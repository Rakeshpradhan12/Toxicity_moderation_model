"""Asynchronous Task Queue & Persistence for Content Moderation.

Provides a persistent SQLite-backed task store and in-memory async worker queue.
Ensures zero dropped requests, instant HTTP 202 responses (<20ms), and background
execution of heavy AI models (NSFW + Light CLIP).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("tasks_data/tasks.db")
DEFAULT_STORAGE_DIR = Path("tasks_data/uploads")
MAX_QUEUE_WORKERS = int(os.getenv("MAX_QUEUE_WORKERS", "2"))


class TaskQueue:
    """Manages persistent task storage and background processing."""

    def __init__(
        self,
        db_path: Path = DEFAULT_DB_PATH,
        storage_dir: Path = DEFAULT_STORAGE_DIR,
        num_workers: int = MAX_QUEUE_WORKERS,
    ):
        self.db_path = db_path
        self.storage_dir = storage_dir
        self.num_workers = num_workers
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.workers: list[asyncio.Task] = []
        self._running = False
        self.app: Optional[Any] = None

        self._ensure_directories()
        self._init_db()

    def _ensure_directories(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS moderation_tasks (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    file_path TEXT,
                    text_content TEXT,
                    language TEXT,
                    webhook_url TEXT,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    result_json TEXT,
                    error_message TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_status ON moderation_tasks(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_created ON moderation_tasks(created_at)"
            )
            conn.commit()

    def start_workers(self, app: Any) -> None:
        """Starts background worker tasks within the asyncio event loop."""
        self.app = app
        self._running = True

        # Recover any orphaned 'queued' or unfinished 'processing' tasks from DB
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT task_id FROM moderation_tasks WHERE status IN ('queued', 'processing') ORDER BY created_at ASC"
            )
            orphaned = [row["task_id"] for row in cursor.fetchall()]

        for tid in orphaned:
            self.queue.put_nowait(tid)

        if orphaned:
            logger.info("Enqueued %d unfinished tasks from database for processing.", len(orphaned))

        # Launch workers
        for i in range(self.num_workers):
            task = asyncio.create_task(self._worker_loop(i))
            self.workers.append(task)
        logger.info("Started %d async queue worker(s).", self.num_workers)

    async def stop_workers(self) -> None:
        """Gracefully terminates worker tasks."""
        self._running = False
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        logger.info("All queue workers stopped.")

    async def enqueue_image_task(
        self,
        image_bytes: bytes,
        filename: str = "upload.jpg",
        webhook_url: Optional[str] = None,
    ) -> str:
        """Saves image to disk, records in DB, and enqueues task in <15ms."""
        task_id = str(uuid.uuid4())
        ext = Path(filename).suffix.lower() or ".jpg"
        file_path = self.storage_dir / f"{task_id}{ext}"

        # Write to disk asynchronously
        await asyncio.to_thread(file_path.write_bytes, image_bytes)

        now = time.time()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO moderation_tasks 
                (task_id, task_type, status, file_path, webhook_url, created_at)
                VALUES (?, 'image', 'queued', ?, ?, ?)
                """,
                (task_id, str(file_path), webhook_url, now),
            )
            conn.commit()

        await self.queue.put(task_id)
        return task_id

    async def enqueue_text_task(
        self,
        text: str,
        lang: Optional[str] = None,
        webhook_url: Optional[str] = None,
    ) -> str:
        """Records text task in DB and enqueues it."""
        task_id = str(uuid.uuid4())
        now = time.time()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO moderation_tasks 
                (task_id, task_type, status, text_content, language, webhook_url, created_at)
                VALUES (?, 'text', 'queued', ?, ?, ?, ?)
                """,
                (task_id, text, lang, webhook_url, now),
            )
            conn.commit()

        await self.queue.put(task_id)
        return task_id

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves current status and result for a task ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM moderation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()

        if not row:
            return None

        result_data = None
        if row["result_json"]:
            try:
                result_data = json.loads(row["result_json"])
            except Exception:
                result_data = row["result_json"]

        duration_sec = None
        if row["started_at"] and row["completed_at"]:
            duration_sec = round(row["completed_at"] - row["started_at"], 3)
        elif row["created_at"] and row["completed_at"]:
            duration_sec = round(row["completed_at"] - row["created_at"], 3)

        return {
            "task_id": row["task_id"],
            "task_type": row["task_type"],
            "status": row["status"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "duration_sec": duration_sec,
            "result": result_data,
            "error": row["error_message"],
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Returns statistics on queued, processing, completed, and failed tasks."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT status, COUNT(*) as count FROM moderation_tasks GROUP BY status"
            )
            counts = {row["status"]: row["count"] for row in cursor.fetchall()}

        return {
            "in_memory_queue_size": self.queue.qsize(),
            "queued": counts.get("queued", 0),
            "processing": counts.get("processing", 0),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
            "total_tasks": sum(counts.values()),
        }

    async def _worker_loop(self, worker_id: int) -> None:
        """Continuous consumer loop popping and processing tasks."""
        logger.debug("Worker %d ready.", worker_id)

        while self._running:
            try:
                task_id = await self.queue.get()
            except asyncio.CancelledError:
                break

            try:
                await self._process_single_task(task_id, worker_id)
            except Exception as e:
                logger.exception("Unexpected error processing task %s in worker %d", task_id, worker_id)
            finally:
                self.queue.task_done()

    async def _process_single_task(self, task_id: str, worker_id: int) -> None:
        """Processes one task from start to finish."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM moderation_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()

        if not row:
            return

        # Mark processing
        start_time = time.time()
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE moderation_tasks SET status = 'processing', started_at = ? WHERE task_id = ?",
                (start_time, task_id),
            )
            conn.commit()

        task_type = row["task_type"]
        file_path_str = row["file_path"]
        webhook_url = row["webhook_url"]

        try:
            if task_type == "image":
                from image_moderation.image_io import decode_frames

                file_path = Path(file_path_str)
                if not file_path.exists():
                    raise FileNotFoundError(f"Image file {file_path_str} was not found on disk.")

                image_bytes = await asyncio.to_thread(file_path.read_bytes)
                frames, width, height = await run_in_threadpool(decode_frames, image_bytes)

                moderator = self.app.state.image_moderator if self.app else None
                if moderator is None:
                    raise RuntimeError("Image moderator is not initialized.")

                result = await run_in_threadpool(moderator.moderate_frames, frames)
                result_payload = {
                    **result,
                    "image_width": width,
                    "image_height": height,
                    "file_size_bytes": len(image_bytes),
                    "inference_time_ms": round((time.time() - start_time) * 1000, 2),
                }

                # Clean up stored file
                try:
                    file_path.unlink(missing_ok=True)
                except Exception:
                    pass

            elif task_type == "text":
                moderator = self.app.state.text_moderator if self.app else None
                if moderator is None:
                    raise RuntimeError("Text moderator is not initialized.")

                result_payload = await run_in_threadpool(
                    moderator.moderate_single, row["text_content"], row["language"]
                )
            else:
                raise ValueError(f"Unsupported task type: {task_type}")

            # Mark completed
            completed_time = time.time()
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE moderation_tasks 
                    SET status = 'completed', completed_at = ?, result_json = ? 
                    WHERE task_id = ?
                    """,
                    (completed_time, json.dumps(result_payload), task_id),
                )
                conn.commit()

            # Fire webhook if requested
            if webhook_url:
                asyncio.create_task(self._deliver_webhook(webhook_url, task_id, "completed", result_payload))

        except Exception as exc:
            logger.exception("Task %s failed: %s", task_id, str(exc))
            completed_time = time.time()
            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE moderation_tasks 
                    SET status = 'failed', completed_at = ?, error_message = ? 
                    WHERE task_id = ?
                    """,
                    (completed_time, str(exc), task_id),
                )
                conn.commit()

            if webhook_url:
                asyncio.create_task(self._deliver_webhook(webhook_url, task_id, "failed", {"error": str(exc)}))

    async def _deliver_webhook(
        self, webhook_url: str, task_id: str, status: str, payload: Dict[str, Any]
    ) -> None:
        """Sends non-blocking HTTP POST webhook callback."""
        body = {
            "task_id": task_id,
            "status": status,
            "data": payload,
            "timestamp": time.time(),
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(webhook_url, json=body)
        except Exception as e:
            logger.warning("Failed to deliver webhook for task %s to %s: %s", task_id, webhook_url, e)


# Global task queue singleton
task_queue = TaskQueue()
