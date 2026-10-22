"""FastAPI routes for image moderation."""
from __future__ import annotations
from typing import Optional

import logging
import os
from time import perf_counter

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from core.auth import verify_api_key
from image_moderation.image_io import ImageRejected, decode_frames
from image_moderation.moderator import ModeratorBusy

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Image Moderation"], dependencies=[Depends(verify_api_key)])

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))


@router.post("/api/v1/moderate-image", summary="Moderate uploaded image")
async def moderate_image(request: Request, file: UploadFile = File(...)) -> dict:
    data = await file.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty image")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {MAX_IMAGE_BYTES}-byte limit",
        )

    started = perf_counter()
    try:
        frames, width, height = await run_in_threadpool(decode_frames, data)
        moderator = request.app.state.image_moderator
        if moderator is None:
            raise HTTPException(status_code=503, detail="Image moderator is not loaded")
        result = await run_in_threadpool(moderator.moderate_frames, frames)
    except HTTPException:
        raise
    except ImageRejected as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except ModeratorBusy as exc:
        raise HTTPException(
            status_code=503,
            detail="Moderation is busy, retry shortly",
            headers={"Retry-After": "5"},
        ) from exc
    except Exception as exc:
        logger.exception("Image moderation failed")
        raise HTTPException(status_code=500, detail="Image moderation failed") from exc

    return {
        **result,
        "file_size_bytes": len(data),
        "image_width": width,
        "image_height": height,
        "inference_time_ms": round((perf_counter() - started) * 1000, 2),
    }


@router.post(
    "/api/v1/moderate-image/async",
    status_code=202,
    summary="Enqueue image for asynchronous moderation (Returns 202 immediately)",
)
async def moderate_image_async(
    file: UploadFile = File(...),
    webhook_url: Optional[str] = None,
) -> dict:
    from core.task_queue import task_queue

    data = await file.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty image")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {MAX_IMAGE_BYTES}-byte limit",
        )

    task_id = await task_queue.enqueue_image_task(
        image_bytes=data,
        filename=file.filename or "upload.jpg",
        webhook_url=webhook_url,
    )

    return {
        "task_id": task_id,
        "status": "queued",
        "message": "Image successfully queued for moderation",
        "check_status_url": f"/api/v1/tasks/{task_id}",
    }
