"""FastAPI routes for text toxicity moderation."""
from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from core.auth import verify_api_key

router = APIRouter(tags=["Text Moderation"], dependencies=[Depends(verify_api_key)])


class Comment(BaseModel):
    text: str = Field(min_length=1, max_length=5000, description="Comment text to moderate")
    lang: Optional[Literal["en", "hi", "hin", "or"]] = Field(
        default=None, description="Force language code ('en', 'hi', 'hin', 'or'), or None to auto-detect."
    )


class BatchComments(BaseModel):
    items: List[Comment] = Field(min_length=1, max_length=100, description="List of comments to moderate in batch")


@router.post("/moderate", summary="Moderate single text (Legacy endpoint)")
@router.post("/api/v1/moderate-text", summary="Moderate single text comment")
def moderate_text(comment: Comment, request: Request):
    moderator = request.app.state.text_moderator
    return moderator.moderate_single(comment.text, comment.lang)


@router.post("/moderate/batch", summary="Moderate batch of texts (Legacy endpoint)")
@router.post("/api/v1/moderate-text/batch", summary="Moderate batch of text comments")
def moderate_text_batch(batch: BatchComments, request: Request):
    moderator = request.app.state.text_moderator
    results = moderator.moderate_many([(c.text, c.lang) for c in batch.items])
    return {"results": results}


class AsyncComment(BaseModel):
    text: str = Field(min_length=1, max_length=5000, description="Comment text to moderate")
    lang: Optional[Literal["en", "hi", "hin", "or"]] = Field(
        default=None, description="Force language code ('en', 'hi', 'hin', 'or'), or None to auto-detect."
    )
    webhook_url: Optional[str] = Field(
        default=None, description="Optional webhook URL to receive notification when moderation completes"
    )


@router.post(
    "/api/v1/moderate-text/async",
    status_code=202,
    summary="Enqueue text comment for asynchronous moderation (Returns 202 immediately)",
)
async def moderate_text_async(comment: AsyncComment) -> dict:
    from core.task_queue import task_queue

    task_id = await task_queue.enqueue_text_task(
        text=comment.text,
        lang=comment.lang,
        webhook_url=comment.webhook_url,
    )
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "Text comment successfully queued for moderation",
        "check_status_url": f"/api/v1/tasks/{task_id}",
    }
