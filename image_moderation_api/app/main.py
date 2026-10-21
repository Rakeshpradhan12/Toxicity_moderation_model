"""FastAPI HTTP interface for the image moderation models."""
from __future__ import annotations

import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager
from time import perf_counter
from dotenv import load_dotenv

load_dotenv()

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from models.image_io import ImageRejected, decode_frames
from models.moderation import ImageModerator, ModeratorBusy


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
# Several keys may be listed (comma separated) so a key can be rotated without downtime.
API_KEYS = [
    k.strip()
    for k in os.getenv("API_KEYS", os.getenv("API_KEY", "")).split(",")
    if k.strip()
]
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() in ("1", "true", "yes")
LOAD_MODELS_AT_STARTUP = os.getenv("LOAD_MODELS_AT_STARTUP", "true").lower() in ("1", "true", "yes")

_load_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if REQUIRE_API_KEY and not API_KEYS:
        raise RuntimeError("REQUIRE_API_KEY is on but no API_KEYS are set")
    if not API_KEYS:
        logger.warning("No API key is configured: anyone who can reach this service can use it")
    app.state.moderator = None
    if LOAD_MODELS_AT_STARTUP:
        try:
            app.state.moderator = await run_in_threadpool(ImageModerator)
            logger.info("Moderation models loaded")
        except Exception:
            logger.exception("Could not load moderation models at startup")
    yield


app = FastAPI(
    title="Image Moderation API",
    version="1.1.0",
    description=(
        "Runs an NSFW classifier and CLIP prompts for potentially offensive "
        "gestures. Storage in S3 and delivery via CloudFront remain in the caller."
    ),
    lifespan=lifespan,
)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not API_KEYS:
        return
    given = (x_api_key or "").encode()
    if not any(hmac.compare_digest(given, key.encode()) for key in API_KEYS):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


async def get_moderator() -> ImageModerator:
    moderator = getattr(app.state, "moderator", None)
    if moderator is not None:
        return moderator
    async with _load_lock:
        moderator = getattr(app.state, "moderator", None)
        if moderator is None:
            try:
                moderator = await run_in_threadpool(ImageModerator)
            except Exception as exc:
                logger.exception("Could not load moderation models")
                raise HTTPException(status_code=503, detail="Models are not ready") from exc
            app.state.moderator = moderator
    return moderator


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, str | bool]:
    await get_moderator()          # raises 503 if the models cannot be loaded
    return {"status": "ready", "models_loaded": True}


@app.post("/api/v1/moderate-image", dependencies=[Depends(require_api_key)])
async def moderate_image(file: UploadFile = File(...)) -> dict:
    """Receive an image that has already passed the main app's type checks.

    Decision values: approved, pending_review, blocked. Treat any other value,
    an error, or a timeout as pending_review. Never show an unchecked image.
    """
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
        moderator = await get_moderator()
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

    logger.info("moderated image: decision=%s reason=%s frames=%s",
                result["decision"], result["reason"], result.get("frames_checked"))
    return {
        **result,
        "file_size_bytes": len(data),
        "image_width": width,
        "image_height": height,
        "inference_time_ms": round((perf_counter() - started) * 1000, 2),
    }