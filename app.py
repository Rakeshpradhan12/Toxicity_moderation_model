"""Unified Content Moderation Service.

Combines Multilingual Text Toxicity Moderation and Image/NSFW Moderation into a single server.
Automatically pulls missing model weights from S3 if configured.
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

# Load environment variables from .env if present
load_dotenv()

from core.auth import get_configured_api_keys
from image_moderation.moderator import ImageModerator
from image_moderation.router import router as image_router
from text_moderation.moderator import TextModerator
from text_moderation.router import router as text_router

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("unified_moderation")

LOAD_MODELS_AT_STARTUP = os.getenv("LOAD_MODELS_AT_STARTUP", "true").lower() in ("1", "true", "yes")
ENABLE_TEXT_MODERATION = os.getenv("ENABLE_TEXT_MODERATION", "true").lower() in ("1", "true", "yes")
ENABLE_IMAGE_MODERATION = os.getenv("ENABLE_IMAGE_MODERATION", "true").lower() in ("1", "true", "yes")

_image_lock = asyncio.Lock()
_text_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_keys = get_configured_api_keys()
    require_key = os.getenv("REQUIRE_API_KEY", "false").lower() in ("1", "true", "yes")

    if require_key and not api_keys:
        raise RuntimeError("REQUIRE_API_KEY is enabled but no API_KEYS are specified in environment.")
    if not api_keys:
        logger.warning("No API keys configured: Service is open to unauthenticated requests.")
    else:
        logger.info("Configured %d API key(s) for authentication.", len(api_keys))

    app.state.text_moderator = None
    app.state.image_moderator = None

    if LOAD_MODELS_AT_STARTUP:
        if ENABLE_TEXT_MODERATION:
            try:
                logger.info("Initializing Text Moderator...")
                app.state.text_moderator = await run_in_threadpool(TextModerator)
                logger.info("Text Moderator successfully loaded.")
            except Exception:
                logger.exception("Failed to load Text Moderator during startup.")

        if ENABLE_IMAGE_MODERATION:
            try:
                logger.info("Initializing Image Moderator...")
                app.state.image_moderator = await run_in_threadpool(ImageModerator)
                logger.info("Image Moderator successfully loaded.")
            except Exception:
                logger.exception("Failed to load Image Moderator during startup.")

    yield


app = FastAPI(
    title="Unified Toxicity & Content Moderation API",
    version="2.0.0",
    description=(
        "All-in-one AI moderation service providing Multilingual Text Toxicity detection "
        "(English, Hindi, Hinglish, Odia) and Visual Content Moderation (NSFW + offensive gestures)."
    ),
    lifespan=lifespan,
)

# Register sub-routers
app.include_router(text_router)
app.include_router(image_router)


@app.get("/", summary="Service Information")
def root_info() -> Dict[str, Any]:
    return {
        "service": "Unified Toxicity & Content Moderation API",
        "version": "2.0.0",
        "modules": {
            "text_moderation": ENABLE_TEXT_MODERATION,
            "image_moderation": ENABLE_IMAGE_MODERATION,
        },
        "endpoints": {
            "text": ["/moderate", "/moderate/batch", "/api/v1/moderate-text", "/api/v1/moderate-text/batch"],
            "image": ["/api/v1/moderate-image"],
            "system": ["/health", "/ready", "/docs"],
        },
    }


@app.get("/health", summary="Health check")
def health() -> Dict[str, Any]:
    text_ready = app.state.text_moderator is not None
    image_ready = app.state.image_moderator is not None

    return {
        "status": "ok",
        "text_moderator_loaded": text_ready,
        "image_moderator_loaded": image_ready,
        "text_device": getattr(app.state.text_moderator, "device", None) if text_ready else None,
    }


@app.get("/ready", summary="Readiness probe")
async def ready() -> Dict[str, Any]:
    if ENABLE_TEXT_MODERATION and app.state.text_moderator is None:
        async with _text_lock:
            if app.state.text_moderator is None:
                app.state.text_moderator = await run_in_threadpool(TextModerator)

    if ENABLE_IMAGE_MODERATION and app.state.image_moderator is None:
        async with _image_lock:
            if app.state.image_moderator is None:
                app.state.image_moderator = await run_in_threadpool(ImageModerator)

    return {
        "status": "ready",
        "text_model_ready": app.state.text_moderator is not None,
        "image_models_ready": app.state.image_moderator is not None,
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=False)
