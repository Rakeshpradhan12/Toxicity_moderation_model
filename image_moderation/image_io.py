"""Safe decoding of uploaded images: size limits, phone rotation, multi-frame."""
from __future__ import annotations

import os
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "20000000"))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "3"))


class ImageRejected(Exception):
    """The upload cannot be processed. Carries the HTTP status to return."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def frame_indexes(n_frames: int, max_frames: int) -> list[int]:
    """Which frames to check: all of them if few, otherwise spread evenly."""
    if n_frames <= 1 or max_frames <= 1:
        return [0]
    if n_frames <= max_frames:
        return list(range(n_frames))
    step = (n_frames - 1) / (max_frames - 1)
    return sorted({round(i * step) for i in range(max_frames)})


def decode_frames(data: bytes) -> tuple[list[Image.Image], int, int]:
    """Returns (RGB frames, width, height). Animated GIF/WebP files yield multiple frames."""
    try:
        with Image.open(BytesIO(data)) as source:
            width, height = source.size
            if width <= 0 or height <= 0:
                raise ValueError("Invalid dimensions")
            if width * height > MAX_IMAGE_PIXELS:
                raise ImageRejected(413, f"Image exceeds the {MAX_IMAGE_PIXELS}-pixel limit")
            n_frames = getattr(source, "n_frames", 1)
            frames = []
            for index in frame_indexes(n_frames, MAX_FRAMES):
                if n_frames > 1:
                    source.seek(index)
                    frame = source.copy()
                else:
                    frame = source
                frames.append(ImageOps.exif_transpose(frame).convert("RGB"))
    except ImageRejected:
        raise
    except Image.DecompressionBombError as exc:
        raise ImageRejected(413, "Image is too large") from exc
    except (UnidentifiedImageError, OSError, ValueError, EOFError) as exc:
        raise ImageRejected(400, "Unable to decode image") from exc
    return frames, width, height
