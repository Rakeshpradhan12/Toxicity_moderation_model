"""Authentication dependency for FastAPI endpoints."""
from __future__ import annotations

import hmac
import logging
import os
from typing import Optional

from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)


def get_configured_api_keys() -> list[str]:
    raw = os.getenv("API_KEYS", os.getenv("API_KEY", ""))
    return [k.strip() for k in raw.split(",") if k.strip()]


def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    api_keys = get_configured_api_keys()
    require_key = os.getenv("REQUIRE_API_KEY", "false").lower() in ("1", "true", "yes")

    if not require_key:
        return

    if not api_keys:
        return

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    given = x_api_key.strip().encode("utf-8")
    if not any(hmac.compare_digest(given, k.encode("utf-8")) for k in api_keys):
        raise HTTPException(status_code=401, detail="Invalid X-API-Key")
