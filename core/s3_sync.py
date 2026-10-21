"""S3 model synchronization utility.

Automatically downloads model weights from S3 if they are not present locally.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def sync_s3_prefix(
    bucket: str,
    prefix: str,
    dest_dir: str | Path,
    region: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
) -> bool:
    """Download all objects matching `prefix` in `bucket` into `dest_dir`.

    Skips files that already exist locally with the exact same size.
    Returns True if sync succeeded or was already up-to-date, False on failure.
    """
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError:
        logger.error("boto3 is not installed. Run: pip install boto3")
        return False

    prefix = prefix.strip("/") + "/"
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    client_kwargs = {}
    if region:
        client_kwargs["region_name"] = region
    if aws_access_key_id and aws_secret_access_key:
        client_kwargs["aws_access_key_id"] = aws_access_key_id
        client_kwargs["aws_secret_access_key"] = aws_secret_access_key

    try:
        s3 = boto3.client("s3", **client_kwargs)
        paginator = s3.get_paginator("list_objects_v2")
        objects = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith("/"):
                    objects.append(obj)

        if not objects:
            logger.warning("No objects found in s3://%s/%s", bucket, prefix)
            return False

        logger.info("Syncing %d files from s3://%s/%s to %s", len(objects), bucket, prefix, dest)
        for obj in objects:
            key = obj["Key"]
            rel_path = key[len(prefix):]
            target_path = dest / rel_path

            if target_path.exists() and target_path.stat().st_size == obj["Size"]:
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading %s (%.1f MB) -> %s", rel_path, obj["Size"] / 1e6, target_path)
            s3.download_file(bucket, key, str(target_path))

            if target_path.stat().st_size != obj["Size"]:
                logger.error("Size mismatch for %s after download", rel_path)
                return False

        logger.info("Successfully synced s3://%s/%s -> %s", bucket, prefix, dest)
        return True

    except (ClientError, NoCredentialsError) as e:
        logger.error("Failed to sync from s3://%s/%s: %s", bucket, prefix, e)
        return False
    except Exception as e:
        logger.exception("Unexpected error syncing from S3: %s", e)
        return False


def ensure_model_files(
    local_dir: str | Path,
    s3_prefix: Optional[str] = None,
    s3_bucket: Optional[str] = None,
    region: Optional[str] = None,
    required_files: tuple[str, ...] = ("config.json",),
) -> bool:
    """Checks if required files exist in `local_dir`.

    If missing and S3 credentials/bucket are available, triggers automatic S3 sync.
    """
    dest = Path(local_dir)
    all_present = dest.is_dir() and all((dest / f).exists() for f in required_files)

    if all_present:
        logger.debug("Model files in %s already exist.", dest)
        return True

    bucket = s3_bucket or os.getenv("MODEL_S3_BUCKET")
    prefix = s3_prefix or os.getenv("MODEL_S3_PREFIX")
    reg = region or os.getenv("AWS_REGION")
    ak = os.getenv("AWS_ACCESS_KEY_ID")
    sk = os.getenv("AWS_SECRET_ACCESS_KEY")

    if not bucket or not prefix:
        logger.warning(
            "Local model files missing in %s, but S3 bucket or prefix is not configured. "
            "(MODEL_S3_BUCKET=%s, prefix=%s)",
            dest,
            bucket,
            prefix,
        )
        return False

    logger.info("Model files missing in %s. Attempting to download from s3://%s/%s ...", dest, bucket, prefix)
    success = sync_s3_prefix(
        bucket=bucket,
        prefix=prefix,
        dest_dir=dest,
        region=reg,
        aws_access_key_id=ak,
        aws_secret_access_key=sk,
    )

    if success and all((dest / f).exists() for f in required_files):
        logger.info("Model directory %s is now ready.", dest)
        return True

    logger.error("Failed to verify model files in %s after S3 sync attempt.", dest)
    return False
