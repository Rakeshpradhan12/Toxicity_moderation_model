"""
Download a model bundle from S3 onto a server and check the file sizes.

Run on the server, from the project folder:
    python download_models.py ml-models/image-moderation/v1/ weights

That recreates:  weights/nsfw_image/...   weights/clip_light/...
Files that are already there with the right size are skipped, so it is safe to rerun.

Needs:  python -m pip install boto3
Environment (or an IAM role on the server, which is better than keys):
    MODEL_S3_BUCKET=your-bucket-name
    AWS_REGION=...
Give the server read-only access (s3:ListBucket and s3:GetObject on the ml-models/ prefix).
"""
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        sys.exit("Usage: python download_models.py <s3 prefix> <destination folder>")
    prefix, dest = argv[0].strip("/") + "/", Path(argv[1])
    bucket = os.environ.get("MODEL_S3_BUCKET")
    if not bucket:
        sys.exit("Set MODEL_S3_BUCKET.")

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION"))
    objects = []
    try:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            objects += [o for o in page.get("Contents", []) if not o["Key"].endswith("/")]
    except ClientError as e:
        sys.exit(f"Cannot list s3://{bucket}/{prefix}: {e}")
    if not objects:
        sys.exit(f"Nothing found at s3://{bucket}/{prefix}. Check the prefix.")

    total = sum(o["Size"] for o in objects)
    print(f"{len(objects)} files, {total / 1e6:.0f} MB, from s3://{bucket}/{prefix} to {dest}/")
    for o in objects:
        rel = o["Key"][len(prefix):]
        target = dest / rel
        if target.exists() and target.stat().st_size == o["Size"]:
            print(f"skip     {rel} (already there)")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"download {rel} ({o['Size'] / 1e6:.1f} MB)")
        s3.download_file(bucket, o["Key"], str(target))
        if target.stat().st_size != o["Size"]:
            sys.exit(f"Size mismatch for {rel}; download again.")
    print(f"\nDone. Models are in {dest}/")


if __name__ == "__main__":
    main()