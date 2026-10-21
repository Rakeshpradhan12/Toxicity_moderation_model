"""
Upload a model folder, or a whole bundle of model folders, to S3.

Examples (run from the project that has your .env with the AWS keys):
    python upload_model.py v1                                   # models/ft_best -> ml-models/ft_best/v1/
    python upload_model.py v1 /path/to/weights --name image-moderation
        # uploads weights/nsfw_image and weights/clip_light together:
        #   ml-models/image-moderation/v1/nsfw_image/...
        #   ml-models/image-moderation/v1/clip_light/...
    python upload_model.py v1 /path/to/weights --name image-moderation --exclude clip
        # same, but skips the subfolder weights/clip

.env needs:
    AWS_ACCESS_KEY_ID=...
    AWS_SECRET_ACCESS_KEY=...
    AWS_REGION=...
    MODEL_S3_BUCKET=your-bucket-name
    MODEL_S3_PREFIX=ml-models          # optional, default ml-models

Install first:  python -m pip install boto3 python-dotenv
"""
import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv


def collect_files(folder: Path, excludes: set):
    files = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.name == ".DS_Store":
            continue
        rel = p.relative_to(folder)
        if rel.parts[0] in excludes:
            continue
        files.append(p)
    return files


def main(argv=None) -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description="Upload a model folder or bundle to S3")
    ap.add_argument("version")
    ap.add_argument("folder", nargs="?", default="models/ft_best")
    ap.add_argument("--name", default=None, help="name used in the S3 path (default: the folder's name)")
    ap.add_argument("--exclude", action="append", default=[], help="top-level subfolder to skip (repeatable)")
    args = ap.parse_args(argv)

    folder = Path(args.folder).expanduser()
    bucket = os.environ.get("MODEL_S3_BUCKET")
    region = os.environ.get("AWS_REGION")
    if not bucket:
        sys.exit("Set MODEL_S3_BUCKET in .env (the bucket name to upload to).")
    if not folder.is_dir():
        sys.exit(f"Folder not found: {folder}")

    files = collect_files(folder, set(args.exclude))
    if not any(p.name == "model.safetensors" for p in files):
        sys.exit(f"No model.safetensors found under {folder}. Is this the right folder?")

    base_prefix = os.environ.get("MODEL_S3_PREFIX", "ml-models").strip("/")
    name = args.name or folder.name
    prefix = "/".join(x for x in (base_prefix, name, args.version) if x) + "/"

    sizes = defaultdict(int)
    for p in files:
        rel = p.relative_to(folder)
        sizes[rel.parts[0] if len(rel.parts) > 1 else "(top level)"] += p.stat().st_size
    total = sum(sizes.values())

    print(f"Bucket : {bucket}")
    print(f"Region : {region}")
    print(f"Target : s3://{bucket}/{prefix}")
    print(f"Files  : {len(files)}  ({total / 1e6:.0f} MB)")
    for part, size in sorted(sizes.items()):
        print(f"   {part:<20} {size / 1e6:8.0f} MB")
    if args.exclude:
        print("Skipped: " + ", ".join(args.exclude))

    s3 = boto3.client("s3", region_name=region)   # credentials come from .env or ~/.aws

    try:
        if s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1).get("KeyCount", 0) > 0:
            sys.exit(f"Something already exists at {prefix}. Use a new version name (for example v2).")
    except NoCredentialsError:
        sys.exit("No AWS credentials found. Check your .env file.")
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("AccessDenied", "403"):
            print("Warning: these credentials cannot list the bucket, so the check for an existing version was skipped.")
        else:
            sys.exit(f"Cannot access the bucket: {e}")

    if input("Type yes to upload: ").strip().lower() != "yes":
        sys.exit("Cancelled.")

    config = TransferConfig(multipart_threshold=64 * 1024 * 1024)
    for p in files:
        key = prefix + p.relative_to(folder).as_posix()
        print(f"uploading {key} ({p.stat().st_size / 1e6:.1f} MB)")
        s3.upload_file(str(p), bucket, key, ExtraArgs={"ServerSideEncryption": "AES256"}, Config=config)

    # verify: every local file exists in S3 with the same size
    problems = []
    for p in files:
        key = prefix + p.relative_to(folder).as_posix()
        try:
            size = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
            if size != p.stat().st_size:
                problems.append(f"{key} (size {size} != {p.stat().st_size})")
        except ClientError as e:
            problems.append(f"{key} ({e.response.get('Error', {}).get('Code')})")
    if problems:
        sys.exit("Could not verify: " + "; ".join(problems))
    print(f"\nDone. {len(files)} files verified at s3://{bucket}/{prefix}")


if __name__ == "__main__":
    main()