
import os
import sys
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

load_dotenv()


version = sys.argv[1] if len(sys.argv) > 1 else "v1"
folder = Path(sys.argv[2] if len(sys.argv) > 2 else "models/ft_best")
bucket = os.environ.get("MODEL_S3_BUCKET")
region = os.environ.get("AWS_REGION")

if not bucket:
    sys.exit("Set MODEL_S3_BUCKET in .env (the bucket name to upload to).")
if not folder.is_dir():
    sys.exit(f"Folder not found: {folder}")
if not (folder / "model.safetensors").exists():
    sys.exit(f"{folder} has no model.safetensors. Is this the right folder?")

base_prefix = os.environ.get("MODEL_S3_PREFIX", "ml-models").strip("/")
prefix = f"{base_prefix}/{folder.name}/{version}/" if base_prefix else f"{folder.name}/{version}/"
files = [p for p in sorted(folder.rglob("*")) if p.is_file() and p.name != ".DS_Store"]
total = sum(p.stat().st_size for p in files)

print(f"Bucket : {bucket}")
print(f"Region : {region}")
print(f"Target : s3://{bucket}/{prefix}")
print(f"Files  : {len(files)}  ({total / 1e6:.0f} MB)")

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