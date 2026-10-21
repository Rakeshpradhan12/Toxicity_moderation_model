# Image Moderation API

Standalone FastAPI service combining:
- `Falconsai/nsfw_image_detection` for NSFW classification.
- `openai/clip-vit-base-patch32` with multiple prompts for possible offensive hand gestures.

This service returns a moderation decision and scores. It does **not** upload to S3 or generate CloudFront URLs; your existing FastAPI app retains those responsibilities.

## Structure

```text
image_moderation_api/
├── app/
│   ├── __init__.py
│   └── main.py                 # REST endpoints
├── models/
│   ├── __init__.py
│   └── moderation.py           # Loads/runs both models and applies rules
├── tests/                      # Add labelled test images/tests here
├── test_clip.py                # Optional local CLIP test script
├── test_image.py               # Optional local NSFW test script
├── requirements.txt
├── .env.example
├── .gitignore
├── Dockerfile
└── README.md
```

## Run locally

You can use your existing `imgmod` Conda environment. From this project directory:

```bash
python -m pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

The first call to `/ready` or the first moderation request loads the models and may download their weights. Open these routes:
- Swagger UI: http://127.0.0.1:8001/docs
- Liveness: http://127.0.0.1:8001/health
- Readiness (loads models): http://127.0.0.1:8001/ready

## Test an image

```bash
curl -X POST "http://127.0.0.1:8001/api/v1/moderate-image" \
  -F "file=@/absolute/path/to/image.jpg"
```

Example response shape (scores and timing are illustrative):

```json
{
  "decision": "pending_review",
  "reason": "possible_offensive_gesture",
  "scores": {
    "nsfw": 0.12,
    "normal": 0.88,
    "clip_rude_gesture_max": 0.81,
    "clip_rude_gesture_best_label": "a person showing an offensive hand gesture",
    "clip_prompts": {},
    "clip_ordinary_photo": 0.10
  },
  "file_size_bytes": 245120,
  "image_width": 800,
  "image_height": 600,
  "inference_time_ms": 500.0
}
```

## Call it from the main FastAPI app

Install `httpx` in the main application's environment. Call this helper after your existing validation and before making the image publicly available:

```python
import httpx

MODERATION_URL = "http://127.0.0.1:8001/api/v1/moderate-image"

async def get_image_moderation(image_bytes: bytes) -> dict:
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            MODERATION_URL,
            files={"file": ("image.jpg", image_bytes, "application/octet-stream")},
        )
        response.raise_for_status()
        return response.json()
```

Then use the response like this:
- `approved`: save to your approved image path and let your existing CloudFront flow serve it.
- `pending_review`: keep it private/quarantined and do not return a public CloudFront URL.
- `blocked`: reject the upload or follow your documented retention/moderation policy.

`127.0.0.1` only works when both apps can reach the same host/network namespace. For Docker Compose, use the service hostname, such as `http://image-moderation:8001/api/v1/moderate-image`.

## Important caveats

- The CLIP scores are relative to the supplied candidate prompts, not calibrated probabilities. Review thresholds using a manually labelled test set before production enforcement.
- This CLIP check is for visual gestures; it does not reliably read swear words within an image. For that, add OCR and text moderation.
- This example blocks high-scoring NSFW results and sends likely gesture matches to manual review. Tune thresholds for your policies and test for false positives/negatives.
- The service intentionally does not check MIME type or file extension because the calling application already performs that validation. It still decodes the image and checks byte size and pixel count to protect inference resources.
- Each Uvicorn worker loads its own copies of the models. Begin with one worker and monitor memory and latency.
- Protect this service with network controls or authentication if reachable outside a trusted private network.
