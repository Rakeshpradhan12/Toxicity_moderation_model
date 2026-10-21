# Unified Toxicity & Content Moderation Service

All-in-one AI content moderation service that merges:
1. **Multilingual Text Toxicity Moderation**: English (`en`), Hindi (`hi`), Hinglish (`hin`), and Odia (`or`) toxicity classification with PII leaks and offensive emoji detection.
2. **Visual Content Moderation**: NSFW / nudity detection and offensive hand gesture identification via CLIP.

It includes **automatic S3 model weight pulling** on startup, so the server can spin up on any fresh environment without manual weight copying.

---

## Architecture & Layout

```text
Toxicity_moderation_model/
├── app.py                   # Unified FastAPI entry point (runs both text & image models)
├── requirements.txt         # Consolidated dependencies
├── .env                     # Configuration (AWS S3, API Keys, thresholds)
├── .env.example             # Template for configuration
├── Dockerfile               # Production container image
├── core/
│   ├── auth.py              # API key verification (X-API-Key header)
│   └── s3_sync.py           # Automatic S3 model download & validation
├── text_moderation/
│   ├── moderator.py         # Text toxicity, script detection, PII, and emojis
│   └── router.py            # Endpoints: /moderate, /api/v1/moderate-text
├── image_moderation/
│   ├── image_io.py          # Image decoding, limits, and multi-frame handling
│   ├── light_clip.py        # Optimized Light CLIP vision inference
│   ├── moderator.py         # Combined NSFW + CLIP decision engine
│   └── router.py            # Endpoint: /api/v1/moderate-image
├── horizen_labs_model/      # Original text moderation sub-module (kept for reference)
└── image_moderation_api/    # Original image moderation sub-module (kept for reference)
```

---

## Automatic S3 Model Sync

When the server starts:
- **Text Model**: Checks if `models/ft_best/config.json` exists. If missing, it automatically pulls weights from `s3://{MODEL_S3_BUCKET}/{TEXT_MODEL_S3_PREFIX}`.
- **Image Models**: Checks if `weights/nsfw_image` and `weights/clip_light` exist. If missing, it downloads them from S3 automatically. If S3 is not configured, it safely falls back to standard Hugging Face checkpoints.

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Copy `.env.example` to `.env` and provide your AWS credentials if downloading from S3:
```env
AWS_ACCESS_KEY_ID=your_key_id
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=ap-south-2
MODEL_S3_BUCKET=your_bucket_name
```

### 3. Run the Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Or run directly:
```bash
python app.py
```

Interactive Swagger documentation:
- **http://127.0.0.1:8000/docs**

---

## API Endpoints

### System Endpoints
| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service information, loaded modules, and routes |
| `GET` | `/health` | Health status of text and image models |
| `GET` | `/ready` | Readiness probe (ensures weights are loaded into memory) |

---

### Text Moderation Endpoints
**Header**: `X-API-Key: <your_key>` (if API keys are enabled)

#### `POST /api/v1/moderate-text` (or `/moderate`)
Request:
```json
{
  "text": "तुम बहुत अच्छे इंसान हो, धन्यवाद!",
  "lang": "hi"
}
```
Response:
```json
{
  "decision": "allow",
  "toxicity": 0.0215,
  "language": "hi",
  "language_source": "provided",
  "review_threshold": 0.3,
  "reasons": [],
  "personal_details": [],
  "chunks": 1,
  "flagged_emojis": [],
  "scores": { "toxicity": 0.0215, "severe_toxicity": 0.001 }
}
```

#### `POST /api/v1/moderate-text/batch` (or `/moderate/batch`)
Moderate up to 100 comments in a single optimized pass.

---

### Image Moderation Endpoints
**Header**: `X-API-Key: <your_key>`

#### `POST /api/v1/moderate-image`
Request:
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/moderate-image" \
  -H "X-API-Key: your_api_key" \
  -F "file=@/path/to/image.jpg"
```
Response:
```json
{
  "decision": "approved",
  "reason": "no_high_risk_signal",
  "scores": {
    "nsfw": 0.04,
    "normal": 0.96,
    "clip_rude_gesture_total": 0.12,
    "clip_rude_gesture_max": 0.08,
    "clip_rude_gesture_best_label": "a person showing an offensive hand gesture",
    "clip_ordinary_photo": 0.88
  },
  "file_size_bytes": 105420,
  "image_width": 800,
  "image_height": 600,
  "inference_time_ms": 145.2
}
```

---

## Docker Deployment

Build and run the unified container:
```bash
docker build -t unified-moderation-api .
docker run -p 8000:8000 --env-file .env unified-moderation-api
```
