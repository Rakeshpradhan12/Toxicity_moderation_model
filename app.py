"""
Toxicity moderation service.

Run from the laya_model folder (the one that contains models/ft_best):
    python -m uvicorn app:app --host 127.0.0.1 --port 8000

Docs and a test form: http://127.0.0.1:8000/docs
"""
import os
import re
import threading
import unicodedata
from typing import List, Literal, Optional

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ----------------------------------------------------------------------------
# Settings you may want to change
# ----------------------------------------------------------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "models/ft_best")
API_KEY = os.environ.get("API_KEY")        # if set, callers must send header X-API-Key
MAX_LEN = 256                              # tokens used per comment

# Starting thresholds from the fine-tuned model's test table.
#   review: scores at or above this go to a human reviewer
#   block:  scores at or above this are blocked automatically (None = never block on score alone)
# Re-tune these on labelled comments from your own platform.
LANG_RULES = {
    "en":  {"review": 0.5, "block": None},
    "hi":  {"review": 0.3, "block": None},
    "hin": {"review": 0.7, "block": None},   # Hinglish: weak model, rely on user reports too
    "or":  {"review": 0.3, "block": None},   # Odia: not trained, check on your own data
}

# What to do when a comment seems to contain personal details (phone number,
# email, address). "review" sends it to a human; "ignore" turns the check off.
PII_ACTION = "review"

# Emoji rules. Both lists need review by people from your community: which
# emojis count as vulgar depends on your platform and audience.
#   EMOJI_REVIEW:  always sent to review (clearly offensive)
#   EMOJI_CONTEXT: sent to review only when the model also gives the comment at
#                  least EMOJI_CONTEXT_MIN_SCORE (these emojis are often harmless)
EMOJI_REVIEW = ["\U0001F595"]                         # middle finger
EMOJI_CONTEXT = ["\U0001F346", "\U0001F351", "\U0001F4A6", "\U0001F51E", "\U0001F445"]  # eggplant, peach, droplets, 18+, tongue
EMOJI_CONTEXT_MIN_SCORE = 0.1

# Optional: describe emojis in words before the model reads them, e.g. the middle
# finger becomes "middle finger". Needs `python -m pip install emoji`.
# Test with and without on your own examples before keeping it on.
EMOJI_TO_TEXT = True

# Obvious threat or curse phrases the model is known to miss. A match sends the
# comment to review (it never blocks). Add your own, lower-case. Example:
#   "hi": ["..."], "or": ["..."], "en": ["i will find you"]
REVIEW_PHRASES = {"en": [], "hi": [], "hin": [], "or": []}

# ----------------------------------------------------------------------------
# Model loading (once, at startup)
# ----------------------------------------------------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()
ID2LABEL = model.config.id2label
TOX_INDEX = model.config.label2id["toxicity"]
_lock = threading.Lock()                   # one inference at a time, safe on Apple GPU


def score_texts(texts: List[str]) -> List[dict]:
    enc = tokenizer(texts, truncation=True, max_length=MAX_LEN, padding=True,
                    return_tensors="pt", return_token_type_ids=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    with _lock, torch.no_grad():
        probs = torch.sigmoid(model(**enc).logits.float()).cpu().tolist()
    return [{ID2LABEL[i]: round(p, 4) for i, p in enumerate(row)} for row in probs]


# ----------------------------------------------------------------------------
# Language by script
# ----------------------------------------------------------------------------
def detect_lang(text: str):
    """Returns (language, source). Latin letters count as English, because
    English and Hinglish cannot be told apart by script (send lang='hin' to
    force Hinglish)."""
    odia = deva = latin = 0
    for ch in text:
        cp = ord(ch)
        if 0x0B00 <= cp <= 0x0B7F:
            odia += 1
        elif 0x0900 <= cp <= 0x097F:
            deva += 1
        elif ch.isalpha() and cp < 0x250:
            latin += 1
    counts = {"or": odia, "hi": deva, "en": latin}
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        return "en", "default"             # digits, emoji, other scripts
    return best, "detected"


# ----------------------------------------------------------------------------
# Personal details (phone numbers, emails, addresses)
# ----------------------------------------------------------------------------
# This is a simple pattern check, not the toxicity model. It will have both
# false alarms and misses (for example numbers spelled out in words).
_NUM_RE = re.compile(r"(?<!\d)(?:\+?\d[\s\-.()]?){10,15}(?!\d)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PIN_RE = re.compile(r"(?<!\d)[1-9]\d{5}(?!\d)")
_HOUSE_RE = re.compile(r"\b(?:house|flat|plot|door|h)\.?\s*(?:no\.?|number|#)\s*\d+", re.I)
ADDRESS_CUES = ["road", "street", "lane", "nagar", "colony", "near", "pin", "sector",
                "apartment", "village", "district", "square"]  # add Hindi/Odia words with a native speaker


def _ascii_digits(text: str) -> str:
    # turns Devanagari, Odia and other digits into 0-9 so numbers in any script are caught
    return "".join(str(unicodedata.digit(c)) if unicodedata.category(c) == "Nd" else c for c in text)


def find_pii(text: str) -> List[str]:
    t = _ascii_digits(text)
    found = []
    for m in _NUM_RE.finditer(t):
        digits = re.sub(r"\D", "", m.group(0))
        if 10 <= len(digits) <= 15:
            found.append("phone or ID number")
            break
    if _EMAIL_RE.search(t):
        found.append("email address")
    lowered = t.lower()
    has_cue = any(cue in lowered for cue in ADDRESS_CUES)
    if _HOUSE_RE.search(t) or (_PIN_RE.search(t) and has_cue):
        found.append("possible address")
    return found


# ----------------------------------------------------------------------------
# Emojis
# ----------------------------------------------------------------------------
try:
    import emoji as _emoji
except ImportError:
    _emoji = None
if EMOJI_TO_TEXT and _emoji is None:
    print("WARNING: EMOJI_TO_TEXT is on but the 'emoji' package is not installed; run: python -m pip install emoji")


def prepare_for_model(text: str) -> str:
    if EMOJI_TO_TEXT and _emoji is not None:
        return _emoji.replace_emoji(
            text, replace=lambda chars, data: " " + data["en"].strip(":").replace("_", " ") + " ")
    return text


def find_emoji_flags(text: str, toxicity: float):
    always = [e for e in EMOJI_REVIEW if e in text]
    context = [e for e in EMOJI_CONTEXT if e in text] if toxicity >= EMOJI_CONTEXT_MIN_SCORE else []
    return always, context


# ----------------------------------------------------------------------------
# Decision
# ----------------------------------------------------------------------------
def decide(lang: str, toxicity: float, text: str, pii: List[str],
           emoji_always: List[str], emoji_context: List[str]):
    rule = LANG_RULES[lang]
    reasons = []
    decision = "allow"
    if toxicity >= rule["review"]:
        decision = "review"
        reasons.append(f"toxicity {toxicity:.2f} >= review threshold {rule['review']}")
    if rule["block"] is not None and toxicity >= rule["block"]:
        decision = "block"
        reasons.append(f"toxicity >= block threshold {rule['block']}")
    lowered = text.lower()
    for phrase in REVIEW_PHRASES.get(lang, []):
        if phrase and phrase in lowered:
            if decision == "allow":
                decision = "review"
            reasons.append(f"phrase list match: {phrase}")
    if pii and PII_ACTION == "review":
        if decision == "allow":
            decision = "review"
        reasons.append("possible personal details: " + ", ".join(pii))
    if emoji_always:
        if decision == "allow":
            decision = "review"
        reasons.append("emoji on review list: " + " ".join(emoji_always))
    if emoji_context:
        if decision == "allow":
            decision = "review"
        reasons.append(f"context emoji with toxicity {toxicity:.2f} >= {EMOJI_CONTEXT_MIN_SCORE}: " + " ".join(emoji_context))
    return decision, reasons


def moderate_many(items):
    # items: list of (text, optional_lang)
    langs = []
    for text, forced in items:
        if forced:
            langs.append((forced, "provided"))
        else:
            langs.append(detect_lang(text))
    all_scores = score_texts([prepare_for_model(t) for t, _ in items])
    out = []
    for (text, _), (lang, source), scores in zip(items, langs, all_scores):
        tox = scores["toxicity"]
        pii = find_pii(text) if PII_ACTION != "ignore" else []
        e_always, e_context = find_emoji_flags(text, tox)
        decision, reasons = decide(lang, tox, text, pii, e_always, e_context)
        out.append({
            "decision": decision,
            "toxicity": tox,
            "language": lang,
            "language_source": source,
            "review_threshold": LANG_RULES[lang]["review"],
            "reasons": reasons,
            "personal_details": pii,
            "flagged_emojis": e_always + e_context,
            # Only 'toxicity' was fine-tuned. The other outputs are untrained
            # and should not be used for decisions.
            "scores": scores,
        })
    return out


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------
app = FastAPI(title="Toxicity moderation", version="1.0")


def check_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


class Comment(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    lang: Optional[Literal["en", "hi", "hin", "or"]] = None


class Batch(BaseModel):
    items: List[Comment] = Field(min_length=1, max_length=100)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_DIR, "device": device}


@app.post("/moderate", dependencies=[Depends(check_key)])
def moderate(c: Comment):
    return moderate_many([(c.text, c.lang)])[0]


@app.post("/moderate/batch", dependencies=[Depends(check_key)])
def moderate_batch(b: Batch):
    return {"results": moderate_many([(c.text, c.lang) for c in b.items])}