"""Multilingual text toxicity moderation engine."""
from __future__ import annotations

import logging
import os
import re
import threading
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.s3_sync import ensure_model_files

logger = logging.getLogger(__name__)

# Default model directories and S3 settings
DEFAULT_TEXT_MODEL_DIR = os.getenv("TEXT_MODEL_DIR", os.getenv("MODEL_DIR", "models/ft_best"))
DEFAULT_TEXT_S3_PREFIX = os.getenv("TEXT_MODEL_S3_PREFIX", "ml-models/ft_best/v1/")
DEFAULT_S3_BUCKET = os.getenv("MODEL_S3_BUCKET")

MAX_LEN = int(os.getenv("TEXT_MAX_LEN", "256"))
LONG_TEXT_OVERLAP = int(os.getenv("TEXT_LONG_TEXT_OVERLAP", "32"))
CHUNK_BATCH = int(os.getenv("TEXT_CHUNK_BATCH", "32"))

LANG_RULES = {
    "en": {"review": 0.5, "block": None},
    "hi": {"review": 0.3, "block": None},
    "hin": {"review": 0.7, "block": None},
    "or": {"review": 0.3, "block": None},
}

PII_ACTION = os.getenv("PII_ACTION", "review")

EMOJI_REVIEW = ["\U0001F595"]  # middle finger
EMOJI_CONTEXT = [
    "\U0001F346",  # eggplant
    "\U0001F351",  # peach
    "\U0001F4A6",  # droplets
    "\U0001F51E",  # 18+
    "\U0001F445",  # tongue
]
EMOJI_CONTEXT_MIN_SCORE = 0.1
EMOJI_CONTEXT_MIN_DISTINCT = 2
EMOJI_TO_TEXT = os.getenv("EMOJI_TO_TEXT", "true").lower() in ("1", "true", "yes")

REVIEW_PHRASES = {
    "en": [],
    "hi": [],
    "hin": ["bc", "mc", "bsdk", "lauda"],
    "or": [],
}

try:
    import emoji as _emoji
except ImportError:
    _emoji = None

_NUM_RE = re.compile(r"(?<!\d)(?:\+?\d[\s\-.()]?){10,15}(?!\d)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PIN_RE = re.compile(r"(?<!\d)[1-9]\d{5}(?!\d)")
_HOUSE_RE = re.compile(r"\b(?:house|flat|plot|door|h)\.?\s*(?:no\.?|number|#)\s*\d+", re.I)
ADDRESS_CUES = [
    "road", "street", "lane", "nagar", "colony", "near", "pin", "sector",
    "apartment", "village", "district", "square",
]


def detect_lang(text: str) -> Tuple[str, str]:
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
        return "en", "default"
    return best, "detected"


def _ascii_digits(text: str) -> str:
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


def prepare_for_model(text: str) -> str:
    if EMOJI_TO_TEXT and _emoji is not None:
        return _emoji.replace_emoji(
            text, replace=lambda chars, data: " " + data["en"].strip(":").replace("_", " ") + " "
        )
    return text


def find_emoji_flags(text: str, toxicity: float) -> Tuple[List[str], List[str]]:
    always = [e for e in EMOJI_REVIEW if e in text]
    found = [e for e in EMOJI_CONTEXT if e in text]
    context = found if (toxicity >= EMOJI_CONTEXT_MIN_SCORE or len(found) >= EMOJI_CONTEXT_MIN_DISTINCT) else []
    return always, context


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s).casefold().split())


def _is_word_char(ch: str) -> bool:
    return ch == "_" or unicodedata.category(ch)[0] in ("L", "M", "N")


def phrase_in_text(text_norm: str, phrase_norm: str) -> bool:
    if not phrase_norm:
        return False
    check_start = _is_word_char(phrase_norm[0])
    check_end = _is_word_char(phrase_norm[-1])
    pos = 0
    while True:
        i = text_norm.find(phrase_norm, pos)
        if i < 0:
            return False
        j = i + len(phrase_norm)
        before_ok = (not check_start) or i == 0 or not _is_word_char(text_norm[i - 1])
        after_ok = (not check_end) or j == len(text_norm) or not _is_word_char(text_norm[j])
        if before_ok and after_ok:
            return True
        pos = i + 1


def make_chunks(ids: List[int], width: int, overlap: int) -> List[List[int]]:
    if len(ids) <= width:
        return [ids]
    step = max(width - overlap, 1)
    chunks = []
    for start in range(0, len(ids), step):
        chunks.append(ids[start:start + width])
        if start + width >= len(ids):
            break
    return chunks


class TextModerator:
    """Manages the NLP model and heuristic toxicity rules."""

    def __init__(
        self,
        model_dir: str = DEFAULT_TEXT_MODEL_DIR,
        s3_bucket: Optional[str] = DEFAULT_S3_BUCKET,
        s3_prefix: Optional[str] = DEFAULT_TEXT_S3_PREFIX,
    ) -> None:
        self.model_dir = model_dir
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        self._lock = threading.Lock()

        # 1. Automatic S3 download if missing locally
        self._ensure_model_available()

        # 2. Load model and tokenizer
        logger.info("Loading text toxicity model from %s on device=%s", self.model_dir, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(self.device).eval()
        self.id2label = self.model.config.id2label
        self.prefix_ids, self.suffix_ids = self._special_template()
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

    def _ensure_model_available(self) -> None:
        p = Path(self.model_dir)
        if not (p.is_dir() and (p / "config.json").exists()):
            logger.info("Text model not found in %s, attempting S3 pull...", self.model_dir)
            ensure_model_files(
                local_dir=self.model_dir,
                s3_bucket=self.s3_bucket,
                s3_prefix=self.s3_prefix,
                required_files=("config.json",),
            )

    def _special_template(self) -> Tuple[List[int], List[int]]:
        probe = "hello world"
        with_special = self.tokenizer(probe)["input_ids"]
        core = self.tokenizer(probe, add_special_tokens=False)["input_ids"]
        n = len(core)
        for i in range(len(with_special) - n + 1):
            if with_special[i:i + n] == core:
                return with_special[:i], with_special[i + n:]
        raise RuntimeError("Could not resolve tokenizer special tokens template")

    def score_texts(self, texts: List[str]) -> Tuple[List[dict], List[int]]:
        width = MAX_LEN - len(self.prefix_ids) - len(self.suffix_ids)
        all_ids = self.tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
        chunk_ids, owners = [], []
        for i, ids in enumerate(all_ids):
            for chunk in make_chunks(list(ids), width, LONG_TEXT_OVERLAP):
                chunk_ids.append(self.prefix_ids + chunk + self.suffix_ids)
                owners.append(i)

        best = [None] * len(texts)
        with self._lock, torch.no_grad():
            for s in range(0, len(chunk_ids), CHUNK_BATCH):
                group = chunk_ids[s:s + CHUNK_BATCH]
                longest = max(len(c) for c in group)
                input_ids = torch.tensor(
                    [c + [self.pad_id] * (longest - len(c)) for c in group], device=self.device
                )
                attention_mask = torch.tensor(
                    [[1] * len(c) + [0] * (longest - len(c)) for c in group], device=self.device
                )
                probs = torch.sigmoid(self.model(input_ids=input_ids, attention_mask=attention_mask).logits.float()).cpu()
                for row, owner in zip(probs, owners[s:s + CHUNK_BATCH]):
                    best[owner] = row if best[owner] is None else torch.maximum(best[owner], row)

        scores = [{self.id2label[i]: round(float(p), 4) for i, p in enumerate(row)} for row in best]
        counts = Counter(owners)
        return scores, [counts[i] for i in range(len(texts))]

    def decide(
        self,
        lang: str,
        toxicity: float,
        text: str,
        pii: List[str],
        emoji_always: List[str],
        emoji_context: List[str],
    ) -> Tuple[str, List[str]]:
        rule = LANG_RULES.get(lang, LANG_RULES["en"])
        reasons = []
        decision = "allow"

        if toxicity >= rule["review"]:
            decision = "review"
            reasons.append(f"toxicity {toxicity:.2f} >= review threshold {rule['review']}")
        if rule.get("block") is not None and toxicity >= rule["block"]:
            decision = "block"
            reasons.append(f"toxicity >= block threshold {rule['block']}")

        text_norm = _norm(text)
        for phrase in REVIEW_PHRASES.get(lang, []):
            if phrase_in_text(text_norm, _norm(phrase)):
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
            if len(emoji_context) >= EMOJI_CONTEXT_MIN_DISTINCT:
                reasons.append("several context emojis together: " + " ".join(emoji_context))
            else:
                reasons.append(f"context emoji with toxicity {toxicity:.2f} >= {EMOJI_CONTEXT_MIN_SCORE}: " + " ".join(emoji_context))

        return decision, reasons

    def moderate_many(self, items: List[Tuple[str, Optional[str]]]) -> List[dict[str, Any]]:
        langs = []
        for text, forced in items:
            if forced:
                langs.append((forced, "provided"))
            else:
                langs.append(detect_lang(text))

        all_scores, chunk_counts = self.score_texts([prepare_for_model(t) for t, _ in items])
        out = []
        for (text, _), (lang, source), scores, n_chunks in zip(items, langs, all_scores, chunk_counts):
            tox = scores.get("toxicity", 0.0)
            pii = find_pii(text) if PII_ACTION != "ignore" else []
            e_always, e_context = find_emoji_flags(text, tox)
            decision, reasons = self.decide(lang, tox, text, pii, e_always, e_context)
            out.append({
                "decision": decision,
                "toxicity": tox,
                "language": lang,
                "language_source": source,
                "review_threshold": LANG_RULES.get(lang, {}).get("review", 0.5),
                "reasons": reasons,
                "personal_details": pii,
                "chunks": n_chunks,
                "flagged_emojis": e_always + e_context,
                "scores": scores,
            })
        return out

    def moderate_single(self, text: str, lang: Optional[str] = None) -> dict[str, Any]:
        return self.moderate_many([(text, lang)])[0]
