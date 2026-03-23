# src/voicehub/lang_detect.py
from __future__ import annotations
from typing import Optional, Tuple

from .config import _LANG_LABELS as CODE2LABEL

SUPPORTED_XTTS_CODES = list(CODE2LABEL.keys())
LANGID_CODES = [("zh" if c == "zh-cn" else c) for c in SUPPORTED_XTTS_CODES]
DETECT2XTTS = {c: c for c in LANGID_CODES}
DETECT2XTTS["zh"] = "zh-cn"

LANGID_AVAILABLE = True
IDENTIFIER = None
try:
    import langid
    IDENTIFIER = langid.langid.LanguageIdentifier.from_modelstring(langid.langid.model, norm_probs=True)
    try:
        IDENTIFIER.set_languages(LANGID_CODES)
    except Exception:
        pass
except Exception:
    LANGID_AVAILABLE = False
    IDENTIFIER = None


def detect_tts_language(text: str, min_conf: float = 0.80) -> Tuple[Optional[str], Optional[str], float, str]:
    txt = (text or "").strip()
    if not txt:
        return None, None, 0.0, "⚠️ Empty text."
    if not LANGID_AVAILABLE or IDENTIFIER is None:
        return None, None, 0.0, "⚠️ Language detector not installed."

    try:
        code_raw, score = IDENTIFIER.classify(txt)
    except Exception as e:
        return None, None, 0.0, f"⚠️ Detection failed: {e}"

    xtts_code = DETECT2XTTS.get(code_raw)
    if not xtts_code or xtts_code not in SUPPORTED_XTTS_CODES:
        return None, None, float(score), f"⚠️ Detected unsupported language: {code_raw} ({score:.2f})."

    label = CODE2LABEL.get(xtts_code, xtts_code)
    note = f"✅ Detected **{label}** ({xtts_code}) with confidence **{score:.2f}**."
    if score < float(min_conf):
        note += " (low confidence; consider setting language manually)"
    return label, xtts_code, float(score), note


def detect_tts_language_display(text: str) -> Tuple[str, str]:
    label, code, score, note = detect_tts_language(text)
    if not code:
        return "English", note
    return CODE2LABEL.get(code, "English"), note
