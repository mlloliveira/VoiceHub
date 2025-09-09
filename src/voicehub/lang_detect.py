# src/voicehub/lang_detect.py
# Lightweight, XTTS-focused language detection utilities.
# Uses langid (pure-Python) and restricts detection to languages present in _LANG_LABELS.

from __future__ import annotations
from typing import Tuple, Optional

# Source of truth for supported languages & display labels
from .config import _LANG_LABELS as CODE2LABEL  # e.g., {"en":"English", "pt":"Português", ..., "zh-cn":"中文 (简体)"}

SUPPORTED_XTTS_CODES = list(CODE2LABEL.keys())  # e.g., ["en", "es", ..., "zh-cn", ...]
# langid expects "zh", not "zh-cn"
LANGID_CODES = [("zh" if c == "zh-cn" else c) for c in SUPPORTED_XTTS_CODES]

# Map langid -> XTTS (identity for most; special-case Chinese)
DETECT2XTTS = {c: c for c in LANGID_CODES}
DETECT2XTTS["zh"] = "zh-cn"  # normalize to XTTS code

LANGID_AVAILABLE = True
IDENTIFIER = None
# try:
#     import langid
#     IDENTIFIER = langid.LanguageIdentifier.from_modelstring(langid.model, norm_probs=True)
#     print(IDENTIFIER)
#     # Restrict detector to exactly the languages we support
#     try:
#         IDENTIFIER.set_languages(LANGID_CODES)
#     except Exception:
#         pass
# except Exception:
#     LANGID_AVAILABLE = False
#     IDENTIFIER = None

import langid
IDENTIFIER = langid.langid.LanguageIdentifier.from_modelstring(langid.langid.model, norm_probs=True)


def detect_tts_language(text: str, min_conf: float = 0.80) -> Tuple[Optional[str], Optional[str], float, str]:
    """
    Detect language of 'text' limited to the languages present in CODE2LABEL.
    Returns:
      (display_label, xtts_code, confidence[0,1], note_markdown)
    """
    txt = (text or "").strip()
    if not txt:
        return None, None, 0.0, "⚠️ Empty text."
    if not LANGID_AVAILABLE or IDENTIFIER is None:
        return None, None, 0.0, "⚠️ Language detector not installed."

    try:
        code_raw, score = IDENTIFIER.classify(txt)   # score is normalized 0..1. Lang is 'es', 'en', 'zh', ...
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
    """
    UI helper: returns (dropdown_value, note_markdown).
    Falls back to 'English' if detection fails.
    """
    label, code, score, note = detect_tts_language(text)
    if not code:
        return "English", note
    # Return the display label used in your dropdown
    return CODE2LABEL.get(code, "English"), note
