# src/voicehub/ollama_config.py
import os
from .prefs import get_pref, set_pref

_PUBLIC_FALLBACK = "gemma3:12b"
_env_model = os.getenv("OLLAMA_MODEL") or os.getenv("OLLAMA_MODEL_DEFAULT")

OLLAMA_ENABLE_DEFAULT = os.getenv("OLLAMA_ENABLE", "0") == "1"
OLLAMA_MODEL_DEFAULT = get_pref("ollama_model_default", _env_model or _PUBLIC_FALLBACK)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))
OLLAMA_MAX_SEG_CHARS = int(os.getenv("OLLAMA_MAX_SEG_CHARS", "200"))

OLLAMA_PRECHUNK_PROMPT_XTTS = """You will receive a piece of text that will be spoken by XTTS, which is sensitive to long or poorly punctuated segments. Follow these rules exactly:

1. Replace emojis with words.
2. Split the text into segments of {max_chars} characters or less.
3. Each segment must end with proper punctuation.
4. Prefer sentence boundaries. If a long sentence must be split, create a natural sentence boundary with a period.
5. Keep the original wording and order. Do not paraphrase, summarize, or reorder.
6. Output one segment per line, with no numbering and no commentary.
7. Preserve connecting words like And / But when they help the flow.
8. Remove weird non-text artifacts and read-safe punctuation problems only.

Input text:
"""

OLLAMA_PRECHUNK_PROMPT_QWEN = """You will receive a piece of text that will be spoken by Qwen-TTS, which handles longer chunks better than XTTS. Your goal is to preserve paragraph flow while still making safe speaking segments.

Rules:
1. Replace emojis with words.
2. Clean weird non-text artifacts, broken symbols, and obvious OCR-like garbage.
3. Prefer keeping sentences together and preserve paragraph flow.
4. Only split when needed for a natural speaking pause or when a segment would become too long.
5. Aim for segments of {max_chars} characters or less, but prioritize natural paragraph rhythm over aggressive punctuation splitting.
6. Keep the original wording and order. Do not paraphrase, summarize, or reorder.
7. Output one segment per line, with no numbering and no commentary.
8. Choose safe semantic break points, not every punctuation mark.

Input text:
"""

OLLAMA_TRANSLATE_PROMPT = """You are a precise translation engine.

Rules:
1) Translate the INPUT text to {target_lang_name}.
2) Preserve meaning, numbers, names, URLs, and formatting markers.
3) Fix obvious punctuation only if needed for fluency; do not add commentary.
4) Do not explain. Output ONLY the translated text.

INPUT text:
"""


def set_ollama_default_model(model: str) -> str:
    m = (model or "").strip()
    if not m:
        return "⚠️ Empty model tag — nothing saved."
    set_pref("ollama_model_default", m)
    return f"✅ Saved default Ollama model: **{m}**"
