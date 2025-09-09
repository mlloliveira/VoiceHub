# src/voicehub/config.py
import os

# ASR knobs (kept from your comments)
ASR_MODEL_NAME = os.getenv("ASR_MODEL", "large-v3")  # Defaults to Whisper large-v3.
ASR_COMPUTE = "int8_float16" if os.getenv("ASR_INT8", "0") == "1" else "float16"  # Compute is FP16 by default; int8_float16 uses 8-bit weights + FP16 activations to reduce VRAM and boost throughput (CTranslate2 feature)

BACKENDS = ["Faster-Whisper (GPU, recommended)", "OpenAI Whisper (GPU)"]
BACKEND_MAP = {"Faster-Whisper (GPU, recommended)": "fw", "OpenAI Whisper (GPU)": "ow"}

# Languages supported by XTTS-v2 (17 total)
_LANG_LABELS = {
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "it": "Italiano",
    "pt": "Português",
    "pl": "Polski",
    "tr": "Türkçe",
    "ru": "Русский",
    "nl": "Nederlands",
    "cs": "Čeština",
    "ar": "العربية",
    "zh-cn": "中文 (简体)",
    "ja": "日本語",
    "hu": "Magyar",
    "ko": "한국어",
    "hi": "हिन्दी",
}

def _parse_codes(var_value: str, default_csv: str):
    raw = (var_value or default_csv).split(",")
    return [c.strip().lower() for c in raw if c.strip()]


# --- XTTS per-language character caps (model-informed limits) ---

PER_LANG_CHAR_CAPS = {
    "en": 250, "de": 253, "fr": 273, "es": 239, "it": 213, "pt": 203,
    "pl": 224, "tr": 226, "ru": 182, "nl": 251, "cs": 186, "ar": 166,
    "zh": 82,  "ja": 71,  "hu": 224, "ko": 95, "hi": 250,
}

def _base_lang(code: str) -> str:
    # 'zh-cn' -> 'zh', 'pt' -> 'pt'
    if not code:
        return "en"
    c = code.lower()
    if c.startswith("zh"):
        return "zh"
    return c.split("-")[0]

def _bucket_down_10(n: int, floor_min: int = 50) -> int:
    """
    Bucket to the next lower multiple of 10.
    If already a multiple of 10, step down one bucket (e.g., 250 -> 240).
    Enforce a minimal sane floor to avoid tiny values (default 50).
    """
    n = int(n)
    if n <= floor_min:
        return floor_min
    q, r = divmod(n, 10)
    bucket = (q - 1) * 10 if r == 0 else q * 10
    return max(floor_min, bucket)

def dynamic_cap_for_lang(code: str) -> int:
    """
    Language-aware cap with 'next lower multiple of 10' bucketing.
    Examples: it(213)->210, tr(226)->220, en(250)->240, fr(273)->270, zh(82)->80, ja(71)->70.
    """
    base = _base_lang(code)
    cap = PER_LANG_CHAR_CAPS.get(base, 250)
    return _bucket_down_10(cap)

def effective_max_chars(lang_code: str, user_cap: int, dynamic: bool) -> int:
    """
    If dynamic=True → use the language-specific bucketed cap.
    If dynamic=False → use the user slider as-is.
    """
    if dynamic:
        return dynamic_cap_for_lang(lang_code)
    return int(user_cap)

# ---- ASR languages (dynamic; includes "auto") ----
_ASR_CODES = _parse_codes(os.getenv("ASR_LANGS"), "auto,en,pt,ru")
ASR_LANG_DISPLAY = []
ASR_LANG_MAP = {}  # display -> code (None for auto)
for code in _ASR_CODES:
    if code == "auto":
        label = "Auto-detect"
        ASR_LANG_DISPLAY.append(label)
        ASR_LANG_MAP[label] = None
    else:
        label = _LANG_LABELS.get(code, code)
        ASR_LANG_DISPLAY.append(label)
        ASR_LANG_MAP[label] = code

# ---- TTS languages (dynamic) ----
_TTS_CODES = _parse_codes(os.getenv("TTS_LANGS"), "en,pt")
TTS_LANG_DISPLAY = []
TTS_LANG_MAP = {}  # display -> code
for code in _TTS_CODES:
    label = _LANG_LABELS.get(code, code)
    TTS_LANG_DISPLAY.append(label)
    TTS_LANG_MAP[label] = code

# TTS knobs
TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_MAX_CHARS_PER_CHUNK = int(os.getenv("TTS_MAX_CHARS", "210"))  # keep < 250 for 'en' and < 203 for 'pt'
DEFAULT_TTS_MAX_MINUTES = 5.0
DEFAULT_TTS_SPEED = 1.0

# Dev-only UI (hidden unless you opt in)
def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}
DEBUG_TOOLS = _env_flag("DEBUG_TOOLS", default=False) # Dev-only