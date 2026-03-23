# src/voicehub/config.py
import os

VOICEHUB_VERSION = "0.2.0"

# ASR knobs
ASR_MODEL_NAME = os.getenv("ASR_MODEL", "turbo")
ASR_COMPUTE = "int8_float16" if os.getenv("ASR_INT8", "0") == "1" else "float16"

BACKENDS = ["Faster-Whisper (GPU, recommended)", "OpenAI Whisper (GPU)"]
BACKEND_MAP = {"Faster-Whisper (GPU, recommended)": "fw", "OpenAI Whisper (GPU)": "ow"}

# Languages supported by XTTS-v2
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

# Qwen supported languages
QWEN_SUPPORTED_CODES = {"zh-cn", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"}
QWEN_LANGUAGE_NAME_MAP = {
    "zh-cn": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}

# XTTS per-language character caps
PER_LANG_CHAR_CAPS = {
    "en": 250, "de": 253, "fr": 273, "es": 239, "it": 213, "pt": 203,
    "pl": 224, "tr": 226, "ru": 182, "nl": 251, "cs": 186, "ar": 166,
    "zh": 82,  "ja": 71,  "hu": 224, "ko": 95, "hi": 250,
}


def _parse_codes(var_value: str, default_csv: str):
    raw = (var_value or default_csv).split(",")
    return [c.strip().lower() for c in raw if c.strip()]


def _base_lang(code: str) -> str:
    if not code:
        return "en"
    c = code.lower()
    if c.startswith("zh"):
        return "zh"
    return c.split("-")[0]


def _bucket_down_10(n: int, floor_min: int = 50) -> int:
    n = int(n)
    if n <= floor_min:
        return floor_min
    q, r = divmod(n, 10)
    bucket = (q - 1) * 10 if r == 0 else q * 10
    return max(floor_min, bucket)


def dynamic_cap_for_lang(code: str) -> int:
    base = _base_lang(code)
    cap = PER_LANG_CHAR_CAPS.get(base, 250)
    return _bucket_down_10(cap)


def effective_max_chars(lang_code: str, user_cap: int, dynamic: bool) -> int:
    manual_cap = max(1, int(user_cap))
    if dynamic:
        return min(dynamic_cap_for_lang(lang_code), manual_cap)
    return manual_cap


# ---- ASR languages ----
_ASR_CODES = _parse_codes(os.getenv("ASR_LANGS"), "auto,en,pt,ru")
ASR_LANG_DISPLAY = []
ASR_LANG_MAP = {}
for code in _ASR_CODES:
    if code == "auto":
        label = "Auto-detect"
        ASR_LANG_DISPLAY.append(label)
        ASR_LANG_MAP[label] = None
    else:
        label = _LANG_LABELS.get(code, code)
        ASR_LANG_DISPLAY.append(label)
        ASR_LANG_MAP[label] = code

# ---- TTS languages ----
_TTS_CODES = _parse_codes(os.getenv("TTS_LANGS"), "en,pt,es,fr,de,it,ru,ja,ko,zh-cn,ar,nl,pl,tr,cs,hu,hi")
TTS_LANG_DISPLAY = ["Auto-detect"]
TTS_LANG_MAP = {"Auto-detect": None}
for code in _TTS_CODES:
    label = _LANG_LABELS.get(code, code)
    if label not in TTS_LANG_MAP:
        TTS_LANG_DISPLAY.append(label)
        TTS_LANG_MAP[label] = code

# TTS knobs / model ids
TTS_MODEL_NAME = os.getenv("TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
QWEN_MODEL_SIZE_OPTIONS = ("1.7B", "0.6B")
QWEN_DEFAULT_MODEL_SIZE = os.getenv("QWEN_MODEL_SIZE", "1.7B")
QWEN_CUSTOM_MODEL_NAME = os.getenv("QWEN_CUSTOM_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
QWEN_CLONE_MODEL_NAME = os.getenv("QWEN_CLONE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
QWEN_TOKENIZER_MODEL_NAME = os.getenv("QWEN_TOKENIZER_MODEL", "Qwen/Qwen3-TTS-Tokenizer-12Hz")

def qwen_model_names_for_size(size: str | None):
    normalized = (size or QWEN_DEFAULT_MODEL_SIZE or "1.7B").strip()
    if normalized not in QWEN_MODEL_SIZE_OPTIONS:
        normalized = "1.7B"
    return {
        "size": normalized,
        "custom": f"Qwen/Qwen3-TTS-12Hz-{normalized}-CustomVoice",
        "clone": f"Qwen/Qwen3-TTS-12Hz-{normalized}-Base",
        "tokenizer": QWEN_TOKENIZER_MODEL_NAME,
    }

DEFAULT_MAX_CHARS_PER_CHUNK = int(os.getenv("TTS_MAX_CHARS", "200"))
DEFAULT_QWEN_CHUNK_SIZE = int(os.getenv("QWEN_TTS_MAX_CHARS", "512"))
DEFAULT_QWEN_MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "1024"))
DEFAULT_TTS_MAX_MINUTES = 10.0
DEFAULT_TTS_SPEED = 1.0
DEFAULT_XTTS_CLONE_REF_SECONDS = float(os.getenv("XTTS_CLONE_REF_SECONDS", "300"))
DEFAULT_QWEN_CLONE_REF_SECONDS = float(os.getenv("QWEN_CLONE_REF_SECONDS", "50"))
CLONE_REF_SECONDS_MIN = 5
CLONE_REF_SECONDS_MAX = 600
XTTS_CHUNK_MIN = 60
XTTS_CHUNK_MAX = 400
QWEN_CHUNK_MIN = 120
QWEN_CHUNK_MAX = 2048

# Dev-only UI

def _env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


DEBUG_TOOLS = _env_flag("DEBUG_TOOLS", default=False)
