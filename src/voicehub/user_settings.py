# src/voicehub/user_settings.py
# Centralized user-tunable settings (per model) with sane defaults.
# Other modules (ASR, TTS, Ollama) read these at call time.

from dataclasses import dataclass, asdict, fields
from typing import Dict
from .config import DEFAULT_MAX_CHARS_PER_CHUNK, DEFAULT_TTS_MAX_MINUTES  # <- tie to config constant
from .prefs import get_pref, set_pref

@dataclass
class UserSettings:
    # ---- Whisper (ASR) ----
    whisper_temperature: float = 0.0      # low randomness by default
    whisper_top_p: float = 0.9            # nucleus sampling (if backend supports)
    whisper_beam_size: int = 5            # default beams (used to init ASR slider)
    whisper_condition_on_prev: bool = True
    asr_stream_max_minutes: float = 5.0

    # ---- XTTS (TTS) ----
    # Only the global chunk size lives here; per-synthesis speed/minutes are TTS-tab only.
    xtts_max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK # keep < 250 for 'en' and < 203 for 'pt'
    xtts_max_minutes_default: float = DEFAULT_TTS_MAX_MINUTES
    xtts_dynamic_per_lang_caps: bool = True

    # ---- Ollama (LLM pre-chunker) ----
    ollama_temperature: float = 0.7
    ollama_top_p: float = 0.9
    ollama_num_predict: int = 4096         # token cap for refinement
    ollama_stop: str = ""                  # comma-separated substrings; optional

# Global mutable settings (kept simple on purpose)
_SETTINGS = UserSettings()

###
_PREFS_KEY = "settings"

def _coerce(name: str, val):
    """Best-effort type coercion to match dataclass field types."""
    ftypes = {f.name: f.type for f in fields(UserSettings)}
    t = ftypes.get(name)
    try:
        if t is bool:
            if isinstance(val, str):  # 'true'/'false' etc.
                return val.strip().lower() in {"1","true","yes","on"}
            return bool(val)
        if t is int:
            return int(val)
        if t is float:
            return float(val)
        if t is str:
            return str(val)
    except Exception:
        pass
    return val

def _load_settings_from_prefs():
    global _SETTINGS
    data = get_pref(_PREFS_KEY, {}) or {}
    if not isinstance(data, dict):
        return
    cur = asdict(_SETTINGS)
    for k, v in data.items():
        if k in cur:
            cur[k] = _coerce(k, v)
    _SETTINGS = UserSettings(**cur)

def _save_settings_to_prefs():
    set_pref(_PREFS_KEY, asdict(_SETTINGS))

# Load once at import
_load_settings_from_prefs()
###

def get_settings() -> UserSettings:
    return _SETTINGS

def update_settings(**kwargs) -> Dict:
    for k, v in kwargs.items():
        if hasattr(_SETTINGS, k):
            setattr(_SETTINGS, k, _coerce(k, v))
    _save_settings_to_prefs()
    return asdict(_SETTINGS)

def reset_settings() -> Dict:
    global _SETTINGS
    _SETTINGS = UserSettings()
    _save_settings_to_prefs()
    return asdict(_SETTINGS)
