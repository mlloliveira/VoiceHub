# src/voicehub/user_settings.py
from dataclasses import dataclass, asdict, fields
from typing import Dict

from .config import (
    DEFAULT_MAX_CHARS_PER_CHUNK,
    DEFAULT_QWEN_CHUNK_SIZE,
    DEFAULT_TTS_MAX_MINUTES,
    DEFAULT_XTTS_CLONE_REF_SECONDS,
    DEFAULT_QWEN_CLONE_REF_SECONDS,
    QWEN_DEFAULT_MODEL_SIZE,
    QWEN_MODEL_SIZE_OPTIONS,
    VOICEHUB_VERSION,
)
from .prefs import get_pref, set_pref, prefs_exist


@dataclass
class UserSettings:
    # ---- Whisper (ASR) ----
    whisper_temperature: float = 0.0
    whisper_top_p: float = 0.9
    whisper_beam_size: int = 5
    whisper_condition_on_prev: bool = True
    asr_stream_max_minutes: float = 5.0

    # ---- TTS ----
    tts_default_family: str = "XTTS"
    xtts_max_chars_per_chunk: int = DEFAULT_MAX_CHARS_PER_CHUNK
    qwen_max_chars_per_chunk: int = DEFAULT_QWEN_CHUNK_SIZE
    qwen_model_size: str = QWEN_DEFAULT_MODEL_SIZE
    xtts_clone_ref_max_seconds: float = DEFAULT_XTTS_CLONE_REF_SECONDS
    qwen_clone_ref_max_seconds: float = DEFAULT_QWEN_CLONE_REF_SECONDS
    xtts_max_minutes_default: float = DEFAULT_TTS_MAX_MINUTES
    xtts_dynamic_per_lang_caps: bool = True
    qwen_voice_default: str = "Ryan"

    # ---- Ollama ----
    ollama_temperature: float = 0.7
    ollama_top_p: float = 0.9
    ollama_num_predict: int = 4096
    ollama_stop: str = ""

    # ---- bookkeeping ----
    voicehub_version: str = VOICEHUB_VERSION


_SETTINGS = UserSettings()
_PREFS_KEY = "settings"


def _coerce(name: str, val):
    ftypes = {f.name: f.type for f in fields(UserSettings)}
    t = ftypes.get(name)
    try:
        if t is bool:
            if isinstance(val, str):
                return val.strip().lower() in {"1", "true", "yes", "on"}
            return bool(val)
        if t is int:
            return int(val)
        if t is float:
            return float(val)
        if t is str:
            s = str(val)
            if name == "qwen_model_size" and s not in QWEN_MODEL_SIZE_OPTIONS:
                return QWEN_DEFAULT_MODEL_SIZE
            return s
    except Exception:
        pass
    return val


def _should_migrate_old_user(data: dict) -> bool:
    return prefs_exist() and isinstance(data, dict) and bool(data) and "tts_default_family" not in data


def _load_settings_from_prefs():
    global _SETTINGS
    data = get_pref(_PREFS_KEY, {}) or {}
    if not isinstance(data, dict):
        return
    cur = asdict(_SETTINGS)
    if _should_migrate_old_user(data):
        cur["tts_default_family"] = "XTTS"
    for k, v in data.items():
        if k in cur:
            cur[k] = _coerce(k, v)
    cur["voicehub_version"] = VOICEHUB_VERSION
    _SETTINGS = UserSettings(**cur)
    _save_settings_to_prefs()


def _save_settings_to_prefs():
    payload = asdict(_SETTINGS)
    payload["voicehub_version"] = VOICEHUB_VERSION
    set_pref(_PREFS_KEY, payload)


_load_settings_from_prefs()


def get_settings() -> UserSettings:
    return _SETTINGS


def apply_runtime_settings(**kwargs) -> Dict:
    """Update the in-memory settings for the current session without saving to disk."""
    for k, v in kwargs.items():
        if hasattr(_SETTINGS, k):
            setattr(_SETTINGS, k, _coerce(k, v))
    _SETTINGS.voicehub_version = VOICEHUB_VERSION
    return asdict(_SETTINGS)


def update_settings(**kwargs) -> Dict:
    apply_runtime_settings(**kwargs)
    _save_settings_to_prefs()
    return asdict(_SETTINGS)


def reset_settings() -> Dict:
    global _SETTINGS
    # respect old-user migration only on initial load; reset means new recommended defaults
    _SETTINGS = UserSettings()
    _SETTINGS.voicehub_version = VOICEHUB_VERSION
    _save_settings_to_prefs()
    return asdict(_SETTINGS)
