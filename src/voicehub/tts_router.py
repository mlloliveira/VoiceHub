from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import (
    DEFAULT_MAX_CHARS_PER_CHUNK,
    DEFAULT_QWEN_CHUNK_SIZE,
    DEFAULT_XTTS_CLONE_REF_SECONDS,
    DEFAULT_QWEN_CLONE_REF_SECONDS,
    CLONE_REF_SECONDS_MAX,
    CLONE_REF_SECONDS_MIN,
    QWEN_LANGUAGE_NAME_MAP,
    QWEN_SUPPORTED_CODES,
    XTTS_CHUNK_MAX,
    XTTS_CHUNK_MIN,
    QWEN_CHUNK_MAX,
    QWEN_CHUNK_MIN,
    _LANG_LABELS,
)

TTS_FAMILIES = ["Qwen", "XTTS"]
TTS_FAMILY_MAP = {"Qwen": "qwen", "XTTS": "xtts", "qwen": "qwen", "xtts": "xtts"}
TTS_FAMILY_DISPLAY = {"qwen": "Qwen", "xtts": "XTTS"}

QWEN_CUSTOM = "qwen_custom"
QWEN_CLONE = "qwen_clone"
XTTS_BACKEND = "xtts"

QWEN_DEFAULT_VOICE_BY_CODE: Dict[str, str] = {
    "zh-cn": "Vivian",
    "en": "Ryan",
    "ja": "Ono_Anna",
    "ko": "Sohee",
    "de": "Ryan",
    "fr": "Ryan",
    "it": "Ryan",
    "pt": "Ryan",
    "es": "Ryan",
    "ru": "Ryan",
}

QWEN_SPEAKER_ORDER = [
    "Ryan",
    "Aiden",
    "Vivian",
    "Serena",
    "Ono_Anna",
    "Sohee",
    "Uncle_Fu",
    "Dylan",
    "Eric",
]

XTTS_DEFAULT_PRIORITY = [
    # female-leaning / American-leaning first when available
    "Ana Florence",
    "Claribel Dervla",
    "Daisy Studious",
    "Tammie Ema",
    "Gracie Wise",
    "Gitta Nikolina",
    "Brenda Stern",
    "Ava",
]

XTTS_CURATED_PRIORITY = XTTS_DEFAULT_PRIORITY + [
    "Aaron Dreschner",
    "Aaron Dreshner",
    "Craig Gutsy",
    "Damien Black",
    "Tom",
    "Emma",
    "Bella",
    "Josh",
    "Sam",
    "Sonia",
    "Arnold",
    "Rosemary",
]


@dataclass(frozen=True)
class TTSRoute:
    preferred_family: str
    resolved_family: str
    backend: str
    language_code: str
    qwen_language: Optional[str]
    fallback_used: bool = False
    fallback_reason: str = ""


def normalize_family(value: str | None) -> str:
    return TTS_FAMILY_MAP.get((value or "").strip(), "xtts")


def qwen_supports_language(code: str | None) -> bool:
    return (code or "").lower() in QWEN_SUPPORTED_CODES


def qwen_language_name_for_code(code: str | None) -> Optional[str]:
    return QWEN_LANGUAGE_NAME_MAP.get((code or "").lower())


def resolve_tts_route(preferred_family: str | None, language_code: str | None, has_reference: bool) -> TTSRoute:
    family = normalize_family(preferred_family)
    code = (language_code or "en").lower()
    if family == "qwen" and qwen_supports_language(code):
        return TTSRoute(
            preferred_family=family,
            resolved_family="qwen",
            backend=QWEN_CLONE if has_reference else QWEN_CUSTOM,
            language_code=code,
            qwen_language=qwen_language_name_for_code(code),
            fallback_used=False,
            fallback_reason="",
        )

    if family == "qwen":
        return TTSRoute(
            preferred_family=family,
            resolved_family="xtts",
            backend=XTTS_BACKEND,
            language_code=code,
            qwen_language=None,
            fallback_used=True,
            fallback_reason="unsupported_language",
        )

    return TTSRoute(
        preferred_family=family,
        resolved_family="xtts",
        backend=XTTS_BACKEND,
        language_code=code,
        qwen_language=None,
        fallback_used=False,
        fallback_reason="",
    )


def format_backend_status(language_code: str, route: TTSRoute, *, detected: bool = True) -> str:
    label = _LANG_LABELS.get(language_code, language_code)
    prefix = "Detected" if detected else "Using"
    if route.fallback_used and route.resolved_family == "xtts":
        return f"✅ {prefix} {label} ({language_code}). Falling back to XTTS as backend model."
    if route.resolved_family == "qwen":
        return f"✅ {prefix} {label} ({language_code}). Using Qwen-TTS as backend model."
    return f"✅ {prefix} {label} ({language_code}). Using XTTS as backend model."


def qwen_default_voice_for_code(code: str | None) -> str:
    return QWEN_DEFAULT_VOICE_BY_CODE.get((code or "").lower(), "Ryan")


def qwen_voice_choices() -> List[str]:
    return ["Default"] + QWEN_SPEAKER_ORDER


def all_backend_voice_choices(available_xtts: Sequence[str]) -> List[str]:
    choices = ["Default"]
    for name in QWEN_SPEAKER_ORDER:
        choices.append(label_voice(name, "qwen"))
    xtts_choices, _ = xtts_voice_choices(available_xtts)
    for name in xtts_choices[1:]:
        choices.append(label_voice(name, "xtts"))
    return choices


def resolve_qwen_voice(selection: str | None, language_code: str | None, saved_default: str | None = None) -> str:
    selected = (selection or "Default").strip()
    if selected and selected != "Default" and selected in QWEN_SPEAKER_ORDER:
        return selected
    if saved_default and saved_default in QWEN_SPEAKER_ORDER:
        return saved_default
    return qwen_default_voice_for_code(language_code)


def curated_xtts_voices(available: Sequence[str], *, max_items: int = 12) -> List[str]:
    seen = set()
    curated: List[str] = []
    avail_map = {name.lower(): name for name in available}
    for candidate in XTTS_CURATED_PRIORITY:
        match = avail_map.get(candidate.lower())
        if match and match.lower() not in seen:
            curated.append(match)
            seen.add(match.lower())
    for name in available:
        if len(curated) >= max_items:
            break
        if name.lower() not in seen:
            curated.append(name)
            seen.add(name.lower())
    return curated


def resolve_xtts_default_voice(available: Sequence[str]) -> Optional[str]:
    if not available:
        return None
    avail_map = {name.lower(): name for name in available}
    for candidate in XTTS_DEFAULT_PRIORITY:
        match = avail_map.get(candidate.lower())
        if match:
            return match
    return available[0]


def label_voice(name: str, family: str) -> str:
    fam = TTS_FAMILY_DISPLAY.get(normalize_family(family), family)
    return f"{name} ({fam})" if name != "Default" else name


def xtts_voice_choices(available: Sequence[str]) -> Tuple[List[str], Optional[str]]:
    curated = curated_xtts_voices(list(available))
    default_voice = resolve_xtts_default_voice(curated) if curated else None
    choices = ["Default"] + curated if curated else ["Default"]
    return choices, default_voice


def resolve_xtts_voice(selection: str | None, available: Sequence[str]) -> Optional[str]:
    selected = (selection or "Default").strip()
    curated = curated_xtts_voices(list(available))
    if selected and selected != "Default" and selected in curated:
        return selected
    return resolve_xtts_default_voice(curated)


def speed_bucket_prompt(speed: float, language_code: str | None = None) -> str:
    speed = float(speed)
    if speed <= 0.75:
        return "Speak much more slowly than usual, with very clear pronunciation and gentle pauses between phrases."
    if speed <= 0.90:
        return "Speak a bit slower than usual, with clear pronunciation and calm pacing."
    if speed < 1.10:
        return ""
    if speed <= 1.25:
        return "Speak slightly faster than usual, while staying natural, smooth, and easy to understand."
    return "Speak clearly at a fast pace, but keep every word intelligible and do not rush or slur syllables."


def chunk_slider_state_for_family(
    family: str,
    *,
    xtts_value: int | None = None,
    qwen_value: int | None = None,
) -> dict:
    fam = normalize_family(family)
    if fam == "qwen":
        return {
            "minimum": QWEN_CHUNK_MIN,
            "maximum": QWEN_CHUNK_MAX,
            "value": int(qwen_value) if qwen_value is not None else DEFAULT_QWEN_CHUNK_SIZE,
            "label": "Chunk size cap / threshold",
            "info": "Qwen: paragraph-friendly chunk target/cap. Internal max_new_tokens stays conservative.",
        }
    return {
        "minimum": XTTS_CHUNK_MIN,
        "maximum": XTTS_CHUNK_MAX,
        "value": int(xtts_value) if xtts_value is not None else DEFAULT_MAX_CHARS_PER_CHUNK,
        "label": "Chunk size cap / threshold",
        "info": "XTTS: fixed cap or language-aware threshold, depending on the dynamic toggle.",
    }


def clone_ref_slider_state_for_family(
    family: str,
    *,
    xtts_value: float | int | None = None,
    qwen_value: float | int | None = None,
) -> dict:
    fam = normalize_family(family)
    if fam == "qwen":
        return {
            "minimum": CLONE_REF_SECONDS_MIN,
            "maximum": CLONE_REF_SECONDS_MAX,
            "value": float(qwen_value) if qwen_value is not None else float(DEFAULT_QWEN_CLONE_REF_SECONDS),
            "label": "Reference audio cap for voice cloning (seconds)",
            "info": "Qwen: uploaded clone reference audio is automatically trimmed to this cap before transcript generation and synthesis.",
        }
    return {
        "minimum": CLONE_REF_SECONDS_MIN,
        "maximum": CLONE_REF_SECONDS_MAX,
        "value": float(xtts_value) if xtts_value is not None else float(DEFAULT_XTTS_CLONE_REF_SECONDS),
        "label": "Reference audio cap for voice cloning (seconds)",
        "info": "XTTS: uploaded clone reference audio is automatically trimmed to this cap before synthesis.",
    }
