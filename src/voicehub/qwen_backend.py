from __future__ import annotations

from threading import Event
from typing import Optional, Sequence

import numpy as np

from .config import DEFAULT_QWEN_MAX_NEW_TOKENS, qwen_model_names_for_size
from .user_settings import get_settings

_QWEN_IMPORT_ERROR: Optional[Exception] = None
_QWEN_MODEL_CLASS = None
_CUSTOM_MODEL = None
_CLONE_MODEL = None
_CUSTOM_MODEL_ID = None
_CLONE_MODEL_ID = None


def _import_qwen():
    global _QWEN_MODEL_CLASS, _QWEN_IMPORT_ERROR
    if _QWEN_MODEL_CLASS is not None:
        return _QWEN_MODEL_CLASS
    try:
        from qwen_tts import Qwen3TTSModel
        _QWEN_MODEL_CLASS = Qwen3TTSModel
        return _QWEN_MODEL_CLASS
    except Exception as exc:
        _QWEN_IMPORT_ERROR = exc
        raise


def qwen_available() -> tuple[bool, str]:
    try:
        _import_qwen()
        return True, "qwen-tts import OK"
    except Exception as exc:
        return False, f"qwen-tts unavailable: {exc}"


def _torch_runtime_kwargs() -> dict:
    try:
        import torch
    except Exception:
        return {}

    use_cuda = bool(torch.cuda.is_available())
    dtype = None
    if use_cuda:
        dtype = getattr(torch, "bfloat16", None) or getattr(torch, "float16", None)
    else:
        dtype = getattr(torch, "float32", None)

    kwargs = {"dtype": dtype}
    kwargs["device_map"] = "cuda:0" if use_cuda else "cpu"
    if use_cuda:
        try:
            import flash_attn  # noqa: F401
            kwargs["attn_implementation"] = "flash_attention_2"
            print("✅ Qwen will use flash_attention_2.")
        except Exception:
            print("⚠️ flash-attn not found. Qwen will use the slower fallback attention path.")
    return kwargs


def _current_qwen_model_names():
    size = getattr(get_settings(), "qwen_model_size", None)
    return qwen_model_names_for_size(size)


def get_qwen_custom_model():
    global _CUSTOM_MODEL, _CUSTOM_MODEL_ID
    model_names = _current_qwen_model_names()
    model_id = model_names["custom"]
    if _CUSTOM_MODEL is None or _CUSTOM_MODEL_ID != model_id:
        model_cls = _import_qwen()
        print(f"⏳ Loading Qwen CustomVoice model: {model_id}")
        print("ℹ️ If this is the first time this model is used here, it may need to download files and can take a while.")
        _CUSTOM_MODEL = model_cls.from_pretrained(model_id, **_torch_runtime_kwargs())
        _CUSTOM_MODEL_ID = model_id
        print("✅ Qwen CustomVoice model loaded.")
    return _CUSTOM_MODEL


def get_qwen_clone_model():
    global _CLONE_MODEL, _CLONE_MODEL_ID
    model_names = _current_qwen_model_names()
    model_id = model_names["clone"]
    if _CLONE_MODEL is None or _CLONE_MODEL_ID != model_id:
        model_cls = _import_qwen()
        print(f"⏳ Loading Qwen Base voice-clone model: {model_id}")
        print("ℹ️ If this is the first time this model is used here, it may need to download files and can take a while.")
        _CLONE_MODEL = model_cls.from_pretrained(model_id, **_torch_runtime_kwargs())
        _CLONE_MODEL_ID = model_id
        print("✅ Qwen Base voice-clone model loaded.")
    return _CLONE_MODEL


def get_qwen_supported_speakers() -> list[str]:
    model = get_qwen_custom_model()
    speakers = model.get_supported_speakers()
    return list(speakers) if speakers else []


def get_qwen_supported_languages() -> list[str]:
    model = get_qwen_custom_model()
    langs = model.get_supported_languages()
    return list(langs) if langs else []


class _StopOnEventCriteria:
    def __init__(self, stop_event: Event):
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs):
        return bool(self.stop_event and self.stop_event.is_set())


def _generation_stop_kwargs(stop_event: Event | None) -> dict:
    if stop_event is None:
        return {}
    try:
        from transformers import StoppingCriteriaList
        return {"stopping_criteria": StoppingCriteriaList([_StopOnEventCriteria(stop_event)])}
    except Exception:
        return {}


def _call_with_safe_retries(func, kwargs: dict, *, allow_instruct_fallback: bool = False):
    attempts = [dict(kwargs)]
    if "stopping_criteria" in kwargs:
        reduced = dict(kwargs)
        reduced.pop("stopping_criteria", None)
        attempts.append(reduced)
        if allow_instruct_fallback and "instruct" in reduced:
            reduced2 = dict(reduced)
            reduced2.pop("instruct", None)
            attempts.append(reduced2)
    elif allow_instruct_fallback and "instruct" in kwargs:
        reduced = dict(kwargs)
        reduced.pop("instruct", None)
        attempts.append(reduced)

    last_exc = None
    seen = set()
    for attempt in attempts:
        key = tuple(sorted(attempt.keys()))
        if key in seen:
            continue
        seen.add(key)
        try:
            return func(**attempt)
        except TypeError as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    return func(**kwargs)


def synthesize_qwen_custom(texts: Sequence[str], *, language: str, speaker: str, instruct: str = "", max_new_tokens: int = DEFAULT_QWEN_MAX_NEW_TOKENS, stop_event: Event | None = None):
    model = get_qwen_custom_model()
    kwargs = {
        "text": list(texts),
        "language": [language] * len(texts),
        "speaker": [speaker] * len(texts),
        "max_new_tokens": int(max_new_tokens),
        **_generation_stop_kwargs(stop_event),
    }
    if instruct:
        kwargs["instruct"] = [instruct] * len(texts)
    wavs, sr = _call_with_safe_retries(model.generate_custom_voice, kwargs, allow_instruct_fallback=False)
    return [np.asarray(w, dtype=np.float32) for w in wavs], int(sr)


def create_qwen_clone_prompt(ref_audio, ref_text: str, *, x_vector_only_mode: bool = False):
    model = get_qwen_clone_model()
    return model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=bool(x_vector_only_mode),
    )


def synthesize_qwen_clone(texts: Sequence[str], *, language: str, voice_clone_prompt, instruct: str = "", max_new_tokens: int = DEFAULT_QWEN_MAX_NEW_TOKENS, stop_event: Event | None = None):
    model = get_qwen_clone_model()
    kwargs = {
        "text": list(texts),
        "language": [language] * len(texts),
        "voice_clone_prompt": voice_clone_prompt,
        "max_new_tokens": int(max_new_tokens),
        **_generation_stop_kwargs(stop_event),
    }
    if instruct:
        kwargs["instruct"] = [instruct] * len(texts)
    wavs, sr = _call_with_safe_retries(model.generate_voice_clone, kwargs, allow_instruct_fallback=True)
    return [np.asarray(w, dtype=np.float32) for w in wavs], int(sr)
