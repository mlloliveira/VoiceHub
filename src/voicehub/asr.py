# src/voicehub/asr.py
import os
import tempfile
import numpy as np
import soundfile as sf

from .user_settings import get_settings
from .progress_utils import console_progress as _console_progress

#Ollama translation:
from .ollama_utils import has_ollama, translate_text_with_ollama
from .ollama_config import OLLAMA_MODEL_DEFAULT

# ===================== ASR (Whisper via faster-whisper / optional openai-whisper) =====================
FW_AVAILABLE = True
WhisperModel = None
try:
    from faster_whisper import WhisperModel
except Exception:
    FW_AVAILABLE = False

OWHISPER_AVAILABLE = True
owhisp = None
try:
    import whisper as owhisp
except Exception:
    OWHISPER_AVAILABLE = False

from .config import (
    ASR_MODEL_NAME, ASR_COMPUTE,
    ASR_LANG_MAP, BACKEND_MAP,
)

_fw_model = None 
_ow_model = None


def _torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _fw_runtime_args():
    if _torch_cuda_available():
        return {"device": "cuda", "compute_type": ASR_COMPUTE}
    return {"device": "cpu", "compute_type": os.getenv("ASR_CPU_COMPUTE", "float32")}


def _ow_device() -> str:
    return "cuda" if _torch_cuda_available() else "cpu"

#Why two backends?
#faster-whisper is much faster and lighter for deployment (CTranslate2). openai-whisper (the original PyTorch impl) is useful for parity checks and debugging.

def get_fw_model(): #faster-whisper (CTranslate2) on CUDA with your chosen compute type.
    global _fw_model
    if _fw_model is None:
        if not FW_AVAILABLE or WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed in this env.")
        runtime = _fw_runtime_args()
        print(f"⏳ Loading faster-whisper model: {ASR_MODEL_NAME} on {runtime.get('device', 'cpu')}")
        print("ℹ️ If this is the first time this model is used here, it may need to download files and can take a while.")
        _fw_model = WhisperModel(ASR_MODEL_NAME, **runtime)
        print("✅ faster-whisper model loaded.")
    return _fw_model

def get_ow_model(): #openai-whisper (PyTorch) as a secondary backend, in case you want to compare or debug.
    global _ow_model
    if _ow_model is None:
        if not OWHISPER_AVAILABLE:
            raise RuntimeError("openai-whisper is not installed in this env.")
        device = _ow_device()
        print(f"⏳ Loading openai-whisper model: {ASR_MODEL_NAME} on {device}")
        print("ℹ️ If this is the first time this model is used here, it may need to download files and can take a while.")
        _ow_model = owhisp.load_model(ASR_MODEL_NAME, device=device)
        print("✅ openai-whisper model loaded.")
    return _ow_model

def _write_temp_wav(sr, y) -> str: #Utility: Normalizes to mono float32, writes an on-disk WAV, and returns its path
    if y.ndim > 1:
        y = y.mean(axis=1) #Normalize to mono
    y = y.astype(np.float32) #Normalize to float32
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False) #creates temp WAV file
    sf.write(f.name, y, sr) #writes temp WAV file
    f.close()
    return f.name #returns temp WAV file path

# Accepts Gradio audio in any of its 4 shapes and returns a temp WAV path
def _normalize_audio_to_wav(audio) -> str:
    """
    Gradio 4.x Audio component may return:
      - str (filepath) when type="filepath"
      - (sr, np.ndarray) tuple or list when type="numpy"
      - dict with keys like {"sample_rate": int, "data": np.ndarray, ...}
    This function converts all of them into a WAV file on disk and returns its path.
    """
    # 1) Already a filepath?
    if isinstance(audio, str):
        if os.path.exists(audio):
            return audio
        raise ValueError(f"Path does not exist: {audio}")

    # 2) Tuple/list style: (sr, array)
    if isinstance(audio, (tuple, list)) and len(audio) >= 2:
        sr, y = audio[0], audio[1]
        return _write_temp_wav(int(sr), np.asarray(y))

    # 3) Dict style: {"sample_rate": ..., "data": ...}
    if isinstance(audio, dict):
        # Gradio commonly uses 'sample_rate' + 'data'
        sr = None
        for key in ("sample_rate", "sr", "sampling_rate", "sampleRate"):
            if key in audio and audio.get(key) is not None:
                sr = audio.get(key)
                break
        y = None
        for key in ("data", "array", "samples"):
            if key in audio and audio.get(key) is not None:
                y = audio.get(key)
                break
        if sr is None or y is None:
            raise ValueError("Missing 'sample_rate' or 'data' in audio dict.")
        return _write_temp_wav(int(sr), np.asarray(y))

    # 4) Unknown
    raise ValueError(f"Unsupported audio input type: {type(audio)}")

def transcribe_reference_audio(path: str, language_code: str | None = None):
    """Structured helper for internal reuse (e.g., Qwen clone transcript generation)."""
    s = get_settings()
    model = get_fw_model()
    segments, info = model.transcribe(
        path,
        language=language_code,
        beam_size=max(1, int(getattr(s, "whisper_beam_size", 5))),
        vad_filter=True,
        condition_on_previous_text=bool(s.whisper_condition_on_prev),
        temperature=float(s.whisper_temperature),
    )
    text = "".join([(seg.text or "") for seg in segments]).strip()
    return {
        "text": text,
        "language": getattr(info, "language", language_code or ""),
        "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
    }

def _transcribe_path(path: str, lang_code, beam_size, use_vad, backend_display):
    s = get_settings()  # current global defaults
    backend = BACKEND_MAP.get(backend_display, "fw")
    if backend == "ow":
        model = get_ow_model()
        result = model.transcribe(
            path,
            language=lang_code,
            fp16=(_ow_device() == "cuda"),
            beam_size=int(beam_size) if beam_size else None,
            temperature=float(s.whisper_temperature),   # <-- use global default
        )
        text = (result.get("text") or "").strip()
        meta = f"Detected: {result.get('language', 'n/a')}" if lang_code is None else ""
        return text, meta

    model = get_fw_model()
    segments, info = model.transcribe(
        path,
        language=lang_code,
        beam_size=int(beam_size),
        vad_filter=bool(use_vad),
        condition_on_previous_text=bool(s.whisper_condition_on_prev),           # <-- from config tab
        temperature=float(s.whisper_temperature),                               # <-- from config tab
    )
    text = "".join([seg.text for seg in segments]).strip()
    meta = f"Detected: {info.language} (p={info.language_probability:.2f})" if lang_code is None else ""
    return text, meta

#Ollama Translation:
def translate_asr_text(source_text: str, target_lang_display: str, model_name: str):
    """
    Backend translation helper for ASR transcripts (Ollama-backed).
    - source_text: current transcript (plain text)
    - target_lang_display: human-readable target (e.g., "English", "Português")
    - model_name: ollama tag (e.g., "llama3.1", "qwen2.5:7b"); can be empty -> default

    Returns:
        (translated_text, status_markdown)
    """
    src = (source_text or "").strip()
    if not src:
        return "", "⚠️ No transcript to translate."

    ok, msg = has_ollama()
    if not ok:
        return "", f"❌ {msg}"

    s = get_settings()  # per-model global decode knobs for Ollama
    stop_list = [t.strip() for t in (s.ollama_stop or "").split(",") if t.strip()]
    options = {
        "temperature": float(s.ollama_temperature),
        "top_p": float(s.ollama_top_p),
        "num_predict": int(s.ollama_num_predict),
    }
    if stop_list:
        options["stop"] = stop_list

    try:
        model = (model_name or OLLAMA_MODEL_DEFAULT).strip()
        out = translate_text_with_ollama(
            raw_text=src,
            target_lang_name=target_lang_display,  # keep human name; LLMs handle names better than ISO codes
            model=model,
            options=options,
        )
        if not out:
            return "", "⚠️ Translation returned empty output."
        return out, f"✅ Translated to **{target_lang_display}** with **{model}**."
    except Exception as e:
        return "", f"⚠️ Translation failed: {e}"
    
### Create conditions and functions to force a stop:
_ASR_STOP_REQUESTED = False

def reset_asr_stop_flag():
    """Clear STOP flag before a new ASR run."""
    global _ASR_STOP_REQUESTED
    _ASR_STOP_REQUESTED = False

def request_asr_stop(backend_display: str) -> str:
    """
    UI hook: set STOP for ASR.
    If backend is OpenAI Whisper (ow), we can't stop; return a warning.
    """
    global _ASR_STOP_REQUESTED
    _ASR_STOP_REQUESTED = True
    from .config import BACKEND_MAP
    backend = BACKEND_MAP.get(backend_display, "fw")
    if backend == "ow":
        return "⛔️ Stop not supported for OpenAI Whisper backend; letting it finish."
    return "⛔️ Stop requested — finishing current segment and returning partial transcript."

###########################################

#Transcribe (Main Func)
def transcribe(audio, language_display, beam_size, use_vad, backend_display): #Handles both backends: OpenAI Whisper and faster-whisper
    if audio is None:
        return "", ""
    # sr, y = audio #Sample Rate , Waveform
    # wav_path = _write_temp_wav(sr, y)
    # Accept str filepath (preferred) or numpy/dict and normalize to WAV on disk
    try:
        wav_path = _normalize_audio_to_wav(audio)
    except Exception as e:
        return f"⚠️ Could not read audio input: {e}", ""

    lang = ASR_LANG_MAP.get(language_display, None) #Auto-detect maps to None
    backend = BACKEND_MAP.get(backend_display, "fw") #Faster-Whisper is default

    s = get_settings()  # per-model defaults
    try:
        if backend == "ow": # OpenAI Whisper (PyTorch)
            model = get_ow_model() #Get model
            result = model.transcribe( #Inference 
            wav_path,
            language=lang,
            fp16=(_ow_device() == "cuda"),
            beam_size=int(beam_size) if beam_size else None,
            temperature=float(s.whisper_temperature),
            #top_p=float(s.whisper_top_p), #Some Whisper builds supports it
        )
            text = (result.get("text") or "").strip() #Text result (STT)
            meta = f"Detected: {result.get('language', 'n/a')}" if lang is None else "" #Detected language if not selected
            return text, meta

        # Faster-Whisper (FW)
        reset_asr_stop_flag() # Reset stop flag
        model = get_fw_model() # Get model
        segments, info = model.transcribe( #Inference 
            wav_path,
            language=lang,
            beam_size=int(beam_size),
            vad_filter=bool(use_vad),
            condition_on_previous_text=bool(s.whisper_condition_on_prev),
            temperature=float(s.whisper_temperature),
            #top_p=float(s.whisper_top_p),  #Some Whisper builds supports it
        )
        # text = "".join([seg.text for seg in segments]).strip() #Text result (STT)
        # meta = f"Detected: {info.language} (p={info.language_probability:.2f})" if lang is None else "" #Detected language if not selected

        ### More Steps to do previous logic, but with more flexibity to check status and to stop:
        total_s = getattr(info, "duration", 0.0) or 0.0
        done_s = 0.0
        acc = []
        user_stopped = False

        _console_progress(0, int(total_s), prefix="ASR")
        for seg in segments:
            # append text as we go
            acc.append(seg.text or "")
            # update progress by audio time processed
            try:
                done_s = max(done_s, float(seg.end))
            except Exception:
                pass
            _console_progress(int(done_s), int(total_s), prefix="ASR")

            # STOP?
            if _ASR_STOP_REQUESTED:
                user_stopped = True
                break

        _console_progress(int(total_s), int(total_s), prefix="ASR", end=True)
        text = "".join(acc).strip()

        # meta line
        meta = ""
        if lang is None:
            try:
                meta = f"Detected: {info.language} (p={info.language_probability:.2f})"
            except Exception:
                meta = ""
        if user_stopped:
            meta = ("⛔️ Stopped by user. " + (meta or "")).strip()
        ###


        return text, meta

    except Exception as e:
        return f"⚠️ ASR error: {e}", ""

###########################################

# --- Simple streaming buffer (no live ASR; just bypass browser cap) ---
def _unpack_chunk(audio):
    """
    Accept a streaming chunk in any Gradio shape and return (sr, mono float32 ndarray in [-1, 1]).
    Handles int8/int16/int32 PCM by scaling to [-1, 1]; keeps float as-is.
    """
    # tuple/list: (sr, data)
    if isinstance(audio, (tuple, list)) and len(audio) >= 2:
        sr, y = int(audio[0]), np.asarray(audio[1])
    elif isinstance(audio, dict):
        sr = None
        for key in ("sample_rate", "sr", "sampling_rate", "sampleRate"):
            if key in audio and audio.get(key) is not None:
                sr = int(audio.get(key))
                break
        y = None
        for key in ("data", "array", "samples"):
            if key in audio and audio.get(key) is not None:
                y = np.asarray(audio.get(key))
                break
        if sr is None or y is None:
            raise ValueError("Unsupported streaming chunk dict: missing sample rate or audio data.")
    else:
        raise ValueError(f"Unsupported streaming chunk type: {type(audio)}")

    # mono
    if y.ndim > 1:
        y = y.mean(axis=1)

    # scale if integer PCM
    if np.issubdtype(y.dtype, np.integer):
        if y.dtype == np.int16:
            y = y.astype(np.float32) / 32768.0
        elif y.dtype == np.int32:
            y = y.astype(np.float32) / 2147483648.0
        elif y.dtype == np.int8:
            # int8 is centered at 0 already: [-128,127]
            y = y.astype(np.float32) / 128.0
        elif y.dtype == np.uint8:
            # unsigned 8-bit: [0,255] -> center at 128
            y = (y.astype(np.float32) - 128.0) / 128.0
        else:
            # generic integer fallback (rare)
            info = np.iinfo(y.dtype)
            denom = float(max(abs(info.min), info.max))
            y = y.astype(np.float32) / (denom if denom > 0 else 1.0)
    else:
        # already float -> just cast (assumed in [-1,1] from browser)
        y = y.astype(np.float32)

    # safety clip
    y = np.clip(y, -1.0, 1.0)
    return sr, y

def stream_reset():
    """Start a fresh capture buffer."""
    return {"sr": None, "buf": [], "nsamp": 0, "cap": False}

def stream_append(audio_chunk, state: dict):
    """
    Append a streaming mic chunk into the buffer.
    Enforce a hard time cap in minutes (truncate last chunk if needed).
    Returns (updated_state, status_text).
    """
    if state is None or not isinstance(state, dict):
        state = stream_reset()
    sr, y = _unpack_chunk(audio_chunk)
    if state["sr"] is None:
        state["sr"] = sr
    elif state["sr"] != sr:
        # In practice the mic sr remains constant; if not, we just coerce by simple cast.
        # (Resampling is overkill for this use case.)
        sr = state["sr"]
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)

    s = get_settings()
    cap_s = max(0.5, float(s.asr_stream_max_minutes)) * 60.0
    have_s = state["nsamp"] / sr
    remain_s = cap_s - have_s

    if remain_s <= 0:
        state["cap"] = True
        return state, f"⏱️ Cap reached ({cap_s/60:.1f} min). Press Stop."

    # trim this chunk to fit remaining time
    keep = int(remain_s * sr)
    if keep < len(y):
        y = y[:keep]
        state["cap"] = True

    state["buf"].append(y)
    state["nsamp"] += len(y)
    have_s = state["nsamp"] / sr
    note = " (CAP REACHED)" if state["cap"] else ""
    return state, f"Recording {have_s:.1f}s / {cap_s/60:.1f}m{note}"

def stream_finalize_and_transcribe(state: dict, language_display, beam_size, use_vad, backend_display):
    """
    Concatenate buffered chunks -> write temp WAV -> run normal transcription path.
    Returns (wav_path, transcript, meta) so the UI can preview the captured audio.
    """
    if not state or state.get("nsamp", 0) == 0 or not state.get("buf"):
        return None, "", "⚠️ No audio captured."

    sr = int(state["sr"])
    y = np.concatenate(state["buf"]).astype(np.float32)

    y = np.clip(y, -1.0, 1.0) # extra safety (browser / driver oddities)

    # Save one WAV for both preview and transcription
    wav_path = _write_temp_wav(sr, y)
    #(It is possible to add some cleaning technique here)
    try:
        # Preferred if you have it:
        text, meta = _transcribe_path(
            wav_path,
            lang_code=ASR_LANG_MAP.get(language_display, None),
            beam_size=beam_size,
            use_vad=use_vad,
            backend_display=backend_display,
        )
    except NameError:
        # Fallback: transcribe() that accepts a filepath input
        text, meta = transcribe(
            wav_path, language_display, beam_size, use_vad, backend_display
        )

    return wav_path, text, meta
