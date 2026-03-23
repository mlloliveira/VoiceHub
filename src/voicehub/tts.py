from __future__ import annotations

import os
import tempfile
from threading import Event
from typing import Optional, Sequence

import gradio as gr
import numpy as np
import soundfile as sf

from .audio_utils import concat_audio_segments
from .asr import transcribe_reference_audio
from .chunking import choose_safe_refined_text, chunk_text
from .config import (
    ASR_MODEL_NAME,
    DEBUG_TOOLS,
    DEFAULT_MAX_CHARS_PER_CHUNK,
    DEFAULT_QWEN_MAX_NEW_TOKENS,
    DEFAULT_QWEN_CHUNK_SIZE,
    DEFAULT_QWEN_CLONE_REF_SECONDS,
    DEFAULT_XTTS_CLONE_REF_SECONDS,
    TTS_LANG_MAP,
    TTS_MODEL_NAME,
    VOICEHUB_VERSION,
    effective_max_chars,
    qwen_model_names_for_size,
)
from .lang_detect import detect_tts_language
from .ollama_config import OLLAMA_ENABLE_DEFAULT, OLLAMA_MODEL_DEFAULT
from .ollama_utils import has_ollama, refine_text_with_ollama
from .progress_utils import console_progress as _console_progress
from .qwen_backend import create_qwen_clone_prompt, qwen_available, synthesize_qwen_clone, synthesize_qwen_custom
from .tts_router import (
    all_backend_voice_choices,
    format_backend_status,
    label_voice,
    qwen_voice_choices,
    resolve_qwen_voice,
    resolve_tts_route,
    resolve_xtts_voice,
    speed_bucket_prompt,
    xtts_voice_choices,
)
from .user_settings import get_settings
from .voice_clone_cache import load_clone_cache, read_cached_transcript, save_clone_cache

_TTS_STOP_REQUESTED = False
_TTS_STOP_EVENT = Event()
_XTTS_MODEL = None
_TORCHAUDIO_LOAD_PATCHED = False
SPEAKER_DISPLAY_TO_META: dict[str, dict] = {}


def _ensure_xtts_audio_loader_compat():
    global _TORCHAUDIO_LOAD_PATCHED
    if _TORCHAUDIO_LOAD_PATCHED:
        return
    try:
        import torchcodec  # noqa: F401
        return
    except Exception:
        pass

    try:
        import sys
        torch = sys.modules.get("torch")
        torchaudio = sys.modules.get("torchaudio")
        if torch is None:
            import torch as _torch
            torch = _torch
        if torchaudio is None:
            import torchaudio as _torchaudio
            torchaudio = _torchaudio
    except Exception as e:
        print(f"⚠️ torchcodec is not installed, and torchaudio could not be patched for XTTS cloning: {e}")
        return

    def _voicehub_sf_load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None, buffer_size=4096, backend=None):
        data, sr = sf.read(uri, always_2d=True, dtype="float32")
        frame_offset = max(0, int(frame_offset or 0))
        num_frames = int(num_frames) if num_frames is not None else -1
        if frame_offset:
            data = data[frame_offset:]
        if num_frames >= 0:
            data = data[:num_frames]
        arr = np.asarray(data.T if channels_first else data, dtype=np.float32, order="C")
        return torch.from_numpy(arr), int(sr)

    torchaudio.load = _voicehub_sf_load
    _TORCHAUDIO_LOAD_PATCHED = True
    print("⚠️ torchcodec is not installed. Using a soundfile-based torchaudio.load fallback for XTTS voice cloning.")


def _import_xtts_api():
    from TTS.api import TTS
    return TTS


def get_xtts_model():
    global _XTTS_MODEL
    if _XTTS_MODEL is None:
        _ensure_xtts_audio_loader_compat()
        TTS = _import_xtts_api()
        print(f"⏳ Loading XTTS model: {TTS_MODEL_NAME}")
        print("ℹ️ If this is the first time this model is used here, it may need to download files and can take a while.")
        model = TTS(TTS_MODEL_NAME)
        try:
            model.to("cuda")
            print("✅ XTTS model loaded on GPU.")
        except Exception as e:
            print(f"⚠️ Could not move XTTS-v2 to GPU, falling back to CPU. Reason: {e}")
            model.to("cpu")
            print("✅ XTTS model loaded on CPU.")
        _XTTS_MODEL = model
    return _XTTS_MODEL


def get_available_xtts_speakers():
    names = []
    try:
        tts = get_xtts_model()
        sm = getattr(tts.synthesizer.tts_model, "speaker_manager", None)
        sp = getattr(sm, "speakers", None) if sm is not None else None
        if isinstance(sp, dict):
            names = list(sp.keys())
        elif isinstance(sp, (list, tuple)):
            names = list(sp)
    except Exception:
        pass
    return sorted({n.strip() for n in names if isinstance(n, str) and n.strip()})


def _build_ollama_options_from_settings():
    s = get_settings()
    stop_list = [t.strip() for t in (s.ollama_stop or "").split(",") if t.strip()]
    opts = {
        "temperature": float(s.ollama_temperature),
        "top_p": float(s.ollama_top_p),
        "num_predict": int(s.ollama_num_predict),
    }
    if stop_list:
        opts["stop"] = stop_list
    return opts


def preprocess_to_chunks(
    text: str,
    use_ollama: bool,
    ollama_model: str,
    max_chars: int,
    lang_code: str = "xx",
    tts_family: str = "xtts",
):
    text = (text or "").strip()
    refined = text
    if not text:
        return "", [], []

    if use_ollama:
        model = (ollama_model or OLLAMA_MODEL_DEFAULT).strip()
        ok, _ = has_ollama()
        if ok and model:
            try:
                print(f"⏳ Running Ollama pre-chunker for {tts_family.upper()}...")
                refined_out = refine_text_with_ollama(
                    text,
                    max_chars=int(max_chars),
                    model=model,
                    options=_build_ollama_options_from_settings(),
                    prompt_mode="qwen" if tts_family == "qwen" else "xtts",
                )
                if refined_out:
                    refined, accepted = choose_safe_refined_text(text, refined_out)
                    if not accepted:
                        print("[TTS] Ollama refine changed content/order; falling back to original text.")
                    else:
                        print(f"✅ Ollama pre-chunker finished for {tts_family.upper()}.")
            except Exception as e:
                print(f"[TTS] Ollama refine failed: {e}. Keeping original text.")

    sents, chunks, backend = chunk_text(refined, max_len=int(max_chars), lang_code=lang_code or "xx")
    if DEBUG_TOOLS:
        print(f"[TTS] family={tts_family} chunk_backend={backend} chunks={len(chunks)}")
    return refined, sents, chunks


def reset_tts_stop_flag():
    global _TTS_STOP_REQUESTED
    _TTS_STOP_REQUESTED = False
    _TTS_STOP_EVENT.clear()


def request_tts_stop():
    global _TTS_STOP_REQUESTED
    _TTS_STOP_REQUESTED = True
    _TTS_STOP_EVENT.set()
    print("⛔️ Stop requested for TTS.")
    return "⛔️ Stop requested — trying to stop generation immediately."


def _resolve_language(text: str, language_display: str):
    det_note = ""
    if language_display == "Auto-detect":
        det_label, det_code, det_score, det_note = detect_tts_language(text)
        return det_code or "en", det_note
    return TTS_LANG_MAP.get(language_display, "en") or "en", det_note


def _current_family() -> str:
    return (getattr(get_settings(), "tts_default_family", "XTTS") or "XTTS").strip()


def make_speaker_choices(language_display: str, family_display: str | None = None):
    SPEAKER_DISPLAY_TO_META.clear()
    xtts = get_available_xtts_speakers()
    choices = all_backend_voice_choices(xtts)
    SPEAKER_DISPLAY_TO_META["Default"] = {"family": "default", "name": "Default"}
    for name in qwen_voice_choices()[1:]:
        lbl = label_voice(name, "qwen")
        SPEAKER_DISPLAY_TO_META[lbl] = {"family": "qwen", "name": name}
    xtts_choices, _ = xtts_voice_choices(xtts)
    for name in xtts_choices[1:]:
        lbl = label_voice(name, "xtts")
        SPEAKER_DISPLAY_TO_META[lbl] = {"family": "xtts", "name": name}
    return choices, "Default"


def refresh_speakers(language_display: str, family_display: str | None = None):
    choices, default = make_speaker_choices(language_display, family_display)
    return gr.update(choices=choices, value=default, interactive=True)


def _strip_voice_label(display_value: str | None) -> str:
    if not display_value or display_value == "Default":
        return "Default"
    if display_value.endswith(" (Qwen)"):
        return display_value[:-7]
    if display_value.endswith(" (XTTS)"):
        return display_value[:-7]
    return display_value


def _tts_chunk_xtts(text_chunk, lang, speaker_name=None, ref_wav=None, speed=1.0):
    tts = get_xtts_model()
    if ref_wav:
        wav = tts.tts(text=text_chunk, language=lang, speaker_wav=[ref_wav], speed=float(speed), split_sentences=False)
    else:
        wav = tts.tts(text=text_chunk, language=lang, speaker=speaker_name, speed=float(speed), split_sentences=False)
    if isinstance(wav, tuple) and len(wav) == 2:
        wav, sr = wav
    else:
        sr = 24000
    return np.asarray(wav, dtype=np.float32), int(sr)


def _maybe_trim_and_collect(wavs_out, srs_out, new_wavs: Sequence[np.ndarray], sr: int, max_seconds: float, total_s: float):
    truncated = False
    truncated_mid_chunk = False
    processed = 0
    for w in new_wavs:
        if _TTS_STOP_REQUESTED:
            truncated = True
            break
        dur = len(w) / sr if sr else 0.0
        remain = max_seconds - total_s
        eps = max(1.0 / sr, 0.001) if sr else 0.001
        if remain <= 0:
            truncated = True
            break
        if dur <= remain + eps:
            wavs_out.append(np.asarray(w, dtype=np.float32))
            srs_out.append(int(sr))
            total_s += dur
            processed += 1
        else:
            n_samples = int(remain * sr + 0.5)
            if n_samples > 0:
                wavs_out.append(np.asarray(w[:n_samples], dtype=np.float32))
                srs_out.append(int(sr))
                total_s += n_samples / sr
                processed += 1
                truncated = True
                truncated_mid_chunk = True
            else:
                truncated = True
            break
    return total_s, truncated, truncated_mid_chunk, processed




def _clone_ref_cap_seconds_for_family(family: str) -> float:
    s = get_settings()
    fam = (family or "XTTS").strip().lower()
    if fam == "qwen":
        return max(1.0, float(getattr(s, "qwen_clone_ref_max_seconds", DEFAULT_QWEN_CLONE_REF_SECONDS) or DEFAULT_QWEN_CLONE_REF_SECONDS))
    return max(1.0, float(getattr(s, "xtts_clone_ref_max_seconds", DEFAULT_XTTS_CLONE_REF_SECONDS) or DEFAULT_XTTS_CLONE_REF_SECONDS))


def _prepare_capped_reference_audio(ref_wav: str | None, family: str):
    if not ref_wav:
        return ref_wav, None, False, 0.0, 0.0
    cap_seconds = _clone_ref_cap_seconds_for_family(family)
    try:
        wav, sr = sf.read(ref_wav, always_2d=False)
        total_seconds = (len(wav) / float(sr)) if sr else 0.0
        if not sr or total_seconds <= cap_seconds + 1e-6:
            return ref_wav, None, False, total_seconds, cap_seconds
        n_samples = max(1, int(cap_seconds * sr))
        trimmed = wav[:n_samples]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, trimmed, sr)
        print(f"✂️ Trimmed reference audio for {family.upper()} cloning from {total_seconds:.1f}s to {cap_seconds:.1f}s.")
        return tmp.name, tmp.name, True, total_seconds, cap_seconds
    except Exception as e:
        print(f"⚠️ Could not preprocess reference audio cap for {family.upper()}: {e}. Using original file.")
        return ref_wav, None, False, 0.0, cap_seconds


def _cleanup_temp_paths(paths):
    for p in paths:
        if p:
            try:
                os.unlink(p)
            except Exception:
                pass

def _build_qwen_clone_prompt_cached(ref_wav: str):
    digest, record = load_clone_cache(ref_wav)
    transcript = ""
    detected_language = ""
    if record is not None:
        transcript = read_cached_transcript(record)
        detected_language = record.detected_language
    if not transcript:
        print("⏳ Generating transcript for Qwen voice cloning...")
        result = transcribe_reference_audio(ref_wav, language_code=None)
        transcript = (result.get("text") or "").strip()
        detected_language = (result.get("language") or "").strip()
        if transcript:
            save_clone_cache(
                digest=digest,
                source_audio_path_original=ref_wav,
                asr_model="faster-whisper:" + ASR_MODEL_NAME,
                detected_language=detected_language,
                transcript=transcript,
                qwen_clone_model=qwen_model_names_for_size(getattr(get_settings(), "qwen_model_size", None))["clone"],
                voicehub_version=VOICEHUB_VERSION,
            )
            print("✅ Saved transcript cache for Qwen voice cloning.")
    else:
        print("✅ Reusing cached transcript for Qwen voice cloning.")
    if not transcript:
        print("⚠️ No transcript available. Falling back to x-vector-only Qwen cloning.")
        prompt = create_qwen_clone_prompt(ref_audio=ref_wav, ref_text="", x_vector_only_mode=True)
        return prompt, True
    prompt = create_qwen_clone_prompt(ref_audio=ref_wav, ref_text=transcript, x_vector_only_mode=False)
    return prompt, False


def _compose_qwen_instruction(style_prompt: str, speed: float, language_code: str) -> str:
    parts = []
    sp = speed_bucket_prompt(float(speed), language_code)
    if sp:
        parts.append(sp)
    if style_prompt and style_prompt.strip():
        parts.append(style_prompt.strip())
    return " ".join(parts).strip()


def _segment_limit_for_route(route, lang: str, settings) -> int:
    if route.resolved_family == "qwen":
        return int(getattr(settings, "qwen_max_chars_per_chunk", DEFAULT_QWEN_CHUNK_SIZE) or DEFAULT_QWEN_CHUNK_SIZE)
    return effective_max_chars(
        lang_code=lang or "en",
        user_cap=int(getattr(settings, "xtts_max_chars_per_chunk", DEFAULT_MAX_CHARS_PER_CHUNK)),
        dynamic=bool(getattr(settings, "xtts_dynamic_per_lang_caps", True)),
    )


def _build_chunks_for_route(text: str, route, lang: str, use_ollama: bool, ollama_model: str):
    seg_chars = _segment_limit_for_route(route, lang, get_settings())
    return preprocess_to_chunks(
        text=text,
        use_ollama=bool(use_ollama),
        ollama_model=ollama_model,
        max_chars=seg_chars,
        lang_code=lang or "en",
        tts_family=route.resolved_family,
    )


def _run_xtts_chunks(chunks, lang, speaker_display, ref_wav, speed, max_seconds, total_s, wavs_out, srs_out, processed, truncated, truncated_mid_chunk):
    speaker_name = resolve_xtts_voice(_strip_voice_label(speaker_display), get_available_xtts_speakers())
    if not speaker_name and not ref_wav:
        return None, None, None, None, None, None, "⚠️ No XTTS voices were discovered."
    print(f"🎙️ Using XTTS backend. Voice: {'reference clone' if ref_wav else speaker_name}.")
    for i, ch in enumerate(chunks, start=1):
        if _TTS_STOP_REQUESTED:
            truncated = True
            break
        print(f"🔊 XTTS chunk {i}/{len(chunks)}")
        _console_progress(i, len(chunks), prefix="TTS")
        speak_text = ch.strip()
        if not speak_text:
            continue
        w, sr = _tts_chunk_xtts(speak_text, lang, speaker_name=speaker_name, ref_wav=ref_wav, speed=speed)
        total_s, tr, tr_mid, done = _maybe_trim_and_collect(wavs_out, srs_out, [w], sr, max_seconds, total_s)
        truncated = truncated or tr
        truncated_mid_chunk = truncated_mid_chunk or tr_mid
        processed += done
        if tr or _TTS_STOP_REQUESTED:
            break
    return total_s, processed, truncated, truncated_mid_chunk, wavs_out, srs_out, None


def synthesize_tts(
    text,
    language_display,
    speaker_display,
    speed,
    ref_wav,
    qwen_style_prompt="",
    use_ollama=False,
    ollama_model="",
):
    reset_tts_stop_flag()
    text = (text or "").strip()
    if not text:
        return None, ""

    temp_cleanup_paths = []
    original_ref_wav = ref_wav

    lang, det_note = _resolve_language(text, language_display)
    try:
        s = get_settings()
        initial_route = resolve_tts_route(getattr(s, "tts_default_family", "XTTS"), lang, has_reference=bool(ref_wav))
        route = initial_route

        used_limit_min = float(s.xtts_max_minutes_default)
        max_seconds = max(0.5, used_limit_min) * 60.0

        if route.resolved_family == "qwen":
            ok, msg = qwen_available()
            if not ok:
                print(f"⚠️ {msg}. Falling back to XTTS.")
                route = resolve_tts_route("XTTS", lang, has_reference=bool(ref_wav))

        ref_wav, cleanup_path, _, _, _ = _prepare_capped_reference_audio(original_ref_wav, route.resolved_family if route else "XTTS")
        if cleanup_path:
            temp_cleanup_paths.append(cleanup_path)

        _, _, chunks = _build_chunks_for_route(
            text=text,
            route=route,
            lang=lang,
            use_ollama=bool(use_ollama),
            ollama_model=ollama_model,
        )
        if not chunks:
            return None, "⚠️ No content after preprocessing."

        wavs_out: list[np.ndarray] = []
        srs_out: list[int] = []
        total_s = 0.0
        processed = 0
        truncated = False
        truncated_mid_chunk = False
        user_stopped = False

        _console_progress(0, len(chunks), prefix="TTS")

        if route.backend == "xtts":
            backend_note = format_backend_status(lang, route, detected=(language_display == "Auto-detect"))
            total_s, processed, truncated, truncated_mid_chunk, wavs_out, srs_out, err = _run_xtts_chunks(
                chunks, lang, speaker_display, ref_wav, speed, max_seconds, total_s, wavs_out, srs_out, processed, truncated, truncated_mid_chunk
            )
            if err:
                return None, err
            if _TTS_STOP_REQUESTED:
                user_stopped = True
        else:
            backend_note = format_backend_status(lang, route, detected=(language_display == "Auto-detect"))
            qwen_lang = route.qwen_language or "English"
            qwen_instruct = _compose_qwen_instruction(qwen_style_prompt, float(speed), route.language_code)
            if route.backend == "qwen_clone":
                try:
                    clone_prompt, xvec_mode = _build_qwen_clone_prompt_cached(ref_wav)
                    print(f"🎙️ Using Qwen voice cloning backend. x-vector-only={xvec_mode}.")
                except Exception as e:
                    print(f"⚠️ Qwen clone setup failed: {e}. Falling back to XTTS.")
                    route = resolve_tts_route("XTTS", lang, has_reference=bool(original_ref_wav))
                    ref_wav, cleanup_path, _, _, _ = _prepare_capped_reference_audio(original_ref_wav, "XTTS")
                    if cleanup_path:
                        temp_cleanup_paths.append(cleanup_path)
                    backend_note = format_backend_status(lang, route, detected=(language_display == "Auto-detect"))
                    _, _, chunks = _build_chunks_for_route(
                        text=text,
                        route=route,
                        lang=lang,
                        use_ollama=bool(use_ollama),
                        ollama_model=ollama_model,
                    )
                    total_s, processed, truncated, truncated_mid_chunk, wavs_out, srs_out, err = _run_xtts_chunks(
                        chunks, lang, speaker_display, ref_wav, speed, max_seconds, total_s, wavs_out, srs_out, processed, truncated, truncated_mid_chunk
                    )
                    if err:
                        return None, err
                    if _TTS_STOP_REQUESTED:
                        user_stopped = True
                    route = None
            else:
                speaker_name = resolve_qwen_voice(_strip_voice_label(speaker_display), route.language_code, getattr(s, "qwen_voice_default", "Ryan"))
                print(f"🎙️ Using Qwen CustomVoice backend. Voice: {speaker_name}.")
            if route is not None:
                for i, ch in enumerate(chunks, start=1):
                    if _TTS_STOP_REQUESTED:
                        truncated = True
                        user_stopped = True
                        break
                    print(f"🔊 Qwen chunk {i}/{len(chunks)}")
                    _console_progress(i, len(chunks), prefix="TTS")
                    speak_text = ch.strip()
                    if not speak_text:
                        continue
                    try:
                        if route.backend == "qwen_clone":
                            wavs, sr = synthesize_qwen_clone(
                                [speak_text],
                                language=qwen_lang,
                                voice_clone_prompt=clone_prompt,
                                instruct=qwen_instruct,
                                max_new_tokens=DEFAULT_QWEN_MAX_NEW_TOKENS,
                                stop_event=_TTS_STOP_EVENT,
                            )
                        else:
                            wavs, sr = synthesize_qwen_custom(
                                [speak_text],
                                language=qwen_lang,
                                speaker=speaker_name,
                                instruct=qwen_instruct,
                                max_new_tokens=DEFAULT_QWEN_MAX_NEW_TOKENS,
                                stop_event=_TTS_STOP_EVENT,
                            )
                    except Exception as e:
                        if _TTS_STOP_REQUESTED:
                            user_stopped = True
                            truncated = True
                            break
                        print(f"⚠️ Qwen generation failed: {e}. Falling back to XTTS.")
                        route = resolve_tts_route("XTTS", lang, has_reference=bool(original_ref_wav))
                        ref_wav, cleanup_path, _, _, _ = _prepare_capped_reference_audio(original_ref_wav, "XTTS")
                        if cleanup_path:
                            temp_cleanup_paths.append(cleanup_path)
                        backend_note = format_backend_status(lang, route, detected=(language_display == "Auto-detect"))
                        _, _, chunks = _build_chunks_for_route(
                            text=text,
                            route=route,
                            lang=lang,
                            use_ollama=bool(use_ollama),
                            ollama_model=ollama_model,
                        )
                        total_s, processed, truncated, truncated_mid_chunk, wavs_out, srs_out, err = _run_xtts_chunks(
                            chunks, lang, speaker_display, ref_wav, speed, max_seconds, total_s, wavs_out, srs_out, processed, truncated, truncated_mid_chunk
                        )
                        if err:
                            return None, err
                        break
                    total_s, tr, tr_mid, done = _maybe_trim_and_collect(wavs_out, srs_out, wavs, sr, max_seconds, total_s)
                    truncated = truncated or tr
                    truncated_mid_chunk = truncated_mid_chunk or tr_mid
                    processed += done
                    if tr or _TTS_STOP_REQUESTED:
                        if _TTS_STOP_REQUESTED:
                            user_stopped = True
                        break

        if not wavs_out:
            if user_stopped:
                return None, "⛔️ Stopped by user before any audio could be synthesized."
            return None, f"⚠️ Output capped at {used_limit_min:.1f} min; nothing could be synthesized."

        final, output_sr = concat_audio_segments(wavs_out, srs_out)
        out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        sf.write(out_path, final, output_sr or 24000)

        warn = ""
        if truncated or processed < len(chunks):
            if not user_stopped:
                detail = "mid-sentence" if truncated_mid_chunk else "at a sentence boundary"
                warn = (
                    f"⚠️ Output capped at **{used_limit_min:.1f} minutes**; synthesized **{processed}/{len(chunks)}** chunk(s) and ignored the rest ({detail})."
                )
            else:
                warn = "⛔️ Stopped by user."

        pieces = [p for p in [warn, backend_note] if p]
        final_note = " ".join(pieces).strip()
        _console_progress(len(chunks), len(chunks), prefix="TTS", end=True)
        return out_path, final_note
    finally:
        _cleanup_temp_paths(temp_cleanup_paths)

# developer helpers

def inspect_chunking(text: str, max_chars: int | None = None):
    text = (text or "").strip()
    if not text:
        return [], [], "Provide some text above."
    max_len = int(max_chars) if max_chars else DEFAULT_MAX_CHARS_PER_CHUNK
    sents, chunks, backend = chunk_text(text, max_len=max_len, lang_code="xx")
    sent_rows = [[idx, len(s), s] for idx, s in enumerate(sents)]
    chunk_rows = [[idx, len(c), c] for idx, c in enumerate(chunks)]
    total_chars = sum(len(c) for c in chunks)
    summary = (
        f"**Sentences:** {len(sents)}\n\n"
        f"**Chunks (max {max_len} chars):** {len(chunks)}  •  "
        f"**Total chars across chunks:** {total_chars}\n\n"
        f"**Chunk backend:** {backend or 'unknown'}"
    )
    return sent_rows, chunk_rows, summary


def inspect_full_pipeline(text: str, max_chars: int | None = None, use_ollama: bool = False, ollama_model: str = ""):
    txt = (text or "").strip()
    if not txt:
        return "", [], [], "Paste some text above.", "⚠️ Empty text."
    det_label, det_code, det_prob, det_note = detect_tts_language(txt)
    dbg_lang = det_code or "en"
    s = get_settings()
    seg_chars = int(max_chars) if max_chars else effective_max_chars(
        lang_code=dbg_lang,
        user_cap=int(getattr(s, "xtts_max_chars_per_chunk", DEFAULT_MAX_CHARS_PER_CHUNK)),
        dynamic=bool(getattr(s, "xtts_dynamic_per_lang_caps", True)),
    )
    refined, sents, chunks = preprocess_to_chunks(
        text=txt,
        use_ollama=bool(use_ollama),
        ollama_model=ollama_model,
        max_chars=seg_chars,
        lang_code=dbg_lang,
        tts_family="xtts",
    )
    sent_rows = [[i, len(s), s] for i, s in enumerate(sents)]
    chunk_rows = [[i, len(c), c] for i, c in enumerate(chunks)]
    summary = (
        f"**Refined length:** {len(refined)} chars  \n"
        f"**Sentences:** {len(sents)}  \n"
        f"**Chunks (max {seg_chars} chars; lang={dbg_lang}):** {len(chunks)}  \n"
        f"**Total chars across chunks:** {sum(len(c) for c in chunks)}"
    )
    return refined, sent_rows, chunk_rows, summary, det_note
