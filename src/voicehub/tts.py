# src/voicehub/tts.py
import re, tempfile, numpy as np, soundfile as sf, gradio as gr
import sys

from TTS.api import TTS
from .config import effective_max_chars, TTS_MODEL_NAME, DEFAULT_MAX_CHARS_PER_CHUNK, TTS_LANG_MAP, DEBUG_TOOLS
from .user_settings import get_settings
from .progress_utils import console_progress as _console_progress
from .lang_detect import detect_tts_language
# Import only Ollama knobs
from .ollama_config import OLLAMA_ENABLE_DEFAULT, OLLAMA_MODEL_DEFAULT
from .ollama_utils import has_ollama, refine_text_with_ollama

# ===================== TTS: Coqui XTTS-v2 (dynamic speakers + optional cloning) =====================
tts = TTS(TTS_MODEL_NAME) #If not stored yet in ~/.local/share/tts it downloads the model from Hugging Face
try: #Try to load in the GPU, if fails, goes to the CPU
    tts.to("cuda")
except Exception as e:
    print(f"⚠️ Could not move XTTS-v2 to GPU, falling back to CPU. Reason: {e}")
    tts.to("cpu")

# Discover speakers dynamically and label with -en-us / -pt-br
SPEAKER_DISPLAY_TO_NAME = {}

def get_available_speakers(): #Inspects the loaded XTTS-v2 model to discover the built-in speakers
    names = []
    try:
        sm = getattr(tts.synthesizer.tts_model, "speaker_manager", None) #XTTS exposes speakers via: tts.synthesizer.tts_model.speaker_manager.speakers
        sp = getattr(sm, "speakers", None) if sm is not None else None
        if isinstance(sp, dict):#Can be a dict OR...
            names = list(sp.keys())
        elif isinstance(sp, (list, tuple)):#... it can be a list, depending on the version
            names = list(sp)
    except Exception:
        pass
    return sorted({n.strip() for n in names if isinstance(n, str) and n.strip()})

def make_speaker_choices(language_display): #Produces *display labels* for the dropdown and updates the mapping from display
    base_names = get_available_speakers()
    choices = list(base_names)
    SPEAKER_DISPLAY_TO_NAME.clear()
    SPEAKER_DISPLAY_TO_NAME.update({name: name for name in base_names}) # mapping stays (identity mapping: display name == backend name)
    default = choices[0] if choices else None
    return choices, default

def refresh_speakers(language_display): #Refresh the list of speakers
    choices, default = make_speaker_choices(language_display)
    # Gradio 4.x: return gr.update(...) for the target component
    if not choices:
        # disable the dropdown if no speakers found
        return gr.update(choices=[], value=None, interactive=False)
    return gr.update(choices=choices, value=default, interactive=True)

# ---- chunking helpers ----
def _split_into_sentences(text: str): #Naive sentence splitter using a regex on punctuation such as . ! ? : ;
    parts = re.split(r'(?<=[\.\!\?\:;\u3002])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()] # Returns clean non-empty sentences. This makes chunk assembly 'sentence-first' for better flow and fewer mid-thought breaks.

def _chunk_by_length(sentences, max_len=DEFAULT_MAX_CHARS_PER_CHUNK): #Greedy chunker
    """
    Greedy chunker to keep each TTS input bellow desired chars size.
    - Builds chunks by appending whole sentences while <= max_len.
    - If a single sentence > max_len, word-wrap it into sub-chunks (<= max_len).
    - Flush any pending 'cur' before splitting a long sentence
      so the relative order is never changed.
    """
    chunks, cur = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue

        if len(s) <= max_len:
            # try to append to current chunk
            if len(cur) + (1 if cur else 0) + len(s) <= max_len:
                cur = f"{cur} {s}".strip() if cur else s
            else:
                if cur:
                    chunks.append(cur)
                cur = s
        else:
            if cur:
                chunks.append(cur)
                cur = ""

            # word-wrap the long sentence
            buf = ""
            for w in s.split():
                if len(buf) + (1 if buf else 0) + len(w) <= max_len:
                    buf = f"{buf} {w}".strip() if buf else w
                else:
                    if buf:
                        chunks.append(buf)
                    buf = w
            if buf:
                chunks.append(buf)

    if cur:
        chunks.append(cur)
    return chunks #Very long tokens (no spaces) can still exceed max_len; those become a single chunk equal to that token, but XTTS will still speak it; it just may be awkward.

def _tts_chunk(text_chunk, lang, speaker_name=None, ref_wav=None, speed=1.0): #Synthesizes ONE chunk with XTTS
    if ref_wav: #Cloning: if ref_wav is provided -> pass speaker_wav=[ref_wav].
        wav = tts.tts(
            text=text_chunk, language=lang, speaker_wav=[ref_wav],
            speed=float(speed), split_sentences=False
        )
    else: #Built-in voice: else -> pass speaker=speaker_name.
        wav = tts.tts(
            text=text_chunk, language=lang, speaker=speaker_name,
            speed=float(speed), split_sentences=False
        )
    if isinstance(wav, tuple) and len(wav) == 2:
        wav, sr = wav
    else:
        sr = 24000
    return np.asarray(wav, dtype=np.float32), int(sr) #We normalize to (np.ndarray, sr) and default sr=24000 when waveform not provided

### Ollama Helper ###
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
):
    """
    Raw text → (optional Ollama refine) → naive sentence split → greedy chunking (<= max_chars).
    Returns: refined_text, sentences_list, chunks_list
    """
    text = (text or "").strip()
    refined = text
    if not text:
        return "", [], []

    if use_ollama:
        model = (ollama_model or OLLAMA_MODEL_DEFAULT).strip()
        ok, _ = has_ollama()
        if ok and model:
            try:
                refined_out = refine_text_with_ollama(
                    text, max_chars=int(max_chars), model=model, options=_build_ollama_options_from_settings()
                )
                if refined_out:
                    refined = refined_out
            except Exception as e:
                print(f"[TTS] Ollama refine failed: {e}. Keeping original text.")

    sents = _split_into_sentences(refined)
    chunks = _chunk_by_length(sents, max_len=int(max_chars))
    return refined, sents, chunks

### Create conditions and functions to force a stop:
_TTS_STOP_REQUESTED = False

def reset_tts_stop_flag():
    """Clear STOP flag before a new TTS run."""
    global _TTS_STOP_REQUESTED
    _TTS_STOP_REQUESTED = False

def request_tts_stop():
    """
    UI hook: set STOP flag.
    Returns a short status string to show in the UI.
    """
    global _TTS_STOP_REQUESTED
    _TTS_STOP_REQUESTED = True
    return "⛔️ Stop requested — finishing current chunk and returning what's done."

#### --- Main TTS Function!! --- ####
def synthesize_tts(
    text,
    language_display,
    speaker_display,
    speed,
    ref_wav,
    use_ollama=False,
    ollama_model="",
):
    # Orchestrates the whole TTS process and enforces a hard duration cap.
    """
    Steps
    -----
    1) Validate inputs; map language_display -> language code dynamically.
    2) (Optional) Pre-process with Ollama to refine punctuation & segment boundaries.
       - Raw text => Ollama => Refined text with better punctuation & <= seg_chars chars per line.
       - We then pass that refined text into our naive splitter + greedy chunker for robustness.
       * This is centralized in `preprocess_to_chunks(...)` so Debug and TTS are identical.
    3) Decide voice:
       - If `ref_wav` present -> cloning mode.
       - Else map `speaker_display` -> real speaker name via SPEAKER_DISPLAY_TO_NAME.
       - Else fallback to first discovered speaker, or raise a friendly error if none.
    4) Split text into sentences, then chunk to <= seg_chars chars (same as Debug).
    5) Synthesize with a hard duration cap, with epsilon tolerance to avoid false truncation warnings.
    """

    # 0) Reset stop flag at the start of each run ---
    reset_tts_stop_flag()

    # --- 1) Validate input text ---
    text = (text or "").strip()
    if not text:
        return None, ""
    
    # Check if Auto-Detect is on display. If it is, detect language. If not, use selected language
    det_note = ""
    if language_display == "Auto-detect":
        det_label, det_code, det_score, det_note = detect_tts_language(text)
        lang = det_code or "en"
    else:
        # dynamic language mapping: UI display -> language code (e.g., "English" -> "en")
        lang = TTS_LANG_MAP.get(language_display, "en")

    # global settings
    s = get_settings()

    # cap on output duration (minutes -> seconds)
    used_limit_min = float(s.xtts_max_minutes_default)  # Config tab value
    max_seconds = max(0.5, used_limit_min) * 60.0

    # --- 2) Unified preprocessing (makes TTS == Debug) ---

    # decide per-run segment size (same used for Ollama + chunker)
    seg_chars = effective_max_chars(
        lang_code=lang or "en",
        user_cap=int(getattr(s, "xtts_max_chars_per_chunk", DEFAULT_MAX_CHARS_PER_CHUNK)),
        dynamic=bool(getattr(s, "xtts_dynamic_per_lang_caps", True)),
    )
    if DEBUG_TOOLS:
        print(f"[TTS] seg_chars={seg_chars}, use_ollama={bool(use_ollama)}, model='{ollama_model}'")

    _, _, chunks = preprocess_to_chunks(
        text=text,
        use_ollama=bool(use_ollama),
        ollama_model=ollama_model,
        max_chars=seg_chars,
    )

    total_chunks = len(chunks)
    if total_chunks == 0:
        print("[TTS] No content after preprocessing.")
        return None, "⚠️ No content after preprocessing."
    print(f"[TTS] Preparing {total_chunks} chunk(s)…")

    # --- 3) Voice selection (built-in vs cloning) ---
    # pick speaker by display label (only used if not cloning)
    speaker_name = SPEAKER_DISPLAY_TO_NAME.get(speaker_display, None)
    if not speaker_name and not ref_wav:
        avail = get_available_speakers()
        if avail:
            speaker_name = avail[0]
        else:
            raise RuntimeError("No built-in speakers found. Upload a reference WAV to clone a voice.")

    # --- 5) Synthesis loop with hard cap & epsilon tolerance ---
    wavs, sample_rate = [], None
    total_s = 0.0
    truncated = False
    truncated_mid_chunk = False
    processed = 0  # how many chunks we actually synthesized
    user_stopped = False # track user stop

    # console progress (non-intrusive)
    _console_progress(0, total_chunks, prefix="TTS")

    for i, ch in enumerate(chunks, start=1):

        # if the user already hit STOP, bail before sending the next chunk
        if _TTS_STOP_REQUESTED:
            truncated = True
            user_stopped = True
            break

        # Simple console bar 1/… n/…
        _console_progress(i, total_chunks, prefix="TTS")  

        # synthesize one chunk (cloning if ref_wav else use speaker_name)
        w, sr = _tts_chunk(ch, lang, speaker_name=speaker_name, ref_wav=ref_wav, speed=speed)
        if sample_rate is None:
            sample_rate = sr

        dur = len(w) / sr
        remain = max_seconds - total_s

        # Tolerance: treat “almost equal” as fitting.
        EPS = max(1.0 / sr, 0.001)  # ≈ 1 sample or 1 ms, whichever is larger

        if remain <= 0:
            truncated = True
            break

        if dur <= remain + EPS:
            # Fits (within tolerance) → take whole chunk
            wavs.append(w)
            total_s += dur
            processed += 1
        else:
            # Needs trimming: compute samples to keep with rounding
            n_samples = int(remain * sr + 0.5)
            if n_samples >= len(w) - 1:
                # Effectively full chunk (numeric noise) → don't warn
                wavs.append(w)
                total_s += dur
                processed += 1
            elif n_samples > 0:
                wavs.append(w[:n_samples])
                total_s += n_samples / sr
                truncated = True
                truncated_mid_chunk = True
                processed += 1
            else:
                truncated = True  # nothing left to add
            break

        # check STOP again after completing a chunk
        if _TTS_STOP_REQUESTED:
            truncated = True
            user_stopped = True
            break    

    if not wavs:
        if user_stopped:
            return None, "⛔️ Stopped by user before any audio could be synthesized."
        return None, f"⚠️ Output capped at {used_limit_min:.1f} min; nothing could be synthesized."

    final = np.concatenate(wavs) if len(wavs) > 1 else wavs[0]
    out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    sf.write(out_path, final, sample_rate or 24000)

    # build warning (include how many chunks we actually generated)
    warn = ""
    if truncated or processed < total_chunks:
        if not user_stopped:
            detail = "mid-sentence" if truncated_mid_chunk else "at a sentence boundary"
            warn = (
                f"⚠️ Output capped at **{used_limit_min:.1f} minutes**; "
                f"synthesized **{processed}/{total_chunks}** chunk(s) and ignored the rest "
                f"({detail})."
            )
        else:  #Warning from user pressing stop
            warn = ("⛔️ Stopped by user. " + (warn or "")).strip()      

    if warn != "":
        print(warn)

    if det_note: #Warning from lang_detect
        warn = (warn + " " + det_note).strip() if warn else det_note  

    _console_progress(total_chunks, total_chunks, prefix="TTS", end=True)  # final console bar
    return out_path, warn


# --- developer helper stays here so your Debug tab can import it from TTS ---
def inspect_chunking(text: str, max_chars: int | None = None):
    """
    Developer helper: show how text is broken into sentences and then chunks.
    Returns (sentences_table_rows, chunks_table_rows, summary_markdown)
    """
    text = (text or "").strip()
    if not text:
        return [], [], "Provide some text above."

    max_len = int(max_chars) if max_chars else DEFAULT_MAX_CHARS_PER_CHUNK
    sents = _split_into_sentences(text)
    chunks = _chunk_by_length(sents, max_len=max_len)

    sent_rows = [[idx, len(s), s] for idx, s in enumerate(sents)]
    chunk_rows = [[idx, len(c), c] for idx, c in enumerate(chunks)]

    total_chars = sum(len(c) for c in chunks)
    summary = (
        f"**Sentences:** {len(sents)}\n\n"
        f"**Chunks (max {max_len} chars):** {len(chunks)}  •  "
        f"**Total chars across chunks:** {total_chars}\n\n"
        "Chunks are built greedily from full sentences, word-wrapping only when a single sentence exceeds the max length."
    )
    return sent_rows, chunk_rows, summary

# --- developer: end-to-end inspection helper (raw -> refined -> sentences -> chunks) ---
from .ollama_config import OLLAMA_MAX_SEG_CHARS, OLLAMA_MODEL_DEFAULT  # top of file with other imports
from .ollama_utils import has_ollama, refine_text_with_ollama           # already used by synthesize_tts

def inspect_full_pipeline(
    text: str,
    max_chars: int | None = None,
    use_ollama: bool = False,
    ollama_model: str = "",
):
    """
    Debug pipeline:
      Raw text → (optional Ollama refine) → sentence split → greedy chunk (<= max_chars)
      PLUS: language detection (normalized prob) for quick verification in Debug UI.
    Returns:
      refined_text, sentence_rows, chunk_rows, summary_md, detect_md
    """
    txt = (text or "").strip()
    if not txt:
        return "", [], [], "Paste some text above.", "⚠️ Empty text."
    
    # Language detection on the original text
    det_label, det_code, det_prob, det_note = detect_tts_language(txt)
    detect_md = det_note if det_note else (
        f"Detected **{det_label}** ({det_code}) with confidence **{det_prob:.2f}**."
        if det_code else "⚠️ Detection unavailable."
    )
    dbg_lang = det_code or "en"

    # use same seg length default as TTS unless overridden by the slider
    s = get_settings()
    seg_chars = effective_max_chars(
        lang_code=dbg_lang,
        user_cap=int(getattr(s, "xtts_max_chars_per_chunk", DEFAULT_MAX_CHARS_PER_CHUNK)),
        dynamic=bool(getattr(s, "xtts_dynamic_per_lang_caps", True)),
    )

    # unified preprocessing (same as TTS)
    refined, sents, chunks = preprocess_to_chunks(
        text=txt,
        use_ollama=bool(use_ollama),
        ollama_model=ollama_model,
        max_chars=seg_chars,
    )

    sent_rows  = [[i, len(s), s] for i, s in enumerate(sents)]
    chunk_rows = [[i, len(c), c] for i, c in enumerate(chunks)]

    summary = (
        f"**Refined length:** {len(refined)} chars  \n"
        f"**Sentences:** {len(sents)}  \n"
        f"**Chunks (max {seg_chars} chars; lang={dbg_lang}):** {len(chunks)}  \n"
        f"**Total chars across chunks:** {sum(len(c) for c in chunks)}"
    )

    

    return refined, sent_rows, chunk_rows, summary, detect_md