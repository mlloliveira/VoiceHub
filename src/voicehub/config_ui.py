# src/voicehub/config_ui.py
import gradio as gr

from .tts_router import chunk_slider_state_for_family, clone_ref_slider_state_for_family
from .config import QWEN_MODEL_SIZE_OPTIONS
from .user_settings import apply_runtime_settings, get_settings, update_settings, reset_settings


def _apply_cfg_runtime(
    whisper_temperature,
    whisper_top_p,
    whisper_beam_size,
    whisper_condition_on_prev,
    asr_stream_max_minutes,
    tts_default_family,
    tts_chunk_size_shared,
    qwen_model_size,
    clone_ref_seconds_shared,
    xtts_max_minutes_default,
    xtts_dynamic_caps,
    ollama_temperature,
    ollama_top_p,
    ollama_num_predict,
    ollama_stop,
):
    family = str(tts_default_family or "XTTS")
    kwargs = dict(
        whisper_temperature=float(whisper_temperature),
        whisper_top_p=float(whisper_top_p),
        whisper_beam_size=int(whisper_beam_size),
        whisper_condition_on_prev=bool(whisper_condition_on_prev),
        asr_stream_max_minutes=float(asr_stream_max_minutes),
        tts_default_family=family,
        qwen_model_size=str(qwen_model_size or "1.7B"),
        xtts_max_minutes_default=float(xtts_max_minutes_default),
        xtts_dynamic_per_lang_caps=bool(xtts_dynamic_caps),
        ollama_temperature=float(ollama_temperature),
        ollama_top_p=float(ollama_top_p),
        ollama_num_predict=int(ollama_num_predict),
        ollama_stop=str(ollama_stop or "").strip(),
    )
    if family == "Qwen":
        kwargs["qwen_max_chars_per_chunk"] = int(tts_chunk_size_shared)
        kwargs["qwen_clone_ref_max_seconds"] = float(clone_ref_seconds_shared)
    else:
        kwargs["xtts_max_chars_per_chunk"] = int(tts_chunk_size_shared)
        kwargs["xtts_clone_ref_max_seconds"] = float(clone_ref_seconds_shared)
    return apply_runtime_settings(**kwargs)


def _save_cfg(
    whisper_temperature,
    whisper_top_p,
    whisper_beam_size,
    whisper_condition_on_prev,
    asr_stream_max_minutes,
    tts_default_family,
    tts_chunk_size_shared,
    qwen_model_size,
    clone_ref_seconds_shared,
    xtts_max_minutes_default,
    xtts_dynamic_caps,
    ollama_temperature,
    ollama_top_p,
    ollama_num_predict,
    ollama_stop,
):
    family = str(tts_default_family or "XTTS")
    kwargs = dict(
        whisper_temperature=float(whisper_temperature),
        whisper_top_p=float(whisper_top_p),
        whisper_beam_size=int(whisper_beam_size),
        whisper_condition_on_prev=bool(whisper_condition_on_prev),
        asr_stream_max_minutes=float(asr_stream_max_minutes),
        tts_default_family=family,
        qwen_model_size=str(qwen_model_size or "1.7B"),
        xtts_max_minutes_default=float(xtts_max_minutes_default),
        xtts_dynamic_per_lang_caps=bool(xtts_dynamic_caps),
        ollama_temperature=float(ollama_temperature),
        ollama_top_p=float(ollama_top_p),
        ollama_num_predict=int(ollama_num_predict),
        ollama_stop=str(ollama_stop or "").strip(),
    )
    if family == "Qwen":
        kwargs["qwen_max_chars_per_chunk"] = int(tts_chunk_size_shared)
        kwargs["qwen_clone_ref_max_seconds"] = float(clone_ref_seconds_shared)
    else:
        kwargs["xtts_max_chars_per_chunk"] = int(tts_chunk_size_shared)
        kwargs["xtts_clone_ref_max_seconds"] = float(clone_ref_seconds_shared)
    return update_settings(**kwargs)


def _slider_update_for_family(family: str):
    s = get_settings()
    state = chunk_slider_state_for_family(
        family,
        xtts_value=getattr(s, "xtts_max_chars_per_chunk", None),
        qwen_value=getattr(s, "qwen_max_chars_per_chunk", None),
    )
    return gr.update(
        minimum=state["minimum"],
        maximum=state["maximum"],
        value=state["value"],
        label=state["label"],
        info=state["info"],
    )


def _xtts_dynamic_interactive(family: str):
    return gr.update(interactive=(str(family) == "XTTS"))


def _clone_slider_update_for_family(family: str):
    s = get_settings()
    state = clone_ref_slider_state_for_family(
        family,
        xtts_value=getattr(s, "xtts_clone_ref_max_seconds", None),
        qwen_value=getattr(s, "qwen_clone_ref_max_seconds", None),
    )
    return gr.update(
        minimum=state["minimum"],
        maximum=state["maximum"],
        value=state["value"],
        label=state["label"],
        info=state["info"],
    )


def _qwen_prompt_visibility(family: str):
    return gr.update(visible=(str(family) == "Qwen"))


def build_config_tab(qwen_prompt_box=None):
    s = get_settings()
    initial_slider = chunk_slider_state_for_family(
        s.tts_default_family,
        xtts_value=s.xtts_max_chars_per_chunk,
        qwen_value=s.qwen_max_chars_per_chunk,
    )
    initial_clone_slider = clone_ref_slider_state_for_family(
        s.tts_default_family,
        xtts_value=getattr(s, "xtts_clone_ref_max_seconds", None),
        qwen_value=getattr(s, "qwen_clone_ref_max_seconds", None),
    )

    gr.Markdown("### ⚙️ Global Settings (per model)")

    gr.Markdown("**Automatic Speech Recognition - ASR (Whisper)**")
    with gr.Row():
        whisper_temperature = gr.Slider(0.0, 1.2, step=0.05, value=s.whisper_temperature, label="Temperature (Whisper)", info="Sampling randomness for decoder search.")
        whisper_top_p = gr.Slider(0.1, 1.0, step=0.05, value=s.whisper_top_p, label="Top-p (Whisper)", info="No current support in this build.", interactive=False)
        whisper_beam_size = gr.Slider(1, 10, step=1, value=s.whisper_beam_size, label="Beam size (Whisper)", info="Higher can improve accuracy at cost of speed.")
        asr_stream_max_minutes = gr.Slider(0.1, 10.0, step=0.1, value=s.asr_stream_max_minutes, label="ASR microphone (minutes)", info="Hard cap for streamed microphone capture.")
    whisper_condition_on_prev = gr.Checkbox(value=s.whisper_condition_on_prev, label="Condition on previous text (Whisper)", info="Provide prior transcript as context for next segment.")

    gr.Markdown("**Text-To-Speech - TTS (Qwen or XTTS)**")
    with gr.Row():
        tts_default_family = gr.Dropdown(["Qwen", "XTTS"], value=s.tts_default_family, label="Default TTS model", info="XTTS is the safe default. Qwen is optional and only loaded when selected.")
        qwen_model_size = gr.Dropdown(list(QWEN_MODEL_SIZE_OPTIONS), value=getattr(s, "qwen_model_size", "1.7B"), label="Qwen model size", info="Optional Qwen checkpoint family. 0.6B is lighter; 1.7B is stronger.")
        xtts_dynamic_caps = gr.Checkbox(value=s.xtts_dynamic_per_lang_caps, label="Dynamically handles max chars chunk per language (XTTS)", info="XTTS only. When on, the chunk cap below becomes a safety threshold on top of the language-aware cap.", interactive=(s.tts_default_family == "XTTS"))
        tts_chunk_size_shared = gr.Slider(
            minimum=initial_slider["minimum"],
            maximum=initial_slider["maximum"],
            step=5,
            value=(s.qwen_max_chars_per_chunk if s.tts_default_family == "Qwen" else s.xtts_max_chars_per_chunk),
            label=initial_slider["label"],
            info=initial_slider["info"],
        )
        clone_ref_seconds_shared = gr.Slider(
            minimum=initial_clone_slider["minimum"],
            maximum=initial_clone_slider["maximum"],
            step=5,
            value=initial_clone_slider["value"],
            label=initial_clone_slider["label"],
            info=initial_clone_slider["info"],
        )
        xtts_max_minutes_default = gr.Slider(0.5, 90.0, step=0.5, value=s.xtts_max_minutes_default, label="Max audio output length (TTS) in minutes", info="Default cap for synthesized audio length.")

    gr.Markdown("**Ollama (pre-chunker)**")
    with gr.Row():
        ollama_temperature = gr.Slider(0.0, 1.2, step=0.05, value=s.ollama_temperature, label="Temperature (Ollama)")
        ollama_top_p = gr.Slider(0.1, 1.0, step=0.05, value=s.ollama_top_p, label="Top-p (Ollama)")
        ollama_num_predict = gr.Slider(64, 16384, step=32, value=s.ollama_num_predict, label="Token cap (Ollama)")
    ollama_stop = gr.Textbox(value=s.ollama_stop, label="Stop sequences (Ollama, optional, comma-separated)", placeholder="e.g., ###, END")

    with gr.Row():
        save_btn = gr.Button("Save settings", variant="primary")
        reset_btn = gr.Button("Reset to recommended defaults")
    cfg_echo = gr.JSON(label="Current settings (for reference)")

    tts_default_family.change(_slider_update_for_family, [tts_default_family], [tts_chunk_size_shared], concurrency_limit=1)
    tts_default_family.change(_clone_slider_update_for_family, [tts_default_family], [clone_ref_seconds_shared], concurrency_limit=1)
    tts_default_family.change(_xtts_dynamic_interactive, [tts_default_family], [xtts_dynamic_caps], concurrency_limit=1)
    if qwen_prompt_box is not None:
        tts_default_family.change(_qwen_prompt_visibility, [tts_default_family], [qwen_prompt_box], concurrency_limit=1)

    runtime_inputs = [
        whisper_temperature,
        whisper_top_p,
        whisper_beam_size,
        whisper_condition_on_prev,
        asr_stream_max_minutes,
        tts_default_family,
        tts_chunk_size_shared,
        qwen_model_size,
        clone_ref_seconds_shared,
        xtts_max_minutes_default,
        xtts_dynamic_caps,
        ollama_temperature,
        ollama_top_p,
        ollama_num_predict,
        ollama_stop,
    ]
    for comp in runtime_inputs:
        comp.change(_apply_cfg_runtime, runtime_inputs, [cfg_echo], concurrency_limit=1)

    save_btn.click(
        _save_cfg,
        [
            whisper_temperature,
            whisper_top_p,
            whisper_beam_size,
            whisper_condition_on_prev,
            asr_stream_max_minutes,
            tts_default_family,
            tts_chunk_size_shared,
            qwen_model_size,
            clone_ref_seconds_shared,
            xtts_max_minutes_default,
            xtts_dynamic_caps,
            ollama_temperature,
            ollama_top_p,
            ollama_num_predict,
            ollama_stop,
        ],
        [cfg_echo],
        concurrency_limit=1,
    )
    reset_btn.click(
        _reset_ui,
        inputs=[],
        outputs=[
            whisper_temperature,
            whisper_top_p,
            whisper_beam_size,
            whisper_condition_on_prev,
            asr_stream_max_minutes,
            tts_default_family,
            tts_chunk_size_shared,
            qwen_model_size,
            clone_ref_seconds_shared,
            xtts_max_minutes_default,
            xtts_dynamic_caps,
            ollama_temperature,
            ollama_top_p,
            ollama_num_predict,
            ollama_stop,
            cfg_echo,
        ],
        concurrency_limit=1,
    )
    return {"tts_default_family": tts_default_family}


def _reset_ui():
    s = reset_settings()
    slider = chunk_slider_state_for_family(
        s["tts_default_family"],
        xtts_value=s.get("xtts_max_chars_per_chunk"),
        qwen_value=s.get("qwen_max_chars_per_chunk"),
    )
    clone_slider = clone_ref_slider_state_for_family(
        s["tts_default_family"],
        xtts_value=s.get("xtts_clone_ref_max_seconds"),
        qwen_value=s.get("qwen_clone_ref_max_seconds"),
    )
    return (
        gr.update(value=s["whisper_temperature"]),
        gr.update(value=s["whisper_top_p"]),
        gr.update(value=s["whisper_beam_size"]),
        gr.update(value=s["whisper_condition_on_prev"]),
        gr.update(value=s["asr_stream_max_minutes"]),
        gr.update(value=s["tts_default_family"]),
        gr.update(minimum=slider["minimum"], maximum=slider["maximum"], value=slider["value"], label=slider["label"], info=slider["info"]),
        gr.update(value=s["qwen_model_size"]),
        gr.update(minimum=clone_slider["minimum"], maximum=clone_slider["maximum"], value=clone_slider["value"], label=clone_slider["label"], info=clone_slider["info"]),
        gr.update(value=s["xtts_max_minutes_default"]),
        gr.update(value=s["xtts_dynamic_per_lang_caps"], interactive=(s["tts_default_family"] == "XTTS")),
        gr.update(value=s["ollama_temperature"]),
        gr.update(value=s["ollama_top_p"]),
        gr.update(value=s["ollama_num_predict"]),
        gr.update(value=s["ollama_stop"]),
        s,
    )
