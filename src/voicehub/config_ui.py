# src/voicehub/config_ui.py
# Builds the "Config" tab UI (after TTS, before Debug).
# Per-model knobs with concise explanations.

import gradio as gr
from .user_settings import get_settings, update_settings, reset_settings
from .config import DEFAULT_MAX_CHARS_PER_CHUNK  # reflect your config default in the UI

def _save_cfg(
    # Whisper
    whisper_temperature, whisper_top_p, whisper_beam_size, whisper_condition_on_prev, asr_stream_max_minutes,
    # XTTS (global chunking only)
    xtts_max_chars_per_chunk, xtts_max_minutes_default, xtts_dynamic_caps,
    # Ollama
    ollama_temperature, ollama_top_p, ollama_num_predict, ollama_stop
):
    return update_settings(
        whisper_temperature=float(whisper_temperature),
        whisper_top_p=float(whisper_top_p),
        whisper_beam_size=int(whisper_beam_size),
        whisper_condition_on_prev=bool(whisper_condition_on_prev),
        asr_stream_max_minutes=float(asr_stream_max_minutes),
        xtts_max_chars_per_chunk=int(xtts_max_chars_per_chunk),
        xtts_max_minutes_default=float(xtts_max_minutes_default),
        xtts_dynamic_per_lang_caps=bool(xtts_dynamic_caps),
        ollama_temperature=float(ollama_temperature),
        ollama_top_p=float(ollama_top_p),
        ollama_num_predict=int(ollama_num_predict),
        ollama_stop=str(ollama_stop or "").strip(),
    )

def build_config_tab():
    s = get_settings()

    with gr.Tab("⚙️ Config"):
        gr.Markdown("### ⚙️ Global Settings (per model)")#\nDefaults apply across the app. You can still override some per-action.")

        # ---- Whisper (ASR) ----
        gr.Markdown("**Whisper (ASR)**")
        with gr.Row():
            whisper_temperature = gr.Slider(
                minimum=0.0, maximum=1.2, step=0.05, value=s.whisper_temperature,
                label="Temperature (Whisper)",
                info="Sampling randomness for decoder search. Lower is more deterministic; higher can recover edge cases."
            )
            whisper_top_p = gr.Slider( #Backend doesn't current support because current Whisper model doesn't support it.
                minimum=0.1, maximum=1.0, step=0.05, value=s.whisper_top_p,
                label="Top-p (Whisper)",
                info="NO CURRENT SUPPORT! Nucleus sampling mass for decoding. Lower = safer.",
                interactive=not True # Freeze (not working anyway)
            )
            whisper_beam_size = gr.Slider(
                minimum=1, maximum=10, step=1, value=s.whisper_beam_size,
                label="Beam size (Whisper)",
                info="Parallel hypotheses explored. Higher can improve accuracy at cost of speed."
            )
            asr_stream_max_minutes = gr.Slider(
                minimum=0.1, maximum=10.0, step=0.1, value=s.asr_stream_max_minutes,
                label="ASR microphone (minutes)",
                info="Max capture length. Hard cap for streamed microphone capture; extra audio is ignored."
            )
        whisper_condition_on_prev = gr.Checkbox(
            value=s.whisper_condition_on_prev,
            label="Condition on previous text (Whisper)",
            info="Provide prior transcript as context for next segment; helps coherence across segments."
        )

        # ---- XTTS (TTS) ----
        gr.Markdown("**XTTS (TTS)**")
        with gr.Row():
            xtts_dynamic_caps = gr.Checkbox(
                value=s.xtts_dynamic_per_lang_caps,
                label="Dynamically handles max chars per chunk per language",
                info="Recommended. Uses model-informed caps (e.g., en≈250, pt≈200, it≈210).",
            )
            xtts_max_chars_per_chunk = gr.Slider(
                minimum=60, maximum=400, step=5, value=s.xtts_max_chars_per_chunk,
                label=f"Max characters per chunk (XTTS) • Manual",
                info="Upper bound per XTTS input string. Keep bellow limit to avoid truncation. Ignored when dynamic is on.",
                interactive=not s.xtts_dynamic_per_lang_caps,   # Freeze when dynamic (xtts_dynamic_caps) is ON
            )
            xtts_max_minutes_default = gr.Slider(
                minimum=0.5, maximum=30.0, step=0.5, value=s.xtts_max_minutes_default,
                label="Max audio output length (XTTS) in minutes",
                info="Default cap for synthesized audio length; longer text will be truncated."
            )
        def _toggle_slider(dyn: bool): # Reactively freeze/unfreeze the slider
            return gr.update(interactive=not dyn)
        xtts_dynamic_caps.change(_toggle_slider, [xtts_dynamic_caps], [xtts_max_chars_per_chunk])

        # ---- Ollama (LLM pre-chunker) ----
        gr.Markdown("**Ollama (pre-chunker)**")
        with gr.Row():
            ollama_temperature = gr.Slider(
                minimum=0.0, maximum=1.2, step=0.05, value=s.ollama_temperature,
                label="Temperature (Ollama)",
                info="Sampling randomness for refining punctuation/segments. Lower is safer."
            )
            ollama_top_p = gr.Slider(
                minimum=0.1, maximum=1.0, step=0.05, value=s.ollama_top_p,
                label="Top-p (Ollama)",
                info="Nucleus sampling mass for the LLM; balances safety vs. variety."
            )
            ollama_num_predict = gr.Slider(
                minimum=64, maximum=16384, step=32, value=s.ollama_num_predict,
                label="Token cap (Ollama)",
                info="Max tokens to generate while refining lines."
            )
        ollama_stop = gr.Textbox(
            value=s.ollama_stop,
            label="Stop sequences (Ollama, optional, comma-separated)",
            placeholder="e.g., ###, END",
            info="End generation if any of these substrings appear."
        )

        # ---- Save / Reset & Echo ----
        with gr.Row():
            save_btn = gr.Button("Save settings", variant="primary")
            reset_btn = gr.Button("Reset to recommended defaults")
        cfg_echo = gr.JSON(label="Current settings (for reference)")

        save_btn.click(
            _save_cfg,
            [
                whisper_temperature, whisper_top_p, whisper_beam_size, whisper_condition_on_prev,asr_stream_max_minutes,
                xtts_max_chars_per_chunk,xtts_max_minutes_default,xtts_dynamic_caps,
                ollama_temperature, ollama_top_p, ollama_num_predict, ollama_stop
            ],
            [cfg_echo],
            concurrency_limit=1,
        )
        #reset_btn.click(lambda: reset_settings(), inputs=[], outputs=[cfg_echo], concurrency_limit=1)
        reset_btn.click(
            _reset_ui,
            inputs=[],
            outputs=[
                whisper_temperature,
                whisper_top_p,
                whisper_beam_size,
                whisper_condition_on_prev,
                asr_stream_max_minutes,
                xtts_dynamic_caps,
                xtts_max_chars_per_chunk,
                xtts_max_minutes_default,
                ollama_temperature,
                ollama_top_p,
                ollama_num_predict,
                ollama_stop,
                cfg_echo,  # keep the JSON echo updated too
            ],
            concurrency_limit=1,
        )

# Reset Everything to the default
def _reset_ui():
    # Reset the underlying settings object to recommended defaults
    s = reset_settings()  # returns dict

    # Return updates for every control + the cfg echo (last)
    return (
        gr.update(value=s["whisper_temperature"]),
        gr.update(value=s["whisper_top_p"]),
        gr.update(value=s["whisper_beam_size"]),
        gr.update(value=s["whisper_condition_on_prev"]),
        gr.update(value=s["asr_stream_max_minutes"]),
        gr.update(value=s["xtts_dynamic_per_lang_caps"]),
        # also re-apply disabled/enabled state for the manual chars slider:
        gr.update(value=s["xtts_max_chars_per_chunk"], interactive=not s["xtts_dynamic_per_lang_caps"]),
        gr.update(value=s["xtts_max_minutes_default"]),
        gr.update(value=s["ollama_temperature"]),
        gr.update(value=s["ollama_top_p"]),
        gr.update(value=s["ollama_num_predict"]),
        gr.update(value=s["ollama_stop"]),
        s  # cfg_echo JSON
    )