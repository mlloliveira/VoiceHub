# src/voicehub/ui.py
import gradio as gr
from .config import ASR_LANG_DISPLAY, ASR_LANG_MAP, BACKENDS, TTS_LANG_DISPLAY, DEBUG_TOOLS, DEFAULT_TTS_SPEED
from .asr import transcribe, translate_asr_text, stream_reset, stream_append, stream_finalize_and_transcribe, request_asr_stop
from .tts import make_speaker_choices, refresh_speakers, synthesize_tts, request_tts_stop
from .config_ui import build_config_tab
from .debug_ui import build_debug_tab
# ---- Pull Ollama defaults + test from dedicated modules
from .ollama_config import OLLAMA_ENABLE_DEFAULT, OLLAMA_MODEL_DEFAULT, set_ollama_default_model
from .ollama_utils import test_ollama_connection, list_ollama_models
# ---- User Settings
from .user_settings import get_settings

#If the user has Ollama, check the available models:
def _refresh_ollama_models(current_value: str | None):
    """
    Backend handler: refresh the dropdown choices from local Ollama.
    - If Ollama down: keep default choice (OLLAMA_MODEL_DEFAULT) and show a warning.
    - If up but no models: keep default and warn.
    - Else: populate with local models and preserve current selection when possible.
    """
    ok, names, msg = list_ollama_models()
    if not ok or not names:
        # Fallback to the configured default; still allow custom entry
        warn = f"{'❌' if not ok else '⚠️'} {msg}\nDefaulting to **{OLLAMA_MODEL_DEFAULT}**."
        return (
            gr.update(choices=[OLLAMA_MODEL_DEFAULT], value=OLLAMA_MODEL_DEFAULT, interactive=True),
            warn,
        )
    # pick a sane value: keep current if still present; else prefer configured default; else first
    value = current_value if current_value in names else (OLLAMA_MODEL_DEFAULT if OLLAMA_MODEL_DEFAULT in names else names[0])
    return gr.update(choices=names, value=value, interactive=True), f"✅ {msg}"

def _save_default_model(current_model: str):
    # Return both a dropdown update and a status line
    import gradio as gr
    msg = set_ollama_default_model(current_model)
    return gr.update(value=current_model), msg

#Helper function to delete Auto-detect:
def _asr_translate_choices():
    # Keep only real languages (i.e., code is not None)
    return [label for label in ASR_LANG_DISPLAY if ASR_LANG_MAP.get(label)]

#Include Auto-Detect in the TTS Language
tts_lang_choices = ["Auto-detect"] + TTS_LANG_DISPLAY

#Build the App UI:
def build_app():
    s = get_settings()  # per-model defaults
    # ===================== UI =====================
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🗣️ VoiceHub: Multilingual ASR + TTS\nAutomatic Speech Recognition + Text To Speech") #Top name

        with gr.Tabs():
            # -------- ASR TAB --------
            with gr.Tab("Speech → Text"): 
                with gr.Row():
                    with gr.Tab("📤 Upload"): 
                        audio_in = gr.Audio(
                            sources=["upload"], type="filepath",  # Do ["upload","microphone"] to add microphone here too. But it caps within 30s
                            format="wav",
                            max_length=600,  # allow up to 10 min mic recordings
                            label="Upload audio",
                        )
                    with gr.Tab("🎤 Microphone"): 
                        mic_stream = gr.Audio(
                        sources=["microphone"], streaming=True, type="numpy",
                        label="Microphone", format="wav"
                        )
                         # Preview of the captured audio (will be filled on Stop)
                        asr_preview_audio = gr.Audio(label="Captured audio (preview)", type="filepath", autoplay=False) 
                        asr_rec_status = gr.Markdown("")             
                    # hidden per-session buffer
                    asr_stream_state = gr.State(stream_reset())
                    with gr.Column():
                        lang_dd = gr.Dropdown(ASR_LANG_DISPLAY, value=ASR_LANG_DISPLAY[0], label="Language")
                        backend_dd = gr.Dropdown(BACKENDS, value="Faster-Whisper (GPU, recommended)", label="ASR backend")
                        beam = gr.Slider(1, 10, value=s.whisper_beam_size, step=1, label="Beam size")
                        use_vad = gr.Checkbox(value=True, label="VAD (silence filter) — FW only")
                        with gr.Row():
                            go_asr = gr.Button("Transcribe", variant="primary")
                            stop_asr = gr.Button("🛑", variant="secondary", scale=0, min_width=60)
                        asr_meta = gr.Markdown("")
                with gr.Row():
                    # add show_copy_button for convenience
                    out_text = gr.Textbox(label="Transcript", lines=6, show_copy_button=True)
                    #Advanced Button for AI Translation using Ollama
                with gr.Row():
                    with gr.Accordion("Advanced", open=False):
                        gr.Markdown("**AI Translate (Ollama backend, optional)**")
                        with gr.Row():
                            _translate_choices = _asr_translate_choices()
                            asr_trg_lang = gr.Dropdown(
                                choices=_translate_choices,
                                value=_translate_choices[0] if _translate_choices else "English",
                                label="Translate transcript to",
                                info="Uses local Ollama LLM to translate the current transcript."
                            )
                            asr_ollama_model = gr.Dropdown(
                                choices=[OLLAMA_MODEL_DEFAULT],
                                value=OLLAMA_MODEL_DEFAULT,
                                label="Ollama model",
                                allow_custom_value=True,
                                info="Select an installed local model (click Refresh), or type a tag."
                            )
                        with gr.Row():
                            asr_test_btn = gr.Button("Test Ollama")
                            asr_refresh_models_btn = gr.Button("Refresh models")
                            asr_save_default_btn = gr.Button("Set as default model")
                        with gr.Row():
                            asr_translate_btn = gr.Button("Translate", variant="primary")
                        with gr.Row():
                            asr_translation = gr.Textbox(label="Translation", lines=6, show_copy_button=True)
                        asr_ollama_status = gr.Markdown("")
                    # wire up model refresh + test
                    asr_refresh_models_btn.click(
                        _refresh_ollama_models,
                        [asr_ollama_model],
                        [asr_ollama_model, asr_ollama_status],
                        concurrency_limit=1,
                    )
                    asr_test_btn.click(
                        test_ollama_connection,
                        [asr_ollama_model],
                        [asr_ollama_status],
                        concurrency_limit=1,
                    )
                    # perform translation (no UI progress — LLMs vary; console is fine if you want logs)
                    asr_translate_btn.click(
                        translate_asr_text,                          # call backend
                        [out_text, asr_trg_lang, asr_ollama_model],  # transcript, target (display), model tag
                        [asr_translation, asr_ollama_status],        # translated text, status line
                        concurrency_limit=1,
                    )
                    asr_save_default_btn.click(
                        _save_default_model,
                        [asr_ollama_model],
                        [asr_ollama_model, asr_ollama_status],
                        concurrency_limit=1,
                    )
                #Event Wire Mic
                mic_stream.start_recording(
                    fn=lambda: (stream_reset(), "🎙️ Recording…", None),
                    inputs=[],
                    outputs=[asr_stream_state, asr_rec_status, asr_preview_audio],
                    concurrency_limit=1,
                )
                mic_stream.stream(
                    fn=stream_append,
                    inputs=[mic_stream, asr_stream_state],
                    outputs=[asr_stream_state, asr_rec_status],
                    concurrency_limit=1,
                )
                mic_stream.stop_recording(
                    fn=stream_finalize_and_transcribe,
                    inputs=[asr_stream_state, lang_dd, beam, use_vad, backend_dd],
                    outputs=[asr_preview_audio, out_text, asr_meta],
                    concurrency_limit=1,
                )
                ### Event Wire Transcribe
                # Run ASR
                go_asr.click(transcribe, [audio_in, lang_dd, beam, use_vad, backend_dd], [out_text, asr_meta], concurrency_limit=2)
                
                # Stop ASR
                stop_asr.click(request_asr_stop,inputs=[backend_dd],outputs=[asr_meta],concurrency_limit=1)

            # -------- TTS TAB --------
            with gr.Tab("Text → Speech"):
                # First row: input text + basic controls (no Advanced here)
                with gr.Row():
                    in_text = gr.Textbox(label="Text", placeholder="Type something…", lines=25)
                    with gr.Column():
                        tts_lang_dd = gr.Dropdown(tts_lang_choices, value=tts_lang_choices[0], label="TTS Language")
                        init_choices, init_default = make_speaker_choices(TTS_LANG_DISPLAY[0])
                        tts_speaker_dd = gr.Dropdown(init_choices, value=init_default, label="Voice (discovered)", allow_custom_value=False)
                        tts_speed = gr.Slider(0.5, 2.0, value=DEFAULT_TTS_SPEED, step=0.05, label="Speed")
                        # Optional cloning: upload a reference WAV to enforce accent/style
                        ref_wav = gr.Audio(sources=["upload"], type="filepath", label="(Optional) Reference voice WAV")
                        with gr.Row():
                            go_tts = gr.Button("Synthesize", variant="primary")
                            stop_tts = gr.Button("🛑", variant="secondary", scale=0, min_width=60)

                tts_warn_md = gr.Markdown("")
                # Second row: left = outputs; right = Advanced (here)
                with gr.Row():
                    with gr.Column():
                        tts_audio = gr.Audio(label="TTS Output", autoplay=True, type="filepath")
                    with gr.Column():
                        with gr.Accordion("Advanced", open=False):
                            # ---- Optional Ollama pre-chunker (dev/advanced) ----
                            use_ollama = gr.Checkbox(value=OLLAMA_ENABLE_DEFAULT, label="Use Ollama pre-chunker (optional)")
                            ollama_model = gr.Dropdown(
                            choices=[OLLAMA_MODEL_DEFAULT],  # will be replaced on refresh/load if Ollama responds
                            value=OLLAMA_MODEL_DEFAULT,
                            label="Ollama model",
                            allow_custom_value=True,         # power users can type names not yet listed
                            info="Select a locally installed model (click Refresh). You can also type a custom tag."
                        )

                            with gr.Row():
                                refresh_models_btn = gr.Button("Refresh models")
                                test_btn = gr.Button("Test Ollama")
                                save_default_btn = gr.Button("Set as default model")
                            ollama_status = gr.Markdown("")

                            # Refresh list (preserve current selection when possible)
                            refresh_models_btn.click(
                                _refresh_ollama_models,
                                [ollama_model],            # pass current value
                                [ollama_model, ollama_status],
                                concurrency_limit=1,
                            )
                            # Quick connectivity/model test
                            test_btn.click(
                                test_ollama_connection,
                                [ollama_model],
                                [ollama_status],
                                concurrency_limit=1,
                            )
                            save_default_btn.click(
                                _save_default_model,
                                [ollama_model],
                                [ollama_model, ollama_status],
                                concurrency_limit=1,
                            )
                # Update speakers when language changes
                tts_lang_dd.change(refresh_speakers, [tts_lang_dd], [tts_speaker_dd])

                # Wire Actions:
                go_tts.click(
                    synthesize_tts,
                    [in_text, tts_lang_dd, tts_speaker_dd, tts_speed, ref_wav, use_ollama, ollama_model],
                    [tts_audio, tts_warn_md],
                    concurrency_limit=2,
                )
                stop_tts.click(
                    request_tts_stop,
                    inputs=[],
                    outputs=[tts_warn_md], # Show an immediate status when STOP is clickedy
                    concurrency_limit=1,
                )
            # -------- CONFIG TAB --------
            with gr.Tab("⚙️ Config"):
                build_config_tab()
                # -------- LOG TAB --------
                from .log_panel import build_log_tab
                build_log_tab(demo)
            # -------- DEV-ONLY: Debug (hidden unless DEBUG_TOOLS=1) --------
            if DEBUG_TOOLS:
                build_debug_tab()
            
    return demo
