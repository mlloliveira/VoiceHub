# src/voicehub/debug_ui.py
import gradio as gr
from .user_settings import get_settings
from .ollama_config import OLLAMA_ENABLE_DEFAULT, OLLAMA_MODEL_DEFAULT
from .ollama_utils import test_ollama_connection
from .tts import inspect_full_pipeline

def build_debug_tab():
    with gr.Tab("🔧 Debug (dev)"):
        gr.Markdown(
            "Paste text and **see the full pipeline**: raw → (optional Ollama) refined → sentences → chunks.  \n"
            "This tab is hidden unless `DEBUG_TOOLS=1`."
        )

        with gr.Row():
            dbg_text = gr.Textbox(label="Raw text", lines=8, placeholder="Paste your TTS text here…")
            with gr.Column():
                s = get_settings()
                dbg_max = gr.Slider(120, 400, value=s.xtts_max_chars_per_chunk, step=5, label="Max chars per segment (for both stages)")
                dbg_use_ollama = gr.Checkbox(value=OLLAMA_ENABLE_DEFAULT, label="Use Ollama pre-chunker")
                dbg_model = gr.Textbox(value=OLLAMA_MODEL_DEFAULT, label="Ollama model", placeholder="llama3.1")
                test_btn = gr.Button("Test Ollama")
                ollama_status = gr.Markdown("")

        test_btn.click(
            test_ollama_connection,
            [dbg_model],
            [ollama_status],
            concurrency_limit=1,
        )

        run_btn = gr.Button("Run pipeline", variant="primary")

        # Outputs: refined text (copyable), sentence table, chunk table, summary
        with gr.Row():
            refined_box = gr.Textbox(label="Refined text (from Ollama, if enabled)", lines=8, show_copy_button=True)
        with gr.Row():
            dbg_sent = gr.Dataframe(
                headers=["# (sentence)", "chars", "text"],
                datatype=["number", "number", "str"],
                wrap=True,
                label="Sentence split"
            )
            dbg_chunks = gr.Dataframe(
                headers=["# (chunk)", "chars", "text"],
                datatype=["number", "number", "str"],
                wrap=True,
                label="Chunk assembly"
            )
        dbg_summary = gr.Markdown("")
        dbg_detect_md = gr.Markdown("")

        # Optional explicit copy button for refined text
        copy_refined = gr.Button("Copy refined text")
        copy_refined.click(
            fn=None,
            inputs=refined_box,
            outputs=[],
            js="(s) => { if (s) navigator.clipboard.writeText(s); }",
        )

        run_btn.click(
            inspect_full_pipeline,
            [dbg_text, dbg_max, dbg_use_ollama, dbg_model],
            [refined_box, dbg_sent, dbg_chunks, dbg_summary,dbg_detect_md],
            concurrency_limit=1,
        )