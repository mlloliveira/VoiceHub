# src/voicehub/log_panel.py
import gradio as gr
import io, sys, threading
from datetime import datetime

# --- Tee logger ---
_log_lock = threading.Lock()
_log_buffer = io.StringIO()
_tee_installed = False
_orig_stdout, _orig_stderr = None, None

class TeeLogger(io.TextIOBase):
    def __init__(self, real_stream):
        self.real_stream = real_stream

    def write(self, s):
        with _log_lock:
            try:
                self.real_stream.write(s)
                self.real_stream.flush()
            except Exception:
                pass
            _log_buffer.write(s)
        return len(s)

    def flush(self):
        with _log_lock:
            try:
                self.real_stream.flush()
            except Exception:
                pass

def _install_tee():
    global _tee_installed, _orig_stdout, _orig_stderr
    if _tee_installed:
        return
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = TeeLogger(_orig_stdout)
    sys.stderr = TeeLogger(_orig_stderr)
    _tee_installed = True

def get_logs(max_lines=400):
    with _log_lock:
        text = _log_buffer.getvalue()
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])

def clear_logs():
    with _log_lock:
        _log_buffer.seek(0)
        _log_buffer.truncate(0)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{stamp}] (log cleared)\n"
    print(msg, end="")  # goes to both console and buffer
    return get_logs()

# --- Gradio UI ---
def build_log_tab(demo: gr.Blocks):
    _install_tee()
    with gr.Tab("🖥️ Log Panel"):
        gr.Markdown("### 💻 In-app Log Panel")
        log_box = gr.Textbox(label="Logs", lines=20, show_copy_button=True)
        clear_btn = gr.Button("Clear Logs")
        clear_btn.click(clear_logs, outputs=log_box, show_progress="hidden")

        # Auto-refresh every 0.5s
        demo.load(fn=get_logs, inputs=None, outputs=log_box, every=0.5)

    return {"log_box": log_box}
