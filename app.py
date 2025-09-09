# app.py
import os, sys
from pathlib import Path

################# Mute little persistent bug
import warnings
import logging
try:
    from h11._util import LocalProtocolError as _H11LPE
except Exception:
    _H11LPE = None

class _DropH11ContentLengthMismatch(logging.Filter):
    PHRASES = (
        "Too much data for declared Content-Length",
        "Too little data for declared Content-Length",
    )
    def filter(self, record: logging.LogRecord) -> bool:
        msg = (record.getMessage() or "")
        if any(p in msg for p in self.PHRASES):
            return False
        exc = getattr(record, "exc_info", None)
        if exc and _H11LPE and isinstance(exc[1], _H11LPE):
            if any(p in str(exc[1]) for p in self.PHRASES):
                return False
        return True

# Attach to noisy channels (same as before)
for name in ("uvicorn.error", "uvicorn.access", "starlette"):
    logging.getLogger(name).addFilter(_DropH11ContentLengthMismatch())

# Mute the specific ctranslate2 deprecation warning about pkg_resources
warnings.filterwarnings(
    "ignore",
    message=r".*pkg_resources is deprecated.*",
    category=UserWarning,
    module=r"ctranslate2(\..*)?$",
)
#################

# Make "src" importable without installing the package yet
ROOT = Path(__file__).resolve().parent
SYS_PATH_SRC = str(ROOT / "src")
if SYS_PATH_SRC not in sys.path:
    sys.path.insert(0, SYS_PATH_SRC)

# Put preferences inside the app directory by default
os.environ.setdefault("VOICEHUB_PREFS_DIR", str(ROOT / "preferences"))

# Keep localhost clean (corporate proxies etc.)
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,::1")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost,::1")

from voicehub.ui import build_app

if __name__ == "__main__":
    demo = build_app()
    demo.queue()  # IMPORTANT for Audio.stream/start/stop events
    demo.launch(
        server_name=os.getenv("SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("SERVER_PORT", "7870")),
        max_threads=20,
        show_error=True,
        debug=True,
        share=False,
        max_file_size=os.getenv("MAX_FILE_SIZE", "300mb")  # <-- allow large uploads
    ) #You can use: python app.py --server-name 127.0.0.1 --server-port 7860