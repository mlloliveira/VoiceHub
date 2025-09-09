# src/voicehub/progress_utils.py
import sys

# --- progress bar helper for dev/user feedback ---
def progress_bar_md(done: int | float, total: int | float, width: int = 28) -> str:
    """
    Renders an ascii progress bar and counter. Accepts ints or floats.
    Example: [███████---------------] 7/24
    """
    try:
        total = max(1, int(round(total)))
    except Exception:
        total = 1
    try:
        done = max(0, min(int(round(done)), total))
    except Exception:
        done = 0
    filled = int(width * (done / total)) if total else 0
    bar = "█" * filled + "·" * (width - filled)
    return f"[{bar}] {done}/{total}"

def console_progress(done: int | float, total: int | float, prefix: str = "PROG", end: bool = False):
    """
    Prints a single-line progress meter (carriage-returned) to stdout.
    Updates in-app log panel in real time.
    """
    line = f"{prefix} {progress_bar_md(done, total)}"
    sys.stdout.write("\r" + line)
    sys.stdout.flush()
    if end:
        sys.stdout.write("\n")
        sys.stdout.flush()
