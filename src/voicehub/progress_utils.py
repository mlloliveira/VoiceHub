# src/voicehub/progress_utils.py
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
    Log-friendly progress meter.
    Prints newline-delimited updates so the in-app log panel can display them.
    """
    line = f"{prefix} {progress_bar_md(done, total)}"
    print(line)
    return line
