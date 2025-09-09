from pathlib import Path
import json, os

# Legacy location we used before (for a one-time migration)
_LEGACY_DIR = Path.home() / ".voicehub"
_LEGACY_FILE = _LEGACY_DIR / "config.json"

def _prefs_dir() -> Path:
    """
    Precedence:
      1) VOICEHUB_PREFS_DIR  (set in app.py → <app>/preferences)
      2) VOICEHUB_HOME       (older env; if someone used it)
      3) CWD/preferences      (safe fallback)
    """
    if os.getenv("VOICEHUB_PREFS_DIR"):
        base = Path(os.getenv("VOICEHUB_PREFS_DIR")).expanduser().resolve()
    elif os.getenv("VOICEHUB_HOME"):
        base = Path(os.getenv("VOICEHUB_HOME")).expanduser().resolve()
        base = base if base.name == "preferences" else (base / "preferences")
    else:
        base = (Path.cwd() / "preferences").resolve()

    base.mkdir(parents=True, exist_ok=True)
    return base

def _prefs_path() -> Path:
    return _prefs_dir() / "config.json"

def _migrate_legacy_once():
    """If there's an old ~/.voicehub/config.json and no new file yet, move it."""
    new = _prefs_path()
    if new.exists():
        return
    if _LEGACY_FILE.exists():
        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            _LEGACY_FILE.replace(new)
        except Exception:
            # best-effort: copy then leave the original
            try:
                new.write_text(_LEGACY_FILE.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass  # ignore if unreadable

def get_prefs() -> dict:
    _migrate_legacy_once()
    p = _prefs_path()
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def save_prefs(prefs: dict) -> None:
    p = _prefs_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(prefs, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)

def get_pref(key: str, default=None):
    return get_prefs().get(key, default)

def set_pref(key: str, value) -> bool:
    prefs = get_prefs()
    if value is None:
        prefs.pop(key, None)
    else:
        prefs[key] = value
    save_prefs(prefs)
    return True
