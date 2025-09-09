# src/voicehub/ollama_utils.py
# All Ollama interactions (client import, HTTP fallback, health checks, refinement) live here.

import json, re, urllib.request
from typing import Optional, Tuple, List, Dict

from .ollama_config import (OLLAMA_HOST, OLLAMA_TIMEOUT, OLLAMA_PRECHUNK_PROMPT, OLLAMA_MAX_SEG_CHARS, OLLAMA_TRANSLATE_PROMPT
)

# Try optional Python client; if missing, we fall back to raw HTTP
OLLAMA_PY_AVAILABLE = False
try:
    import ollama  # optional dependency
    OLLAMA_PY_AVAILABLE = True
except Exception:
    print('No Ollama Python Library detected. Falling back to raw HTTP for Ollama')
    OLLAMA_PY_AVAILABLE = False

def _http_request(method: str, path: str, payload: dict | None = None, timeout: int = OLLAMA_TIMEOUT):
    url = f"{OLLAMA_HOST.rstrip('/')}{path}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method.upper(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        return None if not body else json.loads(body.decode("utf-8"))

def has_ollama() -> Tuple[bool, str]:
    """Check if Ollama is reachable and return (ok: bool, details: str)."""
    try:
        if OLLAMA_PY_AVAILABLE:
            client = ollama.Client(host=OLLAMA_HOST)
            _ = client.list()
            return True, "Ollama is running."
        _ = _http_request("GET", "/api/tags", None, timeout=OLLAMA_TIMEOUT)
        return True, "Ollama is running."
    except Exception as e:
        return False, f"Ollama not reachable: {e}"

def _generate(prompt: str, model: str, options: dict | None = None) -> str:
    options = options or {}
    if OLLAMA_PY_AVAILABLE:
        client = ollama.Client(host=OLLAMA_HOST)
        out = client.generate(model=model, prompt=prompt, stream=False, options=options)
        return (out.get("response") or "").strip()
    out = _http_request(
        "POST", "/api/generate",
        {"model": model, "prompt": prompt, "stream": False, "options": options},
        timeout=OLLAMA_TIMEOUT
    )
    return (out.get("response") or "").strip() if out else ""

def refine_text_with_ollama(raw_text: str, max_chars: int | None, model: str, options: dict | None = None) -> str:
    """
    Pre-process text with Ollama to produce better, punctuation-aware segments (one per line).
    We DO NOT replace your splitter/greedy chunker; this step just improves the input text.
    """
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return ""
    mc = int(max_chars) if max_chars else OLLAMA_MAX_SEG_CHARS
    prompt = OLLAMA_PRECHUNK_PROMPT.format(max_chars=mc) + raw_text + "\n\nOutput text:\n"
    resp = _generate(prompt, model=model, options=options)
    if not resp:
        return ""
    # Normalize to plain lines; model may echo labels or extra spaces
    lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
    # Strip simple numbering like "1. " or "2) "
    cleaned = [re.sub(r"^\s*\d+\s*[\.\)\-:]\s*", "", ln) for ln in lines]
    return "\n".join(cleaned).strip()

def translate_text_with_ollama(raw_text: str, target_lang_name: str, model: str, options: Optional[Dict] = None) -> str:
    """
    Translate `raw_text` to a human-readable language name like 'English', 'Português', 'Deutsch', etc.
    Uses the generic generation API with a translation-specific prompt.
    """
    txt = (raw_text or "").strip()
    if not txt:
        return ""
    prompt = OLLAMA_TRANSLATE_PROMPT.format(target_lang_name=target_lang_name) + txt + "\n\nOUTPUT:\n"
    resp = _generate(prompt, model=model, options=options or {})
    return (resp or "").strip()


def test_ollama_connection(model: str) -> str:
    """For the UI 'Test Ollama' button."""
    ok, msg = has_ollama()
    if not ok:
        return f"❌ {msg}"
    try:
        _ = _generate("Say OK.", model=model, options={"num_predict": 8})
        return f"✅ Ollama reachable and model **{model}** responded."
    except Exception as e:
        return f"⚠️ Ollama is up, but model **{model}** failed: {e}"
    

def list_ollama_models() -> Tuple[bool, List[str], str]:
    """
    Query local Ollama for installed models (like `ollama ls`).
    Returns:
      (ok, names, msg)
      - ok=False if Ollama not reachable
      - names: list of model names (e.g., ["llama3:8b", "qwen2.5:7b"])
      - msg: human-friendly status text
    """
    ok, msg = has_ollama()
    if not ok:
        return False, [], msg
    try:
        names: List[str] = []
        if OLLAMA_PY_AVAILABLE:
            client = ollama.Client(host=OLLAMA_HOST)
            data = client.list()  # typically {"models":[{name,model,size,...}, ...]}
        else:
            data = _http_request("GET", "/api/tags", None, timeout=OLLAMA_TIMEOUT)

        models = []
        if isinstance(data, dict):
            models = data.get("models") or data.get("data") or []

        for m in models:
            if isinstance(m, dict):
                nm = m.get("name") or m.get("model") or m.get("digest")
                if nm:
                    names.append(nm)

        names = sorted({n.strip() for n in names if isinstance(n, str) and n.strip()})
        if not names:
            return True, [], "No local models found (try `ollama pull ...`)."
        return True, names, f"Found {len(names)} model(s)."
    except Exception as e:
        return False, [], f"Failed to list models: {e}"
