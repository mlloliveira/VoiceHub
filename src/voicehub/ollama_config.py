# src/voicehub/ollama_config.py
# ---- Ollama (optional) configuration ----
# Keep all Ollama-related knobs isolated here so regular code stays clean.

import os
from .prefs import get_pref, set_pref

# Public fallback you chose:
_PUBLIC_FALLBACK = "gemma3:12b"

# Accept both env names for convenience
_env_model = os.getenv("OLLAMA_MODEL") or os.getenv("OLLAMA_MODEL_DEFAULT")

# Enable/disable by default; user can override in UI "Advanced" or via env in run.bat/run.sh
OLLAMA_ENABLE_DEFAULT = os.getenv("OLLAMA_ENABLE", "0") == "1"
#OLLAMA_MODEL_DEFAULT  = os.getenv("OLLAMA_MODEL", "gemma3:12b") 
OLLAMA_MODEL_DEFAULT  = get_pref("ollama_model_default", _env_model or _PUBLIC_FALLBACK) # Precedence: saved pref -> env -> fallback
OLLAMA_HOST           = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434") 
OLLAMA_TIMEOUT        = int(os.getenv("OLLAMA_TIMEOUT", "30"))  # seconds
OLLAMA_MAX_SEG_CHARS  = int(os.getenv("OLLAMA_MAX_SEG_CHARS", "200"))  # follows your rule 2

# Prompt template (your rules verbatim; {max_chars} is injected)
OLLAMA_PRECHUNK_PROMPT = """You will receive a piece of text that will be spoken by a TTS system. Follow these rules exactly:

1. Replace emojis with words.
   Example: 🤗 → "hugging face", 😂 → "laughing face".

2. Split the text into segments of {max_chars} characters or less.
   Count spaces and punctuation as characters.
   Each segment must end with proper punctuation.

3. Break long sentences into smaller ones.
   Use period (.), question mark (?) or exclamation mark (!) as the main break points.
   Do not use commas as a break. If a comma makes a sentence too long, change it to a period.
   Example:
   Input: "I was walking down the street, it was raining, I forgot my umbrella"
   Output: "I was walking down the street. It was raining. I forgot my umbrella."

4. If there is no good punctuation, create one.
   Example: Hey cutie, actually I was thinking... → Hey cutie! Actually, I was thinking...

5. Keep the text meaning the same. Do not change words. Only fix punctuation and break into chunks.

6. Output format:
   Each segment on a new line.
   Do not number the segments. Just plain text lines.
   Try to keep connecting words such as And and But.
   Do not separate senteces linked with That 

Full example:
Input text: Hey 🤗 I just wanted to let you know that I’m going to be late to the meeting today because traffic is crazy and I left home more than an hour ago and I’m still stuck on the highway and it doesn’t look like that it’s going to clear up soon, I’m so sorry about this but i will try my best
Output text:
Hey hugging face!
I just wanted to let you know that I’m going to be late to the meeting today.
Traffic is crazy.
And I left home more than an hour ago and I’m still stuck on the highway.
It doesn’t look like that it’s going to clear up soon.
I’m so sorry about this.
But i will try my best.

----
Now it's your turn! Do not change the language. Follow the instructions. Follow the text. Follow the examples.

Input text:
"""


OLLAMA_TRANSLATE_PROMPT = """You are a precise translation engine.

Rules:
1) Translate the INPUT text to {target_lang_name}.
2) Preserve meaning, numbers, names, URLs, and formatting markers.
3) Fix obvious punctuation only if needed for fluency; do not add commentary.
4) Do not explain. Output ONLY the translated text.

INPUT text:
"""


def set_ollama_default_model(model: str) -> str:
    m = (model or "").strip()
    if not m:
        return "⚠️ Empty model tag — nothing saved."
    set_pref("ollama_model_default", m)
    return f"✅ Saved default Ollama model: **{m}**"