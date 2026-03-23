"""Manual local smoke tests for VoiceHub 0.2.0.
Run inside your GPU-enabled env after installing the new dependencies.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-audio", type=str, default="", help="Optional reference audio for Qwen clone smoke test")
    args = parser.parse_args()

    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())

    from src.voicehub.asr import get_fw_model
    from src.voicehub.qwen_backend import get_qwen_custom_model, get_qwen_clone_model, qwen_available
    from src.voicehub.tts import get_xtts_model

    ok, msg = qwen_available()
    print("qwen:", ok, msg)

    print("Loading faster-whisper turbo...")
    _ = get_fw_model()
    print("✓ faster-whisper ready")

    print("Loading XTTS...")
    _ = get_xtts_model()
    print("✓ XTTS ready")

    print("Loading Qwen CustomVoice...")
    q_custom = get_qwen_custom_model()
    wavs, sr = q_custom.generate_custom_voice(
        text="Hello from VoiceHub point two.",
        language="English",
        speaker="Ryan",
        max_new_tokens=1024,
    )
    print("✓ Qwen CustomVoice ready", len(wavs[0]), sr)

    if args.ref_audio:
        print("Loading Qwen Base clone...")
        q_clone = get_qwen_clone_model()
        prompt = q_clone.create_voice_clone_prompt(ref_audio=args.ref_audio, ref_text="Hello from the cached reference audio.")
        wavs, sr = q_clone.generate_voice_clone(
            text="This is a clone smoke test.",
            language="English",
            voice_clone_prompt=prompt,
            max_new_tokens=1024,
        )
        print("✓ Qwen clone ready", len(wavs[0]), sr)


if __name__ == "__main__":
    main()
