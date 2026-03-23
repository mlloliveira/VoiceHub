from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import soundfile as sf

from .prefs import prefs_path


@dataclass
class CloneCacheRecord:
    audio_sha256: str
    source_audio_path_original: str
    asr_model: str
    detected_language: str
    transcript_path: str
    qwen_clone_model: str
    voicehub_version: str
    created_at: str


def _cache_root() -> Path:
    root = prefs_path().parent / "voice_clone_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read_audio_bytes(path: str) -> bytes:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    return bytes(memoryview(audio)) + str(sr).encode("utf-8")


def audio_sha256(path: str) -> str:
    h = hashlib.sha256()
    h.update(_read_audio_bytes(path))
    return h.hexdigest()


def transcript_path_for_hash(digest: str) -> Path:
    return _cache_root() / f"{digest}.txt"


def metadata_path_for_hash(digest: str) -> Path:
    return _cache_root() / f"{digest}.json"


def load_clone_cache(path: str) -> tuple[str, Optional[CloneCacheRecord]]:
    digest = audio_sha256(path)
    meta_path = metadata_path_for_hash(digest)
    if not meta_path.exists():
        return digest, None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return digest, CloneCacheRecord(**data)
    except Exception:
        return digest, None


def save_clone_cache(
    *,
    digest: str,
    source_audio_path_original: str,
    asr_model: str,
    detected_language: str,
    transcript: str,
    qwen_clone_model: str,
    voicehub_version: str,
) -> CloneCacheRecord:
    txt_path = transcript_path_for_hash(digest)
    txt_path.write_text((transcript or "").strip(), encoding="utf-8")
    record = CloneCacheRecord(
        audio_sha256=digest,
        source_audio_path_original=str(source_audio_path_original),
        asr_model=str(asr_model),
        detected_language=str(detected_language or ""),
        transcript_path=str(txt_path),
        qwen_clone_model=str(qwen_clone_model),
        voicehub_version=str(voicehub_version),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    metadata_path_for_hash(digest).write_text(json.dumps(record.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def read_cached_transcript(record: CloneCacheRecord) -> str:
    p = Path(record.transcript_path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8").strip()
