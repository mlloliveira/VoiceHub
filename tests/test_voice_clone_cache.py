from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


class VoiceCloneCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        os.environ["VOICEHUB_PREFS_DIR"] = self.tmpdir.name

    def tearDown(self):
        self.tmpdir.cleanup()
        os.environ.pop("VOICEHUB_PREFS_DIR", None)

    def _make_wav(self, path: str, seed: int):
        rng = np.random.default_rng(seed)
        wav = rng.standard_normal(1600).astype(np.float32) * 0.01
        sf.write(path, wav, 16000)

    def test_same_audio_hash_reuses_metadata(self):
        from src.voicehub.voice_clone_cache import load_clone_cache, save_clone_cache, read_cached_transcript

        wav = Path(self.tmpdir.name) / "voice.wav"
        self._make_wav(str(wav), 1)

        digest, record = load_clone_cache(str(wav))
        self.assertIsNone(record)

        saved = save_clone_cache(
            digest=digest,
            source_audio_path_original=str(wav),
            asr_model="faster-whisper:turbo",
            detected_language="en",
            transcript="hello from cache",
            qwen_clone_model="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            voicehub_version="0.2.0",
        )
        self.assertEqual(read_cached_transcript(saved), "hello from cache")

        digest2, record2 = load_clone_cache(str(wav))
        self.assertEqual(digest2, digest)
        self.assertIsNotNone(record2)
        self.assertEqual(record2.audio_sha256, digest)

    def test_different_audio_produces_different_hash(self):
        from src.voicehub.voice_clone_cache import audio_sha256

        wav1 = Path(self.tmpdir.name) / "voice1.wav"
        wav2 = Path(self.tmpdir.name) / "voice2.wav"
        self._make_wav(str(wav1), 1)
        self._make_wav(str(wav2), 2)

        self.assertNotEqual(audio_sha256(str(wav1)), audio_sha256(str(wav2)))


if __name__ == "__main__":
    unittest.main()
