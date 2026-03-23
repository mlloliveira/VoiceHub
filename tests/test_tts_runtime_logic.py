from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class TTSLogicTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        os.environ["VOICEHUB_PREFS_DIR"] = self.tmpdir.name
        for name in [
            "src.voicehub.prefs",
            "src.voicehub.user_settings",
            "src.voicehub.tts",
        ]:
            sys.modules.pop(name, None)
        import src.voicehub.prefs as prefs
        import src.voicehub.user_settings as user_settings
        import src.voicehub.tts as tts
        self.user_settings = importlib.reload(user_settings)
        self.tts = importlib.reload(tts)

    def tearDown(self):
        os.environ.pop("VOICEHUB_PREFS_DIR", None)
        self.tmpdir.cleanup()
        for name in [
            "src.voicehub.prefs",
            "src.voicehub.user_settings",
            "src.voicehub.tts",
        ]:
            sys.modules.pop(name, None)

    def test_make_speaker_choices_contains_both_families(self):
        with patch.object(self.tts, "get_available_xtts_speakers", return_value=["Ana Florence", "Tom"]):
            choices, default = self.tts.make_speaker_choices("English", "XTTS")
        self.assertEqual(default, "Default")
        self.assertIn("Ryan (Qwen)", choices)
        self.assertIn("Ana Florence (XTTS)", choices)

    def test_compose_qwen_instruction_combines_speed_and_style(self):
        prompt = self.tts._compose_qwen_instruction("Warm, intimate podcast voice.", 1.2, "en")
        self.assertIn("Warm, intimate podcast voice.", prompt)
        self.assertIn("faster", prompt.lower())


    def test_qwen_runtime_failure_falls_back_to_xtts(self):
        self.user_settings.update_settings(tts_default_family="Qwen", xtts_max_chars_per_chunk=120)
        with patch.object(self.tts, "preprocess_to_chunks", return_value=("hello", ["hello"], ["hello"])),              patch.object(self.tts, "qwen_available", return_value=(True, "ok")),              patch.object(self.tts, "synthesize_qwen_custom", side_effect=RuntimeError("missing sox")),              patch.object(self.tts, "get_available_xtts_speakers", return_value=["Ana Florence"]),              patch.object(self.tts, "_tts_chunk_xtts", return_value=(np.zeros(10, dtype=np.float32), 24000)):
            out_path, note = self.tts.synthesize_tts(
                text="hello world",
                language_display="English",
                speaker_display="Ryan (Qwen)",
                speed=1.0,
                ref_wav=None,
                qwen_style_prompt="",
                use_ollama=False,
                ollama_model="",
            )
        self.assertTrue(out_path)
        self.assertIn("XTTS", note)

    def test_request_tts_stop_sets_event(self):
        self.tts.reset_tts_stop_flag()
        self.assertFalse(self.tts._TTS_STOP_EVENT.is_set())
        self.tts.request_tts_stop()
        self.assertTrue(self.tts._TTS_STOP_EVENT.is_set())

    def test_qwen_unavailable_falls_back_to_xtts(self):
        self.user_settings.update_settings(tts_default_family="Qwen", xtts_max_chars_per_chunk=120)
        with patch.object(self.tts, "preprocess_to_chunks", return_value=("hello", ["hello"], ["hello"])), \
             patch.object(self.tts, "qwen_available", return_value=(False, "missing qwen")), \
             patch.object(self.tts, "get_available_xtts_speakers", return_value=["Ana Florence"]), \
             patch.object(self.tts, "_tts_chunk_xtts", return_value=(np.zeros(10, dtype=np.float32), 24000)):
            out_path, note = self.tts.synthesize_tts(
                text="hello world",
                language_display="English",
                speaker_display="Ryan (Qwen)",
                speed=1.0,
                ref_wav=None,
                qwen_style_prompt="",
                use_ollama=False,
                ollama_model="",
            )
        self.assertTrue(out_path)
        self.assertIn("XTTS", note)

    def test_qwen_clone_setup_failure_falls_back_to_xtts(self):
        self.user_settings.update_settings(tts_default_family="Qwen", xtts_max_chars_per_chunk=120)
        with patch.object(self.tts, "preprocess_to_chunks", return_value=("hello", ["hello"], ["hello"])), \
             patch.object(self.tts, "qwen_available", return_value=(True, "ok")), \
             patch.object(self.tts, "_build_qwen_clone_prompt_cached", side_effect=RuntimeError("clone create failed")), \
             patch.object(self.tts, "get_available_xtts_speakers", return_value=["Ana Florence"]), \
             patch.object(self.tts, "_tts_chunk_xtts", return_value=(np.zeros(10, dtype=np.float32), 24000)):
            out_path, note = self.tts.synthesize_tts(
                text="hello world",
                language_display="English",
                speaker_display="Ryan (Qwen)",
                speed=1.0,
                ref_wav="ref.wav",
                qwen_style_prompt="",
                use_ollama=False,
                ollama_model="",
            )
        self.assertTrue(out_path)
        self.assertIn("XTTS", note)


if __name__ == "__main__":
    unittest.main()


class CloneReferenceCapTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        os.environ["VOICEHUB_PREFS_DIR"] = self.tmpdir.name
        for name in ["src.voicehub.prefs", "src.voicehub.user_settings", "src.voicehub.tts"]:
            sys.modules.pop(name, None)
        import src.voicehub.user_settings as user_settings
        import src.voicehub.tts as tts
        self.user_settings = importlib.reload(user_settings)
        self.tts = importlib.reload(tts)

    def tearDown(self):
        os.environ.pop("VOICEHUB_PREFS_DIR", None)
        self.tmpdir.cleanup()
        for name in ["src.voicehub.prefs", "src.voicehub.user_settings", "src.voicehub.tts"]:
            sys.modules.pop(name, None)

    def test_prepare_capped_reference_audio_trims_to_family_cap(self):
        import soundfile as sf
        from pathlib import Path
        wav_path = Path(self.tmpdir.name) / "ref.wav"
        sr = 16000
        wav = np.zeros(sr * 120, dtype=np.float32)
        sf.write(wav_path, wav, sr)
        self.user_settings.update_settings(qwen_clone_ref_max_seconds=50.0, xtts_clone_ref_max_seconds=300.0)

        qwen_path, cleanup_path, trimmed, total_seconds, cap_seconds = self.tts._prepare_capped_reference_audio(str(wav_path), "Qwen")
        self.assertTrue(trimmed)
        self.assertTrue(cleanup_path)
        qwav, qsr = sf.read(qwen_path)
        self.assertAlmostEqual(len(qwav) / qsr, 50.0, places=1)
        self.assertAlmostEqual(total_seconds, 120.0, places=1)
        self.assertAlmostEqual(cap_seconds, 50.0, places=1)
        self.tts._cleanup_temp_paths([cleanup_path])

        xtts_path, cleanup_path2, trimmed2, _, cap_seconds2 = self.tts._prepare_capped_reference_audio(str(wav_path), "XTTS")
        self.assertFalse(trimmed2)
        self.assertEqual(xtts_path, str(wav_path))
        self.assertIsNone(cleanup_path2)
        self.assertAlmostEqual(cap_seconds2, 300.0, places=1)

    def test_synthesize_tts_retrims_reference_audio_on_qwen_to_xtts_fallback(self):
        import soundfile as sf
        from pathlib import Path
        wav_path = Path(self.tmpdir.name) / "ref2.wav"
        sr = 16000
        sf.write(wav_path, np.zeros(sr * 120, dtype=np.float32), sr)
        self.user_settings.update_settings(tts_default_family="Qwen", qwen_clone_ref_max_seconds=50.0, xtts_clone_ref_max_seconds=300.0, xtts_max_chars_per_chunk=120)

        captured = {}
        def fake_xtts_chunk(text_chunk, lang, speaker_name=None, ref_wav=None, speed=1.0):
            captured["ref_wav"] = ref_wav
            q, qs = sf.read(ref_wav)
            captured["seconds"] = len(q) / qs
            return np.zeros(10, dtype=np.float32), 24000

        with patch.object(self.tts, "preprocess_to_chunks", return_value=("hello", ["hello"], ["hello"])), \
             patch.object(self.tts, "qwen_available", return_value=(True, "ok")), \
             patch.object(self.tts, "_build_qwen_clone_prompt_cached", side_effect=RuntimeError("clone create failed")), \
             patch.object(self.tts, "get_available_xtts_speakers", return_value=["Ana Florence"]), \
             patch.object(self.tts, "_tts_chunk_xtts", side_effect=fake_xtts_chunk):
            out_path, note = self.tts.synthesize_tts(
                text="hello world",
                language_display="English",
                speaker_display="Ryan (Qwen)",
                speed=1.0,
                ref_wav=str(wav_path),
                qwen_style_prompt="",
                use_ollama=False,
                ollama_model="",
            )
        self.assertTrue(out_path)
        self.assertIn("XTTS", note)
        self.assertGreaterEqual(captured["seconds"], 119.0)
