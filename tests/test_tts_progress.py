from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.voicehub import tts


class TTSProgressGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.out_path = str(Path(self.tmpdir.name) / "out.wav")
        self.settings = SimpleNamespace(
            tts_default_family="XTTS",
            xtts_max_minutes_default=5.0,
            qwen_max_chars_per_chunk=1024,
            xtts_max_chars_per_chunk=200,
            xtts_dynamic_per_lang_caps=True,
            qwen_voice_default="Ryan",
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_xtts_generator_reports_loading_and_synthesis_stages(self):
        route = SimpleNamespace(
            resolved_family="xtts",
            backend="xtts",
            language_code="en",
            qwen_language=None,
            fallback_used=False,
        )
        with patch.object(tts, "get_settings", return_value=self.settings), \
             patch.object(tts, "_resolve_language", return_value=("en", "")), \
             patch.object(tts, "resolve_tts_route", return_value=route), \
             patch.object(tts, "format_backend_status", return_value="✅ Using XTTS as backend model."), \
             patch.object(tts, "preprocess_to_chunks", return_value=("", [], ["Hello there.", "General Kenobi."])), \
             patch.object(tts, "get_available_xtts_speakers", return_value=["Ana Florence"]), \
             patch.object(tts, "resolve_xtts_voice", return_value="Ana Florence"), \
             patch.object(tts, "get_xtts_model", return_value=object()), \
             patch.object(tts, "_tts_chunk_xtts", return_value=(np.zeros(2400, dtype=np.float32), 24000)), \
             patch.object(tts, "concat_audio_segments", return_value=(np.zeros(4800, dtype=np.float32), 24000)), \
             patch.object(tts.sf, "write"), \
             patch.object(tts.tempfile, "NamedTemporaryFile", return_value=SimpleNamespace(name=self.out_path)):
            steps = list(tts.synthesize_tts("Some text.", "English", "Default", 1.0, None, False, ""))

        statuses = [status for _audio, status in steps if status]
        self.assertTrue(any("Resolving language and backend" in s for s in statuses))
        self.assertTrue(any("Loading XTTS backend" in s for s in statuses))
        self.assertTrue(any("Synthesizing 2 chunk(s) with XTTS" in s for s in statuses))
        self.assertEqual(steps[-1][0], self.out_path)
        self.assertIn("Using XTTS as backend model", steps[-1][1])

    def test_qwen_clone_generator_reports_clone_prep_stages(self):
        self.settings.tts_default_family = "Qwen"
        route = SimpleNamespace(
            resolved_family="qwen",
            backend="qwen_clone",
            language_code="en",
            qwen_language="English",
            fallback_used=False,
        )
        with patch.object(tts, "get_settings", return_value=self.settings), \
             patch.object(tts, "_resolve_language", return_value=("en", "")), \
             patch.object(tts, "resolve_tts_route", return_value=route), \
             patch.object(tts, "format_backend_status", return_value="✅ Using Qwen-TTS as backend model."), \
             patch.object(tts, "preprocess_to_chunks", return_value=("", [], ["First paragraph.", "Second paragraph."])), \
             patch.object(tts, "qwen_available", return_value=(True, "qwen ok")), \
             patch.object(tts, "get_qwen_clone_model", return_value=object()), \
             patch.object(tts, "_build_qwen_clone_prompt_cached", return_value=({"prompt": "ok"}, False)), \
             patch.object(tts, "synthesize_qwen_clone", return_value=([np.zeros(2400, dtype=np.float32)], 24000)), \
             patch.object(tts, "concat_audio_segments", return_value=(np.zeros(2400, dtype=np.float32), 24000)), \
             patch.object(tts.sf, "write"), \
             patch.object(tts.tempfile, "NamedTemporaryFile", return_value=SimpleNamespace(name=self.out_path)):
            steps = list(tts.synthesize_tts("Some text.", "Auto-detect", "Default", 1.0, "ref.wav", False, ""))

        statuses = [status for _audio, status in steps if status]
        self.assertTrue(any("Loading Qwen cloning backend" in s for s in statuses))
        self.assertTrue(any("Preparing Qwen voice clone" in s for s in statuses))
        self.assertTrue(any("Synthesizing 2 chunk(s) with Qwen clone" in s for s in statuses))
        self.assertEqual(steps[-1][0], self.out_path)
        self.assertIn("Using Qwen-TTS as backend model", steps[-1][1])


if __name__ == "__main__":
    unittest.main()
