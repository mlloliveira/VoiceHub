from __future__ import annotations

import io
import unittest
from unittest.mock import patch

from src.voicehub import tts


class TTSLogOnlyTests(unittest.TestCase):
    def test_wrapper_yields_audio_only_and_prints_statuses(self):
        def fake_synth(*args, **kwargs):
            yield None, "⏳ Loading backend..."
            yield None, "⏳ Synthesizing..."
            yield "/tmp/out.wav", "✅ Done."

        captured = io.StringIO()
        with patch.object(tts, "synthesize_tts", side_effect=fake_synth), \
             patch("sys.stdout", new=captured):
            steps = list(tts.synthesize_tts_log_only("text", "English", "Default", 1.0, None, False, ""))

        self.assertEqual(steps, [None, None, "/tmp/out.wav"])
        out = captured.getvalue()
        self.assertIn("Loading backend", out)
        self.assertIn("Synthesizing", out)
        self.assertIn("Done.", out)


if __name__ == "__main__":
    unittest.main()
