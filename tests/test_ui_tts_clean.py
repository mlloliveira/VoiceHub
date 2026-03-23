from __future__ import annotations

import unittest

from src.voicehub.ui import build_app


class UITTSCleanTests(unittest.TestCase):
    def test_tts_tab_no_status_markdown_component(self):
        demo = build_app()
        markdown_labels = []
        audio_count = 0
        for comp in demo.blocks.values():
            label = getattr(comp, "label", None)
            if comp.__class__.__name__ == "Markdown":
                markdown_labels.append(label)
            if comp.__class__.__name__ == "Audio" and label == "TTS Output":
                audio_count += 1
        self.assertEqual(audio_count, 1)
        self.assertNotIn("TTS status", markdown_labels)


if __name__ == "__main__":
    unittest.main()
