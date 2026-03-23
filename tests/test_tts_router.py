from __future__ import annotations

import unittest

from src.voicehub.tts_router import (
    all_backend_voice_choices,
    chunk_slider_state_for_family,
    format_backend_status,
    qwen_default_voice_for_code,
    resolve_qwen_voice,
    resolve_tts_route,
    resolve_xtts_default_voice,
    speed_bucket_prompt,
)


class TTSRouterTests(unittest.TestCase):
    def test_qwen_supported_language_routes_to_custom_or_clone(self):
        custom = resolve_tts_route("Qwen", "en", has_reference=False)
        clone = resolve_tts_route("Qwen", "en", has_reference=True)
        self.assertEqual(custom.backend, "qwen_custom")
        self.assertEqual(clone.backend, "qwen_clone")
        self.assertEqual(custom.qwen_language, "English")

    def test_qwen_unsupported_language_falls_back_to_xtts(self):
        route = resolve_tts_route("Qwen", "ar", has_reference=False)
        self.assertEqual(route.backend, "xtts")
        self.assertTrue(route.fallback_used)
        self.assertIn("XTTS", format_backend_status("ar", route))

    def test_manual_language_status_uses_using_not_detected(self):
        route = resolve_tts_route("XTTS", "en", has_reference=False)
        status = format_backend_status("en", route, detected=False)
        self.assertIn("Using English", status)
        self.assertNotIn("Detected English", status)

    def test_qwen_default_voice_mapping(self):
        self.assertEqual(qwen_default_voice_for_code("en"), "Ryan")
        self.assertEqual(qwen_default_voice_for_code("zh-cn"), "Vivian")
        self.assertEqual(qwen_default_voice_for_code("fr"), "Ryan")

    def test_qwen_voice_resolution_prefers_selection_then_saved_default(self):
        self.assertEqual(resolve_qwen_voice("Ryan", "en", saved_default="Vivian"), "Ryan")
        self.assertEqual(resolve_qwen_voice("Default", "en", saved_default="Vivian"), "Vivian")
        self.assertEqual(resolve_qwen_voice("Default", "en", saved_default=""), "Ryan")

    def test_xtts_default_voice_prefers_priority_candidates(self):
        avail = ["Tom", "Ana Florence", "Aaron Dreschner"]
        self.assertEqual(resolve_xtts_default_voice(avail), "Ana Florence")

    def test_speed_bucket_prompts_exist_for_extremes(self):
        self.assertTrue(speed_bucket_prompt(0.75, "en"))
        self.assertEqual(speed_bucket_prompt(1.0, "en"), "")
        self.assertTrue(speed_bucket_prompt(1.25, "zh-cn"))
        self.assertIn("slower", speed_bucket_prompt(0.85, "en").lower())

    def test_chunk_slider_state_changes_by_family(self):
        qwen = chunk_slider_state_for_family("Qwen")
        xtts = chunk_slider_state_for_family("XTTS")
        self.assertEqual(qwen["value"], 512)
        self.assertEqual(qwen["maximum"], 2048)
        self.assertEqual(xtts["value"], 200)
        self.assertEqual(xtts["maximum"], 400)

    def test_chunk_slider_state_can_preserve_family_specific_values(self):
        qwen = chunk_slider_state_for_family("Qwen", qwen_value=1500)
        xtts = chunk_slider_state_for_family("XTTS", xtts_value=180)
        self.assertEqual(qwen["value"], 1500)
        self.assertEqual(xtts["value"], 180)

    def test_all_backend_voice_choices_include_both_families(self):
        choices = all_backend_voice_choices(["Ana Florence", "Tom"])
        self.assertIn("Ryan (Qwen)", choices)
        self.assertIn("Ana Florence (XTTS)", choices)
        self.assertEqual(choices[0], "Default")


if __name__ == "__main__":
    unittest.main()
