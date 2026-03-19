from __future__ import annotations

import unittest
from unittest.mock import patch

from src.voicehub.config import effective_max_chars
from src.voicehub.chunking import (
    choose_safe_refined_text,
    chunk_text,
)


class ChunkingTests(unittest.TestCase):
    def test_long_text_roundtrip_and_chunk_limits(self):
        paragraph = (
            "Invoice ID INV-2026-000123 must stay before ticket TCK-7788 and user_45. "
            "This paragraph is intentionally long so the chunker has to keep splitting it without losing order. "
            "We also keep repeated references to build a very long text for validation.\n"
        )
        text = paragraph * 180

        sentences, chunks, backend = chunk_text(text, max_len=210, lang_code="en")

        self.assertTrue(sentences)
        self.assertTrue(chunks)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(0 < len(chunk) <= 210 for chunk in chunks))
        self.assertIn(backend, {"spacy-sentencizer", "legacy-regex"})

    def test_extremely_long_token_is_split_without_loss(self):
        text = "prefix " + ("A" * 1200) + " suffix."
        _, chunks, _ = chunk_text(text, max_len=128, lang_code="en")
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(0 < len(chunk) <= 128 for chunk in chunks))

    def test_ollama_refine_is_rejected_when_ids_or_order_change(self):
        original = "First ID A-100 comes before B-200 and then C-300."
        changed = "First ID B-200 comes before A-100 and then C-300."

        refined, accepted = choose_safe_refined_text(original, changed)

        self.assertFalse(accepted)
        self.assertEqual(refined, original)

    def test_legacy_backup_is_still_available(self):
        text = (
            "Sentence one keeps its place. Sentence two keeps its place as well. "
            "Sentence three is here to make the fallback path run."
        )
        with patch("src.voicehub.chunking._get_sentencizer", return_value=None):
            _, chunks, backend = chunk_text(text, max_len=60, lang_code="en")

        self.assertEqual(backend, "legacy-regex")
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(0 < len(chunk) <= 60 for chunk in chunks))


class ConfigThresholdTests(unittest.TestCase):
    def test_manual_mode_uses_user_cap_directly(self):
        self.assertEqual(effective_max_chars("en", user_cap=185, dynamic=False), 185)

    def test_dynamic_mode_uses_user_cap_as_threshold(self):
        self.assertEqual(effective_max_chars("en", user_cap=200, dynamic=True), 200)
        self.assertEqual(effective_max_chars("en", user_cap=260, dynamic=True), 240)
        self.assertEqual(effective_max_chars("pt", user_cap=220, dynamic=True), 200)


if __name__ == "__main__":
    unittest.main()
