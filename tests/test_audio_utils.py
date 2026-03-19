from __future__ import annotations

import unittest

import numpy as np

from src.voicehub.audio_utils import concat_audio_segments


class AudioConcatTests(unittest.TestCase):
    def test_concat_same_sample_rate_without_crossfade_is_exact(self):
        a = np.array([0.0, 0.25, 0.5], dtype=np.float32)
        b = np.array([0.75, 1.0], dtype=np.float32)

        out, sr = concat_audio_segments([a, b], [24000, 24000], crossfade_ms=0.0)

        np.testing.assert_allclose(out, np.concatenate([a, b]))
        self.assertEqual(sr, 24000)

    def test_concat_resamples_mismatched_sample_rates(self):
        a = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
        b = np.linspace(1.0, -1.0, 4, dtype=np.float32)

        out, sr = concat_audio_segments([a, b], [24000, 12000], crossfade_ms=0.0)

        self.assertEqual(sr, 24000)
        self.assertGreaterEqual(len(out), len(a) + len(b))
        self.assertEqual(out.dtype, np.float32)

    def test_concat_crossfade_softens_hard_boundary_click(self):
        left = np.zeros(400, dtype=np.float32)
        right = np.ones(400, dtype=np.float32)

        hard, _ = concat_audio_segments([left, right], [24000, 24000], crossfade_ms=0.0)
        soft, _ = concat_audio_segments([left, right], [24000, 24000], crossfade_ms=4.0)

        hard_jump = abs(float(hard[400] - hard[399]))
        soft_jump = abs(float(soft[400] - soft[399]))
        self.assertLess(soft_jump, hard_jump)


if __name__ == "__main__":
    unittest.main()
