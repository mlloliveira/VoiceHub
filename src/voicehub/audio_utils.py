from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


def _as_mono_float32(wav) -> np.ndarray:
    arr = np.asarray(wav, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1).astype(np.float32, copy=False)
    if arr.ndim == 1:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 2:
        if arr.shape[0] == 1:
            return arr[0].astype(np.float32, copy=False)
        if arr.shape[1] == 1:
            return arr[:, 0].astype(np.float32, copy=False)
        return arr.mean(axis=1, dtype=np.float32)
    return arr.reshape(-1).astype(np.float32, copy=False)


def _resample_linear(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    wav = _as_mono_float32(wav)
    src_sr = int(src_sr)
    dst_sr = int(dst_sr)
    if src_sr <= 0 or dst_sr <= 0 or wav.size == 0 or src_sr == dst_sr:
        return wav.astype(np.float32, copy=False)

    src_x = np.arange(wav.size, dtype=np.float64)
    dst_len = max(1, int(round(wav.size * (dst_sr / float(src_sr)))))
    dst_x = np.linspace(0.0, max(wav.size - 1, 0), num=dst_len, dtype=np.float64)
    out = np.interp(dst_x, src_x, wav.astype(np.float64, copy=False))
    return out.astype(np.float32, copy=False)


def _crossfade_join(left: np.ndarray, right: np.ndarray, crossfade_samples: int) -> np.ndarray:
    left = _as_mono_float32(left)
    right = _as_mono_float32(right)
    n = int(crossfade_samples)
    if n <= 0 or left.size == 0 or right.size == 0:
        return np.concatenate([left, right]).astype(np.float32, copy=False)

    n = min(n, left.size, right.size)
    if n <= 0:
        return np.concatenate([left, right]).astype(np.float32, copy=False)

    fade_out = np.linspace(1.0, 0.0, num=n, endpoint=False, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, num=n, endpoint=False, dtype=np.float32)
    overlap = left[-n:] * fade_out + right[:n] * fade_in
    merged = np.concatenate([left[:-n], overlap, right[n:]])
    return merged.astype(np.float32, copy=False)


def concat_audio_segments(
    wavs: Sequence[np.ndarray],
    sample_rates: Sequence[int],
    *,
    crossfade_ms: float = 4.0,
) -> Tuple[np.ndarray, int]:
    """
    Robustly join synthesized mono waveforms.

    - normalizes each segment to mono float32
    - resamples later segments to the first segment's sample rate if needed
    - applies a very small crossfade to reduce boundary clicks between chunks
    """
    if not wavs:
        raise ValueError("No audio segments to concatenate.")
    if len(wavs) != len(sample_rates):
        raise ValueError("wavs and sample_rates must have the same length.")

    target_sr = int(sample_rates[0]) if sample_rates else 24000
    if target_sr <= 0:
        target_sr = 24000

    crossfade_samples = max(0, int(round(target_sr * (float(crossfade_ms) / 1000.0))))

    merged = None
    for wav, sr in zip(wavs, sample_rates):
        seg = _resample_linear(_as_mono_float32(wav), int(sr), target_sr)
        if seg.size == 0:
            continue
        if merged is None:
            merged = seg.astype(np.float32, copy=False)
        else:
            merged = _crossfade_join(merged, seg, crossfade_samples)

    if merged is None:
        merged = np.zeros(1, dtype=np.float32)
    return merged.astype(np.float32, copy=False), target_sr
