#!/usr/bin/env python3
"""
Trinity Sonifier — map 48D composition vectors to 2+1 (L/C/R) audio stems.

Design:
- Left: keven (even indices) as cosine partials
- Right: kodd (odd indices) as sine partials
- Center: kore/certainty drone (subset of harmonics) scaled by certainty per window
- 48 ticks per bar; stride determines step between windows externally

Dependencies: numpy, torch, scipy (for wav write)
"""
from __future__ import annotations

import numpy as np
import torch
try:
    from scipy.io import wavfile
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
from typing import Iterable, Optional, Tuple


def _normalize_wave(w: np.ndarray, peak_dbfs: float = -1.0) -> np.ndarray:
    if w.size == 0:
        return w
    peak = np.max(np.abs(w))
    if peak <= 1e-12:
        return w
    target = 10 ** (peak_dbfs / 20.0)
    scale = target / peak
    return (w * min(1.0, scale)).astype(np.float32)


class TrinitySonifier:
    def __init__(
        self,
        sample_rate: int = 48000,
        bpm: float = 96.0,
        tonic_hz: float = 220.0,  # A3
        tick_per_bar: int = 48,
        stride_ticks: int = 16,
        partials: int = 24,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.bpm = float(bpm)
        self.tonic_hz = float(tonic_hz)
        self.tick_per_bar = int(tick_per_bar)
        self.stride_ticks = int(stride_ticks)
        self.partials = int(partials)
        # Duration per tick: 4 beats per bar assumed (common time)
        sec_per_beat = 60.0 / self.bpm
        beats_per_bar = 4.0
        self.bar_duration = sec_per_beat * beats_per_bar
        self.tick_duration = self.bar_duration / self.tick_per_bar
        # One window duration = stride_ticks ticks
        self.window_duration = self.tick_duration * self.stride_ticks
        self.window_n = int(round(self.window_duration * self.sample_rate))
        # Precompute partial frequencies
        self.part_freqs = self.tonic_hz * np.arange(1, self.partials + 1, dtype=np.float64)
        # Center (drone) chosen harmonics (1,2,3,5)
        self.center_part_idx = np.array([1, 2, 3, 5], dtype=np.int64) - 1
        self.center_freqs = self.part_freqs[self.center_part_idx]

    def _synth_partials(self, amps: np.ndarray, freqs: np.ndarray, shape: str = "cos") -> np.ndarray:
        t = np.linspace(0.0, self.window_duration, self.window_n, endpoint=False, dtype=np.float64)
        phase = 2.0 * np.pi * freqs.reshape(-1, 1) * t.reshape(1, -1)
        if shape == "cos":
            waves = np.cos(phase)
        else:
            waves = np.sin(phase)
        # broadcast amps to [partials, 1]
        a = amps.reshape(-1, 1)
        y = (a * waves).sum(axis=0)
        return y.astype(np.float32)

    def sonify_composition(
        self,
        composition_vectors: torch.Tensor,
        certainty: Optional[torch.Tensor] = None,
        qc_clash_flags: Optional[Iterable[bool]] = None,
        center_gain: float = 0.7,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        composition_vectors: [W, 48] torch tensor
        certainty: [W] torch tensor in [0,1] (optional)
        qc_clash_flags: iterable of bool per window (optional) — if True, add a quiet click
        Returns (left, center, right) mono stems as float32 in [-1,1] (not normalized)
        """
        comp = composition_vectors.detach().cpu()
        W, D = comp.shape
        assert D == 48, "Expected 48D composition vectors"
        if certainty is None:
            certainty = torch.ones((W,), dtype=torch.float32)
        cert = certainty.detach().cpu().clamp(0, 1)
        if qc_clash_flags is None:
            qc_clash_flags = [False] * W

        left_list = []
        center_list = []
        right_list = []

        for i in range(W):
            v = comp[i]
            keven = v[0::2].numpy()
            kodd = v[1::2].numpy()
            # Pad/crop to partials
            if keven.shape[0] < self.partials:
                ke_p = np.pad(keven, (0, self.partials - keven.shape[0]))
                ko_p = np.pad(kodd, (0, self.partials - kodd.shape[0]))
            else:
                ke_p = keven[: self.partials]
                ko_p = kodd[: self.partials]

            # Gentle amplitude companding to avoid harshness
            ke_p = np.tanh(ke_p)
            ko_p = np.tanh(ko_p)

            left_wave = self._synth_partials(ke_p, self.part_freqs, "cos")
            right_wave = self._synth_partials(ko_p, self.part_freqs, "sin")

            # Center: certainty-controlled drone
            c_amp = float(cert[i].item()) * center_gain
            c_amps = np.ones((self.center_freqs.shape[0],), dtype=np.float32) * c_amp
            center_wave = self._synth_partials(c_amps, self.center_freqs, "cos")

            # Optional: add quiet click if clash flagged
            if qc_clash_flags and qc_clash_flags[i]:
                click_len = min(256, center_wave.shape[0])
                window = np.hanning(click_len).astype(np.float32)
                center_wave[:click_len] += 0.05 * window  # subtle

            left_list.append(left_wave)
            center_list.append(center_wave)
            right_list.append(right_wave)

        L = np.concatenate(left_list) if left_list else np.zeros((0,), dtype=np.float32)
        C = np.concatenate(center_list) if center_list else np.zeros((0,), dtype=np.float32)
        R = np.concatenate(right_list) if right_list else np.zeros((0,), dtype=np.float32)
        return L, C, R

    def save_wav(self, wave: np.ndarray, path: str, peak_dbfs: float = -1.0) -> None:
        w = _normalize_wave(wave, peak_dbfs=peak_dbfs)
        if _HAS_SCIPY:
            wavfile.write(path, self.sample_rate, w)
        else:
            # Fallback: write 16-bit PCM using builtin wave module
            import wave, struct
            # Convert float32 [-1,1] to int16
            w16 = np.clip((w * 32767.0), -32768, 32767).astype(np.int16)
            with wave.open(path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(w16.tobytes())
