#!/usr/bin/env python3
"""
Tests to lock in current behavior for Conductor refinement wrapper and TrinitySonifier.
"""
from __future__ import annotations

import os
import numpy as np
import torch

from bio import pipeline
from bio.sonifier import TrinitySonifier


def test_refine_backbone_wrapper_kwargs_fallback(tmp_path):
    # Compose a deterministic sequence
    seq = "A" * 64
    mean, cert = pipeline.compose_sequence(seq, samples=1, variability=0.0, seed=7)
    bb, phi, psi, modes, cond = pipeline.conduct_backbone(mean, seq)

    # Call refine_backbone with an unsupported kwarg to trigger fallback
    refined_torsions, refined_bb = pipeline.refine_backbone(
        cond,
        bb,
        phi,
        psi,
        modes,
        seq,
        max_iters=1,
        step_deg=1.0,
        seed=123,
        weights={"ca": 1.0, "clash": 1.0},
        nonexistent_flag=True,  # forces TypeError in the first attempt
    )

    # Types and shapes preserved
    assert isinstance(refined_torsions, list)
    assert isinstance(refined_bb, np.ndarray) and refined_bb.shape == bb.shape
    assert len(refined_torsions) == len(phi) == len(psi)
    # Ensure it is convertible to (L,2) numeric array (phi, psi per residue)
    arr = np.asarray(refined_torsions, dtype=np.float32)
    assert arr.shape == (len(phi), 2)


def test_sonify_3ch_shape_and_center_weights_effect(tmp_path):
    # Prepare tiny inputs for speed
    W = 4
    comp = torch.randn(W, 48, dtype=torch.float32)
    kore = torch.randn(48, dtype=torch.float32)
    cert = torch.rand(W, dtype=torch.float32)
    diss = torch.rand(W, dtype=torch.float32)

    # Use a small sample rate and small stride to keep arrays small
    son_lo = TrinitySonifier(sample_rate=8000, bpm=120.0, stride_ticks=8, partials=8)

    cw_low = {"kore": 0.5, "cert": 0.2, "diss": 0.1}
    wav_low = son_lo.sonify_composition_3ch(comp, kore, cert, diss, cw_low)
    assert isinstance(wav_low, np.ndarray) and wav_low.ndim == 2 and wav_low.shape[1] == 3
    # Duration = W * window_n samples
    assert wav_low.shape[0] == son_lo.window_n * W

    # Increase center weights -> expect higher RMS in center channel on average
    cw_high = {"kore": 2.0, "cert": 2.0, "diss": 0.0}
    wav_high = son_lo.sonify_composition_3ch(comp, kore, cert, diss, cw_high)

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x.astype(np.float64))) + 1e-12))

    rms_low = rms(wav_low[:, 2])
    rms_high = rms(wav_high[:, 2])
    assert rms_high >= rms_low  # monotonic w.r.t center weights

    # Validate save_wav writes a file
    out_path = tmp_path / "test.wav"
    son_lo.save_wav(wav_low, str(out_path))
    assert os.path.exists(out_path)
    # file should be non-empty
    assert os.path.getsize(out_path) > 0
