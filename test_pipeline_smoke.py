#!/usr/bin/env python3
import numpy as np
import torch

from bio import pipeline


def test_compose_sequence_shapes():
    seq = "A" * 64
    mean, cert = pipeline.compose_sequence(seq, samples=2, variability=0.1, seed=123, window_jitter=True)
    assert isinstance(mean, torch.Tensor) and mean.ndim == 2 and mean.shape[1] == 48
    assert isinstance(cert, torch.Tensor) and cert.ndim == 1 and cert.shape[0] == mean.shape[0]
    # Reasonable numeric ranges
    assert torch.isfinite(mean).all()
    assert torch.isfinite(cert).all()


def test_conduct_and_qc():
    seq = "A" * 64
    mean, cert = pipeline.compose_sequence(seq, samples=1, variability=0.0, seed=7)
    bb, phi, psi, modes, cond = pipeline.conduct_backbone(mean, seq)
    assert isinstance(bb, np.ndarray) and bb.ndim == 3 and bb.shape[1:] == (3, 3)
    assert isinstance(phi, np.ndarray) and isinstance(psi, np.ndarray)
    rep = pipeline.quality_report(cond, bb, phi, psi, modes)
    assert "summary" in rep and "num_clashes" in rep["summary"]


def test_sonify_3ch_shapes():
    seq = "A" * 64
    mean, cert = pipeline.compose_sequence(seq, samples=1, variability=0.0, seed=11)
    kore = pipeline.estimate_kore(mean, seq)
    diss = pipeline.dissonance_scalar_to_vec(0.0, int(cert.shape[0]))
    wav = pipeline.sonify_3ch(mean, kore, cert, diss, bpm=96.0, stride_ticks=16)
    assert isinstance(wav, np.ndarray) and wav.ndim == 2 and wav.shape[1] == 3
    # finite and float32
    assert wav.dtype == np.float32 and np.isfinite(wav).all()
