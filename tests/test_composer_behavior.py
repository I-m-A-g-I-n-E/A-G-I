#!/usr/bin/env python3
"""
Behavioral tests for bio/composer.HarmonicPropagator to lock in current semantics.
"""
import pytest
import torch

from bio.composer import HarmonicPropagator


def make_seq(n: int) -> str:
    return "ACDEFGHIKLMNPQRSTVWY" * (n // 20 + 1)


def test_requires_min_length_48():
    hp = HarmonicPropagator(n_layers=2, variability=0.0, seed=0, window_jitter=False)
    short = "A" * 47
    with pytest.raises(ValueError):
        _ = hp(short)


def test_rejects_invalid_amino_acids():
    hp = HarmonicPropagator(n_layers=1, variability=0.0, seed=0, window_jitter=False)
    seq = ("A" * 47) + "Z" + ("A" * 10)
    with pytest.raises(ValueError):
        _ = hp(seq)


def test_seed_determinism_no_jitter_variability_zero():
    seq = make_seq(96)[:96]
    hp1 = HarmonicPropagator(n_layers=2, variability=0.0, seed=123, window_jitter=False)
    hp2 = HarmonicPropagator(n_layers=2, variability=0.0, seed=123, window_jitter=False)
    with torch.no_grad():
        out1 = hp1(seq)
        out2 = hp2(seq)
    # Exactly equal tensors
    assert torch.allclose(out1, out2)
    # shape: (num_windows, 48)
    assert out1.ndim == 2 and out1.shape[1] == 48


def test_different_seed_affects_output_with_variability_and_jitter():
    seq = make_seq(128)[:128]
    # High variability and jitter to maximize stochastic differences
    hp1 = HarmonicPropagator(n_layers=3, variability=1.0, seed=111, window_jitter=True)
    hp2 = HarmonicPropagator(n_layers=3, variability=1.0, seed=222, window_jitter=True)
    with torch.no_grad():
        out1 = hp1(seq)
        out2 = hp2(seq)
    # Expect not exactly equal in general
    assert out1.shape[1] == 48 and out2.shape[1] == 48
    # Allow for equal length or off-by-one due to jitter alignment, but values should differ
    assert not torch.allclose(out1, out2)


def test_window_count_alignment_rules():
    # For length L and stride 16 (48/3), number of windows is floor((L - 48)/16)+1 when no jitter
    L = 112  # yields floor((112-48)/16)+1 = floor(64/16)+1 = 4+1 = 5
    seq = make_seq(L)[:L]
    hp = HarmonicPropagator(n_layers=1, variability=0.0, seed=0, window_jitter=False)
    with torch.no_grad():
        out = hp(seq)
    assert out.shape == (5, 48)
