#!/usr/bin/env python3
"""
Conductor sanity tests: confirm internal consistency and invariants without benchmarking.
These tests lock current behavior so we can safely refactor representations later.
"""
from __future__ import annotations

from typing import Tuple
import pytest
import numpy as np
import torch

from bio import pipeline
from bio.conductor import Conductor


def _finite_array(x: np.ndarray) -> bool:
    return np.isfinite(x).all()


def test_build_backbone_basic_invariants(sequence):
    # Compose a deterministic sequence and build a backbone
    mean, cert = pipeline.compose_sequence(sequence, samples=1, variability=0.0, seed=11)
    bb, phi, psi, modes, cond = pipeline.conduct_backbone(mean, sequence)

    # Shapes
    L = len(sequence)
    assert isinstance(bb, np.ndarray) and bb.shape == (L, 3, 3)
    assert isinstance(phi, np.ndarray) and phi.shape == (L,)
    assert isinstance(psi, np.ndarray) and psi.shape == (L,)
    assert isinstance(modes, list) and len(modes) == L

    # Finite values
    assert _finite_array(bb)
    assert _finite_array(phi)
    assert _finite_array(psi)

    # CA-CA distances should be positive and broadly clustered near 3.8 Å
    ca = bb[:, 1, :]
    if L > 1:
        d = np.linalg.norm(ca[1:] - ca[:-1], axis=1)
        assert (d > 0).all()
        # Loose sanity bounds around canonical ~3.8 Å
        assert float(d.min()) > 2.0 and float(d.max()) < 6.0


def test_quality_report_schema_and_ranges(sequence):
    mean, cert = pipeline.compose_sequence(sequence, samples=1, variability=0.0, seed=13)
    bb, phi, psi, modes, cond = pipeline.conduct_backbone(mean, sequence)
    rep = pipeline.quality_report(cond, bb, phi, psi, modes)

    # Basic schema
    assert isinstance(rep, dict)
    for k in ["lengths", "angles", "ca_ca", "phi", "psi", "modes", "summary"]:
        assert k in rep

    # Summary keys and non-negative counts
    for k in [
        "out_of_range_lengths",
        "out_of_range_angles",
        "num_clashes",
        "min_ca_ca",
        "max_ca_ca",
        "orthogonality_index",
    ]:
        assert k in rep["summary"]

    s = rep["summary"]
    assert s["out_of_range_lengths"] >= 0
    assert s["out_of_range_angles"] >= 0
    assert s["num_clashes"] >= 0
    # orthogonality_index in [0,1]
    oi = float(s["orthogonality_index"])
    assert 0.0 <= oi <= 1.0


def test_build_backbone_single_vector_expansion():
    # Provide a single 48D vector and custom length via sequence; expect expansion to L
    L = 48
    seq = "A" * L
    mean = torch.randn(1, 48)
    cond = Conductor()
    bb, phi, psi, modes = cond.build_backbone(mean, sequence=seq)

    assert isinstance(bb, np.ndarray) and bb.shape == (L, 3, 3)
    assert isinstance(phi, np.ndarray) and phi.shape == (L,)
    assert isinstance(psi, np.ndarray) and psi.shape == (L,)
    assert isinstance(modes, list) and len(modes) == L


@pytest.mark.timeout(60)
def test_refine_backbone_minimal_iteration_path(sequence):
    # Smoke the refinement wrapper with very small settings to avoid heavy compute
    mean, cert = pipeline.compose_sequence(sequence, samples=1, variability=0.0, seed=17)
    bb, phi, psi, modes, cond = pipeline.conduct_backbone(mean, sequence)

    refined_torsions, refined_bb = pipeline.refine_backbone(
        cond,
        bb,
        phi,
        psi,
        modes,
        sequence,
        max_iters=1,
        step_deg=1.0,
        seed=123,
        weights={"ca": 1.0, "clash": 1.0},
        # Keep runtime minimal and deterministic
        num_workers=0,
        spacing_max_attempts=0,
        final_attempts=0,
        eval_batch=32,
        wall_timeout_sec=2.0,
    )

    # Shapes preserved and numeric
    assert isinstance(refined_bb, np.ndarray) and refined_bb.shape == bb.shape
    arr = np.asarray(refined_torsions, dtype=np.float32)
    assert arr.shape == (len(phi), 2) and np.isfinite(arr).all()
