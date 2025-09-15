#!/usr/bin/env python3
"""
Unit tests for bio/scale_and_meter.py to lock in current behavior prior to refactors.
"""
from typing import List, Tuple
import numpy as np
import pytest

from bio.scale_and_meter import snap_to_scale, enforce_meter, SCALE_TABLE


def test_snap_to_scale_helix_nearest_bin():
    # pick a raw point close to one of the helix bins
    phi_raw, psi_raw = -60.5, -45.2
    snapped = snap_to_scale('helix', phi_raw, psi_raw)
    allowed = SCALE_TABLE['helix']
    # snapped must be one of the allowed bins
    assert any(np.allclose(snapped, b) for b in allowed)
    # nearest by Euclidean distance
    dists = np.linalg.norm(allowed - np.array([phi_raw, psi_raw]), axis=1)
    best = tuple(allowed[int(np.argmin(dists))])
    assert tuple(snapped) == best


def test_snap_to_scale_sheet_nearest_bin():
    phi_raw, psi_raw = -140.0, 135.0
    snapped = snap_to_scale('sheet', phi_raw, psi_raw)
    allowed = SCALE_TABLE['sheet']
    assert any(np.allclose(snapped, b) for b in allowed)


def test_snap_to_scale_loop_nearest_bin():
    phi_raw, psi_raw = -80.0, 130.0
    snapped = snap_to_scale('loop', phi_raw, psi_raw)
    allowed = SCALE_TABLE['loop']
    assert any(np.allclose(snapped, b) for b in allowed)


def test_enforce_meter_helix_smoothing_behavior():
    # Construct a sequence of torsions with slight noise
    rng = np.random.default_rng(123)
    base = (-60.0, -45.0)
    torsions: List[Tuple[float, float]] = [
        (base[0] + float(rng.normal(0, 1.0)), base[1] + float(rng.normal(0, 1.0)))
        for _ in range(10)
    ]
    smoothed = enforce_meter('helix', torsions)
    # length preserved
    assert len(smoothed) == len(torsions)
    # endpoints preserved per current implementation
    assert smoothed[0] == torsions[0]
    assert smoothed[1] == torsions[1]
    assert smoothed[-2] == torsions[-2]
    assert smoothed[-1] == torsions[-1]
    # middle elements should differ from raw in general (smoothed)
    changed = sum(1 for i in range(2, len(torsions)-2) if smoothed[i] != torsions[i])
    assert changed >= 1


def test_enforce_meter_sheet_and_loop_passthrough():
    torsions = [(-60.0, -45.0) for _ in range(6)]
    assert enforce_meter('sheet', torsions) == torsions
    assert enforce_meter('loop', torsions) == torsions
