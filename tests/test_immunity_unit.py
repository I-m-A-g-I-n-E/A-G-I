#!/usr/bin/env python3
"""
Deterministic unit tests for immunity.py core behaviors.
Run with: pytest -q test_immunity_unit.py
"""
from typing import List

import torch

from immunity import (
    get_divisors,
    MHC_SIZE,
    ImmuneSystem,
    ImmuneCell,
    CellType,
    Antigen,
)


def test_get_divisors_48():
    divs = sorted(get_divisors(48))
    assert divs == [2, 3, 4, 6, 8, 12, 16, 24, 48]


def test_complement_requires_full_coverage():
    # Setup
    immune = ImmuneSystem()
    b_cell = immune.create_immune_cell(CellType.B_CELL)

    # Create a foreign antigen with proper folding
    peptides = torch.arange(MHC_SIZE)
    ag = Antigen(
        epitope_id="ag_full",
        peptides=peptides,
        mhc_signature="",
        presented_by="dendritic",
        folding_score=1.0,
    )

    # Generate full coverage using factor=2 (two antibodies covering 24 sites each)
    b_cell.mount_adaptive_response(ag, factors=[2])
    abs_all = b_cell.antibodies[ag.epitope_id]
    assert len(abs_all) == 2

    # Full coverage should form a MAC
    mac_full = immune.complement.activate_cascade(abs_all)
    assert mac_full is not None
    assert mac_full.shape[0] == MHC_SIZE

    # Partial coverage (single antibody) should NOT form a MAC
    mac_partial = immune.complement.activate_cascade(abs_all[:1])
    assert mac_partial is None


def test_illegal_factors_are_ignored():
    immune = ImmuneSystem()
    b_cell = immune.create_immune_cell(CellType.B_CELL)

    peptides = torch.arange(MHC_SIZE)
    ag = Antigen(
        epitope_id="ag_illegal",
        peptides=peptides,
        mhc_signature="",
        presented_by="dendritic",
        folding_score=1.0,
    )

    # Include an illegal factor (5). Only legal factor 3 should be used.
    b_cell.mount_adaptive_response(ag, factors=[5, 3])
    antibodies = b_cell.antibodies[ag.epitope_id]
    # Expect exactly 3 antibodies (from factor=3), none generated for factor=5
    assert len(antibodies) == 3

    # Verify coverage is 48 only if combined with appropriate complementary splits.
    # With just factor=3 antibodies, union size should be 48 because three segments of length 16 cover all.
    all_sites = torch.cat([ab.binding_sites for ab in antibodies])
    unique_sites = torch.unique(all_sites)
    assert unique_sites.numel() == MHC_SIZE
