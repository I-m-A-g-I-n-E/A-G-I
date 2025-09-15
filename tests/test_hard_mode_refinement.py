import numpy as np
import torch

from bio.conductor import Conductor
from bio.pipeline import refine_backbone, quality_report


def test_hard_mode_refinement_restores_sanity():
    torch.manual_seed(123)
    np.random.seed(123)

    # Setup a simple sequence and synthetic composition vectors
    L = 48
    seq = 'A' * L
    comp = torch.randn(L, 48)

    # Build initial backbone and torsions
    c = Conductor()
    backbone, phi, psi, modes = c.build_backbone(comp, sequence=seq)

    # Pre-refinement sanity gates
    qc0 = c.quality_check(backbone, phi, psi, modes)
    s0 = qc0['summary']
    assert s0['out_of_range_lengths'] == 0
    assert s0['out_of_range_angles'] == 0

    # Induce dissonance: jitter torsions (small but noticeable)
    jitter_deg = 10.0
    phi_j = phi + np.random.randn(*phi.shape).astype(np.float32) * jitter_deg
    psi_j = psi + np.random.randn(*psi.shape).astype(np.float32) * jitter_deg
    torsions_j = np.stack([phi_j, psi_j], axis=1)

    # Rebuild a perturbed backbone as starting point
    backbone_j = c.build_backbone_from_torsions(torsions_j, seq)

    # Assert that the jitter likely worsened spacing (non-critical; stochastic)
    qc_j = c.quality_check(backbone_j, phi_j, psi_j, modes)
    # Proceed to refinement with assertive weights (hard mode)
    weights = {
        'clash': 12.0,
        'ca': 2.0,
        'smooth': 0.2,
        'snap': 0.5,
        'neighbor_ca': 4.0,
        'nonadj_ca': 3.0,
        'dihedral': 1.0,
    }

    refined_torsions, refined_backbone = refine_backbone(
        c,
        backbone_j,
        phi_j,
        psi_j,
        modes,
        seq,
        max_iters=400,
        step_deg=2.0,
        seed=42,
        weights=weights,
        neighbor_threshold=3.2,
        spacing_max_attempts=300,
        spacing_top_bins=4,
        spacing_continue_full=False,
        final_attempts=500,
        spacing_cross_mode=True,
        critical_override_iters=50,
        num_workers=0,
        eval_batch=256,
    )

    rphi = np.array([t[0] for t in refined_torsions], dtype=np.float32)
    rpsi = np.array([t[1] for t in refined_torsions], dtype=np.float32)
    qc_ref = quality_report(c, refined_backbone, rphi, rpsi, modes)
    s = qc_ref['summary']

    # Hard mode acceptance gates
    assert s['num_clashes'] == 0
    assert s['min_ca_ca'] > 3.2
    assert s['out_of_range_lengths'] == 0
    assert s['out_of_range_angles'] == 0
