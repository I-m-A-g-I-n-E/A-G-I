#!/usr/bin/env python3
"""
Phase 2 Demonstration: The Harmonic Propagator.

Takes a raw amino acid sequence and uses the compositional engine to generate
a series of 48-dimensional Composition Vectors.
"""
from __future__ import annotations

import argparse
import os
import torch
import numpy as np

from bio import pipeline


def compose(
    sequence: str,
    *,
    samples: int = 1,
    variability: float = 0.0,
    seed: int | None = None,
    window_jitter: bool = False,
):
    """Runs the full ensemble composition process for a given amino acid sequence."""
    print(f"🎼 Beginning ensemble composition ({samples} samples)...")
    print(f"   - Input Sequence Length: {len(sequence)}")
    print(f"   - Variability: {variability}")
    print(f"   - Seed: {seed}")
    print(f"   - Window Jitter: {window_jitter}")

    mean_composition, harmonic_certainty = pipeline.compose_sequence(
        sequence,
        samples=samples,
        variability=variability,
        seed=seed,
        window_jitter=window_jitter,
    )

    # Report
    print("\n" + "=" * 50)
    print("  Ensemble Composition Complete")
    print(f"  - Mean Composition Shape: {mean_composition.shape}")
    print(f"  - Harmonic Certainty Shape: {harmonic_certainty.shape}")
    print("=" * 50)
    print("\nThis provides a rich, statistical view of the protein's harmonic landscape.")
    print("We now have not only the structure's signature but also a confidence")
    print("metric derived entirely from first principles.")
    return mean_composition, harmonic_certainty


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compose amino acid sequences into 48D Composition Vectors.")
    parser.add_argument("--sequence", type=str, default="MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG", help="Amino acid sequence (>=48 residues)")
    parser.add_argument("--samples", type=int, default=1, help="Number of stochastic samples to generate (ensemble size)")
    parser.add_argument("--variability", type=float, default=0.0, help="Variability in [0,1]: stochastic distillation span, small noise, optional jitter")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (optional)")
    parser.add_argument("--window-jitter", action="store_true", help="Enable random window start offset within stride")
    parser.add_argument("--save-prefix", type=str, default=None, help="Prefix path to save outputs (without extension). If provided, saves mean and certainty.")
    parser.add_argument(
        "--save-format",
        type=str,
        default="pt",
        choices=["pt", "npy"],
        help="Output format when saving tensors: 'pt' for torch.save, 'npy' for NumPy",
    )
    args = parser.parse_args()

    seq = args.sequence.strip().upper()
    if len(seq) < 48:
        print("❌ Error: Sequence must be at least 48 amino acids long.")
    else:
        mean_comp, certainty = compose(
            seq,
            samples=max(1, int(args.samples)),
            variability=max(0.0, min(1.0, args.variability)),
            seed=args.seed,
            window_jitter=bool(args.window_jitter),
        )
        # Optional saving
        if args.save_prefix:
            prefix = os.path.expanduser(args.save_prefix)
            os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
            if args.save_format == "pt":
                torch.save(mean_comp, f"{prefix}_mean.pt")
                torch.save(certainty, f"{prefix}_certainty.pt")
                print(f"💾 Saved: {prefix}_mean.pt, {prefix}_certainty.pt")
            else:
                np.save(f"{prefix}_mean.npy", mean_comp.cpu().numpy())
                np.save(f"{prefix}_certainty.npy", certainty.cpu().numpy())
                print(f"💾 Saved: {prefix}_mean.npy, {prefix}_certainty.npy")
