#!/usr/bin/env python3
"""
Phase 2 Demonstration: The Harmonic Propagator.

Takes a raw amino acid sequence and uses the compositional engine to generate
a series of 48-dimensional Composition Vectors.
"""
from __future__ import annotations

import torch

from bio.composer import HarmonicPropagator


def compose(sequence: str):
    """Runs the full composition process for a given amino acid sequence."""
    print("🎼 Beginning composition...")
    print(f"   - Input Sequence Length: {len(sequence)}")

    # 1. Instantiate the Composer
    composer = HarmonicPropagator(n_layers=4)

    # 2. Generate the Composition Vectors
    # No training needed. This is a deterministic, principled transformation.
    with torch.no_grad():
        composition_vectors = composer(sequence)

    # 3. Report the result (The Falsifiable Output)
    print("\n" + "=" * 50)
    print("  Composition Complete: Generated Composition Vectors")
    print(f"  - Output Shape: {composition_vectors.shape}")
    print("=" * 50)
    print("\nThis tensor is the resolved harmonic signature of the protein.")
    print("It is the input for Phase 3, where this 'music' will be")
    print("translated into a 3D geometric structure.")


if __name__ == "__main__":
    # A sample sequence (first 76 residues of Ubiquitin)
    sample_sequence = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

    if len(sample_sequence) < 48:
        print("❌ Error: Sequence must be at least 48 amino acids long.")
    else:
        compose(sample_sequence)
