#!/usr/bin/env python3
"""
Falsification test for the Amino Acid Timbre Map.

Computes the Pearson correlation between the cosine similarity of our
engineered 48D timbre vectors and the BLOSUM62 substitution matrix.
"""
from __future__ import annotations

import numpy as np
import torch

from bio.amino_acids import get_blosum_vector
from bio.timbre import TimbreGenerator, get_cosine_similarity_vector


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Simple Pearson correlation for validation."""
    return float(np.corrcoef(x, y)[0, 1])


if __name__ == "__main__":
    print("🔬 Validating Amino Acid Timbre Map...")

    # 1. Generate our engineered vectors and their similarity matrix
    print("   - Engineering 48D timbre vectors for 20 amino acids...")
    generator = TimbreGenerator()
    cosine_sim_vec = get_cosine_similarity_vector(generator.timbre_map)

    # 2. Get the ground truth vector from BLOSUM62
    print("   - Loading BLOSUM62 ground truth...")
    blosum_vec = get_blosum_vector()

    # 3. Compute the correlation (The Falsifiable Test)
    print("   - Computing Pearson correlation...")
    correlation = pearson_correlation(cosine_sim_vec.numpy(), blosum_vec)

    # 4. Report the result
    print("\n" + "=" * 40)
    print(f"  Correlation (Cosine Sim vs. BLOSUM62): {correlation:.4f}")
    print("=" * 40)

    # 5. State the conclusion
    if correlation > 0.5:
        print("✅ SUCCESS: The Timbre Map shows significant alignment with biological reality.")
        print("   This provides a sound, falsifiable foundation for Phase 2.")
    else:
        print("❌ FAILURE: The Timbre Map does not align with BLOSUM62.")
        print("   ACTION: Re-engineer the property-to-manifold mapping in bio/timbre.py.")
