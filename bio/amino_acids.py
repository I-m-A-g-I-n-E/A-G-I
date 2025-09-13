#!/usr/bin/env python3
"""
Canonical biological constants for amino acids and substitution scores.

This module centralizes:
- AMINO_ACIDS: the 20 standard amino acids (1-letter codes), sorted alphabetically
- BLOSUM62: represented as a (20x20) matrix in the standard order with a
  lookup function that is agnostic to ordering
- AA_PROPERTIES: physicochemical properties per amino acid
- get_blosum_vector(): flattened upper-triangle vector (190 elements) aligning
  to AMINO_ACIDS alphabetical order for correlation tests
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

# --- 1. The 20 Standard Amino Acids (alphabetical) ---
AMINO_ACIDS: List[str] = sorted([
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
])

# --- 2. Ground Truth: BLOSUM62 Substitution Matrix ---
# Stored in the conventional BLOSUM62 order: ARNDCQEGHILKMFPSTWYV
BLOSUM62_ORDER: List[str] = [
    'A','R','N','D','C','Q','E','G','H','I',
    'L','K','M','F','P','S','T','W','Y','V'
]

# Matrix from the standard BLOSUM62 scoring matrix
# Source: NCBI/BLAST and widely cited references
# Each row i and column j corresponds to BLOSUM62_ORDER[i], BLOSUM62_ORDER[j]
BLOSUM62_MATRIX: List[List[int]] = [
    #  A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
    [  4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0], # A
    [ -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3], # R
    [ -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3], # N
    [ -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3], # D
    [  0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1], # C
    [ -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2], # Q
    [ -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2], # E
    [  0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3], # G
    [ -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3], # H
    [ -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3], # I
    [ -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1], # L
    [ -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2], # K
    [ -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1], # M
    [ -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1], # F
    [ -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2], # P
    [  1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2], # S
    [  0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0], # T
    [ -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3], # W
    [ -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1], # Y
    [  0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4], # V
]

# Quick index mapping for lookup
_B62_INDEX: Dict[str, int] = {aa: i for i, aa in enumerate(BLOSUM62_ORDER)}


def blosum62_score(aa1: str, aa2: str) -> int:
    """Return the BLOSUM62 substitution score for a pair (order-agnostic)."""
    i = _B62_INDEX[aa1]
    j = _B62_INDEX[aa2]
    return BLOSUM62_MATRIX[i][j]


# --- 3. Physicochemical Properties ---
# Format per amino acid: [Hydrophobicity(Kyte-Doolittle), SideChainVolume(Å^3), Charge@pH7 (-1,0,1), HelixPropensity, SheetPropensity]
# Helix/Sheet propensities approximate Chou-Fasman scales; values are relative and unitless.
AA_PROPERTIES: Dict[str, List[float]] = {
    'A': [ 1.8,  88.6,  0, 1.42, 0.83],  # Alanine
    'R': [-4.5, 173.4, +1, 0.98, 0.93],  # Arginine
    'N': [-3.5, 114.1,  0, 0.67, 0.89],  # Asparagine
    'D': [-3.5, 111.1, -1, 1.01, 0.54],  # Aspartic acid
    'C': [ 2.5, 108.5,  0, 0.70, 1.19],  # Cysteine
    'Q': [-3.5, 143.8,  0, 1.11, 1.10],  # Glutamine
    'E': [-3.5, 138.4, -1, 1.51, 0.37],  # Glutamic acid
    'G': [-0.4,  60.1,  0, 0.57, 0.75],  # Glycine
    'H': [-3.2, 153.2, +1, 1.00, 0.87],  # Histidine (weakly + at pH ~7)
    'I': [ 4.5, 166.7,  0, 1.08, 1.60],  # Isoleucine
    'L': [ 3.8, 166.7,  0, 1.21, 1.30],  # Leucine
    'K': [-3.9, 168.6, +1, 1.16, 0.74],  # Lysine
    'M': [ 1.9, 162.9,  0, 1.45, 1.05],  # Methionine
    'F': [ 2.8, 189.9,  0, 1.13, 1.38],  # Phenylalanine
    'P': [-1.6, 112.7,  0, 0.57, 0.55],  # Proline
    'S': [-0.8,  89.0,  0, 0.77, 0.75],  # Serine
    'T': [-0.7, 116.1,  0, 0.83, 1.19],  # Threonine
    'W': [-0.9, 227.8,  0, 1.08, 1.37],  # Tryptophan
    'Y': [-1.3, 193.6,  0, 0.69, 1.47],  # Tyrosine
    'V': [ 4.2, 140.0,  0, 1.06, 1.70],  # Valine
}


def get_blosum_vector() -> np.ndarray:
    """Returns a flattened 190-element vector of the upper triangle of BLOSUM62.

    The order of pairs follows the alphabetical `AMINO_ACIDS`. This function
    uses the canonical BLOSUM62 matrix (in its standard order) under the hood.
    """
    vec: List[int] = []
    for i in range(len(AMINO_ACIDS)):
        for j in range(i + 1, len(AMINO_ACIDS)):
            aa1, aa2 = AMINO_ACIDS[i], AMINO_ACIDS[j]
            score = blosum62_score(aa1, aa2)
            vec.append(int(score))
    return np.array(vec, dtype=np.float32)
