#!/usr/bin/env python3
"""
Phase 3 Demonstration: The Conductor.

Loads the mean composition vector from Phase 2 and generates a 3D structure
in PDB format.
"""
import argparse
import json
import numpy as np
import torch
from bio.conductor import Conductor


def _load_sequence(seq_arg: str | None, seq_file: str | None, fallback_len: int) -> str:
    if seq_arg:
        return seq_arg.strip().upper()
    if seq_file:
        with open(seq_file, 'r') as fh:
            s = fh.read().strip().upper()
            return ''.join([c for c in s if c.isalpha()])
    return 'A' * fallback_len


def generate(input_prefix: str, output_pdb: str, sequence: str | None = None, sequence_file: str | None = None):
    """
    Runs the full structure generation process.
    """
    print("🎻 Tuning up for geometric realization...")

    # 1. Load the score from Phase 2
    mean_composition_path = f"{input_prefix}_mean.npy"
    print(f"   - Loading mean composition from {mean_composition_path}")
    mean_composition = torch.from_numpy(np.load(mean_composition_path))
    # Determine fallback length for default sequence
    if mean_composition.ndim == 1:
        L = mean_composition.shape[0]
    else:
        L = mean_composition.shape[0] if mean_composition.shape[0] > 1 else mean_composition.shape[1]
    seq = _load_sequence(sequence, sequence_file, fallback_len=int(L if L > 0 else 48))
    print(f"   - Using sequence of length {len(seq)}")

    # 2. Instantiate the Conductor and build the structure
    conductor = Conductor()
    print("   - Building full backbone with Harmony Constraint Layer...")
    backbone, phi, psi, modes = conductor.build_backbone(mean_composition, sequence=seq)

    # 3. Save the final structure to a PDB file
    conductor.save_to_pdb(backbone, output_pdb)

    # 4. Run QC and save a report
    report = conductor.quality_check(backbone, phi, psi, modes)
    qc_path = output_pdb.rsplit('.', 1)[0] + "_qc.json"
    with open(qc_path, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(f"   - Wrote QC report to {qc_path}")
    print("\n" + "=" * 50)
    print("  Structure Generation Complete")
    print(f"  - View the result with: pymol {output_pdb}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Generate 3D structure from harmonic composition.")
    parser.add_argument("--input-prefix", required=True, help="Prefix of the .npy files from Phase 2 (e.g., outputs/my_run)")
    parser.add_argument("--output-pdb", required=True, help="Path to save the final PDB file (e.g., outputs/structure.pdb)")
    parser.add_argument("--sequence", required=False, help="Protein sequence as one-letter codes (e.g., ACDEFGH)")
    parser.add_argument("--sequence-file", required=False, help="Path to a file containing the protein sequence")
    args = parser.parse_args()

    generate(args.input_prefix, args.output_pdb, args.sequence, args.sequence_file)
