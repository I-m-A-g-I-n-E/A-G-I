#!/usr/bin/env python3
"""
Phase 3 Demonstration: The Conductor.

Loads the mean composition vector from Phase 2 and generates a 3D structure
in PDB format.
"""
import argparse
import numpy as np
import torch
from bio.conductor import Conductor


def generate(input_prefix: str, output_pdb: str):
    """
    Runs the full structure generation process.
    """
    print("🎻 Tuning up for geometric realization...")

    # 1. Load the score from Phase 2
    mean_composition_path = f"{input_prefix}_mean.npy"
    print(f"   - Loading mean composition from {mean_composition_path}")
    mean_composition = torch.from_numpy(np.load(mean_composition_path))

    # 2. Instantiate the Conductor and build the structure
    conductor = Conductor()
    print("   - Building C-alpha backbone from harmonic composition...")
    ca_trace = conductor.build_backbone(mean_composition)

    # 3. Save the final structure to a PDB file
    conductor.save_to_pdb(ca_trace, output_pdb)
    print("\n" + "=" * 50)
    print("  Structure Generation Complete")
    print(f"  - View the result with: pymol {output_pdb}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Generate 3D structure from harmonic composition.")
    parser.add_argument("--input-prefix", required=True, help="Prefix of the .npy files from Phase 2 (e.g., outputs/my_run)")
    parser.add_argument("--output-pdb", required=True, help="Path to save the final PDB file (e.g., outputs/structure.pdb)")
    args = parser.parse_args()

    generate(args.input_prefix, args.output_pdb)
