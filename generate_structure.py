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
    # Compute initial dissonance score for sonification center weighting
    init_weights = {
        'clash': 1.5,
        'ca': 1.0,
        'smooth': 0.2,
        'snap': 0.5,
    }
    torsions_init = np.stack([phi, psi], axis=1)
    dissonance_initial = conductor.calculate_dissonance(backbone, torsions_init, modes, init_weights)

    # 5. Optional refinement
    print("🎼 Beginning refinement rehearsal...")
    if args.refine:
        weights = {
            'clash': args.w_clash,
            'ca': args.w_ca,
            'smooth': args.w_smooth,
            'snap': args.w_snap,
        }
        print(f"   - Refinement weights: {weights}")
        initial_torsions = [(float(phi[i]), float(psi[i])) for i in range(len(phi))]
        refined_torsions, refined_backbone = conductor.refine_torsions(
            initial_torsions, modes, seq, max_iters=args.refine_iters, step_deg=args.refine_step, seed=args.refine_seed,
            weights=weights
        )
        refined_pdb_path = output_pdb.replace('.pdb', '_refined.pdb')
        conductor.save_to_pdb(refined_backbone, refined_pdb_path)
        # QC for refined
        rphi = np.array([t[0] for t in refined_torsions])
        rpsi = np.array([t[1] for t in refined_torsions])
        qc_refined = conductor.quality_check(refined_backbone, rphi, rpsi, modes)
        qc_refined_path = output_pdb.rsplit('.', 1)[0] + "_refined_qc.json"
        with open(qc_refined_path, 'w') as fh:
            json.dump(qc_refined, fh, indent=2)
        # Dissonance for refined
        torsions_ref = np.stack([rphi, rpsi], axis=1)
        dissonance_refined = conductor.calculate_dissonance(refined_backbone, torsions_ref, modes, weights)
        print("\n" + "=" * 50)
        print("  Rehearsal Complete")
        print(f"  - Initial Clashes: {report['summary']['num_clashes']}")
        print(f"  - Refined Clashes: {qc_refined['summary']['num_clashes']}")
        print(f"  - View refined structure: pymol {refined_pdb_path}")
        print("=" * 50)
    else:
        dissonance_refined = dissonance_initial
    
    # 6. Optional 3-channel sonification
    if args.sonify_3ch and args.audio_wav:
        print("🎶 Orchestrating 3-channel sonification...")
        if args.amplify != 1.0:
            mean_composition = mean_composition * args.amplify

        # Key (kore) from key estimator
        from bio.key_estimator import estimate_key_and_modes
        kore_vector, _ = estimate_key_and_modes(mean_composition if mean_composition.ndim == 2 else mean_composition.unsqueeze(0), seq)

        # Certainty from ensemble file if available
        cert_path = f"{input_prefix}_certainty.npy"
        try:
            certainty_arr = np.load(cert_path)
            harmonic_certainty = torch.from_numpy(certainty_arr.astype(np.float32)).view(-1)
            print(f"   - Loaded certainty from {cert_path} (len={len(harmonic_certainty)})")
        except Exception as e:
            print(f"   - Warning: could not load certainty at {cert_path}: {e}. Using uniform certainty.")
            # Default to 1.0 per window; infer windows from composition
            W = mean_composition.shape[0] if mean_composition.ndim == 2 else 48
            harmonic_certainty = torch.ones((int(W),), dtype=torch.float32)

        # Prepare dissonance vectors per window
        Wc = int(harmonic_certainty.shape[0])
        diss_init_vec = torch.tensor([dissonance_initial] * Wc, dtype=torch.float32)
        diss_ref_vec = torch.tensor([dissonance_refined] * Wc, dtype=torch.float32)

        # Sonify using TrinitySonifier
        from bio.sonifier import TrinitySonifier
        center_weights = {'kore': args.wc_kore, 'cert': args.wc_cert, 'diss': args.wc_diss}
        sonifier = TrinitySonifier(bpm=args.bpm)
        # composition input may be [48] or [W,48]; both are supported by method
        wav_initial = sonifier.sonify_composition_3ch(
            mean_composition, kore_vector, harmonic_certainty, diss_init_vec, center_weights
        )
        sonifier.save_wav(wav_initial, args.audio_wav.replace('.wav', '_initial.wav'))

        if args.refine:
            # Using the same composition but different dissonance highlights improvement
            wav_refined = sonifier.sonify_composition_3ch(
                mean_composition, kore_vector, harmonic_certainty, diss_ref_vec, center_weights
            )
            sonifier.save_wav(wav_refined, args.audio_wav.replace('.wav', '_refined.wav'))

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
    parser.add_argument("--refine", action="store_true", help="Enable torsion refinement pass")
    parser.add_argument("--refine-iters", type=int, default=150, dest="refine_iters", help="Max refinement iterations")
    parser.add_argument("--refine-step", type=float, default=2.0, dest="refine_step", help="Refinement step size in degrees")
    parser.add_argument("--refine-seed", type=int, default=None, dest="refine_seed", help="Random seed for refinement")
    parser.add_argument("--w-clash", type=float, default=1.5, dest="w_clash", help="Weight for clash penalty in dissonance")
    parser.add_argument("--w-ca", type=float, default=1.0, dest="w_ca", help="Weight for CA-CA distance penalty in dissonance")
    parser.add_argument("--w-smooth", type=float, default=0.2, dest="w_smooth", help="Weight for torsion smoothness penalty")
    parser.add_argument("--w-snap", type=float, default=0.5, dest="w_snap", help="Weight for scale snapping penalty")
    # Sonification controls
    parser.add_argument("--sonify-3ch", action="store_true", dest="sonify_3ch", help="Enable 3-channel sonification output.")
    parser.add_argument("--audio-wav", default=None, dest="audio_wav", help="Output path for the 3-channel WAV file.")
    parser.add_argument("--bpm", type=float, default=96.0, help="Tempo for time grid (48 ticks per bar).")
    parser.add_argument("--amplify", type=float, default=1.0, help="Linear gain factor to apply to the composition vector before sonification.")
    parser.add_argument("--wc-kore", type=float, default=1.5, dest="wc_kore", help="Weight for kore projection in center channel.")
    parser.add_argument("--wc-cert", type=float, default=1.0, dest="wc_cert", help="Weight for harmonic certainty in center channel.")
    parser.add_argument("--wc-diss", type=float, default=2.5, dest="wc_diss", help="Weight for dissonance in center channel.")
    args = parser.parse_args()

    generate(args.input_prefix, args.output_pdb, args.sequence, args.sequence_file)
    