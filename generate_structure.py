#!/usr/bin/env python3
"""
Phase 3 Demonstration: The Conductor.

Loads the mean composition vector from Phase 2 and generates a 3D structure
in PDB format.
"""
import argparse
import os
import numpy as np
import torch

from bio.utils import load_ensemble, ensure_dir_for, save_json
from bio import pipeline


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
    print(f"   - Loading ensemble from prefix: {input_prefix}")
    mean_composition, certainty = load_ensemble(input_prefix)
    # Determine fallback length for default sequence
    if mean_composition.ndim == 1:
        L = mean_composition.shape[0]
    else:
        L = mean_composition.shape[0] if mean_composition.shape[0] > 1 else mean_composition.shape[1]
    seq = _load_sequence(sequence, sequence_file, fallback_len=int(L if L > 0 else 48))
    print(f"   - Using sequence of length {len(seq)}")

    # Optional amplification and repetition from args if present
    comp = mean_composition.clone()
    if hasattr(args, 'amplify') and float(args.amplify) != 1.0:
        comp = comp * float(args.amplify)
    if hasattr(args, 'repeat_windows') and int(args.repeat_windows) > 1:
        comp = comp.repeat(int(max(1, args.repeat_windows)), 1)
        certainty = certainty.repeat(int(max(1, args.repeat_windows)))

    # 2. Build the structure via pipeline
    print("   - Building full backbone with Harmony Constraint Layer...")
    backbone, phi, psi, modes, conductor = pipeline.conduct_backbone(comp, seq)

    # 3. Save the final structure to a PDB file
    ensure_dir_for(output_pdb)
    conductor.save_to_pdb(backbone, output_pdb)
    base_noext = output_pdb.rsplit('.', 1)[0]
    initial_pdb = base_noext + "_initial.pdb"
    try:
        # Create an initial-labeled copy for artifact continuity
        conductor.save_to_pdb(backbone, initial_pdb)
    except Exception as e:
        print(f"   - Warning: could not create initial-labeled PDB copy: {e}")

    # 4. Run QC and save a report
    report = pipeline.quality_report(conductor, backbone, phi, psi, modes)
    qc_path = base_noext + "_qc.json"
    save_json(report, qc_path)
    print(f"   - Wrote QC report to {qc_path}")
    # Also save an initial-labeled QC copy for artifact continuity
    initial_qc_path = base_noext + "_initial_qc.json"
    try:
        save_json(report, initial_qc_path)
    except Exception as e:
        print(f"   - Warning: could not create initial-labeled QC copy: {e}")
    # Compute initial dissonance score for sonification center weighting
    init_weights = {'clash': 1.5, 'ca': 1.0, 'smooth': 0.2, 'snap': 0.5}
    torsions_init = np.stack([phi, psi], axis=1)
    dissonance_initial = conductor.calculate_dissonance(backbone, torsions_init, modes, init_weights)

    # 5. Optional refinement
    print("🎼 Beginning refinement rehearsal...")
    if args.refine:
        weights = {'clash': args.w_clash, 'ca': args.w_ca, 'smooth': args.w_smooth, 'snap': args.w_snap}
        print(f"   - Refinement weights: {weights}")
        refined_torsions, refined_backbone = pipeline.refine_backbone(
            conductor, backbone, phi, psi, modes, seq,
            max_iters=args.refine_iters, step_deg=args.refine_step, seed=args.refine_seed, weights=weights,
        )
        refined_pdb_path = output_pdb.replace('.pdb', '_refined.pdb')
        conductor.save_to_pdb(refined_backbone, refined_pdb_path)
        # QC for refined
        rphi = np.array([t[0] for t in refined_torsions])
        rpsi = np.array([t[1] for t in refined_torsions])
        qc_refined = pipeline.quality_report(conductor, refined_backbone, rphi, rpsi, modes)
        qc_refined_path = base_noext + "_refined_qc.json"
        save_json(qc_refined, qc_refined_path)
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
        kore_vector = pipeline.estimate_kore(comp, seq)
        Wc = int(certainty.shape[0])
        diss_init_vec = pipeline.dissonance_scalar_to_vec(float(dissonance_initial), Wc)
        diss_ref_vec = pipeline.dissonance_scalar_to_vec(float(dissonance_refined), Wc)
        center_weights = {'kore': args.wc_kore, 'cert': args.wc_cert, 'diss': args.wc_diss}
        wav_initial = pipeline.sonify_3ch(comp, kore_vector, certainty, diss_init_vec, bpm=args.bpm, stride_ticks=args.stride_ticks)
        pipeline.save_wav(wav_initial, args.audio_wav.replace('.wav', '_initial.wav'))
        if args.refine:
            wav_refined = pipeline.sonify_3ch(comp, kore_vector, certainty, diss_ref_vec, bpm=args.bpm, stride_ticks=args.stride_ticks)
            pipeline.save_wav(wav_refined, args.audio_wav.replace('.wav', '_refined.wav'))

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
    parser.add_argument(
        "--stride-ticks",
        type=int,
        default=16,
        choices=[1, 2, 3, 4, 6, 8, 12, 16, 24, 48],
        dest="stride_ticks",
        help="Number of ticks per window (controls note duration)."
    )
    parser.add_argument(
        "--repeat-windows",
        type=int,
        default=1,
        dest="repeat_windows",
        help="Tile the composition across time to extend short sequences."
    )
    parser.add_argument("--wc-kore", type=float, default=1.5, dest="wc_kore", help="Weight for kore projection in center channel.")
    parser.add_argument("--wc-cert", type=float, default=1.0, dest="wc_cert", help="Weight for harmonic certainty in center channel.")
    parser.add_argument("--wc-diss", type=float, default=2.5, dest="wc_diss", help="Weight for dissonance in center channel.")
    args = parser.parse_args()

    generate(args.input_prefix, args.output_pdb, args.sequence, args.sequence_file)
    