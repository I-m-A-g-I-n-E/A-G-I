#!/usr/bin/env python3
"""
Phase 3 Demonstration: The Conductor.

Loads the mean composition vector from Phase 2 and generates a 3D structure
in PDB format.
"""
import argparse
import json
import os
import shutil
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
    # Also save an initial-labeled copy for artifact continuity
    base_noext = output_pdb.rsplit('.', 1)[0]
    initial_pdb = base_noext + "_initial.pdb"
    try:
        shutil.copyfile(output_pdb, initial_pdb)
    except Exception as e:
        print(f"   - Warning: could not create initial-labeled PDB copy: {e}")

    # 4. Run QC and save a report
    report = conductor.quality_check(backbone, phi, psi, modes)
    qc_path = output_pdb.rsplit('.', 1)[0] + "_qc.json"
    with open(qc_path, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(f"   - Wrote QC report to {qc_path}")
    # Also save an initial-labeled QC copy for artifact continuity
    initial_qc_path = base_noext + "_initial_qc.json"
    try:
        with open(initial_qc_path, 'w') as fh:
            json.dump(report, fh, indent=2)
    except Exception as e:
        print(f"   - Warning: could not create initial-labeled QC copy: {e}")
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
        # Start with composition; apply amplification
        comp = mean_composition.clone() if isinstance(mean_composition, torch.Tensor) else mean_composition
        if args.amplify != 1.0:
            comp = comp * float(args.amplify)

        # Certainty from ensemble file if available
        cert_path = f"{input_prefix}_certainty.npy"
        try:
            certainty_arr = np.load(cert_path)
            harmonic_certainty = torch.from_numpy(certainty_arr.astype(np.float32)).view(-1)
            print(f"   - Loaded certainty from {cert_path} (len={len(harmonic_certainty)})")
        except Exception as e:
            print(f"   - Warning: could not load certainty at {cert_path}: {e}. Using uniform certainty.")
            # Default to 1.0 per window; infer windows from composition
            W_def = comp.shape[0] if getattr(comp, 'ndim', 1) == 2 else 48
            harmonic_certainty = torch.ones((int(W_def),), dtype=torch.float32)

        # Repeat windows if requested
        repeat = int(max(1, args.repeat_windows))
        if repeat > 1:
            if isinstance(comp, torch.Tensor):
                if comp.ndim == 1:
                    comp = comp.unsqueeze(0).repeat(repeat, 1)
                else:
                    comp = comp.repeat(repeat, 1)
            else:
                # fall back to numpy then convert back to torch if needed later
                arr = comp if isinstance(comp, np.ndarray) else np.asarray(comp)
                if arr.ndim == 1:
                    arr = np.tile(arr.reshape(1, -1), (repeat, 1))
                else:
                    arr = np.tile(arr, (repeat, 1))
                comp = torch.from_numpy(arr.astype(np.float32))
            # repeat certainty as well
            harmonic_certainty = harmonic_certainty.repeat(repeat)

        # Key (kore) from key estimator — use windowed comp if 2D, else expand
        from bio.key_estimator import estimate_key_and_modes
        kore_input = comp if (isinstance(comp, torch.Tensor) and comp.ndim == 2) else comp.unsqueeze(0)
        kore_vector, _ = estimate_key_and_modes(kore_input, seq)

        # Prepare dissonance vectors per window (match certainty length)
        Wc = int(harmonic_certainty.shape[0])
        diss_init_vec = torch.tensor([dissonance_initial] * Wc, dtype=torch.float32)
        diss_ref_vec = torch.tensor([dissonance_refined] * Wc, dtype=torch.float32)

        # Sonify using TrinitySonifier with ergonomic stride control
        from bio.sonifier import TrinitySonifier
        center_weights = {'kore': args.wc_kore, 'cert': args.wc_cert, 'diss': args.wc_diss}
        sonifier = TrinitySonifier(bpm=args.bpm, stride_ticks=args.stride_ticks)
        wav_initial = sonifier.sonify_composition_3ch(
            comp, kore_vector, harmonic_certainty, diss_init_vec, center_weights
        )
        sonifier.save_wav(wav_initial, args.audio_wav.replace('.wav', '_initial.wav'))

        if args.refine:
            wav_refined = sonifier.sonify_composition_3ch(
                comp, kore_vector, harmonic_certainty, diss_ref_vec, center_weights
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
    