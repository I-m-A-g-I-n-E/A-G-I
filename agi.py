#!/usr/bin/env python3
"""
AGI Unified CLI

Subcommands:
- compose: Compose an AA sequence into 48D windows and save ensemble artifacts.
- structure: Generate PDB from composition; optional refinement and 3-channel sonification.
- sonify: Sonify a composition prefix into a single 3-channel WAV.

This CLI wraps the pipeline in bio/pipeline.py and does not break existing scripts.
"""
from __future__ import annotations

import os
import json
from typing import Optional
import subprocess

import click
import numpy as np
import torch

from bio import pipeline
from bio.utils import ensure_dir_for, load_ensemble, save_json


@click.group()
def cli():
    """AGI Unified CLI: compose → structure → sonify"""
    pass


# -------------------------
# compose
# -------------------------

@cli.command()
@click.option('--sequence', required=True, help='Amino acid sequence (>=48 residues).')
@click.option('--samples', type=int, default=1, show_default=True)
@click.option('--variability', type=float, default=0.0, show_default=True)
@click.option('--seed', type=int, default=None)
@click.option('--window-jitter', is_flag=True, default=False, show_default=True)
@click.option('--save-prefix', type=str, required=True, help='Prefix path to save outputs (without extension).')
@click.option('--save-format', type=click.Choice(['pt', 'npy']), default='npy', show_default=True)
def compose(sequence: str, samples: int, variability: float, seed: Optional[int], window_jitter: bool,
            save_prefix: str, save_format: str):
    """Compose an AA sequence into 48D windows and save mean + certainty."""
    mean, certainty = pipeline.compose_sequence(
        sequence.strip().upper(),
        samples=max(1, int(samples)),
        variability=max(0.0, min(1.0, variability)),
        seed=seed,
        window_jitter=bool(window_jitter),
    )
    prefix = os.path.expanduser(save_prefix)
    os.makedirs(os.path.dirname(prefix) or '.', exist_ok=True)
    if save_format == 'pt':
        torch.save(mean, f"{prefix}_mean.pt")
        torch.save(certainty, f"{prefix}_certainty.pt")
        click.echo(f"💾 Saved: {prefix}_mean.pt, {prefix}_certainty.pt")
    else:
        np.save(f"{prefix}_mean.npy", mean.cpu().numpy())
        np.save(f"{prefix}_certainty.npy", certainty.cpu().numpy())
        click.echo(f"💾 Saved: {prefix}_mean.npy, {prefix}_certainty.npy")


# -------------------------
# structure
# -------------------------

def _load_sequence_arg(seq_arg: Optional[str], seq_file: Optional[str], fallback_len: int) -> str:
    if seq_arg:
        return seq_arg.strip().upper()
    if seq_file:
        with open(seq_file, 'r') as fh:
            s = fh.read().strip().upper()
            return ''.join([c for c in s if c.isalpha()])
    return 'A' * fallback_len


@cli.command()
@click.option('--input-prefix', required=True, help='Prefix of the ensemble files (e.g., outputs/my_run)')
@click.option('--output-pdb', required=True, help='Path to save the final PDB file (e.g., outputs/structure.pdb)')
@click.option('--sequence', required=False, help='Protein sequence as one-letter codes (e.g., ACDEFGH)')
@click.option('--sequence-file', required=False, help='Path to a file containing the protein sequence')
@click.option('--refine', is_flag=True, default=False, show_default=True, help='Enable torsion refinement pass')
@click.option('--refine-iters', type=int, default=150, show_default=True)
@click.option('--refine-step', type=float, default=2.0, show_default=True)
@click.option('--refine-seed', type=int, default=None)
@click.option('--w-clash', type=float, default=1.5, show_default=True)
@click.option('--w-ca', type=float, default=1.0, show_default=True)
@click.option('--w-smooth', type=float, default=0.2, show_default=True)
@click.option('--w-snap', type=float, default=0.5, show_default=True)
@click.option('--w-neighbor-ca', type=float, default=0.0, show_default=True)
@click.option('--w-nonadj-ca', type=float, default=0.0, show_default=True)
@click.option('--w-dihedral', type=float, default=0.0, show_default=True)
# Debug/trace
@click.option('--trace', 'debug_trace_path', type=str, default=None, help='Path to write JSONL refinement trace')
@click.option('--trace-log-every', 'debug_log_every', type=int, default=50, show_default=True, help='Trace every N iterations')
@click.option('--trace-verbose', 'debug_verbose', is_flag=True, default=False, show_default=True, help='Trace every iteration with details')
@click.option('--timeout-sec', 'wall_timeout_sec', type=float, default=None, help='Wall-clock timeout (seconds) for refinement')
# Spacing/repair tuning
@click.option('--neighbor-threshold', type=float, default=3.2, show_default=True)
@click.option('--spacing-max-attempts', type=int, default=300, show_default=True)
@click.option('--spacing-top-bins', type=int, default=4, show_default=True)
@click.option('--spacing-continue-full', is_flag=True, default=False, show_default=True)
@click.option('--final-attempts', type=int, default=2000, show_default=True)
@click.option('--spacing-cross-mode', is_flag=True, default=False, show_default=True, help='Allow spacing pass to pull from any mode bins')
@click.option('--critical-override-iters', type=int, default=0, show_default=True, help='Run greedy min_ca_ca override for N iterations at end')
# Parallelism
@click.option('--num-workers', type=int, default=0, show_default=True, help='Process workers for candidate eval (0=auto)')
@click.option('--eval-batch', type=int, default=256, show_default=True, help='Batch size per worker for candidate eval')
# Sonification
@click.option('--sonify-3ch', is_flag=True, default=False, show_default=True)
@click.option('--audio-wav', type=str, default=None, help='Output path for the 3-channel WAV file.')
@click.option('--bpm', type=float, default=96.0, show_default=True)
@click.option('--stride-ticks', type=int, default=16, show_default=True)
@click.option('--amplify', type=float, default=1.0, show_default=True, help='Gain factor for composition before sonify')
@click.option('--wc-kore', type=float, default=1.5, show_default=True)
@click.option('--wc-cert', type=float, default=1.0, show_default=True)
@click.option('--wc-diss', type=float, default=2.5, show_default=True)
@click.option('--repeat-windows', type=int, default=1, show_default=True)
@click.option('--reference-pdb', type=str, default=None, help='Path to reference PDB for TM-score/RMSD validation')
def structure(input_prefix: str, output_pdb: str, sequence: Optional[str], sequence_file: Optional[str],
              refine: bool, refine_iters: int, refine_step: float, refine_seed: Optional[int],
              w_clash: float, w_ca: float, w_smooth: float, w_snap: float,
              w_neighbor_ca: float, w_nonadj_ca: float, w_dihedral: float,
              debug_trace_path: Optional[str], debug_log_every: int, debug_verbose: bool, wall_timeout_sec: Optional[float],
              neighbor_threshold: float, spacing_max_attempts: int, spacing_top_bins: int, spacing_continue_full: bool,
              final_attempts: int, spacing_cross_mode: bool, critical_override_iters: int,
              num_workers: int, eval_batch: int,
              sonify_3ch: bool, audio_wav: Optional[str], bpm: float, stride_ticks: int, amplify: float,
              wc_kore: float, wc_cert: float, wc_diss: float, repeat_windows: int, reference_pdb: Optional[str]):
    """Generate PDB from composition mean; optional refine and 3-channel sonification."""
    mean, certainty = load_ensemble(input_prefix)
    if mean.ndim == 1:
        L = mean.shape[0]
    else:
        L = mean.shape[0] if mean.shape[0] > 1 else mean.shape[1]
    seq = _load_sequence_arg(sequence, sequence_file, fallback_len=int(L if L > 0 else 48))

    # Optional amplification and repetition
    comp = mean.clone()
    if amplify != 1.0:
        comp = comp * float(amplify)
    if repeat_windows > 1:
        comp = comp.repeat(int(max(1, repeat_windows)), 1)
        certainty = certainty.repeat(int(max(1, repeat_windows)))

    # Conduct
    backbone, phi, psi, modes, conductor = pipeline.conduct_backbone(comp, seq)
    ensure_dir_for(output_pdb)
    conductor.save_to_pdb(backbone, output_pdb)

    # QC
    qc = pipeline.quality_report(conductor, backbone, phi, psi, modes)
    # Record weights provenance (initial)
    qc.setdefault('summary', {})['weights'] = {
        'clash': 1.5, 'ca': 1.0, 'smooth': 0.2, 'snap': 0.5,
        'neighbor_ca': 0.0, 'nonadj_ca': 0.0, 'dihedral': 0.0,
    }
    base_noext = output_pdb.rsplit('.', 1)[0]
    qc_path = base_noext + '_qc.json'
    save_json(qc, qc_path)
    # Also store initial-labeled copy for provenance
    init_qc_path = base_noext + '_initial_qc.json'
    save_json(qc, init_qc_path)

    # Dissonance (initial)
    torsions_init = np.stack([phi, psi], axis=1)
    diss_initial = conductor.calculate_dissonance(backbone, torsions_init, modes, {
        'clash': 1.5, 'ca': 1.0, 'smooth': 0.2, 'snap': 0.5,
    })

    # Optional refine
    if refine:
        weights = {
            'clash': w_clash,
            'ca': w_ca,
            'smooth': w_smooth,
            'snap': w_snap,
            'neighbor_ca': w_neighbor_ca,
            'nonadj_ca': w_nonadj_ca,
            'dihedral': w_dihedral,
        }
        refined_torsions, refined_backbone = pipeline.refine_backbone(
            conductor, backbone, phi, psi, modes, seq,
            max_iters=int(refine_iters), step_deg=float(refine_step), seed=refine_seed, weights=weights,
            debug_trace_path=debug_trace_path, debug_log_every=int(debug_log_every),
            debug_verbose=bool(debug_verbose), wall_timeout_sec=wall_timeout_sec,
            neighbor_threshold=float(neighbor_threshold), spacing_max_attempts=int(spacing_max_attempts),
            spacing_top_bins=int(spacing_top_bins), spacing_continue_full=bool(spacing_continue_full),
            final_attempts=int(final_attempts), spacing_cross_mode=bool(spacing_cross_mode),
            critical_override_iters=int(critical_override_iters),
            num_workers=int(num_workers), eval_batch=int(eval_batch),
        )
        if debug_trace_path:
            click.echo(f"   - Trace: {debug_trace_path}")
        ref_pdb = output_pdb.replace('.pdb', '_refined.pdb')
        conductor.save_to_pdb(refined_backbone, ref_pdb)
        # QC refined
        rphi = np.array([t[0] for t in refined_torsions], dtype=np.float32)
        rpsi = np.array([t[1] for t in refined_torsions], dtype=np.float32)
        qc_ref = pipeline.quality_report(conductor, refined_backbone, rphi, rpsi, modes)
        # Attach refine stats if available
        try:
            stats = getattr(conductor, 'last_refine_stats', None)
            if stats is not None:
                qc_ref.setdefault('summary', {})['refine_stats'] = stats
                click.echo(f"   - Spacing pass: attempts={stats.get('spacing_attempts')}, bins_tried={stats.get('bins_tried')}, final_min_ca_ca={stats.get('final_min_ca_ca_after_spacing')}")
        except Exception:
            pass
        # Record weights provenance (refined)
        qc_ref.setdefault('summary', {})['weights'] = weights
        qc_ref_path = base_noext + '_refined_qc.json'
        save_json(qc_ref, qc_ref_path)
        # Treat refined QC as the definitive final QC report as well
        save_json(qc_ref, qc_path)
        # Dissonance refined
        torsions_ref = np.stack([rphi, rpsi], axis=1)
        diss_refined = conductor.calculate_dissonance(refined_backbone, torsions_ref, modes, weights)
    else:
        diss_refined = diss_initial

    # Optional validation against reference structure
    if reference_pdb:
        try:
            from agi.metro.validation import compare_structures
            val = compare_structures(ref_pdb if refine else output_pdb, reference_pdb)
            # Append to final qc
            try:
                # read-modify-write final qc
                with open(qc_path, 'r') as fh:
                    qc_final = json.load(fh)
            except Exception:
                qc_final = {}
            qc_final['validation'] = val
            save_json(qc_final, qc_path)
            click.echo(f"   - Validation: TM-score={val.get('tm_score'):.3f}, RMSD={val.get('rmsd'):.3f}")
        except Exception as e:
            click.echo(f"[warn] Validation skipped: {e}")

    # Optional 3ch sonify
    if sonify_3ch and audio_wav:
        kore = pipeline.estimate_kore(comp, seq)
        W = int(certainty.shape[0])
        diss_init_vec = pipeline.dissonance_scalar_to_vec(float(diss_initial), W)
        diss_ref_vec = pipeline.dissonance_scalar_to_vec(float(diss_refined), W)
        weights_center = {'kore': wc_kore, 'cert': wc_cert, 'diss': wc_diss}
        htags = getattr(conductor, 'handedness', None)
        wave_init = pipeline.sonify_3ch(comp, kore, certainty, diss_init_vec, bpm=bpm, stride_ticks=stride_ticks, center_weights=weights_center, handedness=htags)
        pipeline.save_wav(wave_init, audio_wav.replace('.wav', '_initial.wav'))
        if refine:
            wave_ref = pipeline.sonify_3ch(comp, kore, certainty, diss_ref_vec, bpm=bpm, stride_ticks=stride_ticks, center_weights=weights_center, handedness=htags)
            pipeline.save_wav(wave_ref, audio_wav.replace('.wav', '_refined.wav'))

    click.echo("\n=== Structure Generation Complete ===")
    click.echo(f"PDB: {output_pdb}")
    click.echo(f"QC:  {qc_path}")


# -------------------------
# sonify
# -------------------------

@cli.command()
@click.option('--input-prefix', required=True, help='Prefix of the ensemble files (e.g., outputs/my_run)')
@click.option('--output-wav', required=True, help='Path to save a 3-channel WAV (T,3)')
@click.option('--sequence', required=False, help='Sequence (used to estimate kore)')
@click.option('--bpm', type=float, default=96.0, show_default=True)
@click.option('--stride-ticks', type=int, default=16, show_default=True)
@click.option('--dissonance', type=float, default=0.0, show_default=True, help='Scalar dissonance per window')
@click.option('--wc-kore', type=float, default=1.5, show_default=True)
@click.option('--wc-cert', type=float, default=1.0, show_default=True)
@click.option('--wc-diss', type=float, default=2.5, show_default=True)
@click.option('--amplify', type=float, default=1.0, show_default=True)
def sonify(input_prefix: str, output_wav: str, sequence: Optional[str], bpm: float, stride_ticks: int,
           dissonance: float, wc_kore: float, wc_cert: float, wc_diss: float, amplify: float):
    """Sonify a composition prefix into a single 3-channel WAV file."""
    mean, certainty = load_ensemble(input_prefix)
    seq = sequence or ('A' * int(mean.shape[0] if mean.ndim == 1 else mean.shape[0]))
    comp = mean.clone()
    if amplify != 1.0:
        comp = comp * float(amplify)
    kore = pipeline.estimate_kore(comp, seq)
    diss_vec = pipeline.dissonance_scalar_to_vec(float(dissonance), int(certainty.shape[0]))
    wave = pipeline.sonify_3ch(comp, kore, certainty, diss_vec, bpm=bpm, stride_ticks=stride_ticks,
                               center_weights={'kore': wc_kore, 'cert': wc_cert, 'diss': wc_diss})
    ensure_dir_for(output_wav)
    pipeline.save_wav(wave, output_wav)
    click.echo(f"Saved: {output_wav}")


# -------------------------
# play (compose → structure → sonify)
# -------------------------

@cli.command()
@click.option('--sequence', required=True, help='Amino acid sequence (>=48 residues).')
@click.option('--samples', type=int, default=1, show_default=True)
@click.option('--variability', type=float, default=0.0, show_default=True)
@click.option('--seed', type=int, default=None)
@click.option('--window-jitter', is_flag=True, default=False, show_default=True)
@click.option('--save-prefix', type=str, required=True, help='Prefix path to save ensemble outputs (without extension).')
@click.option('--output-pdb', required=True, help='Path to save the final PDB file (e.g., outputs/structure.pdb)')
@click.option('--refine', is_flag=True, default=False, show_default=True, help='Enable torsion refinement pass')
@click.option('--refine-iters', type=int, default=150, show_default=True)
@click.option('--refine-step', type=float, default=2.0, show_default=True)
@click.option('--refine-seed', type=int, default=None)
@click.option('--w-clash', type=float, default=1.5, show_default=True)
@click.option('--w-ca', type=float, default=1.0, show_default=True)
@click.option('--w-smooth', type=float, default=0.2, show_default=True)
@click.option('--w-snap', type=float, default=0.5, show_default=True)
# Sonification
@click.option('--sonify-3ch', is_flag=True, default=False, show_default=True)
@click.option('--audio-wav', type=str, default=None, help='Output path for the 3-channel WAV file.')
@click.option('--bpm', type=float, default=96.0, show_default=True)
@click.option('--stride-ticks', type=int, default=16, show_default=True)
@click.option('--amplify', type=float, default=1.0, show_default=True, help='Gain factor for composition before sonify')
@click.option('--wc-kore', type=float, default=1.5, show_default=True)
@click.option('--wc-cert', type=float, default=1.0, show_default=True)
@click.option('--wc-diss', type=float, default=2.5, show_default=True)
@click.option('--repeat-windows', type=int, default=1, show_default=True)
def play(sequence: str, samples: int, variability: float, seed: Optional[int], window_jitter: bool,
         save_prefix: str, output_pdb: str, refine: bool, refine_iters: int, refine_step: float, refine_seed: Optional[int],
         w_clash: float, w_ca: float, w_smooth: float, w_snap: float,
         w_neighbor_ca: float, w_nonadj_ca: float, w_dihedral: float,
         debug_trace_path: Optional[str], debug_log_every: int, debug_verbose: bool, wall_timeout_sec: Optional[float],
         neighbor_threshold: float, spacing_max_attempts: int, spacing_top_bins: int, spacing_continue_full: bool,
         final_attempts: int, spacing_cross_mode: bool, critical_override_iters: int,
         num_workers: int, eval_batch: int,
         sonify_3ch: bool, audio_wav: Optional[str], bpm: float,
         stride_ticks: int, amplify: float, wc_kore: float, wc_cert: float, wc_diss: float, repeat_windows: int):
    """One-shot pipeline: compose → structure (optional refine) → optional 3ch sonify."""
    # Compose and save ensemble
    mean, certainty = pipeline.compose_sequence(
        sequence.strip().upper(),
        samples=max(1, int(samples)),
        variability=max(0.0, min(1.0, variability)),
        seed=seed,
        window_jitter=bool(window_jitter),
    )
    prefix = os.path.expanduser(save_prefix)
    os.makedirs(os.path.dirname(prefix) or '.', exist_ok=True)
    np.save(f"{prefix}_mean.npy", mean.cpu().numpy())
    np.save(f"{prefix}_certainty.npy", certainty.cpu().numpy())
    click.echo(f"💾 Saved ensemble: {prefix}_mean.npy, {prefix}_certainty.npy")

    # Optional amplification and repetition
    comp = mean.clone()
    if amplify != 1.0:
        comp = comp * float(amplify)
    if repeat_windows > 1:
        comp = comp.repeat(int(max(1, repeat_windows)), 1)
        certainty = certainty.repeat(int(max(1, repeat_windows)))

    # Conduct
    backbone, phi, psi, modes, conductor = pipeline.conduct_backbone(comp, sequence.strip().upper())
    ensure_dir_for(output_pdb)
    conductor.save_to_pdb(backbone, output_pdb)

    # QC
    qc = pipeline.quality_report(conductor, backbone, phi, psi, modes)
    qc_path = output_pdb.rsplit('.', 1)[0] + '_qc.json'
    save_json(qc, qc_path)

    # Compute initial dissonance scalar
    torsions_init = np.stack([phi, psi], axis=1)
    diss_initial = conductor.calculate_dissonance(backbone, torsions_init, modes, {
        'clash': 1.5, 'ca': 1.0, 'smooth': 0.2, 'snap': 0.5,
    })

    # Optional refine
    if refine:
        weights = {
            'clash': w_clash,
            'ca': w_ca,
            'smooth': w_smooth,
            'snap': w_snap,
            'neighbor_ca': w_neighbor_ca,
            'nonadj_ca': w_nonadj_ca,
            'dihedral': w_dihedral,
        }
        refined_torsions, refined_backbone = pipeline.refine_backbone(
            conductor, backbone, phi, psi, modes, sequence.strip().upper(),
            max_iters=int(refine_iters), step_deg=float(refine_step), seed=refine_seed, weights=weights,
            debug_trace_path=debug_trace_path, debug_log_every=int(debug_log_every),
            debug_verbose=bool(debug_verbose), wall_timeout_sec=wall_timeout_sec,
            neighbor_threshold=float(neighbor_threshold), spacing_max_attempts=int(spacing_max_attempts),
            spacing_top_bins=int(spacing_top_bins), spacing_continue_full=bool(spacing_continue_full),
            final_attempts=int(final_attempts),
            num_workers=int(num_workers), eval_batch=int(eval_batch),
        )
        if debug_trace_path:
            click.echo(f"   - Trace: {debug_trace_path}")
        ref_pdb = output_pdb.replace('.pdb', '_refined.pdb')
        conductor.save_to_pdb(refined_backbone, ref_pdb)
        rphi = np.array([t[0] for t in refined_torsions], dtype=np.float32)
        rpsi = np.array([t[1] for t in refined_torsions], dtype=np.float32)
        qc_ref = pipeline.quality_report(conductor, refined_backbone, rphi, rpsi, modes)
        qc_ref.setdefault('summary', {})['weights'] = weights
        qc_ref_path = output_pdb.rsplit('.', 1)[0] + '_refined_qc.json'
        save_json(qc_ref, qc_ref_path)
        save_json(qc_ref, qc_path)
        # Refined dissonance
        torsions_ref = np.stack([rphi, rpsi], axis=1)
        diss_refined = conductor.calculate_dissonance(refined_backbone, torsions_ref, modes, weights)
    else:
        diss_refined = diss_initial

    # Optional 3ch sonify
    if sonify_3ch and audio_wav:
        kore = pipeline.estimate_kore(comp, sequence.strip().upper())
        W = int(certainty.shape[0])
        diss_init_vec = pipeline.dissonance_scalar_to_vec(float(diss_initial), W)
        htags = getattr(conductor, 'handedness', None)
        wave_init = pipeline.sonify_3ch(comp, kore, certainty, diss_init_vec, bpm=bpm, stride_ticks=stride_ticks,
                                        center_weights={'kore': wc_kore, 'cert': wc_cert, 'diss': wc_diss}, handedness=htags)
        pipeline.save_wav(wave_init, audio_wav.replace('.wav', '_initial.wav'))
        if refine:
            diss_ref_vec = pipeline.dissonance_scalar_to_vec(float(diss_refined), W)
            wave_ref = pipeline.sonify_3ch(comp, kore, certainty, diss_ref_vec, bpm=bpm, stride_ticks=stride_ticks,
                                           center_weights={'kore': wc_kore, 'cert': wc_cert, 'diss': wc_diss}, handedness=htags)
            pipeline.save_wav(wave_ref, audio_wav.replace('.wav', '_refined.wav'))

    click.echo("\n=== Play Complete ===")
    click.echo(f"PDB: {output_pdb}")
    click.echo(f"QC:  {qc_path}")


# -------------------------
# immunity passthrough (delegates to immunity.py)
# -------------------------

@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
def immunity(args):
    """Delegate to the existing immunity.py CLI with passthrough options."""
    cmd = ["python3", "immunity.py", *list(args)]
    click.echo("Running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    cli()
