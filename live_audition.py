#!/usr/bin/env python3
"""
Live Audition — interactive duet loop to compose, conduct/refine, and sonify.

Minimal dependency on existing pipeline artifacts:
- Loads ensemble mean and certainty from --input-prefix *_mean.npy / *_certainty.npy if present
- Otherwise, attempts to run `compose_protein.py` to generate them
- Invokes Conductor to build backbone (+ optional refine)
- Sonifies composition into 3 stems (L/C/R)

Usage:
  python3 live_audition.py --input-prefix outputs/ubiquitin_ensemble --sequence-file path/to/seq.txt
Then interact:
  play            # run a performance with current parameters
  v=0.6           # set variability (if we use composer on the fly in future)
  clash=15        # set clash weight
  bpm=100         # set tempo
  exit            # quit
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

# Local imports
from bio.sonifier import TrinitySonifier
from bio.conductor import Conductor
from bio.utils import load_ensemble


def ensure_ensemble(prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        mean, cert = load_ensemble(prefix)
        return mean.to(torch.float32), cert.to(torch.float32)
    except Exception:
        # Fallback: run compose_protein.py for a quick ensemble
        print("[live] Ensemble files not found; running compose_protein.py to generate...")
        cmd = [
            "python3", "compose_protein.py",
            "--samples", "10",
            "--variability", "0.5",
            "--seed", "42",
            "--window-jitter",
            "--save-prefix", prefix,
            "--save-format", "npy",
        ]
        subprocess.run(cmd, check=True)
        mean, cert = load_ensemble(prefix)
        return mean.to(torch.float32), cert.to(torch.float32)


def run_performance(params: Dict[str, Any], iteration: int) -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / f"live_run_{iteration:02d}"

    # 1) Load or create ensemble composition
    mean, certainty = ensure_ensemble(params["input_prefix"])

    # 2) Conduct + optional refinement
    conductor = Conductor()
    sequence = params.get("sequence", None)

    # Build initial backbone via Conductor (key-aware pipeline inside)
    backbone, phi, psi, modes = conductor.build_backbone(mean, sequence=sequence)
    qc_initial = conductor.quality_check(backbone, phi, psi, modes)

    # Save initial PDB and QC
    pdb_path = f"{prefix}_initial.pdb"
    conductor.save_to_pdb(backbone, pdb_path)
    with open(f"{prefix}_initial_qc.json", "w") as f:
        json.dump(qc_initial, f, indent=2)

    if params.get("refine", False):
        print("[live] Starting refinement rehearsal...")
        weights = {
            "clash": float(params.get("w_clash", 10.0)),
            "ca": float(params.get("w_ca", 2.0)),
            "smooth": float(params.get("w_smooth", 0.2)),
            "snap": float(params.get("w_snap", 0.5)),
        }
        refined_torsions, refined_backbone = conductor.refine_torsions(
            list(zip(phi, psi)), modes, sequence,
            max_iters=int(params.get("refine_iters", 1500)),
            step_deg=float(params.get("refine_step", 4.0)),
            seed=int(params.get("refine_seed", 42)),
            weights=weights,
        )
        qc_ref = conductor.quality_check(refined_backbone, refined_torsions, modes)
        ref_pdb = f"{prefix}_refined.pdb"
        conductor.save_to_pdb(refined_backbone, ref_pdb)
        with open(f"{prefix}_refined_qc.json", "w") as f:
            json.dump(qc_ref, f, indent=2)
        print(f"[live] QC refined: clashes={qc_ref['summary']['num_clashes']}, min_ca_ca={qc_ref['summary']['min_ca_ca']:.3f} Å")
    else:
        print(f"[live] QC initial: clashes={qc_initial['summary']['num_clashes']}, min_ca_ca={qc_initial['summary']['min_ca_ca']:.3f} Å")

    # 3) Sonify L/C/R
    son = TrinitySonifier(
        sample_rate=int(params.get("sr", 48000)),
        bpm=float(params.get("bpm", 96.0)),
        tonic_hz=float(params.get("tonic_hz", 220.0)),
        stride_ticks=int(params.get("stride_ticks", 16)),
    )
    L, C, R = son.sonify_composition(mean, certainty)
    son.save_wav(L, f"{prefix}_L.wav")
    son.save_wav(C, f"{prefix}_C.wav")
    son.save_wav(R, f"{prefix}_R.wav")
    print(f"[live] Saved audio stems: {prefix}_L.wav / _C.wav / _R.wav")


def main():
    ap = argparse.ArgumentParser(description="Live Audition — interactive duet loop")
    ap.add_argument("--input-prefix", type=str, required=True, help="Prefix of ensemble files (e.g., outputs/ubi)")
    ap.add_argument("--sequence-file", type=str, default=None, help="Path to plain sequence file (one-letter codes)")
    ap.add_argument("--refine", action="store_true", help="Enable refinement rehearsal")
    ap.add_argument("--refine-iters", type=int, default=1500)
    ap.add_argument("--refine-step", type=float, default=4.0)
    ap.add_argument("--refine-seed", type=int, default=42)
    ap.add_argument("--w-clash", type=float, default=10.0)
    ap.add_argument("--w-ca", type=float, default=2.0)
    ap.add_argument("--w-smooth", type=float, default=0.2)
    ap.add_argument("--w-snap", type=float, default=0.5)
    ap.add_argument("--bpm", type=float, default=96.0)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--tonic-hz", type=float, default=220.0)
    ap.add_argument("--stride-ticks", type=int, default=16)

    args = ap.parse_args()

    params: Dict[str, Any] = {
        "input_prefix": args.input_prefix,
        "refine": bool(args.refine),
        "refine_iters": args.refine_iters,
        "refine_step": args.refine_step,
        "refine_seed": args.refine_seed,
        "w_clash": args.w_clash,
        "w_ca": args.w_ca,
        "w_smooth": args.w_smooth,
        "w_snap": args.w_snap,
        "bpm": args.bpm,
        "sr": args.sr,
        "tonic_hz": args.tonic_hz,
        "stride_ticks": args.stride_ticks,
    }

    # Load sequence if provided
    sequence = None
    if args.sequence_file:
        with open(args.sequence_file, "r") as f:
            sequence = f.read().strip().splitlines()[0].strip()
            # Remove FASTA header if present
            if sequence.startswith(">"):
                sequence = "".join([ln.strip() for ln in f.read().splitlines() if ln and not ln.startswith(">")])
    params["sequence"] = sequence

    iteration = 0
    while True:
        print("\n--- Current Parameters ---")
        print(json.dumps(params, indent=2))
        cmd = input("Enter command (play / key=value / exit): ").strip()
        if cmd.lower() == "exit":
            break
        elif cmd.lower() == "play":
            run_performance(params, iteration)
            iteration += 1
        else:
            if "=" in cmd:
                k, v = cmd.split("=", 1)
                k = k.strip()
                v = v.strip()
                # Try to cast numeric
                try:
                    if "." in v:
                        params[k] = float(v)
                    else:
                        params[k] = int(v)
                except ValueError:
                    params[k] = v
            else:
                print("Unrecognized command. Use 'play', 'exit', or key=value.")


if __name__ == "__main__":
    main()
