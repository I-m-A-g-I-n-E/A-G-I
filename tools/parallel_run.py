#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


def build_cmd(args, seed: int) -> str:
    input_prefix = args.input_template.format(seed=seed)
    output_pdb = args.output_template.format(seed=seed)
    cmd = [
        sys.executable, 'generate_structure.py',
        '--input-prefix', input_prefix,
        '--output-pdb', output_pdb,
        '--sequence-file', args.sequence_file,
        '--refine',
        '--refine-seed', str(seed),
        '--metrics-db', args.metrics_db,
    ]
    if args.ref_pdb:
        cmd += ['--ref-pdb', args.ref_pdb]
        if args.ref_chain:
            cmd += ['--ref-chain', args.ref_chain]
    # Optional advanced refinement params
    if args.refine_phaseA_frac is not None:
        cmd += ['--refine-phaseA-frac', str(args.refine_phaseA_frac)]
    if args.refine_step_clash is not None:
        cmd += ['--refine-step-clash', str(args.refine_step_clash)]
    if args.refine_clash_weight is not None:
        cmd += ['--refine-clash-weight', str(args.refine_clash_weight)]
    if args.refine_steric_only_phaseA:
        cmd += ['--refine-steric-only-phaseA']
    if args.refine_final_attempts is not None:
        cmd += ['--refine-final-attempts', str(args.refine_final_attempts)]
    if args.refine_final_step is not None:
        cmd += ['--refine-final-step', str(args.refine_final_step)]
    if args.refine_final_window_inc is not None:
        cmd += ['--refine-final-window-inc', str(args.refine_final_window_inc)]
    return ' '.join(shlex.quote(c) for c in cmd)


def run_one(cmd: str, cwd: str) -> tuple[int, str, str]:
    proc = subprocess.Popen(cmd, cwd=cwd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return proc.returncode, out, err


def main():
    ap = argparse.ArgumentParser(description='Parallel runner for generate_structure.py with SQLite metrics sink')
    ap.add_argument('--seeds', type=int, nargs='+', required=True, help='List of seeds to run')
    ap.add_argument('--input-template', type=str, required=True, help='Input prefix template, e.g. outputs/ubi_{seed}')
    ap.add_argument('--output-template', type=str, required=True, help='Output PDB template, e.g. outputs/ubi_{seed}.pdb')
    ap.add_argument('--sequence-file', type=str, required=True, help='Sequence file path')
    ap.add_argument('--ref-pdb', type=str, default=None, help='Reference PDB path')
    ap.add_argument('--ref-chain', type=str, default=None, help='Reference chain ID')
    ap.add_argument('--metrics-db', type=str, required=True, help='SQLite DB path to store metrics')
    ap.add_argument('--concurrency', type=int, default=4, help='Max parallel jobs')
    # Advanced refinement options
    ap.add_argument('--refine-phaseA-frac', type=float, default=None)
    ap.add_argument('--refine-step-clash', type=float, default=None)
    ap.add_argument('--refine-clash-weight', type=float, default=None)
    ap.add_argument('--refine-steric-only-phaseA', action='store_true')
    ap.add_argument('--refine-final-attempts', type=int, default=None)
    ap.add_argument('--refine-final-step', type=float, default=None)
    ap.add_argument('--refine-final-window-inc', type=int, default=None)
    args = ap.parse_args()

    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Launching {len(args.seeds)} jobs with concurrency={args.concurrency}")
    futures = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        for s in args.seeds:
            cmd = build_cmd(args, s)
            print(f" -> seed={s} cmd: {cmd}")
            futures.append((s, ex.submit(run_one, cmd, cwd)))
        for seed, fut in futures:
            rc, out, err = fut.result()
            print(f"\n===== seed {seed} completed (rc={rc}) =====")
            if out:
                print(out)
            if err:
                print('--- stderr ---')
                print(err)


if __name__ == '__main__':
    main()
