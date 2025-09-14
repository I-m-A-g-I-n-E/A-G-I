#!/usr/bin/env python3
import csv
import os
from collections import defaultdict

def parse_float(x):
    try:
        return float(x)
    except Exception:
        return None

def load_metrics(csv_path):
    rows = []
    with open(csv_path, newline='') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    return rows

def aggregate(rows):
    by_base = defaultdict(dict)
    for r in rows:
        base = r.get('output_base','')
        stage = r.get('stage','')
        by_base[base][stage] = r
    out = []
    for base, stages in by_base.items():
        init = stages.get('initial')
        ref = stages.get('refined')
        if not init:
            continue
        # Parse values
        init_orth = parse_float(init.get('orthogonality_index'))
        ref_orth = parse_float(ref.get('orthogonality_index')) if ref else None
        init_clashes = parse_float(init.get('num_clashes'))
        ref_clashes = parse_float(ref.get('num_clashes')) if ref else None
        init_tm = parse_float(init.get('tm_score_ca'))
        ref_tm = parse_float(ref.get('tm_score_ca')) if ref else None
        t_build = parse_float(ref.get('t_build_s') if ref else init.get('t_build_s')) or 0.0
        t_qc_initial = parse_float(ref.get('t_qc_initial_s') if ref else init.get('t_qc_initial_s')) or 0.0
        t_refine = parse_float(ref.get('t_refine_s')) if ref else 0.0
        t_qc_refined = parse_float(ref.get('t_qc_refined_s')) if ref else 0.0
        t_total = (t_build or 0.0) + (t_qc_initial or 0.0) + (t_refine or 0.0) + (t_qc_refined or 0.0)
        # Derived metrics
        orth_gain = None
        orth_gain_rate = None
        clash_red = None
        clash_red_rate = None
        tm_gain = None
        tm_gain_rate = None
        if ref and ref_orth is not None and init_orth is not None:
            orth_gain = ref_orth - init_orth
            orth_gain_rate = orth_gain / t_total if t_total else None
        if ref and ref_clashes is not None and init_clashes is not None:
            clash_red = init_clashes - ref_clashes
            clash_red_rate = clash_red / t_total if t_total else None
        if ref and ref_tm is not None and init_tm is not None:
            tm_gain = ref_tm - init_tm
            tm_gain_rate = tm_gain / t_total if t_total else None
        out.append({
            'output_base': base,
            't_total_s': t_total,
            'init_orth': init_orth,
            'ref_orth': ref_orth,
            'orth_gain': orth_gain,
            'orth_gain_per_s': orth_gain_rate,
            'init_clashes': init_clashes,
            'ref_clashes': ref_clashes,
            'clash_reduction': clash_red,
            'clash_reduction_per_s': clash_red_rate,
            'init_tm': init_tm,
            'ref_tm': ref_tm,
            'tm_gain': tm_gain,
            'tm_gain_per_s': tm_gain_rate,
        })
    return out

def main():
    csv_path = os.path.expanduser('outputs/metrics.csv')
    if not os.path.exists(csv_path):
        print(f"No metrics CSV at {csv_path}. Run generate_structure.py with --metrics-csv first.")
        return
    rows = load_metrics(csv_path)
    agg = aggregate(rows)
    # Sort by orthogonality gain per second (desc), fallback to clash reduction per second
    def sort_key(r):
        og = r.get('orth_gain_per_s')
        cr = r.get('clash_reduction_per_s')
        return (og if og is not None else float('-inf'), cr if cr is not None else float('-inf'))
    agg.sort(key=sort_key, reverse=True)
    # Print a compact report
    def fnum(x, fmt="{:.3f}"):
        return fmt.format(x) if isinstance(x, (int, float)) and x is not None else "NA"
    print("\n=== Metrics Performance Summary ===")
    for r in agg:
        out_base = r.get('output_base', '')
        t_total = fnum(r.get('t_total_s'))
        og = r.get('orth_gain')
        ogr = r.get('orth_gain_per_s')
        cr = r.get('clash_reduction')
        crr = r.get('clash_reduction_per_s')
        tg = r.get('tm_gain')
        tgr = r.get('tm_gain_per_s')
        print(f"{out_base}: t_total={t_total}s, "
              f"orth_gain={fnum(og)} ({fnum(ogr, '{:.4g}')}/s), "
              f"clash_red={fnum(cr)} ({fnum(crr, '{:.4g}')}/s), "
              f"tm_gain={fnum(tg)} ({fnum(tgr, '{:.4g}')}/s)")

if __name__ == '__main__':
    main()
