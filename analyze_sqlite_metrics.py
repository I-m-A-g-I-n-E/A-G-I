#!/usr/bin/env python3
"""
Quick SQLite-based metrics analyzer for protein refinement batches.
Similar to tools/analyze_metrics.py but works directly with the SQLite DB.
"""

import argparse
import sqlite3
from pathlib import Path

def analyze_db(db_path):
    """Analyze the metrics database and provide insights."""
    if not Path(db_path).exists():
        print(f"Error: Database {db_path} not found")
        return
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Get improvement analysis
    query = """
    SELECT 
        i.output_base,
        ROUND(i.tm_score_ca, 6) as initial_tm,
        ROUND(r.tm_score_ca, 6) as refined_tm,
        ROUND(r.tm_score_ca - i.tm_score_ca, 9) as tm_improvement,
        ROUND(i.orthogonality_index, 6) as initial_orth, 
        ROUND(r.orthogonality_index, 6) as refined_orth,
        ROUND(r.orthogonality_index - i.orthogonality_index, 9) as orth_improvement,
        i.num_clashes as initial_clashes,
        r.num_clashes as refined_clashes,
        (r.num_clashes - i.num_clashes) as clash_change,
        ROUND(r.t_refine_s, 2) as refine_time_s,
        ROUND(r.min_ca_ca, 3) as min_ca_ca_refined,
        ROUND(i.min_ca_ca, 3) as min_ca_ca_initial,
        ROUND(r.min_ca_ca - i.min_ca_ca, 6) as min_ca_improvement
    FROM runs i 
    JOIN runs r ON i.output_base = r.output_base 
    WHERE i.stage = 'initial' AND r.stage = 'refined'
    ORDER BY orth_improvement DESC, tm_improvement DESC;
    """
    
    results = conn.execute(query).fetchall()
    
    print("="*80)
    print("PROTEIN REFINEMENT BATCH ANALYSIS")
    print("="*80)
    
    print(f"\nAnalyzing {len(results)} runs from {db_path}\n")
    
    # Summary stats
    total_time = sum(r['refine_time_s'] for r in results)
    avg_time = total_time / len(results) if results else 0
    
    tm_improvements = [r['tm_improvement'] for r in results]
    orth_improvements = [r['orth_improvement'] for r in results]  
    clash_changes = [r['clash_change'] for r in results]
    
    print("SUMMARY:")
    print(f"  Total refinement time: {total_time:.1f}s ({avg_time:.1f}s avg)")
    print(f"  TM-score improvements: {min(tm_improvements):.2e} to {max(tm_improvements):.2e}")
    print(f"  Orthogonality improvements: {min(orth_improvements):.6f} to {max(orth_improvements):.6f}")
    print(f"  Clash changes: {min(clash_changes)} to {max(clash_changes)}")
    
    nonzero_orth = sum(1 for x in orth_improvements if abs(x) > 1e-6)
    nonzero_tm = sum(1 for x in tm_improvements if abs(x) > 1e-8)
    nonzero_clash = sum(1 for x in clash_changes if x != 0)
    
    print(f"  Runs with meaningful orthogonality gain: {nonzero_orth}/{len(results)}")
    print(f"  Runs with meaningful TM gain: {nonzero_tm}/{len(results)}")
    print(f"  Runs with clash reduction: {nonzero_clash}/{len(results)}")
    
    print(f"\nDETAILED RESULTS:")
    print("%-15s %8s %8s %10s %8s %8s %10s %6s %6s %8s %8s" % (
        "Run", "Init_TM", "Ref_TM", "TM_Δ", "Init_O", "Ref_O", "Orth_Δ", 
        "Clash", "ClΔ", "MinCA_Δ", "Time"
    ))
    print("-" * 100)
    
    for r in results:
        base = r['output_base'].replace('outputs/', '').replace('ubi_', '')
        print("%-15s %8.5f %8.5f %10.2e %8.5f %8.5f %10.6f %6d %6d %8.5f %8.1f" % (
            base, r['initial_tm'], r['refined_tm'], r['tm_improvement'],
            r['initial_orth'], r['refined_orth'], r['orth_improvement'],
            r['initial_clashes'], r['clash_change'], r['min_ca_improvement'], r['refine_time_s']
        ))
    
    # Rate analysis
    print(f"\nIMPROVEMENT RATES (per second):")
    for r in results:
        base = r['output_base'].replace('outputs/', '').replace('ubi_', '')
        if r['refine_time_s'] > 0:
            orth_rate = r['orth_improvement'] / r['refine_time_s']
            tm_rate = r['tm_improvement'] / r['refine_time_s']
            print(f"  {base}: orth={orth_rate:.2e}/s, tm={tm_rate:.2e}/s")
    
    conn.close()
    
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS:")
    if nonzero_orth == 0:
        print("❌ Zero orthogonality improvements detected!")
        print("   - Increase --refine-clash-weight (try 15.0+)")
        print("   - Increase --refine-step-clash (try 5.0+)")  
        print("   - Increase --refine-final-step (try 8.0+)")
        print("   - Try segment-based moves (6-12 residue windows)")
    
    if nonzero_clash == 0:
        print("❌ Zero clash reductions detected!")
        print("   - Check clash detection threshold")
        print("   - Try steric-only pre-pass with higher repulsion")
        
    if avg_time > 40:
        print("⚠️  Refinement time is high")
        print("   - Consider reducing --refine-iters or early stopping")
        
    if nonzero_tm == 0:
        print("❌ No meaningful TM-score improvements")  
        print("   - Current refinement may be too conservative")
        print("   - Try larger perturbations in early phases")
        
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze SQLite metrics database")
    parser.add_argument("--db", default="outputs/metrics.db", help="Path to SQLite database")
    args = parser.parse_args()
    
    analyze_db(args.db)
