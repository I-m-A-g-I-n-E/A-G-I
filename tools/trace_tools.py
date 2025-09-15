#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

Event = Dict[str, Any]


def load_events(path: Path) -> List[Event]:
    events: List[Event] = []
    with path.open('r') as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] skipping line {ln}: {e}", file=sys.stderr)
    return events


def fmt_secs(x: float | None) -> str:
    return f"{x:.3f}s" if isinstance(x, (int, float)) else "-"


def summarize(events: List[Event]) -> None:
    if not events:
        print("No events found.")
        return

    first_ts = events[0].get('ts')
    last_ts = events[-1].get('ts')

    # Buckets
    counts = {}
    for e in events:
        evt = e.get('event', 'unknown')
        counts[evt] = counts.get(evt, 0) + 1

    start = next((e for e in events if e.get('event') == 'start'), None)
    spacing = next((e for e in events if e.get('event') == 'spacing_done'), None)
    end = next((e for e in events if e.get('event') == 'end'), None)
    timeouts = [e for e in events if e.get('event') == 'timeout_abort']

    print("=== Trace Summary ===")
    print(f"Total events: {len(events)}  |  Duration: {fmt_secs((last_ts or 0) - (first_ts or 0))}")
    for k in sorted(counts.keys()):
        print(f"  {k}: {counts[k]}")

    if start:
        qc0 = start.get('qc_initial', {})
        print("\nInitial QC:")
        print(f"  min_ca_ca={qc0.get('min_ca_ca')}  num_clashes={qc0.get('num_clashes')}  orth_idx={qc0.get('orthogonality_index')}")

    if spacing:
        print("\nSpacing Pass:")
        print(f"  attempts={spacing.get('spacing_attempts')}  bins_tried={spacing.get('bins_tried')}  final_min_ca_ca={spacing.get('final_min_ca_ca')}")

    if end:
        qcf = end.get('final_qc', {})
        print("\nFinal QC:")
        print(f"  min_ca_ca={qcf.get('min_ca_ca')}  num_clashes={qcf.get('num_clashes')}  orth_idx={qcf.get('orthogonality_index')}")

    if timeouts:
        print("\nTimeouts:")
        for t in timeouts:
            print(f"  phase={t.get('phase')}  iter={t.get('iter') or t.get('attempts2')}  at_ts={t.get('ts')}")

    # Acceptance stats per phase
    def phase_acceptance(phase_key: str) -> None:
        evname = 'phaseA_iter' if phase_key == 'A' else 'phaseB_iter'
        phase = [e for e in events if e.get('event') == evname]
        if not phase:
            return
        acc = sum(1 for e in phase if e.get('accepted'))
        print(f"\nPhase {phase_key} acceptance: {acc}/{len(phase)} ({(100.0*acc/len(phase)):.1f}%)")

    phase_acceptance('A')
    phase_acceptance('B')


def timeline(events: List[Event], phase: str | None, limit: int | None) -> None:
    shown = 0
    for e in events:
        if limit is not None and shown >= limit:
            break
        evt = e.get('event')
        if phase == 'A' and evt != 'phaseA_iter':
            continue
        if phase == 'B' and evt != 'phaseB_iter':
            continue
        if phase == 'spacing' and evt not in ('spacing_done', 'timeout_abort'):
            continue
        if phase == 'final' and evt != 'final_repair_iter':
            continue
        print(json.dumps(e))
        shown += 1


def main():
    ap = argparse.ArgumentParser(description="Refinement trace analyzer (JSONL)")
    ap.add_argument('trace', type=Path, help='Path to JSONL trace file')
    ap.add_argument('--summary', action='store_true', help='Print summary')
    ap.add_argument('--timeline', action='store_true', help='Print event timeline')
    ap.add_argument('--phase', choices=['A','B','spacing','final'], help='Filter timeline to a phase')
    ap.add_argument('--limit', type=int, default=None, help='Limit timeline entries')
    args = ap.parse_args()

    if not args.trace.exists():
        print(f"Trace file not found: {args.trace}", file=sys.stderr)
        sys.exit(1)

    events = load_events(args.trace)
    if args.summary or (not args.summary and not args.timeline):
        summarize(events)
    if args.timeline:
        timeline(events, args.phase, args.limit)


if __name__ == '__main__':
    main()
