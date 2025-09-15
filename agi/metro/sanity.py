"""
Sanity audit utilities for measuring contextualization of values in modules.
Simple heuristics: counts numeric literals vs. named constants usage ratio.
"""
from __future__ import annotations
from pathlib import Path
import re
from typing import Dict

NUMERIC_LITERAL_RE = re.compile(r"(?<![\w.])(\d+\.?\d*|\.\d+)(?![\w.])")
CONTEXT_WORDS = (
    'Laws.', 'IDEAL_GEOMETRY', 'REFINEMENT_POLICY', 'Handedness', 'Gesture', 'Movement',
    'DYADIC', 'TRIADIC', 'MANIFOLD', 'CA-C', 'N-CA', 'C-N',
)


def count_naked_numbers(text: str) -> int:
    # crude heuristic: numbers not part of comments with context annotations
    # remove obvious floats in scientific notation if part of arrays? keep simple
    return len(NUMERIC_LITERAL_RE.findall(text))


def count_contextualized_values(text: str) -> int:
    return sum(text.count(w) for w in CONTEXT_WORDS)


def audit_sanity(module_path: str | Path) -> Dict:
    p = Path(module_path)
    if not p.exists() or not p.is_file():
        return {'module': str(p), 'naked_values': 0, 'sanity_score': 1.0}
    try:
        text = p.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return {'module': str(p), 'naked_values': 0, 'sanity_score': 1.0}
    naked = count_naked_numbers(text)
    anchored = count_contextualized_values(text)
    denom = naked + anchored
    sanity = (anchored / denom) if denom > 0 else 1.0
    return {
        'module': str(p),
        'naked_values': naked,
        'sanity_score': sanity,
    }
