#!/usr/bin/env python3
"""
Lightweight datasources for protein folding QA without external dependencies.

Provides:
- fetch_pdb(pdb_id): Download a PDB file (RCSB) as text.
- parse_pdb_bfactors(pdb_text, chain=None): Parse ATOM lines and aggregate per-residue
  average B-factors by residue (resseq+insertion code). Returns a list of tuples
  (res_id, bfactor_mean).
- windowize(values, size=48, stride=16): Iterate windows of fixed size with given stride.
- bfactors_to_peptides(values): Map B-factor-like values to [0,1] per-residue scores
  where lower B (more confident/ordered) -> higher score.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple
import numpy as np

import torch

RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def fetch_pdb(pdb_id: str) -> str:
    url = RCSB_PDB_URL.format(pdb_id=pdb_id.upper())
    with urllib.request.urlopen(url) as resp:
        data = resp.read().decode("utf-8", errors="replace")
    return data


def read_pdb(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="replace")


def parse_pdb_bfactors(pdb_text: str, chain: Optional[str] = None) -> List[Tuple[str, float]]:
    """
    Parse ATOM records from PDB text and compute per-residue average B-factors.
    Returns a sorted list [(res_uid, mean_bfactor), ...] in residue order.

    res_uid is f"{chain}{resseq}{icode}" for uniqueness.
    """
    residues = []  # list of (res_uid, bfactor)
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        # Fixed-column PDB parsing (1-based indices per PDB spec)
        # https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html#ATOM
        try:
            line = line.rstrip("\n")
            chain_id = line[21].strip()
            if chain and chain_id != chain:
                continue
            resseq = line[22:26].strip()
            icode = line[26].strip()  # insertion code
            bfac_str = line[60:66].strip()
            b = float(bfac_str) if bfac_str else 0.0
            res_uid = f"{chain_id}{resseq}{icode}"
            residues.append((res_uid, b))
        except Exception:
            continue

    # Aggregate by residue uid
    if not residues:
        return []
    agg: List[Tuple[str, List[float]]] = []
    cur_uid = residues[0][0]
    cur_vals: List[float] = []
    for uid, b in residues:
        if uid != cur_uid:
            agg.append((cur_uid, cur_vals))
            cur_uid = uid
            cur_vals = [b]
        else:
            cur_vals.append(b)
    agg.append((cur_uid, cur_vals))

    out: List[Tuple[str, float]] = [(uid, sum(vals) / max(1, len(vals))) for uid, vals in agg]
    return out


def windowize(values: Sequence[float], size: int = 48, stride: int = 16) -> Iterator[Tuple[int, Sequence[float]]]:
    n = len(values)
    i = 0
    while i + size <= n:
        yield i, values[i:i+size]
        i += stride


def pad_to_length(values: Sequence[float], target: int = 48, mode: str = "edge") -> List[float]:
    """
    Pad a short sequence to `target` length.
    - mode="edge": repeat the last value to reach target length.
    - mode="mirror": reflect around the end.
    """
    vals = list(values)
    n = len(vals)
    if n >= target:
        return vals[:target]
    if n == 0:
        return [0.0] * target
    if mode == "edge":
        last = vals[-1]
        return vals + [last] * (target - n)
    elif mode == "mirror":
        extra: List[float] = []
        idx = list(range(n - 2, -1, -1))  # mirror excluding the last element
        while len(vals) + len(extra) < target:
            for j in idx:
                extra.append(vals[j])
                if len(vals) + len(extra) >= target:
                    break
        return vals + extra[: (target - n)]
    else:
        raise ValueError("Unsupported padding mode")


def _minmax(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mn = x.min()
    mx = x.max()
    return (x - mn) / (mx - mn + eps)


def bfactors_to_peptides(bfactors: Sequence[float], mode: str = "b_factor") -> torch.Tensor:
    """
    Map B-like values to [0,1] per-residue wholeness scores.
    Modes:
    - 'b_factor': higher B -> lower score (score = 1 - norm(B))
    - 'plddt': higher pLDDT -> higher score (score = clamp(plddt/100,0,1))
    - 'auto': detect pLDDT if max<=100, else use b_factor
    """
    t = torch.tensor(bfactors, dtype=torch.float32)
    use_mode = mode
    if mode == "auto":
        mx = float(t.max().item()) if t.numel() else 0.0
        use_mode = "plddt" if mx <= 100.0 else "b_factor"
    if use_mode == "plddt":
        score = (t / 100.0).clamp(0.0, 1.0)
    elif use_mode == "b_factor":
        t_norm = _minmax(t)
        score = 1.0 - t_norm
    else:
        raise ValueError("bfactors_to_peptides: unknown mode")
    return score.to(dtype=torch.float32).clamp(0.0, 1.0)


def parse_pdb_ca_coords(pdb_text: str, chain: Optional[str] = None) -> np.ndarray:
    """Parse CA atom coordinates from PDB text for an optional chain.
    Returns an array of shape (L,3) in residue order.
    """
    ca_records: List[Tuple[str, float, float, float]] = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            atom_name = line[12:16]
            if atom_name.strip() != "CA":
                continue
            chain_id = line[21].strip()
            if chain and chain_id != chain:
                continue
            resseq = line[22:26].strip()
            icode = line[26].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            res_uid = f"{chain_id}{resseq}{icode}"
            ca_records.append((res_uid, x, y, z))
        except Exception:
            continue
    if not ca_records:
        return np.zeros((0, 3), dtype=np.float32)
    # Preserve file order which is already residue order for a chain
    coords = np.array([[x, y, z] for (_uid, x, y, z) in ca_records], dtype=np.float32)
    return coords
