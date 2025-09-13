#!/usr/bin/env python3
from __future__ import annotations

import os
import json
from typing import Tuple

import numpy as np
import torch


def ensure_dir_for(path: str) -> None:
    """Ensure the directory for a file path exists."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def load_ensemble(prefix: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load ensemble mean and certainty arrays for a given prefix.

    Returns:
        mean: torch.FloatTensor with shape [W, 48]
        certainty: torch.FloatTensor with shape [W]
    """
    mean_path = f"{prefix}_mean.npy"
    cert_path = f"{prefix}_certainty.npy"

    if not os.path.exists(mean_path):
        raise FileNotFoundError(f"Mean composition file not found: {mean_path}")

    mean = torch.from_numpy(np.load(mean_path)).to(torch.float32)
    if mean.ndim == 1 and mean.shape[0] == 48:
        mean = mean.unsqueeze(0)

    if os.path.exists(cert_path):
        cert = torch.from_numpy(np.load(cert_path)).to(torch.float32).view(-1)
    else:
        # Default certainty to 1.0 per window
        cert = torch.ones((mean.shape[0],), dtype=torch.float32)

    return mean, cert


def save_json(data: dict, path: str) -> None:
    ensure_dir_for(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
