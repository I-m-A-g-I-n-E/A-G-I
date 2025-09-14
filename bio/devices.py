"""
Device selection and safe tensor operations across CPU, CUDA, and MPS.

Centralizes backend quirks so the rest of the codebase can stay simple.

Usage:
    from bio.devices import (
        get_device, to_device, module_to_device,
        is_float64_supported, svd_safe, same_device,
    )

Environment overrides:
    AGI_DEVICE=cpu|cuda|mps   # force a backend if available
    AGI_FORCE_CPU=1           # hard override to CPU

Notes on MPS (Apple Silicon):
    - float64 (torch.float64) is not supported; prefer float32
    - torch.linalg.svd falls back to CPU and can be slow; use svd_safe
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterable, Optional, Tuple

import torch

# ---------------------------------------------------------
# Device selection
# ---------------------------------------------------------

def get_device() -> torch.device:
    """Select an execution device with sane defaults and env overrides.

    Priority (unless overridden): CUDA > MPS > CPU.
    """
    if os.getenv("AGI_FORCE_CPU"):
        return torch.device("cpu")

    env_dev = os.getenv("AGI_DEVICE", "").strip().lower()
    if env_dev == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if env_dev == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if env_dev == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_float64_supported(device: torch.device) -> bool:
    """Whether float64 tensors are supported on the given device.

    - CUDA: yes
    - CPU: yes
    - MPS: no
    """
    if device.type == "mps":
        return False
    return True


# ---------------------------------------------------------
# Utilities for placing tensors and modules
# ---------------------------------------------------------

def to_device(x, device: Optional[torch.device] = None):
    """Move a tensor or a nested structure of tensors to device.

    Works for torch.Tensor and objects exposing .to(device).
    """
    if device is None:
        device = get_device()
    if hasattr(x, "to"):
        return x.to(device)
    # Fallback: try iterables
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(t, device) for t in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x  # unknown type; return as-is


def module_to_device(module: torch.nn.Module, device: Optional[torch.device] = None) -> torch.nn.Module:
    if device is None:
        device = get_device()
    return module.to(device)


def same_device(*tensors: torch.Tensor, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, ...]:
    """Ensure all tensors are on the same device (default: get_device())."""
    if device is None:
        device = get_device()
    return tuple(t.to(device) for t in tensors)


# ---------------------------------------------------------
# Safe numerical ops that account for backend quirks
# ---------------------------------------------------------

def svd_safe(x: torch.Tensor, full_matrices: bool = False):
    """Perform SVD robustly across backends.

    - On MPS, torch.linalg.svd falls back to CPU and can be very slow.
      We move the computation to CPU explicitly to avoid device mismatch.
    - On CUDA/CPU, use the native op.
    Returns (U, S, Vh).
    """
    dev = x.device
    if dev.type == "mps":
        xc = x.detach().to("cpu")
        U, S, Vh = torch.linalg.svd(xc, full_matrices=full_matrices)
        # Move results back to original device if needed
        return U.to(dev), S.to(dev), Vh.to(dev)
    else:
        return torch.linalg.svd(x, full_matrices=full_matrices)


@contextmanager
def device_ctx(device: Optional[torch.device] = None):
    """Context manager to set default device for ops that respect torch.set_default_device.

    Note: Many tensor factory functions respect the default device in modern PyTorch.
    We still recommend explicitly passing device where feasible.
    """
    if device is None:
        device = get_device()
    try:
        prev = torch.empty(0).device  # probe
        # torch.set_default_device is available in newer versions; guard if absent
        set_default = getattr(torch, "set_default_device", None)
        if callable(set_default):
            set_default(str(device))
        yield device
    finally:
        # Reset to CPU default if function exists
        set_default = getattr(torch, "set_default_device", None)
        if callable(set_default):
            set_default("cpu")
