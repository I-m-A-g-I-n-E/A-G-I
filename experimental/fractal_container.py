#!/usr/bin/env python3
"""
Fractal48 reversible container for lossless storage/transport of arbitrary bytes.

Core ideas:
- Exact permutations (3x3 then 2x2x2) via Fractal48Layer.space_to_depth_3/2
- Inversion via depth_to_space_2/3 for perfect recovery
- Blocks are fixed-size 48x48xC uint8 tensors (default C=3)
- Any trailing bytes that do not fill a whole block are treated as "poop" (pixels out of place)
  and cached in the container header so decoding can restore the exact original stream.

This module provides:
- encode_bytes(data: bytes, channels=3, ops=("s3","s2")) -> container_bytes
- decode_bytes(container_bytes: bytes) -> data: bytes
- encode_path(input_path, output_path, channels=3, ops=("s3","s2"))
- decode_path(input_path, output_path)

Container format (little-endian):
  magic:    b'F48C' (4 bytes)
  version:  uint16 (currently 1)
  hdr_len:  uint32 (byte length of JSON header)
  header:   JSON utf-8 dict with fields listed below
  payload:  raw bytes of concatenated permuted blocks (uint8), size = num_blocks * C*48*48

Header JSON fields:
  {
    "channels": int,
    "ops": ["s3", "s2"],  # sequence of permutations applied in encode order
    "block_h": 48,
    "block_w": 48,
    "dtype": "uint8",
    "num_blocks": int,
    "poop_len": int,  # number of trailing input bytes not encoded into full blocks
    "poop_b64": str    # base64 of trailing bytes
  }

Notes:
- We keep the pipeline purely in uint8 to avoid any numeric drift.
- You can extend this by adding an optional reversible mixing stage on even channel counts
  using floating-point and the provided lifting ops, but then you'd need quantization. For now,
  we keep it strictly permutation-only for exactness and simplicity.
"""
from __future__ import annotations

import base64
import io
import json
import os
import struct
from typing import Iterable, List, Sequence, Tuple

import torch

from fractal48_torch import Fractal48Layer

MAGIC = b"F48C"
VERSION = 1
BLOCK_H = 48
BLOCK_W = 48


def _block_capacity(channels: int) -> int:
    return channels * BLOCK_H * BLOCK_W  # bytes per block (uint8)


def _apply_ops_uint8(x: torch.Tensor, ops: Sequence[str]) -> torch.Tensor:
    """Apply a sequence of permutation ops to a uint8 tensor.

    x: (B, C, H, W), dtype=uint8, device=cpu
    ops: elements from {"s3", "s2"} where:
      - "s3" == space_to_depth_3
      - "s2" == space_to_depth_2

    Returns tensor with the same number of elements (pure permutation).
    """
    assert x.dtype == torch.uint8
    for op in ops:
        if op == "s3":
            x = Fractal48Layer.space_to_depth_3(x)
        elif op == "s2":
            x = Fractal48Layer.space_to_depth_2(x)
        else:
            raise ValueError(f"Unsupported op: {op}")
    return x


def _invert_ops_uint8(x: torch.Tensor, ops: Sequence[str]) -> torch.Tensor:
    """Invert a sequence of permutation ops on a uint8 tensor.

    ops must be the same list used in encode; we apply the exact inverses in reverse order.
    """
    assert x.dtype == torch.uint8
    for op in reversed(list(ops)):
        if op == "s2":
            x = Fractal48Layer.depth_to_space_2(x)
        elif op == "s3":
            x = Fractal48Layer.depth_to_space_3(x)
        else:
            raise ValueError(f"Unsupported op in inverse: {op}")
    return x


def encode_bytes(data: bytes, channels: int = 3, ops: Sequence[str] = ("s3", "s2")) -> bytes:
    """Encode arbitrary bytes into a reversible Fractal48 container.

    - Packs data into (B, C, 48, 48) uint8 blocks
    - Applies pure permutations listed in `ops`
    - Stores any trailing leftover (poop) bytes in the header for exact recovery
    """
    if channels < 1:
        raise ValueError("channels must be >= 1")

    cap = _block_capacity(channels)
    total = len(data)
    num_full_blocks = total // cap
    poop_len = total % cap

    # Separate full-block payload and poop bytes
    payload = memoryview(data)[: num_full_blocks * cap]
    poop = bytes(memoryview(data)[num_full_blocks * cap :])  # may be empty

    # Prepare tensor blocks
    if num_full_blocks > 0:
        t = torch.frombuffer(payload, dtype=torch.uint8).clone()  # (N,)
        t = t.view(num_full_blocks, channels, BLOCK_H, BLOCK_W)
        t = _apply_ops_uint8(t, ops)
        # Flatten back to bytes
        t_bytes = t.contiguous().view(-1).numpy().tobytes()
    else:
        t_bytes = b""

    header = {
        "channels": channels,
        "ops": list(ops),
        "block_h": BLOCK_H,
        "block_w": BLOCK_W,
        "dtype": "uint8",
        "num_blocks": num_full_blocks,
        "poop_len": poop_len,
        "poop_b64": base64.b64encode(poop).decode("ascii") if poop_len > 0 else "",
    }
    hdr_json = json.dumps(header, separators=(",", ":")).encode("utf-8")

    # Assemble container
    out = io.BytesIO()
    out.write(MAGIC)
    out.write(struct.pack("<H", VERSION))
    out.write(struct.pack("<I", len(hdr_json)))
    out.write(hdr_json)
    out.write(t_bytes)
    return out.getvalue()


def decode_bytes(container: bytes) -> bytes:
    """Decode a Fractal48 container to recover the original byte stream exactly."""
    bio = io.BytesIO(container)
    magic = bio.read(4)
    if magic != MAGIC:
        raise ValueError("Invalid container magic")
    ver_bytes = bio.read(2)
    if len(ver_bytes) != 2:
        raise ValueError("Truncated container (version)")
    (version,) = struct.unpack("<H", ver_bytes)
    if version != VERSION:
        raise ValueError(f"Unsupported version: {version}")

    hdr_len_bytes = bio.read(4)
    if len(hdr_len_bytes) != 4:
        raise ValueError("Truncated container (header length)")
    (hdr_len,) = struct.unpack("<I", hdr_len_bytes)
    hdr_json = bio.read(hdr_len)
    if len(hdr_json) != hdr_len:
        raise ValueError("Truncated container (header)")

    header = json.loads(hdr_json)
    channels = int(header["channels"])  # type: ignore[arg-type]
    ops = list(header.get("ops", ["s3", "s2"]))
    num_blocks = int(header["num_blocks"])  # type: ignore[arg-type]
    poop_len = int(header.get("poop_len", 0))  # type: ignore[arg-type]
    poop_b64 = header.get("poop_b64", "")
    poop = base64.b64decode(poop_b64) if poop_b64 else b""
    if len(poop) != poop_len:
        raise ValueError("Corrupt poop in header: length mismatch")

    cap = _block_capacity(channels)
    expected_payload_bytes = num_blocks * cap

    payload_bytes = bio.read()
    if len(payload_bytes) != expected_payload_bytes:
        raise ValueError("Truncated container (payload)")

    if num_blocks > 0:
        t = torch.frombuffer(memoryview(payload_bytes), dtype=torch.uint8).clone()
        t = t.view(num_blocks, channels, BLOCK_H, BLOCK_W)
        t = _invert_ops_uint8(t, ops)
        restored = t.contiguous().view(-1).numpy().tobytes()
    else:
        restored = b""

    return restored + poop


def encode_path(input_path: str, output_path: str, channels: int = 3, ops: Sequence[str] = ("s3", "s2")) -> None:
    with open(input_path, "rb") as f:
        data = f.read()
    encoded = encode_bytes(data, channels=channels, ops=ops)
    with open(output_path, "wb") as f:
        f.write(encoded)


def decode_path(input_path: str, output_path: str) -> None:
    with open(input_path, "rb") as f:
        container = f.read()
    decoded = decode_bytes(container)
    with open(output_path, "wb") as f:
        f.write(decoded)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fractal48 reversible container")
    sub = parser.add_subparsers(dest="cmd", required=True)

    enc = sub.add_parser("encode", help="Encode a file into a Fractal48 container")
    enc.add_argument("input", help="Path to input file")
    enc.add_argument("output", help="Path to output .f48 file")
    enc.add_argument("--channels", type=int, default=3, help="Channels per block (default: 3)")
    enc.add_argument(
        "--ops",
        nargs="*",
        default=["s3", "s2"],
        help='Permutation ops in order (subset of {"s3","s2"}); default: s3 s2',
    )

    dec = sub.add_parser("decode", help="Decode a Fractal48 container back to original bytes")
    dec.add_argument("input", help="Path to input .f48 file")
    dec.add_argument("output", help="Path to decoded output file")

    args = parser.parse_args()

    if args.cmd == "encode":
        encode_path(args.input, args.output, channels=args.channels, ops=args.ops)
    elif args.cmd == "decode":
        decode_path(args.input, args.output)
