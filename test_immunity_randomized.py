#!/usr/bin/env python3
"""
Randomized robustness tests for immunity.py separated from core system code.
Run directly as a script or via pytest.
"""
from typing import Dict
import argparse
import pytest

from immunity import (
    run_randomized_trials,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run randomized immunity tests")
    parser.add_argument("--trials", type=int, default=42, help="Number of randomized trials")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    args = parser.parse_args()

    results: Dict[str, int] = run_randomized_trials(trials=args.trials, seed=args.seed)

    # Basic sanity checks (non-fatal):
    total = results.get("trials", 0)
    for k in [
        "self_accept_pass",
        "misfold_reject_pass",
        "foreign_detect_pass",
        "complement_full_pass",
        "complement_partial_pass",
    ]:
        v = results.get(k, -1)
        if not (0 <= v <= total):
            print(f"Warning: result out of bounds: {k}={v} (total={total})")


# Optional pytest entrypoint

@pytest.mark.slow
def test_randomized_trials_smoke():
    """Smoke test: ensure randomized trials run and produce bounded counts."""
    results = run_randomized_trials(trials=5, seed=123)
    total = results["trials"]
    assert total == 5
    assert 0 <= results["self_accept_pass"] <= total
    assert 0 <= results["misfold_reject_pass"] <= total
    assert 0 <= results["foreign_detect_pass"] <= total
    assert 0 <= results["complement_full_pass"] <= total
    assert 0 <= results["complement_partial_pass"] <= total


if __name__ == "__main__":
    main()
