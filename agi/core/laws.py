"""
Harmonia Laws: central constants and symbolic gestures for the 48-manifold system.
"""
from dataclasses import dataclass
from typing import Dict

# Core manifold facts
class Laws:
    # 48 = 3 * 2^4
    MANIFOLD_DIM: int = 48
    DYADIC_BASE: int = 16   # 2^4
    TRIADIC_BASE: int = 3
    # Useful masks and multipliers
    DYADIC_MASK: int = DYADIC_BASE - 1  # 0b1111
    # After 3×3 then 2×2, 2×2, 2×2 permutations: channels scale by 9 * 4 * 4 * 4 = 576
    BOTTLENECK_MULT: int = (TRIADIC_BASE ** 2) * (2 ** 6)


@dataclass(frozen=True)
class GestureShift:
    name: str
    shift: int

# Canonical lifting intents for integer lifting steps
GESTURES: Dict[str, GestureShift] = {
    'HARMONIC': GestureShift('HARMONIC', 1),
    'TENSIVE': GestureShift('TENSIVE', 2),
}
