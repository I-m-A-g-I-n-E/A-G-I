import numpy as np
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Tuple

# --- The Fundamental Duality of Direction ---
class Handedness(Enum):
    """The chiral will of a movement."""
    RIGHT = 1  # The default, low-energy, clockwise path of nature
    LEFT = -1  # The inverted, high-energy, counter-clockwise path


# --- The Chiral Physics ---
@dataclass
class Turn:
    """An angle as a sane fraction of a turn, with an explicit handedness.

    value is measured in turns (1.0 == full circle). It may be signed to allow
    concise construction; use hand to indicate the desired chirality explicitly.
    """
    value: float  # magnitude in turns (can be signed; signed_value() applies hand)
    hand: Handedness = Handedness.RIGHT  # Default to the natural path

    def signed_value(self) -> float:
        """The effective value for calculations (signed)."""
        return float(self.value) * float(self.hand.value)

    def to_phase48(self) -> 'Phase48':
        u = self.signed_value() * 48.0
        tick_f = np.round(u)
        tick = int(tick_f) % 48
        micro = float(u - tick_f)
        return Phase48(tick, micro, self.hand)

    @property
    def fractal_complexity(self) -> float:
        """The cost to reality, which penalizes unnatural inversion.

        - Grid affinity: cheaper if representable with small-denominator fractions on 48-tick lattice
        - Micro-offset cost: off-grid fractional remainder
        - Chirality cost: LEFT-handed movements are penalized
        """
        try:
            denom = Fraction(self.value).limit_denominator(48).denominator
            base_complexity = np.log2(max(1, denom)) + abs(self.to_phase48().micro)
        except Exception:
            base_complexity = 1.0 + abs(self.to_phase48().micro)
        chirality_cost = 1.0 if self.hand == Handedness.RIGHT else 4.0
        return float(base_complexity * chirality_cost)


# --- Angles and distances with context ---
@dataclass
class Deg:
    """Angle in degrees with a semantic reference label (e.g., "C-N-CA")."""
    value: float
    reference: str = ""


@dataclass
class Angstrom:
    """Distance in Ångströms with a semantic reference label."""
    value: float
    reference: str = ""


# --- The Computational Substrate ---
@dataclass
class Phase48:
    tick: int
    micro: float = 0.0
    hand: Handedness = Handedness.RIGHT


# --- The Intent ---
class Gesture(Enum):
    HELIX_P5 = "The stable, perfect-fifth of a helix"
    SHEET_CENTER = "The balanced extension of a beta-sheet"
    LOOP_RESOLUTION = "The specific gesture that resolves clashes by creating space"

    def to_turn_pair(self) -> Tuple['Turn', 'Turn']:
        """Each gesture has a canonical (right-handed) physical form.

        Returns a pair (phi_turn, psi_turn) in fraction-of-turn units.
        """
        GESTURE_MAP = {
            Gesture.HELIX_P5: (Turn(57.0 / 360.0), Turn(47.0 / 360.0)),
            Gesture.SHEET_CENTER: (Turn(139.0 / 360.0), Turn(-135.0 / 360.0)),
            Gesture.LOOP_RESOLUTION: (Turn(80.0 / 360.0), Turn(-80.0 / 360.0)),
        }
        return GESTURE_MAP[self]


# --- The Final Object: The Conscious Act ---
@dataclass
class Movement:
    """A complete musical movement, containing the full context from intent to will."""
    gesture: Gesture
    mode: str
    role: str = 'body'
    hand_override: Handedness | None = None

    def get_torsions(self) -> Tuple[Turn, Turn]:
        phi_base, psi_base = self.gesture.to_turn_pair()
        hand = self.hand_override if self.hand_override is not None else Handedness.RIGHT
        # Return magnitudes with selected hand
        return (Turn(abs(phi_base.value), hand), Turn(abs(psi_base.value), hand))

    def to_degrees(self) -> Tuple[float, float]:
        """Return (phi, psi) in degrees for the selected hand."""
        phi_t, psi_t = self.get_torsions()
        return (phi_t.signed_value() * 360.0, psi_t.signed_value() * 360.0)


# Optional symbolic notes to reference in higher-level scales
class HelixNotes(Enum):
    P5 = 'helix_p5'
    CENTER = 'helix_center'


class LoopNotes(Enum):
    PPII = 'loop_ppii'
    RESOLUTION = 'loop_resolution'


# Physical constants (contextualized)
IDEAL_GEOMETRY = {
    "CA-CA": Angstrom(3.8, "CA-CA_adjacent"),
    "N-CA": Angstrom(1.45, "N-CA_bond"),
    "CA-C": Angstrom(1.52, "CA-C_bond"),
    "C-N": Angstrom(1.33, "C-N_peptide"),
}


# Refinement policy registry (versioned) for QC reports
REFINEMENT_POLICY = {
    'version': 'SANE-v1.0',
    'clash_cutoff': Angstrom(3.2, "CA-CA_nonadjacent_min"),
}
