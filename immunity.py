#!/usr/bin/env python3
"""
The 48-Manifold Immune System Analog
A computational model inspired by immune system mechanisms for integrity checking
"""

import torch
import torch.nn.functional as F
import hashlib
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import random
import time
import click
import statistics
import csv

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === Core Immune Constants ===
MHC_SIZE = 48  # MHC presents peptides of specific lengths
EPITOPE_FACTORS = [k_even := 2, k_odd := 3]  # Binary (self/non-self) and ternary (helper/killer/regulatory)

def get_divisors(n: int) -> List[int]:
    """Return all divisors of n greater than 1 (legal share factors)."""
    divs = []
    for k in range(2, n + 1):
        if n % k == 0:
            divs.append(k)
    return divs

# Centralized colored output
_COLORS = {
    "info": None,
    "notice": "cyan",
    "success": "green",
    "warn": "yellow",
    "error": "red",
}

def emit(msg: str, kind: str = "info") -> None:
    color = _COLORS.get(kind, None)
    click.secho(msg, fg=color)

class CellType(Enum):
    """Representative immune cell types with mapped roles (analog)"""
    THYMUS = "thymus"  # Training ground (genesis account)
    T_CELL = "t_cell"  # Verifier (whole account)
    B_CELL = "b_cell"  # Antibody producer (share manager)
    DENDRITIC = "dendritic"  # Presenter (router)
    MACROPHAGE = "macrophage"  # Cleaner (wholification pool)

@dataclass
class Antigen:
    """
    Antigen ~ WholeBundle (analog)
    Must be properly formatted to be recognized
    """
    epitope_id: str  # Like bundle_id
    peptides: torch.Tensor  # Shape: [48], like atoms
    mhc_signature: str  # Like merkle_root
    presented_by: str  # Which cell presents it
    folding_score: float = 1.0  # Proper folding = wholeness
    
    def is_properly_folded(self) -> bool:
        """Check if protein is whole (not denatured/decimated)"""
        # Proper folding means maintaining 48-structure
        if self.peptides.shape[0] != MHC_SIZE:
            return False
        
        # Check for fragmentation (like checking for decimation)
        if self.folding_score < 0.95:  # Partially denatured = decimated
            return False
        
        return True
    
    def compute_self_signature(self) -> str:
        """Generate self-signature (like merkle root)"""
        data = self.peptides.cpu().numpy().tobytes()
        return hashlib.sha256(data).hexdigest()[:16]

@dataclass
class Antibody:
    """
    Antibody ~ Share (analog)
    Binds to specific epitope regions
    """
    epitope_id: str  # Which antigen it targets
    binding_sites: torch.Tensor  # Which peptides it binds to
    affinity: float  # Binding strength
    produced_by: str  # Which B-cell made it
    
    @property
    def valency(self) -> int:
        """Number of binding sites (like share size)"""
        return len(self.binding_sites)

class ImmuneCell:
    """
    Immune Cell ~ WholenessAccount (analog)
    Maintains integrity through recognition
    """
    def __init__(self, cell_id: str, cell_type: CellType):
        self.cell_id = cell_id
        self.cell_type = cell_type
        self.recognized_self: Set[str] = set()  # Self-antigens (whole bundles)
        self.bound_antigens: Dict[str, Antigen] = {}
        self.antibodies: Dict[str, List[Antibody]] = {}
        self.activation_state: float = 0.0
        self.memory: List[Dict] = []
    
    def present_antigen(self, antigen: Antigen) -> bool:
        """
        Present antigen for recognition
        Like receiving a bundle - must be whole
        """
        # Check if properly folded (not decimated)
        if not antigen.is_properly_folded():
            emit(f"Rejected: misfolded protein (decimated value)", kind="error")
            self.trigger_inflammation()
            return False
        
        # Check peptide count (must be 48)
        if antigen.peptides.shape[0] != MHC_SIZE:
            emit(f"REJECTED: Invalid epitope size {antigen.peptides.shape[0]} != {MHC_SIZE}", kind="error")
            return False
        
        # Compute self-check
        signature = antigen.compute_self_signature()
        
        if signature in self.recognized_self:
            # Self-antigen: accept without inflammation
            emit(f"Recognized self-antigen {antigen.epitope_id}", kind="success")
            self.bound_antigens[antigen.epitope_id] = antigen
            return True
        else:
            # Foreign antigen detected; adaptive response can be mounted by effector pathways (e.g., B-cells)
            emit(f"Foreign antigen detected: {antigen.epitope_id}", kind="warn")
            return False
    
    def trigger_inflammation(self):
        """
        Inflammation (analog): rejection signal when integrity is violated
        """
        self.activation_state = 1.0
        emit(f"Inflammatory signal: cell {self.cell_id} detected an integrity violation", kind="warn")
    
    def mount_adaptive_response(self, antigen: Antigen, factors: Optional[List[int]] = None, shuffle_sites: bool = False):
        """
        Adaptive response (analog): attempt to bind and neutralize detected foreign material
        """
        # Generate antibodies (shares) that sum to whole
        antibodies = []
        use_factors = factors if factors is not None else EPITOPE_FACTORS
        for factor in use_factors:
            if MHC_SIZE % factor != 0:
                # Skip illegal factors to avoid fractional coverage
                continue
            sites_per_antibody = MHC_SIZE // factor
            # Optionally shuffle the site indices before partitioning
            if shuffle_sites:
                perm = torch.randperm(MHC_SIZE, device=device)
            else:
                perm = torch.arange(MHC_SIZE, device=device)
            for i in range(factor):
                start = i * sites_per_antibody
                stop = (i + 1) * sites_per_antibody
                binding_sites = perm[start:stop]
                
                antibody = Antibody(
                    epitope_id=antigen.epitope_id,
                    binding_sites=binding_sites,
                    affinity=0.9,
                    produced_by=self.cell_id
                )
                antibodies.append(antibody)
        
        self.antibodies[antigen.epitope_id] = antibodies
        emit(f"Generated {len(antibodies)} antibodies for {antigen.epitope_id}", kind="notice")
    
    def check_tolerance(self, pattern: torch.Tensor) -> bool:
        """
        Immune tolerance (analog): accepting certain patterns without activation
        """
        # Central tolerance: trained in thymus (genesis)
        if self.cell_type == CellType.THYMUS:
            # Learn what is self
            signature = hashlib.sha256(pattern.cpu().numpy().tobytes()).hexdigest()[:16]
            self.recognized_self.add(signature)
            return True
        
        # Peripheral tolerance: regulatory suppression
        if self.activation_state < 0.1:  # Suppressed state
            return True
        
        return False

class ClonalSelection:
    """
    Clonal selection (analog): only cells that recognize complete antigens proliferate
    """
    def __init__(self):
        self.cell_pool: Dict[str, ImmuneCell] = {}
        self.selection_pressure: float = 0.5
    
    def add_cell(self, cell: ImmuneCell):
        """Add cell to selection pool"""
        self.cell_pool[cell.cell_id] = cell
    
    def select_and_proliferate(self, antigen: Antigen) -> List[ImmuneCell]:
        """
        Select cells that properly bind antigen
        Like selecting accounts that maintain wholeness
        """
        selected_cells = []
        
        for cell_id, cell in self.cell_pool.items():
            # Test if cell can properly present antigen
            if cell.present_antigen(antigen):
                # Cell successfully bound - proliferate
                for i in range(3):  # Factor of 3 proliferation
                    clone = ImmuneCell(f"{cell_id}_clone_{i}", cell.cell_type)
                    clone.recognized_self = cell.recognized_self.copy()
                    selected_cells.append(clone)
                    
                emit(f"Cell {cell_id} selected for clonal expansion", kind="success")
        
        return selected_cells

class ComplementSystem:
    """
    Complement ~ Wholification Pool (analog)
    Combines antibody fragments to eliminate threats
    """
    def __init__(self):
        self.complement_cascade: List[torch.Tensor] = []
        self.c3_convertase_active: bool = False  # Like pool lock
    
    def activate_cascade(self, antibodies: List[Antibody]) -> Optional[torch.Tensor]:
        """
        Complement cascade (analog): multiple antibodies must combine to form MAC (Membrane Attack Complex)
        """
        if self.c3_convertase_active:
            emit("Cascade already active (atomic operation in progress)", kind="notice")
            return None
        
        self.c3_convertase_active = True
        
        try:
            # Check if antibodies form complete coverage
            all_sites = torch.cat([ab.binding_sites for ab in antibodies])
            unique_sites = torch.unique(all_sites)
            
            if len(unique_sites) != MHC_SIZE:
                emit(f"Incomplete opsonization: {len(unique_sites)}/{MHC_SIZE} sites", kind="warn")
                return None
            
            # Form MAC (Membrane Attack Complex) - like creating whole bundle
            mac_complex = torch.ones(MHC_SIZE, device=device)
            emit(f"MAC formed: complete neutralization achieved", kind="success")
            return mac_complex
            
        finally:
            self.c3_convertase_active = False

class ImmuneSystem:
    """
    Complete Immune System ~ Wholeness Ledger (analog)
    Maintains integrity through recognition and verification
    """
    def __init__(self):
        self.cells: Dict[str, ImmuneCell] = {}
        self.clonal_selector = ClonalSelection()
        self.complement = ComplementSystem()
        self.thymus = self._create_thymus()
        self.inflammation_level: float = 0.0
    
    def _create_thymus(self) -> ImmuneCell:
        """
        Thymus ~ Genesis account (analog)
        Where self-tolerance is established
        """
        thymus = ImmuneCell("thymus", CellType.THYMUS)
        
        # Train on self-antigens (like minting genesis bundles)
        for i in range(3):
            self_pattern = torch.arange(MHC_SIZE, device=device) + (i * 100)
            thymus.check_tolerance(self_pattern)
        
        emit(f"Thymus initialized: {len(thymus.recognized_self)} self-antigens recognized", kind="notice")
        return thymus
    
    def create_immune_cell(self, cell_type: CellType) -> ImmuneCell:
        """Create new immune cell with inherited tolerance"""
        cell_id = f"{cell_type.value}_{len(self.cells)}"
        cell = ImmuneCell(cell_id, cell_type)
        
        # Inherit self-recognition from thymus
        cell.recognized_self = self.thymus.recognized_self.copy()
        
        self.cells[cell_id] = cell
        self.clonal_selector.add_cell(cell)
        
        emit(f"Created {cell_type.value}: {cell_id}", kind="notice")
        return cell
    
    def check_system_integrity(self) -> bool:
        """
        Verify immune homeostasis
        Like verifying ledger wholeness
        """
        total_antigens = 0
        misfolded_count = 0
        
        for cell_id, cell in self.cells.items():
            for antigen_id, antigen in cell.bound_antigens.items():
                total_antigens += 1
                if not antigen.is_properly_folded():
                    misfolded_count += 1
                    emit(f"INTEGRITY VIOLATION: Misfolded protein in {cell_id}", kind="error")
        
        # Check if all antigens are properly sized
        if misfolded_count > 0:
            self.inflammation_level = misfolded_count / max(total_antigens, 1)
            emit(f"System inflammation: {self.inflammation_level:.1%}", kind="warn")
            return False
        
        emit(f"Immune homeostasis maintained: {total_antigens} antigens, 0 misfolded", kind="success")
        return True

def demonstrate_immune_analog():
    """
    Demonstrate an immune-system-inspired analog within the 48-manifold framework
    """
    emit("=" * 60, kind="notice")
    emit("IMMUNE SYSTEM ANALOG DEMONSTRATION", kind="notice")
    emit("=" * 60, kind="notice")
    
    # Initialize immune system
    immune = ImmuneSystem()
    
    emit("\n1. CELL DIFFERENTIATION (Account Creation):", kind="info")
    t_cell = immune.create_immune_cell(CellType.T_CELL)
    b_cell = immune.create_immune_cell(CellType.B_CELL)
    dendritic = immune.create_immune_cell(CellType.DENDRITIC)
    
    emit("\n2. SELF-ANTIGEN PRESENTATION (analog: whole-bundle reception):", kind="info")
    # Create properly folded self-antigen
    self_antigen = Antigen(
        epitope_id="self_protein_1",
        peptides=torch.arange(MHC_SIZE, device=device),
        mhc_signature="",
        presented_by="dendritic",
        folding_score=1.0
    )
    dendritic.present_antigen(self_antigen)
    
    emit("\n3. MISFOLDED PROTEIN REJECTION (decimation prevention):", kind="info")
    # Try to present misfolded protein (decimated value)
    misfolded = Antigen(
        epitope_id="prion_1",
        peptides=torch.randn(MHC_SIZE, device=device),
        mhc_signature="",
        presented_by="dendritic",
        folding_score=0.3  # Severely misfolded
    )
    dendritic.present_antigen(misfolded)
    
    emit("\n4. FOREIGN ANTIGEN RESPONSE (non-self detection):", kind="info")
    # Present foreign but properly structured antigen
    foreign_antigen = Antigen(
        epitope_id="virus_spike_1",
        peptides=torch.arange(MHC_SIZE, device=device) + 1000,
        mhc_signature="",
        presented_by="dendritic",
        folding_score=1.0
    )
    dendritic.present_antigen(foreign_antigen)
    
    emit("\n5. ANTIBODY GENERATION (share creation):", kind="info")
    # B-cell produces antibodies (explicitly mounted here)
    b_cell.mount_adaptive_response(foreign_antigen)
    if foreign_antigen.epitope_id in b_cell.antibodies:
        antibodies = b_cell.antibodies[foreign_antigen.epitope_id]
        emit(f"   B-cell generated {len(antibodies)} antibodies", kind="info")
        emit(f"   Each antibody covers {antibodies[0].valency} epitopes", kind="info")
    
    emit("\n6. COMPLEMENT CASCADE (wholification):", kind="info")
    # Activate complement to form MAC
    if foreign_antigen.epitope_id in b_cell.antibodies:
        mac = immune.complement.activate_cascade(
            b_cell.antibodies[foreign_antigen.epitope_id]
        )
        if mac is not None:
            emit("   Complete neutralization achieved", kind="success")
    
    emit("\n7. CLONAL SELECTION (atomic verification):", kind="info")
    # Select cells that maintain integrity
    selected = immune.clonal_selector.select_and_proliferate(self_antigen)
    emit(f"   {len(selected)} cells selected for proliferation", kind="info")
    
    emit("\n8. SYSTEM INTEGRITY CHECK (homeostasis):", kind="info")
    immune.check_system_integrity()
    
    emit("\n" + "=" * 60, kind="notice")
    emit("KEY ANALOG CORRESPONDENCES:", kind="notice")
    emit("=" * 60, kind="notice")
    
    parallels = [
        ("Whole Bundle", "=", "Properly Folded Protein"),
        ("Decimated Value", "=", "Misfolded/Denatured Protein"),
        ("Account", "=", "Immune Cell"),
        ("Merkle Root", "=", "MHC Signature"),
        ("Share", "=", "Antibody"),
        ("Wholification Pool", "=", "Complement System"),
        ("Genesis Minting", "=", "Thymic Education"),
        ("Transfer", "=", "Antigen Presentation"),
        ("Atomic Reception", "=", "Clonal Selection"),
        ("48 Atoms", "=", "48 Peptides in MHC"),
        ("Factor 2/3 Split", "=", "Binary/Ternary Immune Response"),
        ("Reversible Operations", "=", "Immune Memory"),
    ]
    
    for left, eq, right in parallels:
        emit(f"  {left:20s} {eq} {right}", kind="info")
    
    emit("\n" + "=" * 60, kind="notice")
    emit("MODEL SIGNALS:", kind="notice")
    emit("  In this model, integrity is enforced by:", kind="info")
    emit("  - Rejecting fragmented/decimated entities", kind="info")
    emit("  - Accepting only properly formatted wholes", kind="info")
    emit("  - Using cryptographic/molecular signatures", kind="info")
    emit("  - Providing repair mechanisms for fragments", kind="info")
    emit("  - Maintaining memory of what belongs", kind="info")
    emit("  - Operating through reversible recognition", kind="info")
    emit("=" * 60, kind="notice")

def run_randomized_trials(
    trials: int = 5,
    seed: int = 42,
    fold_low: float = 0.0,
    fold_high: float = 0.9,
    offset_min: int = 1000,
    offset_max: int = 100000,
    factor_k_max: int = 3,
    shuffle_sites: bool = True,
) -> Dict[str, int]:
    """
    Run randomized robustness checks with seedable defaults.
    Checks:
    - self acceptance (properly folded, recognized pattern)
    - misfolded rejection (decimation analog)
{{ ... }}
    - complement behavior on full vs partial coverage
    """
    random.seed(seed)
    torch.manual_seed(seed)

    immune = ImmuneSystem()
    dendritic = immune.create_immune_cell(CellType.DENDRITIC)
    b_cell = immune.create_immune_cell(CellType.B_CELL)

    # Self patterns the thymus was initialized on
    bases = [0, 100, 200]
    divs = get_divisors(MHC_SIZE)

    results: Dict[str, int] = {
        "self_accept_pass": 0,
        "misfold_reject_pass": 0,
        "foreign_detect_pass": 0,
        "complement_full_pass": 0,
        "complement_partial_pass": 0,
        "trials": trials,
    }

    for t in range(trials):
        # Self antigen (recognized)
        base = random.choice(bases)
        peptides_self = torch.arange(MHC_SIZE, device=device) + base
        self_ag = Antigen(
            epitope_id=f"self_{t}",
            peptides=peptides_self,
            mhc_signature="",
            presented_by="dendritic",
            folding_score=1.0,
        )
        accepted = dendritic.present_antigen(self_ag)
        if accepted:
            results["self_accept_pass"] += 1

        # Misfolded antigen (rejected)
        fold_score = random.uniform(fold_low, fold_high)
        mis_pep = torch.randn(MHC_SIZE, device=device)
        mis_ag = Antigen(
            epitope_id=f"mis_{t}",
            peptides=mis_pep,
            mhc_signature="",
            presented_by="dendritic",
            folding_score=fold_score,
        )
        mis_ok = dendritic.present_antigen(mis_ag)
        if not mis_ok:
            results["misfold_reject_pass"] += 1
        dendritic.activation_state = 0.0  # reset between trials

        # Foreign antigen (non-self)
        offset = random.randint(offset_min, offset_max)
        foreign_pep = torch.arange(MHC_SIZE, device=device) + offset
        foreign_ag = Antigen(
            epitope_id=f"foreign_{t}",
            peptides=foreign_pep,
            mhc_signature="",
            presented_by="dendritic",
            folding_score=1.0,
        )
        detected = not dendritic.present_antigen(foreign_ag)
        if detected:
            results["foreign_detect_pass"] += 1

        # Randomized antibody factors and complement behavior
        k = random.randint(1, min(factor_k_max, len(divs)))
        factors = random.sample(divs, k)
        b_cell.mount_adaptive_response(foreign_ag, factors=factors, shuffle_sites=shuffle_sites)
        abs_full = b_cell.antibodies.get(foreign_ag.epitope_id, [])
        if abs_full:
            all_sites = torch.cat([ab.binding_sites for ab in abs_full])
            unique_sites = torch.unique(all_sites)
            mac = immune.complement.activate_cascade(abs_full)
            expect_full = len(unique_sites) == MHC_SIZE
            if (mac is not None) == expect_full:
                results["complement_full_pass"] += 1

            # Partial coverage: drop one antibody when possible
            abs_partial = abs_full[:-1] if len(abs_full) > 0 else []
            if abs_partial:
                all_sites_p = torch.cat([ab.binding_sites for ab in abs_partial])
                unique_sites_p = torch.unique(all_sites_p)
                mac_p = immune.complement.activate_cascade(abs_partial)
                expect_partial = len(unique_sites_p) == MHC_SIZE
                if (mac_p is not None) == expect_partial:
                    results["complement_partial_pass"] += 1

    # Print concise summary
    emit("\nRANDOMIZED TRIALS SUMMARY", kind="notice")
    emit(f"  Trials: {results['trials']}", kind="info")
    emit(f"  Self acceptance passes: {results['self_accept_pass']}", kind="success")
    emit(f"  Misfold rejection passes: {results['misfold_reject_pass']}", kind="success")
    emit(f"  Foreign detection passes: {results['foreign_detect_pass']}", kind="success")
    emit(f"  Complement full-coverage behavior passes: {results['complement_full_pass']}", kind="success")
    emit(f"  Complement partial-coverage behavior passes: {results['complement_partial_pass']}", kind="success")
    return results

def timed_call(label: str, func, *args, **kwargs):
    """Run func(*args, **kwargs) and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    dt = time.perf_counter() - t0
    return result, dt


@click.command()
@click.option("--demo/--no-demo", default=True, help="Run the analog demonstration.")
@click.option("--trials", type=int, default=0, help="Run randomized trials (0 to skip).")
@click.option("--seed", type=int, default=42, help="PRNG seed for randomized trials.")
@click.option("--benchmark/--no-benchmark", default=True, help="Display timing benchmarks.")
@click.option("--repeat", type=int, default=1, help="Repeat runs for benchmarking; prints may be verbose.")
@click.option("--json-summary", type=click.Path(dir_okay=False, writable=True), default=None, help="Write JSON summary to file (or '-' for stdout).")
@click.option("--csv-summary", type=click.Path(dir_okay=False, writable=True), default=None, help="Write CSV summary to file (or '-' for stdout).")
# Randomization controls
@click.option("--fold-low", type=float, default=0.0, help="Lower bound for misfolded folding_score.")
@click.option("--fold-high", type=float, default=0.9, help="Upper bound for misfolded folding_score.")
@click.option("--offset-min", type=int, default=1000, help="Minimum offset for foreign antigen peptides.")
@click.option("--offset-max", type=int, default=100000, help="Maximum offset for foreign antigen peptides.")
@click.option("--factor-k-max", type=int, default=3, help="Max number of divisor factors sampled per trial.")
@click.option("--shuffle-sites/--no-shuffle-sites", default=True, help="Randomize binding site partitions for antibodies.")
def cli(
    demo: bool,
    trials: int,
    seed: int,
    benchmark: bool,
    repeat: int,
    json_summary: Optional[str],
    csv_summary: Optional[str],
    fold_low: float,
    fold_high: float,
    offset_min: int,
    offset_max: int,
    factor_k_max: int,
    shuffle_sites: bool,
):
    """CLI for the immune-system analog demo and randomized tests with timing and summaries."""
    click.secho(f"Device: {device}", fg="cyan")

    demo_times: List[float] = []
    trial_times: List[float] = []
    last_results: Optional[Dict[str, int]] = None

    if demo:
        for i in range(max(1, repeat)):
            _, dt = timed_call("demo", demonstrate_immune_analog)
            demo_times.append(dt)
        if benchmark:
            mean = statistics.mean(demo_times)
            std = statistics.pstdev(demo_times) if len(demo_times) > 1 else 0.0
            click.secho(f"Demo time: mean={mean*1000:.2f} ms, std={std*1000:.2f} ms over {len(demo_times)} run(s)", fg="green")

    if trials and trials > 0:
        for i in range(max(1, repeat)):
            results, dt = timed_call(
                "randomized_trials",
                run_randomized_trials,
                trials=trials,
                seed=seed,
                fold_low=fold_low,
                fold_high=fold_high,
                offset_min=offset_min,
                offset_max=offset_max,
                factor_k_max=factor_k_max,
                shuffle_sites=shuffle_sites,
            )
            trial_times.append(dt)
            last_results = results
        if benchmark:
            mean = statistics.mean(trial_times)
            std = statistics.pstdev(trial_times) if len(trial_times) > 1 else 0.0
            rate = trials / mean if mean > 0 else float('inf')
            click.secho(f"Trials: {trials} | mean={mean:.3f}s, std={std:.3f}s ({rate:.1f} trials/s)", fg="green")

    # Centralized summary construction
    import json as _json
    summary = {
        "device": str(device),
        "demo_ms": [t * 1000.0 for t in demo_times],
        "trials": trials,
        "trial_times_s": trial_times,
        "repeat": repeat,
        "seed": seed,
        "randomization": {
            "fold_low": fold_low,
            "fold_high": fold_high,
            "offset_min": offset_min,
            "offset_max": offset_max,
            "factor_k_max": factor_k_max,
            "shuffle_sites": shuffle_sites,
        },
        "last_results": last_results or {},
    }

    # JSON output
    if json_summary:
        payload = _json.dumps(summary, indent=2)
        if json_summary == "-":
            click.echo(payload)
        else:
            with open(json_summary, "w") as f:
                f.write(payload)

    # CSV output (one-line rollup)
    if csv_summary:
        demo_mean = statistics.mean(demo_times) if demo_times else 0.0
        demo_std = statistics.pstdev(demo_times) if len(demo_times) > 1 else 0.0
        trial_mean = statistics.mean(trial_times) if trial_times else 0.0
        trial_std = statistics.pstdev(trial_times) if len(trial_times) > 1 else 0.0
        rate = (trials / trial_mean) if (trials and trial_mean > 0) else 0.0
        row = {
            "device": str(device),
            "trials": trials,
            "repeat": repeat,
            "seed": seed,
            "demo_mean_ms": round(demo_mean * 1000.0, 3),
            "demo_std_ms": round(demo_std * 1000.0, 3),
            "trials_mean_s": round(trial_mean, 6),
            "trials_std_s": round(trial_std, 6),
            "trials_rate_per_s": round(rate, 2),
            "self_accept_pass": (last_results or {}).get("self_accept_pass", 0),
            "misfold_reject_pass": (last_results or {}).get("misfold_reject_pass", 0),
            "foreign_detect_pass": (last_results or {}).get("foreign_detect_pass", 0),
            "complement_full_pass": (last_results or {}).get("complement_full_pass", 0),
            "complement_partial_pass": (last_results or {}).get("complement_partial_pass", 0),
            "fold_low": fold_low,
            "fold_high": fold_high,
            "offset_min": offset_min,
            "offset_max": offset_max,
            "factor_k_max": factor_k_max,
            "shuffle_sites": shuffle_sites,
        }
        headers = list(row.keys())
        if csv_summary == "-":
            writer = csv.DictWriter(click.get_text_stream('stdout'), fieldnames=headers)
            writer.writeheader()
            writer.writerow(row)
        else:
            with open(csv_summary, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerow(row)


if __name__ == "__main__":
    cli()
