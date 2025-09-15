import numpy as np
import torch
import time
import json
import os
from concurrent.futures import ProcessPoolExecutor
from bio.devices import get_device
try:
    from .geometry import INTERVAL_TO_TORSION, IDEAL_GEOMETRY, place_next_atom
    from .key_estimator import estimate_key_and_modes
    from bio.scale_and_meter import snap_to_scale, enforce_meter
except ImportError:  # allow running as a script: python bio/conductor.py
    from bio.geometry import INTERVAL_TO_TORSION, IDEAL_GEOMETRY, place_next_atom
    from bio.key_estimator import estimate_key_and_modes
    from bio.scale_and_meter import snap_to_scale, enforce_meter

# Chirality-aware notation
try:
    from agi.harmonia.notation import Handedness
    from agi.harmonia.notation import Movement, Gesture
except Exception:
    # Fallback placeholder to avoid hard failure if module not found
    class Handedness:
        RIGHT = type("H", (), {"name": "RIGHT"})()
        LEFT = type("H", (), {"name": "LEFT"})()
    Movement = None
    Gesture = None
try:
    from agi.metro.sanity import audit_sanity
    from agi.harmonia import notation
except Exception:
    audit_sanity = None
    notation = None

def _min_caca_for_torsions_batch(args: tuple[str, list[np.ndarray]]) -> tuple[float, int]:
    """Compute the best (max) min CA-CA distance among a batch of candidate torsions.

    Returns (best_min_caca_value, relative_index_in_batch). If batch is empty, returns (-inf, -1).
    """
    sequence, batch = args
    if not batch:
        return float('-inf'), -1
    # Ideal parameters (mirror build_backbone_from_torsions)
    b_N_CA = IDEAL_GEOMETRY["N-CA"]
    b_CA_C = IDEAL_GEOMETRY["CA-C"]
    b_C_N = IDEAL_GEOMETRY["C-N"]
    ang_CA_C_N = IDEAL_GEOMETRY["CA-C-N_angle"]
    ang_C_N_CA = IDEAL_GEOMETRY["C-N-CA_angle"]
    ang_N_CA_C = IDEAL_GEOMETRY["N-CA-C_angle"]
    omega = 180.0
    L = len(sequence)

    best_val = float('-inf')
    best_idx = -1
    for idx, torsions in enumerate(batch):
        # Seed first residue (NumPy baseline)
        N0 = np.array([0.0, 0.0, 0.0])
        CA0 = np.array([b_N_CA, 0.0, 0.0])
        theta = np.deg2rad(180.0 - ang_N_CA_C)
        C0 = CA0 + np.array([b_CA_C * np.cos(theta), b_CA_C * np.sin(theta), 0.0])
        N_coords = [N0]
        CA_coords = [CA0]
        C_coords = [C0]
        phi = torsions[:, 0]
        psi = torsions[:, 1]

        # Optional accelerated path using torch (CUDA/MPS) with NeRF in torch
        use_accel = False
        try:
            use_accel = bool(int(os.getenv('AGI_ACCEL_TORCH', '0')))
        except Exception:
            use_accel = False
        if use_accel:
            try:
                import torch
                dev = get_device()
                if dev.type in ("cuda", "mps"):
                    # Constants
                    b_N_CA = float(IDEAL_GEOMETRY["N-CA"])  # ~1.45
                    b_CA_C = float(IDEAL_GEOMETRY["CA-C"])  # ~1.52
                    b_C_N = float(IDEAL_GEOMETRY["C-N"])    # ~1.33
                    ang_CA_C_N = float(IDEAL_GEOMETRY["CA-C-N_angle"])  # deg
                    ang_C_N_CA = float(IDEAL_GEOMETRY["C-N-CA_angle"])  # deg
                    ang_N_CA_C = float(IDEAL_GEOMETRY["N-CA-C_angle"])  # deg
                    omega = 180.0

                    def deg2rad(x: float | torch.Tensor) -> torch.Tensor:
                        return torch.as_tensor(x, dtype=torch.float32, device=dev) * (torch.pi / 180.0)

                    # Seed first residue on device
                    N0 = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=dev)
                    CA0 = torch.tensor([b_N_CA, 0.0, 0.0], dtype=torch.float32, device=dev)
                    theta0 = deg2rad(180.0 - ang_N_CA_C)
                    C0 = CA0 + torch.stack([
                        torch.cos(theta0) * b_CA_C,
                        torch.sin(theta0) * b_CA_C,
                        torch.tensor(0.0, device=dev),
                    ])

                    N_coords = [N0]
                    CA_coords = [CA0]
                    C_coords = [C0]

                    phi_t = torch.as_tensor(phi, dtype=torch.float32, device=dev)
                    psi_t = torch.as_tensor(psi, dtype=torch.float32, device=dev)

                    def place_next_atom_torch(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor,
                                               *, length: float, angle_deg: float, torsion_deg: float) -> torch.Tensor:
                        # Implements NeRF placement on device
                        eps = 1e-8
                        v1 = p1 - p2
                        v2 = p2 - p3
                        e1 = v2 / (torch.linalg.norm(v2) + eps)
                        n = torch.cross(e1, v1)
                        e2 = n / (torch.linalg.norm(n) + eps)
                        e3 = torch.cross(e2, e1)
                        R = torch.stack([e1, e2, e3], dim=1)  # 3x3 (columns)
                        theta = deg2rad(angle_deg)
                        tau = deg2rad(torsion_deg)
                        r = torch.stack([
                            -torch.cos(theta) * length,
                            torch.sin(theta) * torch.cos(tau) * length,
                            torch.sin(theta) * torch.sin(tau) * length,
                        ])
                        return p3 + R @ r

                    # Build subsequent residues (sequential dependency)
                    for i in range(1, L):
                        Ni = place_next_atom_torch(N_coords[i - 1], CA_coords[i - 1], C_coords[i - 1],
                                                   length=b_C_N, angle_deg=ang_CA_C_N, torsion_deg=omega)
                        CAi = place_next_atom_torch(CA_coords[i - 1], C_coords[i - 1], Ni,
                                                    length=b_N_CA, angle_deg=ang_C_N_CA, torsion_deg=float(phi_t[i].item()))
                        Ci = place_next_atom_torch(C_coords[i - 1], Ni, CAi,
                                                   length=b_CA_C, angle_deg=ang_N_CA_C, torsion_deg=float(psi_t[i].item()))
                        N_coords.append(Ni)
                        CA_coords.append(CAi)
                        C_coords.append(Ci)

                    backbone_t = torch.stack([
                        torch.stack([N_coords[i], CA_coords[i], C_coords[i]], dim=0)
                        for i in range(L)
                    ], dim=0)
                    return backbone_t.detach().to('cpu').numpy()
            except Exception:
                # Fallback to NumPy baseline below
                pass
        for i in range(1, L):
            Ni = place_next_atom(N_coords[i - 1], CA_coords[i - 1], C_coords[i - 1], length=b_C_N, angle=ang_CA_C_N, torsion=omega)
            CAi = place_next_atom(CA_coords[i - 1], C_coords[i - 1], Ni, length=b_N_CA, angle=ang_C_N_CA, torsion=phi[i])
            Ci = place_next_atom(C_coords[i - 1], Ni, CAi, length=b_CA_C, angle=ang_N_CA_C, torsion=psi[i])
            N_coords.append(Ni)
            CA_coords.append(CAi)
            C_coords.append(Ci)
        CA = np.stack(CA_coords, axis=0)
        if CA.shape[0] >= 2:
            d = np.linalg.norm(CA[1:] - CA[:-1], axis=1)
            dmin = float(np.min(d)) if d.size > 0 else float('-inf')
        else:
            dmin = float('-inf')
        if dmin > best_val:
            best_val = dmin
            best_idx = idx
    return best_val, best_idx

class Conductor:
    """
    Translates a sequence of 48D composition vectors into a 3D protein structure.
    """
    def __init__(self):
        # Create "template vectors" for each interval to find the closest match.
        # This is a simple but effective way to classify the harmonic transition.
        self.interval_templates = {
            "Perfect 5th": torch.sin(torch.linspace(0, 5, 48)),
            "Minor 2nd":   torch.cos(torch.linspace(0, 2, 48)),
            "Major 3rd":   torch.sin(torch.linspace(0, 3, 48) ** 2),
            "Unison":      torch.ones(48) * 0.1,
        }
        # Normalize templates
        for k, v in self.interval_templates.items():
            self.interval_templates[k] = v / torch.linalg.norm(v)

    def _get_dominant_interval(self, transition_vector: torch.Tensor,
                               current_vec: torch.Tensor | None = None,
                               global_kore: torch.Tensor | None = None) -> str:
        """Finds the musical interval that best matches the transition.

        If a global_kore is provided, bias the match by the alignment of the
        current composition vector to the kore (key-aware decision).
        """
        # If transition is near-zero, treat as Unison-like to avoid NaNs
        if torch.linalg.norm(transition_vector) < 1e-8:
            return "Unison"
        transition_vector = transition_vector / torch.linalg.norm(transition_vector)
        
        # Use cosine similarity to find the best match
        best_match = "Dissonant"
        max_sim = -1.0
        for name, template in self.interval_templates.items():
            sim = torch.dot(transition_vector, template).item()
            # Key-aware bias: weight similarity by projection onto kore
            if current_vec is not None and global_kore is not None:
                denom = (torch.linalg.norm(current_vec) * torch.linalg.norm(global_kore) + 1e-8)
                key_align = torch.dot(current_vec, global_kore) / denom
                # Blend: 80% transition similarity, 20% key alignment
                sim = 0.8 * sim + 0.2 * key_align.item()
            if sim > max_sim:
                max_sim = sim
                best_match = name
        return best_match

    def build_backbone(self, composition_vectors: torch.Tensor, sequence: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """
        Generates a full N-CA-C backbone using the Harmony Constraint Layer:
        - Estimate global kore and per-residue modes
        - Map transitions to raw torsions (phi, psi) in a key-aware fashion
        - Snap torsions to the allowed scale per mode and enforce simple meter
        - Build backbone with NERF using IDEAL_GEOMETRY

        Returns:
            backbone: np.ndarray of shape (L, 3, 3) with atoms ordered (N, CA, C)
            phi: np.ndarray of shape (L,)
            psi: np.ndarray of shape (L,)
            modes: list[str] of length L
        """
        # Accept either a single 48D vector or a sequence of 48D vectors.
        # If we got shape (1, 48), squeeze it to 1D and expand as a toy sequence.
        if composition_vectors.ndim == 2 and composition_vectors.shape[0] == 1:
            composition_vectors = composition_vectors.squeeze(0)

        # Derive a default sequence if not provided
        if sequence is None:
            # Default to poly-Ala length 48
            sequence = "A" * 48

        num_residues = len(sequence)

        # Ensure we have a composition vector per residue (expand/jitter if needed)
        if composition_vectors.ndim == 1:
            base = composition_vectors
            composition_vectors = torch.stack([
                base + 0.01 * torch.randn_like(base) * i for i in range(num_residues)
            ], dim=0)
        else:
            if composition_vectors.shape[0] < num_residues:
                base = composition_vectors[-1]
                extra = torch.stack([
                    base + 0.01 * torch.randn_like(base) * (i+1)
                    for i in range(num_residues - composition_vectors.shape[0])
                ], dim=0)
                composition_vectors = torch.cat([composition_vectors, extra], dim=0)
            elif composition_vectors.shape[0] > num_residues:
                composition_vectors = composition_vectors[:num_residues]

        # 1. Determine Key and Modes
        global_kore, modes = estimate_key_and_modes(composition_vectors, sequence)

        # Initialize default handedness map (all RIGHT by default)
        try:
            self.handedness = [Handedness.RIGHT] * num_residues
        except Exception:
            pass

        # Initialize Movement intents if available
        try:
            if Movement is not None and Gesture is not None:
                mode_to_gesture = {
                    'helix': Gesture.HELIX_P5,
                    'sheet': Gesture.SHEET_CENTER,
                    'loop': Gesture.LOOP_RESOLUTION,
                }
                self.movements = [
                    Movement(gesture=mode_to_gesture.get(modes[i], Gesture.LOOP_RESOLUTION), mode=modes[i], hand_override=self.handedness[i])
                    for i in range(num_residues)
                ]
            else:
                self.movements = None
        except Exception:
            self.movements = None

        # 2. Generate Raw Torsion Angles from Music (key-aware)
        raw_torsions = [( -57.0, -47.0 )]  # placeholder for residue 0
        for i in range(1, num_residues):
            transition_vec = composition_vectors[i] - composition_vectors[i-1]
            interval = self._get_dominant_interval(transition_vec, composition_vectors[i], global_kore)
            raw_torsions.append(INTERVAL_TO_TORSION.get(interval, INTERVAL_TO_TORSION["Dissonant"]))

        # 3. Snap Torsions to the Correct Scale per residue mode
        scaled_torsions = []
        for i in range(num_residues):
            mode = modes[i] if i < len(modes) else 'loop'
            phi_raw, psi_raw = raw_torsions[i]
            phi_scaled, psi_scaled = snap_to_scale(mode, phi_raw, psi_raw)
            scaled_torsions.append((phi_scaled, psi_scaled))

        # 4. Enforce Meter (simple smoothing for helices)
        # Segment by contiguous mode and apply meter
        final_torsions: list[tuple[float, float]] = [(0.0, 0.0)] * num_residues
        start = 0
        while start < num_residues:
            mode = modes[start]
            end = start
            while end < num_residues and modes[end] == mode:
                end += 1
            segment = scaled_torsions[start:end]
            segment_refined = enforce_meter(mode, segment)
            final_torsions[start:end] = segment_refined
            start = end

        # Convert to phi/psi arrays (per-residue). Canonical convention:
        # For i >= 1, we place CA(i) with phi(i) and C(i) with psi(i).
        phi = np.zeros(num_residues)
        psi = np.zeros(num_residues)
        for i in range(num_residues):
            ph, ps = final_torsions[i]
            phi[i] = ph
            psi[i] = ps

        # Ideal parameters
        b_N_CA = IDEAL_GEOMETRY["N-CA"]
        b_CA_C = IDEAL_GEOMETRY["CA-C"]
        b_C_N = IDEAL_GEOMETRY["C-N"]
        ang_CA_C_N = IDEAL_GEOMETRY["CA-C-N_angle"]
        ang_C_N_CA = IDEAL_GEOMETRY["C-N-CA_angle"]
        ang_N_CA_C = IDEAL_GEOMETRY["N-CA-C_angle"]
        omega = 180.0  # trans peptide bond

        # Seed first residue atoms in a canonical pose
        N0 = np.array([0.0, 0.0, 0.0])
        CA0 = np.array([b_N_CA, 0.0, 0.0])
        # Place C0 in xy-plane to satisfy N-CA-C angle
        theta = np.deg2rad(180.0 - ang_N_CA_C)  # angle from +x axis
        C0 = CA0 + np.array([
            b_CA_C * np.cos(theta),
            b_CA_C * np.sin(theta),
            0.0,
        ])

        N_coords = [N0]
        CA_coords = [CA0]
        C_coords = [C0]

        # Build subsequent residues using canonical sequential NeRF
        for i in range(1, num_residues):
            # Place N(i) using (N(i-1), CA(i-1), C(i-1)) and omega (trans peptide)
            Ni = place_next_atom(
                N_coords[i - 1], CA_coords[i - 1], C_coords[i - 1],
                length=b_C_N,
                angle=ang_CA_C_N,
                torsion=omega,
            )
            # Place CA(i) using (CA(i-1), C(i-1), N(i)) and phi(i)
            CAi = place_next_atom(
                CA_coords[i - 1], C_coords[i - 1], Ni,
                length=b_N_CA,
                angle=ang_C_N_CA,
                torsion=phi[i],
            )
            # Place C(i) using (C(i-1), N(i), CA(i)) and psi(i)
            Ci = place_next_atom(
                C_coords[i - 1], Ni, CAi,
                length=b_CA_C,
                angle=ang_N_CA_C,
                torsion=psi[i],
            )

            N_coords.append(Ni)
            CA_coords.append(CAi)
            C_coords.append(Ci)

        # Pack into (L, 3, 3): atoms ordered N, CA, C
        backbone = np.stack([
            np.stack([N_coords[i], CA_coords[i], C_coords[i]], axis=0)
            for i in range(num_residues)
        ], axis=0)
        return backbone, phi, psi, modes

    # -------------------------
    # Backbone builder from torsions
    # -------------------------
    def build_backbone_from_torsions(self, torsions: np.ndarray, sequence: str) -> np.ndarray:
        """Builds the N-CA-C backbone from an explicit list/array of (phi, psi) torsions.

        Args:
            torsions: array-like (L, 2) of (phi, psi) in degrees per residue
            sequence: protein sequence; used only for length L

        Returns:
            backbone np.ndarray (L, 3, 3)
        """
        L = len(sequence)
        torsions = np.asarray(torsions, dtype=np.float32)
        assert torsions.shape[0] == L, "torsions length must equal sequence length"

        phi = torsions[:, 0]
        psi = torsions[:, 1]

        # Ideal parameters
        b_N_CA = IDEAL_GEOMETRY["N-CA"]
        b_CA_C = IDEAL_GEOMETRY["CA-C"]
        b_C_N = IDEAL_GEOMETRY["C-N"]
        ang_CA_C_N = IDEAL_GEOMETRY["CA-C-N_angle"]
        ang_C_N_CA = IDEAL_GEOMETRY["C-N-CA_angle"]
        ang_N_CA_C = IDEAL_GEOMETRY["N-CA-C_angle"]
        omega = 180.0

        # Seed first residue
        N0 = np.array([0.0, 0.0, 0.0])
        CA0 = np.array([b_N_CA, 0.0, 0.0])
        theta = np.deg2rad(180.0 - ang_N_CA_C)
        C0 = CA0 + np.array([b_CA_C * np.cos(theta), b_CA_C * np.sin(theta), 0.0])

        N_coords = [N0]
        CA_coords = [CA0]
        C_coords = [C0]

        for i in range(1, L):
            # Canonical sequential NeRF (original mapping):
            # N(i): torsion omega, CA(i): torsion phi(i), C(i): torsion psi(i)
            Ni = place_next_atom(N_coords[i - 1], CA_coords[i - 1], C_coords[i - 1],
                                 length=b_C_N, angle=ang_CA_C_N, torsion=omega)
            CAi = place_next_atom(CA_coords[i - 1], C_coords[i - 1], Ni,
                                  length=b_N_CA, angle=ang_C_N_CA, torsion=phi[i])
            Ci = place_next_atom(C_coords[i - 1], Ni, CAi,
                                 length=b_CA_C, angle=ang_N_CA_C, torsion=psi[i])
            N_coords.append(Ni)
            CA_coords.append(CAi)
            C_coords.append(Ci)

        backbone = np.stack([
            np.stack([N_coords[i], CA_coords[i], C_coords[i]], axis=0)
            for i in range(L)
        ], axis=0)
        return backbone

    def save_to_pdb(self, coords: np.ndarray, filename: str):
        """Saves backbone coordinates to a PDB file.

        - If coords has shape (L, 3), writes CA trace.
        - If coords has shape (L, 3, 3), writes N, CA, C per residue.
        """
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        with open(filename, 'w') as f:
            serial = 1
            if coords.ndim == 2 and coords.shape[1] == 3:
                # CA trace only
                for i, pos in enumerate(coords):
                    f.write(
                        f"ATOM  {serial:5d}  CA  ALA A{i+1:4d}    "
                        f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           C  \n"
                    )
                    serial += 1
            elif coords.ndim == 3 and coords.shape[1:] == (3, 3):
                atom_names = ["N ", "CA", "C "]
                elem = ["N", "C", "C"]
                for i in range(coords.shape[0]):
                    for a in range(3):
                        pos = coords[i, a]
                        f.write(
                            f"ATOM  {serial:5d} {atom_names[a]:>3s} ALA A{i+1:4d}    "
                            f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           {elem[a]:1s}  \n"
                        )
                        serial += 1
            else:
                raise ValueError("coords must be (L,3) for CA trace or (L,3,3) for N-CA-C backbone")
        print(f"✅ Saved backbone to {filename}")

    # -------------------------
    # Quality Control Utilities
    # -------------------------
    def _min_inter_residue_distances(self, backbone: np.ndarray, skip_near: int = 2) -> np.ndarray:
        """Compute a matrix D where D[i,j] is the minimum atom-atom distance between
        residues i and j (over N, CA, C), with D[i,j] = inf for |i-j| <= skip_near.

        This is a vectorized implementation replacing nested loops.
        """
        # Optional acceleration via torch on non-CPU devices (controlled by env)
        try:
            use_accel = bool(int(os.getenv('AGI_ACCEL_TORCH', '0')))
        except Exception:
            use_accel = False

        if use_accel:
            try:
                import torch  # already imported at top
                from bio.devices import get_device
                dev = get_device()
                if dev.type in ("cuda", "mps"):
                    P = torch.as_tensor(backbone, dtype=torch.float32, device=dev)  # (L,3,3)
                    L = P.shape[0]
                    A = P[:, None, :, None, :]  # (L,1,3,1,3)
                    B = P[None, :, None, :, :]  # (1,L,1,3,3)
                    diff = A - B
                    dist = torch.linalg.vector_norm(diff, dim=-1)  # (L,L,3,3)
                    D = dist.min(dim=2).values.min(dim=2).values  # (L,L)
                    # Mask near neighbors
                    idx = torch.arange(L, device=dev)
                    ii, jj = torch.meshgrid(idx, idx, indexing='ij')
                    mask_near = (torch.abs(ii - jj) <= int(skip_near))
                    D = D.masked_fill(mask_near, float('inf'))
                    return D.detach().to('cpu').numpy()
            except Exception:
                # Fallback to numpy path on any error
                pass

        # Numpy vectorized baseline (default path)
        L = backbone.shape[0]
        P = backbone.astype(np.float32)
        A = P[:, None, :, None, :]
        B = P[None, :, None, :, :]
        diff = A - B  # (L, L, 3, 3, 3)
        dist = np.linalg.norm(diff, axis=-1)  # (L, L, 3, 3)
        D = dist.min(axis=(2, 3))  # (L, L)
        idx = np.arange(L)
        ii, jj = np.meshgrid(idx, idx, indexing='ij')
        mask_near = (np.abs(ii - jj) <= int(skip_near))
        D[mask_near] = np.inf
        return D
    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        # angle ABC in degrees
        v1 = a - b
        v2 = c - b
        v1 /= (np.linalg.norm(v1) + 1e-8)
        v2 /= (np.linalg.norm(v2) + 1e-8)
        cosang = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cosang)))

    def quality_check(self, backbone: np.ndarray, phi: np.ndarray, psi: np.ndarray, modes: list[str]) -> dict:
        """
        Computes a simple quality assessment for the generated backbone.
        Checks:
          - Bond lengths: N-CA, CA-C, C-N(next)
          - Bond angles: N-CA-C, CA-C-N(next), C-N-CA(next)
          - CA-CA distances
          - Simple clash detection (short inter-atomic distances)
        Returns a report dictionary.
        """
        L = backbone.shape[0]
        N = backbone[:, 0, :]
        CA = backbone[:, 1, :]
        C = backbone[:, 2, :]

        tol_len = 0.12  # Å
        tol_ang = 8.0   # deg
        ideal = IDEAL_GEOMETRY

        len_N_CA = [self._dist(N[i], CA[i]) for i in range(L)]
        len_CA_C = [self._dist(CA[i], C[i]) for i in range(L)]
        len_C_Np = [self._dist(C[i], N[i+1]) for i in range(L-1)]

        ang_N_CA_C = [self._angle(N[i], CA[i], C[i]) for i in range(L)]
        ang_CA_C_Np = [self._angle(CA[i], C[i], N[i+1]) for i in range(L-1)]
        ang_C_N_CA = [self._angle(C[i-1], N[i], CA[i]) for i in range(1, L)]

        ca_ca = [self._dist(CA[i-1], CA[i]) for i in range(1, L)]

        # Deviations
        dev_len_N_CA = [abs(x - ideal["N-CA"]) for x in len_N_CA]
        dev_len_CA_C = [abs(x - ideal["CA-C"]) for x in len_CA_C]
        dev_len_C_Np = [abs(x - ideal["C-N"]) for x in len_C_Np]

        dev_ang_N_CA_C = [abs(x - ideal["N-CA-C_angle"]) for x in ang_N_CA_C]
        dev_ang_CA_C_Np = [abs(x - ideal["CA-C-N_angle"]) for x in ang_CA_C_Np]
        dev_ang_C_N_CA = [abs(x - ideal["C-N-CA_angle"]) for x in ang_C_N_CA]

        out_len = (
            sum(d > tol_len for d in dev_len_N_CA) +
            sum(d > tol_len for d in dev_len_CA_C) +
            sum(d > tol_len for d in dev_len_C_Np)
        )
        out_ang = (
            sum(d > tol_ang for d in dev_ang_N_CA_C) +
            sum(d > tol_ang for d in dev_ang_CA_C_Np) +
            sum(d > tol_ang for d in dev_ang_C_N_CA)
        )

        # Simple clash detection: vectorized via inter-residue min distances
        clash_threshold = 2.0  # Å
        Dmin = self._min_inter_residue_distances(backbone, skip_near=2)  # allow i+3 and beyond
        # Compute total number of clashes (not truncated)
        clash_mask = np.triu((Dmin < clash_threshold), k=1)
        total_clashes = int(np.sum(clash_mask))
        # Find indices for detailed listing and truncate to 50 entries
        ii, jj = np.where(clash_mask)
        clashes = [(int(i), int(j), float(Dmin[i, j])) for i, j in zip(ii, jj)][:50]
        # Orthogonality index: fraction of non-neighbor residue pairs whose min atom-atom distance
        # exceeds a safe separation threshold (use the same repulsive threshold as in dissonance, 3.2 Å)
        safe_thr = 3.2
        valid = np.isfinite(Dmin)
        total_pairs = int(np.sum(valid))
        if total_pairs > 0:
            orth_frac = float(np.sum((Dmin >= safe_thr) & valid) / total_pairs)
        else:
            orth_frac = 0.0

        # Chirality export (best effort)
        try:
            chiral_map = [getattr(h, 'name', str(h)) for h in getattr(self, 'handedness', [Handedness.RIGHT] * L)]
        except Exception:
            chiral_map = ["RIGHT"] * L
        # Gesture export (best effort)
        try:
            if getattr(self, 'movements', None) is not None:
                gesture_map = [str(getattr(getattr(m, 'gesture', None), 'name', 'UNKNOWN')) for m in self.movements]
            else:
                gesture_map = []
        except Exception:
            gesture_map = []

        report = {
            "lengths": {
                "N-CA": len_N_CA,
                "CA-C": len_CA_C,
                "C-N_next": len_C_Np,
            },
            "angles": {
                "N-CA-C": ang_N_CA_C,
                "CA-C-N_next": ang_CA_C_Np,
                "C-N-CA": ang_C_N_CA,
            },
            "ca_ca": ca_ca,
            "phi": phi.tolist(),
            "psi": psi.tolist(),
            "modes": modes,
            "summary": {
                "out_of_range_lengths": int(out_len),
                "out_of_range_angles": int(out_ang),
                "num_clashes": int(total_clashes),
                "min_ca_ca": float(min(ca_ca) if ca_ca else 0.0),
                "max_ca_ca": float(max(ca_ca) if ca_ca else 0.0),
                "orthogonality_index": float(orth_frac),
            },
            "clashes": clashes,
            "chirality_map": chiral_map,
            "gesture_map": gesture_map,
        }
        # Optional sanity audit and policy version (non-breaking addition)
        try:
            if audit_sanity is not None:
                me = __file__
                san1 = audit_sanity(me)
                # Also audit notation if available
                if notation is not None and getattr(notation, '__file__', None):
                    san2 = audit_sanity(getattr(notation, '__file__'))
                    anchor_ratio = float((san1.get('sanity_score', 1.0) + san2.get('sanity_score', 1.0)) / 2.0)
                else:
                    anchor_ratio = float(san1.get('sanity_score', 1.0))
            else:
                anchor_ratio = 1.0
        except Exception:
            anchor_ratio = 1.0

        policy_version = None
        try:
            if notation is not None:
                policy_version = str(notation.REFINEMENT_POLICY.get('version'))
        except Exception:
            policy_version = None

        report["sanity"] = {
            "anchor_ratio": anchor_ratio,
            "policy_version": policy_version,
        }
        return report

    # -------------------------
    # Dissonance and Refinement
    # -------------------------
    def calculate_dissonance(self, backbone: np.ndarray, torsions: np.ndarray, modes: list[str], weights: dict) -> float:
        """Calculates a weighted Dissonance Score for a given conformation."""
        L = backbone.shape[0]
        N = backbone[:, 0, :]
        CA = backbone[:, 1, :]
        C = backbone[:, 2, :]

        # Physical: CA-CA deviation from 3.8 Å (legacy term)
        ca_ca = np.array([np.linalg.norm(CA[i] - CA[i-1]) for i in range(1, L)], dtype=np.float32)
        E_ca = float(np.mean((ca_ca - 3.8) ** 2)) if L > 1 else 0.0

        # New: Neighbor CA-CA tether (explicit, identical target 3.8 Å)
        E_neighbor_ca = float(np.mean((ca_ca - 3.8) ** 2)) if L > 1 else 0.0

        # Physical: repulsive for non-adjacent atoms closer than 3.2 Å (vectorized) on min-atom distances
        thr = 3.2
        Dmin = self._min_inter_residue_distances(backbone, skip_near=2)
        # Only distances below threshold contribute
        mask = Dmin < thr
        # Exclude infs (masked near neighbors)
        contrib = (thr - Dmin[mask]) ** 2 if np.any(mask) else np.array([], dtype=np.float32)
        rep = float(contrib.sum())
        E_clash = rep / max(1, L)

        # New: Non-adjacent CA-CA repulsion using softplus((r_cut - d)^2), averaged over number of clashes
        if L > 2:
            # Pairwise CA-CA distances for |i-j| > 1
            CAi = CA[:, None, :]  # (L,1,3)
            CAj = CA[None, :, :]  # (1,L,3)
            dCA = np.linalg.norm(CAi - CAj, axis=-1)  # (L,L)
            idx = np.arange(L)
            ii, jj = np.meshgrid(idx, idx, indexing='ij')
            nonadj = np.abs(ii - jj) > 1
            valid = nonadj
            dCA_valid = dCA[valid]
            clashes_mask = dCA_valid < thr
            d_clash = dCA_valid[clashes_mask]
            if d_clash.size > 0:
                x = (thr - d_clash) ** 2
                # softplus(x) = log(1 + exp(x))
                soft = np.log1p(np.exp(x))
                E_nonadj_ca = float(np.mean(soft))
            else:
                E_nonadj_ca = 0.0
        else:
            E_nonadj_ca = 0.0

        # Harmony: smoothness (finite difference of torsions)
        t = np.asarray(torsions, dtype=np.float32)
        if t.shape[0] > 1:
            dt = t[1:] - t[:-1]
            E_smooth = float(np.mean(dt ** 2))
        else:
            E_smooth = 0.0

        # New: Dihedral smoothness with proper angle wrapping to [-180, 180]
        def wrap180(a: np.ndarray) -> np.ndarray:
            return ((a + 180.0) % 360.0) - 180.0
        if t.shape[0] > 1:
            phi = wrap180(t[:, 0])
            psi = wrap180(t[:, 1])
            dphi = wrap180(phi[1:] - phi[:-1])
            dpsi = wrap180(psi[1:] - psi[:-1])
            E_dihedral = float(np.mean(np.concatenate([dphi**2, dpsi**2], axis=0)))
        else:
            E_dihedral = 0.0

        # Harmony: snap distance to nearest bin per mode
        from .scale_and_meter import SCALE_TABLE
        snap_errs = []
        for i in range(L):
            bins = SCALE_TABLE[modes[i]]
            diff = bins - t[i]
            d = np.min(np.linalg.norm(diff, axis=1))
            snap_errs.append(d)
        E_snap = float(np.mean(np.square(snap_errs))) if snap_errs else 0.0

        w = weights or {}
        total = (
            w.get('ca', 0.0) * E_ca +
            w.get('clash', 0.0) * E_clash +
            w.get('smooth', 0.0) * E_smooth +
            w.get('snap', 0.0) * E_snap +
            w.get('neighbor_ca', 0.0) * E_neighbor_ca +
            w.get('nonadj_ca', 0.0) * E_nonadj_ca +
            w.get('dihedral', 0.0) * E_dihedral
        )
        return float(total)

    def refine_torsions(self, phi_psi_initial: list, modes: list[str], sequence: str,
                       max_iters: int = 150, step_deg: float = 2.0, seed: int | None = None,
                       weights: dict | None = None, patience: int = 50,
                       *,
                       phaseA_frac: float = 0.4,
                       step_deg_clash: float | None = None,
                       clash_weight: float | None = None,
                       steric_only_phaseA: bool = True,
                       final_attempts: int = 2000,
                       final_step: float = 5.0,
                       final_window_increment: int = 25,
                       # Spacing control knobs
                       neighbor_threshold: float = 3.2,
                       spacing_max_attempts: int = 300,
                       spacing_top_bins: int = 4,
                       spacing_continue_full: bool = False,
                       spacing_cross_mode: bool = False,
                       critical_override_iters: int = 0,
                       # Parallelism
                       num_workers: int = 0,
                       eval_batch: int = 256,
                       # Debug/trace controls
                       debug_trace_path: str | None = None,
                       debug_log_every: int = 25,
                       debug_verbose: bool = False,
                       wall_timeout_sec: float | None = None) -> tuple[list, np.ndarray]:
        """
        Refines a set of torsion angles to minimize dissonance while staying on-key.
        Returns (refined_torsions_list, refined_backbone)
        """
        torsions = np.array(phi_psi_initial, dtype=np.float32)
        L = torsions.shape[0]
        rng = np.random.default_rng(seed)
        if weights is None:
            weights = {'ca': 1.0, 'clash': 1.5, 'smooth': 0.2, 'snap': 0.5}

        # Auto workers fallback
        try:
            if int(num_workers) <= 0:
                cpu = os.cpu_count() or 2
                num_workers = max(1, cpu - 1)
        except Exception:
            num_workers = 1

        # Setup tracing and timeout
        t0 = time.time()
        trace_f = None
        def _trace(event: dict):
            nonlocal trace_f
            if debug_trace_path is None:
                return
            try:
                if trace_f is None:
                    os.makedirs(os.path.dirname(debug_trace_path) or '.', exist_ok=True)
                    trace_f = open(debug_trace_path, 'a', buffering=1)
                event = dict(event)
                event['ts'] = time.time()
                trace_f.write(json.dumps(event) + "\n")
                trace_f.flush()
            except Exception:
                pass

        # Evaluate initial
        bb_curr = self.build_backbone_from_torsions(torsions, sequence)
        best_score = self.calculate_dissonance(bb_curr, torsions, modes, weights)
        qc_curr = self.quality_check(bb_curr, np.array([t[0] for t in torsions]), np.array([t[1] for t in torsions]), modes)
        best_num_clashes = qc_curr['summary']['num_clashes']
        best_min_caca = qc_curr['summary']['min_ca_ca']

        iters_since_improvement = 0

        _trace({
            'event': 'start',
            'L': int(L),
            'max_iters': int(max_iters),
            'step_deg': float(step_deg),
            'weights': weights,
            'phaseA_frac': float(phaseA_frac),
            'seed': int(seed) if seed is not None else None,
            'qc_initial': qc_curr['summary'],
        })

        # Stage 1: Spacing Pass (Physical Integrity First)
        # Push all adjacent CA-CA distances above neighbor_threshold by discrete note jumps
        # Use provided spacing controls
        neighbor_threshold = float(neighbor_threshold)
        max_spacing_attempts = int(spacing_max_attempts)
        spacing_attempts = 0
        bins_tried = 0
        from .scale_and_meter import SCALE_TABLE

        def adjacent_pairs_below_thr(backbone_arr: np.ndarray, thr: float) -> list[tuple[int, int]]:
            CA_loc = backbone_arr[:, 1, :]
            d = np.linalg.norm(CA_loc[1:] - CA_loc[:-1], axis=1)
            pairs = []
            for k in range(1, len(d) + 1):
                if d[k - 1] < thr:
                    pairs.append((k - 1, k))
            return pairs

        def global_min_caca(backbone_arr: np.ndarray) -> float:
            CA_loc = backbone_arr[:, 1, :]
            if CA_loc.shape[0] < 2:
                return float('inf')
            d = np.linalg.norm(CA_loc[1:] - CA_loc[:-1], axis=1)
            return float(np.min(d)) if d.size > 0 else float('inf')

        pairs = adjacent_pairs_below_thr(bb_curr, neighbor_threshold)
        while pairs and spacing_attempts < max_spacing_attempts:
            if wall_timeout_sec is not None and (time.time() - t0) > float(wall_timeout_sec):
                _trace({'event': 'timeout_abort', 'phase': 'spacing', 'spacing_attempts': int(spacing_attempts)})
                break
            spacing_attempts += 1
            improved = False
            # Work each problematic adjacent pair
            for (ia, ib) in pairs:
                # Enumerate up to K nearest bins for each residue in the pair
                def top_bins(bins: np.ndarray, curr: np.ndarray, top: int = spacing_top_bins) -> np.ndarray:
                    d = np.linalg.norm(bins - curr, axis=1)
                    ords = np.argsort(d)
                    return bins[ords[:min(top, len(ords))]]

                if spacing_cross_mode:
                    bins_union = np.vstack([SCALE_TABLE['helix'], SCALE_TABLE['sheet'], SCALE_TABLE['loop']])
                    bins_a = bins_union
                    bins_b = bins_union
                else:
                    bins_a = SCALE_TABLE[modes[ia]]
                    bins_b = SCALE_TABLE[modes[ib]]
                A = top_bins(bins_a, torsions[ia])
                B = top_bins(bins_b, torsions[ib])

                best_local_min = global_min_caca(bb_curr)
                best_tmp = None
                # Prepare candidates
                cand_tors_list: list[np.ndarray] = []
                for a in A:
                    for b in B:
                        tmp = torsions.copy()
                        if spacing_cross_mode:
                            tmp[ia, 0] = float(a[0]); tmp[ia, 1] = float(a[1])
                            tmp[ib, 0] = float(b[0]); tmp[ib, 1] = float(b[1])
                        else:
                            sphi_a, spsi_a = snap_to_scale(modes[ia], float(a[0]), float(a[1]))
                            tmp[ia, 0] = sphi_a
                            tmp[ia, 1] = spsi_a
                            sphi_b, spsi_b = snap_to_scale(modes[ib], float(b[0]), float(b[1]))
                            tmp[ib, 0] = sphi_b
                            tmp[ib, 1] = spsi_b
                        cand_tors_list.append(tmp)
                bins_tried += len(cand_tors_list)

                def _eval_min(batch: list[np.ndarray]) -> tuple[float, int]:
                    best_val = best_local_min
                    best_idx = -1
                    for idx, cand in enumerate(batch):
                        bb_tmp = self.build_backbone_from_torsions(cand, sequence)
                        mmin = global_min_caca(bb_tmp)
                        if mmin > best_val:
                            best_val = mmin
                            best_idx = idx
                    return best_val, best_idx

                if int(num_workers) > 1 and len(cand_tors_list) >= 2:
                    # Batch the candidates
                    batches = [cand_tors_list[i:i+int(eval_batch)] for i in range(0, len(cand_tors_list), int(eval_batch))]
                    best_val = best_local_min
                    best_pair_idx = -1
                    offset = 0
                    with ProcessPoolExecutor(max_workers=int(num_workers)) as ex:
                        # Evaluate in parallel by mapping over small batches to reduce IPC
                        results = list(ex.map(_min_caca_for_torsions_batch, [(sequence, batch) for batch in batches]))
                    for bidx, (val, rel_idx) in enumerate(results):
                        if rel_idx >= 0 and val > best_val:
                            best_val = val
                            best_pair_idx = offset + rel_idx
                        offset += len(batches[bidx])
                    if best_pair_idx >= 0:
                        best_local_min = best_val
                        best_tmp = cand_tors_list[best_pair_idx]
                else:
                    # Sequential fallback
                    for idx, cand in enumerate(cand_tors_list):
                        bb_tmp = self.build_backbone_from_torsions(cand, sequence)
                        mmin = global_min_caca(bb_tmp)
                        if mmin > best_local_min:
                            best_local_min = mmin
                            best_tmp = cand

                # Accept immediately if we found a combination that increases global min CA-CA
                if best_tmp is not None:
                    torsions = best_tmp
                    bb_curr = self.build_backbone_from_torsions(torsions, sequence)
                    best_min_caca = best_local_min
                    improved = True
                    # Reset Phase trackers as we changed state
                    iters_since_improvement = 0
            if not improved:
                if not spacing_continue_full:
                    break
            # Recompute list of problematic pairs
            pairs = adjacent_pairs_below_thr(bb_curr, neighbor_threshold)

        # record spacing stats on the instance for external reporting
        try:
            self.last_refine_stats = {
                'spacing_attempts': int(spacing_attempts),
                'bins_tried': int(bins_tried),
                'final_min_ca_ca_after_spacing': float(global_min_caca(bb_curr)),
            }
        except Exception:
            self.last_refine_stats = {'spacing_attempts': int(spacing_attempts), 'bins_tried': int(bins_tried)}

        _trace({
            'event': 'spacing_done',
            'spacing_attempts': int(spacing_attempts),
            'bins_tried': int(bins_tried),
            'final_min_ca_ca': float(self.last_refine_stats.get('final_min_ca_ca_after_spacing', 0.0)),
        })

        # Precompute contiguous same-mode segments for meter-aware updates
        segments: list[tuple[int, int, str]] = []  # (start, end, mode) with end exclusive
        s = 0
        while s < L:
            m = modes[s]
            e = s
            while e < L and modes[e] == m:
                e += 1
            segments.append((s, e, m))
            s = e

        # Phase A: clash-focused
        phaseA_iters = max(1, int(max_iters * float(phaseA_frac)))
        wA = dict(weights)
        wA['clash'] = float(clash_weight) if clash_weight is not None else max(wA.get('clash', 1.5), 10.0)
        stepA = float(step_deg_clash) if step_deg_clash is not None else max(float(step_deg), 3.5)
        # Phase-A scoring weights (steric-only if requested)
        weightsA = {'ca': 0.0, 'smooth': 0.0, 'snap': 0.0, 'clash': wA['clash']} if steric_only_phaseA else wA

        for i in range(phaseA_iters):
            if wall_timeout_sec is not None and (time.time() - t0) > float(wall_timeout_sec):
                _trace({'event': 'timeout_abort', 'phase': 'A', 'iter': int(i)})
                break
            progress = i / max(1, phaseA_iters)
            current_step = stepA * (1.0 - 0.5 * progress)  # gentler anneal in clash phase

            # Meter-aware proposal: select one contiguous same-mode segment
            seg_idx = int(rng.integers(0, len(segments)))
            seg_start, seg_end, seg_mode = segments[seg_idx]
            span = seg_end - seg_start
            if span <= 0:
                continue
            prop = torsions.copy()
            # Note-jump proposal: choose an alternate allowed bin per residue, then snap
            from .scale_and_meter import SCALE_TABLE
            for idx in range(seg_start, seg_end):
                mode_i = modes[idx]
                bins = SCALE_TABLE[mode_i]
                curr = torsions[idx]
                # Distance of current torsion to each allowed bin
                dists = np.linalg.norm(bins - curr, axis=1)
                order = np.argsort(dists)
                # Prefer the second-best allowed bin if available; otherwise keep best
                cand_idx = int(order[1]) if order.shape[0] > 1 else int(order[0])
                cand = bins[cand_idx].astype(np.float32)
                # Small jitter to encourage escaping plateaus
                jitter = rng.uniform(-current_step, current_step, size=(2,)).astype(np.float32)
                trial = cand + jitter
                sphi, spsi = snap_to_scale(mode_i, float(trial[0]), float(trial[1]))
                prop[idx, 0] = sphi
                prop[idx, 1] = spsi

            bb_prop = self.build_backbone_from_torsions(prop, sequence)
            score_curr = self.calculate_dissonance(bb_curr, torsions, modes, weightsA)
            score_prop = self.calculate_dissonance(bb_prop, prop, modes, weightsA)
            qc_prop = self.quality_check(bb_prop, np.array([t[0] for t in prop]), np.array([t[1] for t in prop]), modes)
            reduce_clash = qc_prop['summary']['num_clashes'] < best_num_clashes
            increase_min = qc_prop['summary']['min_ca_ca'] > best_min_caca
            accepted = (score_prop < score_curr) or reduce_clash or increase_min
            if (debug_verbose or (i % max(1, int(debug_log_every)) == 0)):
                _trace({'event': 'phaseA_iter', 'iter': int(i), 'progress': float(progress), 'current_step': float(current_step),
                        'score_curr': float(score_curr), 'score_prop': float(score_prop), 'accepted': bool(accepted),
                        'reduce_clash': bool(reduce_clash), 'increase_min': bool(increase_min),
                        'best_num_clashes': int(best_num_clashes), 'best_min_caca': float(best_min_caca)})
            if accepted:
                torsions = prop
                bb_curr = bb_prop
                best_score = self.calculate_dissonance(bb_curr, torsions, modes, weights)  # track balanced score
                qc_curr = qc_prop
                best_num_clashes = qc_curr['summary']['num_clashes']
                best_min_caca = qc_curr['summary']['min_ca_ca']
                iters_since_improvement = 0
            else:
                iters_since_improvement += 1
            if iters_since_improvement >= patience:
                break

        # Phase B: balanced refinement
        remain = max_iters - phaseA_iters
        if remain > 0:
            for j in range(remain):
                if wall_timeout_sec is not None and (time.time() - t0) > float(wall_timeout_sec):
                    _trace({'event': 'timeout_abort', 'phase': 'B', 'iter': int(j)})
                    break
                progress = j / max(1, remain)
                current_step = float(step_deg) * (1.0 - progress)

                # Meter-aware proposal: select one contiguous same-mode segment
                seg_idx = int(rng.integers(0, len(segments)))
                seg_start, seg_end, seg_mode = segments[seg_idx]
                span = seg_end - seg_start
                if span <= 0:
                    continue
                prop = torsions.copy()
                # Note-jump proposal: choose alternate allowed bins in this segment
                from .scale_and_meter import SCALE_TABLE
                for idx in range(seg_start, seg_end):
                    mode_i = modes[idx]
                    bins = SCALE_TABLE[mode_i]
                    curr = torsions[idx]
                    dists = np.linalg.norm(bins - curr, axis=1)
                    order = np.argsort(dists)
                    cand_idx = int(order[1]) if order.shape[0] > 1 else int(order[0])
                    cand = bins[cand_idx].astype(np.float32)
                    jitter = rng.uniform(-current_step, current_step, size=(2,)).astype(np.float32)
                    trial = cand + jitter
                    sphi, spsi = snap_to_scale(mode_i, float(trial[0]), float(trial[1]))
                    prop[idx, 0] = sphi
                    prop[idx, 1] = spsi

                bb_prop = self.build_backbone_from_torsions(prop, sequence)
                score_prop = self.calculate_dissonance(bb_prop, prop, modes, weights)
                if score_prop < best_score:
                    torsions = prop
                    bb_curr = bb_prop
                    best_score = score_prop
                    # Update QC baselines occasionally to allow objective drift toward better geometry
                    qc_curr = self.quality_check(bb_curr, np.array([t[0] for t in torsions]), np.array([t[1] for t in torsions]), modes)
                    best_num_clashes = min(best_num_clashes, qc_curr['summary']['num_clashes'])
                    best_min_caca = max(best_min_caca, qc_curr['summary']['min_ca_ca'])
                    iters_since_improvement = 0
                else:
                    # Multi-objective acceptance in balanced phase too
                    qc_prop = self.quality_check(bb_prop, np.array([t[0] for t in prop]), np.array([t[1] for t in prop]), modes)
                    reduce_clash = qc_prop['summary']['num_clashes'] < qc_curr['summary']['num_clashes']
                    increase_min = qc_prop['summary']['min_ca_ca'] > qc_curr['summary']['min_ca_ca']
                    if reduce_clash or increase_min:
                        torsions = prop
                        bb_curr = bb_prop
                        qc_curr = qc_prop
                        best_num_clashes = min(best_num_clashes, qc_curr['summary']['num_clashes'])
                        best_min_caca = max(best_min_caca, qc_curr['summary']['min_ca_ca'])
                        # balanced score update
                        best_score = min(best_score, self.calculate_dissonance(bb_curr, torsions, modes, weights))
                        iters_since_improvement = 0
                    else:
                        iters_since_improvement += 1

                if iters_since_improvement >= patience:
                    print(f"   - Rehearsal converged after {phaseA_iters + j + 1} iterations.")
                    break

        # Chiral inversion attempt: consciously try LEFT-handed local inversion near tightest CA-CA pair
        try:
            CA_now = bb_curr[:, 1, :]
            ca_d = np.linalg.norm(CA_now[1:] - CA_now[:-1], axis=1)
            if ca_d.size > 0:
                tight_k = int(np.argmin(ca_d) + 1)
                local_span = 2
                idxs = np.unique(list(range(max(0, tight_k - local_span), min(L, tight_k + local_span + 1))))
                prop = torsions.copy()
                for idx in idxs:
                    # invert sign then resnap to scale (mirror path)
                    inv = -prop[idx]
                    sphi, spsi = snap_to_scale(modes[idx], float(inv[0]), float(inv[1]))
                    prop[idx, 0] = sphi
                    prop[idx, 1] = spsi
                bb_prop = self.build_backbone_from_torsions(prop, sequence)
                score_curr = self.calculate_dissonance(bb_curr, torsions, modes, weights)
                score_prop = self.calculate_dissonance(bb_prop, prop, modes, weights)
                qc_prop = self.quality_check(bb_prop, np.array([t[0] for t in prop]), np.array([t[1] for t in prop]), modes)
                qc_curr = self.quality_check(bb_curr, np.array([t[0] for t in torsions]), np.array([t[1] for t in torsions]), modes)
                reduce_clash = qc_prop['summary']['num_clashes'] < qc_curr['summary']['num_clashes']
                increase_min = qc_prop['summary']['min_ca_ca'] > qc_curr['summary']['min_ca_ca']
                if (score_prop < score_curr) or reduce_clash or increase_min:
                    torsions = prop
                    bb_curr = bb_prop
                    # Update handedness map for these residues to LEFT
                    try:
                        if not hasattr(self, 'handedness') or len(self.handedness) != L:
                            self.handedness = [Handedness.RIGHT] * L
                        for idx in idxs:
                            self.handedness[idx] = Handedness.LEFT
                            # also reflect in Movement intents if available
                            if getattr(self, 'movements', None) is not None and 0 <= int(idx) < len(self.movements):
                                self.movements[int(idx)].hand_override = Handedness.LEFT
                    except Exception:
                        pass
        except Exception:
            pass

        # Final authoritative pass: targeted clash resolution
        # If clashes remain, aggressively fix them by local torsion tweaks
        rep_weights = dict(weights)
        rep_weights['clash'] = float(clash_weight) if clash_weight is not None else max(rep_weights.get('clash', 1.5), 10.0)
        # Evaluate current clashes using QC
        rphi = np.array([t[0] for t in torsions])
        rpsi = np.array([t[1] for t in torsions])
        qc = self.quality_check(bb_curr, rphi, rpsi, modes)
        clashes = qc['clashes']
        attempts = 0
        max_attempts = int(final_attempts)
        local_window = 2
        while clashes and attempts < max_attempts:
            # Work on the worst clash (smallest distance)
            i, j, dmin = min(clashes, key=lambda x: x[2])
            # Propose localized updates around i and j
            idxs = list(range(max(0, i - local_window), min(L, i + local_window + 1))) + \
                   list(range(max(0, j - local_window), min(L, j + local_window + 1)))
            idxs = np.unique(idxs)
            prop = torsions.copy()
            # Use a modest step that can still move apart
            step = float(final_step)
            rng = np.random.default_rng(seed)
            deltas = rng.uniform(-step, step, size=(len(idxs), 2)).astype(np.float32)
            for k_idx, idx in enumerate(idxs):
                prop[idx] += deltas[k_idx]
                sphi, spsi = snap_to_scale(modes[idx], float(prop[idx, 0]), float(prop[idx, 1]))
                prop[idx, 0] = sphi
                prop[idx, 1] = spsi

            bb_prop = self.build_backbone_from_torsions(prop, sequence)
            score_curr = self.calculate_dissonance(bb_curr, torsions, modes, rep_weights)
            score_prop = self.calculate_dissonance(bb_prop, prop, modes, rep_weights)
            # Accept if improves dissonance OR specifically reduces number of clashes or increases min distance
            qc_prop = self.quality_check(bb_prop, np.array([t[0] for t in prop]), np.array([t[1] for t in prop]), modes)
            reduce_clash = qc_prop['summary']['num_clashes'] < qc['summary']['num_clashes']
            increase_min = qc_prop['summary']['min_ca_ca'] > qc['summary']['min_ca_ca']
            if score_prop < score_curr or reduce_clash or increase_min:
                torsions = prop
                bb_curr = bb_prop
                qc = qc_prop
                clashes = qc['clashes']
            else:
                attempts += 1
                # small adaptive change of window/step if stuck
                if attempts % max(1, int(final_window_increment)) == 0 and local_window < 4:
                    local_window += 1

        # Dedicated neighbor CA-CA repair pass: push min CA-CA above 3.2 Å
        try:
            CA = bb_curr[:, 1, :]
            ca_dists = np.linalg.norm(CA[1:] - CA[:-1], axis=1)
            min_idx = int(np.argmin(ca_dists) + 1)
            min_val = float(ca_dists[min_idx - 1]) if ca_dists.size > 0 else float('inf')
        except Exception:
            min_idx, min_val = 1, float('inf')
        target_thr = 3.2
        attempts2 = 0
        max_attempts2 = int(final_attempts)
        local_window2 = 2
        from .scale_and_meter import SCALE_TABLE
        while min_val < target_thr and attempts2 < max_attempts2:
            if wall_timeout_sec is not None and (time.time() - t0) > float(wall_timeout_sec):
                _trace({'event': 'timeout_abort', 'phase': 'final_repair', 'attempts2': int(attempts2)})
                break
            l = max(0, min_idx - local_window2 - 1)
            r = min(L, min_idx + local_window2 + 1)
            prop = torsions.copy()
            # Note-jump in the neighborhood of the tightest CA-CA pair
            for idx in range(l, r):
                mode_i = modes[idx]
                bins = SCALE_TABLE[mode_i]
                curr = torsions[idx]
                dists = np.linalg.norm(bins - curr, axis=1)
                order = np.argsort(dists)
                cand_idx = int(order[1]) if order.shape[0] > 1 else int(order[0])
                cand = bins[cand_idx].astype(np.float32)
                jitter = rng.uniform(-final_step, final_step, size=(2,)).astype(np.float32)
                trial = cand + jitter
                sphi, spsi = snap_to_scale(mode_i, float(trial[0]), float(trial[1]))
                prop[idx, 0] = sphi
                prop[idx, 1] = spsi

            bb_prop = self.build_backbone_from_torsions(prop, sequence)
            # Evaluate CA-CA improvement
            CAp = bb_prop[:, 1, :]
            ca_dp = np.linalg.norm(CAp[1:] - CAp[:-1], axis=1)
            minp = float(np.min(ca_dp)) if ca_dp.size > 0 else float('inf')
            improved = (minp > min_val) or (minp >= target_thr)
            if (debug_verbose or (attempts2 % max(1, int(debug_log_every)) == 0)):
                _trace({'event': 'final_repair_iter', 'attempts2': int(attempts2), 'span': int(r - l),
                        'min_val': float(min_val), 'minp': float(minp), 'improved': bool(improved)})
            if improved:
                torsions = prop
                bb_curr = bb_prop
                min_val = minp
                # Recompute min index for next round
                if ca_dp.size > 0:
                    min_idx = int(np.argmin(ca_dp) + 1)
            else:
                # Direct pairwise optimization over (k-1,k) bins
                k = int(min_idx)
                cand_prop = torsions.copy()
                best_local = min_val
                best_pair = None
                for iidx in [k-1, k]:
                    if iidx < 0 or iidx >= L:
                        continue
                # Build candidate sets
                idx_a = k - 1
                idx_b = k
                if 0 <= idx_a < L and 0 <= idx_b < L:
                    bins_a = SCALE_TABLE[modes[idx_a]]
                    bins_b = SCALE_TABLE[modes[idx_b]]
                    # Try up to 4 nearest bins for each
                    def top_bins(bins, curr, top=4):
                        d = np.linalg.norm(bins - curr, axis=1)
                        ords = np.argsort(d)
                        return bins[ords[:min(top, len(ords))]]
                    A = top_bins(bins_a, torsions[idx_a])
                    B = top_bins(bins_b, torsions[idx_b])
                    cand_list = []
                    for a in A:
                        for b in B:
                            tmp = torsions.copy()
                            sphi, spsi = snap_to_scale(modes[idx_a], float(a[0]), float(a[1]))
                            tmp[idx_a, 0] = sphi
                            tmp[idx_a, 1] = spsi
                            sphi2, spsi2 = snap_to_scale(modes[idx_b], float(b[0]), float(b[1]))
                            tmp[idx_b, 0] = sphi2
                            tmp[idx_b, 1] = spsi2
                            cand_list.append(tmp)
                    if int(num_workers) > 1 and cand_list:
                        batches = [cand_list[i:i+int(eval_batch)] for i in range(0, len(cand_list), int(eval_batch))]
                        with ProcessPoolExecutor(max_workers=int(num_workers)) as ex:
                            results = list(ex.map(_min_caca_for_torsions_batch, [(sequence, batch) for batch in batches]))
                        offset = 0
                        best_idx = -1
                        for bidx, (val, rel_idx) in enumerate(results):
                            if rel_idx >= 0 and val > best_local:
                                best_local = val
                                best_idx = offset + rel_idx
                            offset += len(batches[bidx])
                        if best_idx >= 0:
                            cand_prop = cand_list[best_idx]
                    else:
                        for tmp in cand_list:
                            bb_tmp = self.build_backbone_from_torsions(tmp, sequence)
                            CAx = bb_tmp[:, 1, :]
                            ca_dx = np.linalg.norm(CAx[1:] - CAx[:-1], axis=1)
                            if ca_dx.size > 0:
                                dmin = float(np.min(ca_dx))
                                if dmin > best_local:
                                    best_local = dmin
                                    cand_prop = tmp
                if best_local > min_val:
                    torsions = cand_prop
                    bb_curr = self.build_backbone_from_torsions(torsions, sequence)
                    min_val = best_local
                    if min_val >= target_thr:
                        break
                attempts2 += 1
                if attempts2 % max(1, int(final_window_increment)) == 0 and local_window2 < 4:
                    local_window2 += 1

        # Critical override: greedily maximize global min CA-CA regardless of other energies
        it_co = 0
        if int(critical_override_iters) > 0:
            while min_val < target_thr and it_co < int(critical_override_iters):
                if wall_timeout_sec is not None and (time.time() - t0) > float(wall_timeout_sec):
                    _trace({'event': 'timeout_abort', 'phase': 'critical_override', 'iter': int(it_co)})
                    break
                # Recompute tightest pair
                CA = bb_curr[:, 1, :]
                ca_dists = np.linalg.norm(CA[1:] - CA[:-1], axis=1)
                if ca_dists.size == 0:
                    break
                min_idx = int(np.argmin(ca_dists) + 1)
                min_val = float(ca_dists[min_idx - 1])
                k = min_idx
                idx_a = k - 1
                idx_b = k
                bins_union = np.vstack([SCALE_TABLE['helix'], SCALE_TABLE['sheet'], SCALE_TABLE['loop']])
                def top_bins_union(curr, top=spacing_top_bins):
                    d = np.linalg.norm(bins_union - curr, axis=1)
                    ords = np.argsort(d)
                    return bins_union[ords[:min(top, len(ords))]]
                A = top_bins_union(torsions[idx_a]) if 0 <= idx_a < L else np.empty((0,2), dtype=np.float32)
                B = top_bins_union(torsions[idx_b]) if 0 <= idx_b < L else np.empty((0,2), dtype=np.float32)
                best_local = min_val
                best_tmp = None
                for a in A:
                    for b in B:
                        tmp = torsions.copy()
                        tmp[idx_a, 0] = float(a[0]); tmp[idx_a, 1] = float(a[1])
                        tmp[idx_b, 0] = float(b[0]); tmp[idx_b, 1] = float(b[1])
                        bb_tmp = self.build_backbone_from_torsions(tmp, sequence)
                        CAx = bb_tmp[:, 1, :]
                        ca_dx = np.linalg.norm(CAx[1:] - CAx[:-1], axis=1)
                        if ca_dx.size > 0:
                            dmin = float(np.min(ca_dx))
                            if dmin > best_local:
                                best_local = dmin
                                best_tmp = (tmp, bb_tmp)
                _trace({'event': 'critical_override_iter', 'iter': int(it_co), 'min_val': float(min_val), 'best_local': float(best_local)})
                if best_tmp is not None:
                    torsions, bb_curr = best_tmp
                    min_val = best_local
                    if min_val >= target_thr:
                        break
                else:
                    # No improvement found; terminate early
                    break
                it_co += 1

        # Final event
        try:
            final_qc = self.quality_check(bb_curr, np.array([t[0] for t in torsions]), np.array([t[1] for t in torsions]), modes)
            _trace({'event': 'end', 'final_qc': final_qc['summary']})
        except Exception:
            pass
        if trace_f is not None:
            try: trace_f.close()
            except Exception: pass
        return torsions.tolist(), bb_curr

    # -------------------------
    # Reference Metrics (CA-only alignment)
    # -------------------------
    @staticmethod
    def _extract_ca(backbone: np.ndarray) -> np.ndarray:
        """Return CA coordinates (L,3) from a backbone shaped (L,3,3)."""
        assert backbone.ndim == 3 and backbone.shape[1] == 3 and backbone.shape[2] == 3, "backbone must be (L,3,3)"
        return backbone[:, 1, :]  # CA is index 1

    @staticmethod
    def _kabsch(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute optimal rotation R and translation t s.t. R@P + t ~= Q.
        P, Q: (L,3). Returns (R (3,3), t (3,))
        """
        assert P.shape == Q.shape and P.ndim == 2 and P.shape[1] == 3
        Pc = P - P.mean(axis=0)
        Qc = Q - Q.mean(axis=0)
        H = Pc.T @ Qc
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = Q.mean(axis=0) - R @ P.mean(axis=0)
        return R, t

    def align_ca(self, pred_backbone: np.ndarray, ref_ca: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Align predicted CA to reference CA via Kabsch.
        Returns (pred_aligned (L,3), ref_used (L,3), R@P+t).
        """
        P = self._extract_ca(pred_backbone)
        L = min(P.shape[0], ref_ca.shape[0])
        if L == 0:
            raise ValueError("Empty coordinates for alignment")
        P = P[:L]
        Q = ref_ca[:L]
        R, t = self._kabsch(P, Q)
        P_aligned = (R @ P.T).T + t
        return P_aligned, Q, P_aligned

    def rmsd_ca(self, pred_backbone: np.ndarray, ref_ca: np.ndarray) -> float:
        """Backbone CA RMSD after Kabsch alignment."""
        P = self._extract_ca(pred_backbone)
        L = min(P.shape[0], ref_ca.shape[0])
        if L == 0:
            return float("nan")
        P = P[:L]
        Q = ref_ca[:L]
        R, t = self._kabsch(P, Q)
        P_aligned = (R @ P.T).T + t
        diff2 = np.sum((P_aligned - Q) ** 2, axis=1)
        return float(np.sqrt(np.mean(diff2)))

    def lddt_ca(self, pred_backbone: np.ndarray, ref_ca: np.ndarray, cutoffs: list[float] | None = None) -> float:
        """Approximate lDDT using CA-CA distances only.
        Computes fraction of neighbor pairs whose distance error falls within thresholds.
        cutoffs in Angstroms; default [0.5,1,2,4] per standard lDDT bins.
        """
        if cutoffs is None:
            cutoffs = [0.5, 1.0, 2.0, 4.0]
        P = self._extract_ca(pred_backbone)
        L = min(P.shape[0], ref_ca.shape[0])
        if L < 3:
            return float("nan")
        P = P[:L]
        Q = ref_ca[:L]
        # Use aligned P to avoid rigid-body differences
        R, t = self._kabsch(P, Q)
        P = (R @ P.T).T + t
        # Pairwise distances within a neighborhood (|i-j| between 1 and 15)
        max_seq_sep = 15
        counts = 0
        score = 0.0
        for i in range(L):
            for j in range(i + 1, min(L, i + max_seq_sep + 1)):
                dP = np.linalg.norm(P[i] - P[j])
                dQ = np.linalg.norm(Q[i] - Q[j])
                err = abs(dP - dQ)
                # lDDT assigns partial credit across thresholds
                hits = sum(1 for c in cutoffs if err < c) / len(cutoffs)
                score += hits
                counts += 1
        if counts == 0:
            return float("nan")
        return float(score / counts)

    def tm_score_ca(self, pred_backbone: np.ndarray, ref_ca: np.ndarray) -> float:
        """Compute TM-score using CA atoms after Kabsch alignment.
        Uses standard d0 = 1.24*(L-15)^(1/3) - 1.8, clipped to >= 0.5.
        """
        P = self._extract_ca(pred_backbone)
        L = min(P.shape[0], ref_ca.shape[0])
        if L == 0:
            return float("nan")
        P = P[:L]
        Q = ref_ca[:L]
        R, t = self._kabsch(P, Q)
        P_aligned = (R @ P.T).T + t
        d2 = np.sum((P_aligned - Q) ** 2, axis=1)
        d0 = 1.24 * np.cbrt(max(L - 15, 1)) - 1.8
        d0 = max(d0, 0.5)
        tm = np.mean(1.0 / (1.0 + (np.sqrt(d2) / d0) ** 2))
        return float(tm)

    def evaluate_against_reference(self,
                                   pred_backbone: np.ndarray,
                                   ref_ca: np.ndarray,
                                   pred_phi: np.ndarray | None = None,
                                   pred_psi: np.ndarray | None = None,
                                   ref_phi: np.ndarray | None = None,
                                   ref_psi: np.ndarray | None = None) -> dict:
        """Compute standard metrics vs a reference CA trace and optional torsions.
        Returns a dict with keys: rmsd_ca, lddt_ca, tm_score_ca, torsion_mae (if available).
        """
        metrics: dict = {}
        try:
            metrics["rmsd_ca"] = self.rmsd_ca(pred_backbone, ref_ca)
        except Exception:
            metrics["rmsd_ca"] = float("nan")
        try:
            metrics["lddt_ca"] = self.lddt_ca(pred_backbone, ref_ca)
        except Exception:
            metrics["lddt_ca"] = float("nan")
        try:
            metrics["tm_score_ca"] = self.tm_score_ca(pred_backbone, ref_ca)
        except Exception:
            metrics["tm_score_ca"] = float("nan")

        # Torsion MAE if both provided
        if (pred_phi is not None and pred_psi is not None and
                ref_phi is not None and ref_psi is not None):
            L = min(len(pred_phi), len(ref_phi))
            if L > 0:
                dphi = np.abs((pred_phi[:L] - ref_phi[:L]))
                dpsi = np.abs((pred_psi[:L] - ref_psi[:L]))
                # wrap to [-180,180]
                dphi = np.minimum(dphi, 360.0 - dphi)
                dpsi = np.minimum(dpsi, 360.0 - dpsi)
                metrics["torsion_mae"] = float((np.mean(dphi) + np.mean(dpsi)) / 2.0)
        return metrics
