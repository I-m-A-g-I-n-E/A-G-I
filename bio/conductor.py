import numpy as np
import torch
from .geometry import INTERVAL_TO_TORSION, IDEAL_GEOMETRY, place_next_atom
from .key_estimator import estimate_key_and_modes
from .scale_and_meter import snap_to_scale, enforce_meter

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

        # Convert to phi/psi arrays with the convention used by the NERF builder
        phi = np.zeros(num_residues)
        psi = np.zeros(num_residues)
        # Use phi[i] for residue i and psi[i-1] for peptide between i-1 and i
        for i in range(num_residues):
            ph, ps = final_torsions[i]
            phi[i] = ph
            if i > 0:
                psi[i-1] = ps
        # Reasonable defaults for termini
        if num_residues > 1:
            psi[num_residues - 1] = psi[num_residues - 2]
            phi[0] = phi[1]
        else:
            psi[0] = -47.0
            phi[0] = -57.0

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

        # Build subsequent residues using NERF
        for i in range(1, num_residues):
            # Place N(i) using (N(i-1), CA(i-1), C(i-1)) and psi(i-1)
            Ni = place_next_atom(
                N_coords[i - 1], CA_coords[i - 1], C_coords[i - 1],
                length=b_C_N,
                angle=ang_CA_C_N,
                torsion=psi[i - 1],
            )
            # Place CA(i) using (CA(i-1), C(i-1), N(i)) and omega
            CAi = place_next_atom(
                CA_coords[i - 1], C_coords[i - 1], Ni,
                length=b_N_CA,
                angle=ang_C_N_CA,
                torsion=omega,
            )
            # Place C(i) using (C(i-1), N(i), CA(i)) and phi(i)
            Ci = place_next_atom(
                C_coords[i - 1], Ni, CAi,
                length=b_CA_C,
                angle=ang_N_CA_C,
                torsion=phi[i],
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
            Ni = place_next_atom(N_coords[i - 1], CA_coords[i - 1], C_coords[i - 1],
                                 length=b_C_N, angle=ang_CA_C_N, torsion=psi[i - 1])
            CAi = place_next_atom(CA_coords[i - 1], C_coords[i - 1], Ni,
                                  length=b_N_CA, angle=ang_C_N_CA, torsion=omega)
            Ci = place_next_atom(C_coords[i - 1], Ni, CAi,
                                 length=b_CA_C, angle=ang_N_CA_C, torsion=phi[i])
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

        # Simple clash detection: check minimum distance between heavy atoms for non-adjacent residues
        atoms = [N, CA, C]
        clash_threshold = 2.0  # Å
        clashes = []
        for i in range(L):
            for j in range(i+3, L):  # skip close sequence neighbors
                min_d = 1e9
                for A in atoms:
                    for B in atoms:
                        d = self._dist(A[i], B[j])
                        if d < min_d:
                            min_d = d
                if min_d < clash_threshold:
                    clashes.append((i, j, min_d))
                    if len(clashes) >= 50:
                        break
            if len(clashes) >= 50:
                break

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
                "num_clashes": int(len(clashes)),
                "min_ca_ca": float(min(ca_ca) if ca_ca else 0.0),
                "max_ca_ca": float(max(ca_ca) if ca_ca else 0.0),
            },
            "clashes": clashes,
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

        # Physical: CA-CA deviation from 3.8 Å
        ca_ca = np.array([np.linalg.norm(CA[i] - CA[i-1]) for i in range(1, L)], dtype=np.float32)
        E_ca = float(np.mean((ca_ca - 3.8) ** 2)) if L > 1 else 0.0

        # Physical: repulsive for non-adjacent atoms closer than 3.2 Å
        atoms = [N, CA, C]
        rep = 0.0
        thr = 3.2
        for i in range(L):
            for j in range(i+3, L):
                # min atom-atom distance between residues i and j
                dmin = min(np.linalg.norm(A[i] - B[j]) for A in atoms for B in atoms)
                if dmin < thr:
                    rep += float((thr - dmin) ** 2)
        E_clash = rep / max(1, L)

        # Harmony: smoothness (finite difference of torsions)
        t = np.asarray(torsions, dtype=np.float32)
        if t.shape[0] > 1:
            dt = t[1:] - t[:-1]
            E_smooth = float(np.mean(dt ** 2))
        else:
            E_smooth = 0.0

        # Harmony: snap distance to nearest bin per mode
        from .scale_and_meter import SCALE_TABLE
        snap_errs = []
        for i in range(L):
            bins = SCALE_TABLE[modes[i]]
            diff = bins - t[i]
            d = np.min(np.linalg.norm(diff, axis=1))
            snap_errs.append(d)
        E_snap = float(np.mean(np.square(snap_errs))) if snap_errs else 0.0

        w = weights
        total = (w['ca'] * E_ca + w['clash'] * E_clash + w['smooth'] * E_smooth + w['snap'] * E_snap)
        return float(total)

    def refine_torsions(self, phi_psi_initial: list, modes: list[str], sequence: str,
                        max_iters: int = 150, step_deg: float = 2.0, seed: int | None = None,
                        weights: dict | None = None, patience: int = 50) -> tuple[list, np.ndarray]:
        """
        Refines a set of torsion angles to minimize dissonance while staying on-key.
        Returns (refined_torsions_list, refined_backbone)
        """
        torsions = np.array(phi_psi_initial, dtype=np.float32)
        L = torsions.shape[0]
        rng = np.random.default_rng(seed)
        if weights is None:
            weights = {'ca': 1.0, 'clash': 1.5, 'smooth': 0.2, 'snap': 0.5}

        # Evaluate initial
        bb_curr = self.build_backbone_from_torsions(torsions, sequence)
        best_score = self.calculate_dissonance(bb_curr, torsions, modes, weights)
        iters_since_improvement = 0

        for i in range(max_iters):
            # Annealed step size (linear decay)
            progress = i / max(1, max_iters)
            current_step = float(step_deg) * (1.0 - progress)

            k = max(1, L // 5)
            idxs = rng.choice(L, size=k, replace=False)
            prop = torsions.copy()
            prop[idxs] += rng.uniform(-current_step, current_step, size=(k, 2)).astype(np.float32)
            # snap to scale
            for idx in idxs:
                phi, psi = prop[idx]
                sphi, spsi = snap_to_scale(modes[idx], float(phi), float(psi))
                prop[idx, 0] = sphi
                prop[idx, 1] = spsi

            bb_prop = self.build_backbone_from_torsions(prop, sequence)
            score_prop = self.calculate_dissonance(bb_prop, prop, modes, weights)
            if score_prop < best_score:
                torsions = prop
                bb_curr = bb_prop
                best_score = score_prop
                iters_since_improvement = 0
            else:
                iters_since_improvement += 1

            if iters_since_improvement >= patience:
                print(f"   - Rehearsal converged after {i+1} iterations.")
                break

        # Final authoritative pass: targeted clash resolution
        # If clashes remain, aggressively fix them by local torsion tweaks
        rep_weights = dict(weights)
        rep_weights['clash'] = max(rep_weights.get('clash', 1.5), 10.0)
        # Evaluate current clashes using QC
        rphi = np.array([t[0] for t in torsions])
        rpsi = np.array([t[1] for t in torsions])
        qc = self.quality_check(bb_curr, rphi, rpsi, modes)
        clashes = qc['clashes']
        attempts = 0
        max_attempts = 1000
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
            step = 3.5
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
                if attempts % 50 == 0 and local_window < 4:
                    local_window += 1
        return torsions.tolist(), bb_curr
