import numpy as np
import torch
try:
    from .geometry import INTERVAL_TO_TORSION, IDEAL_GEOMETRY, place_next_atom
    from .key_estimator import estimate_key_and_modes
    from .scale_and_meter import snap_to_scale, enforce_meter
except ImportError:  # allow running as a script: python bio/conductor.py
    from bio.geometry import INTERVAL_TO_TORSION, IDEAL_GEOMETRY, place_next_atom
    from bio.key_estimator import estimate_key_and_modes
    from bio.scale_and_meter import snap_to_scale, enforce_meter

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
            # Canonical sequential NeRF
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
        # backbone: (L, 3, 3)
        L = backbone.shape[0]
        # Positions shaped (L, A=3, C=3)
        P = backbone.astype(np.float32)
        # Compute all atom-atom distances between residue pairs using broadcasting:
        # A: (L,1,3,1,3), B: (1,L,1,3,3) -> diff: (L,L,3,3,3)
        A = P[:, None, :, None, :]
        B = P[None, :, None, :, :]
        diff = A - B  # (L, L, 3, 3, 3)
        # Euclidean distances per atom pair across last axis -> (L, L, 3, 3)
        dist = np.linalg.norm(diff, axis=-1)
        # Min over atom pairs -> (L, L)
        D = dist.min(axis=(2, 3))
        # Mask near-sequence neighbors (|i-j| <= skip_near)
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
        # Find indices where clash occurs
        ii, jj = np.where(np.triu((Dmin < clash_threshold), k=1))
        # Construct clashes list up to 50 entries
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
                "orthogonality_index": float(orth_frac),
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

        # Physical: repulsive for non-adjacent atoms closer than 3.2 Å (vectorized)
        thr = 3.2
        Dmin = self._min_inter_residue_distances(backbone, skip_near=2)
        # Only distances below threshold contribute
        mask = Dmin < thr
        # Exclude infs (masked near neighbors)
        contrib = (thr - Dmin[mask]) ** 2 if np.any(mask) else np.array([], dtype=np.float32)
        rep = float(contrib.sum())
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
        qc_curr = self.quality_check(bb_curr, np.array([t[0] for t in torsions]), np.array([t[1] for t in torsions]), modes)
        best_num_clashes = qc_curr['summary']['num_clashes']
        best_min_caca = qc_curr['summary']['min_ca_ca']

        iters_since_improvement = 0

        # Advanced refinement config defaults (kept local to avoid signature bloat)
        phaseA_frac = 0.4
        step_deg_clash = None
        clash_weight = None
        steric_only_phaseA = True
        final_attempts = 2000
        final_step = 5.0
        final_window_increment = 25

        # Phase A: clash-focused
        phaseA_iters = max(1, int(max_iters * 0.4))
        wA = dict(weights)
        wA['clash'] = float(clash_weight) if clash_weight is not None else max(wA.get('clash', 1.5), 10.0)
        stepA = float(step_deg_clash) if step_deg_clash is not None else max(float(step_deg), 3.5)
        # Phase-A scoring weights (steric-only if requested)
        weightsA = {'ca': 0.0, 'smooth': 0.0, 'snap': 0.0, 'clash': wA['clash']} if steric_only_phaseA else wA

        for i in range(phaseA_iters):
            progress = i / max(1, phaseA_iters)
            current_step = stepA * (1.0 - 0.5 * progress)  # gentler anneal in clash phase

            k = max(1, L // 3)
            idxs = rng.choice(L, size=k, replace=False)
            prop = torsions.copy()
            prop[idxs] += rng.uniform(-current_step, current_step, size=(k, 2)).astype(np.float32)
            for idx in idxs:
                phi, psi = prop[idx]
                sphi, spsi = snap_to_scale(modes[idx], float(phi), float(psi))
                prop[idx, 0] = sphi
                prop[idx, 1] = spsi

            bb_prop = self.build_backbone_from_torsions(prop, sequence)
            score_curr = self.calculate_dissonance(bb_curr, torsions, modes, weightsA)
            score_prop = self.calculate_dissonance(bb_prop, prop, modes, weightsA)
            qc_prop = self.quality_check(bb_prop, np.array([t[0] for t in prop]), np.array([t[1] for t in prop]), modes)
            reduce_clash = qc_prop['summary']['num_clashes'] < best_num_clashes
            increase_min = qc_prop['summary']['min_ca_ca'] > best_min_caca
            if score_prop < score_curr or reduce_clash or increase_min:
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
                progress = j / max(1, remain)
                current_step = float(step_deg) * (1.0 - progress)

                k = max(1, L // 5)
                idxs = rng.choice(L, size=k, replace=False)
                prop = torsions.copy()
                prop[idxs] += rng.uniform(-current_step, current_step, size=(k, 2)).astype(np.float32)
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
