import numpy as np
import torch
from .geometry import INTERVAL_TO_TORSION, IDEAL_GEOMETRY, place_next_atom

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

    def _get_dominant_interval(self, transition_vector: torch.Tensor) -> str:
        """Finds the musical interval that best matches the transition."""
        # If transition is near-zero, treat as Unison-like to avoid NaNs
        if torch.linalg.norm(transition_vector) < 1e-8:
            return "Unison"
        transition_vector = transition_vector / torch.linalg.norm(transition_vector)
        
        # Use cosine similarity to find the best match
        best_match = "Dissonant"
        max_sim = -1.0
        for name, template in self.interval_templates.items():
            sim = torch.dot(transition_vector, template).item()
            if sim > max_sim:
                max_sim = sim
                best_match = name
        return best_match

    def build_backbone(self, composition_vectors: torch.Tensor) -> np.ndarray:
        """
        Generates a simplified C-alpha (CA) trace for the entire protein based on
        harmonic transitions. This is a first-pass toy model for visualization.
        """
        # Accept either a single 48D vector or a sequence of 48D vectors.
        # If we got shape (1, 48), squeeze it to 1D and expand as a toy sequence.
        if composition_vectors.ndim == 2 and composition_vectors.shape[0] == 1:
            composition_vectors = composition_vectors.squeeze(0)

        if composition_vectors.ndim == 1:
            # Expand a single 48D vector into a short sequence by small jitters
            base = composition_vectors
            composition_vectors = torch.stack([
                base + 0.01 * torch.randn_like(base) * i for i in range(48)
            ], dim=0)
        num_residues = composition_vectors.shape[0]
        # Limit to one window for now (48 long)
        num_residues = int(min(num_residues, 48))

        # --- Simplified CA-Trace Generation for First Pass ---
        ca_coords = [np.zeros((3,))]
        # Initialize a roughly forward direction
        prev_dir = np.array([1.0, 0.0, 0.0])
        for i in range(1, num_residues):
            transition_vec = composition_vectors[i] - composition_vectors[i-1]
            interval = self._get_dominant_interval(transition_vec)
            phi, psi = INTERVAL_TO_TORSION.get(interval, INTERVAL_TO_TORSION["Dissonant"])
            
            # Map angles to a 3D direction in a stable way
            # Convert to radians
            phi_r = np.deg2rad(phi)
            psi_r = np.deg2rad(psi)
            # Spherical-like mapping, blend with previous direction for smoothness
            dir_vec = np.array([
                np.cos(phi_r) * np.cos(psi_r),
                np.cos(phi_r) * np.sin(psi_r),
                np.sin(phi_r),
            ])
            dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-8)
            dir_vec = 0.6 * prev_dir + 0.4 * dir_vec
            dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-8)
            next_coord = ca_coords[-1] + dir_vec * 3.8  # Approx CA-CA distance
            ca_coords.append(next_coord)
            prev_dir = dir_vec
            
        return np.array(ca_coords)

    def save_to_pdb(self, coords: np.ndarray, filename: str):
        """Saves a C-alpha trace to a PDB file for visualization."""
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        with open(filename, 'w') as f:
            for i, pos in enumerate(coords):
                f.write(
                    f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
                    f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                    "  1.00  0.00           C  \n"
                )
        print(f"✅ Saved C-alpha trace to {filename}")
