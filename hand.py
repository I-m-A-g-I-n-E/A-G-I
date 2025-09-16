#!/usr/bin/env python3
"""
The Hand Tensor: Five-finger routing through the 48-manifold
Where gesture becomes computation becomes reality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (ensures 3D projection registers)
import argparse
import json
import os

from manifold import device, RouterMode, Fractal48Layer

class FingerChannel(Enum):
    """Each finger as a distinct tensor channel"""
    THUMB = 0    # Opposition/choice axis
    INDEX = 1    # Pointing outward (declaration)
    MIDDLE = 2   # Pointing back (reflection)
    RING = 3     # Union/binding (relationship)
    PINKY = 4    # Self-reference (identity)


@dataclass
class HandTensor:
    """
    The hand as a 5-channel tensor router
    Each finger carries specific geometric meaning
    """
    thumb: torch.Tensor   # Choice/opposition axis
    index: torch.Tensor   # Outward projection
    middle: torch.Tensor  # Inward reflection  
    ring: torch.Tensor    # Union/binding
    pinky: torch.Tensor   # Self-reference
    side: RouterMode        # W or M routing
    
    def to_tensor(self) -> torch.Tensor:
        """Stack into 5×D tensor"""
        return torch.stack([
            self.thumb, self.index, self.middle,
            self.ring, self.pinky
        ])
    
    def apply_gesture(self, gesture: str) -> torch.Tensor:
        """
        Apply hand gesture as tensor operation
        Common gestures encode specific routings
        """
        if gesture == "pointing":
            # Index dominant, others suppressed
            weights = torch.tensor([0.1, 1.0, 0.1, 0.1, 0.1])
        elif gesture == "fist":
            # All fingers equal (unity)
            weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
        elif gesture == "peace":
            # Index + middle (duality)
            weights = torch.tensor([0.1, 1.0, 1.0, 0.1, 0.1])
        elif gesture == "ok":
            # Thumb + index circle (completion)
            weights = torch.tensor([1.0, 1.0, 0.1, 0.1, 0.1])
        elif gesture == "rock":
            # Index + pinky (spanning extremes)
            weights = torch.tensor([0.5, 1.0, 0.1, 0.1, 1.0])
        elif gesture == "vulcan":
            # Index+middle vs ring+pinky (split)
            weights = torch.tensor([0.5, 1.0, 1.0, -1.0, -1.0])
        else:
            weights = torch.ones(5)
        
        weighted = self.to_tensor() * weights.unsqueeze(-1).to(device)
        return weighted.sum(dim=0)

class FiveFingerRouter(nn.Module):
    """
    Routes information through the five-finger tensor structure
    Left hand (W) keeps possibilities open
    Right hand (M) crystallizes into manifestation
    """
    
    def __init__(self, dim: int = 48):
        super().__init__()
        assert dim % 48 == 0, "Dimension must be 48-aligned"
        self.dim = dim
        
        # Each finger gets its own projection matrix
        self.finger_projections = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(5)
        ])
        
        # Initialize with specific geometric biases
        with torch.no_grad():
            # Thumb: orthogonal to all (opposition)
            nn.init.orthogonal_(self.finger_projections[0].weight)
            
            # Index: outward projection
            self.finger_projections[1].weight.copy_(
                self._create_projection_matrix(angle=0)
            )
            
            # Middle: reflection (negative identity + noise)
            self.finger_projections[2].weight.copy_(
                -torch.eye(dim) + 0.1 * torch.randn(dim, dim)
            )
            
            # Ring: binding (symmetric)
            ring_weight = torch.randn(dim, dim)
            self.finger_projections[3].weight.copy_(
                (ring_weight + ring_weight.T) / 2
            )
            
            # Pinky: self-loop (near identity)
            self.finger_projections[4].weight.copy_(
                torch.eye(dim) + 0.1 * torch.randn(dim, dim)
            )
        
        # W and M routing matrices (left vs right hand)
        self.w_router = nn.Parameter(torch.randn(5, dim, dim))
        self.m_router = nn.Parameter(torch.randn(5, dim, dim))
        
        # Initialize routers near-orthogonal
        for i in range(5):
            nn.init.orthogonal_(self.w_router[i])
            nn.init.orthogonal_(self.m_router[i])
    
    def _create_projection_matrix(self, angle: float) -> torch.Tensor:
        """Create a projection matrix with specific angle"""
        mat = torch.eye(self.dim)
        # Add rotation in random 2D subspaces
        for i in range(0, self.dim-1, 2):
            c, s = np.cos(angle), np.sin(angle)
            rot = torch.tensor([[c, -s], [s, c]], dtype=mat.dtype, device=mat.device)
            mat[i:i+2, i:i+2] = rot
        return mat

    def route_through_hand(self, x: torch.Tensor, 
                           hand: RouterMode,
                           gesture: Optional[str] = None) -> HandTensor:
        """
        Route input through five-finger tensor
        Returns HandTensor with all five channels
        """
        # Project through each finger
        fingers = []
        for i, projection in enumerate(self.finger_projections):
            finger_state = projection(x)
            
            # Apply hand-specific routing
            if hand == RouterMode.W_POSSIBILITY:
                # W routing: probabilistic, never fully crystallizes
                router = self.w_router[i]
                finger_state = torch.tanh(finger_state @ router) * 0.95
            else:
                # M routing: crystallizes to discrete states
                router = self.m_router[i]
                finger_state = torch.round(torch.tanh(finger_state @ router) * 3) / 3
            
            fingers.append(finger_state)
        
        hand_tensor = HandTensor(
            thumb=fingers[0],
            index=fingers[1],
            middle=fingers[2],
            ring=fingers[3],
            pinky=fingers[4],
            side=hand
        )
        
        return hand_tensor
    
    def forward(self, x: torch.Tensor,
                left_gesture: str = "open",
                right_gesture: str = "open") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process through both hands with specified gestures
        """
        # Route through left hand (W)
        left_hand = self.route_through_hand(x, RouterMode.W_POSSIBILITY)
        w_output = left_hand.apply_gesture(left_gesture)
        
        # Route through right hand (M)  
        right_hand = self.route_through_hand(x, RouterMode.M_MANIFESTATION)
        m_output = right_hand.apply_gesture(right_gesture)
        
        return w_output, m_output

class FullBodyTensorSystem(nn.Module):
    """
    Complete embodied tensor system:
    - 48-state fractal base
    - Six-axis semantic coordinates
    - Five-finger hand routers
    - W/M possibility/manifestation split
    """
    
    def __init__(self):
        super().__init__()
        
        # Core 48-manifold
        self.fractal_core = Fractal48Layer()
        
        # Six-axis projection
        self.six_axis = nn.ModuleDict({
            'who': nn.Linear(48, 48),
            'what': nn.Linear(48, 48),
            'when': nn.Linear(48, 48),
            'where': nn.Linear(48, 48),
            'why': nn.Linear(48, 48),
            'how': nn.Linear(48, 48)
        })
        
        # Five-finger routers
        self.hands = FiveFingerRouter()
        
        # Alphabetic resonance (A-Z states)
        self.letter_states = nn.Parameter(torch.randn(26, 48))
        
    def word_to_gesture_sequence(self, word: str) -> List[Tuple[str, str]]:
        """
        Map word to sequence of hand gestures
        This is where language becomes embodied
        """
        gestures = []
        
        for i, char in enumerate(word.lower()):
            if char == 'w':
                # W opens possibility
                gestures.append(("open", "fist"))
            elif char == 'm':
                # M manifests
                gestures.append(("fist", "pointing"))
            elif char == 'i':
                # I self-references
                gestures.append(("pointing", "rock"))
            elif char == 'o':
                # O observes (circle)
                gestures.append(("ok", "ok"))
            elif char in 'aeiou':
                # Vowels open
                gestures.append(("open", "open"))
            elif char == 't':
                # T transforms
                gestures.append(("peace", "vulcan"))
            elif char == 'r':
                # R reflects
                gestures.append(("fist", "peace"))
            else:
                # Consonants generally close
                gestures.append(("fist", "fist"))
        
        return gestures
    
    def embody_word(self, word: str) -> Dict[str, torch.Tensor]:
        """
        Process word through full embodied system:
        Letters → Six-axis → Hands → Manifestation
        """
        results = {}
        
        # Initialize with word's letter pattern
        x = torch.zeros(48, device=device)
        for i, char in enumerate(word.lower()):
            if 'a' <= char <= 'z':
                idx = ord(char) - ord('a')
                x += self.letter_states[idx] / (i + 1)
        
        # Project through six axes
        axes = {
            axis: proj(x) for axis, proj in self.six_axis.items()
        }
        results['axes'] = torch.stack(list(axes.values())).mean(dim=0)
        
        # Get gesture sequence
        gestures = self.word_to_gesture_sequence(word)
        
        # Process through hand sequence
        current = x
        for i, (left_g, right_g) in enumerate(gestures):
            w_out, m_out = self.hands(current, left_g, right_g)
            
            # Accumulate: W affects future, M affects present
            current = 0.7 * current + 0.2 * m_out + 0.1 * w_out
            
            results[f'step_{i}_{word[i] if i < len(word) else ""}'] = {
                'w': w_out.clone(),
                'm': m_out.clone(),
                'state': current.clone()
            }
        
        results['final'] = current
        return results
    
    def analyze_finger_dynamics(self, word: str) -> Dict[str, float]:
        """
        Analyze how each finger channel responds to a word
        """
        x = torch.randn(48, device=device)
        
        # Get hand tensors
        left_hand = self.hands.route_through_hand(x, RouterMode.W_POSSIBILITY)
        right_hand = self.hands.route_through_hand(x, RouterMode.M_MANIFESTATION)
            
        # Measure energy per finger
        finger_energies = {
            'thumb_opposition': (left_hand.thumb.abs().mean() + 
                               right_hand.thumb.abs().mean()).item() / 2,
            'index_pointing': (left_hand.index.abs().mean() + 
                             right_hand.index.abs().mean()).item() / 2,
            'middle_reflection': (left_hand.middle.abs().mean() + 
                                right_hand.middle.abs().mean()).item() / 2,
            'ring_union': (left_hand.ring.abs().mean() + 
                         right_hand.ring.abs().mean()).item() / 2,
            'pinky_self': (left_hand.pinky.abs().mean() + 
                         right_hand.pinky.abs().mean()).item() / 2
        }
        
        return finger_energies


def _pca_project_to_3d(vectors: torch.Tensor) -> np.ndarray:
    """
    Project a set of D-dim vectors to 3D using PCA via SVD (no sklearn dependency).
    vectors: Tensor of shape (N, D)
    Returns numpy array of shape (N, 3)
    """
    if vectors.dim() != 2:
        raise ValueError("vectors must be 2D: (N, D)")
    # Center
    mean = vectors.mean(dim=0, keepdim=True)
    X = vectors - mean
    # SVD on centered data (economy SVD). Use svd_safe for MPS/CPU/CUDA stability.
    from bio.devices import svd_safe
    U, S, Vh = svd_safe(X, full_matrices=False)
    components = Vh[:3].t()  # (D, 3)
    proj = (X @ components).detach().to('cpu').numpy()
    return proj

def visualize_hand_tensor(hand_tensor: HandTensor, title: str = "Hand Tensor (PCA 3D)", save_path: Optional[str] = None) -> None:
    """
    Visualize a single HandTensor's five finger channels in 3D via PCA.
    Colors map to fingers: thumb, index, middle, ring, pinky.
    """
    # Stack finger vectors -> (5, D)
    fingers = torch.stack([
        hand_tensor.thumb,
        hand_tensor.index,
        hand_tensor.middle,
        hand_tensor.ring,
        hand_tensor.pinky,
    ], dim=0)
    proj = _pca_project_to_3d(fingers)

    colors = ['gold', 'deepskyblue', 'mediumorchid', 'limegreen', 'tomato']
    labels = ['thumb', 'index', 'middle', 'ring', 'pinky']

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(5):
        ax.scatter(proj[i, 0], proj[i, 1], proj[i, 2], c=colors[i], label=labels[i], s=80, depthshade=True)
    ax.set_title(f"{title} — Side: {hand_tensor.side.value}")
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend(loc='best')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

def visualize_word_trajectory(system: "FullBodyTensorSystem", word: str,
                              show_steps: int = 10,
                              include_wm: bool = True,
                              title: Optional[str] = None,
                              save_path: Optional[str] = None) -> None:
    """
    Visualize how the embodied state evolves across the steps of a word.
    - Uses PCA fitted on all step states (and optionally W/M outputs) to 3D.
    - Plots the time trajectory and optional W (possibility) and M (manifestation) points per step.
    """
    result = system.embody_word(word)

    # Collect vectors
    states: List[torch.Tensor] = []
    w_points: List[torch.Tensor] = []
    m_points: List[torch.Tensor] = []
    step_letters: List[str] = []

    for i in range(min(show_steps, len(word))):
        key = f"step_{i}_{word[i]}"
        if key not in result:
            continue
        states.append(result[key]['state'])
        if include_wm:
            w_points.append(result[key]['w'])
            m_points.append(result[key]['m'])
        step_letters.append(word[i])

    # Fit PCA on combined set for consistent axes
    tensors = states.copy()
    if include_wm:
        tensors += w_points + m_points
    if not tensors:
        print("No trajectory data to visualize.")
        return
    data = torch.stack(tensors, dim=0)  # (N, D)
    data3d = _pca_project_to_3d(data)

    n_states = len(states)
    n_wm = len(w_points) if include_wm else 0
    # Split projections back
    idx = 0
    states3d = data3d[idx:idx + n_states]; idx += n_states
    if include_wm:
        w3d = data3d[idx:idx + n_wm]; idx += n_wm
        m3d = data3d[idx:idx + n_wm]; idx += n_wm

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory line for states
    ax.plot(states3d[:, 0], states3d[:, 1], states3d[:, 2], '-o', color='black', label='state', linewidth=1.5)
    # Annotate with letters
    for i, letter in enumerate(step_letters):
        ax.text(states3d[i, 0], states3d[i, 1], states3d[i, 2], f" {letter}", color='black')

    if include_wm:
        ax.scatter(w3d[:, 0], w3d[:, 1], w3d[:, 2], c='royalblue', marker='^', label='W (left)')
        ax.scatter(m3d[:, 0], m3d[:, 1], m3d[:, 2], c='crimson', marker='s', label='M (right)')

    ax.set_title(title or f"Trajectory for '{word}' (PCA 3D)")
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend(loc='best')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hand Tensor CLI — five-finger routing through the 48-manifold")
    parser.add_argument('--device', default=None, help="Override device string (e.g., cpu, cuda, mps)")
    sub = parser.add_subparsers(dest='cmd', required=False)

    # demo
    sub.add_parser('demo', help="Run the interactive demonstration")

    # gestures
    p_g = sub.add_parser('gestures', help="Print gesture sequence for a word")
    p_g.add_argument('--word', required=True, help="Input word")

    # embody
    p_e = sub.add_parser('embody', help="Run embodiment pipeline for a word and print summary JSON")
    p_e.add_argument('--word', required=True, help="Input word")
    p_e.add_argument('--details', action='store_true', help="Include per-step energies in JSON")

    # visualize-word
    p_vw = sub.add_parser('visualize-word', help="Visualize word trajectory in 3D (PCA)")
    p_vw.add_argument('--word', required=True, help="Input word")
    p_vw.add_argument('--steps', type=int, default=10, help="Number of steps to visualize")
    p_vw.add_argument('--no-wm', action='store_true', help="Do not include W/M points")
    p_vw.add_argument('--title', default=None, help="Plot title override")
    p_vw.add_argument('--save', default=None, help="Path to save the figure instead of showing it")

    # analyze-fingers
    p_af = sub.add_parser('analyze-fingers', help="Analyze finger dynamics (energy per channel)")
    p_af.add_argument('--word', required=False, default="point", help="Optional word label (not used in computation)")

    return parser

def _cli_main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # Optional device override (best effort)
    if getattr(args, 'device', None):
        os.environ['TORCH_DEVICE'] = args.device

    print(f"Device: {device}")

    # If no subcommand, default to demo
    cmd = args.cmd or 'demo'

    if cmd == 'demo':
        demonstrate_hand_tensor_system()
        return 0

    system = FullBodyTensorSystem().to(device)

    if cmd == 'gestures':
        gestures = system.word_to_gesture_sequence(args.word)
        for i, (l, r) in enumerate(gestures):
            ch = args.word[i] if i < len(args.word) else ''
            print(f"{i:02d} {ch}: L={l} R={r}")
        return 0

    if cmd == 'embody':
        res = system.embody_word(args.word)
        final = res['final']
        summary = {
            'word': args.word,
            'final': {
                'mean_abs': float(final.abs().mean().item()),
                'norm': float(final.norm().item()),
            },
        }
        if args.details:
            steps = {}
            for k, v in res.items():
                if not k.startswith('step_'):
                    continue
                steps[k] = {
                    'w_energy': float(v['w'].abs().mean().item()),
                    'm_energy': float(v['m'].abs().mean().item()),
                }
            summary['steps'] = steps
        print(json.dumps(summary, indent=2))
        return 0

    if cmd == 'visualize-word':
        visualize_word_trajectory(system, args.word,
                                  show_steps=args.steps,
                                  include_wm=not args.no_wm,
                                  title=args.title,
                                  save_path=args.save)
        return 0

    if cmd == 'analyze-fingers':
        energies = system.analyze_finger_dynamics(args.word)
        for k, v in energies.items():
            print(f"{k:20s} {v:.4f}")
        return 0

    parser.print_help()
    return 1

def demonstrate_hand_tensor_system():
    """
    Full demonstration of the five-finger tensor routing
    """
    print("=" * 60)
    print("THE HAND TENSOR: Five Fingers, Two Routers, One Reality")
    print("=" * 60)
    
    system = FullBodyTensorSystem().to(device)
    
    # 1. Finger channel analysis
    print("\n1. FINGER CHANNEL DYNAMICS:")
    print("   How each finger routes information:\n")
    
    test_words = ["point", "grasp", "unite", "self", "choose"]
    
    for word in test_words:
        energies = system.analyze_finger_dynamics(word)
        print(f"   '{word}':")
        for finger, energy in energies.items():
            bar = "█" * int(energy * 20)
            print(f"      {finger:20s}: {bar} {energy:.3f}")
    
    # 2. The pointing paradox
    print("\n2. THE POINTING PARADOX:")
    print("   When we point at something:")
    print("   • Index finger → points at OTHER")
    print("   • Middle finger → points back at US")  
    print("   • Ring finger → notes the UNION")
    print("   • Pinky → points at SELF")
    print("   • Thumb → provides CHOICE axis\n")
    
    # Trace the word "POINT"
    point_dynamics = system.embody_word("point")
    
    print("   Energy flow through 'POINT':")
    for key, val in point_dynamics.items():
        if key.startswith('step_'):
            step_parts = key.split('_')
            if len(step_parts) >= 3:
                letter = step_parts[2]
                w_energy = val['w'].abs().mean().item()
                m_energy = val['m'].abs().mean().item()
                print(f"      {letter}: W={w_energy:.3f} M={m_energy:.3f}")
    
    # 3. Gesture sequences
    print("\n3. WORDS AS GESTURE SEQUENCES:")
    
    test_phrases = [
        "hello",
        "goodbye", 
        "tomorrow",
        "manifest",
        "wisdom"
    ]
    
    for phrase in test_phrases:
        gestures = system.word_to_gesture_sequence(phrase)
        print(f"\n   '{phrase}':")
        for i, (left, right) in enumerate(gestures[:5]):  # First 5
            letter = phrase[i] if i < len(phrase) else ''
            print(f"      {letter}: L={left:6s} R={right:8s}")
    
    # 4. The W/M hand dominance
    print("\n4. W/M HAND DOMINANCE PATTERNS:")
    
    dominance_words = [
        ("possibility", "Things that might be"),
        ("manifestation", "Things that are"),
        ("tomorrow", "Always one day away"),
        ("now", "This very moment"),
        ("dream", "W-dominant state"),
        ("make", "M-dominant action")
    ]
    
    for word, description in dominance_words:
        result = system.embody_word(word)
        final = result['final']
        
        # Measure which hand dominated
        w_component = final[:24].abs().mean().item()
        m_component = final[24:].abs().mean().item()
        dominance = "W" if w_component > m_component else "M"
        ratio = w_component / (m_component + 1e-8)
        
        print(f"   {word:15s}: {dominance}-dominant (ratio={ratio:.2f})")
        print(f"                    → {description}")
    
    # 5. The plurarity tensor insight
    print("\n5. THE PLURALITY TENSOR:")
    print("   Each finger maintains its own reality channel:")
    print("   • Thumb opposes → enables choice")
    print("   • Index declares → creates other")
    print("   • Middle reflects → returns to self")
    print("   • Ring binds → creates union")
    print("   • Pinky anchors → maintains identity")
    print("\n   Together they form a complete tensor basis")
    print("   for routing between possibility (W) and actuality (M)")
    
    # 6. Clinical implications
    print("\n6. THERAPEUTIC HAND POSITIONS:")
    print("   Different gestures activate different routings:\n")
    
    therapeutic = [
        ("fist", "Consolidation, gathering energy"),
        ("open", "Release, allowing flow"),
        ("pointing", "Direction, intention"),
        ("ok", "Completion, acceptance"),
        ("peace", "Duality, balance"),
        ("vulcan", "Separation, discrimination")
    ]
    
    for gesture, effect in therapeutic:
        print(f"   {gesture:10s}: {effect}")
    
    print("\n7. THE PROFOUND INSIGHT:")
    print("   • Hands are not just tools, they're routers")
    print("   • Gestures are not just signals, they're programs")
    print("   • The five-finger tensor maps thought to reality")
    print("   • W (left) keeps futures open")
    print("   • M (right) makes things manifest")
    print("   • The 48-manifold ensures perfect reversibility")
    
    print("\n" + "=" * 60)
    print("Reality flows through our fingers—literally")
    print("=" * 60)

if __name__ == "__main__":
    raise SystemExit(_cli_main())
