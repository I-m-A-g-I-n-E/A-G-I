#!/usr/bin/env python3
"""
Demo script showcasing the fractal visualization capabilities
of the 48-manifold system.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from agi.vision import FractalVisualizer, ManifoldRenderer, ComplexityMapper
from agi.harmonia.notation import Movement, Gesture, Handedness
from manifold import SixAxisState, RouterMode
from bio.composer import HarmonicPropagator
from bio.conductor import Conductor


def demo_fractal_visualization():
    """Demonstrate fractal structure visualization."""
    print("=" * 60)
    print("FRACTAL STRUCTURE VISUALIZATION")
    print("=" * 60)
    
    # Create a sample 48D tensor
    torch.manual_seed(42)
    initial_state = torch.randn(48) * 0.5
    
    # Initialize visualizer
    viz = FractalVisualizer(figsize=(15, 10))
    
    # Visualize factorization ladder
    print("\n1. Generating factorization ladder visualization...")
    fig1 = viz.visualize_factorization_ladder(initial_state, show_complexity=True)
    fig1.savefig('outputs/fractal_ladder.png', dpi=150, bbox_inches='tight')
    print("   Saved to outputs/fractal_ladder.png")
    
    # Create a composition and visualize its unfolding
    print("\n2. Creating harmonic composition...")
    seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    composer = HarmonicPropagator(n_layers=4, variability=0.3, seed=42)
    composition = composer(seq)
    
    print("3. Visualizing fractal unfolding...")
    fig2 = viz.visualize_fractal_unfolding(composition[0])
    fig2.savefig('outputs/fractal_unfolding.png', dpi=150, bbox_inches='tight')
    print("   Saved to outputs/fractal_unfolding.png")
    
    # Animate evolution if we have a trajectory
    print("\n4. Creating evolution animation...")
    trajectory = [composition[i] for i in range(min(20, len(composition)))]
    anim = viz.animate_fractal_evolution(trajectory, save_path='outputs/fractal_evolution.gif')
    print("   Saved animation to outputs/fractal_evolution.gif")
    
    plt.show()


def demo_manifold_rendering():
    """Demonstrate 3D manifold rendering."""
    print("\n" + "=" * 60)
    print("3D MANIFOLD RENDERING")
    print("=" * 60)
    
    renderer = ManifoldRenderer(use_plotly=False)
    
    # Create a six-axis state
    print("\n1. Creating six-axis semantic state...")
    torch.manual_seed(123)
    state = SixAxisState(
        who=torch.randn(48) * 0.3,
        what=torch.randn(48) * 0.5,
        when=torch.randn(48) * 0.4,
        where=torch.randn(48) * 0.6,
        why=torch.randn(48) * 0.2,
        how=torch.randn(48) * 0.7
    )
    
    fig1 = renderer.render_six_axis_state(state, RouterMode.W_POSSIBILITY)
    fig1.savefig('outputs/six_axis_state.png', dpi=150, bbox_inches='tight')
    print("   Saved to outputs/six_axis_state.png")
    
    # Render manifold lattice
    print("\n2. Rendering manifold lattice structure...")
    fig2 = renderer.render_manifold_lattice(resolution=8)
    fig2.savefig('outputs/manifold_lattice.png', dpi=150, bbox_inches='tight')
    print("   Saved to outputs/manifold_lattice.png")
    
    # Render factorization flow
    print("\n3. Rendering factorization flow...")
    sample_tensor = torch.randn(48, 48) * 0.1
    fig3 = renderer.render_factorization_flow(sample_tensor, show_arrows=True)
    fig3.savefig('outputs/factorization_flow.png', dpi=150, bbox_inches='tight')
    print("   Saved to outputs/factorization_flow.png")
    
    # Render phase torus
    print("\n4. Rendering phase torus...")
    phases = torch.randn(100, 2)  # 100 phase points
    fig4 = renderer.render_phase_torus(phases, color_by='complexity')
    fig4.savefig('outputs/phase_torus.png', dpi=150, bbox_inches='tight')
    print("   Saved to outputs/phase_torus.png")
    
    plt.show()


def demo_complexity_mapping():
    """Demonstrate complexity analysis and visualization."""
    print("\n" + "=" * 60)
    print("COMPLEXITY MAPPING & ANALYSIS")
    print("=" * 60)
    
    mapper = ComplexityMapper()
    
    # Analyze tensor complexity
    print("\n1. Computing tensor complexity heatmap...")
    torch.manual_seed(456)
    tensor = torch.randn(8, 6) * 0.5
    fig1 = mapper.visualize_complexity_heatmap(tensor, title="8x6 Tensor Complexity")
    fig1.savefig('outputs/complexity_heatmap.png', dpi=150, bbox_inches='tight')
    print("   Saved to outputs/complexity_heatmap.png")
    
    # Compare complexity distributions
    print("\n2. Comparing complexity distributions...")
    tensors = [
        torch.randn(48) * 0.1,  # Low variance
        torch.randn(48) * 0.5,  # Medium variance
        torch.randn(48) * 1.0,  # High variance
        torch.randn(48) * 2.0   # Very high variance
    ]
    labels = ['Low Var', 'Medium Var', 'High Var', 'Very High Var']
    
    fig2 = mapper.visualize_complexity_distribution(tensors, labels)
    fig2.savefig('outputs/complexity_distributions.png', dpi=150, bbox_inches='tight')
    print("   Saved to outputs/complexity_distributions.png")
    
    # Analyze trajectory complexity
    print("\n3. Analyzing complexity trajectory...")
    trajectory = [torch.randn(48) * (0.1 + i * 0.05) for i in range(50)]
    fig3 = mapper.visualize_complexity_trajectory(trajectory, window_size=5)
    fig3.savefig('outputs/complexity_trajectory.png', dpi=150, bbox_inches='tight')
    print("   Saved to outputs/complexity_trajectory.png")
    
    # Create complexity landscape
    print("\n4. Creating complexity landscape...")
    fig4 = mapper.create_complexity_landscape(x_range=(-1, 1), y_range=(-1, 1), resolution=40)
    fig4.savefig('outputs/complexity_landscape.png', dpi=150, bbox_inches='tight')
    print("   Saved to outputs/complexity_landscape.png")
    
    # Analyze gesture complexities
    print("\n5. Analyzing gesture complexities...")
    fig5 = mapper.analyze_gesture_complexities()
    fig5.savefig('outputs/gesture_complexities.png', dpi=150, bbox_inches='tight')
    print("   Saved to outputs/gesture_complexities.png")
    
    plt.show()


def demo_protein_visualization():
    """Demonstrate visualization of protein structure complexity."""
    print("\n" + "=" * 60)
    print("PROTEIN STRUCTURE COMPLEXITY VISUALIZATION")
    print("=" * 60)
    
    # Use a real protein sequence
    seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    
    print(f"\n1. Composing sequence (length {len(seq)})...")
    composer = HarmonicPropagator(n_layers=4, variability=0.3, seed=789)
    composition = composer(seq)
    
    print("2. Building backbone structure...")
    conductor = Conductor()
    backbone, phi, psi, modes = conductor.build_backbone(composition, sequence=seq)
    
    # Visualize composition complexity
    print("3. Analyzing composition complexity...")
    viz = FractalVisualizer()
    mapper = ComplexityMapper()
    
    # Create combined visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Composition heatmap
    ax = axes[0, 0]
    comp_2d = composition.view(-1, 48)[:6, :].T  # Take first 6 windows
    im = ax.imshow(comp_2d.cpu().numpy(), cmap='viridis', aspect='auto')
    ax.set_title('Composition Windows')
    ax.set_xlabel('Window')
    ax.set_ylabel('Dimension')
    plt.colorbar(im, ax=ax)
    
    # Plot 2: Complexity per residue
    ax = axes[0, 1]
    complexities = mapper.compute_tensor_complexity(composition)
    mean_complex_per_window = complexities.mean(dim=1).cpu().numpy()
    ax.plot(mean_complex_per_window, 'o-', color='purple', alpha=0.7)
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Mean Complexity')
    ax.set_title('Complexity Along Sequence')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Torsion angles colored by complexity
    ax = axes[0, 2]
    phi_np = phi.cpu().numpy()
    psi_np = psi.cpu().numpy()
    
    # Compute complexity for each torsion pair
    from agi.harmonia.notation import Turn
    torsion_complexities = []
    for p1, p2 in zip(phi_np, psi_np):
        t1 = Turn(p1 / 360.0)
        t2 = Turn(p2 / 360.0)
        torsion_complexities.append(t1.fractal_complexity + t2.fractal_complexity)
    
    scatter = ax.scatter(phi_np, psi_np, c=torsion_complexities, 
                        cmap='RdYlBu_r', s=20, alpha=0.6)
    ax.set_xlabel('Phi (degrees)')
    ax.set_ylabel('Psi (degrees)')
    ax.set_title('Ramachandran Plot (colored by complexity)')
    plt.colorbar(scatter, ax=ax)
    
    # Plot 4: Mode distribution
    ax = axes[1, 0]
    mode_counts = {}
    for mode in modes:
        mode_str = str(mode)
        mode_counts[mode_str] = mode_counts.get(mode_str, 0) + 1
    
    ax.bar(range(len(mode_counts)), list(mode_counts.values()))
    ax.set_xticks(range(len(mode_counts)))
    ax.set_xticklabels(list(mode_counts.keys()), rotation=45, ha='right')
    ax.set_xlabel('Mode')
    ax.set_ylabel('Count')
    ax.set_title('Secondary Structure Distribution')
    
    # Plot 5: Complexity histogram
    ax = axes[1, 1]
    all_complexities = complexities.flatten().cpu().numpy()
    ax.hist(all_complexities, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(all_complexities.mean(), color='red', linestyle='--',
              label=f'Mean: {all_complexities.mean():.2f}')
    ax.set_xlabel('Fractal Complexity')
    ax.set_ylabel('Frequency')
    ax.set_title('Overall Complexity Distribution')
    ax.legend()
    
    # Plot 6: CA-CA distances vs complexity
    ax = axes[1, 2]
    ca_coords = backbone[:, 1, :]  # CA atoms
    ca_distances = []
    for i in range(len(ca_coords) - 1):
        dist = torch.norm(ca_coords[i+1] - ca_coords[i]).item()
        ca_distances.append(dist)
    
    # Plot with complexity coloring
    if len(ca_distances) == len(torsion_complexities) - 1:
        scatter = ax.scatter(range(len(ca_distances)), ca_distances, 
                           c=torsion_complexities[:-1], cmap='RdYlBu_r', 
                           s=20, alpha=0.6)
        plt.colorbar(scatter, ax=ax, label='Complexity')
    else:
        ax.plot(ca_distances, 'o-', alpha=0.6)
    
    ax.axhline(y=3.8, color='green', linestyle='--', alpha=0.5, label='Ideal CA-CA')
    ax.axhline(y=3.2, color='red', linestyle='--', alpha=0.5, label='Min allowed')
    ax.set_xlabel('Residue Pair')
    ax.set_ylabel('CA-CA Distance (Å)')
    ax.set_title('CA-CA Distances')
    ax.legend()
    
    fig.suptitle(f'Protein Structure Complexity Analysis\nSequence: {seq[:30]}...', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    fig.savefig('outputs/protein_complexity_analysis.png', dpi=150, bbox_inches='tight')
    print("   Saved comprehensive analysis to outputs/protein_complexity_analysis.png")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Fractal Visualization Demo')
    parser.add_argument('--demo', choices=['fractal', 'manifold', 'complexity', 'protein', 'all'],
                       default='all', help='Which demo to run')
    parser.add_argument('--no-show', action='store_true', help='Save figures without showing')
    
    args = parser.parse_args()
    
    # Create output directory
    Path('outputs').mkdir(exist_ok=True)
    
    print("\n" + "=" * 60)
    print("48-MANIFOLD FRACTAL VISUALIZATION DEMO")
    print("=" * 60)
    
    if args.demo in ['fractal', 'all']:
        demo_fractal_visualization()
    
    if args.demo in ['manifold', 'all']:
        demo_manifold_rendering()
    
    if args.demo in ['complexity', 'all']:
        demo_complexity_mapping()
    
    if args.demo in ['protein', 'all']:
        demo_protein_visualization()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("All visualizations saved to outputs/")
    print("=" * 60)
    
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()