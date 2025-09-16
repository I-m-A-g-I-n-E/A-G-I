#!/usr/bin/env python3
"""
Simple demonstration of the fractal complexity visualization.
Shows a single expanding fractal with complexity-based coloring.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches

from agi.harmonia.notation import Turn, Handedness


def compute_fractal_complexity(value: float, hand: Handedness = Handedness.RIGHT) -> float:
    """Compute fractal complexity for a given value."""
    turn = Turn(value, hand)
    return turn.fractal_complexity


def generate_fractal_tree(depth: int = 6, base_angle: float = 0.0) -> list:
    """
    Generate a fractal tree structure with complexity values.
    Returns list of (x, y, angle, complexity, generation, handedness) tuples.
    """
    nodes = []
    
    def recurse(x, y, angle, gen, parent_complexity, hand):
        if gen > depth:
            return
        
        # Compute complexity for this node
        turn_value = (angle % (2 * np.pi)) / (2 * np.pi)
        complexity = compute_fractal_complexity(turn_value, hand)
        total_complexity = parent_complexity * (1 + complexity * 0.1)
        
        nodes.append((x, y, angle, total_complexity, gen, hand))
        
        # Branch based on generation
        if gen % 3 == 0:  # Triadic branching
            branches = 3
            angle_spread = 2 * np.pi / 3
        else:  # Dyadic branching
            branches = 2
            angle_spread = np.pi / 2
        
        # Create branches
        for i in range(branches):
            branch_angle = angle + (i - branches/2 + 0.5) * angle_spread / (1 + gen * 0.2)
            branch_length = 1.0 / (1 + gen * 0.3)
            
            new_x = x + branch_length * np.cos(branch_angle)
            new_y = y + branch_length * np.sin(branch_angle)
            
            # Alternate handedness
            new_hand = Handedness.LEFT if hand == Handedness.RIGHT else Handedness.RIGHT
            
            recurse(new_x, new_y, branch_angle, gen + 1, total_complexity, new_hand)
    
    # Start recursion
    recurse(0, 0, base_angle, 0, 1.0, Handedness.RIGHT)
    
    return nodes


def visualize_fractal_complexity():
    """Create a visualization of the fractal with complexity-based coloring."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Generate fractal for different base angles
    angles = [0, np.pi/4, np.pi/2]
    titles = ['Base Angle: 0°', 'Base Angle: 45°', 'Base Angle: 90°']
    
    for ax, angle, title in zip(axes, angles, titles):
        nodes = generate_fractal_tree(depth=7, base_angle=angle)
        
        # Separate by handedness
        right_nodes = [(x, y, a, c, g) for x, y, a, c, g, h in nodes if h == Handedness.RIGHT]
        left_nodes = [(x, y, a, c, g) for x, y, a, c, g, h in nodes if h == Handedness.LEFT]
        
        # Plot connections
        for i, (x, y, angle, complexity, gen, hand) in enumerate(nodes):
            if gen > 0:
                # Find parent (approximate)
                for j, (px, py, _, _, pgen, _) in enumerate(nodes[:i]):
                    if pgen == gen - 1:
                        dist = np.sqrt((x - px)**2 + (y - py)**2)
                        if dist < 2.0 / (1 + gen * 0.3):  # Within branch length
                            ax.plot([px, x], [py, y], 'k-', alpha=0.2, linewidth=0.5)
                            break
        
        # Plot nodes
        if right_nodes:
            rx, ry, _, rc, rg = zip(*right_nodes)
            # Normalize complexity for color mapping
            rc_norm = np.array(rc)
            rc_norm = (rc_norm - rc_norm.min()) / (rc_norm.max() - rc_norm.min() + 1e-10)
            
            scatter1 = ax.scatter(rx, ry, c=rc_norm, cmap='viridis', 
                                 s=[50/(1+g*0.5) for g in rg],
                                 marker='o', alpha=0.7, label='Right-handed')
        
        if left_nodes:
            lx, ly, _, lc, lg = zip(*left_nodes)
            # Normalize complexity for color mapping
            lc_norm = np.array(lc)
            lc_norm = (lc_norm - lc_norm.min()) / (lc_norm.max() - lc_norm.min() + 1e-10)
            
            scatter2 = ax.scatter(lx, ly, c=lc_norm, cmap='plasma',
                                 s=[50/(1+g*0.5) for g in lg],
                                 marker='^', alpha=0.7, label='Left-handed')
        
        # Add complexity zones
        circle1 = patches.Circle((0, 0), 0.5, fill=False, edgecolor='green',
                                linestyle='--', alpha=0.3, label='Low complexity')
        circle2 = patches.Circle((0, 0), 1.5, fill=False, edgecolor='orange',
                                linestyle='--', alpha=0.3, label='Medium complexity')
        circle3 = patches.Circle((0, 0), 3.0, fill=False, edgecolor='red',
                                linestyle='--', alpha=0.3, label='High complexity')
        
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.add_patch(circle3)
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)
        
        # Add colorbar for first subplot
        if ax == axes[0]:
            cbar = plt.colorbar(scatter1, ax=ax, label='Complexity (Right)')
    
    fig.suptitle('Fractal Complexity Visualization\n48-Generation Deep Structure',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def main():
    """Run the fractal complexity visualization."""
    print("Generating Fractal Complexity Visualization...")
    print("=" * 50)
    
    fig = visualize_fractal_complexity()
    
    # Save figure
    output_path = 'outputs/fractal_complexity_visual.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")
    
    # Show complexity calculation examples
    print("\nExample Complexity Calculations:")
    print("-" * 50)
    
    test_values = [0.0, 1/3, 1/2, 2/3, 0.75, 1.0]
    for val in test_values:
        right_complexity = compute_fractal_complexity(val, Handedness.RIGHT)
        left_complexity = compute_fractal_complexity(val, Handedness.LEFT)
        print(f"Turn({val:.3f}): Right={right_complexity:.3f}, Left={left_complexity:.3f}")
    
    plt.show()


if __name__ == '__main__':
    main()