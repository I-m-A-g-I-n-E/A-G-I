#!/usr/bin/env python3
"""
Demo for the 48-generation fractal navigator.
Shows the interactive fractal that expands based on viewer perspective.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Check for optional dependencies
try:
    import pygame
    from pygame.locals import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Note: pygame/PyOpenGL not available. Interactive mode disabled.")
    print("Install with: pip install pygame PyOpenGL PyOpenGL_accelerate")

from agi.vision.fractal_navigator import (
    FractalNavigator, 
    InteractiveFractalNavigator,
    create_matplotlib_animation
)


def demo_static_views():
    """Generate static views of the fractal from different perspectives."""
    print("\n" + "="*60)
    print("STATIC FRACTAL VIEWS")
    print("="*60)
    
    navigator = FractalNavigator(generations=12)  # Reduced for static demo
    
    # Create different viewing angles
    views = [
        ("Origin View", np.array([0, 0, 5]), np.array([0, 0, 0])),
        ("Side View", np.array([10, 0, 0]), np.array([0, np.pi/2, 0])),
        ("Top View", np.array([0, 10, 0]), np.array([np.pi/2, 0, 0])),
        ("Diagonal View", np.array([5, 5, 5]), np.array([np.pi/6, np.pi/4, 0]))
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    
    for idx, (title, pos, orient) in enumerate(views):
        navigator.viewer.position = pos
        navigator.viewer.orientation = orient
        navigator.viewer.velocity = np.random.randn(3) * 0.1  # Small random velocity
        
        ax = axes[idx]
        navigator.render_matplotlib(ax)
        ax.set_title(title)
    
    fig.suptitle('48-Generation Fractal Navigator - Multiple Views', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = Path('outputs')
    output_path.mkdir(exist_ok=True)
    fig.savefig(output_path / 'fractal_navigator_views.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved static views to {output_path / 'fractal_navigator_views.png'}")
    
    return fig


def demo_movement_constraints():
    """Demonstrate the Y-axis movement constraint."""
    print("\n" + "="*60)
    print("MOVEMENT CONSTRAINT DEMONSTRATION")
    print("="*60)
    
    navigator = FractalNavigator(generations=8)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})
    
    # Scenario 1: No orthogonal movement
    ax = axes[0]
    navigator.viewer.position = np.array([5.0, 0.0, 5.0])
    navigator.viewer.has_moved_orthogonally = False
    navigator.render_matplotlib(ax)
    ax.set_title('Before Orthogonal Movement\n(Free to move in all directions)')
    
    # Scenario 2: Just after orthogonal movement
    ax = axes[1]
    navigator.viewer.position = np.array([5.0, 3.0, 5.0])
    navigator.viewer.has_moved_orthogonally = True
    navigator.viewer.initial_y = 3.0
    navigator.render_matplotlib(ax)
    ax.set_title('After Orthogonal Movement\n(Y≥3 constraint active, floor visible)')
    
    # Scenario 3: Attempting to go below floor
    ax = axes[2]
    navigator.viewer.position = np.array([8.0, 3.0, 8.0])  # Constrained to floor
    navigator.viewer.velocity = np.array([0.5, -1.0, 0.5])  # Trying to go down
    navigator.viewer.update(0.1, {'down': True})  # Will be stopped at floor
    navigator.render_matplotlib(ax)
    ax.set_title('Constrained to Floor\n(Cannot descend below Y=3)')
    
    fig.suptitle('Y-Axis Movement Constraint Demonstration', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = Path('outputs')
    fig.savefig(output_path / 'fractal_navigator_constraints.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved constraint demo to {output_path / 'fractal_navigator_constraints.png'}")
    
    return fig


def demo_velocity_sampling():
    """Demonstrate how velocity affects sampling rate."""
    print("\n" + "="*60)
    print("VELOCITY-BASED SAMPLING DEMONSTRATION")
    print("="*60)
    
    navigator = FractalNavigator(generations=16)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    
    # Different velocity scenarios
    velocities = [
        ("Stationary", np.array([0, 0, 0])),
        ("Slow (v=0.5)", np.array([0.3, 0.2, 0.3])),
        ("Medium (v=2)", np.array([1.2, 0.8, 1.2])),
        ("Fast (v=5)", np.array([3, 2, 3])),
        ("Very Fast (v=8)", np.array([5, 3, 5])),
        ("Maximum (v=10)", np.array([6, 4, 6]))
    ]
    
    for idx, (title, vel) in enumerate(velocities):
        navigator.viewer.velocity = vel
        navigator.calculate_sample_rate()
        
        ax = axes[idx]
        navigator.render_matplotlib(ax)
        
        speed = np.linalg.norm(vel)
        ax.set_title(f'{title}\nSample Rate: {navigator.sample_rate:.2f}')
    
    fig.suptitle('Velocity-Based Factorial Sampling Rate', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = Path('outputs')
    fig.savefig(output_path / 'fractal_navigator_sampling.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved sampling demo to {output_path / 'fractal_navigator_sampling.png'}")
    
    return fig


def demo_complexity_distribution():
    """Show how fractal complexity distributes through generations."""
    print("\n" + "="*60)
    print("FRACTAL COMPLEXITY DISTRIBUTION")
    print("="*60)
    
    navigator = FractalNavigator(generations=20)
    
    # Generate fractal and collect complexity data
    navigator.generate_visible_fractals(max_distance=30)
    
    # Organize by generation
    generation_complexities = {}
    for node in navigator.visible_nodes:
        gen = node['generation']
        if gen not in generation_complexities:
            generation_complexities[gen] = []
        generation_complexities[gen].append(node['complexity'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Complexity by generation
    ax = axes[0, 0]
    generations = sorted(generation_complexities.keys())
    mean_complexities = [np.mean(generation_complexities[g]) for g in generations]
    ax.plot(generations, mean_complexities, 'o-', color='purple', linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mean Complexity')
    ax.set_title('Mean Complexity by Generation')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Complexity distribution histogram
    ax = axes[0, 1]
    all_complexities = [c for cs in generation_complexities.values() for c in cs]
    ax.hist(all_complexities, bins=30, color='teal', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Complexity')
    ax.set_ylabel('Frequency')
    ax.set_title('Overall Complexity Distribution')
    ax.axvline(np.mean(all_complexities), color='red', linestyle='--', 
              label=f'Mean: {np.mean(all_complexities):.2f}')
    ax.legend()
    
    # Plot 3: Handedness distribution
    ax = axes[1, 0]
    right_count = sum(1 for n in navigator.visible_nodes if n['handedness'].value == 1)
    left_count = len(navigator.visible_nodes) - right_count
    ax.bar(['Right-handed', 'Left-handed'], [right_count, left_count], 
          color=['blue', 'red'], alpha=0.7)
    ax.set_ylabel('Count')
    ax.set_title('Handedness Distribution')
    
    # Plot 4: Complexity vs depth scatter
    ax = axes[1, 1]
    for gen in generations[:10]:  # First 10 generations for clarity
        complexities = generation_complexities[gen]
        ax.scatter([gen] * len(complexities), complexities, 
                  alpha=0.5, s=20, label=f'Gen {gen}' if gen < 3 else '')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Complexity')
    ax.set_title('Complexity Scatter by Generation')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Fractal Complexity Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = Path('outputs')
    fig.savefig(output_path / 'fractal_complexity_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved complexity analysis to {output_path / 'fractal_complexity_analysis.png'}")
    
    return fig


def demo_animated():
    """Create an animated tour through the fractal."""
    print("\n" + "="*60)
    print("CREATING ANIMATED FRACTAL TOUR")
    print("="*60)
    
    fig, anim = create_matplotlib_animation()
    
    # Save animation
    output_path = Path('outputs')
    output_path.mkdir(exist_ok=True)
    
    try:
        anim.save(output_path / 'fractal_navigator_tour.gif', writer='pillow', fps=20)
        print(f"✓ Saved animation to {output_path / 'fractal_navigator_tour.gif'}")
    except Exception as e:
        print(f"Could not save animation: {e}")
        print("Showing live animation instead...")
        plt.show()
    
    return fig, anim


def demo_interactive():
    """Run the interactive OpenGL navigator."""
    if not OPENGL_AVAILABLE:
        print("\n⚠️  Interactive mode requires pygame and PyOpenGL")
        print("Install with: pip install pygame PyOpenGL PyOpenGL_accelerate")
        return
    
    print("\n" + "="*60)
    print("INTERACTIVE FRACTAL NAVIGATOR")
    print("="*60)
    print("\nControls:")
    print("  WASD: Move horizontally")
    print("  Space/Shift: Move up/down")
    print("  Arrow Keys: Rotate view")
    print("  ESC: Exit")
    print("\nRemember: Once you move up (orthogonally), you cannot")
    print("descend below that Y level (spacetime constraint)")
    print("="*60)
    
    navigator = InteractiveFractalNavigator(generations=48)
    navigator.run()


def main():
    parser = argparse.ArgumentParser(description='48-Generation Fractal Navigator Demo')
    parser.add_argument('--mode', choices=['static', 'constraints', 'sampling', 
                                          'complexity', 'animated', 'interactive', 'all'],
                       default='all', help='Which demo to run')
    parser.add_argument('--no-show', action='store_true', help='Save without displaying')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("48-GENERATION FRACTAL NAVIGATOR")
    print("Adaptive fractal visualization with spacetime constraints")
    print("="*60)
    
    # Create output directory
    Path('outputs').mkdir(exist_ok=True)
    
    figures = []
    
    if args.mode in ['static', 'all']:
        fig = demo_static_views()
        figures.append(fig)
    
    if args.mode in ['constraints', 'all']:
        fig = demo_movement_constraints()
        figures.append(fig)
    
    if args.mode in ['sampling', 'all']:
        fig = demo_velocity_sampling()
        figures.append(fig)
    
    if args.mode in ['complexity', 'all']:
        fig = demo_complexity_distribution()
        figures.append(fig)
    
    if args.mode in ['animated', 'all']:
        fig, anim = demo_animated()
        figures.append(fig)
    
    if args.mode == 'interactive':
        demo_interactive()
    
    if not args.no_show and figures:
        plt.show()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("All visualizations saved to outputs/")
    print("="*60)


if __name__ == '__main__':
    main()