#!/usr/bin/env python3
"""
Simple test of the visualization module
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path

from agi.vision import FractalVisualizer, ComplexityMapper

def main():
    # Create output directory
    Path('outputs').mkdir(exist_ok=True)
    
    print("Testing Fractal Visualization Module")
    print("=" * 40)
    
    # Create sample data
    torch.manual_seed(42)
    tensor_48d = torch.randn(48) * 0.5
    
    # Test 1: FractalVisualizer
    print("\n1. Testing FractalVisualizer...")
    viz = FractalVisualizer(figsize=(12, 8))
    
    # Visualize factorization ladder
    fig = viz.visualize_factorization_ladder(tensor_48d, show_complexity=True)
    fig.savefig('outputs/test_fractal_ladder.png', dpi=100, bbox_inches='tight')
    print("   ✓ Saved factorization ladder visualization")
    plt.close(fig)
    
    # Test 2: ComplexityMapper
    print("\n2. Testing ComplexityMapper...")
    mapper = ComplexityMapper()
    
    # Create complexity heatmap
    tensor_2d = torch.randn(6, 8) * 0.3
    fig = mapper.visualize_complexity_heatmap(tensor_2d, title="Test Complexity Heatmap")
    fig.savefig('outputs/test_complexity_heatmap.png', dpi=100, bbox_inches='tight')
    print("   ✓ Saved complexity heatmap")
    plt.close(fig)
    
    # Test 3: Complexity landscape
    print("\n3. Testing complexity landscape...")
    fig = mapper.create_complexity_landscape(x_range=(-0.5, 0.5), y_range=(-0.5, 0.5), resolution=30)
    fig.savefig('outputs/test_complexity_landscape.png', dpi=100, bbox_inches='tight')
    print("   ✓ Saved complexity landscape")
    plt.close(fig)
    
    print("\n✅ All tests passed!")
    print(f"Visualizations saved to outputs/")

if __name__ == '__main__':
    main()