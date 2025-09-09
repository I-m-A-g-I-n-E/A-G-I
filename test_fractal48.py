#!/usr/bin/env python3
"""
Test suite for the Fractal48 PyTorch implementation
Validates reversibility, performance, and mathematical properties
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from fractal48_torch import (
    Fractal48AutoEncoder, 
    FractalCoordinateSystem,
    create_test_data,
    DEVICE
)

def visualize_factorization():
    """Visualize the 48-manifold factorization process"""
    print("\n" + "="*60)
    print("VISUALIZING 48-MANIFOLD FACTORIZATION")
    print("="*60)
    
    # Create model
    model = Fractal48AutoEncoder(in_channels=1, base_channels=8).to(DEVICE)
    model.eval()
    
    # Create a test pattern that shows the structure
    size = 48
    x = torch.zeros(1, 1, size, size).to(DEVICE)
    
    # Create a pattern that reveals the factorization
    for i in range(size):
        for j in range(size):
            # Checkerboard at different scales
            x[0, 0, i, j] = (i // 3) % 2 + (j // 3) % 2
            x[0, 0, i, j] += 0.5 * ((i // 2) % 2 + (j // 2) % 2)
    
    # Get the factorization stages
    with torch.no_grad():
        z, prov = model.encoder(x)
        phases = prov.phase_history
    
    # Plot the progression
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    stages = [
        (x[0, 0], "Original 48×48"),
        (phases[1][0, 0], "After 3×3 (16×16)"),
        (phases[2][0, 0], "After 2×2 (8×8)"),
        (phases[3][0, 0], "After 2×2 (4×4)"),
        (phases[4][0, 0], "After 2×2 (2×2)"),
        (None, "Factorization Path")
    ]
    
    for ax, (data, title) in zip(axes.flat, stages):
        if data is not None:
            im = ax.imshow(data.cpu().numpy(), cmap='twilight')
            ax.set_title(title)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            # Show the factorization tree
            ax.text(0.5, 0.8, "48×48", ha='center', fontsize=12, weight='bold')
            ax.arrow(0.5, 0.75, 0, -0.1, head_width=0.02, fc='black')
            ax.text(0.5, 0.6, "÷3 → 16×16", ha='center', fontsize=10)
            ax.arrow(0.5, 0.55, 0, -0.1, head_width=0.02, fc='black')
            ax.text(0.5, 0.4, "÷2 → 8×8", ha='center', fontsize=10)
            ax.arrow(0.5, 0.35, 0, -0.1, head_width=0.02, fc='black')
            ax.text(0.5, 0.2, "÷2 → 4×4", ha='center', fontsize=10)
            ax.arrow(0.5, 0.15, 0, -0.1, head_width=0.02, fc='black')
            ax.text(0.5, 0.05, "÷2 → 2×2", ha='center', fontsize=10, weight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title("Factorization Path")
    
    plt.suptitle("48-Manifold Factorization Process", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('fractal48_factorization.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to fractal48_factorization.png")


def test_perfect_reconstruction():
    """Test that the system achieves perfect reconstruction"""
    print("\n" + "="*60)
    print("TESTING PERFECT RECONSTRUCTION")
    print("="*60)
    
    model = Fractal48AutoEncoder(in_channels=3, base_channels=32).to(DEVICE)
    model.eval()
    
    # Test with different data types
    test_cases = [
        ("Random uniform", torch.rand(2, 3, 48, 48).to(DEVICE)),
        ("Random normal", torch.randn(2, 3, 48, 48).to(DEVICE)),
        ("Structured", create_test_data(2, 3, 48)),
        ("Binary", (torch.rand(2, 3, 48, 48) > 0.5).float().to(DEVICE)),
    ]
    
    for name, data in test_cases:
        with torch.no_grad():
            result = model(data)
            error = (result['reconstruction'] - data).abs().max()
            
        print(f"{name:15s} | Max error: {error:.2e} | Perfect: {error < 1e-5}")


def test_coordinate_system():
    """Validate the fractal coordinate system properties"""
    print("\n" + "="*60)
    print("VALIDATING FRACTAL COORDINATE SYSTEM")
    print("="*60)
    
    coord_sys = FractalCoordinateSystem()
    
    # Test bijectivity
    print("\n1. Testing bijectivity (one-to-one mapping):")
    all_coords = set()
    for i in range(48):
        d, t, p = coord_sys.to_fractal_coords(i)
        j = coord_sys.from_fractal_coords(d, t)
        all_coords.add((d, t))
        if i != j:
            print(f"   ERROR: {i} → ({d},{t}) → {j}")
    print(f"   ✓ All 48 indices map to unique coordinates: {len(all_coords) == 48}")
    
    # Test local opposites
    print("\n2. Testing local opposite symmetry:")
    for i in [0, 1, 23, 24, 47]:
        opp = coord_sys.get_local_opposite(i)
        opp_opp = coord_sys.get_local_opposite(opp)
        # Note: double opposite might not return to original due to the specific mapping
        print(f"   {i:2d} → {opp:2d} → {opp_opp:2d}")
    
    # Visualize the structure
    print("\n3. Visualizing dyadic-triadic decomposition:")
    grid = np.zeros((16, 3))
    for i in range(48):
        d, t, p = coord_sys.to_fractal_coords(i)
        grid[d, t] = i
    
    print("   Triadic →")
    print("   " + "  0    1    2")
    print("   " + "-" * 15)
    for d in range(16):
        row = f"{d:2d} |"
        for t in range(3):
            val = int(grid[d, t])
            row += f" {val:2d}  "
        print(f"   {row}")
    print("   ↓ Dyadic")


def benchmark_vs_standard():
    """Compare 48-system against standard convolutions"""
    print("\n" + "="*60)
    print("BENCHMARKING VS STANDARD CONVOLUTIONS")
    print("="*60)
    
    # 48-manifold system
    model_48 = Fractal48AutoEncoder(in_channels=3, base_channels=64).to(DEVICE)
    
    # Standard conv system (similar parameter count)
    class StandardAutoEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=3, padding=0),  # 48→16
                nn.ReLU(),
                nn.Conv2d(64, 256, 3, stride=2, padding=1),  # 16→8
                nn.ReLU(),
                nn.Conv2d(256, 512, 3, stride=2, padding=1),  # 8→4
                nn.ReLU(),
                nn.Conv2d(512, 1024, 3, stride=2, padding=1),  # 4→2
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, 3, stride=3, padding=0),
            )
        
        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)
    
    model_std = StandardAutoEncoder().to(DEVICE)
    
    # Compare
    x = create_test_data(8, 3, 48)
    
    # Parameter count
    params_48 = sum(p.numel() for p in model_48.parameters())
    params_std = sum(p.numel() for p in model_std.parameters())
    
    print(f"\nParameter count:")
    print(f"  48-manifold: {params_48:,}")
    print(f"  Standard:    {params_std:,}")
    
    # Speed test
    import time
    
    for name, model in [("48-manifold", model_48), ("Standard", model_std)]:
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                if name == "48-manifold":
                    _ = model(x)['reconstruction']
                else:
                    _ = model(x)
        
        # Benchmark
        if DEVICE.type == "mps":
            torch.mps.synchronize()
        
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                if name == "48-manifold":
                    out = model(x)['reconstruction']
                else:
                    out = model(x)
        
        if DEVICE.type == "mps":
            torch.mps.synchronize()
        
        elapsed = time.time() - start
        
        print(f"\n{name}:")
        print(f"  Time/batch: {elapsed/100*1000:.2f}ms")
        print(f"  Output shape: {out.shape}")


def test_gradient_flow():
    """Test gradient flow through the reversible system"""
    print("\n" + "="*60)
    print("TESTING GRADIENT FLOW")
    print("="*60)
    
    model = Fractal48AutoEncoder(in_channels=3, base_channels=32).to(DEVICE)
    model.train()
    
    x = create_test_data(4, 3, 48)
    x.requires_grad = True
    
    # Forward and backward
    result = model(x)
    loss = result['reconstruction'].mean()
    loss.backward()
    
    # Check gradient statistics
    print("\nGradient statistics:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            print(f"  {name:30s} | shape: {str(grad.shape):15s} | "
                  f"mean: {grad.mean():.2e} | std: {grad.std():.2e} | "
                  f"max: {grad.abs().max():.2e}")
    
    # Check input gradient (should be well-behaved due to reversibility)
    print(f"\nInput gradient: mean={x.grad.mean():.2e}, std={x.grad.std():.2e}, "
          f"max={x.grad.abs().max():.2e}")


if __name__ == "__main__":
    print("🔬 TESTING THE 48-MANIFOLD SYSTEM 🔬")
    print(f"Device: {DEVICE}")
    
    # Run all tests
    test_perfect_reconstruction()
    test_coordinate_system()
    test_gradient_flow()
    benchmark_vs_standard()
    visualize_factorization()
    
    print("\n" + "="*60)
    print("✨ ALL TESTS COMPLETE ✨")
    print("The trinity of duality shines through the divine decode!")
    print("="*60)