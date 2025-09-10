#!/usr/bin/env python3
"""
Test suite for the Fractal48 PyTorch implementation
Validates reversibility, performance, and mathematical properties
"""

import matplotlib


import torch
import torch.nn as nn
import torch.nn.functional as Fw
import matplotlib.pyplot as plt
import sys
import numpy as np
from fractal48_torch import (
    Fractal48AutoEncoder, 
    FractalCoordinateSystem,
    create_test_data,
    DEVICE,
    Fractal48Layer
)
from main import Fractal48Transfer, FractalCoordinate

def set_seed(seed: int = 1234):
    """Deterministic seeding for reproducible tests."""
    torch.manual_seed(seed)
    np.random.seed(seed)

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


def test_polyphase_permutation_roundtrip():
    """Verify pure polyphase (3 then 2×2×2) permutations are exactly reversible.
    This demonstrates 'transfer, not transform' on the native 48×48 grid without interpolation.
    """
    print("\n" + "="*60)
    print("TESTING POLYPHASE PERMUTATION ROUND-TRIP (48 = 3 × 2 × 2 × 2)")
    print("="*60)

    B, C, H, W = 2, 1, 48, 48
    x = torch.randn(B, C, H, W, device=DEVICE)
    frac = Fractal48Layer(use_learnable_lifting=False)

    # Forward polyphase (permutations only)
    y = frac.space_to_depth_3(x)   # 48→16, C→C*9
    y = frac.space_to_depth_2(y)   # 16→8,  C→C*36
    y = frac.space_to_depth_2(y)   # 8→4,   C→C*144
    y = frac.space_to_depth_2(y)   # 4→2,   C→C*576

    # Inverse polyphase
    z = frac.depth_to_space_2(y)   # 2→4
    z = frac.depth_to_space_2(z)   # 4→8
    z = frac.depth_to_space_2(z)   # 8→16
    z = frac.depth_to_space_3(z)   # 16→48

    max_err = (z - x).abs().max().item()
    mse = torch.mean((z - x) ** 2).item()
    print(f"Max error: {max_err:.2e} | MSE: {mse:.2e}")
    print(f"Perfect (within float tolerance): {max_err < 1e-6}")


def test_permutation_identities_parameterized():
    """Stress identity for space_to_depth/depth_to_space across sizes and channels."""
    print("\n" + "="*60)
    print("TESTING PERMUTATION IDENTITIES (PARAMETERIZED)")
    print("="*60)
    set_seed(1234)

    sizes = [48, 96]
    batches = [1, 3]
    channels = [1, 2, 3, 4]
    frac = Fractal48Layer(use_learnable_lifting=False)

    for H in sizes:
        for B in batches:
            for C in channels:
                x = torch.randn(B, C, H, H, device=DEVICE)
                # 3× path only if divisible by 3 (true for 48,96)
                y = frac.space_to_depth_3(x)
                y = frac.depth_to_space_3(y)
                e3 = (y - x).abs().max().item()

                z = frac.space_to_depth_2(x)
                z = frac.depth_to_space_2(z)
                e2 = (z - x).abs().max().item()

                print(f"  B={B} C={C} H={H} | e3={e3:.2e} e2={e2:.2e}")
                assert e3 < 1e-6 and e2 < 1e-6


def test_fft_ortho_unitary_roundtrip():
    """Verify 2D FFT with orthonormal scaling is unitary (round-trip is identity)."""
    print("\n" + "="*60)
    print("TESTING ORTHONORMAL FFT ROUND-TRIP (UNITARY)")
    print("="*60)

    B, C, H, W = 2, 3, 48, 48
    x = torch.randn(B, C, H, W, device=DEVICE)
    # Orthonormal FFT/IFT
    X = torch.fft.fft2(x, norm='ortho')
    z = torch.fft.ifft2(X, norm='ortho').real

    max_err = (z - x).abs().max().item()
    mse = torch.mean((z - x) ** 2).item()
    print(f"Max error: {max_err:.2e} | MSE: {mse:.2e}")
    print(f"Perfect (within float tolerance): {max_err < 1e-6}")


def test_bilinear_aliasing_nonzero():
    """Show that bilinear down/up introduces nonzero error (aliasing/smoothing)."""
    print("\n" + "="*60)
    print("TESTING BILINEAR DOWNSAMPLE/UPSAMPLE ALIASING")
    print("="*60)

    size = 48
    x = torch.randn(1, 1, size, size, device=DEVICE)
    down = F.interpolate(x, size=(size//2, size//2), mode='bilinear')
    up = F.interpolate(down, size=(size, size), mode='bilinear')
    err = (up - x).abs().mean().item()
    max_err = (up - x).abs().max().item()
    print(f"Mean error: {err:.6f} | Max error: {max_err:.6f}")
    assert err > 1e-3, "Bilinear down/up should introduce noticeable error"


def test_integer_lifting_inverse():
    """Verify that integer_lift_mix followed by integer_lift_unmix returns the original.
    Test both learnable and non-learnable modes across different shifts.
    """
    print("\n" + "="*60)
    print("TESTING INTEGER LIFTING INVERSE (MIX → UNMIX)")
    print("="*60)

    B, C, H, W = 2, 8, 16, 16  # C even
    x = torch.randn(B, C, H, W, device=DEVICE)

    for use_learnable in [False, True]:
        layer = Fractal48Layer(use_learnable_lifting=use_learnable)
        for shift in [1, 2, 3]:
            y = layer.integer_lift_mix(x, shift=shift)
            z = layer.integer_lift_unmix(y, shift=shift)
            max_err = (z - x).abs().max().item()
            mse = torch.mean((z - x) ** 2).item()
            mode = "learnable" if use_learnable else "integer"
            print(f"  mode={mode:9s} shift={shift} | Max: {max_err:.2e} | MSE: {mse:.2e}")
            assert max_err < 1e-5, "Lifting inverse should be near-perfect"


def test_pixel_unshuffle_shuffle_roundtrip():
    """Pixel unshuffle/shuffle permutations are exactly reversible (baseline)."""
    print("\n" + "="*60)
    print("TESTING PIXEL_UNSHUFFLE/SHUFFLE ROUND-TRIP")
    print("="*60)
    set_seed(2468)
    B, C, H, W = 2, 3, 48, 48
    x = torch.randn(B, C, H, W, device=DEVICE)

    # Unshuffle by 3 then 2,2,2 (mirrors 3×2×2×2 factorization)
    y = F.pixel_unshuffle(x, downscale_factor=3)     # H→H/3, C→C*9
    y = F.pixel_unshuffle(y, downscale_factor=2)     # /2
    y = F.pixel_unshuffle(y, downscale_factor=2)
    y = F.pixel_unshuffle(y, downscale_factor=2)     # H→H/48

    # Shuffle back (reverse order)
    z = F.pixel_shuffle(y, upscale_factor=2)
    z = F.pixel_shuffle(z, upscale_factor=2)
    z = F.pixel_shuffle(z, upscale_factor=2)
    z = F.pixel_shuffle(z, upscale_factor=3)

    max_err = (z - x).abs().max().item()
    print(f"Max error: {max_err:.2e}")
    assert max_err < 1e-6


def test_dtype_float64_precision():
    """Double precision reduces round-trip error further for FFT identity."""
    print("\n" + "="*60)
    print("TESTING FLOAT64 PRECISION (FFT IDENTITY)")
    print("="*60)
    set_seed(1357)
    x32 = torch.randn(1, 1, 48, 48, device=DEVICE)
    x64 = x32.double()
    X64 = torch.fft.fft2(x64, norm='ortho')
    z64 = torch.fft.ifft2(X64, norm='ortho').real
    max_err64 = (z64 - x64).abs().max().item()
    print(f"Max error (float64): {max_err64:.2e}")
    assert max_err64 < 1e-10


def test_encoder_decoder_perfect_recon_for_sizes():
    """Fractal48AutoEncoder reconstruction quality for 48- and 96-sized inputs."""
    print("\n" + "="*60)
    print("TESTING ENCODER/DECODER PERFECT RECONSTRUCTION (48, 96)")
    print("="*60)
    set_seed(97531)
    sizes = [48, 96]
    for sz in sizes:
        # Use base_channels == in_channels and set 1x1 projections to identity
        in_ch = 3
        base_ch = 3
        model = Fractal48AutoEncoder(in_channels=in_ch, base_channels=base_ch).to(DEVICE)
        model.eval()

        # Force 1x1 convs to identity (perfectly invertible)
        with torch.no_grad():
            # Encoder input projection
            w_in = model.encoder.input_proj.weight
            w_in.zero_()
            for c in range(min(in_ch, base_ch)):
                w_in[c, c, 0, 0] = 1.0
            # Decoder output projection
            w_out = model.decoder.output_proj.weight
            w_out.zero_()
            for c in range(min(in_ch, base_ch)):
                w_out[c, c, 0, 0] = 1.0

        x = torch.randn(2, in_ch, sz, sz, device=DEVICE)
        with torch.no_grad():
            out = model(x)
        err = (out['reconstruction'] - out['input']).abs().max().item()
        print(f"  size={sz} | Max error: {err:.2e}")
        assert err < 1e-5


def test_coordinate_local_opposite_bijection():
    """Local opposite mapping should be a bijection over the 48 indices."""
    print("\n" + "="*60)
    print("TESTING COORDINATE LOCAL OPPOSITE BIJECTION")
    print("="*60)
    cs = FractalCoordinateSystem()
    mapping = {}
    for i in range(48):
        opp = cs.get_local_opposite(i)
        assert 0 <= opp < 48
        mapping[i] = opp
    image = set(mapping.values())
    print(f"Unique images: {len(image)} (expected 48)")
    assert len(image) == 48


def test_main_transfer_reversibility_and_coupling():
    """Validate the algebra in main.Fractal48Transfer: reversibility and coupling outputs."""
    print("\n" + "="*60)
    print("TESTING main.Fractal48Transfer REVERSIBILITY & COUPLING")
    print("="*60)
    system = Fractal48Transfer()
    coord = FractalCoordinate(level=0, branch=17, parity=0, phase=(1, 1, 1))
    ok = system.verify_reversibility(coord, depth=20)
    print(f"verify_reversibility(depth=20): {ok}")
    assert ok is True

    evolved, dual = system.couple_with_local_opposite(coord)
    # Basic sanity checks on ranges
    for c in [evolved, dual]:
        assert 0 <= c.branch < 48
        assert c.parity in (0, 1, 2)

if __name__ == "__main__":
    print("🔬 TESTING THE 48-MANIFOLD SYSTEM 🔬")
    print(f"Device: {DEVICE}")

    quick = any(arg.lower() == "quick" for arg in sys.argv[1:])

    if quick:
        # Fast, focused tests that demonstrate transfer vs transform
        set_seed(1234)
        test_polyphase_permutation_roundtrip()
        test_permutation_identities_parameterized()
        test_fft_ortho_unitary_roundtrip()
        test_fft_energy_conservation()
        test_dtype_float64_precision()
        test_pixel_unshuffle_shuffle_roundtrip()
        test_integer_lifting_inverse()
        test_bilinear_aliasing_nonzero()
        test_coordinate_local_opposite_bijection()
    else:
        # Full suite
        test_perfect_reconstruction()
        test_coordinate_system()
        test_gradient_flow()
        test_polyphase_permutation_roundtrip()
        test_permutation_identities_parameterized()
        test_fft_ortho_unitary_roundtrip()
        test_fft_energy_conservation()
        test_dtype_float64_precision()
        test_pixel_unshuffle_shuffle_roundtrip()
        test_integer_lifting_inverse()
        test_bilinear_aliasing_nonzero()
        benchmark_vs_standard()
        visualize_factorization()

    print("\n" + "="*60)
    print("✨ TESTS COMPLETE ✨")
    print("The trinity of duality shines through the divine decode!")
    print("="*60)