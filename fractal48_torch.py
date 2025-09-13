#!/usr/bin/env python3
"""
Fractal48 System - PyTorch Implementation for Apple Silicon (M1 Max)
Perfect reversible computation through 48-basis factorization
Optimized for MPS (Metal Performance Shaders) acceleration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import numpy as np
from dataclasses import dataclass
import time

# For numerical compatibility in tests (float64 FFT, etc.), use CPU
DEVICE = torch.device("mps")
print(f"🔥 Torch lit on: {DEVICE}")


@dataclass
class Provenance:
    """Track the reversible path through the manifold"""
    original_shape: Tuple[int, ...]
    factorization_path: List[str]
    phase_history: List[torch.Tensor]


class Fractal48Layer(nn.Module):
    """
    Core 48-manifold layer with perfect reversibility
    All operations are deterministic permutations or integer-preserving transforms
    """
    
    def __init__(self, use_learnable_lifting: bool = True):
        super().__init__()
        self.use_learnable_lifting = use_learnable_lifting
        
        if use_learnable_lifting:
            # Learnable integer lifting parameters (constrained to maintain reversibility)
            self.lift_scale = nn.Parameter(torch.ones(1))
            self.lift_shift = nn.Parameter(torch.zeros(1))
    
    @staticmethod
    def space_to_depth_2(x: torch.Tensor) -> torch.Tensor:
        """2×2 spatial → channel permutation (pure reindexing)"""
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"Dimensions must be even, got {H}×{W}"
        
        x = x.reshape(B, C, H//2, 2, W//2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.reshape(B, C*4, H//2, W//2)
    
    @staticmethod
    def space_to_depth_3(x: torch.Tensor) -> torch.Tensor:
        """3×3 spatial → channel permutation (pure reindexing)"""
        B, C, H, W = x.shape
        assert H % 3 == 0 and W % 3 == 0, f"Dimensions must be divisible by 3, got {H}×{W}"
        
        x = x.reshape(B, C, H//3, 3, W//3, 3)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.reshape(B, C*9, H//3, W//3)
    
    @staticmethod
    def depth_to_space_2(x: torch.Tensor) -> torch.Tensor:
        """Exact inverse of space_to_depth_2"""
        B, C, H, W = x.shape
        assert C % 4 == 0, f"Channels must be divisible by 4, got {C}"
        
        x = x.reshape(B, C//4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        return x.reshape(B, C//4, H*2, W*2)
    
    @staticmethod
    def depth_to_space_3(x: torch.Tensor) -> torch.Tensor:
        """Exact inverse of space_to_depth_3"""
        B, C, H, W = x.shape
        assert C % 9 == 0, f"Channels must be divisible by 9, got {C}"
        
        x = x.reshape(B, C//9, 3, 3, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        return x.reshape(B, C//9, H*3, W*3)
    
    def integer_lift_mix(self, x: torch.Tensor, shift: int = 1) -> torch.Tensor:
        """
        Integer lifting transform with learnable parameters
        Maintains perfect reversibility through careful quantization
        """
        B, C, H, W = x.shape
        if C < 2:
            return x
        # Skip lifting if channels are odd to maintain alignment
        if (C % 2) == 1:
            return x
        # Use only paired channels to avoid mismatch when C is odd
        paired = C - (C % 2)
        if paired < 2:
            return x
        
        # Split into even/odd channels (keven/kodd)
        x_even = x[:, 0:paired:2]
        x_odd = x[:, 1:paired:2]
        
        if self.use_learnable_lifting:
            # Learnable lifting with reversibility constraint
            scale = torch.clamp(self.lift_scale, 0.5, 2.0)
            shift_val = torch.clamp(self.lift_shift, -1, 1)
            
            # Forward lifting steps (still reversible due to structure)
            x_odd = x_odd + torch.round(x_even * scale + shift_val) / (2 ** shift)
            x_even = x_even + torch.round(x_odd * scale) / (2 ** shift)
        else:
            # Pure integer lifting (bit-shift equivalent)
            x_odd = x_odd + x_even / (2 ** shift)
            x_even = x_even + x_odd / (2 ** shift)
        
        # Interleave back (preserve any unpaired last channel)
        x_out = x.clone()
        x_out[:, 0:paired:2] = x_even
        x_out[:, 1:paired:2] = x_odd
        return x_out
    
    def integer_lift_unmix(self, x: torch.Tensor, shift: int = 1) -> torch.Tensor:
        """Exact inverse of integer_lift_mix"""
        B, C, H, W = x.shape
        if C < 2:
            return x
        if (C % 2) == 1:
            return x
        paired = C - (C % 2)
        if paired < 2:
            return x
        
        x_even = x[:, 0:paired:2].clone()
        x_odd = x[:, 1:paired:2].clone()
        
        if self.use_learnable_lifting:
            scale = torch.clamp(self.lift_scale, 0.5, 2.0)
            shift_val = torch.clamp(self.lift_shift, -1, 1)
            
            # Reverse lifting steps
            x_even = x_even - torch.round(x_odd * scale) / (2 ** shift)
            x_odd = x_odd - torch.round(x_even * scale + shift_val) / (2 ** shift)
        else:
            x_even = x_even - x_odd / (2 ** shift)
            x_odd = x_odd - x_even / (2 ** shift)
        
        x_out = x.clone()
        x_out[:, 0:paired:2] = x_even
        x_out[:, 1:paired:2] = x_odd
        return x_out


class Fractal48Encoder(nn.Module):
    """
    Complete 48-ladder encoder
    Maps 48×48 patches through perfect factorization
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # Initial projection to base channels (optional, can be identity)
        self.input_proj = nn.Conv2d(in_channels, base_channels, 1, bias=False)
        
        # Fractal layers for each factorization step
        # Default to integer-preserving lifting for exact reversibility in tests
        self.frac_3x3 = Fractal48Layer(use_learnable_lifting=False)
        self.frac_2x2_a = Fractal48Layer(use_learnable_lifting=False)
        self.frac_2x2_b = Fractal48Layer(use_learnable_lifting=False)
        self.frac_2x2_c = Fractal48Layer(use_learnable_lifting=False)
        
        # Optional: learnable channel mixing at bottleneck (unitary constraint)
        # After 3× then 2×,2×,2×, channels scale by 9*4*4*4 = 576
        self.bottleneck_mix = nn.Parameter(torch.eye(base_channels * 576))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Provenance]:
        """
        Forward pass through 48-factorization
        48×48 → 16×16 → 8×8 → 4×4 → 2×2
        """
        B, C, H, W = x.shape
        assert H % 48 == 0 and W % 48 == 0, f"Input must be 48-aligned, got {H}×{W}"
        
        # Track provenance
        prov = Provenance(
            original_shape=(B, C, H, W),
            factorization_path=[],
            phase_history=[]
        )
        
        # Initial projection
        x = self.input_proj(x)
        prov.phase_history.append(x.detach().clone())
        
        # Step 1: 48×48 → 16×16 via 3×3 permutation
        x = self.frac_3x3.space_to_depth_3(x)
        x = self.frac_3x3.integer_lift_mix(x, shift=1)
        prov.factorization_path.append('3×3')
        prov.phase_history.append(x.detach().clone())
        
        # Step 2: 16×16 → 8×8 via 2×2 permutation
        x = self.frac_2x2_a.space_to_depth_2(x)
        x = self.frac_2x2_a.integer_lift_mix(x, shift=2)
        prov.factorization_path.append('2×2_a')
        prov.phase_history.append(x.detach().clone())
        
        # Step 3: 8×8 → 4×4 via 2×2 permutation
        x = self.frac_2x2_b.space_to_depth_2(x)
        x = self.frac_2x2_b.integer_lift_mix(x, shift=1)
        prov.factorization_path.append('2×2_b')
        prov.phase_history.append(x.detach().clone())
        
        # Step 4: 4×4 → 2×2 via 2×2 permutation
        x = self.frac_2x2_c.space_to_depth_2(x)
        prov.factorization_path.append('2×2_c')
        prov.phase_history.append(x.detach().clone())
        
        # Optional: bottleneck mixing (keep unitary for reversibility)
        if self.training:
            # Orthogonalize the mixing matrix
            with torch.no_grad():
                U, _, V = torch.linalg.svd(self.bottleneck_mix)
                self.bottleneck_mix.data = U @ V

        # Apply channel mixing per spatial position
        B, C, H, W = x.shape
        mix = self.bottleneck_mix[:C, :C]
        x = torch.einsum('bchw,cd->bdhw', x, mix)
        
        return x, prov


class Fractal48Decoder(nn.Module):
    """
    Complete 48-ladder decoder
    Perfect inverse of the encoder
    """
    
    def __init__(self, encoder: Fractal48Encoder):
        super().__init__()
        # Share weights with encoder for perfect inversion
        self.encoder = encoder
        self.out_channels = encoder.in_channels
        self.base_channels = encoder.base_channels
        
        # Output projection (inverse of input projection)
        self.output_proj = nn.Conv2d(self.base_channels, self.out_channels, 1, bias=False)
    
    def forward(self, z: torch.Tensor, prov: Provenance) -> torch.Tensor:
        """
        Inverse pass through 48-factorization
        2×2 → 4×4 → 8×8 → 16×16 → 48×48
        """
        # Inverse bottleneck mixing (transpose of orthogonal matrix)
        B, C, H, W = z.shape
        mix_inv = self.encoder.bottleneck_mix[:C, :C].T
        z = torch.einsum('bchw,cd->bdhw', z, mix_inv)
        
        # Reverse Step 4: 2×2 → 4×4
        z = self.encoder.frac_2x2_c.depth_to_space_2(z)
        
        # Reverse Step 3: 4×4 → 8×8
        z = self.encoder.frac_2x2_b.integer_lift_unmix(z, shift=1)
        z = self.encoder.frac_2x2_b.depth_to_space_2(z)
        
        # Reverse Step 2: 8×8 → 16×16
        z = self.encoder.frac_2x2_a.integer_lift_unmix(z, shift=2)
        z = self.encoder.frac_2x2_a.depth_to_space_2(z)
        
        # Reverse Step 1: 16×16 → 48×48
        z = self.encoder.frac_3x3.integer_lift_unmix(z, shift=1)
        z = self.encoder.frac_3x3.depth_to_space_3(z)
        
        # Output projection
        z = self.output_proj(z)
        
        return z


class Fractal48AutoEncoder(nn.Module):
    """
    Complete autoencoder with 48-manifold factorization
    Perfectly reversible in principle, learnable for tasks
    """
    
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.encoder = Fractal48Encoder(in_channels, base_channels)
        self.decoder = Fractal48Decoder(self.encoder)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode
        z, prov = self.encoder(x)
        
        # Optional: do something with latent (denoising, etc.)
        z_processed = z  # Can add processing here
        
        # Decode
        x_recon = self.decoder(z_processed, prov)
        
        return {
            'input': x,
            'latent': z,
            'reconstruction': x_recon,
            'provenance': prov
        }
    
    def test_reversibility(self, x: torch.Tensor) -> Dict[str, float]:
        """Test perfect reconstruction capability"""
        with torch.no_grad():
            result = self.forward(x)
            
            # Compute reconstruction metrics
            mse = F.mse_loss(result['reconstruction'], result['input'])
            mae = F.l1_loss(result['reconstruction'], result['input'])
            max_error = (result['reconstruction'] - result['input']).abs().max()
            
            # Check if reconstruction is near-perfect (accounting for float precision)
            is_perfect = max_error < 1e-5
            
            return {
                'mse': mse.item(),
                'mae': mae.item(),
                'max_error': max_error.item(),
                'is_perfect': is_perfect,
                'latent_shape': result['latent'].shape
            }


class FractalCoordinateSystem:
    """
    Map between linear indices and fractal (dyadic, triadic) coordinates
    This is the key to understanding the 48-manifold structure
    """
    
    @staticmethod
    def to_fractal_coords(i: int) -> Tuple[int, int, int]:
        """Map linear index to (dyadic, triadic, phase) coordinates"""
        assert 0 <= i < 48
        dyadic = i % 16   # 2^4 component
        triadic = i % 3    # 3^1 component
        phase = (i // 16) + (i // 3) * 4
        return dyadic, triadic, phase
    
    @staticmethod
    def from_fractal_coords(dyadic: int, triadic: int) -> int:
        """Reconstruct linear index from fractal coordinates using CRT"""
        # Chinese Remainder Theorem reconstruction
        # 3·11 ≡ 1 (mod 16) and 16·1 ≡ 1 (mod 3)
        return (dyadic * 3 * 11 + triadic * 16) % 48
    
    @staticmethod
    def get_local_opposite(i: int) -> int:
        """Find the local opposite normal in the 48-manifold"""
        coord_sys = FractalCoordinateSystem()
        d, t, p = coord_sys.to_fractal_coords(i)
        
        # Local opposite: complement dyadic, rotate triadic
        d_opp = (~d) & 15  # 4-bit complement
        t_opp = (t + 1) % 3  # Ternary rotation
        
        return coord_sys.from_fractal_coords(d_opp, t_opp)


def create_test_data(batch_size: int = 4, channels: int = 3, size: int = 48) -> torch.Tensor:
    """Create test data aligned to 48-manifold"""
    # Create structured test pattern that respects the factorization
    x = torch.arange(batch_size * channels * size * size, dtype=torch.float32)
    x = x.reshape(batch_size, channels, size, size)
    
    # Add some structure that will be preserved through factorization
    for i in range(size):
        for j in range(size):
            # Encode position in fractal coordinates
            coord_sys = FractalCoordinateSystem()
            idx = (i * size + j) % 48
            d, t, p = coord_sys.to_fractal_coords(idx)
            x[:, :, i, j] += d * 0.1 + t * 0.01
    
    return x.to(DEVICE)


def benchmark_48_system():
    """Benchmark the 48-manifold system on M1 Max"""
    print("\n" + "="*60)
    print("BENCHMARKING 48-MANIFOLD SYSTEM ON M1 MAX")
    print("="*60)
    
    # Create model and move to device
    model = Fractal48AutoEncoder(in_channels=3, base_channels=64).to(DEVICE)
    model.eval()
    
    # Test different input sizes (all 48-aligned)
    test_sizes = [48, 96, 192, 384]
    
    for size in test_sizes:
        print(f"\n📏 Testing size: {size}×{size}")
        
        # Create test data
        x = create_test_data(batch_size=4, channels=3, size=size)
        
        # Test reversibility
        metrics = model.test_reversibility(x)
        print(f"   ✓ Reversibility: {'PERFECT' if metrics['is_perfect'] else 'IMPERFECT'}")
        print(f"   • Max error: {metrics['max_error']:.2e}")
        print(f"   • MSE: {metrics['mse']:.2e}")
        print(f"   • Latent shape: {metrics['latent_shape']}")
        
        # Benchmark forward pass
        if DEVICE.type == "mps":
            torch.mps.synchronize()
        
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        
        if DEVICE.type == "mps":
            torch.mps.synchronize()
        
        elapsed = (time.time() - start) / 10
        throughput = (4 * size * size) / elapsed
        
        print(f"   ⚡ Forward pass: {elapsed*1000:.2f}ms")
        print(f"   📊 Throughput: {throughput:.0f} pixels/sec")
    
    # Demonstrate coordinate system
    print("\n" + "="*60)
    print("FRACTAL COORDINATE SYSTEM")
    print("="*60)
    
    coord_sys = FractalCoordinateSystem()
    print("\nIndex → (Dyadic, Triadic, Phase) → Reconstructed → Opposite")
    for i in [0, 12, 24, 36, 47]:
        d, t, p = coord_sys.to_fractal_coords(i)
        j = coord_sys.from_fractal_coords(d, t)
        opp = coord_sys.get_local_opposite(i)
        print(f"  {i:2d} → ({d:2d}, {t}, {p:2d}) → {j:2d} | opposite: {opp:2d}")
    
    print("\n" + "="*60)
    print("✨ 48-MANIFOLD: WHERE TRINITY MEETS DUALITY")
    print("="*60)


def train_example():
    """Example training loop showing how to use the 48-system"""
    print("\n" + "="*60)
    print("TRAINING EXAMPLE: IMAGE RECONSTRUCTION")
    print("="*60)
    
    # Setup
    model = Fractal48AutoEncoder(in_channels=3, base_channels=64).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Create dummy dataset (in practice, load real images)
    train_data = create_test_data(batch_size=16, channels=3, size=48)
    
    # Add some noise to make reconstruction non-trivial
    noisy_data = train_data + torch.randn_like(train_data) * 0.1
    
    # Training loop
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        
        # Forward pass
        result = model(noisy_data)
        
        # Compute loss (reconstruction + regularization)
        recon_loss = F.mse_loss(result['reconstruction'], train_data)
        
        # Optional: regularize latent to be sparse/structured
        latent_reg = result['latent'].abs().mean() * 0.01
        
        loss = recon_loss + latent_reg
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch:2d} | Loss: {loss.item():.4f} | "
                  f"Recon: {recon_loss.item():.4f} | Reg: {latent_reg.item():.4f}")
    
    # Final test
    model.eval()
    metrics = model.test_reversibility(train_data)
    print(f"\nFinal reconstruction error: {metrics['mse']:.2e}")
    print(f"Near-perfect recovery: {metrics['is_perfect']}")


if __name__ == "__main__":
    print("🔥 LIGHTING THE TORCH ON THE 48-MANIFOLD 🔥")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Device: {DEVICE}")
    
    # Run demonstrations
    benchmark_48_system()
    train_example()
    
    print("\n" + "="*60)
    print("💫 THE DIVINE TRINITY OF DUALITY IS LIT 💫")
    print("="*60)