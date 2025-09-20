"""
Simple tests for the fractal_48 system.
"""

import numpy as np
import tempfile
import os

from fractal_48.config import FractalConfig
from fractal_48.crt import crt_phase48, phi_field, keven_kodd_mask
from fractal_48.kernels import mandelbrot, julia, newton3, cphi
from fractal_48.color import hsl_to_srgb, palette48
from fractal_48.perms import verify_permutation_invertibility
from fractal_48.render import render_frame


def test_config_validation():
    """Test configuration validation."""
    print("Testing configuration validation...")
    
    # Valid config
    config = FractalConfig(width=480, height=288)
    assert config.width == 480
    assert config.height == 288
    
    # Invalid width (not 48-aligned)
    try:
        FractalConfig(width=479, height=288)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Invalid height (not 48-aligned)
    try:
        FractalConfig(width=480, height=287)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    print("✓ Configuration validation works")


def test_crt_mapping():
    """Test CRT phase mapping."""
    print("Testing CRT phase mapping...")
    
    # Test specific coordinates
    phi, parity = crt_phase48(0, 0)
    assert phi == 0 and parity == 0
    
    phi, parity = crt_phase48(15, 2)
    assert phi == 15 * 3 + 2 == 47
    
    phi, parity = crt_phase48(1, 1)
    assert parity == 0  # even sum
    
    phi, parity = crt_phase48(1, 0)
    assert parity == 1  # odd sum
    
    # Test field generation
    phi_map = phi_field(96, 96)
    parity_map = keven_kodd_mask(96, 96)
    
    assert phi_map.shape == (96, 96)
    assert parity_map.shape == (96, 96)
    assert np.max(phi_map) == 47
    assert np.min(phi_map) == 0
    
    print("✓ CRT mapping works correctly")


def test_fractal_kernels():
    """Test fractal kernel implementations."""
    print("Testing fractal kernels...")
    
    # Test Mandelbrot
    n, z = mandelbrot(0+0j, 100, 4.0)
    assert n == 100  # Origin should not escape
    
    n, z = mandelbrot(2+0j, 100, 4.0)
    assert n < 100  # Should escape quickly
    
    # Test Julia
    n, z = julia(0+0j, 0.3+0.5j, 100, 4.0)
    assert isinstance(n, int) and isinstance(z, complex)
    
    # Test Newton
    steps, basin = newton3(1+0j, 100, 1e-6)
    assert basin == 0  # Should converge to root 1
    
    # Test cphi
    c = cphi(0, 0.4, 0.0)
    assert abs(c - 0.4) < 1e-10
    
    c = cphi(12, 0.4, 0.0)  # 1/4 turn
    assert abs(c.real) < 1e-10  # Should be on imaginary axis
    
    print("✓ Fractal kernels work correctly")


def test_color_system():
    """Test color palette system."""
    print("Testing color system...")
    
    # Test HSL to sRGB conversion
    r, g, b = hsl_to_srgb(0, 1, 0.5)  # Pure red
    assert r > 0.9 and g < 0.1 and b < 0.1
    
    r, g, b = hsl_to_srgb(120, 1, 0.5)  # Pure green
    assert r < 0.1 and g > 0.9 and b < 0.1
    
    # Test palette function
    config = FractalConfig()
    r, g, b = palette48(10.0, 0, 0, config)
    assert 0 <= r <= 1 and 0 <= g <= 1 and 0 <= b <= 1
    
    print("✓ Color system works correctly")


def test_permutation_invertibility():
    """Test permutation invertibility for all 48 frames."""
    print("Testing permutation invertibility...")
    
    config = FractalConfig(width=480, height=288, animate=True)
    test_img = np.random.rand(288, 480, 3).astype(np.float32)
    
    for frame_idx in range(48):
        assert verify_permutation_invertibility(test_img, frame_idx, config), \
            f"Frame {frame_idx} permutation not invertible"
    
    print("✓ All 48 permutations are invertible")


def test_rendering():
    """Test basic rendering functionality."""
    print("Testing rendering...")
    
    config = FractalConfig(width=480, height=288, max_iters=10)
    
    # Test Mandelbrot render
    config.kernel = "mandelbrot"
    frame = render_frame(config, 0)
    assert frame.shape == (288, 480, 3)
    assert 0 <= np.min(frame) <= np.max(frame) <= 1
    
    # Test Julia render
    config.kernel = "julia"
    frame = render_frame(config, 0)
    assert frame.shape == (288, 480, 3)
    
    # Test Newton render
    config.kernel = "newton"
    frame = render_frame(config, 0)
    assert frame.shape == (288, 480, 3)
    
    print("✓ Rendering works for all kernels")


def test_backends():
    """Test backend functionality and performance."""
    print("Testing backends...")
    
    config = FractalConfig(width=96, height=96, max_iters=10, backend="numpy")
    
    # Test NumPy backend
    frame_numpy = render_frame(config, 0)
    assert frame_numpy.shape == (96, 96, 3)
    assert 0 <= np.min(frame_numpy) <= np.max(frame_numpy) <= 1
    
    # Test Numba backend if available
    try:
        from .kernels_numba import NUMBA_AVAILABLE
        if NUMBA_AVAILABLE:
            config.backend = "numba"
            frame_numba = render_frame(config, 0)
            assert frame_numba.shape == (96, 96, 3)
            assert 0 <= np.min(frame_numba) <= np.max(frame_numba) <= 1
            
            # Check that outputs are very similar (allowing for floating point differences)
            diff = np.abs(frame_numpy - frame_numba)
            max_diff = np.max(diff)
            assert max_diff < 0.01, f"Backends differ too much: max_diff={max_diff}"
            
        # Test invalid backend
        config.backend = "invalid"
        try:
            FractalConfig(width=96, height=96, backend="invalid")
            assert False, "Should have raised ValueError for invalid backend"
        except ValueError:
            pass  # Expected
    except ImportError:
        pass  # Numba not available, skip test
    
    print("✓ Backend selection works correctly")


def test_48_alignment():
    """Test that only 48-aligned dimensions are accepted."""
    print("Testing 48-alignment requirement...")
    
    valid_sizes = [48, 96, 144, 192, 240, 288, 480, 576, 960, 1152, 1920]
    invalid_sizes = [47, 49, 100, 500, 1000, 1919, 1921]
    
    for size in valid_sizes:
        try:
            config = FractalConfig(width=size, height=size)
            # Should not raise
        except ValueError:
            assert False, f"Valid size {size} was rejected"
    
    for size in invalid_sizes:
        try:
            config = FractalConfig(width=size, height=size)
            assert False, f"Invalid size {size} was accepted"
        except ValueError:
            pass  # Expected
    
    print("✓ 48-alignment requirement enforced")


def run_all_tests():
    """Run all tests."""
    print("Running Fractal-48 test suite...")
    print("=" * 50)
    
    test_config_validation()
    test_crt_mapping()
    test_fractal_kernels()
    test_color_system()
    test_permutation_invertibility()
    test_rendering()
    test_backends()
    test_48_alignment()
    
    print("=" * 50)
    print("✓ All tests passed!")


if __name__ == "__main__":
    run_all_tests()