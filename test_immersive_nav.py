#!/usr/bin/env python3
"""
Quick test script for the immersive navigator.
Run this to verify the 3D fractal visualization is working.
"""

import sys
import time
from agi.vision.immersive_navigator import ImmersiveNavigator, FractalNode
import numpy as np


def test_fractal_node():
    """Test FractalNode creation and methods."""
    print("Testing FractalNode...")
    
    node = FractalNode(
        position=np.array([0.0, 0.0, 0.0]),
        generation=0,
        parent=None,
        children=[],
        complexity=0.0,
        departure_angle=0.0,
        is_folded=False,
        color=(1.0, 0.0, 0.0),
        radius=1.0,
        pulse_phase=0.0,
        manifold_level=0
    )
    
    # Test distance calculation
    point = np.array([3.0, 4.0, 0.0])
    distance = node.distance_to(point)
    assert abs(distance - 5.0) < 0.001, f"Distance calculation failed: {distance}"
    
    # Test folding/unfolding
    node.fold()
    assert node.is_folded
    node.unfold()
    assert not node.is_folded
    
    print("✓ FractalNode tests passed")


def test_navigator_initialization():
    """Test ImmersiveNavigator initialization."""
    print("\nTesting ImmersiveNavigator initialization...")
    
    nav = ImmersiveNavigator(
        width=640,
        height=480,
        max_generations=6
    )
    
    # Check initial state
    assert nav.width == 640
    assert nav.height == 480
    assert nav.max_generations == 6
    assert nav.current_generations == 6
    
    # Check fractal tree was generated
    assert nav.root_node is not None
    assert len(nav.all_nodes) > 0
    print(f"  Generated {len(nav.all_nodes)} nodes")
    
    # Check manifold floor was created
    assert 0 in nav.manifold_floors
    assert nav.manifold_floors[0].level == 0
    
    # Test generation toggling
    if nav.toggleable_generations:
        original_count = len(nav.all_nodes)
        nav.current_generations = 3
        nav.all_nodes.clear()
        nav._generate_fractal_tree()
        new_count = len(nav.all_nodes)
        assert new_count < original_count, "Reducing generations should reduce node count"
        print(f"  Generation toggle: {original_count} → {new_count} nodes")
    
    print("✓ ImmersiveNavigator initialization tests passed")


def test_color_conversion():
    """Test HSV to RGB conversion."""
    print("\nTesting HSV to RGB conversion...")
    
    nav = ImmersiveNavigator(width=640, height=480, max_generations=3)
    
    # Test some known conversions
    tests = [
        ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),  # Red
        ((1/3, 1.0, 1.0), (0.0, 1.0, 0.0)),   # Green (approximately)
        ((2/3, 1.0, 1.0), (0.0, 0.0, 1.0)),   # Blue (approximately)
    ]
    
    for (h, s, v), expected in tests:
        result = nav._hsv_to_rgb(h, s, v)
        # Allow some tolerance for floating point
        for i in range(3):
            assert abs(result[i] - expected[i]) < 0.1, f"HSV({h},{s},{v}) conversion failed"
    
    print("✓ HSV to RGB conversion tests passed")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Immersive Fractal Navigator")
    print("=" * 50)
    
    try:
        test_fractal_node()
        test_navigator_initialization()
        test_color_conversion()
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
        
        print("\nThe navigator is ready to run.")
        print("To launch the interactive 3D visualization:")
        print("  python3 agi/vision/immersive_navigator.py")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())