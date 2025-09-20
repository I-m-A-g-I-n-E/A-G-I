"""
Reversible permutation operations using manifold.py space-to-depth functions.
"""

import numpy as np
from typing import Optional
import sys
import os

# Import space-to-depth operations from the parent manifold system
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fractal48_torch import Fractal48Layer
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def space_to_depth_2(img: np.ndarray) -> np.ndarray:
    """
    2×2 spatial → channel permutation (pure reindexing).
    
    Args:
        img: Array of shape (H, W, C) or (H, W)
        
    Returns:
        Transformed array with 2x2 blocks moved to channel dimension
    """
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    
    H, W, C = img.shape
    assert H % 2 == 0 and W % 2 == 0, f"Dimensions must be even, got {H}×{W}"
    
    # Reshape to expose 2x2 blocks
    img_blocks = img.reshape(H//2, 2, W//2, 2, C)
    # Move spatial dimensions to channel dimension
    img_transformed = img_blocks.transpose(0, 2, 4, 1, 3)
    # Flatten the 2x2 block into channels
    return img_transformed.reshape(H//2, W//2, C*4)


def depth_to_space_2(img: np.ndarray) -> np.ndarray:
    """
    Exact inverse of space_to_depth_2.
    
    Args:
        img: Array of shape (H, W, C) where C is divisible by 4
        
    Returns:
        Transformed array with channels moved back to 2x2 spatial blocks
    """
    H, W, C = img.shape
    assert C % 4 == 0, f"Channels must be divisible by 4, got {C}"
    
    # Reshape to separate the 2x2 block channels
    img_blocks = img.reshape(H, W, C//4, 2, 2)
    # Move channels back to spatial dimensions
    img_spatial = img_blocks.transpose(0, 3, 1, 4, 2)
    # Reshape to final spatial dimensions
    return img_spatial.reshape(H*2, W*2, C//4)


def space_to_depth_3(img: np.ndarray) -> np.ndarray:
    """
    3×3 spatial → channel permutation (pure reindexing).
    
    Args:
        img: Array of shape (H, W, C) or (H, W)
        
    Returns:
        Transformed array with 3x3 blocks moved to channel dimension
    """
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    
    H, W, C = img.shape
    assert H % 3 == 0 and W % 3 == 0, f"Dimensions must be divisible by 3, got {H}×{W}"
    
    # Reshape to expose 3x3 blocks
    img_blocks = img.reshape(H//3, 3, W//3, 3, C)
    # Move spatial dimensions to channel dimension
    img_transformed = img_blocks.transpose(0, 2, 4, 1, 3)
    # Flatten the 3x3 block into channels
    return img_transformed.reshape(H//3, W//3, C*9)


def depth_to_space_3(img: np.ndarray) -> np.ndarray:
    """
    Exact inverse of space_to_depth_3.
    
    Args:
        img: Array of shape (H, W, C) where C is divisible by 9
        
    Returns:
        Transformed array with channels moved back to 3x3 spatial blocks
    """
    H, W, C = img.shape
    assert C % 9 == 0, f"Channels must be divisible by 9, got {C}"
    
    # Reshape to separate the 3x3 block channels
    img_blocks = img.reshape(H, W, C//9, 3, 3)
    # Move channels back to spatial dimensions
    img_spatial = img_blocks.transpose(0, 3, 1, 4, 2)
    # Reshape to final spatial dimensions
    return img_spatial.reshape(H*3, W*3, C//9)


def apply_perm_for_frame(img: np.ndarray, frame_idx: int, config) -> np.ndarray:
    """
    Apply reversible permutation composition for animation frame.
    
    Frame composition: P(f) = (S2)^{f mod 4} ∘ (S3)^{f mod 3}
    
    Args:
        img: Input image array (H, W, 3)
        frame_idx: Frame index [0, 47]
        config: FractalConfig instance
        
    Returns:
        Permuted image for the frame
    """
    if not config.animate:
        return img
    
    result = img.copy()
    
    # Apply S3 permutation (f mod 3) times
    s3_count = frame_idx % 3
    for _ in range(s3_count):
        if result.shape[0] % 3 == 0 and result.shape[1] % 3 == 0:
            result = space_to_depth_3(result)
            # For display, we might want to invert immediately or apply a different strategy
            # For now, let's keep the permutation
    
    # Apply S2 permutation (f mod 4) times
    s2_count = frame_idx % 4
    for _ in range(s2_count):
        if result.shape[0] % 2 == 0 and result.shape[1] % 2 == 0:
            result = space_to_depth_2(result)
            # Similar consideration for display
    
    return result


def invert_perm_for_frame(img: np.ndarray, frame_idx: int, config) -> np.ndarray:
    """
    Invert the permutation applied by apply_perm_for_frame.
    
    Args:
        img: Permuted image array
        frame_idx: Frame index [0, 47]
        config: FractalConfig instance
        
    Returns:
        Original image with permutation inverted
    """
    if not config.animate:
        return img
    
    result = img.copy()
    
    # Invert S2 permutation
    s2_count = frame_idx % 4
    for _ in range(s2_count):
        if result.shape[2] % 4 == 0:
            result = depth_to_space_2(result)
    
    # Invert S3 permutation
    s3_count = frame_idx % 3
    for _ in range(s3_count):
        if result.shape[2] % 9 == 0:
            result = depth_to_space_3(result)
    
    return result


def verify_permutation_invertibility(img: np.ndarray, frame_idx: int, config) -> bool:
    """
    Verify that permutation operations are truly invertible.
    
    Args:
        img: Test image
        frame_idx: Frame index
        config: Configuration
        
    Returns:
        True if round-trip is exact
    """
    try:
        permuted = apply_perm_for_frame(img, frame_idx, config)
        restored = invert_perm_for_frame(permuted, frame_idx, config)
        return np.allclose(img, restored, atol=1e-10)
    except Exception:
        return False