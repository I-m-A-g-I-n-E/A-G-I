"""
Provenance tracking and JSON export for reproducibility.
"""

import json
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
import os
import subprocess

from .config import FractalConfig


def get_git_info() -> Dict[str, str]:
    """Get current git commit and status information."""
    try:
        # Get current commit hash
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], 
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        # Get branch name
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        # Check if working directory is clean
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        return {
            'commit': commit,
            'branch': branch,
            'clean': len(status) == 0,
            'status': status if status else None
        }
    except Exception:
        return {
            'commit': 'unknown',
            'branch': 'unknown',
            'clean': False,
            'status': 'git info unavailable'
        }


def create_provenance(config: FractalConfig, render_time: float = 0.0, 
                     frames: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
    """
    Create comprehensive provenance record for reproducibility.
    
    Args:
        config: Fractal configuration used
        render_time: Total rendering time in seconds
        frames: Optional list of rendered frames for checksum
        
    Returns:
        Provenance dictionary
    """
    git_info = get_git_info()
    
    # Compute checksums if frames provided
    frame_checksums = []
    if frames:
        for i, frame in enumerate(frames):
            frame_bytes = frame.astype(np.float32).tobytes()
            checksum = hashlib.sha256(frame_bytes).hexdigest()[:16]
            frame_checksums.append({
                'frame': i,
                'checksum': checksum,
                'shape': frame.shape,
                'dtype': str(frame.dtype)
            })
    
    provenance = {
        # Metadata
        'version': '0.1.0',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'render_time_seconds': render_time,
        
        # Git information
        'git': git_info,
        
        # Configuration
        'config': {
            'width': config.width,
            'height': config.height,
            'kernel': config.kernel,
            'center': config.center,
            'scale': config.scale,
            'rotation': config.rotation,
            'max_iters': config.max_iters,
            'bailout': config.bailout,
            'julia_r': config.julia_r,
            'julia_theta': config.julia_theta,
            'palette_mode': config.palette_mode,
            'base_hue': config.base_hue,
            'delta_s': config.delta_s,
            'delta_l': config.delta_l,
            'animate': config.animate,
            'loop_frames': config.loop_frames,
            'output_path': config.output_path
        },
        
        # 48-manifold specific parameters
        'manifold': {
            'dimension': 48,
            'factorization': '48 = 2^4 × 3 = 16 × 3',
            'crt_mapping': 'phi = (x mod 16) * 3 + (y mod 3)',
            'parity_mapping': 'p = (x + y) mod 2',
            'phase_scheduler': '48-phase CRT',
            'permutation_schedule': 'P(f) = (S2)^{f mod 4} ∘ (S3)^{f mod 3}'
        },
        
        # Operation details
        'operations': {
            'space_to_depth_2': 'applied for S2 permutations',
            'space_to_depth_3': 'applied for S3 permutations',
            'measurement_first': True,
            'reversible_permutations': config.animate,
            'color_space': 'HSL → sRGB with gamma 2.2',
            'smooth_coloring': config.palette_mode == 'smooth'
        },
        
        # Frame checksums for verification
        'frames': frame_checksums,
        
        # Performance metrics
        'performance': {
            'total_pixels': config.width * config.height,
            'pixels_per_frame': config.width * config.height,
            'total_frames': config.loop_frames if config.animate else 1,
            'pixels_per_second': (config.width * config.height * 
                                (config.loop_frames if config.animate else 1)) / max(render_time, 0.001)
        }
    }
    
    return provenance


def save_provenance(provenance: Dict[str, Any], output_path: str) -> str:
    """
    Save provenance record to JSON file.
    
    Args:
        provenance: Provenance dictionary
        output_path: Base output path
        
    Returns:
        Path to saved JSON file
    """
    json_path = f"{output_path}.json"
    
    with open(json_path, 'w') as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)
    
    return json_path


def load_provenance(json_path: str) -> Dict[str, Any]:
    """
    Load provenance record from JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Provenance dictionary
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def config_from_provenance(provenance: Dict[str, Any]) -> FractalConfig:
    """
    Reconstruct FractalConfig from provenance record.
    
    Args:
        provenance: Provenance dictionary
        
    Returns:
        Reconstructed FractalConfig
    """
    cfg = provenance['config']
    
    return FractalConfig(
        width=cfg['width'],
        height=cfg['height'],
        kernel=cfg['kernel'],
        center=tuple(cfg['center']),
        scale=cfg['scale'],
        rotation=cfg['rotation'],
        max_iters=cfg['max_iters'],
        bailout=cfg['bailout'],
        julia_r=cfg['julia_r'],
        julia_theta=cfg['julia_theta'],
        palette_mode=cfg['palette_mode'],
        base_hue=cfg['base_hue'],
        delta_s=cfg['delta_s'],
        delta_l=cfg['delta_l'],
        animate=cfg['animate'],
        loop_frames=cfg['loop_frames'],
        output_path=cfg['output_path']
    )


def verify_reproducibility(original_provenance: Dict[str, Any], 
                          new_frames: List[np.ndarray]) -> Dict[str, Any]:
    """
    Verify that re-rendered frames match original checksums.
    
    Args:
        original_provenance: Original provenance record
        new_frames: Newly rendered frames
        
    Returns:
        Verification results
    """
    original_checksums = {f['frame']: f['checksum'] 
                         for f in original_provenance.get('frames', [])}
    
    results = {
        'total_frames': len(new_frames),
        'matches': 0,
        'mismatches': 0,
        'details': []
    }
    
    for i, frame in enumerate(new_frames):
        frame_bytes = frame.astype(np.float32).tobytes()
        new_checksum = hashlib.sha256(frame_bytes).hexdigest()[:16]
        original_checksum = original_checksums.get(i, None)
        
        match = original_checksum == new_checksum
        if match:
            results['matches'] += 1
        else:
            results['mismatches'] += 1
        
        results['details'].append({
            'frame': i,
            'original_checksum': original_checksum,
            'new_checksum': new_checksum,
            'match': match
        })
    
    results['reproducible'] = results['mismatches'] == 0
    return results