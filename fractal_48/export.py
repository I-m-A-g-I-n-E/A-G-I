"""
Export functionality for PNG, GIF, and MP4 with metadata.
"""

import numpy as np
from typing import List, Optional
import os
import json

try:
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    PngInfo = None

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

from .config import FractalConfig
from .provenance import create_provenance, save_provenance


def save_frame_png(frame: np.ndarray, output_path: str, 
                   config: FractalConfig, provenance: Optional[dict] = None) -> str:
    """
    Save a single frame as PNG with embedded metadata.
    
    Args:
        frame: RGB frame array [0, 1]
        output_path: Output file path (without extension)
        config: Fractal configuration
        provenance: Optional provenance data
        
    Returns:
        Path to saved PNG file
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL/Pillow is required for PNG export")
    
    # Convert to 8-bit RGB
    frame_8bit = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    
    # Create PIL Image
    img = Image.fromarray(frame_8bit, 'RGB')
    
    # Add metadata if available
    if provenance and PngInfo is not None:
        # PNG text metadata
        pnginfo = PngInfo()
        pnginfo.add_text("Description", f"Fractal-48 {config.kernel} render")
        pnginfo.add_text("Software", "Fractal-48 v0.1.0")
        pnginfo.add_text("Configuration", json.dumps(provenance.get('config', {})))
        pnginfo.add_text("Manifold", "48 = 2^4 × 3")
        img.save(f"{output_path}.png", pnginfo=pnginfo)
    else:
        img.save(f"{output_path}.png")
    
    return f"{output_path}.png"


def save_animation_gif(frames: List[np.ndarray], output_path: str,
                      config: FractalConfig, duration: float = 2.0) -> str:
    """
    Save 48-frame loop as GIF.
    
    Args:
        frames: List of RGB frame arrays
        output_path: Output file path (without extension)
        config: Fractal configuration
        duration: Total duration in seconds
        
    Returns:
        Path to saved GIF file
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL/Pillow is required for GIF export")
    
    # Convert frames to 8-bit RGB
    frames_8bit = []
    for frame in frames:
        frame_8bit = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        frames_8bit.append(Image.fromarray(frame_8bit, 'RGB'))
    
    # Calculate frame duration
    frame_duration = int(1000 * duration / len(frames))  # milliseconds
    
    # Save as animated GIF
    gif_path = f"{output_path}.gif"
    frames_8bit[0].save(
        gif_path,
        save_all=True,
        append_images=frames_8bit[1:],
        duration=frame_duration,
        loop=0,  # infinite loop
        optimize=False  # preserve quality
    )
    
    return gif_path


def save_animation_mp4(frames: List[np.ndarray], output_path: str,
                      config: FractalConfig, fps: float = 24.0) -> str:
    """
    Save 48-frame loop as MP4 video.
    
    Args:
        frames: List of RGB frame arrays
        output_path: Output file path (without extension)
        config: Fractal configuration
        fps: Frames per second
        
    Returns:
        Path to saved MP4 file
    """
    if not IMAGEIO_AVAILABLE:
        raise ImportError("imageio is required for MP4 export")
    
    # Convert frames to 8-bit RGB
    frames_8bit = []
    for frame in frames:
        frame_8bit = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        frames_8bit.append(frame_8bit)
    
    # Save as MP4
    mp4_path = f"{output_path}.mp4"
    
    with imageio.get_writer(mp4_path, fps=fps, codec='libx264', quality=8) as writer:
        for frame in frames_8bit:
            writer.append_data(frame)
    
    return mp4_path


def export_complete_render(frames: List[np.ndarray], config: FractalConfig,
                          render_time: float = 0.0) -> dict:
    """
    Export complete render with all formats and provenance.
    
    Args:
        frames: Rendered frames
        config: Fractal configuration
        render_time: Total rendering time
        
    Returns:
        Dictionary with paths to all exported files
    """
    # Create provenance record
    provenance = create_provenance(config, render_time, frames)
    
    # Create output directory if needed
    output_dir = os.path.dirname(config.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    exports = {}
    
    # Save provenance JSON
    json_path = save_provenance(provenance, config.output_path)
    exports['provenance'] = json_path
    
    if len(frames) == 1:
        # Single frame export
        png_path = save_frame_png(frames[0], config.output_path, config, provenance)
        exports['png'] = png_path
    else:
        # Multi-frame export
        # Save first frame as reference PNG
        png_path = save_frame_png(frames[0], f"{config.output_path}_frame_0", config, provenance)
        exports['reference_png'] = png_path
        
        # Save animation as GIF
        try:
            gif_path = save_animation_gif(frames, config.output_path, config, duration=2.0)
            exports['gif'] = gif_path
        except ImportError as e:
            print(f"Warning: Could not export GIF: {e}")
        
        # Save animation as MP4
        try:
            mp4_path = save_animation_mp4(frames, config.output_path, config, fps=24.0)
            exports['mp4'] = mp4_path
        except ImportError as e:
            print(f"Warning: Could not export MP4: {e}")
    
    return exports


def validate_exports(exports: dict, config: FractalConfig) -> dict:
    """
    Validate that exported files exist and have expected properties.
    
    Args:
        exports: Dictionary of export paths
        config: Fractal configuration
        
    Returns:
        Validation results
    """
    results = {
        'valid': True,
        'files': {},
        'errors': []
    }
    
    for export_type, path in exports.items():
        if not os.path.exists(path):
            results['valid'] = False
            results['errors'].append(f"Missing file: {path}")
            results['files'][export_type] = {'exists': False}
        else:
            file_size = os.path.getsize(path)
            results['files'][export_type] = {
                'exists': True,
                'size_bytes': file_size,
                'path': path
            }
            
            # Additional validation for specific formats
            if path.endswith('.png') and PIL_AVAILABLE:
                try:
                    img = Image.open(path)
                    results['files'][export_type].update({
                        'width': img.width,
                        'height': img.height,
                        'mode': img.mode
                    })
                    
                    # Check dimensions match config
                    if img.width != config.width or img.height != config.height:
                        results['errors'].append(f"PNG dimensions mismatch: {img.width}x{img.height} != {config.width}x{config.height}")
                        results['valid'] = False
                        
                except Exception as e:
                    results['errors'].append(f"PNG validation error: {e}")
                    results['valid'] = False
    
    return results