#!/usr/bin/env python3
"""
Command-line interface for the Fractal-48 renderer.

Usage:
    python -m fractal_48.cli render --kernel mandelbrot --width 1920 --height 1152 \\
        --center -0.75 0.0 --scale 600 --iters 1000 --animate --out output/mandelbrot

    python -m fractal_48.cli from-json output/mandelbrot.json
"""

import click
import time
import sys
from typing import Tuple

from .config import FractalConfig
from .render import render_frame, render_loop, benchmark_render_performance
from .export import export_complete_render, validate_exports
from .provenance import load_provenance, config_from_provenance, verify_reproducibility


@click.group()
def cli():
    """Fractal-48: 48-manifold fractal renderer with reversible animations."""
    pass


@cli.command()
@click.option('--kernel', type=click.Choice(['mandelbrot', 'julia', 'newton']), 
              default='mandelbrot', help='Fractal kernel to use')
@click.option('--width', type=int, default=1920, help='Canvas width (must be 48-aligned)')
@click.option('--height', type=int, default=1152, help='Canvas height (must be 48-aligned)')
@click.option('--center', type=(float, float), default=(-0.75, 0.0), 
              help='Complex plane center coordinates')
@click.option('--scale', type=float, default=600.0, help='Pixels per unit scale')
@click.option('--rotation', type=float, default=0.0, help='Rotation in degrees')
@click.option('--iters', type=int, default=1000, help='Maximum iterations')
@click.option('--bailout', type=float, default=4.0, help='Escape threshold')
@click.option('--julia-r', type=float, default=0.4, help='Julia parameter radius')
@click.option('--julia-theta', type=float, default=0.0, help='Julia parameter phase offset')
@click.option('--palette', type=click.Choice(['smooth']), default='smooth', 
              help='Color palette mode')
@click.option('--base-hue', type=float, default=210.0, help='Base hue for color cycling')
@click.option('--delta-s', type=float, default=0.05, help='Saturation parity modulation')
@click.option('--delta-l', type=float, default=0.04, help='Lightness parity modulation')
@click.option('--animate/--no-animate', default=False, help='Generate 48-frame animation')
@click.option('--out', type=str, default='fractal_48_output', help='Output path (without extension)')
@click.option('--benchmark/--no-benchmark', default=False, help='Run performance benchmark')
def render(kernel: str, width: int, height: int, center: Tuple[float, float],
          scale: float, rotation: float, iters: int, bailout: float,
          julia_r: float, julia_theta: float, palette: str, base_hue: float,
          delta_s: float, delta_l: float, animate: bool, out: str, benchmark: bool):
    """Render fractal with specified parameters."""
    
    try:
        # Create configuration
        config = FractalConfig(
            width=width,
            height=height,
            kernel=kernel,
            center=center,
            scale=scale,
            rotation=rotation,
            max_iters=iters,
            bailout=bailout,
            julia_r=julia_r,
            julia_theta=julia_theta,
            palette_mode=palette,
            base_hue=base_hue,
            delta_s=delta_s,
            delta_l=delta_l,
            animate=animate,
            output_path=out
        )
        
        click.echo(f"Rendering {kernel} fractal at {width}×{height}")
        click.echo(f"Center: {center}, Scale: {scale}")
        if animate:
            click.echo(f"Animation: {config.loop_frames} frames")
        
        # Run benchmark if requested
        if benchmark:
            click.echo("Running performance benchmark...")
            metrics = benchmark_render_performance(config)
            click.echo(f"Single frame: {metrics['single_frame_time']:.3f}s")
            click.echo(f"Pixels/second: {metrics['pixels_per_second']:.0f}")
            if animate:
                click.echo(f"Estimated 48-frame time: {metrics['estimated_48_frame_time']:.1f}s")
        
        # Render frames
        start_time = time.time()
        if animate:
            frames = render_loop(config)
        else:
            frames = [render_frame(config, 0)]
        render_time = time.time() - start_time
        
        click.echo(f"Rendering completed in {render_time:.2f}s")
        
        # Export results
        click.echo("Exporting files...")
        exports = export_complete_render(frames, config, render_time)
        
        # Validate exports
        validation = validate_exports(exports, config)
        
        if validation['valid']:
            click.echo("✓ Export successful!")
            for export_type, file_info in validation['files'].items():
                if file_info['exists']:
                    size_mb = file_info['size_bytes'] / (1024 * 1024)
                    click.echo(f"  {export_type}: {file_info['path']} ({size_mb:.2f} MB)")
        else:
            click.echo("✗ Export validation failed:")
            for error in validation['errors']:
                click.echo(f"  Error: {error}")
            sys.exit(1)
            
    except ValueError as e:
        click.echo(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Render error: {e}")
        sys.exit(1)


@cli.command('from-json')
@click.argument('json_path', type=click.Path(exists=True))
@click.option('--verify/--no-verify', default=True, help='Verify reproducibility')
@click.option('--out', type=str, default=None, help='Override output path')
def from_json(json_path: str, verify: bool, out: str):
    """Reproduce render from provenance JSON file."""
    
    try:
        # Load provenance
        click.echo(f"Loading provenance from {json_path}")
        provenance = load_provenance(json_path)
        
        # Reconstruct configuration
        config = config_from_provenance(provenance)
        if out:
            config.output_path = out
        
        click.echo(f"Reproducing {config.kernel} render from {provenance['timestamp']}")
        click.echo(f"Original git commit: {provenance['git']['commit'][:8]}")
        
        # Render frames
        start_time = time.time()
        if config.animate:
            frames = render_loop(config)
        else:
            frames = [render_frame(config, 0)]
        render_time = time.time() - start_time
        
        # Verify reproducibility if requested
        if verify and 'frames' in provenance:
            click.echo("Verifying reproducibility...")
            verification = verify_reproducibility(provenance, frames)
            
            if verification['reproducible']:
                click.echo(f"✓ Reproducible! {verification['matches']}/{verification['total_frames']} frames match")
            else:
                click.echo(f"✗ Not reproducible: {verification['mismatches']}/{verification['total_frames']} frames differ")
                if verification['mismatches'] <= 3:  # Show details for small number of mismatches
                    for detail in verification['details']:
                        if not detail['match']:
                            click.echo(f"  Frame {detail['frame']}: {detail['original_checksum']} → {detail['new_checksum']}")
        
        # Export results
        click.echo("Exporting files...")
        exports = export_complete_render(frames, config, render_time)
        
        click.echo("✓ Reproduction complete!")
        for export_type, path in exports.items():
            click.echo(f"  {export_type}: {path}")
            
    except Exception as e:
        click.echo(f"Reproduction error: {e}")
        sys.exit(1)


@cli.command()
@click.option('--width', type=int, default=1920, help='Test canvas width')
@click.option('--height', type=int, default=1152, help='Test canvas height')
def test_permutations(width: int, height: int):
    """Test reversible permutation operations."""
    import numpy as np
    from .perms import verify_permutation_invertibility
    from .config import FractalConfig
    
    click.echo(f"Testing permutation invertibility on {width}×{height} canvas")
    
    # Create test configuration
    config = FractalConfig(width=width, height=height, animate=True)
    
    # Create test image
    test_img = np.random.rand(height, width, 3).astype(np.float32)
    
    # Test all 48 frame permutations
    passed = 0
    failed = 0
    
    for frame_idx in range(48):
        if verify_permutation_invertibility(test_img, frame_idx, config):
            passed += 1
        else:
            failed += 1
            click.echo(f"  Frame {frame_idx}: FAILED")
    
    click.echo(f"Results: {passed}/48 passed, {failed}/48 failed")
    
    if failed == 0:
        click.echo("✓ All permutations are invertible!")
    else:
        click.echo(f"✗ {failed} permutations failed invertibility test")
        sys.exit(1)


@cli.command()
def examples():
    """Show example command lines for common use cases."""
    examples_text = """
Fractal-48 Example Commands:

1. Basic Mandelbrot render (single frame):
   python -m fractal_48.cli render --kernel mandelbrot --width 1920 --height 1152 \\
       --center -0.75 0.0 --scale 600 --iters 1000 --out mandelbrot_basic

2. Mandelbrot 48-frame animation:
   python -m fractal_48.cli render --kernel mandelbrot --animate \\
       --width 1536 --height 1152 --center -0.75 0.0 --scale 400 \\
       --out animations/mandelbrot_loop

3. Julia set with parameter sweep:
   python -m fractal_48.cli render --kernel julia --animate \\
       --width 1536 --height 1152 --julia-r 0.4 --julia-theta 0.0 \\
       --iters 600 --out animations/julia_sweep

4. Newton fractal (z³-1):
   python -m fractal_48.cli render --kernel newton --width 1728 --height 1152 \\
       --center 0.0 0.0 --scale 800 --iters 100 --out newton_basins

5. Performance benchmark:
   python -m fractal_48.cli render --kernel mandelbrot --benchmark \\
       --width 960 --height 576

6. Reproduce from JSON:
   python -m fractal_48.cli from-json output/mandelbrot_basic.json

7. Test permutation invertibility:
   python -m fractal_48.cli test-permutations --width 1920 --height 1152

Note: All width/height values must be divisible by 48 for proper 48-manifold alignment.
"""
    click.echo(examples_text)


if __name__ == '__main__':
    cli()