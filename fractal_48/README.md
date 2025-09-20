# Fractal-48: 48-Manifold Fractal Renderer

A reversible, measurement-first fractal generator that operates on the 2^4×3 (48) manifold using principles of Transfer, not Transform and USK/Harmonic-Integrity.

## Features

- **48-Aligned Canvas**: Width and height must be multiples of 48 for proper manifold alignment
- **CRT-Based 48-Phase Scheduler**: Uses Chinese Remainder Theorem mapping for phase indexing
- **Keven/Kodd Parity Gating**: Separates even/odd pixels for enhanced visual structure
- **Reversible Animations**: 48-frame loops using space-to-depth permutations without aliasing
- **Three Fractal Kernels**:
  - Mandelbrot set (z ← z² + c)
  - Julia sets with 48-phase parameter sweep
  - Newton fractal for z³ - 1
- **Smooth, Parity-Aware Coloring**: HSL→sRGB with gamma 2.2 correction
- **Full Provenance Tracking**: Reproducible renders with embedded metadata
- **Export Formats**: PNG stills, GIF/MP4 animations with JSON provenance

## Installation

Install required dependencies:

```bash
pip install numpy torch matplotlib pillow imageio click
pip install imageio[ffmpeg]  # For MP4 export
```

## Quick Start

### Basic Mandelbrot Render

```bash
python -m fractal_48.cli render \
    --kernel mandelbrot \
    --width 1920 --height 1152 \
    --center -0.75 0.0 \
    --scale 600 \
    --iters 1000 \
    --out mandelbrot_basic
```

### Julia Set Animation

```bash
python -m fractal_48.cli render \
    --kernel julia \
    --animate \
    --width 1536 --height 1152 \
    --julia-r 0.4 \
    --iters 600 \
    --out julia_animation
```

### Newton Fractal

```bash
python -m fractal_48.cli render \
    --kernel newton \
    --width 1728 --height 1152 \
    --center 0.0 0.0 \
    --scale 800 \
    --iters 100 \
    --out newton_basins
```

## 48-Manifold Theory

The system operates on a 48-dimensional manifold based on the factorization:

```
48 = 2^4 × 3 = 16 × 3
```

### CRT Phase Mapping

Each pixel (x, y) maps to a 48-phase coordinate:

```python
d = x % 16        # Dyadic component [0, 15]
t = y % 3         # Triadic component [0, 2]  
phi = d * 3 + t   # Phase index [0, 47]
parity = (x + y) % 2  # 0=keven, 1=kodd
```

### Reversible Permutations

Animation frames use composition of space-to-depth operations:

```
P(f) = (S2)^{f mod 4} ∘ (S3)^{f mod 3}
```

Where S2 and S3 are 2×2 and 3×3 space-to-depth permutations.

## CLI Reference

### Commands

- `render` - Generate fractal images/animations
- `from-json` - Reproduce render from provenance file
- `test-permutations` - Verify permutation invertibility
- `examples` - Show example command lines

### Common Options

- `--kernel` - Fractal type: mandelbrot, julia, newton
- `--width/--height` - Canvas dimensions (must be 48-aligned)
- `--center` - Complex plane center coordinates
- `--scale` - Pixels per unit
- `--iters` - Maximum iterations
- `--animate` - Generate 48-frame animation
- `--out` - Output path (without extension)

### Animation Options

- `--julia-r` - Julia parameter radius
- `--julia-theta` - Julia parameter phase offset

### Color Options

- `--base-hue` - Base hue for color cycling [0, 360)
- `--delta-s` - Saturation parity modulation
- `--delta-l` - Lightness parity modulation

## Python API

```python
from fractal_48 import FractalConfig, render_frame, render_loop

# Create configuration
config = FractalConfig(
    width=1920,
    height=1152,
    kernel="mandelbrot",
    center=(-0.75, 0.0),
    scale=600,
    max_iters=1000,
    animate=True
)

# Render single frame
frame = render_frame(config, 0)

# Render 48-frame loop
frames = render_loop(config)
```

## File Outputs

Each render produces:

- **PNG Image**: High-quality fractal image with embedded metadata
- **JSON Provenance**: Complete configuration and reproducibility data
- **GIF Animation**: 48-frame loop (for animated renders)
- **MP4 Video**: High-quality video (for animated renders)

## Reproducibility

All renders include comprehensive provenance tracking:

```bash
# Reproduce exact render from JSON
python -m fractal_48.cli from-json output.json

# Verify reproducibility
python -m fractal_48.cli from-json output.json --verify
```

## Performance

Typical performance on modern hardware:

- **1920×1152 Mandelbrot**: ~1-2s per frame
- **48-frame animation**: ~2-3 minutes
- **960×576 test render**: ~0.2-0.5s per frame

Use `--benchmark` to measure performance on your system.

## 48-Alignment Requirement

All canvas dimensions must be divisible by 48:

**Valid sizes**: 48, 96, 144, 192, 288, 480, 576, 960, 1152, 1536, 1728, 1920...

**Invalid sizes**: Any number not divisible by 48

## Theory and References

This implementation demonstrates:

- **Transfer, not Transform**: All operations are reversible permutations
- **Measurement-First**: Only final sRGB output constitutes measurement
- **USK/Harmonic-Integrity**: Maintains mathematical structure throughout
- **48-Manifold Properties**: Leverages 2^4×3 factorization for perfect tiling

## Testing

Run the test suite to verify functionality:

```bash
python -m fractal_48.tests
```

Test specific features:

```bash
# Test permutation invertibility
python -m fractal_48.cli test-permutations --width 1920 --height 1152

# Performance benchmark
python -m fractal_48.cli render --benchmark --width 960 --height 576
```

## Examples

See comprehensive examples:

```bash
python -m fractal_48.cli examples
```

## License

Part of the A-G-I 48-Manifold system. See repository license for details.