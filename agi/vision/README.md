# Vision Module - Fractal Structure Visualization

The `agi.vision` module provides comprehensive visualization tools for exploring and understanding the fractal structures inherent in the 48-manifold system.

## Overview

This module models the unfolding fractal structure as described by the mathematical complexity functions in the 48-manifold framework. It provides three main components:

1. **FractalVisualizer**: Visualizes the hierarchical decomposition through the factorization ladder (48 → 16 → 8 → 4 → 2)
2. **ManifoldRenderer**: Renders the 48-manifold structure in 3D space
3. **ComplexityMapper**: Maps and analyzes fractal complexity across different representations

## Mathematical Foundation

The visualization is based on the 48-manifold's unique factorization:
- **48 = 2^4 × 3**: Perfect factorability enabling exact space/time-to-depth transformations
- **Factorization Ladder**: (3, 2, 2, 2) decomposition steps
- **Fractal Complexity**: Measured through grid affinity, micro-offset costs, and chirality penalties

## Key Features

### FractalVisualizer
- **Factorization Ladder Visualization**: Shows the complete decomposition from 48 down to 2
- **Fractal Unfolding**: Visualizes how composition vectors unfold into fractal structures
- **Evolution Animation**: Animates the evolution of fractal states over time
- **Complexity Tracking**: Color-codes visualizations by fractal complexity

### ManifoldRenderer
- **Six-Axis Semantic State**: Renders the 6D semantic manifold (who, what, when, where, why, how)
- **Lattice Structure**: 3D visualization of the 48-manifold lattice (16 × 3 structure)
- **Factorization Flow**: Shows information flow through decomposition levels
- **Phase Torus**: Renders phase space on a torus to show cyclic nature

### ComplexityMapper
- **Tensor Complexity**: Computes fractal complexity for each element
- **Movement Complexity**: Analyzes gesture and handedness effects
- **Complexity Landscapes**: Creates 2D/3D visualizations of complexity topology
- **Trajectory Analysis**: Tracks complexity evolution over time

## Usage

### Basic Example

```python
from agi.vision import FractalVisualizer, ComplexityMapper
import torch

# Create sample data
tensor = torch.randn(48) * 0.5

# Visualize factorization ladder
viz = FractalVisualizer()
fig = viz.visualize_factorization_ladder(tensor, show_complexity=True)
fig.savefig('fractal_ladder.png')

# Analyze complexity
mapper = ComplexityMapper()
complexities = mapper.compute_tensor_complexity(tensor)
fig = mapper.visualize_complexity_heatmap(tensor)
fig.savefig('complexity_heatmap.png')
```

### Protein Structure Visualization

```python
from bio.composer import HarmonicPropagator
from agi.vision import FractalVisualizer

# Compose a protein sequence
seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
composer = HarmonicPropagator()
composition = composer(seq)

# Visualize fractal unfolding
viz = FractalVisualizer()
fig = viz.visualize_fractal_unfolding(composition[0])
fig.savefig('protein_fractal.png')
```

### 3D Manifold Rendering

```python
from agi.vision import ManifoldRenderer
from manifold import SixAxisState, RouterMode
import torch

# Create six-axis state
state = SixAxisState(
    who=torch.randn(48),
    what=torch.randn(48),
    when=torch.randn(48),
    where=torch.randn(48),
    why=torch.randn(48),
    how=torch.randn(48)
)

# Render in 3D
renderer = ManifoldRenderer()
fig = renderer.render_six_axis_state(state, RouterMode.W_POSSIBILITY)
fig.savefig('six_axis.png')
```

## Complexity Metrics

The fractal complexity calculation considers:

1. **Grid Affinity**: Preference for values representable with small-denominator fractions on the 48-tick lattice
2. **Micro-offset Cost**: Penalty for off-grid fractional remainders
3. **Chirality Cost**: Left-handed movements incur higher complexity (4x multiplier)

Formula:
```
complexity = base_complexity × chirality_cost
where:
  base_complexity = log₂(denominator) + |micro_offset|
  chirality_cost = 1.0 (RIGHT) or 4.0 (LEFT)
```

## Demo Scripts

Run the comprehensive demo:
```bash
python demo_fractal_viz.py --demo all
```

Individual demos:
- `--demo fractal`: Fractal structure visualization
- `--demo manifold`: 3D manifold rendering
- `--demo complexity`: Complexity analysis
- `--demo protein`: Protein structure complexity

## Dependencies

Required:
- `torch`
- `numpy`
- `matplotlib`

Optional (for enhanced features):
- `plotly`: Interactive 3D visualizations
- `seaborn`: Enhanced heatmaps

## Output Examples

The module generates various visualization types:
- **Factorization Ladder**: 3D surfaces showing each decomposition level
- **Complexity Heatmaps**: 2D color-coded complexity distributions
- **Phase Space Plots**: Torsion angle distributions in Ramachandran-like plots
- **Fractal Trees**: Hierarchical decomposition structures
- **Complexity Landscapes**: 3D surfaces showing complexity topology

## Integration with Core System

The visualization module integrates seamlessly with:
- `bio.composer`: Visualize harmonic compositions
- `bio.conductor`: Analyze backbone structures and torsions
- `agi.harmonia.notation`: Compute gesture and movement complexities
- `manifold`: Render six-axis states and router modes

## Future Enhancements

Planned features:
- Real-time visualization during refinement
- WebGL-based interactive rendering
- VR/AR support for immersive exploration
- Complexity-guided optimization visualizations
- Multi-scale fractal zoom capabilities

## Kid-Friendly Games

Three interactive versions for children:

### 1. Web Browser Game (`fractal_web_game.html`)
- **No installation needed** - just open in browser
- Touch/click to grow fractal friends
- Emoji characters with personalities
- Sound effects and particles
- Works on tablets and phones

### 2. Python Playground (`fractal_playground.py`)
- Animated creatures with moods
- Rainbow and dance modes
- Story mode for learning
- Educational diagrams

### 3. Pygame Engine (`fractal_game.py`)
- Full 60 FPS game engine
- Advanced particle systems
- Score and level system
- Real-time rendering

## Troubleshooting

### Pygame Color Error Fix
The game has been updated to handle RGB color validation properly. All color values are ensured to be integers in the range 0-255.

### Running Without Dependencies
- Web version works without any Python dependencies
- Matplotlib playground works without pygame
- All versions have graceful fallbacks