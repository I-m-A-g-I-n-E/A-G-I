# Jupyter Integration Strategy for 48-Manifold Site
## Architectural Options for Interactive Notebooks

### Overview
While GitHub Pages serves static content, there are several strategies to incorporate Jupyter notebooks with `ipywidgets` into the site architecture. Each approach offers different levels of interactivity, cost, and complexity.

---

## Option 1: JupyterLite (Recommended for Initial Launch)
**Run Jupyter entirely in the browser using WebAssembly**

### Architecture
```
GitHub Pages (Static Host)
    ├── Main Site (Jekyll)
    └── /lab/* → JupyterLite Instance
         ├── Pyodide kernel
         ├── ipywidgets support
         └── Pre-loaded notebooks
```

### Implementation
```yaml
# Add to site structure
/jupyter-lite/
  ├── jupyter-lite.json     # Configuration
  ├── files/                # Pre-loaded notebooks
  │   ├── protein_composer.ipynb
  │   ├── fractal_explorer.ipynb
  │   ├── hand_tensor_demo.ipynb
  │   └── tutorials/
  └── requirements.txt      # Packages to pre-install
```

### Example JupyterLite Configuration
```json
{
  "jupyter-lite-schema-version": 1,
  "jupyter-config-data": {
    "appUrl": "/lab",
    "notebookPage": "tree",
    "exposeAppInBrowser": true,
    "collaborative": false
  },
  "pyodide-url": "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js"
}
```

### Advantages
- ✅ **No server required** - runs entirely in browser
- ✅ **Free hosting** via GitHub Pages
- ✅ **Full ipywidgets support** (most widgets work)
- ✅ **Seamless integration** with static site
- ✅ **Offline capable** after initial load

### Limitations
- ⚠️ Initial load time (~30-60 seconds)
- ⚠️ Limited to pure Python packages
- ⚠️ No GPU acceleration
- ⚠️ Memory limited by browser (~2-4GB)

### Sample Notebook with ipywidgets
```python
# protein_composer.ipynb
import ipywidgets as widgets
from IPython.display import display, HTML
import numpy as np

# Create interactive protein composer
sequence_input = widgets.Textarea(
    value='MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG',
    placeholder='Enter amino acid sequence',
    description='Sequence:',
    layout=widgets.Layout(width='100%', height='100px')
)

samples_slider = widgets.IntSlider(
    value=8, min=1, max=32, step=1,
    description='Samples:',
    style={'description_width': 'initial'}
)

variability_slider = widgets.FloatSlider(
    value=0.5, min=0, max=1, step=0.1,
    description='Variability:',
    style={'description_width': 'initial'}
)

output_area = widgets.Output()

def compose_structure(btn):
    with output_area:
        output_area.clear_output()
        print(f"Composing with {samples_slider.value} samples...")
        # Run actual 48-manifold composition
        from bio.composer import HarmonicPropagator
        composer = HarmonicPropagator(
            n_layers=4, 
            variability=variability_slider.value,
            seed=42
        )
        comp = composer(sequence_input.value)
        print(f"Generated {comp.shape[0]} windows")
        
        # Create interactive 3D plot
        plot_3d_structure(comp)

compose_btn = widgets.Button(
    description='Generate Structure',
    button_style='primary',
    icon='check'
)
compose_btn.on_click(compose_structure)

# Layout
display(widgets.VBox([
    widgets.HTML('<h2>🧬 Interactive Protein Composer</h2>'),
    sequence_input,
    widgets.HBox([samples_slider, variability_slider]),
    compose_btn,
    output_area
]))
```

---

## Option 2: Binder Integration
**Launch full Jupyter environments on-demand**

### Architecture
```
GitHub Pages
    ├── Main Site
    └── "Launch in Binder" buttons → MyBinder.org
         └── Spawns full Jupyter instance
              ├── Full Python environment
              ├── GPU support possible
              └── All packages available
```

### Implementation
```markdown
<!-- In GitHub Pages -->
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/48manifold/notebooks/main?labpath=protein_composer.ipynb)
```

### Binder Configuration
```yaml
# environment.yml
name: manifold48
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - scipy
  - matplotlib
  - ipywidgets
  - pythreejs  # 3D visualization
  - ipyvolume  # 3D volume rendering
  - bqplot     # Interactive plotting
  - pip:
    - torch
    - bio-composer
```

### Advantages
- ✅ Full Python environment
- ✅ No infrastructure to maintain
- ✅ Free for public repos
- ✅ GPU instances available

### Limitations
- ⚠️ 2-5 minute startup time
- ⚠️ Sessions expire after inactivity
- ⚠️ Limited concurrent users
- ⚠️ External dependency

---

## Option 3: JupyterHub Deployment
**Dedicated server for authenticated users**

### Architecture
```
GitHub Pages (Frontend)
    ├── Main Site
    └── "Launch Notebook" → JupyterHub (Separate Server)
         ├── User authentication
         ├── Persistent storage
         ├── GPU support
         └── Custom environments
```

### Deployment Options

#### 3a. Kubernetes with Zero to JupyterHub
```yaml
# config.yaml for helm
hub:
  config:
    Authenticator:
      allowed_users:
        - researcher1
        - researcher2
    
singleuser:
  image:
    name: 48manifold/notebook
    tag: latest
  memory:
    guarantee: 2G
    limit: 8G
  storage:
    capacity: 10Gi
```

#### 3b. The Littlest JupyterHub (TLJH)
```bash
# For smaller deployments (< 100 users)
curl -L https://tljh.jupyter.org/bootstrap.py | sudo python3 - \
  --admin admin-user \
  --plugin git+https://github.com/48manifold/tljh-plugin
```

### Advantages
- ✅ Full control over environment
- ✅ GPU support
- ✅ Persistent user storage
- ✅ Custom authentication
- ✅ Real-time collaboration

### Limitations
- ⚠️ Requires server infrastructure ($50-500/month)
- ⚠️ Maintenance overhead
- ⚠️ Security considerations

---

## Option 4: Hybrid Approach (Recommended for Production)
**Combine static demos with optional full notebooks**

### Architecture
```
┌─────────────────────────────────────────┐
│         GitHub Pages (Main Site)         │
├─────────────────────────────────────────┤
│                                          │
│  Simple Demos → JupyterLite (in-browser) │
│                                          │
│  Advanced Work → Binder (free tier)      │
│                                          │
│  Premium Users → JupyterHub (dedicated)  │
│                                          │
└─────────────────────────────────────────┘
```

### Implementation Strategy

#### Phase 1: JupyterLite for Everyone
```html
<!-- Embedded in GitHub Pages -->
<iframe
  src="/jupyter-lite/lab/index.html?path=tutorials/quickstart.ipynb"
  width="100%"
  height="600px"
></iframe>
```

#### Phase 2: Binder for Researchers
```html
<!-- For compute-intensive work -->
<div class="notebook-launcher">
  <h3>Need more compute power?</h3>
  <a href="https://mybinder.org/v2/gh/48manifold/core/main?labpath=research/advanced.ipynb" 
     class="btn btn-primary">
    Launch Full Environment
  </a>
  <p>Free cloud instance with GPU support (may take 2-5 minutes to start)</p>
</div>
```

#### Phase 3: JupyterHub for Partners
```python
# Custom spawner for premium users
c.JupyterHub.spawner_class = 'kubespawner.KubeSpawner'
c.KubeSpawner.profile_list = [
    {
        'display_name': 'Minimal (2 CPU, 4GB RAM)',
        'kubespawner_override': {
            'cpu_limit': 2,
            'mem_limit': '4G',
        }
    },
    {
        'display_name': 'GPU Instance (4 CPU, 16GB RAM, 1 GPU)',
        'kubespawner_override': {
            'cpu_limit': 4,
            'mem_limit': '16G',
            'extra_resource_limits': {"nvidia.com/gpu": "1"},
        }
    }
]
```

---

## Interactive Widget Examples for 48-Manifold

### 1. Fractal Navigator Widget
```python
import ipywidgets as widgets
import numpy as np
from ipycanvas import Canvas
import ipyvolume as ipv

class Fractal48Navigator(widgets.VBox):
    def __init__(self):
        # Create canvas for 2D view
        self.canvas = Canvas(width=800, height=400)
        
        # Controls
        self.zoom = widgets.FloatSlider(
            value=1.0, min=0.1, max=10.0,
            description='Zoom:',
            continuous_update=False
        )
        
        self.depth = widgets.IntSlider(
            value=3, min=1, max=8,
            description='Depth:',
            continuous_update=False
        )
        
        self.mode = widgets.Dropdown(
            options=['keven', 'kodd', 'kore', 'full'],
            value='full',
            description='Mode:'
        )
        
        # 3D view
        self.figure_3d = ipv.figure()
        
        # Wire up events
        self.zoom.observe(self.update_view, 'value')
        self.depth.observe(self.update_view, 'value')
        self.mode.observe(self.update_view, 'value')
        
        # Layout
        super().__init__([
            widgets.HTML('<h3>48-Manifold Fractal Navigator</h3>'),
            widgets.HBox([self.zoom, self.depth, self.mode]),
            self.canvas,
            self.figure_3d
        ])
        
        self.update_view()
    
    def update_view(self, change=None):
        # Generate fractal based on parameters
        self.render_fractal()
        self.render_3d()
    
    def render_fractal(self):
        # Render 2D fractal on canvas
        with self.canvas:
            # Clear and draw fractal pattern
            pass
    
    def render_3d(self):
        # Update 3D visualization
        ipv.clear()
        # Generate 3D manifold visualization
        x, y, z = self.generate_manifold_points()
        ipv.scatter(x, y, z, marker='sphere', size=0.5)
        
# Display widget
navigator = Fractal48Navigator()
display(navigator)
```

### 2. Protein Structure Viewer
```python
import nglview as nv
import ipywidgets as widgets

class ProteinViewer(widgets.VBox):
    def __init__(self, pdb_path=None):
        # 3D molecular viewer
        self.viewer = nv.NGLWidget()
        if pdb_path:
            self.viewer.add_structure(nv.FileStructure(pdb_path))
        
        # Controls
        self.style = widgets.Dropdown(
            options=['cartoon', 'ball+stick', 'surface', 'ribbon'],
            value='cartoon',
            description='Style:'
        )
        
        self.color = widgets.Dropdown(
            options=['chainid', 'residue', 'secondary structure', 'hydrophobicity'],
            value='chainid',
            description='Color:'
        )
        
        self.style.observe(self.update_style, 'value')
        self.color.observe(self.update_color, 'value')
        
        super().__init__([
            widgets.HTML('<h3>3D Protein Structure</h3>'),
            widgets.HBox([self.style, self.color]),
            self.viewer
        ])
    
    def update_style(self, change):
        self.viewer.clear_representations()
        self.viewer.add_representation(change['new'])
    
    def update_color(self, change):
        # Update coloring scheme
        pass

# Usage
viewer = ProteinViewer('outputs/ubiquitin.pdb')
display(viewer)
```

### 3. Real-time Sonification Player
```python
from ipywidgets import Audio, Play
import numpy as np

class SonificationPlayer(widgets.VBox):
    def __init__(self, composition_data):
        self.composition = composition_data
        
        # Generate audio
        self.audio_data = self.sonify(composition_data)
        
        # Audio widget
        self.audio = Audio(
            value=self.audio_data,
            autoplay=False,
            controls=True,
            loop=False
        )
        
        # Playback controls
        self.play = Play(
            value=0,
            min=0,
            max=len(self.audio_data),
            step=1,
            interval=100,
            description="Playing:"
        )
        
        # Visualization
        self.waveform = widgets.Output()
        
        super().__init__([
            widgets.HTML('<h3>🎵 Protein Sonification</h3>'),
            self.audio,
            self.play,
            self.waveform
        ])
    
    def sonify(self, data):
        # Convert composition to audio
        sample_rate = 48000
        duration = len(data) / 16  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Three channels: keven, kore, kodd
        left = np.sin(2 * np.pi * 440 * t)  # keven
        center = np.sin(2 * np.pi * 523.25 * t)  # kore (C5)
        right = np.sin(2 * np.pi * 659.25 * t)  # kodd
        
        return np.stack([left, center, right])

# Usage
player = SonificationPlayer(composition_data)
display(player)
```

---

## Recommended Implementation Path

### Phase 1: Launch with JupyterLite (Week 1-2)
1. Set up JupyterLite build process
2. Create starter notebooks with ipywidgets
3. Embed in GitHub Pages via iframes
4. Test browser compatibility

### Phase 2: Add Binder Support (Week 3)
1. Create binder/ directory with environment specs
2. Add "Launch in Binder" badges
3. Create advanced notebooks requiring more compute
4. Document usage patterns

### Phase 3: Evaluate JupyterHub Need (Month 2+)
1. Monitor usage metrics
2. Survey user needs
3. If justified, deploy TLJH on cloud VM
4. Implement authentication system

---

## Cost-Benefit Analysis

| Solution | Initial Cost | Monthly Cost | Setup Time | User Experience | Best For |
|----------|-------------|--------------|------------|-----------------|----------|
| JupyterLite | $0 | $0 | 1-2 days | Good (slow start) | Demos, tutorials |
| Binder | $0 | $0 | 2-3 hours | Good (wait time) | Workshops, research |
| JupyterHub (small) | $100 | $50-100 | 1 week | Excellent | Team collaboration |
| JupyterHub (large) | $500 | $200-500 | 2 weeks | Excellent | Production/partners |
| Hybrid | $0-100 | $0-100 | 1-2 weeks | Excellent | All audiences |

---

## Technical Integration Code

### Seamless Embedding in Jekyll
```html
---
layout: page
title: Interactive Notebook
---

<style>
.notebook-container {
    width: 100%;
    height: 800px;
    border: 1px solid #ddd;
    border-radius: 8px;
    overflow: hidden;
}

.notebook-controls {
    padding: 1rem;
    background: #f5f5f5;
    border-bottom: 1px solid #ddd;
}
</style>

<div class="notebook-container">
    <div class="notebook-controls">
        <button onclick="openNotebook('basic')">Basic Tutorial</button>
        <button onclick="openNotebook('advanced')">Advanced Features</button>
        <button onclick="launchBinder()">Full Environment</button>
    </div>
    <iframe id="notebook-frame" 
            src="/jupyter-lite/lab/index.html?path=tutorials/intro.ipynb"
            width="100%" 
            height="750px">
    </iframe>
</div>

<script>
function openNotebook(level) {
    const frame = document.getElementById('notebook-frame');
    frame.src = `/jupyter-lite/lab/index.html?path=tutorials/${level}.ipynb`;
}

function launchBinder() {
    window.open('https://mybinder.org/v2/gh/48manifold/notebooks/main', '_blank');
}
</script>
```

---

## Conclusion

For the 48-Manifold project, I recommend starting with **JupyterLite** for immediate deployment with zero infrastructure cost, then expanding to include **Binder** integration for users needing more computational resources. This provides:

1. **Instant interactivity** without leaving the GitHub Pages site
2. **Full ipywidgets support** for rich visualizations
3. **Zero infrastructure cost** initially
4. **Scalable path** to JupyterHub if needed

The hybrid approach allows you to serve different user segments effectively:
- **Casual visitors**: Quick demos in JupyterLite
- **Researchers**: Full environments via Binder
- **Partners/Premium**: Dedicated JupyterHub instances

This strategy maintains the simplicity of GitHub Pages while providing the full power of Jupyter notebooks with interactive widgets when needed.