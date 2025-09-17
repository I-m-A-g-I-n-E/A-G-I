#!/usr/bin/env python3
"""
Prepare notebooks and files for Deepnote upload.
Creates a ZIP file that can be easily uploaded through Deepnote's UI.
"""

import zipfile
import json
from pathlib import Path
from datetime import datetime

def create_remaining_notebooks():
    """Create the Motor and Manifold notebooks that weren't created yet."""
    
    # Motor - Hand Visualization Notebook
    motor_notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🖐️ Motor - Hand Tensor Visualization\\n",
                    "## Five-Finger Routing Through the 48-Manifold\\n",
                    "\\n",
                    "This notebook implements gesture-based tensor routing and visualization."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Setup\\n",
                    "import sys\\n",
                    "sys.path.append('/work')\\n",
                    "\\n",
                    "from deepnote.cuda_devices import get_device\\n",
                    "import torch\\n",
                    "import numpy as np\\n",
                    "import plotly.graph_objects as go\\n",
                    "import ipywidgets as widgets\\n",
                    "from IPython.display import display, HTML\\n",
                    "\\n",
                    "device = get_device()\\n",
                    "print(f'🖐️ Motor notebook initialized on {device}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Hand Tensor Implementation (Placeholder)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "class HandTensorCUDA:\\n",
                    "    '''CUDA-optimized five-finger tensor router.'''\\n",
                    "    \\n",
                    "    def __init__(self, device=None):\\n",
                    "        self.device = device or get_device()\\n",
                    "        self.fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']\\n",
                    "        \\n",
                    "    def create_gesture(self, gesture_type='pointing'):\\n",
                    "        '''Create gesture tensor configuration.'''\\n",
                    "        gestures = {\\n",
                    "            'pointing': [0.1, 1.0, 0.1, 0.1, 0.1],\\n",
                    "            'fist': [1.0, 1.0, 1.0, 1.0, 1.0],\\n",
                    "            'peace': [0.1, 1.0, 1.0, 0.1, 0.1],\\n",
                    "            'ok': [1.0, 1.0, 0.1, 0.1, 0.1],\\n",
                    "            'rock': [0.5, 1.0, 0.1, 0.1, 1.0]\\n",
                    "        }\\n",
                    "        weights = torch.tensor(gestures.get(gesture_type, [1.0]*5), device=self.device)\\n",
                    "        return weights\\n",
                    "\\n",
                    "# TODO: Implement full hand tensor visualization\\n",
                    "print('Hand tensor placeholder ready')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Interactive Gesture Control"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create interactive widget\\n",
                    "@widgets.interact(\\n",
                    "    gesture=widgets.Dropdown(\\n",
                    "        options=['pointing', 'fist', 'peace', 'ok', 'rock'],\\n",
                    "        value='pointing'\\n",
                    "    )\\n",
                    ")\\n",
                    "def visualize_gesture(gesture):\\n",
                    "    hand = HandTensorCUDA(device)\\n",
                    "    weights = hand.create_gesture(gesture)\\n",
                    "    \\n",
                    "    fig = go.Figure(data=[\\n",
                    "        go.Bar(\\n",
                    "            x=['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'],\\n",
                    "            y=weights.cpu().numpy()\\n",
                    "        )\\n",
                    "    ])\\n",
                    "    fig.update_layout(title=f'Gesture: {gesture}', height=400)\\n",
                    "    fig.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Next Steps for AI Developers\\n",
                    "\\n",
                    "### TODO:\\n",
                    "1. Port full HandTensor from hand.py\\n",
                    "2. Implement 3D hand visualization\\n",
                    "3. Add real-time gesture recognition\\n",
                    "4. Create haptic feedback simulation\\n",
                    "5. Build gesture composition interface"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "deepnote_notebook_id": "motor-hand-visualization"
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Manifold - Visualization Playground Notebook
    manifold_notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🔮 Manifold Visualization Playground\\n",
                    "## Interactive 48-Manifold Explorer\\n",
                    "\\n",
                    "Explore the mathematical beauty of the 48-manifold through interactive visualizations."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Setup\\n",
                    "import sys\\n",
                    "sys.path.append('/work')\\n",
                    "\\n",
                    "from deepnote.cuda_devices import get_device\\n",
                    "import torch\\n",
                    "import numpy as np\\n",
                    "import plotly.graph_objects as go\\n",
                    "from plotly.subplots import make_subplots\\n",
                    "import ipywidgets as widgets\\n",
                    "\\n",
                    "device = get_device()\\n",
                    "print(f'🔮 Manifold playground initialized on {device}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Fractal Navigator"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "class Fractal48Navigator:\\n",
                    "    '''Interactive 48-manifold fractal explorer.'''\\n",
                    "    \\n",
                    "    def __init__(self, device=None):\\n",
                    "        self.device = device or get_device()\\n",
                    "        self.dims = 48\\n",
                    "        \\n",
                    "    def generate_fractal(self, depth=5, zoom=1.0):\\n",
                    "        '''Generate fractal pattern based on 48-factorization.'''\\n",
                    "        # Create base pattern\\n",
                    "        size = 48 * 4  # 192x192 grid\\n",
                    "        pattern = torch.zeros(size, size, device=self.device)\\n",
                    "        \\n",
                    "        # Apply factorization patterns\\n",
                    "        for level in range(depth):\\n",
                    "            scale = 3 ** level\\n",
                    "            if scale < size:\\n",
                    "                noise = torch.randn(size//scale, size//scale, device=self.device)\\n",
                    "                noise = torch.nn.functional.interpolate(\\n",
                    "                    noise.unsqueeze(0).unsqueeze(0),\\n",
                    "                    size=(size, size),\\n",
                    "                    mode='nearest'\\n",
                    "                ).squeeze()\\n",
                    "                pattern += noise / (level + 1)\\n",
                    "        \\n",
                    "        return pattern * zoom\\n",
                    "\\n",
                    "navigator = Fractal48Navigator(device)\\n",
                    "print('Fractal navigator ready')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Interactive Parameter Explorer"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "@widgets.interact(\\n",
                    "    depth=widgets.IntSlider(min=1, max=8, value=5),\\n",
                    "    zoom=widgets.FloatSlider(min=0.1, max=10, value=1.0),\\n",
                    "    colorscale=widgets.Dropdown(\\n",
                    "        options=['Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis'],\\n",
                    "        value='Viridis'\\n",
                    "    )\\n",
                    ")\\n",
                    "def explore_manifold(depth, zoom, colorscale):\\n",
                    "    '''Interactive manifold exploration.'''\\n",
                    "    pattern = navigator.generate_fractal(depth, zoom)\\n",
                    "    \\n",
                    "    fig = go.Figure(data=go.Heatmap(\\n",
                    "        z=pattern.cpu().numpy(),\\n",
                    "        colorscale=colorscale\\n",
                    "    ))\\n",
                    "    \\n",
                    "    fig.update_layout(\\n",
                    "        title=f'48-Manifold Fractal (Depth={depth}, Zoom={zoom:.1f})',\\n",
                    "        width=600,\\n",
                    "        height=600\\n",
                    "    )\\n",
                    "    fig.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 3D Manifold Visualization"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def create_3d_manifold():\\n",
                    "    '''Create 3D visualization of the 48-manifold.'''\\n",
                    "    # Generate 3D points on manifold\\n",
                    "    n_points = 48 * 48\\n",
                    "    theta = torch.linspace(0, 2*np.pi, n_points, device=device)\\n",
                    "    phi = torch.linspace(0, np.pi, n_points, device=device)\\n",
                    "    \\n",
                    "    # Create torus-like structure\\n",
                    "    R, r = 3, 1  # Major and minor radius\\n",
                    "    x = (R + r * torch.cos(phi)) * torch.cos(theta)\\n",
                    "    y = (R + r * torch.cos(phi)) * torch.sin(theta)\\n",
                    "    z = r * torch.sin(phi)\\n",
                    "    \\n",
                    "    # Add 48-based modulation\\n",
                    "    modulation = torch.sin(theta * 48) * 0.2\\n",
                    "    x += modulation\\n",
                    "    y += modulation\\n",
                    "    \\n",
                    "    fig = go.Figure(data=[go.Scatter3d(\\n",
                    "        x=x.cpu().numpy(),\\n",
                    "        y=y.cpu().numpy(),\\n",
                    "        z=z.cpu().numpy(),\\n",
                    "        mode='markers',\\n",
                    "        marker=dict(\\n",
                    "            size=2,\\n",
                    "            color=z.cpu().numpy(),\\n",
                    "            colorscale='Viridis',\\n",
                    "        )\\n",
                    "    )])\\n",
                    "    \\n",
                    "    fig.update_layout(\\n",
                    "        title='48-Manifold in 3D',\\n",
                    "        scene=dict(\\n",
                    "            xaxis_title='X',\\n",
                    "            yaxis_title='Y',\\n",
                    "            zaxis_title='Z'\\n",
                    "        ),\\n",
                    "        height=600\\n",
                    "    )\\n",
                    "    fig.show()\\n",
                    "\\n",
                    "create_3d_manifold()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Next Steps for AI Developers\\n",
                    "\\n",
                    "### TODO:\\n",
                    "1. Implement WebGL export for web visualization\\n",
                    "2. Add real-time parameter animation\\n",
                    "3. Create preset gallery\\n",
                    "4. Build fractal zoom navigation\\n",
                    "5. Add VR/AR visualization support"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "deepnote_notebook_id": "manifold-visualization-playground"
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Save notebooks
    motor_path = Path("notebooks/3_Motor_Hand.ipynb")
    manifold_path = Path("notebooks/4_Manifold_Playground.ipynb")
    
    with open(motor_path, 'w') as f:
        json.dump(motor_notebook, f, indent=2)
    
    with open(manifold_path, 'w') as f:
        json.dump(manifold_notebook, f, indent=2)
    
    print("✅ Created Motor and Manifold notebooks")
    return motor_path, manifold_path

def create_deepnote_zip():
    """Create a ZIP file with all Deepnote files for easy upload."""
    
    # Create remaining notebooks
    create_remaining_notebooks()
    
    # Create ZIP file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"deepnote_48manifold_{timestamp}.zip"
    
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        # Add all Python files
        for py_file in Path('.').glob('*.py'):
            if not py_file.name.startswith('.'):
                zipf.write(py_file, py_file.name)
        
        # Add all notebooks
        notebooks_dir = Path('notebooks')
        if notebooks_dir.exists():
            for notebook in notebooks_dir.glob('*.ipynb'):
                zipf.write(notebook, f'notebooks/{notebook.name}')
        
        # Add other files
        other_files = ['requirements.txt', 'setup_deepnote.sh', 
                       'AI_DEVELOPER_GUIDE.md', 'DEEPNOTE_SETUP.md']
        for file in other_files:
            if Path(file).exists():
                zipf.write(file, file)
    
    print(f"\n✅ Created ZIP file: {zip_name}")
    print(f"   Size: {Path(zip_name).stat().st_size / 1024:.1f} KB")
    
    return zip_name

def main():
    print("="*60)
    print("📦 PREPARING DEEPNOTE PACKAGE")
    print("="*60)
    
    zip_file = create_deepnote_zip()
    
    print("\n" + "="*60)
    print("✅ PACKAGE READY!")
    print("="*60)
    print("\n📋 Upload Instructions:")
    print("1. Go to https://deepnote.com")
    print("2. Create new project '48-Manifold'")
    print("3. Enable GPU (Settings → Hardware → GPU)")
    print(f"4. Upload {zip_file}")
    print("5. Extract and run setup_deepnote.sh")
    print("\n🚀 Ready for deployment!")

if __name__ == "__main__":
    main()