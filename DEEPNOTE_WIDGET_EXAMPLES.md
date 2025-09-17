# Deepnote Widget & Cell Embedding Guide
## Creating Clean Consumer-Facing Interfaces

### Overview
Deepnote offers multiple ways to create professional, widget-only interfaces that hide code complexity while maintaining full interactivity. This guide shows how to embed individual cells, create consumer-facing apps, and build clean interfaces for the 48-Manifold project.

---

## 1. Cell-Level Embedding

### Embedding Individual Output Cells
```html
<!-- Embed ONLY the output of a specific cell -->
<iframe
  src="https://deepnote.com/embed/48-manifold/Protein-Composer-uuid?cellId=3d4f5g6h&showCode=false"
  height="400"
  width="100%"
  frameborder="0">
</iframe>
```

### Multiple Cells as a Dashboard
```html
<!-- Create a dashboard by embedding multiple output cells -->
<div class="dashboard-grid">
  <!-- Input Widget Cell -->
  <div class="widget-container">
    <iframe
      src="https://deepnote.com/embed/48-manifold/notebook-uuid?cellId=input-widget&showCode=false"
      height="200"
      width="100%">
    </iframe>
  </div>
  
  <!-- Visualization Cell -->
  <div class="viz-container">
    <iframe
      src="https://deepnote.com/embed/48-manifold/notebook-uuid?cellId=3d-structure&showCode=false"
      height="500"
      width="100%">
    </iframe>
  </div>
  
  <!-- Metrics Cell -->
  <div class="metrics-container">
    <iframe
      src="https://deepnote.com/embed/48-manifold/notebook-uuid?cellId=quality-metrics&showCode=false"
      height="150"
      width="100%">
    </iframe>
  </div>
</div>
```

---

## 2. Deepnote App Mode (Clean Consumer Interface)

### Configure Notebook for App Mode
```python
# In the notebook's first cell (hidden in app mode)
import deepnote

# Configure app settings
deepnote.app.configure({
    "hide_code": True,              # Hide all code cells
    "hide_markdown": False,          # Keep markdown for instructions
    "layout": "single-column",       # or "dashboard"
    "theme": "light",
    "width": "full",
    "allow_interactivity": True,
    "show_cell_numbers": False,
    "show_cell_borders": False,
    "custom_css": """
        .cell-output { 
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .widget-container {
            max-width: 800px;
            margin: 0 auto;
        }
    """
})
```

### Create Widget-Only Interface
```python
# Cell 1: Configuration (hidden)
from ipywidgets import VBox, HBox, Button, Text, FloatSlider, Output, HTML
import plotly.graph_objects as go
from IPython.display import display, clear_output
import deepnote

# Cell 2: Title (shown as markdown)
# # 🧬 48-Manifold Protein Composer
# Transform amino acid sequences into 3D structures using harmonic composition

# Cell 3: Input Widgets (shown, code hidden)
@deepnote.app.widget_only
def create_input_panel():
    """This docstring and code will be hidden, only widgets shown"""
    
    # Create styled widgets
    sequence_input = Text(
        value='MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG',
        placeholder='Enter amino acid sequence (min 48 residues)',
        description='',
        layout={'width': '100%', 'height': '80px'}
    )
    
    # Style with custom HTML label
    sequence_label = HTML("""
        <h3 style='color: #1a237e; margin-bottom: 10px;'>
            Amino Acid Sequence
        </h3>
    """)
    
    samples_slider = FloatSlider(
        value=8,
        min=1,
        max=32,
        step=1,
        description='Ensemble Samples',
        style={'description_width': '120px'},
        layout={'width': '100%'}
    )
    
    variability_slider = FloatSlider(
        value=0.5,
        min=0,
        max=1,
        step=0.1,
        description='Variability',
        style={'description_width': '120px'},
        layout={'width': '100%'}
    )
    
    generate_btn = Button(
        description='Generate Structure',
        button_style='primary',
        icon='check',
        layout={'width': '200px', 'height': '40px'}
    )
    
    # Layout
    input_panel = VBox([
        sequence_label,
        sequence_input,
        HBox([samples_slider, variability_slider]),
        generate_btn
    ], layout={'padding': '20px', 'background': '#f5f5f5', 'border-radius': '8px'})
    
    return input_panel, generate_btn, sequence_input, samples_slider, variability_slider

# Display only the widgets
panel, btn, seq, samples, var = create_input_panel()
display(panel)

# Cell 4: Output Area (shown, code hidden)
@deepnote.app.widget_only
def create_output_area():
    output = Output(layout={
        'border': '1px solid #ddd',
        'padding': '20px',
        'border-radius': '8px',
        'min-height': '400px',
        'background': 'white'
    })
    return output

output_area = create_output_area()
display(output_area)

# Cell 5: Processing Logic (hidden completely)
@deepnote.app.hidden
def process_sequence(b):
    with output_area:
        clear_output()
        
        # Show loading state
        display(HTML("""
            <div style='text-align: center; padding: 40px;'>
                <div class='spinner'></div>
                <h3 style='color: #666;'>Generating structure...</h3>
            </div>
            <style>
                .spinner {
                    width: 40px;
                    height: 40px;
                    margin: 0 auto;
                    border: 4px solid #f3f3f3;
                    border-top: 4px solid #1a237e;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        """))
        
        # Process (actual computation would go here)
        import time
        time.sleep(2)
        
        clear_output()
        
        # Display results in a clean format
        display(HTML("""
            <div style='display: grid; grid-template-columns: 2fr 1fr; gap: 20px;'>
                <div>
                    <h3 style='color: #1a237e;'>Generated Structure</h3>
                    <div id='structure-viz'></div>
                </div>
                <div>
                    <h3 style='color: #1a237e;'>Quality Metrics</h3>
                    <div style='background: #f8f9fa; padding: 15px; border-radius: 8px;'>
                        <div style='margin-bottom: 10px;'>
                            <strong>Clashes:</strong> 
                            <span style='color: green; float: right;'>0</span>
                        </div>
                        <div style='margin-bottom: 10px;'>
                            <strong>Min CA-CA:</strong> 
                            <span style='color: green; float: right;'>3.54 Å</span>
                        </div>
                        <div style='margin-bottom: 10px;'>
                            <strong>Certainty:</strong> 
                            <span style='color: blue; float: right;'>0.816</span>
                        </div>
                        <div>
                            <strong>Harmony:</strong> 
                            <span style='color: blue; float: right;'>0.923</span>
                        </div>
                    </div>
                </div>
            </div>
        """))
        
        # Add 3D structure visualization
        fig = go.Figure(data=[go.Scatter3d(
            x=list(range(50)),
            y=[i**2 % 20 - 10 for i in range(50)],
            z=[i**3 % 30 - 15 for i in range(50)],
            mode='markers+lines',
            marker=dict(size=5, color=list(range(50)), colorscale='Viridis'),
            line=dict(color='darkblue', width=2)
        )])
        
        fig.update_layout(
            height=400,
            showlegend=False,
            scene=dict(
                xaxis_title="X (Å)",
                yaxis_title="Y (Å)", 
                zaxis_title="Z (Å)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        fig.show()

btn.on_click(process_sequence)
```

---

## 3. Embedding Strategies for GitHub Pages

### Strategy 1: Full App Embed (Simplest)
```html
<!-- Embed entire notebook in app mode -->
<iframe
  src="https://deepnote.com/app/48-manifold/Protein-Composer-uuid"
  height="800"
  width="100%"
  frameborder="0">
</iframe>
```

### Strategy 2: Widget Dashboard (Most Flexible)
```html
<!DOCTYPE html>
<html>
<head>
    <style>
        .widget-dashboard {
            display: grid;
            grid-template-areas:
                "header header"
                "input viz"
                "metrics audio";
            grid-template-columns: 1fr 2fr;
            gap: 20px;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .widget-frame {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header { grid-area: header; }
        .input { grid-area: input; }
        .viz { grid-area: viz; }
        .metrics { grid-area: metrics; }
        .audio { grid-area: audio; }
        
        /* Hide Deepnote branding if needed */
        .widget-frame iframe {
            margin-top: -40px;
            height: calc(100% + 40px);
        }
    </style>
</head>
<body>
    <div class="widget-dashboard">
        <div class="widget-frame header">
            <h1>48-Manifold Protein Composer</h1>
            <p>Interactive structure generation using harmonic composition</p>
        </div>
        
        <div class="widget-frame input">
            <iframe
                src="https://deepnote.com/embed/48-manifold/notebook?cellId=input-widgets&hideCode=true"
                height="300"
                width="100%"
                frameborder="0">
            </iframe>
        </div>
        
        <div class="widget-frame viz">
            <iframe
                src="https://deepnote.com/embed/48-manifold/notebook?cellId=structure-viz&hideCode=true"
                height="400"
                width="100%"
                frameborder="0">
            </iframe>
        </div>
        
        <div class="widget-frame metrics">
            <iframe
                src="https://deepnote.com/embed/48-manifold/notebook?cellId=metrics&hideCode=true"
                height="200"
                width="100%"
                frameborder="0">
            </iframe>
        </div>
        
        <div class="widget-frame audio">
            <iframe
                src="https://deepnote.com/embed/48-manifold/notebook?cellId=sonification&hideCode=true"
                height="200"
                width="100%"
                frameborder="0">
            </iframe>
        </div>
    </div>
</body>
</html>
```

### Strategy 3: Progressive Disclosure
```javascript
// Show different levels of detail based on user expertise
function setInterfaceLevel(level) {
    const frames = {
        basic: "https://deepnote.com/app/48-manifold/basic-uuid",
        advanced: "https://deepnote.com/app/48-manifold/advanced-uuid",
        research: "https://deepnote.com/app/48-manifold/research-uuid"
    };
    
    document.getElementById('notebook-frame').src = frames[level];
    
    // Update UI indicators
    document.querySelectorAll('.level-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.getElementById(`btn-${level}`).classList.add('active');
}
```

```html
<div class="interface-selector">
    <button id="btn-basic" class="level-btn active" onclick="setInterfaceLevel('basic')">
        👶 Basic
    </button>
    <button id="btn-advanced" class="level-btn" onclick="setInterfaceLevel('advanced')">
        🧑‍🔬 Advanced
    </button>
    <button id="btn-research" class="level-btn" onclick="setInterfaceLevel('research')">
        🔬 Research
    </button>
</div>

<iframe id="notebook-frame" src="https://deepnote.com/app/48-manifold/basic-uuid"></iframe>
```

---

## 4. Advanced Widget Examples

### Interactive Parameter Explorer
```python
# Create a parameter exploration interface with live updates
@deepnote.app.reactive_widget
def parameter_explorer():
    from ipywidgets import interactive, FloatSlider, IntSlider, Dropdown
    import numpy as np
    import plotly.express as px
    
    def update_manifold(dimension, factorization, parity, routing):
        # Hide the actual computation
        data = np.random.randn(100, 3) * dimension
        
        # Show only the visualization
        fig = px.scatter_3d(
            x=data[:, 0], 
            y=data[:, 1], 
            z=data[:, 2],
            color=np.arange(100),
            title=f"48-Manifold: {factorization} | {parity} | {routing}"
        )
        fig.update_layout(showlegend=False, height=500)
        fig.show()
        
        # Display metrics in a clean card
        display(HTML(f"""
            <div style='display: flex; gap: 20px; margin-top: 20px;'>
                <div style='flex: 1; padding: 15px; background: #e3f2fd; border-radius: 8px;'>
                    <h4 style='margin: 0; color: #1565c0;'>Dimension</h4>
                    <p style='font-size: 24px; margin: 5px 0;'>{dimension}</p>
                </div>
                <div style='flex: 1; padding: 15px; background: #f3e5f5; border-radius: 8px;'>
                    <h4 style='margin: 0; color: #7b1fa2;'>Complexity</h4>
                    <p style='font-size: 24px; margin: 5px 0;'>{dimension * 2}</p>
                </div>
                <div style='flex: 1; padding: 15px; background: #e8f5e9; border-radius: 8px;'>
                    <h4 style='margin: 0; color: #2e7d32;'>Reversibility</h4>
                    <p style='font-size: 24px; margin: 5px 0;'>100%</p>
                </div>
            </div>
        """))
    
    # Create interactive widget with custom styling
    w = interactive(
        update_manifold,
        dimension=IntSlider(min=12, max=96, step=12, value=48, description='Base Dimension'),
        factorization=Dropdown(options=['2^4 × 3', '2^3 × 6', '4 × 12'], value='2^4 × 3', description='Factorization'),
        parity=Dropdown(options=['keven', 'kodd', 'both'], value='both', description='Parity Mode'),
        routing=Dropdown(options=['W (possibility)', 'M (manifestation)'], value='W (possibility)', description='Routing')
    )
    
    # Hide the control layout and show custom UI
    w.layout.visibility = 'hidden'
    
    # Create custom control panel
    display(HTML("""
        <style>
            .widget-label { display: none !important; }
            .widget-readout { font-weight: bold; color: #1a237e; }
            .widget-slider { width: 100% !important; }
        </style>
    """))
    
    return w

explorer = parameter_explorer()
display(explorer)
```

### Multi-Tab Interface
```python
# Create a tabbed interface with different views
from ipywidgets import Tab, VBox, HTML

tab = Tab()

# Tab 1: Simple Interface
simple_view = VBox([
    HTML("<h3>Quick Start</h3>"),
    # Minimal controls
])

# Tab 2: Advanced Interface
advanced_view = VBox([
    HTML("<h3>Advanced Options</h3>"),
    # Full controls
])

# Tab 3: Research Interface
research_view = VBox([
    HTML("<h3>Research Tools</h3>"),
    # Complete toolkit
])

tab.children = [simple_view, advanced_view, research_view]
tab.set_title(0, '🎯 Simple')
tab.set_title(1, '⚙️ Advanced')
tab.set_title(2, '🔬 Research')

display(tab)
```

---

## 5. Deepnote API for Dynamic Embedding

### Generate Embeds Programmatically
```python
import requests

# Deepnote API endpoint
api_url = "https://api.deepnote.com/v1/projects/{project_id}/embed"

# Create custom embed configuration
embed_config = {
    "notebook_id": "protein-composer-uuid",
    "cells": ["input-widget", "output-viz"],  # Only these cells
    "options": {
        "hide_code": True,
        "hide_markdown": False,
        "theme": "light",
        "height": 600,
        "auto_run": True,
        "cache_outputs": True
    }
}

# Get embed URL
response = requests.post(
    api_url,
    json=embed_config,
    headers={"Authorization": f"Bearer {api_token}"}
)

embed_url = response.json()["embed_url"]
```

### Dynamic Widget Loading
```javascript
// Load different widgets based on user interaction
class DeepnoteWidgetLoader {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.baseUrl = "https://deepnote.com/embed/48-manifold/";
    }
    
    loadWidget(widgetName, options = {}) {
        const defaultOptions = {
            hideCode: true,
            height: 400,
            theme: 'light'
        };
        
        const settings = {...defaultOptions, ...options};
        const params = new URLSearchParams(settings).toString();
        
        const iframe = document.createElement('iframe');
        iframe.src = `${this.baseUrl}${widgetName}?${params}`;
        iframe.width = '100%';
        iframe.height = settings.height;
        iframe.frameborder = '0';
        
        this.container.innerHTML = '';
        this.container.appendChild(iframe);
    }
    
    // Load specific cell output only
    loadCell(notebookId, cellId, options = {}) {
        const url = `${this.baseUrl}${notebookId}?cellId=${cellId}&hideCode=true`;
        this.loadFromUrl(url, options);
    }
}

// Usage
const loader = new DeepnoteWidgetLoader('widget-container');

// Load different widgets based on user selection
document.getElementById('protein-btn').onclick = () => {
    loader.loadWidget('protein-composer', {height: 600});
};

document.getElementById('fractal-btn').onclick = () => {
    loader.loadWidget('fractal-navigator', {height: 500});
};
```

---

## 6. Best Practices for Consumer-Facing Interfaces

### 1. **Hide Complexity**
```python
# Use Deepnote's @app decorators
@deepnote.app.hidden  # Completely hidden
@deepnote.app.widget_only  # Show only output
@deepnote.app.collapsible  # User can expand if curious
```

### 2. **Responsive Design**
```python
# Detect screen size and adjust
@deepnote.app.responsive
def adaptive_interface():
    if deepnote.screen.width < 768:
        return mobile_layout()
    else:
        return desktop_layout()
```

### 3. **Loading States**
```python
# Show progress for long computations
with deepnote.progress_bar(total=100) as progress:
    for i in range(100):
        progress.update(1)
        progress.set_description(f"Processing: {i}%")
```

### 4. **Error Handling**
```python
# User-friendly error messages
@deepnote.app.error_handler
def safe_computation():
    try:
        result = complex_calculation()
    except Exception as e:
        display(HTML(f"""
            <div style='background: #ffebee; padding: 20px; border-radius: 8px;'>
                <h3 style='color: #c62828;'>⚠️ Unable to process</h3>
                <p>Please check your input and try again.</p>
                <details>
                    <summary>Technical details</summary>
                    <code>{str(e)}</code>
                </details>
            </div>
        """))
```

---

## Conclusion

Deepnote's flexibility allows you to create everything from simple embedded widgets to complex multi-panel dashboards, all while hiding the underlying code complexity. The key advantages for consumer-facing interfaces are:

1. **Granular Control** - Embed individual cells, entire notebooks, or custom apps
2. **Clean Presentation** - Hide code, show only outputs and widgets
3. **Responsive Design** - Adapts to different screen sizes and contexts
4. **Progressive Disclosure** - Different interfaces for different user levels
5. **Performance** - Outputs can be cached and served quickly
6. **Interactivity** - Full widget support without showing implementation

This makes Deepnote perfect for creating professional, accessible interfaces for the 48-Manifold project that work seamlessly within your GitHub Pages site.