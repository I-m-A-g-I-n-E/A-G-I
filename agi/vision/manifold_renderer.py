"""
3D renderer for the 48-manifold structure.
Visualizes the manifold as a navigable 3D space.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, List

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    px = None

from agi.harmonia.laws import Laws
from manifold import RouterMode, SixAxisState


class ManifoldRenderer:
    """
    Renders the 48-manifold in various 3D representations.
    Supports both matplotlib and plotly for interactive visualization.
    """
    
    def __init__(self, use_plotly: bool = False):
        self.use_plotly = use_plotly and PLOTLY_AVAILABLE
        if use_plotly and not PLOTLY_AVAILABLE:
            print("Warning: Plotly not available, falling back to matplotlib")
        self.manifold_dim = Laws.MANIFOLD_DIM
        
    def render_six_axis_state(self, 
                             state: SixAxisState,
                             mode: RouterMode = RouterMode.W_POSSIBILITY) -> plt.Figure:
        """
        Render a SixAxisState as a 3D radar/spider plot.
        Shows the 6 semantic dimensions: who, what, when, where, why, how.
        """
        if self.use_plotly:
            return self._render_six_axis_plotly(state, mode)
        else:
            return self._render_six_axis_matplotlib(state, mode)
    
    def render_manifold_lattice(self,
                              resolution: int = 8,
                              highlight_path: Optional[torch.Tensor] = None) -> plt.Figure:
        """
        Render the 48-manifold as a 3D lattice structure.
        Shows the dyadic (2^4) and triadic (3) components.
        """
        if self.use_plotly:
            return self._render_lattice_plotly(resolution, highlight_path)
        else:
            return self._render_lattice_matplotlib(resolution, highlight_path)
    
    def render_factorization_flow(self,
                                 tensor: torch.Tensor,
                                 show_arrows: bool = True) -> plt.Figure:
        """
        Visualize the flow of information through factorization levels.
        Shows how 48 → 16 → 8 → 4 → 2 decomposition works.
        """
        fig = plt.figure(figsize=(14, 10))
        
        # Create 3D axis
        ax = fig.add_subplot(111, projection='3d')
        
        # Define levels and their positions
        levels = [
            (48, 0, 'Full Manifold (48 = 2^4 × 3)'),
            (16, -2, 'Dyadic Core (16 = 2^4)'),
            (8, -4, 'Octahedral (8 = 2^3)'),
            (4, -6, 'Quaternary (4 = 2^2)'),
            (2, -8, 'Binary (2 = 2^1)')
        ]
        
        # Plot each level as a plane
        for dim, z, label in levels:
            self._plot_level_plane(ax, dim, z, label, tensor)
        
        # Add flow arrows if requested
        if show_arrows:
            self._add_flow_arrows(ax, levels)
        
        ax.set_xlabel('X Dimension')
        ax.set_ylabel('Y Dimension')
        ax.set_zlabel('Factorization Level')
        ax.set_title('Information Flow Through Factorization', fontsize=14, fontweight='bold')
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        return fig
    
    def render_phase_torus(self,
                         phases: torch.Tensor,
                         color_by: str = 'complexity') -> plt.Figure:
        """
        Render phases on a torus to show the cyclic nature of the manifold.
        The torus represents the phase space with major radius for coarse phase
        and minor radius for fine phase.
        """
        if self.use_plotly:
            return self._render_torus_plotly(phases, color_by)
        else:
            return self._render_torus_matplotlib(phases, color_by)
    
    def _render_six_axis_matplotlib(self, state: SixAxisState, mode: RouterMode) -> plt.Figure:
        """Matplotlib implementation of six-axis rendering."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert state to numpy
        axes_data = state.to_tensor().cpu().numpy()
        
        # Define axis labels
        labels = ['Who', 'What', 'When', 'Where', 'Why', 'How']
        
        # Create vertices for each axis
        n_points = axes_data.shape[1] if axes_data.ndim > 1 else 1
        
        # Plot each axis as a vector from origin
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan']
        
        for i, (axis, label, color) in enumerate(zip(axes_data, labels, colors)):
            # Normalize axis for visualization
            if axis.ndim == 0:
                axis = np.array([axis])
            
            # Create points along this semantic axis
            t = np.linspace(0, 1, len(axis))
            x = t * np.cos(i * np.pi / 3)
            y = t * np.sin(i * np.pi / 3)
            z = axis
            
            ax.plot(x, y, z, color=color, label=label, linewidth=2, alpha=0.7)
            ax.scatter(x[-1], y[-1], z[-1], color=color, s=100, alpha=0.8)
        
        # Add mode indicator
        mode_color = 'orange' if mode == RouterMode.W_POSSIBILITY else 'purple'
        ax.text2D(0.05, 0.95, f'Mode: {mode.value}', 
                 transform=ax.transAxes, fontsize=12,
                 color=mode_color, fontweight='bold')
        
        ax.set_xlabel('Semantic X')
        ax.set_ylabel('Semantic Y')
        ax.set_zlabel('Magnitude')
        ax.set_title('Six-Axis Semantic State', fontsize=14, fontweight='bold')
        ax.legend()
        
        return fig
    
    def _render_six_axis_plotly(self, state: SixAxisState, mode: RouterMode):
        """Plotly implementation for interactive six-axis rendering."""
        if not PLOTLY_AVAILABLE:
            return self._render_six_axis_matplotlib(state, mode)
            
        axes_data = state.to_tensor().cpu().numpy()
        labels = ['Who', 'What', 'When', 'Where', 'Why', 'How']
        
        fig = go.Figure()
        
        # Create 3D scatter plot for each axis
        colors = px.colors.qualitative.Set1[:6]
        
        for i, (axis, label, color) in enumerate(zip(axes_data, labels, colors)):
            if axis.ndim == 0:
                axis = np.array([axis])
            
            t = np.linspace(0, 1, len(axis))
            x = t * np.cos(i * np.pi / 3)
            y = t * np.sin(i * np.pi / 3)
            z = axis
            
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=4),
                marker=dict(size=5)
            ))
        
        fig.update_layout(
            title=f'Six-Axis Semantic State (Mode: {mode.value})',
            scene=dict(
                xaxis_title='Semantic X',
                yaxis_title='Semantic Y',
                zaxis_title='Magnitude'
            ),
            height=800
        )
        
        return fig
    
    def _render_lattice_matplotlib(self, resolution: int, highlight_path: Optional[torch.Tensor]) -> plt.Figure:
        """Render manifold lattice with matplotlib."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create lattice points
        # Dyadic dimensions (16 points)
        dyadic_points = np.arange(16)
        # Triadic dimension (3 points)
        triadic_points = np.arange(3)
        
        # Create 3D grid
        X, Y, Z = np.meshgrid(
            dyadic_points[:4],
            dyadic_points[4:8] if len(dyadic_points) > 4 else [0],
            triadic_points
        )
        
        # Plot lattice points
        ax.scatter(X, Y, Z, c=Z, cmap='viridis', s=50, alpha=0.6)
        
        # Draw lattice connections
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    # Connect to neighbors
                    if i < X.shape[0] - 1:
                        ax.plot([X[i,j,k], X[i+1,j,k]], 
                               [Y[i,j,k], Y[i+1,j,k]], 
                               [Z[i,j,k], Z[i+1,j,k]], 
                               'k-', alpha=0.2)
                    if j < X.shape[1] - 1:
                        ax.plot([X[i,j,k], X[i,j+1,k]], 
                               [Y[i,j,k], Y[i,j+1,k]], 
                               [Z[i,j,k], Z[i,j+1,k]], 
                               'k-', alpha=0.2)
                    if k < X.shape[2] - 1:
                        ax.plot([X[i,j,k], X[i,j,k+1]], 
                               [Y[i,j,k], Y[i,j,k+1]], 
                               [Z[i,j,k], Z[i,j,k+1]], 
                               'r-', alpha=0.3)
        
        # Highlight path if provided
        if highlight_path is not None:
            path_np = highlight_path.cpu().numpy()
            if len(path_np) >= 3:
                ax.plot(path_np[:, 0], path_np[:, 1], path_np[:, 2],
                       'g-', linewidth=3, alpha=0.8, label='Highlighted Path')
        
        ax.set_xlabel('Dyadic Dim 1 (2^2)')
        ax.set_ylabel('Dyadic Dim 2 (2^2)')
        ax.set_zlabel('Triadic Dim (3)')
        ax.set_title('48-Manifold Lattice Structure\n(16 × 3 = 48)', fontsize=14, fontweight='bold')
        
        if highlight_path is not None:
            ax.legend()
        
        return fig
    
    def _render_lattice_plotly(self, resolution: int, highlight_path: Optional[torch.Tensor]):
        """Interactive lattice rendering with plotly."""
        if not PLOTLY_AVAILABLE:
            return self._render_lattice_matplotlib(resolution, highlight_path)
            
        # Similar structure but with plotly implementation
        dyadic_points = np.arange(16)
        triadic_points = np.arange(3)
        
        X, Y, Z = np.meshgrid(
            dyadic_points[:4],
            dyadic_points[4:8] if len(dyadic_points) > 4 else [0],
            triadic_points
        )
        
        fig = go.Figure(data=[go.Scatter3d(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            mode='markers',
            marker=dict(
                size=8,
                color=Z.flatten(),
                colorscale='Viridis',
                showscale=True
            )
        )])
        
        if highlight_path is not None:
            path_np = highlight_path.cpu().numpy()
            if len(path_np) >= 3:
                fig.add_trace(go.Scatter3d(
                    x=path_np[:, 0],
                    y=path_np[:, 1],
                    z=path_np[:, 2],
                    mode='lines',
                    line=dict(color='green', width=6),
                    name='Path'
                ))
        
        fig.update_layout(
            title='48-Manifold Lattice Structure',
            scene=dict(
                xaxis_title='Dyadic Dim 1',
                yaxis_title='Dyadic Dim 2',
                zaxis_title='Triadic Dim'
            ),
            height=800
        )
        
        return fig
    
    def _plot_level_plane(self, ax, dim: int, z: float, label: str, tensor: torch.Tensor):
        """Plot a single factorization level as a plane."""
        # Create grid for this dimension
        grid_size = int(np.sqrt(dim))
        if grid_size * grid_size < dim:
            grid_size += 1
        
        x = np.linspace(-grid_size/2, grid_size/2, grid_size)
        y = np.linspace(-grid_size/2, grid_size/2, grid_size)
        X, Y = np.meshgrid(x, y)
        Z = np.ones_like(X) * z
        
        # Sample tensor values for coloring
        if tensor.numel() >= dim:
            values = tensor.flatten()[:dim].cpu().numpy()
            values_grid = values[:grid_size*grid_size].reshape(grid_size, grid_size)
        else:
            values_grid = np.random.randn(grid_size, grid_size) * 0.1
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(values_grid),
                              alpha=0.6, antialiased=True)
        
        # Add label
        ax.text(0, grid_size/2 + 1, z, label, fontsize=10, ha='center')
        
    def _add_flow_arrows(self, ax, levels: List[Tuple[int, float, str]]):
        """Add arrows showing information flow between levels."""
        for i in range(len(levels) - 1):
            dim1, z1, _ = levels[i]
            dim2, z2, _ = levels[i + 1]
            
            # Draw arrow from level to level
            ax.quiver(0, 0, z1, 0, 0, z2 - z1,
                     color='red', alpha=0.5, arrow_length_ratio=0.1)
            
            # Add factorization label
            factor = dim1 // dim2 if dim2 > 0 else dim1
            ax.text(1, 0, (z1 + z2) / 2, f'÷{factor}',
                   fontsize=9, color='red')
    
    def _render_torus_matplotlib(self, phases: torch.Tensor, color_by: str) -> plt.Figure:
        """Render phase space as a torus."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Torus parameters
        R = 3  # Major radius
        r = 1  # Minor radius
        
        # Create torus mesh
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, 2 * np.pi, 50)
        U, V = np.meshgrid(u, v)
        
        X = (R + r * np.cos(V)) * np.cos(U)
        Y = (R + r * np.cos(V)) * np.sin(U)
        Z = r * np.sin(V)
        
        # Map phases to torus
        if phases.numel() > 0:
            phases_np = phases.cpu().numpy()
            
            # Color based on complexity or phase
            if color_by == 'complexity':
                from agi.harmonia.notation import Turn
                colors = np.zeros_like(U)
                for i in range(U.shape[0]):
                    for j in range(U.shape[1]):
                        phase_val = U[i, j] / (2 * np.pi)
                        turn = Turn(phase_val)
                        colors[i, j] = turn.fractal_complexity
            else:
                colors = U + V
            
            surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.twilight(colors / colors.max()),
                                  alpha=0.8, antialiased=True)
        else:
            surf = ax.plot_surface(X, Y, Z, cmap='twilight', alpha=0.8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Phase Torus (colored by {color_by})', fontsize=14, fontweight='bold')
        
        return fig
    
    def _render_torus_plotly(self, phases: torch.Tensor, color_by: str):
        """Interactive torus rendering with plotly."""
        if not PLOTLY_AVAILABLE:
            return self._render_torus_matplotlib(phases, color_by)
            
        R = 3
        r = 1
        
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, 2 * np.pi, 50)
        U, V = np.meshgrid(u, v)
        
        X = (R + r * np.cos(V)) * np.cos(U)
        Y = (R + r * np.cos(V)) * np.sin(U)
        Z = r * np.sin(V)
        
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Twilight',
            showscale=True
        )])
        
        fig.update_layout(
            title=f'Phase Torus (colored by {color_by})',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            height=800
        )
        
        return fig