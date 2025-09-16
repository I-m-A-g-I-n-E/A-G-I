"""
Core fractal visualization module for the 48-manifold system.
Visualizes the unfolding fractal structure through decomposition layers.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import matplotlib.animation as animation

from agi.harmonia.laws import Laws
from agi.harmonia.notation import Turn, Handedness, Phase48


@dataclass
class FractalState:
    """Represents a state in the fractal decomposition."""
    level: int  # 0=48, 1=16, 2=8, 3=4, 4=2
    tensor: torch.Tensor
    complexity: float
    phase: Phase48
    factorization: str
    
    @property
    def dimension(self) -> int:
        """Current dimension at this level."""
        factors = [48, 16, 8, 4, 2]
        return factors[min(self.level, len(factors)-1)]


class FractalVisualizer:
    """
    Visualizes the 48-manifold fractal structure and its unfolding.
    Shows how information flows through the factorization ladder.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        self.ladder = (3, 2, 2, 2)  # Factorization ladder: 48 = 3 × 2^4
        self.levels = [48, 16, 8, 4, 2]
        
    def visualize_factorization_ladder(self, 
                                      initial_state: torch.Tensor,
                                      show_complexity: bool = True) -> plt.Figure:
        """
        Visualize the complete factorization ladder from 48 down to 2.
        Shows how the manifold decomposes through dyadic and triadic steps.
        """
        fig = plt.figure(figsize=self.figsize)
        
        # Create subplots for each level
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        states = self._compute_factorization_states(initial_state)
        
        # Plot each level
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        
        for idx, (pos, state) in enumerate(zip(positions, states)):
            ax = fig.add_subplot(gs[pos[0], pos[1]], projection='3d')
            self._plot_level_3d(ax, state, show_complexity)
            
        # Add overall title
        fig.suptitle('48-Manifold Factorization Ladder', fontsize=16, fontweight='bold')
        
        # Add complexity legend if enabled
        if show_complexity:
            self._add_complexity_colorbar(fig)
            
        return fig
    
    def visualize_fractal_unfolding(self,
                                   composition: torch.Tensor,
                                   torsions: Optional[torch.Tensor] = None) -> plt.Figure:
        """
        Visualize how a composition vector unfolds into fractal structure.
        Shows the hierarchical decomposition and recomposition.
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # 1. Original 48D composition as heatmap
        ax = axes[0]
        if composition.dim() == 1:
            comp_2d = composition.view(6, 8)
        else:
            comp_2d = composition[:48].view(6, 8)
        
        im = ax.imshow(comp_2d.cpu().numpy(), cmap='viridis', aspect='auto')
        ax.set_title('48D Composition (6×8 view)')
        ax.set_xlabel('Dyadic dimension')
        ax.set_ylabel('Hexadic dimension')
        plt.colorbar(im, ax=ax)
        
        # 2. Dyadic/Triadic decomposition
        ax = axes[1]
        self._plot_dyadic_triadic_split(ax, composition)
        
        # 3. Complexity landscape
        ax = axes[2]
        self._plot_complexity_landscape(ax, composition)
        
        # 4. Phase space if torsions provided
        ax = axes[3]
        if torsions is not None:
            self._plot_phase_space(ax, torsions)
        else:
            self._plot_fractal_tree(ax, composition)
            
        fig.suptitle('Fractal Unfolding of Composition', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def animate_fractal_evolution(self,
                                 trajectory: List[torch.Tensor],
                                 save_path: Optional[str] = None) -> animation.FuncAnimation:
        """
        Animate the evolution of fractal states over time.
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            state = FractalState(
                level=0,
                tensor=trajectory[frame],
                complexity=self._compute_complexity(trajectory[frame]),
                phase=self._tensor_to_phase(trajectory[frame]),
                factorization="48"
            )
            self._plot_level_3d(ax, state, show_complexity=True)
            ax.set_title(f'Frame {frame}/{len(trajectory)}')
            
        anim = animation.FuncAnimation(fig, update, frames=len(trajectory), 
                                     interval=50, blit=False)
        
        if save_path:
            anim.save(save_path, writer='pillow')
            
        return anim
    
    def _compute_factorization_states(self, tensor: torch.Tensor) -> List[FractalState]:
        """Compute states at each level of factorization."""
        states = []
        current = tensor.clone()
        
        # Level 0: Original 48
        states.append(FractalState(
            level=0,
            tensor=current,
            complexity=self._compute_complexity(current),
            phase=self._tensor_to_phase(current),
            factorization="48 = 2^4 × 3"
        ))
        
        # Level 1: 48 → 16 (factor out 3)
        if current.numel() >= 48:
            current = self._factor_triadic(current)
            states.append(FractalState(
                level=1,
                tensor=current,
                complexity=self._compute_complexity(current),
                phase=self._tensor_to_phase(current),
                factorization="16 = 2^4"
            ))
        
        # Level 2: 16 → 8 (factor out 2)
        if current.numel() >= 16:
            current = self._factor_dyadic(current)
            states.append(FractalState(
                level=2,
                tensor=current,
                complexity=self._compute_complexity(current),
                phase=self._tensor_to_phase(current),
                factorization="8 = 2^3"
            ))
        
        # Level 3: 8 → 4 (factor out 2)
        if current.numel() >= 8:
            current = self._factor_dyadic(current)
            states.append(FractalState(
                level=3,
                tensor=current,
                complexity=self._compute_complexity(current),
                phase=self._tensor_to_phase(current),
                factorization="4 = 2^2"
            ))
        
        # Level 4: 4 → 2 (factor out 2)
        if current.numel() >= 4:
            current = self._factor_dyadic(current)
            states.append(FractalState(
                level=4,
                tensor=current,
                complexity=self._compute_complexity(current),
                phase=self._tensor_to_phase(current),
                factorization="2 = 2^1"
            ))
            
        return states
    
    def _plot_level_3d(self, ax, state: FractalState, show_complexity: bool):
        """Plot a single factorization level in 3D."""
        data = state.tensor.cpu().numpy()
        
        # Reshape for 3D visualization
        if data.ndim == 1:
            # Create a grid based on factorization level
            grid_size = int(np.sqrt(data.size))
            if grid_size * grid_size < data.size:
                grid_size += 1
            padded = np.zeros(grid_size * grid_size)
            padded[:data.size] = data.flatten()
            data = padded.reshape(grid_size, grid_size)
        
        # Create mesh
        x = np.arange(data.shape[0])
        y = np.arange(data.shape[1])
        X, Y = np.meshgrid(x, y)
        
        # Color based on complexity if requested
        if show_complexity:
            colors = self._complexity_to_color(data, state.complexity)
            surf = ax.plot_surface(X, Y, data.T, facecolors=colors, 
                                  antialiased=True, alpha=0.8)
        else:
            surf = ax.plot_surface(X, Y, data.T, cmap='viridis',
                                  antialiased=True, alpha=0.8)
        
        ax.set_title(f'Level {state.level}: {state.factorization}\n'
                    f'Complexity: {state.complexity:.3f}')
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.set_zlabel('Value')
        
    def _plot_dyadic_triadic_split(self, ax, composition: torch.Tensor):
        """Visualize the dyadic/triadic decomposition."""
        comp_np = composition.cpu().numpy()
        
        # Compute dyadic and triadic components
        dyadic_mask = np.array([i % 3 != 0 for i in range(len(comp_np))])
        triadic_mask = ~dyadic_mask
        
        dyadic_component = comp_np * dyadic_mask
        triadic_component = comp_np * triadic_mask
        
        x = np.arange(len(comp_np))
        width = 0.35
        
        ax.bar(x - width/2, dyadic_component, width, label='Dyadic (2^n)', 
               color='blue', alpha=0.7)
        ax.bar(x + width/2, triadic_component, width, label='Triadic (3×)', 
               color='red', alpha=0.7)
        
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Value')
        ax.set_title('Dyadic/Triadic Decomposition')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_complexity_landscape(self, ax, composition: torch.Tensor):
        """Plot the complexity landscape of the composition."""
        comp_np = composition.cpu().numpy()
        
        # Compute local complexity for each element
        complexities = []
        for i, val in enumerate(comp_np):
            # Simulate Turn complexity calculation
            turn = Turn(val / 48.0)
            complexities.append(turn.fractal_complexity)
        
        ax.plot(complexities, 'o-', color='purple', alpha=0.7)
        ax.fill_between(range(len(complexities)), complexities, 
                        alpha=0.3, color='purple')
        
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Fractal Complexity')
        ax.set_title('Complexity Landscape')
        ax.grid(True, alpha=0.3)
        
        # Add complexity threshold lines
        ax.axhline(y=1.0, color='green', linestyle='--', 
                  label='Low complexity', alpha=0.5)
        ax.axhline(y=4.0, color='red', linestyle='--', 
                  label='High complexity', alpha=0.5)
        ax.legend()
        
    def _plot_phase_space(self, ax, torsions: torch.Tensor):
        """Plot the phase space of torsion angles."""
        if torsions.shape[-1] >= 2:
            phi = torsions[..., 0].cpu().numpy()
            psi = torsions[..., 1].cpu().numpy()
            
            # Create scatter plot with density coloring
            scatter = ax.scatter(phi, psi, c=np.arange(len(phi)), 
                               cmap='twilight', s=20, alpha=0.6)
            
            ax.set_xlabel('Phi (degrees)')
            ax.set_ylabel('Psi (degrees)')
            ax.set_title('Torsion Phase Space')
            ax.grid(True, alpha=0.3)
            
            # Add Ramachandran-like regions
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
            
            plt.colorbar(scatter, ax=ax, label='Position')
        else:
            ax.text(0.5, 0.5, 'Insufficient torsion data', 
                   transform=ax.transAxes, ha='center')
            ax.set_title('Phase Space (No Data)')
            
    def _plot_fractal_tree(self, ax, composition: torch.Tensor):
        """Plot the fractal tree structure of decomposition."""
        # Create tree structure showing factorization
        levels = [48, 16, 8, 4, 2, 1]
        positions = []
        connections = []
        
        for i, level in enumerate(levels):
            y = -i
            # Calculate x positions for nodes at this level
            n_nodes = min(2**i, 8)  # Cap for visualization
            x_positions = np.linspace(-n_nodes/2, n_nodes/2, n_nodes)
            
            for x in x_positions:
                positions.append((x, y))
                
                # Connect to parent nodes
                if i > 0:
                    parent_idx = len(positions) - n_nodes - 1
                    if parent_idx >= 0:
                        connections.append((parent_idx, len(positions) - 1))
        
        # Plot connections
        for start_idx, end_idx in connections:
            if start_idx < len(positions) and end_idx < len(positions):
                start = positions[start_idx]
                end = positions[end_idx]
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       'k-', alpha=0.3)
        
        # Plot nodes
        for i, (x, y) in enumerate(positions):
            level_idx = min(-int(y), len(levels) - 1)
            size = levels[level_idx] * 2
            ax.scatter(x, y, s=size, c='blue', alpha=0.7, zorder=5)
            ax.text(x, y, str(levels[level_idx]), ha='center', va='center',
                   fontsize=8, color='white', zorder=6)
        
        ax.set_xlim(-10, 10)
        ax.set_ylim(-len(levels), 1)
        ax.set_xlabel('Branch')
        ax.set_ylabel('Level')
        ax.set_title('Fractal Decomposition Tree')
        ax.grid(True, alpha=0.3)
        
    def _factor_triadic(self, tensor: torch.Tensor) -> torch.Tensor:
        """Factor out triadic component (divide by 3)."""
        if tensor.numel() % 3 == 0:
            return tensor.view(-1, 3).mean(dim=1)
        return tensor
    
    def _factor_dyadic(self, tensor: torch.Tensor) -> torch.Tensor:
        """Factor out dyadic component (divide by 2)."""
        if tensor.numel() % 2 == 0:
            return tensor.view(-1, 2).mean(dim=1)
        return tensor
    
    def _compute_complexity(self, tensor: torch.Tensor) -> float:
        """Compute fractal complexity of a tensor."""
        # Use statistical measures as proxy for complexity
        if tensor.numel() == 0:
            return 0.0
        
        t_np = tensor.cpu().numpy().flatten()
        
        # Compute various complexity metrics
        entropy = -np.sum(np.abs(t_np) * np.log(np.abs(t_np) + 1e-10))
        variance = np.var(t_np)
        
        # Normalize and combine
        complexity = np.log(1 + entropy) + np.sqrt(variance)
        
        return float(complexity)
    
    def _tensor_to_phase(self, tensor: torch.Tensor) -> Phase48:
        """Convert tensor to Phase48 representation."""
        # Use first element as representative phase
        if tensor.numel() > 0:
            val = float(tensor.flatten()[0])
            tick = int(val * 48) % 48
            micro = (val * 48) - tick
            return Phase48(tick, micro)
        return Phase48(0, 0.0)
    
    def _complexity_to_color(self, data: np.ndarray, complexity: float):
        """Map complexity to color gradient."""
        # Normalize complexity to [0, 1]
        norm_complexity = np.clip(complexity / 10.0, 0, 1)
        
        # Create color array
        cmap = cm.get_cmap('RdYlBu_r')
        colors = np.ones((*data.shape, 4))
        
        # Apply complexity-based coloring
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = abs(data[i, j])
                colors[i, j] = cmap(norm_complexity * val)
                
        return colors
    
    def _add_complexity_colorbar(self, fig):
        """Add a complexity colorbar to the figure."""
        cmap = cm.get_cmap('RdYlBu_r')
        sm = cm.ScalarMappable(cmap=cmap)
        sm.set_array([0, 10])
        
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Fractal Complexity', rotation=270, labelpad=20)