"""
Maps and visualizes fractal complexity across different representations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Optional, List, Tuple, Dict

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None

from agi.harmonia.notation import Turn, Handedness, Movement, Gesture
from agi.harmonia.laws import Laws


class ComplexityMapper:
    """
    Maps fractal complexity across tensors, movements, and trajectories.
    Provides visualization tools for understanding complexity distributions.
    """
    
    def __init__(self):
        self.complexity_cache = {}
        
    def compute_tensor_complexity(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute fractal complexity for each element in a tensor.
        Returns a tensor of the same shape with complexity values.
        """
        complexities = torch.zeros_like(tensor)
        
        # Flatten for processing
        flat_tensor = tensor.flatten()
        flat_complex = complexities.flatten()
        
        for i, val in enumerate(flat_tensor):
            # Convert to Turn and compute complexity
            turn_val = float(val) / Laws.MANIFOLD_DIM
            turn = Turn(turn_val)
            flat_complex[i] = turn.fractal_complexity
        
        return flat_complex.reshape(tensor.shape)
    
    def compute_movement_complexity(self, movement: Movement) -> float:
        """
        Compute the total complexity of a movement.
        Accounts for gesture, handedness, and torsion angles.
        """
        phi_turn, psi_turn = movement.get_torsions()
        
        # Sum complexities of both torsion angles
        total_complexity = phi_turn.fractal_complexity + psi_turn.fractal_complexity
        
        # Add penalty for certain gestures
        gesture_penalties = {
            Gesture.LOOP_RESOLUTION: 2.0,  # Resolution moves are complex
            Gesture.HELIX_P5: 0.5,        # Helices are natural
            Gesture.SHEET_CENTER: 1.0      # Sheets are neutral
        }
        
        if movement.gesture in gesture_penalties:
            total_complexity *= gesture_penalties[movement.gesture]
            
        return total_complexity
    
    def visualize_complexity_heatmap(self,
                                    tensor: torch.Tensor,
                                    title: str = "Fractal Complexity Heatmap") -> plt.Figure:
        """
        Create a heatmap visualization of tensor complexity.
        """
        complexities = self.compute_tensor_complexity(tensor)
        comp_np = complexities.cpu().numpy()
        
        # Reshape for 2D visualization if needed
        if comp_np.ndim == 1:
            # Try to make it square-ish
            size = int(np.sqrt(comp_np.size))
            if size * size == comp_np.size:
                comp_np = comp_np.reshape(size, size)
            else:
                # Pad to make square
                pad_size = (size + 1) ** 2 - comp_np.size
                comp_np = np.pad(comp_np, (0, pad_size), mode='constant', constant_values=0)
                comp_np = comp_np.reshape(size + 1, size + 1)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        if SEABORN_AVAILABLE:
            sns.heatmap(comp_np, annot=False, fmt='.2f', cmap='RdYlBu_r',
                       cbar_kws={'label': 'Fractal Complexity'},
                       ax=ax)
        else:
            # Fallback to matplotlib imshow
            im = ax.imshow(comp_np, cmap='RdYlBu_r', aspect='auto')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Fractal Complexity')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        
        return fig
    
    def visualize_complexity_distribution(self,
                                        tensors: List[torch.Tensor],
                                        labels: Optional[List[str]] = None) -> plt.Figure:
        """
        Visualize the distribution of complexity values across multiple tensors.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        if labels is None:
            labels = [f'Tensor {i}' for i in range(len(tensors))]
        
        all_complexities = []
        
        for i, (tensor, label) in enumerate(zip(tensors[:4], labels[:4])):
            complexities = self.compute_tensor_complexity(tensor)
            comp_flat = complexities.flatten().cpu().numpy()
            all_complexities.append(comp_flat)
            
            if i < 4:
                ax = axes[i]
                
                # Histogram
                ax.hist(comp_flat, bins=30, alpha=0.7, color=f'C{i}', edgecolor='black')
                ax.axvline(comp_flat.mean(), color='red', linestyle='--', 
                          label=f'Mean: {comp_flat.mean():.2f}')
                ax.axvline(1.0, color='green', linestyle=':', alpha=0.5,
                          label='Low complexity')
                ax.axvline(4.0, color='orange', linestyle=':', alpha=0.5,
                          label='High complexity')
                
                ax.set_xlabel('Fractal Complexity')
                ax.set_ylabel('Frequency')
                ax.set_title(label)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
        
        fig.suptitle('Complexity Distribution Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def visualize_complexity_trajectory(self,
                                      trajectory: List[torch.Tensor],
                                      window_size: int = 10) -> plt.Figure:
        """
        Visualize how complexity evolves over a trajectory.
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Compute complexities for each timestep
        mean_complexities = []
        max_complexities = []
        min_complexities = []
        
        for tensor in trajectory:
            complexities = self.compute_tensor_complexity(tensor)
            comp_flat = complexities.flatten().cpu().numpy()
            
            mean_complexities.append(np.mean(comp_flat))
            max_complexities.append(np.max(comp_flat))
            min_complexities.append(np.min(comp_flat))
        
        timesteps = np.arange(len(trajectory))
        
        # Plot 1: Mean complexity over time
        ax = axes[0]
        ax.plot(timesteps, mean_complexities, 'b-', linewidth=2, label='Mean')
        
        # Add moving average
        if len(mean_complexities) > window_size:
            moving_avg = np.convolve(mean_complexities, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            ax.plot(timesteps[window_size-1:], moving_avg, 'r--', 
                   alpha=0.7, label=f'Moving Avg (w={window_size})')
        
        ax.set_ylabel('Mean Complexity')
        ax.set_title('Mean Fractal Complexity Evolution')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 2: Complexity range (min to max)
        ax = axes[1]
        ax.fill_between(timesteps, min_complexities, max_complexities,
                       alpha=0.3, color='purple', label='Range')
        ax.plot(timesteps, mean_complexities, 'k-', linewidth=1, alpha=0.7)
        
        ax.set_ylabel('Complexity Range')
        ax.set_title('Complexity Range Evolution')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 3: Complexity variance
        ax = axes[2]
        variances = []
        for tensor in trajectory:
            complexities = self.compute_tensor_complexity(tensor)
            comp_flat = complexities.flatten().cpu().numpy()
            variances.append(np.var(comp_flat))
        
        ax.plot(timesteps, variances, 'g-', linewidth=2)
        ax.fill_between(timesteps, 0, variances, alpha=0.3, color='green')
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Complexity Variance')
        ax.set_title('Complexity Variance Evolution')
        ax.grid(True, alpha=0.3)
        
        fig.suptitle('Fractal Complexity Trajectory Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_complexity_landscape(self,
                                  x_range: Tuple[float, float] = (-1, 1),
                                  y_range: Tuple[float, float] = (-1, 1),
                                  resolution: int = 50) -> plt.Figure:
        """
        Create a 2D landscape showing complexity as a function of two parameters.
        Useful for understanding the complexity topology.
        """
        fig = plt.figure(figsize=(14, 6))
        
        # Create parameter grid
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Compute complexity for each point
        Z = np.zeros_like(X)
        
        for i in range(resolution):
            for j in range(resolution):
                # Create Turn with both parameters
                turn_x = Turn(X[i, j], Handedness.RIGHT)
                turn_y = Turn(Y[i, j], Handedness.RIGHT if Y[i, j] > 0 else Handedness.LEFT)
                
                # Combined complexity
                Z[i, j] = turn_x.fractal_complexity + turn_y.fractal_complexity
        
        # Plot 1: 3D surface
        ax1 = fig.add_subplot(121, projection='3d')
        surf = ax1.plot_surface(X, Y, Z, cmap='RdYlBu_r', alpha=0.8,
                               antialiased=True, linewidth=0)
        
        ax1.set_xlabel('Parameter X (turns)')
        ax1.set_ylabel('Parameter Y (turns)')
        ax1.set_zlabel('Fractal Complexity')
        ax1.set_title('3D Complexity Landscape')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
        
        # Plot 2: 2D contour
        ax2 = fig.add_subplot(122)
        contour = ax2.contourf(X, Y, Z, levels=20, cmap='RdYlBu_r')
        ax2.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        # Mark special points
        ax2.plot(0, 0, 'ko', markersize=10, label='Origin')
        ax2.plot([1/3, 1/2, 2/3], [1/3, 1/2, 2/3], 'go', 
                markersize=8, label='Low complexity')
        
        ax2.set_xlabel('Parameter X (turns)')
        ax2.set_ylabel('Parameter Y (turns)')
        ax2.set_title('2D Complexity Contours')
        ax2.legend()
        
        fig.colorbar(contour, ax=ax2)
        
        fig.suptitle('Fractal Complexity Landscape', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def analyze_gesture_complexities(self) -> plt.Figure:
        """
        Analyze and visualize the complexity of different gestures.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        gestures = list(Gesture)
        handedness_options = [Handedness.RIGHT, Handedness.LEFT]
        
        # Analyze each gesture
        gesture_data = {}
        
        for gesture in gestures:
            complexities_right = []
            complexities_left = []
            
            for hand in handedness_options:
                movement = Movement(
                    gesture=gesture,
                    mode='test',
                    hand_override=hand
                )
                
                complexity = self.compute_movement_complexity(movement)
                
                if hand == Handedness.RIGHT:
                    complexities_right.append(complexity)
                else:
                    complexities_left.append(complexity)
            
            gesture_data[gesture.value] = {
                'right': np.mean(complexities_right),
                'left': np.mean(complexities_left)
            }
        
        # Plot 1: Gesture complexity comparison
        ax = axes[0]
        gesture_names = list(gesture_data.keys())
        right_complexities = [gesture_data[g]['right'] for g in gesture_names]
        left_complexities = [gesture_data[g]['left'] for g in gesture_names]
        
        x = np.arange(len(gesture_names))
        width = 0.35
        
        ax.bar(x - width/2, right_complexities, width, label='Right-handed', 
               color='blue', alpha=0.7)
        ax.bar(x + width/2, left_complexities, width, label='Left-handed',
               color='red', alpha=0.7)
        
        ax.set_xlabel('Gesture')
        ax.set_ylabel('Complexity')
        ax.set_title('Gesture Complexity by Handedness')
        ax.set_xticks(x)
        ax.set_xticklabels(gesture_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Complexity ratio (left/right)
        ax = axes[1]
        ratios = [gesture_data[g]['left'] / gesture_data[g]['right'] 
                 for g in gesture_names]
        
        ax.bar(x, ratios, color='purple', alpha=0.7)
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Gesture')
        ax.set_ylabel('Complexity Ratio (L/R)')
        ax.set_title('Handedness Complexity Ratio')
        ax.set_xticks(x)
        ax.set_xticklabels(gesture_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Total complexity
        ax = axes[2]
        total_complexities = [gesture_data[g]['right'] + gesture_data[g]['left']
                             for g in gesture_names]
        
        colors = ['green' if tc < 5 else 'yellow' if tc < 10 else 'red' 
                 for tc in total_complexities]
        ax.bar(x, total_complexities, color=colors, alpha=0.7)
        
        ax.set_xlabel('Gesture')
        ax.set_ylabel('Total Complexity')
        ax.set_title('Total Gesture Complexity')
        ax.set_xticks(x)
        ax.set_xticklabels(gesture_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        ax = axes[3]
        ax.axis('off')
        
        summary_text = "Complexity Summary:\n\n"
        summary_text += f"Most Complex: {gesture_names[np.argmax(total_complexities)]}\n"
        summary_text += f"Least Complex: {gesture_names[np.argmin(total_complexities)]}\n"
        summary_text += f"Mean Complexity: {np.mean(total_complexities):.2f}\n"
        summary_text += f"Complexity Range: {np.ptp(total_complexities):.2f}\n\n"
        summary_text += "Handedness Effect:\n"
        summary_text += f"Mean L/R Ratio: {np.mean(ratios):.2f}\n"
        summary_text += f"Max L/R Ratio: {np.max(ratios):.2f}"
        
        ax.text(0.1, 0.5, summary_text, fontsize=11, 
               transform=ax.transAxes, verticalalignment='center')
        
        fig.suptitle('Gesture Complexity Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig