"""
Interactive 48-generation fractal navigator based on fractal_complexity.
Represents the unfolding fractal structure with viewer-adaptive rendering
and movement constraints preserving logical spacetime.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import time
import math

try:
    import pygame
    from pygame.locals import *
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    pygame = None

from agi.harmonia.notation import Turn, Handedness, Phase48
from agi.harmonia.laws import Laws


@dataclass
class ViewerState:
    """Represents the viewer's position and orientation in fractal space."""
    position: np.ndarray  # (x, y, z) position
    velocity: np.ndarray  # (vx, vy, vz) velocity vector
    orientation: np.ndarray  # (pitch, yaw, roll) in radians
    initial_y: float  # The initial y-value when moving orthogonally
    has_moved_orthogonally: bool  # Track if we've left the initial axis
    time: float  # Current time
    trajectory: List[np.ndarray]  # History of positions
    
    def __post_init__(self):
        """Ensure arrays are float type."""
        self.position = self.position.astype(np.float64)
        self.velocity = self.velocity.astype(np.float64)
        self.orientation = self.orientation.astype(np.float64)
    
    def update(self, dt: float, controls: Dict[str, bool]):
        """Update viewer state based on controls and constraints."""
        # Store previous position
        prev_pos = self.position.copy()
        
        # Apply controls to velocity
        acceleration = 0.5
        if controls.get('forward', False):
            self.velocity[2] -= acceleration * dt
        if controls.get('backward', False):
            self.velocity[2] += acceleration * dt
        if controls.get('left', False):
            self.velocity[0] -= acceleration * dt
        if controls.get('right', False):
            self.velocity[0] += acceleration * dt
        if controls.get('up', False):
            self.velocity[1] += acceleration * dt
        if controls.get('down', False):
            self.velocity[1] -= acceleration * dt
            
        # Apply rotation
        if controls.get('yaw_left', False):
            self.orientation[1] -= dt
        if controls.get('yaw_right', False):
            self.orientation[1] += dt
        if controls.get('pitch_up', False):
            self.orientation[0] -= dt
        if controls.get('pitch_down', False):
            self.orientation[0] += dt
            
        # Apply velocity with rotation
        rot_matrix = self._get_rotation_matrix()
        world_velocity = rot_matrix @ self.velocity
        self.position += world_velocity * dt
        
        # Check if we've moved orthogonally
        if not self.has_moved_orthogonally:
            if abs(self.position[1] - prev_pos[1]) > 0.001:  # Y movement threshold
                self.has_moved_orthogonally = True
                self.initial_y = prev_pos[1]
        
        # Enforce Y constraint: cannot go below initial_y once moved orthogonally
        if self.has_moved_orthogonally:
            self.position[1] = max(self.position[1], self.initial_y)
            if self.position[1] == self.initial_y and self.velocity[1] < 0:
                self.velocity[1] = 0  # Stop downward velocity at floor
        
        # Apply damping
        self.velocity *= 0.95
        
        # Update time
        self.time += dt
        
        # Record trajectory
        self.trajectory.append(self.position.copy())
        if len(self.trajectory) > 1000:  # Limit history
            self.trajectory.pop(0)
    
    def _get_rotation_matrix(self) -> np.ndarray:
        """Get 3x3 rotation matrix from orientation."""
        pitch, yaw, roll = self.orientation
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])
        
        Ry = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])
        
        Rz = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ])
        
        return Ry @ Rx @ Rz
    
    @property
    def speed(self) -> float:
        """Current speed magnitude."""
        return np.linalg.norm(self.velocity)


class FractalNavigator:
    """
    48-generation deep fractal navigator with adaptive rendering.
    The fractal expands to accommodate the viewer's vantage point.
    """
    
    def __init__(self, 
                 generations: int = 48,
                 base_complexity: float = 1.0,
                 viewport_size: Tuple[int, int] = (1280, 720)):
        self.generations = generations
        self.base_complexity = base_complexity
        self.viewport_size = viewport_size
        
        # Initialize viewer state
        self.viewer = ViewerState(
            position=np.array([0.0, 0.0, 5.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0]),
            initial_y=0.0,
            has_moved_orthogonally=False,
            time=0.0,
            trajectory=[]
        )
        
        # Fractal parameters
        self.fractal_cache = {}
        self.visible_nodes = []
        self.sample_rate = 1.0
        
    def compute_fractal_node(self, 
                            generation: int, 
                            position: np.ndarray,
                            parent_complexity: float = 1.0) -> Dict:
        """
        Compute a single fractal node with its complexity.
        Based on the fractal_complexity from notation.py.
        """
        # Create Turn to compute complexity
        angle = (generation * np.pi / 24) % (2 * np.pi)  # Map generation to angle
        turn = Turn(angle / (2 * np.pi))  # Convert to turns
        
        # Node complexity inherits from parent and adds its own
        node_complexity = parent_complexity * turn.fractal_complexity
        
        # Determine handedness based on generation parity
        hand = Handedness.RIGHT if generation % 2 == 0 else Handedness.LEFT
        
        # Adjust complexity for handedness
        if hand == Handedness.LEFT:
            node_complexity *= 1.5  # Less penalty than full 4x for visual balance
        
        # Create child positions using 48-manifold structure
        children = []
        if generation < self.generations:
            # Use dyadic (2^4) and triadic (3) branching
            # Branch based on generation depth
            if generation % 3 == 0:  # Triadic branching
                branch_count = 3
                angle_step = 2 * np.pi / 3
            else:  # Dyadic branching
                branch_count = 2
                angle_step = np.pi
            
            for i in range(branch_count):
                angle = i * angle_step + generation * 0.1  # Slight rotation per generation
                
                # Calculate child position
                radius = 1.0 / (1 + generation * 0.1)  # Decrease with depth
                offset = np.array([
                    radius * np.cos(angle),
                    radius * np.sin(angle) * 0.5,  # Flatten Y slightly
                    radius * np.sin(angle)
                ])
                
                child_pos = position + offset
                children.append({
                    'position': child_pos,
                    'generation': generation + 1,
                    'parent_complexity': node_complexity
                })
        
        return {
            'position': position,
            'generation': generation,
            'complexity': node_complexity,
            'handedness': hand,
            'children': children,
            'radius': 0.5 / (1 + generation * 0.2)  # Visual size
        }
    
    def generate_visible_fractals(self, max_distance: float = 50.0):
        """
        Generate fractal nodes visible from current viewpoint.
        Adaptively generates based on viewer position and orientation.
        """
        self.visible_nodes = []
        
        # Start with root node
        root = self.compute_fractal_node(0, np.array([0, 0, 0]))
        
        # BFS to generate visible nodes
        queue = [root]
        
        while queue:
            node = queue.pop(0)
            
            # Check if node is within view distance
            distance = np.linalg.norm(node['position'] - self.viewer.position)
            
            if distance < max_distance:
                # Check if node is in view frustum (simplified)
                to_node = node['position'] - self.viewer.position
                if np.dot(to_node, self.viewer._get_rotation_matrix()[2]) < 0:  # In front
                    self.visible_nodes.append(node)
                    
                    # Add children to queue if close enough
                    if distance < max_distance * 0.7:  # LOD threshold
                        for child_data in node['children']:
                            child = self.compute_fractal_node(
                                child_data['generation'],
                                child_data['position'],
                                child_data['parent_complexity']
                            )
                            queue.append(child)
    
    def calculate_sample_rate(self):
        """
        Calculate sample rate based on velocity.
        Sample rate is factorial of present velocity magnitude.
        """
        speed = self.viewer.speed
        
        # Factorial approximation for continuous values using Stirling's approximation
        if speed < 1:
            self.sample_rate = 1.0
        else:
            # Stirling's approximation: n! ≈ √(2πn) * (n/e)^n
            n = min(speed, 10)  # Cap at 10 for performance
            self.sample_rate = np.sqrt(2 * np.pi * n) * (n / np.e) ** n
            self.sample_rate = min(self.sample_rate, 100)  # Cap maximum
    
    def render_matplotlib(self, ax: Optional[plt.Axes] = None) -> plt.Figure:
        """
        Render current fractal state using matplotlib.
        """
        if ax is None:
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.get_figure()
        
        ax.clear()
        
        # Generate visible fractals
        self.generate_visible_fractals()
        
        # Calculate sample rate
        self.calculate_sample_rate()
        
        # Sample nodes based on rate
        sample_step = max(1, int(len(self.visible_nodes) / self.sample_rate))
        sampled_nodes = self.visible_nodes[::sample_step]
        
        # Plot nodes
        for node in sampled_nodes:
            pos = node['position']
            complexity = node['complexity']
            generation = node['generation']
            
            # Color based on complexity and handedness
            if node['handedness'] == Handedness.RIGHT:
                color = plt.cm.viridis(complexity / 10)
            else:
                color = plt.cm.plasma(complexity / 10)
            
            # Size based on generation (closer = larger)
            size = node['radius'] * 100 * (1 + 1/(1 + generation))
            
            ax.scatter(pos[0], pos[1], pos[2], 
                      c=[color], s=size, alpha=0.7,
                      marker='o' if node['handedness'] == Handedness.RIGHT else '^')
        
        # Plot viewer trajectory
        if len(self.viewer.trajectory) > 1:
            traj = np.array(self.viewer.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                   'r-', alpha=0.3, linewidth=1)
        
        # Plot viewer position
        vpos = self.viewer.position
        ax.scatter(vpos[0], vpos[1], vpos[2],
                  c='red', s=200, marker='*', label='Viewer')
        
        # Add floor plane if orthogonal movement occurred
        if self.viewer.has_moved_orthogonally:
            xx, zz = np.meshgrid(np.linspace(-20, 20, 10),
                                np.linspace(-20, 20, 10))
            yy = np.ones_like(xx) * self.viewer.initial_y
            ax.plot_surface(xx, yy, zz, alpha=0.1, color='blue')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y (constrained)')
        ax.set_zlabel('Z')
        ax.set_title(f'Fractal Navigator - Generation Depth: {self.generations}\n'
                    f'Sample Rate: {self.sample_rate:.2f} (Speed: {self.viewer.speed:.2f})\n'
                    f'Y-Constrained: {self.viewer.has_moved_orthogonally}')
        
        # Set view based on viewer orientation
        ax.view_init(elev=np.degrees(self.viewer.orientation[0]),
                    azim=np.degrees(self.viewer.orientation[1]))
        
        ax.legend()
        
        return fig


if OPENGL_AVAILABLE:
    class InteractiveFractalNavigator(FractalNavigator):
        """
        Interactive version using pygame for real-time navigation.
        """
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.running = False
            self.clock = None
            self.screen = None
        
        def init_pygame(self):
            """Initialize pygame and OpenGL."""
            pygame.init()
            self.screen = pygame.display.set_mode(self.viewport_size, DOUBLEBUF | OPENGL)
            pygame.display.set_caption("48-Generation Fractal Navigator")
            
            # OpenGL setup
            gluPerspective(45, (self.viewport_size[0] / self.viewport_size[1]), 0.1, 100.0)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            self.clock = pygame.time.Clock()
        
        def draw_fractal_opengl(self):
            """Draw fractal using OpenGL."""
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Set view matrix based on viewer state
            glLoadIdentity()
            
            # Apply viewer transformation
            glRotatef(-np.degrees(self.viewer.orientation[0]), 1, 0, 0)
            glRotatef(-np.degrees(self.viewer.orientation[1]), 0, 1, 0)
            glTranslatef(-self.viewer.position[0], 
                         -self.viewer.position[1], 
                         -self.viewer.position[2])
            
            # Generate and draw visible fractals
            self.generate_visible_fractals()
            self.calculate_sample_rate()
            
            sample_step = max(1, int(len(self.visible_nodes) / self.sample_rate))
            sampled_nodes = self.visible_nodes[::sample_step]
            
            for node in sampled_nodes:
                pos = node['position']
                complexity = node['complexity']
                
                # Set color based on complexity
                normalized_complexity = min(complexity / 10, 1.0)
                if node['handedness'] == Handedness.RIGHT:
                    glColor4f(0, normalized_complexity, 1 - normalized_complexity, 0.7)
                else:
                    glColor4f(normalized_complexity, 0, 1 - normalized_complexity, 0.7)
                
                # Draw as point or sphere (simplified)
                glPushMatrix()
                glTranslatef(pos[0], pos[1], pos[2])
                
                # Draw a simple cube for performance
                size = node['radius']
                glBegin(GL_QUADS)
                # Front face
                glVertex3f(-size, -size, size)
                glVertex3f(size, -size, size)
                glVertex3f(size, size, size)
                glVertex3f(-size, size, size)
                # Back face
                glVertex3f(-size, -size, -size)
                glVertex3f(size, -size, -size)
                glVertex3f(size, size, -size)
                glVertex3f(-size, size, -size)
                glEnd()
                
                glPopMatrix()
            
            # Draw floor plane if needed
            if self.viewer.has_moved_orthogonally:
                glColor4f(0, 0, 1, 0.1)
                glBegin(GL_QUADS)
                glVertex3f(-50, self.viewer.initial_y, -50)
                glVertex3f(50, self.viewer.initial_y, -50)
                glVertex3f(50, self.viewer.initial_y, 50)
                glVertex3f(-50, self.viewer.initial_y, 50)
                glEnd()
        
        def run(self):
            """Main interactive loop."""
            self.init_pygame()
            self.running = True
            
            controls = {}
            
            while self.running:
                dt = self.clock.tick(60) / 1000.0  # 60 FPS target
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_w:
                        controls['forward'] = True
                    elif event.key == pygame.K_s:
                        controls['backward'] = True
                    elif event.key == pygame.K_a:
                        controls['left'] = True
                    elif event.key == pygame.K_d:
                        controls['right'] = True
                    elif event.key == pygame.K_SPACE:
                        controls['up'] = True
                    elif event.key == pygame.K_LSHIFT:
                        controls['down'] = True
                    elif event.key == pygame.K_LEFT:
                        controls['yaw_left'] = True
                    elif event.key == pygame.K_RIGHT:
                        controls['yaw_right'] = True
                    elif event.key == pygame.K_UP:
                        controls['pitch_up'] = True
                    elif event.key == pygame.K_DOWN:
                        controls['pitch_down'] = True
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_w:
                        controls['forward'] = False
                    elif event.key == pygame.K_s:
                        controls['backward'] = False
                    elif event.key == pygame.K_a:
                        controls['left'] = False
                    elif event.key == pygame.K_d:
                        controls['right'] = False
                    elif event.key == pygame.K_SPACE:
                        controls['up'] = False
                    elif event.key == pygame.K_LSHIFT:
                        controls['down'] = False
                    elif event.key == pygame.K_LEFT:
                        controls['yaw_left'] = False
                    elif event.key == pygame.K_RIGHT:
                        controls['yaw_right'] = False
                    elif event.key == pygame.K_UP:
                        controls['pitch_up'] = False
                    elif event.key == pygame.K_DOWN:
                        controls['pitch_down'] = False
            
            # Update viewer state
            self.viewer.update(dt, controls)
            
            # Render
            self.draw_fractal_opengl()
            
            # Display info
            font = pygame.font.Font(None, 36)
            text = font.render(f"Speed: {self.viewer.speed:.2f} | Sample Rate: {self.sample_rate:.2f}", 
                             True, (255, 255, 255))
            # Note: Text rendering in OpenGL mode requires special handling
            
            pygame.display.flip()
        
        pygame.quit()
else:
    # Dummy class when pygame not available
    class InteractiveFractalNavigator:
        def __init__(self, *args, **kwargs):
            raise ImportError("pygame and PyOpenGL are required for interactive mode")
        def run(self):
            raise ImportError("pygame and PyOpenGL are required for interactive mode")


def create_matplotlib_animation():
    """Create an animated matplotlib visualization."""
    navigator = FractalNavigator()
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        # Simulate movement
        t = frame * 0.1
        navigator.viewer.position = np.array([
            5 * np.cos(t),
            2 + np.sin(t * 0.5),
            5 * np.sin(t)
        ])
        navigator.viewer.velocity = np.array([
            -5 * np.sin(t) * 0.1,
            0.5 * np.cos(t * 0.5) * 0.1,
            5 * np.cos(t) * 0.1
        ])
        
        # Check for orthogonal movement
        if frame > 30 and not navigator.viewer.has_moved_orthogonally:
            navigator.viewer.has_moved_orthogonally = True
            navigator.viewer.initial_y = navigator.viewer.position[1]
        
        navigator.viewer.time = frame * 0.1
        navigator.viewer.trajectory.append(navigator.viewer.position.copy())
        
        navigator.render_matplotlib(ax)
        
        return ax.collections + ax.lines
    
    anim = animation.FuncAnimation(fig, update, frames=200, 
                                 interval=50, blit=False)
    
    return fig, anim