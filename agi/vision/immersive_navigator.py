"""
Immersive 3D fractal navigation system with dynamic manifold creation.
 
Features:
- Mouse/keyboard navigation in 3D space
- Dynamic manifold floor creation when moving up
- Fractal folding/unfolding based on departure angles
- Spherical aura system with toggleable layer count
- Angular fractal spawning (360°/angle = fractal count)
"""
 
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum
import math
import time
 
 
class NavigationMode(Enum):
    """Navigation control modes."""
    WALK = "walk"  # Ground-based movement
    FLY = "fly"    # Free 3D movement
    ORBIT = "orbit" # Orbit around focused fractal
 
 
@dataclass
class FractalNode:
    """A fractal node in 3D space."""
    position: np.ndarray  # [x, y, z]
    generation: int
    parent: Optional['FractalNode']
    children: List['FractalNode']
    complexity: float
    departure_angle: float  # Angle of departure from parent
    is_folded: bool
    color: Tuple[float, float, float]
    radius: float
    pulse_phase: float
    manifold_level: int  # Which manifold floor this belongs to
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def distance_to(self, point: np.ndarray) -> float:
        """Calculate distance to a point."""
        return np.linalg.norm(self.position - point)
    
    def fold(self):
        """Fold this fractal and its children."""
        self.is_folded = True
        for child in self.children:
            child.fold()
    
    def unfold(self):
        """Unfold this fractal and its children."""
        self.is_folded = False
        for child in self.children:
            child.unfold()
 
 
@dataclass
class ManifoldFloor:
    """A manifold floor that can be created dynamically."""
    level: int
    y_position: float
    nodes: List[FractalNode]
    grid_size: int
    color: Tuple[float, float, float]
    opacity: float
    
    def add_node(self, node: FractalNode):
        """Add a node to this manifold floor."""
        self.nodes.append(node)
        node.manifold_level = self.level
 
 
class ImmersiveNavigator:
    """3D fractal navigation system with dynamic manifold creation."""
    
    def __init__(self, width: int = 1280, height: int = 720, 
                 max_generations: int = 48, toggleable_generations: bool = True):
        """Initialize the immersive navigator.
        
        Args:
            width: Window width
            height: Window height
            max_generations: Maximum fractal generations (default 48)
            toggleable_generations: Allow toggling generation count
        """
        self.width = width
        self.height = height
        self.max_generations = max_generations
        self.current_generations = 6  # Start with fewer for performance
        self.toggleable_generations = toggleable_generations
        
        # Camera state - start further back to see the fractal
        self.camera_pos = np.array([0.0, 2.0, 10.0])
        self.camera_look = np.array([0.0, 0.0, 0.0])
        self.camera_up = np.array([0.0, 1.0, 0.0])
        self.yaw = 0.0  # Horizontal rotation
        self.pitch = 0.0  # Vertical rotation
        
        # Movement state
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.move_speed = 0.1
        self.sprint_multiplier = 2.0
        self.is_sprinting = False
        self.nav_mode = NavigationMode.FLY
        
        # Fractal system
        self.root_node: Optional[FractalNode] = None
        self.all_nodes: List[FractalNode] = []
        self.focused_node: Optional[FractalNode] = None
        self.aura_radius = 2.0
        self.aura_active = True
        
        # Manifold system
        self.manifold_floors: Dict[int, ManifoldFloor] = {}
        self.current_floor = 0
        self.floor_height = 10.0
        
        # Interaction state
        self.keys_pressed: Set[int] = set()
        self.mouse_captured = False
        self.mouse_sensitivity = 0.002
        
        # Rendering state
        self.show_wireframe = False
        self.show_aura = True
        self.show_manifolds = True
        self.show_stats = True
        
        # Animation
        self.time = 0.0
        self.dt = 0.016  # ~60 FPS
        
        # Initialize pygame and OpenGL
        self._init_pygame()
        self._init_opengl()
        self._generate_fractal_tree()
        self._create_initial_manifold()
    
    def _init_pygame(self):
        """Initialize pygame and create window."""
        pygame.init()
        pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Immersive Fractal Navigator - 48-Manifold System")
        pygame.event.set_grab(False)  # Start with mouse not captured
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
    
    def _init_opengl(self):
        """Initialize OpenGL settings."""
        # Set clear color to dark blue for visibility
        glClearColor(0.05, 0.05, 0.15, 1.0)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_LINE_SMOOTH)
        
        # Lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [0.0, 10.0, 0.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.4, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.9, 1.0])
        
        # Material properties for better visibility
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
        glMaterialf(GL_FRONT, GL_SHININESS, 30.0)
        
        # Fog for depth (reduced density for better visibility)
        glEnable(GL_FOG)
        glFogi(GL_FOG_MODE, GL_EXP)
        glFogfv(GL_FOG_COLOR, [0.05, 0.05, 0.15, 1.0])
        glFogf(GL_FOG_DENSITY, 0.005)
        
        self._setup_projection()
    
    def _setup_projection(self):
        """Setup projection matrix."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, self.width / self.height, 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)
    
    def _generate_fractal_tree(self):
        """Generate the initial fractal tree structure."""
        # Create root node at origin
        self.root_node = FractalNode(
            position=np.array([0.0, 0.0, 0.0]),
            generation=0,
            parent=None,
            children=[],
            complexity=0.0,
            departure_angle=0.0,
            is_folded=False,
            color=(1.0, 1.0, 1.0),
            radius=1.0,
            pulse_phase=0.0,
            manifold_level=0
        )
        self.all_nodes = [self.root_node]
        
        # Generate children recursively
        self._generate_children(self.root_node, self.current_generations)
    
    def _generate_children(self, parent: FractalNode, generations_left: int):
        """Recursively generate fractal children based on departure angles."""
        if generations_left <= 0:
            return
        
        # Limit depth to prevent infinite recursion
        if parent.generation >= self.current_generations - 1:
            return
        
        # Calculate number of children based on generation
        # Start with fewer children, decrease with depth for performance
        if parent.generation == 0:
            angles = [0, 90, 180, 270]  # 4 cardinal directions
        elif parent.generation == 1:
            angles = [0, 120, 240]  # 3 children
        elif parent.generation == 2:
            angles = [0, 180]  # 2 children
        else:
            # Fewer children at deeper levels
            angles = [0]  # 1 child
        
        for angle in angles:
            # Calculate child position using spherical coordinates
            theta = np.radians(angle)
            phi = np.radians(30 + parent.generation * 10)  # Vary elevation
            
            # Distance decreases with generation
            distance = 3.0 / (1 + parent.generation * 0.3)
            
            offset = np.array([
                distance * np.sin(phi) * np.cos(theta),
                distance * np.cos(phi) * 0.3,  # Less vertical spread
                distance * np.sin(phi) * np.sin(theta)
            ])
            
            child_pos = parent.position + offset
            
            # Calculate complexity based on angle and generation
            complexity = (parent.complexity + 1.0) * (1 + abs(np.sin(theta)))
            
            # Color based on angle and generation
            hue = (angle / 360.0 + parent.generation * 0.1) % 1.0
            color = self._hsv_to_rgb(hue, 0.8, 0.9)
            
            child = FractalNode(
                position=child_pos,
                generation=parent.generation + 1,
                parent=parent,
                children=[],
                complexity=complexity,
                departure_angle=angle,
                is_folded=False,
                color=color,
                radius=0.5 / (1 + parent.generation * 0.2),
                pulse_phase=np.random.random() * np.pi * 2,
                manifold_level=parent.manifold_level
            )
            
            parent.children.append(child)
            self.all_nodes.append(child)
            
            # Recursive generation
            self._generate_children(child, generations_left - 1)
    
    def _create_initial_manifold(self):
        """Create the initial manifold floor."""
        floor = ManifoldFloor(
            level=0,
            y_position=0.0,
            nodes=[self.root_node],
            grid_size=20,
            color=(0.2, 0.3, 0.5),
            opacity=0.3
        )
        self.manifold_floors[0] = floor
    
    def _create_new_manifold(self, level: int):
        """Create a new manifold floor when moving up."""
        y_pos = level * self.floor_height
        
        # Gather nodes at this level
        nodes_at_level = [n for n in self.all_nodes 
                         if abs(n.position[1] - y_pos) < self.floor_height / 2]
        
        floor = ManifoldFloor(
            level=level,
            y_position=y_pos,
            nodes=nodes_at_level,
            grid_size=20 + level * 5,  # Larger grids at higher levels
            color=(0.2 + level * 0.1, 0.3, 0.5 - level * 0.05),
            opacity=0.3 - level * 0.02
        )
        
        self.manifold_floors[level] = floor
        
        # Generate new fractals on this manifold
        for node in nodes_at_level:
            if node.generation < self.current_generations / 2:
                self._spawn_manifold_fractals(node, level)
    
    def _spawn_manifold_fractals(self, origin: FractalNode, level: int):
        """Spawn new fractals on a manifold based on departure angle."""
        # Calculate number of fractals: 360 / departure_angle
        departure_angle = 30 + level * 5  # Vary by manifold level
        num_fractals = int(360 / departure_angle)
        
        for i in range(num_fractals):
            angle = i * departure_angle
            theta = np.radians(angle)
            
            # Position on manifold plane
            distance = 2.0 + level * 0.5
            position = origin.position + np.array([
                distance * np.cos(theta),
                0.1,  # Slightly above manifold
                distance * np.sin(theta)
            ])
            
            # Create new fractal node
            node = FractalNode(
                position=position,
                generation=origin.generation + 1,
                parent=origin,
                children=[],
                complexity=origin.complexity * 1.5,
                departure_angle=angle,
                is_folded=False,
                color=self._hsv_to_rgb((angle / 360 + level * 0.2) % 1, 0.7, 0.8),
                radius=0.3,
                pulse_phase=np.random.random() * np.pi * 2,
                manifold_level=level
            )
            
            origin.children.append(node)
            self.all_nodes.append(node)
    
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[float, float, float]:
        """Convert HSV to RGB color."""
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c
        
        if h < 1/6:
            r, g, b = c, x, 0
        elif h < 2/6:
            r, g, b = x, c, 0
        elif h < 3/6:
            r, g, b = 0, c, x
        elif h < 4/6:
            r, g, b = 0, x, c
        elif h < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return (r + m, g + m, b + m)
    
    def _update_camera(self):
        """Update camera based on yaw/pitch and look direction."""
        # Calculate look direction from yaw/pitch
        look_dir = np.array([
            np.sin(self.yaw) * np.cos(self.pitch),
            np.sin(self.pitch),
            -np.cos(self.yaw) * np.cos(self.pitch)
        ])
        
        self.camera_look = self.camera_pos + look_dir
        
        # Update camera up vector for roll
        self.camera_up = np.array([0, 1, 0])
    
    def _handle_movement(self):
        """Handle keyboard movement input."""
        move_vec = np.array([0.0, 0.0, 0.0])
        
        # Calculate forward/right vectors
        forward = self.camera_look - self.camera_pos
        forward[1] = 0 if self.nav_mode == NavigationMode.WALK else forward[1]
        forward = forward / np.linalg.norm(forward) if np.linalg.norm(forward) > 0 else np.array([0, 0, 1])
        
        right = np.cross(forward, self.camera_up)
        right = right / np.linalg.norm(right) if np.linalg.norm(right) > 0 else np.array([1, 0, 0])
        
        speed = self.move_speed * (self.sprint_multiplier if self.is_sprinting else 1.0)
        
        # WASD movement
        if K_w in self.keys_pressed:
            move_vec += forward * speed
        if K_s in self.keys_pressed:
            move_vec -= forward * speed
        if K_a in self.keys_pressed:
            move_vec -= right * speed
        if K_d in self.keys_pressed:
            move_vec += right * speed
        
        # Vertical movement (Space/Shift)
        if K_SPACE in self.keys_pressed:
            move_vec[1] += speed
            # Check if we've moved to a new manifold level
            new_level = int(self.camera_pos[1] / self.floor_height)
            if new_level > self.current_floor and new_level not in self.manifold_floors:
                self._create_new_manifold(new_level)
                self.current_floor = new_level
        
        if K_LSHIFT in self.keys_pressed and self.nav_mode == NavigationMode.FLY:
            move_vec[1] -= speed
        
        # Apply movement with smoothing
        self.velocity = self.velocity * 0.9 + move_vec * 0.1
        self.camera_pos += self.velocity
        
        # Ground collision in WALK mode
        if self.nav_mode == NavigationMode.WALK:
            floor_y = self.current_floor * self.floor_height
            if self.camera_pos[1] < floor_y + 1.6:  # Eye height
                self.camera_pos[1] = floor_y + 1.6
                self.velocity[1] = 0
    
    def _handle_folding_unfolding(self):
        """Handle fractal folding/unfolding based on proximity and movement."""
        if not self.aura_active:
            return
        
        # Find nodes within aura radius
        for node in self.all_nodes:
            distance = node.distance_to(self.camera_pos)
            
            if distance < self.aura_radius:
                # Moving toward node - unfold
                to_node = node.position - self.camera_pos
                if np.dot(self.velocity, to_node) > 0:
                    node.unfold()
                # Moving away - fold
                elif np.dot(self.velocity, to_node) < 0:
                    node.fold()
    
    def _render_scene(self):
        """Render the entire 3D scene."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Set camera
        gluLookAt(
            self.camera_pos[0], self.camera_pos[1], self.camera_pos[2],
            self.camera_look[0], self.camera_look[1], self.camera_look[2],
            self.camera_up[0], self.camera_up[1], self.camera_up[2]
        )
        
        # Render manifold floors
        if self.show_manifolds:
            self._render_manifolds()
        
        # Render fractal tree
        self._render_fractals()
        
        # Render aura
        if self.show_aura:
            self._render_aura()
        
        # Render HUD
        if self.show_stats:
            self._render_hud()
    
    def _render_manifolds(self):
        """Render manifold floors."""
        glDisable(GL_LIGHTING)
        
        for level, floor in self.manifold_floors.items():
            glPushMatrix()
            glTranslatef(0, floor.y_position, 0)
            
            # Set color with transparency
            glColor4f(*floor.color, floor.opacity)
            
            # Draw grid
            glBegin(GL_LINES)
            grid_range = floor.grid_size
            for i in range(-grid_range, grid_range + 1):
                # X lines
                glVertex3f(i, 0, -grid_range)
                glVertex3f(i, 0, grid_range)
                # Z lines
                glVertex3f(-grid_range, 0, i)
                glVertex3f(grid_range, 0, i)
            glEnd()
            
            glPopMatrix()
        
        glEnable(GL_LIGHTING)
    
    def _render_fractals(self):
        """Render all fractal nodes."""
        glEnable(GL_LIGHTING)
        
        for node in self.all_nodes:
            if node.is_folded:
                continue
            
            glPushMatrix()
            glTranslatef(*node.position)
            
            # Pulsing effect
            pulse = 1.0 + 0.2 * np.sin(self.time * 2 + node.pulse_phase)
            scale = node.radius * pulse
            
            # Set material color for better lighting
            glMaterialfv(GL_FRONT, GL_AMBIENT, [*node.color, 1.0])
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [*node.color, 1.0])
            glColor4f(*node.color, 0.9)
            
            # Draw sphere using our custom method
            if self.show_wireframe:
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                self._draw_sphere(scale, 12, 12)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            else:
                # Draw using GL quads for compatibility
                self._draw_sphere(scale, 12, 12)
            
            glPopMatrix()
            
            # Draw connections to children
            if not node.is_folded:
                glDisable(GL_LIGHTING)
                glBegin(GL_LINES)
                for child in node.children:
                    if not child.is_folded:
                        glColor4f(*node.color, 0.3)
                        glVertex3f(*node.position)
                        glColor4f(*child.color, 0.3)
                        glVertex3f(*child.position)
                glEnd()
                glEnable(GL_LIGHTING)
    
    def _draw_sphere(self, radius: float, slices: int, stacks: int):
        """Draw a sphere using quads (no GLUT dependency)."""
        for i in range(stacks):
            lat0 = np.pi * (-0.5 + float(i) / stacks)
            z0 = radius * np.sin(lat0)
            zr0 = radius * np.cos(lat0)
            
            lat1 = np.pi * (-0.5 + float(i + 1) / stacks)
            z1 = radius * np.sin(lat1)
            zr1 = radius * np.cos(lat1)
            
            glBegin(GL_QUAD_STRIP)
            for j in range(slices + 1):
                lng = 2 * np.pi * float(j) / slices
                x = np.cos(lng)
                y = np.sin(lng)
                
                glNormal3f(x * zr0, y * zr0, z0)
                glVertex3f(x * zr0, y * zr0, z0)
                glNormal3f(x * zr1, y * zr1, z1)
                glVertex3f(x * zr1, y * zr1, z1)
            glEnd()
    
    def _render_aura(self):
        """Render the spherical aura around the camera."""
        glDisable(GL_LIGHTING)
        glPushMatrix()
        glTranslatef(*self.camera_pos)
        
        # Draw translucent sphere
        glColor4f(0.5, 0.7, 1.0, 0.1)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        self._draw_sphere(self.aura_radius, 16, 16)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        glPopMatrix()
        glEnable(GL_LIGHTING)
    
    def _render_hud(self):
        """Render HUD overlay with stats."""
        # Switch to 2D rendering
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        # Render text (would need proper text rendering in production)
        stats = [
            f"Generations: {self.current_generations}/{self.max_generations}",
            f"Manifold Level: {self.current_floor}",
            f"Nodes: {len([n for n in self.all_nodes if not n.is_folded])}/{len(self.all_nodes)}",
            f"Mode: {self.nav_mode.value}",
            f"Pos: ({self.camera_pos[0]:.1f}, {self.camera_pos[1]:.1f}, {self.camera_pos[2]:.1f})",
            "",
            "Controls:",
            "WASD: Move | Mouse: Look",
            "Space: Up | Shift: Down",
            "Tab: Capture mouse",
            "G: Toggle generations",
            "F: Toggle wireframe",
            "M: Cycle nav mode",
            "ESC: Exit"
        ]
        
        # Draw semi-transparent background
        glColor4f(0, 0, 0, 0.5)
        glBegin(GL_QUADS)
        glVertex2f(10, 10)
        glVertex2f(250, 10)
        glVertex2f(250, 20 + len(stats) * 20)
        glVertex2f(10, 20 + len(stats) * 20)
        glEnd()
        
        # Would render text here with proper font rendering
        # For now, just indicate where text would be
        glColor4f(1, 1, 1, 1)
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    def handle_event(self, event):
        """Handle pygame events."""
        if event.type == QUIT:
            return False
        
        elif event.type == KEYDOWN:
            self.keys_pressed.add(event.key)
            
            if event.key == K_ESCAPE:
                return False
            elif event.key == K_TAB:
                self.mouse_captured = not self.mouse_captured
                pygame.event.set_grab(self.mouse_captured)
                pygame.mouse.set_visible(not self.mouse_captured)
            elif event.key == K_f:
                self.show_wireframe = not self.show_wireframe
            elif event.key == K_m:
                # Cycle navigation mode
                modes = list(NavigationMode)
                idx = (modes.index(self.nav_mode) + 1) % len(modes)
                self.nav_mode = modes[idx]
            elif event.key == K_g and self.toggleable_generations:
                # Cycle generation count
                options = [3, 6, 9, 12, 24, 48]
                try:
                    idx = options.index(self.current_generations)
                    self.current_generations = options[(idx + 1) % len(options)]
                except ValueError:
                    self.current_generations = 6
                # Regenerate fractal tree
                self.all_nodes.clear()
                self._generate_fractal_tree()
            elif event.key == K_LCTRL:
                self.is_sprinting = True
        
        elif event.type == KEYUP:
            self.keys_pressed.discard(event.key)
            if event.key == K_LCTRL:
                self.is_sprinting = False
        
        elif event.type == MOUSEMOTION and self.mouse_captured:
            dx, dy = event.rel
            self.yaw += dx * self.mouse_sensitivity
            self.pitch -= dy * self.mouse_sensitivity
            self.pitch = np.clip(self.pitch, -np.pi/2 + 0.01, np.pi/2 - 0.01)
        
        return True
    
    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if not self.handle_event(event):
                    running = False
            
            # Update
            self.time += self.dt
            self._handle_movement()
            self._update_camera()
            self._handle_folding_unfolding()
            
            # Render
            self._render_scene()
            pygame.display.flip()
            
            # Frame rate
            self.clock.tick(60)
        
        pygame.quit()
 
 
def main():
    """Run the immersive fractal navigator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Immersive 3D Fractal Navigator")
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
    parser.add_argument("--generations", type=int, default=48, 
                       help="Maximum fractal generations")
    parser.add_argument("--fixed-generations", action="store_true",
                       help="Disable generation toggling")
    
    args = parser.parse_args()
    
    navigator = ImmersiveNavigator(
        width=args.width,
        height=args.height,
        max_generations=args.generations,
        toggleable_generations=not args.fixed_generations
    )
    
    print("=== Immersive Fractal Navigator ===")
    print(f"Manifold dimensions: {args.generations} generations")
    print("\nControls:")
    print("  WASD: Movement")
    print("  Mouse: Look around (Tab to capture)")
    print("  Space: Move up (creates new manifold)")
    print("  Shift: Move down")
    print("  Ctrl: Sprint")
    print("  G: Toggle generation count")
    print("  F: Toggle wireframe")
    print("  M: Cycle navigation mode")
    print("  ESC: Exit")
    print("\nStarting navigator...")
    
    navigator.run()
 
 
if __name__ == "__main__":
    main()