"""
Kid-friendly fractal playground with dynamic animations.
Designed for ages 5-10 with colorful, interactive fractals that grow and dance.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Polygon, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
from typing import List, Tuple, Optional, Dict
import time
import math
import colorsys
from dataclasses import dataclass
import random

# Try to import sound capabilities
try:
    import pygame
    import pygame.mixer
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False


@dataclass
class FractalCreature:
    """A friendly fractal creature that lives in the playground."""
    name: str
    x: float
    y: float
    size: float
    color: str
    shape: str  # 'circle', 'star', 'heart', 'flower', 'butterfly'
    mood: str  # 'happy', 'excited', 'sleepy', 'dancing'
    age: int  # Generation in fractal terms
    friends: List['FractalCreature']
    wobble_phase: float = 0.0
    spin_angle: float = 0.0
    pulse_size: float = 1.0
    
    def update(self, dt: float):
        """Update creature animation state."""
        # Wobble back and forth
        self.wobble_phase += dt * 2.0
        
        # Spin if excited
        if self.mood == 'excited':
            self.spin_angle += dt * 3.0
        elif self.mood == 'dancing':
            self.spin_angle = np.sin(self.wobble_phase) * 0.5
            
        # Pulse size when happy
        if self.mood == 'happy':
            self.pulse_size = 1.0 + 0.2 * np.sin(self.wobble_phase * 2)
        
    def get_position(self) -> Tuple[float, float]:
        """Get animated position."""
        wobble_x = self.x + 0.1 * np.sin(self.wobble_phase) * self.size
        wobble_y = self.y + 0.05 * np.cos(self.wobble_phase * 1.5) * self.size
        return wobble_x, wobble_y


class FractalPlayground:
    """
    An interactive, animated fractal playground for kids.
    Creates colorful, friendly fractals that grow and play together.
    """
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.creatures = []
        self.time = 0
        self.generation_count = 0
        self.max_generations = 7  # Keep it simple for kids
        
        # Fun color palettes for kids
        self.color_palettes = {
            'rainbow': ['#FF0000', '#FF7F00', '#FFFF00', '#00FF00', '#0000FF', '#4B0082', '#9400D3'],
            'candy': ['#FF69B4', '#FFB6C1', '#FFA0D2', '#FF1493', '#C71585', '#DB7093'],
            'ocean': ['#00CED1', '#4682B4', '#5F9EA0', '#6495ED', '#00BFFF', '#1E90FF'],
            'forest': ['#228B22', '#32CD32', '#00FF00', '#7CFC00', '#90EE90', '#98FB98'],
            'sunset': ['#FF4500', '#FF6347', '#FF8C00', '#FFA500', '#FFD700', '#FFFFE0'],
            'space': ['#4B0082', '#8A2BE2', '#9370DB', '#BA55D3', '#DDA0DD', '#EE82EE']
        }
        self.current_palette = 'rainbow'
        
        # Friendly creature names
        self.creature_names = [
            'Sparkle', 'Twinkle', 'Bubble', 'Giggles', 'Sunshine', 'Rainbow',
            'Stardust', 'Moonbeam', 'Flutter', 'Blossom', 'Shimmer', 'Glimmer',
            'Dazzle', 'Sprinkle', 'Cupcake', 'Jellybean', 'Marshmallow', 'Pixie'
        ]
        
        # Animation settings
        self.animation_speed = 1.0
        self.growth_rate = 0.5
        self.dance_mode = False
        self.rainbow_mode = True
        
        # Initialize playground
        self.setup_playground()
        
    def setup_playground(self):
        """Initialize the playground with a parent fractal creature."""
        # Create the first friendly fractal
        parent = FractalCreature(
            name=random.choice(self.creature_names),
            x=0,
            y=0,
            size=1.0,
            color=self.get_generation_color(0),
            shape='star',
            mood='happy',
            age=0,
            friends=[]
        )
        self.creatures = [parent]
        
    def get_generation_color(self, generation: int) -> str:
        """Get a fun color for this generation."""
        colors = self.color_palettes[self.current_palette]
        return colors[generation % len(colors)]
        
    def create_rainbow_color(self, phase: float) -> str:
        """Create a rainbow color based on phase."""
        hue = (phase % 1.0)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return mcolors.to_hex(rgb)
        
    def grow_fractal_family(self, parent: FractalCreature) -> List[FractalCreature]:
        """Create children fractals from a parent."""
        if parent.age >= self.max_generations:
            return []
            
        children = []
        
        # Number of friends based on generation
        if parent.age % 3 == 0:
            num_friends = 3  # Triangle of friends
        else:
            num_friends = 2  # Pair of friends
            
        # Create child creatures
        for i in range(num_friends):
            angle = (2 * np.pi * i / num_friends) + parent.spin_angle
            
            # Calculate child position
            distance = parent.size * 2.5
            child_x = parent.x + distance * np.cos(angle)
            child_y = parent.y + distance * np.sin(angle)
            
            # Pick a fun shape
            shapes = ['circle', 'star', 'heart', 'flower', 'butterfly']
            shape = random.choice(shapes)
            
            # Pick a mood
            moods = ['happy', 'excited', 'dancing']
            mood = random.choice(moods)
            
            # Create the child creature
            child = FractalCreature(
                name=random.choice(self.creature_names),
                x=child_x,
                y=child_y,
                size=parent.size * 0.6,
                color=self.get_generation_color(parent.age + 1),
                shape=shape,
                mood=mood,
                age=parent.age + 1,
                friends=[],
                wobble_phase=random.random() * 2 * np.pi
            )
            
            children.append(child)
            parent.friends.append(child)
            
        return children
        
    def draw_creature_shape(self, ax, creature: FractalCreature, alpha: float = 1.0):
        """Draw a friendly creature shape."""
        x, y = creature.get_position()
        size = creature.size * creature.pulse_size * 0.5
        
        if self.rainbow_mode:
            color = self.create_rainbow_color(self.time * 0.1 + creature.age * 0.1)
        else:
            color = creature.color
            
        if creature.shape == 'circle':
            circle = Circle((x, y), size, color=color, alpha=alpha, 
                          edgecolor='white', linewidth=2)
            ax.add_patch(circle)
            
        elif creature.shape == 'star':
            # Create a 5-pointed star
            angles = np.linspace(0, 2*np.pi, 11)[:-1] + creature.spin_angle
            radii = np.array([size, size*0.4] * 5)
            points = [(x + r*np.cos(a), y + r*np.sin(a)) 
                     for r, a in zip(radii, angles)]
            star = Polygon(points, color=color, alpha=alpha,
                          edgecolor='white', linewidth=2)
            ax.add_patch(star)
            
        elif creature.shape == 'heart':
            # Create a simple heart shape
            t = np.linspace(0, 2*np.pi, 100)
            heart_x = x + size * (16*np.sin(t)**3) / 16
            heart_y = y + size * (13*np.cos(t) - 5*np.cos(2*t) - 
                                  2*np.cos(3*t) - np.cos(4*t)) / 16
            ax.fill(heart_x, heart_y, color=color, alpha=alpha,
                   edgecolor='white', linewidth=2)
            
        elif creature.shape == 'flower':
            # Create a simple flower
            for i in range(6):
                angle = i * np.pi / 3 + creature.spin_angle
                petal_x = x + size * 0.5 * np.cos(angle)
                petal_y = y + size * 0.5 * np.sin(angle)
                petal = Circle((petal_x, petal_y), size*0.3, 
                             color=color, alpha=alpha*0.8,
                             edgecolor='white', linewidth=1)
                ax.add_patch(petal)
            # Center
            center = Circle((x, y), size*0.2, color='yellow', 
                          alpha=alpha, edgecolor='white', linewidth=2)
            ax.add_patch(center)
            
        elif creature.shape == 'butterfly':
            # Create butterfly wings
            wing_size = size * 0.8
            # Left wing
            left_wing = FancyBboxPatch((x - wing_size, y - wing_size/2),
                                      wing_size, wing_size,
                                      boxstyle="round,pad=0.1",
                                      color=color, alpha=alpha*0.8,
                                      edgecolor='white', linewidth=2,
                                      transform=ax.transData)
            # Right wing
            right_wing = FancyBboxPatch((x, y - wing_size/2),
                                       wing_size, wing_size,
                                       boxstyle="round,pad=0.1",
                                       color=color, alpha=alpha*0.8,
                                       edgecolor='white', linewidth=2,
                                       transform=ax.transData)
            ax.add_patch(left_wing)
            ax.add_patch(right_wing)
            
        # Add creature's name
        if creature.age <= 2:  # Only show names for first few generations
            ax.text(x, y - size * 1.5, creature.name, 
                   fontsize=8, ha='center', color='black',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='white', alpha=0.7))
            
    def draw_connections(self, ax, creature: FractalCreature, alpha: float = 1.0):
        """Draw playful connections between friends."""
        x1, y1 = creature.get_position()
        
        for friend in creature.friends:
            x2, y2 = friend.get_position()
            
            # Create a wavy connection
            t = np.linspace(0, 1, 20)
            wave_amplitude = 0.1 * creature.size
            wave_freq = 3 + creature.age
            
            # Bezier curve with wave
            ctrl_x = (x1 + x2) / 2 + wave_amplitude * np.sin(self.time * 2)
            ctrl_y = (y1 + y2) / 2 + wave_amplitude * np.cos(self.time * 2)
            
            curve_x = (1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2
            curve_y = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2
            
            # Add wave
            perp_x = -(y2 - y1) / np.linalg.norm([x2-x1, y2-y1])
            perp_y = (x2 - x1) / np.linalg.norm([x2-x1, y2-y1])
            
            wave = wave_amplitude * np.sin(t * wave_freq * np.pi + self.time * 3)
            curve_x += perp_x * wave
            curve_y += perp_y * wave
            
            if self.rainbow_mode:
                color = self.create_rainbow_color(self.time * 0.2 + creature.age * 0.1)
            else:
                color = creature.color
                
            ax.plot(curve_x, curve_y, color=color, alpha=alpha*0.5, 
                   linewidth=2, linestyle='-')
            
    def animate_frame(self, ax):
        """Render one frame of the animation."""
        ax.clear()
        
        # Set up the playground background
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        
        # Create a fun background
        if self.rainbow_mode:
            # Gradient background
            gradient = np.linspace(0, 1, 256).reshape(256, 1)
            gradient = np.hstack((gradient, gradient))
            
            extent = [-10, 10, -10, 10]
            ax.imshow(gradient.T, extent=extent, aspect='auto', 
                     cmap='rainbow', alpha=0.2)
        else:
            ax.set_facecolor('#E6F3FF')  # Light blue sky
            
        # Remove axes for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Update all creatures
        dt = 0.05 * self.animation_speed
        for creature in self.creatures:
            creature.update(dt)
            
        # Draw connections first (behind creatures)
        for creature in self.creatures:
            self.draw_connections(ax, creature, alpha=0.6)
            
        # Draw all creatures
        for creature in self.creatures:
            self.draw_creature_shape(ax, creature)
            
        # Add title with animation info
        title = f"🌟 Fractal Friends Playground 🌟\n"
        title += f"Generation: {self.generation_count} | Friends: {len(self.creatures)}"
        ax.set_title(title, fontsize=16, fontweight='bold', color='purple')
        
        # Add instructions
        instructions = "Watch the fractals grow and play!"
        ax.text(0, -9.5, instructions, ha='center', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='yellow', alpha=0.7))
        
    def grow_generation(self):
        """Grow one generation of fractals."""
        if self.generation_count >= self.max_generations:
            return
            
        current_generation = [c for c in self.creatures if c.age == self.generation_count]
        new_creatures = []
        
        for creature in current_generation:
            children = self.grow_fractal_family(creature)
            new_creatures.extend(children)
            
        self.creatures.extend(new_creatures)
        self.generation_count += 1
        
    def update(self, frame):
        """Update function for animation."""
        self.time = frame * 0.05
        
        # Grow fractals periodically
        if frame % 30 == 0 and frame > 0:
            self.grow_generation()
            
        # Reset if we've grown all generations
        if frame % 300 == 0 and frame > 0:
            self.setup_playground()
            self.generation_count = 0
            
    def create_animation(self, fig=None, ax=None):
        """Create the animated playground."""
        if fig is None:
            fig, ax = plt.subplots(figsize=(12, 9))
            
        def animate(frame):
            self.update(frame)
            self.animate_frame(ax)
            return ax.patches + ax.lines + ax.texts
            
        anim = animation.FuncAnimation(
            fig, animate, interval=50, blit=False, frames=600
        )
        
        return fig, anim


class FractalStoryMode:
    """
    Story mode where fractals tell a story as they grow.
    Perfect for educational settings.
    """
    
    def __init__(self):
        self.playground = FractalPlayground()
        self.story_chapters = [
            {
                'title': "The Lonely Star",
                'text': "Once upon a time, there was a lonely star named Sparkle...",
                'action': 'create_parent'
            },
            {
                'title': "Finding Friends",
                'text': "Sparkle wished for friends to play with...",
                'action': 'grow_first_generation'
            },
            {
                'title': "The Fractal Family",
                'text': "Soon, Sparkle's friends had friends of their own!",
                'action': 'grow_second_generation'
            },
            {
                'title': "The Big Party",
                'text': "Everyone started dancing together!",
                'action': 'dance_party'
            },
            {
                'title': "Rainbow Celebration",
                'text': "The whole family turned into a rainbow!",
                'action': 'rainbow_mode'
            }
        ]
        self.current_chapter = 0
        
    def next_chapter(self):
        """Move to the next chapter of the story."""
        if self.current_chapter < len(self.story_chapters):
            chapter = self.story_chapters[self.current_chapter]
            
            if chapter['action'] == 'create_parent':
                self.playground.setup_playground()
            elif chapter['action'] == 'grow_first_generation':
                self.playground.grow_generation()
            elif chapter['action'] == 'grow_second_generation':
                self.playground.grow_generation()
            elif chapter['action'] == 'dance_party':
                for creature in self.playground.creatures:
                    creature.mood = 'dancing'
                self.playground.dance_mode = True
            elif chapter['action'] == 'rainbow_mode':
                self.playground.rainbow_mode = True
                
            self.current_chapter += 1
            return chapter
        return None


def create_interactive_playground():
    """
    Create an interactive playground with keyboard controls.
    Perfect for kids to explore fractals.
    """
    
    playground = FractalPlayground()
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Keyboard event handler
    def on_key(event):
        if event.key == ' ':  # Spacebar grows fractals
            playground.grow_generation()
        elif event.key == 'r':  # R for rainbow
            playground.rainbow_mode = not playground.rainbow_mode
        elif event.key == 'd':  # D for dance
            playground.dance_mode = not playground.dance_mode
            for creature in playground.creatures:
                creature.mood = 'dancing' if playground.dance_mode else 'happy'
        elif event.key == 'c':  # C to change colors
            palettes = list(playground.color_palettes.keys())
            current_idx = palettes.index(playground.current_palette)
            playground.current_palette = palettes[(current_idx + 1) % len(palettes)]
        elif event.key == 'n':  # N for new/reset
            playground.setup_playground()
            playground.generation_count = 0
        elif event.key == '+':  # Speed up
            playground.animation_speed = min(3.0, playground.animation_speed + 0.2)
        elif event.key == '-':  # Slow down
            playground.animation_speed = max(0.2, playground.animation_speed - 0.2)
            
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Create animation
    fig, anim = playground.create_animation(fig, ax)
    
    # Add control instructions
    fig.text(0.5, 0.02, 
            "Controls: SPACE=Grow | R=Rainbow | D=Dance | C=Colors | N=New | +/- =Speed",
            ha='center', fontsize=10, color='blue',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    return fig, anim, playground


def create_educational_demo():
    """
    Create an educational demo that explains fractals to kids.
    """
    
    fig = plt.figure(figsize=(16, 10))
    
    # Create subplots for different concepts
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. What is a fractal?
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("What is a Fractal?", fontsize=12, fontweight='bold')
    ax1.text(0.5, 0.5, 
            "A fractal is a pattern that\nrepeats itself at different sizes!\n\n"
            "Like a tree with branches,\nand branches on branches!",
            ha='center', va='center', fontsize=10,
            transform=ax1.transAxes)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # 2. Simple fractal tree
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Fractal Tree", fontsize=12, fontweight='bold')
    draw_simple_tree(ax2, 0, 0, 90, 5, 3)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-1, 4)
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # 3. Fractal snowflake
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("Fractal Snowflake", fontsize=12, fontweight='bold')
    draw_koch_snowflake(ax3, 3)
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 2)
    ax3.set_aspect('equal')
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    # 4-6. Growing fractal animation frames
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        ax.set_title(f"Generation {i+1}", fontsize=12, fontweight='bold')
        
        # Create a simple fractal at this generation
        playground = FractalPlayground()
        playground.max_generations = i + 1
        
        for _ in range(i + 1):
            playground.grow_generation()
            
        playground.animate_frame(ax)
        
    fig.suptitle("🌈 Learn About Fractals! 🌈", fontsize=16, fontweight='bold')
    
    return fig


def draw_simple_tree(ax, x, y, angle, length, depth):
    """Draw a simple fractal tree for educational purposes."""
    if depth == 0:
        return
        
    # Calculate end point
    end_x = x + length * np.cos(np.radians(angle))
    end_y = y + length * np.sin(np.radians(angle))
    
    # Draw branch
    ax.plot([x, end_x], [y, end_y], 'brown', linewidth=depth)
    
    # Draw child branches
    new_length = length * 0.7
    draw_simple_tree(ax, end_x, end_y, angle - 25, new_length, depth - 1)
    draw_simple_tree(ax, end_x, end_y, angle + 25, new_length, depth - 1)
    

def draw_koch_snowflake(ax, order):
    """Draw a Koch snowflake for educational purposes."""
    def koch_line(start, end, order):
        if order == 0:
            return [start, end]
        
        # Divide line into thirds
        x1, y1 = start
        x2, y2 = end
        
        dx = x2 - x1
        dy = y2 - y1
        
        p1 = (x1 + dx/3, y1 + dy/3)
        p2 = (x1 + 2*dx/3, y1 + 2*dy/3)
        
        # Calculate peak point
        angle = np.arctan2(dy, dx) + np.pi/3
        length = np.sqrt((dx/3)**2 + (dy/3)**2)
        peak = (p1[0] + length * np.cos(angle),
                p1[1] + length * np.sin(angle))
        
        # Recursively create smaller segments
        points = []
        points.extend(koch_line(start, p1, order-1)[:-1])
        points.extend(koch_line(p1, peak, order-1)[:-1])
        points.extend(koch_line(peak, p2, order-1)[:-1])
        points.extend(koch_line(p2, end, order-1))
        
        return points
    
    # Create initial triangle
    angles = [0, 120, 240]
    vertices = [(np.cos(np.radians(a)), np.sin(np.radians(a))) 
                for a in angles]
    
    # Generate Koch curves for each side
    all_points = []
    for i in range(3):
        start = vertices[i]
        end = vertices[(i+1) % 3]
        points = koch_line(start, end, order)
        all_points.extend(points[:-1])
    all_points.append(all_points[0])  # Close the shape
    
    # Draw the snowflake
    xs, ys = zip(*all_points)
    ax.fill(xs, ys, color='lightblue', edgecolor='blue', linewidth=1)


def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kid-Friendly Fractal Playground')
    parser.add_argument('--mode', choices=['playground', 'story', 'education', 'all'],
                       default='playground',
                       help='Which mode to run')
    parser.add_argument('--save', action='store_true',
                       help='Save animations as GIF')
    
    args = parser.parse_args()
    
    print("🌟 Welcome to the Fractal Playground! 🌟")
    print("=" * 50)
    
    if args.mode == 'playground' or args.mode == 'all':
        print("\n🎮 Interactive Playground Mode")
        print("Use keyboard controls to play with fractals!")
        fig, anim, playground = create_interactive_playground()
        
        if args.save:
            print("Saving playground animation...")
            anim.save('outputs/fractal_playground.gif', writer='pillow', fps=20)
            print("✓ Saved to outputs/fractal_playground.gif")
        
    if args.mode == 'story' or args.mode == 'all':
        print("\n📖 Story Mode")
        story = FractalStoryMode()
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Animate through the story
        def story_update(frame):
            if frame % 60 == 0:  # New chapter every 3 seconds
                chapter = story.next_chapter()
                if chapter:
                    print(f"Chapter: {chapter['title']}")
                    print(f"  {chapter['text']}")
            
            story.playground.update(frame)
            story.playground.animate_frame(ax)
            
        anim = animation.FuncAnimation(fig, story_update, frames=300,
                                     interval=50, blit=False)
        
        if args.save:
            print("Saving story animation...")
            anim.save('outputs/fractal_story.gif', writer='pillow', fps=20)
            print("✓ Saved to outputs/fractal_story.gif")
    
    if args.mode == 'education' or args.mode == 'all':
        print("\n🎓 Educational Mode")
        print("Learn about fractals with examples!")
        edu_fig = create_educational_demo()
        
        if args.save:
            print("Saving educational diagram...")
            edu_fig.savefig('outputs/fractal_education.png', dpi=150, bbox_inches='tight')
            print("✓ Saved to outputs/fractal_education.png")
    
    plt.show()
    
    print("\n" + "=" * 50)
    print("Thanks for playing with fractals! 🌈")


if __name__ == '__main__':
    main()