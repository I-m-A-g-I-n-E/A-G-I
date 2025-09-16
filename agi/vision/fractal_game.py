#!/usr/bin/env python3
"""
Kid-friendly fractal game with real-time rendering engine.
Designed for ages 5-10 with colorful sprites, sounds, and fun interactions.
"""

import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

# Try importing pygame for real-time rendering
try:
    import pygame
    import pygame.gfxdraw
    from pygame import Surface, Rect, Color
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("🎮 Please install pygame for the full experience!")
    print("Run: pip install pygame")

# Fallback to tkinter if pygame not available
try:
    import tkinter as tk
    from tkinter import Canvas
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False


@dataclass
class FractalSprite:
    """A fun, animated sprite for our fractal creatures."""
    x: float
    y: float
    vx: float = 0  # velocity
    vy: float = 0
    size: float = 32
    color: Tuple[int, int, int] = (255, 100, 200)
    emoji: str = "😊"
    rotation: float = 0
    scale: float = 1.0
    glow: float = 0
    bounce: float = 0
    parent: Optional['FractalSprite'] = None
    children: List['FractalSprite'] = None
    generation: int = 0
    personality: str = "happy"  # happy, silly, sleepy, excited, curious
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        # Animation parameters
        self.wobble = random.random() * math.pi * 2
        self.pulse = 0
        self.sparkles = []
        
    def update(self, dt: float):
        """Update sprite animation and physics."""
        # Wobble animation
        self.wobble += dt * 3
        self.x += math.sin(self.wobble) * 0.5
        self.y += math.cos(self.wobble * 0.7) * 0.3
        
        # Pulse effect
        self.pulse += dt * 4
        self.scale = 1.0 + math.sin(self.pulse) * 0.1
        
        # Rotation for excited sprites
        if self.personality == "excited":
            self.rotation += dt * 200
        elif self.personality == "silly":
            self.rotation = math.sin(self.wobble) * 30
            
        # Bounce effect
        if self.personality == "happy":
            self.bounce = abs(math.sin(self.wobble * 2)) * 10
            
        # Update sparkles
        self.sparkles = [(x + random.random() * 2 - 1, 
                         y - dt * 50,
                         life - dt)
                         for x, y, life in self.sparkles if life > 0]
        
        # Add new sparkles occasionally
        if random.random() < 0.02:
            self.sparkles.append((0, 0, 1.0))


class KidFractalEngine:
    """
    A fun, game-like rendering engine for fractal exploration.
    Built specifically for children with bright colors and playful animations.
    """
    
    def __init__(self, width: int = 1024, height: int = 768):
        self.width = width
        self.height = height
        self.running = True
        self.clock = None
        self.screen = None
        self.sprites: List[FractalSprite] = []
        self.particles = []
        self.time = 0
        self.score = 0
        self.level = 1
        
        # Fun emoji sets for different themes
        self.emoji_themes = {
            'animals': ['🐶', '🐱', '🐭', '🐹', '🐰', '🦊', '🐻', '🐼', '🐨', '🐯'],
            'space': ['🌟', '✨', '💫', '⭐', '🌙', '☄️', '🚀', '🛸', '👽', '🌎'],
            'nature': ['🌸', '🌺', '🌻', '🌹', '🌷', '🌲', '🌳', '🍄', '🦋', '🐝'],
            'food': ['🍎', '🍊', '🍋', '🍓', '🍇', '🍉', '🍑', '🍒', '🧁', '🍭'],
            'ocean': ['🐠', '🐟', '🦈', '🐙', '🦑', '🦐', '🦀', '🐚', '🐋', '🐬'],
            'fantasy': ['🦄', '🐲', '🧚', '🧙', '👑', '💎', '🔮', '⚡', '🌈', '✨']
        }
        self.current_theme = 'animals'
        
        # Color palettes
        self.color_schemes = {
            'rainbow': [(255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0), 
                       (0, 0, 255), (75, 0, 130), (148, 0, 211)],
            'candy': [(255, 182, 193), (255, 192, 203), (255, 105, 180),
                     (255, 20, 147), (219, 112, 147), (238, 130, 238)],
            'ocean': [(0, 119, 190), (0, 180, 216), (72, 202, 228),
                     (144, 224, 239), (173, 232, 244), (202, 240, 248)],
            'sunset': [(255, 94, 77), (255, 154, 0), (237, 117, 57),
                      (255, 206, 84), (255, 220, 128), (255, 237, 185)],
            'forest': [(34, 139, 34), (50, 205, 50), (124, 252, 0),
                      (173, 255, 47), (154, 205, 50), (144, 238, 144)]
        }
        self.color_scheme = 'rainbow'
        
        # Sound settings
        self.sound_enabled = True
        self.music_volume = 0.5
        self.sfx_volume = 0.7
        
        # Game state
        self.game_mode = 'sandbox'  # sandbox, adventure, learn
        self.selected_sprite = None
        self.camera_x = 0
        self.camera_y = 0
        self.zoom = 1.0
        
        # Initialize rendering engine
        if PYGAME_AVAILABLE:
            self.init_pygame()
        elif TKINTER_AVAILABLE:
            self.init_tkinter()
            
    def init_pygame(self):
        """Initialize pygame rendering engine."""
        pygame.init()
        
        # Create window
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("🌟 Fractal Friends Adventure 🌟")
        
        # Load fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # Create emoji font (system emoji support)
        try:
            # Try to use a system font that supports emoji
            self.emoji_font = pygame.font.SysFont('segoeuiemoji', 48)
        except:
            self.emoji_font = self.font_large
            
        # Clock for FPS
        self.clock = pygame.time.Clock()
        
        # Initialize sound system
        if self.sound_enabled:
            try:
                pygame.mixer.init()
                self.load_sounds()
            except:
                self.sound_enabled = False
                
        # Create initial sprite
        self.create_starter_fractal()
        
    def init_tkinter(self):
        """Fallback to tkinter for basic rendering."""
        self.root = tk.Tk()
        self.root.title("🌟 Fractal Friends (Basic Mode) 🌟")
        
        self.canvas = Canvas(self.root, width=self.width, height=self.height, bg='lightblue')
        self.canvas.pack()
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        
        # Create initial fractal
        self.create_starter_fractal()
        
    def load_sounds(self):
        """Generate fun sound effects."""
        # Since we can't load files, we'll generate sounds
        self.sounds = {}
        
        # Generate simple tones for sound effects
        sample_rate = 22050
        duration = 0.2
        
        # Pop sound
        freq = 800
        samples = int(sample_rate * duration)
        waves = np.sin(2 * np.pi * freq * np.linspace(0, duration, samples))
        waves = (waves * 32767).astype(np.int16)
        waves = np.repeat(waves.reshape(-1, 1), 2, axis=1)  # Stereo
        
        try:
            self.sounds['pop'] = pygame.sndarray.make_sound(waves)
        except:
            pass
            
    def create_starter_fractal(self):
        """Create the initial fractal sprite."""
        starter = FractalSprite(
            x=self.width // 2,
            y=self.height // 2,
            size=64,
            color=self.get_generation_color(0),
            emoji=self.get_random_emoji(),
            personality="happy"
        )
        self.sprites = [starter]
        
    def get_random_emoji(self) -> str:
        """Get a random emoji from current theme."""
        theme_emojis = self.emoji_themes[self.current_theme]
        return random.choice(theme_emojis)
        
    def get_generation_color(self, generation: int) -> Tuple[int, int, int]:
        """Get color for a generation."""
        colors = self.color_schemes[self.color_scheme]
        return colors[generation % len(colors)]
        
    def spawn_children(self, parent: FractalSprite):
        """Create child fractals from parent."""
        if parent.generation >= 5:  # Limit depth for performance
            return
            
        num_children = 2 if parent.generation % 2 == 0 else 3
        
        for i in range(num_children):
            angle = (i * 2 * math.pi / num_children) + self.time
            distance = parent.size * 2
            
            child_x = parent.x + math.cos(angle) * distance
            child_y = parent.y + math.sin(angle) * distance
            
            # Random personality
            personalities = ["happy", "silly", "excited", "curious", "sleepy"]
            
            child = FractalSprite(
                x=child_x,
                y=child_y,
                size=parent.size * 0.7,
                color=self.get_generation_color(parent.generation + 1),
                emoji=self.get_random_emoji(),
                generation=parent.generation + 1,
                personality=random.choice(personalities),
                parent=parent
            )
            
            parent.children.append(child)
            self.sprites.append(child)
            
            # Create particle burst
            self.create_particle_burst(child_x, child_y, child.color)
            
            # Play sound
            if self.sound_enabled and 'pop' in self.sounds:
                self.sounds['pop'].play()
                
            # Increase score
            self.score += 10 * (parent.generation + 1)
            
    def create_particle_burst(self, x: float, y: float, color: Tuple[int, int, int]):
        """Create a burst of particles for visual effects."""
        for _ in range(20):
            self.particles.append({
                'x': x,
                'y': y,
                'vx': random.uniform(-5, 5),
                'vy': random.uniform(-5, 5),
                'color': color,
                'life': 1.0,
                'size': random.uniform(2, 8)
            })
            
    def update_particles(self, dt: float):
        """Update particle system."""
        updated_particles = []
        for p in self.particles:
            p['x'] += p['vx']
            p['y'] += p['vy']
            p['vy'] += 0.5  # gravity
            p['life'] -= dt * 2
            
            if p['life'] > 0:
                updated_particles.append(p)
                
        self.particles = updated_particles
        
    def draw_sprite_pygame(self, sprite: FractalSprite):
        """Draw a sprite using pygame."""
        # Transform to camera space
        screen_x = (sprite.x - self.camera_x) * self.zoom + self.width // 2
        screen_y = (sprite.y - self.camera_y) * self.zoom + self.height // 2
        size = sprite.size * sprite.scale * self.zoom
        
        # Draw glow effect
        if sprite.glow > 0:
            glow_surf = pygame.Surface((size * 3, size * 3), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*sprite.color, 50), 
                             (size * 1.5, size * 1.5), size * 1.5)
            self.screen.blit(glow_surf, (screen_x - size * 1.5, screen_y - size * 1.5))
            
        # Draw main circle
        pygame.draw.circle(self.screen, sprite.color, 
                          (int(screen_x), int(screen_y - sprite.bounce)), 
                          int(size))
        
        # Draw face
        eye_size = max(2, int(size * 0.15))
        eye_offset = size * 0.3
        
        # Eyes
        pygame.draw.circle(self.screen, (0, 0, 0),
                          (int(screen_x - eye_offset), int(screen_y - sprite.bounce - eye_offset)),
                          eye_size)
        pygame.draw.circle(self.screen, (0, 0, 0),
                          (int(screen_x + eye_offset), int(screen_y - sprite.bounce - eye_offset)),
                          eye_size)
        
        # Smile
        if sprite.personality == "happy":
            # Draw smile arc
            rect = pygame.Rect(screen_x - size * 0.5, screen_y - sprite.bounce - size * 0.2,
                              size, size * 0.6)
            pygame.draw.arc(self.screen, (0, 0, 0), rect, 0, math.pi, max(2, int(size * 0.1)))
        elif sprite.personality == "silly":
            # Tongue out
            pygame.draw.circle(self.screen, (255, 100, 100),
                             (int(screen_x), int(screen_y - sprite.bounce + size * 0.3)),
                             max(2, int(size * 0.2)))
        elif sprite.personality == "sleepy":
            # Closed eyes (lines)
            pygame.draw.line(self.screen, (0, 0, 0),
                           (screen_x - eye_offset - eye_size, screen_y - sprite.bounce - eye_offset),
                           (screen_x - eye_offset + eye_size, screen_y - sprite.bounce - eye_offset),
                           max(2, eye_size // 2))
            pygame.draw.line(self.screen, (0, 0, 0),
                           (screen_x + eye_offset - eye_size, screen_y - sprite.bounce - eye_offset),
                           (screen_x + eye_offset + eye_size, screen_y - sprite.bounce - eye_offset),
                           max(2, eye_size // 2))
            # Zzz
            if int(self.time * 2) % 2 == 0:
                text = self.font_small.render("z", True, (100, 100, 255))
                self.screen.blit(text, (screen_x + size, screen_y - sprite.bounce - size))
                
        # Draw sparkles
        for sx, sy, life in sprite.sparkles:
            spark_x = screen_x + sx * size
            spark_y = screen_y - sprite.bounce + sy * size
            spark_size = int(life * 5)
            pygame.draw.circle(self.screen, (255, 255, 100), 
                             (int(spark_x), int(spark_y)), spark_size)
                             
        # Draw emoji (if font supports it)
        try:
            emoji_text = self.emoji_font.render(sprite.emoji, True, (255, 255, 255))
            emoji_rect = emoji_text.get_rect(center=(screen_x, screen_y - sprite.bounce))
            self.screen.blit(emoji_text, emoji_rect)
        except:
            pass
            
    def draw_connections(self):
        """Draw connections between parent and child sprites."""
        for sprite in self.sprites:
            if sprite.parent:
                # Transform positions
                x1 = (sprite.parent.x - self.camera_x) * self.zoom + self.width // 2
                y1 = (sprite.parent.y - self.camera_y) * self.zoom + self.height // 2
                x2 = (sprite.x - self.camera_x) * self.zoom + self.width // 2
                y2 = (sprite.y - self.camera_y) * self.zoom + self.height // 2
                
                # Draw wavy line
                points = []
                for t in np.linspace(0, 1, 20):
                    wave = math.sin(t * math.pi * 3 + self.time * 2) * 10
                    px = x1 + (x2 - x1) * t
                    py = y1 + (y2 - y1) * t + wave
                    points.append((px, py))
                    
                if len(points) > 1:
                    pygame.draw.lines(self.screen, sprite.color, False, points, 2)
                    
    def draw_particles(self):
        """Draw particle effects."""
        for p in self.particles:
            screen_x = (p['x'] - self.camera_x) * self.zoom + self.width // 2
            screen_y = (p['y'] - self.camera_y) * self.zoom + self.height // 2
            size = p['size'] * p['life'] * self.zoom
            
            if size > 0:
                color = (*p['color'], int(255 * p['life']))
                pygame.gfxdraw.filled_circle(self.screen, int(screen_x), int(screen_y),
                                            int(size), color)
                
    def draw_ui(self):
        """Draw user interface elements."""
        # Score
        score_text = self.font_large.render(f"Score: {self.score}", True, (255, 255, 255))
        score_shadow = self.font_large.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(score_shadow, (52, 22))
        self.screen.blit(score_text, (50, 20))
        
        # Level
        level_text = self.font_medium.render(f"Level {self.level}", True, (255, 255, 255))
        self.screen.blit(level_text, (50, 70))
        
        # Sprite count
        count_text = self.font_small.render(f"Fractals: {len(self.sprites)}", True, (255, 255, 255))
        self.screen.blit(count_text, (50, 110))
        
        # Instructions
        instructions = [
            "🖱️ Click on fractals to grow them!",
            "⌨️ SPACE: Random growth | R: Reset | T: Change theme",
            "⌨️ C: Change colors | M: Toggle music | ESC: Exit"
        ]
        
        y_offset = self.height - 100
        for instruction in instructions:
            inst_text = self.font_small.render(instruction, True, (255, 255, 255))
            inst_shadow = self.font_small.render(instruction, True, (0, 0, 0))
            self.screen.blit(inst_shadow, (self.width // 2 - inst_text.get_width() // 2 + 1, y_offset + 1))
            self.screen.blit(inst_text, (self.width // 2 - inst_text.get_width() // 2, y_offset))
            y_offset += 25
            
        # Theme indicator
        theme_text = self.font_small.render(f"Theme: {self.current_theme.title()}", True, (255, 255, 255))
        self.screen.blit(theme_text, (self.width - 200, 20))
        
    def handle_click(self, x: int, y: int):
        """Handle mouse click."""
        # Convert to world coordinates
        world_x = (x - self.width // 2) / self.zoom + self.camera_x
        world_y = (y - self.height // 2) / self.zoom + self.camera_y
        
        # Find clicked sprite
        for sprite in self.sprites:
            dx = sprite.x - world_x
            dy = sprite.y - world_y
            if math.sqrt(dx*dx + dy*dy) < sprite.size:
                # Spawn children for this sprite
                self.spawn_children(sprite)
                sprite.glow = 1.0
                break
        else:
            # No sprite clicked, create a new one
            new_sprite = FractalSprite(
                x=world_x,
                y=world_y,
                size=32,
                color=self.get_generation_color(0),
                emoji=self.get_random_emoji(),
                personality=random.choice(["happy", "silly", "excited", "curious"])
            )
            self.sprites.append(new_sprite)
            self.create_particle_burst(world_x, world_y, new_sprite.color)
            
    def run(self):
        """Main game loop."""
        if not PYGAME_AVAILABLE:
            print("Pygame not available! Please install pygame for the full experience.")
            return
            
        running = True
        dt = 0
        
        while running:
            # Timing
            dt = self.clock.tick(60) / 1000.0  # 60 FPS
            self.time += dt
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # Random growth
                        if self.sprites:
                            random_sprite = random.choice(self.sprites)
                            self.spawn_children(random_sprite)
                    elif event.key == pygame.K_r:
                        # Reset
                        self.sprites.clear()
                        self.particles.clear()
                        self.score = 0
                        self.create_starter_fractal()
                    elif event.key == pygame.K_t:
                        # Change theme
                        themes = list(self.emoji_themes.keys())
                        idx = themes.index(self.current_theme)
                        self.current_theme = themes[(idx + 1) % len(themes)]
                    elif event.key == pygame.K_c:
                        # Change colors
                        schemes = list(self.color_schemes.keys())
                        idx = schemes.index(self.color_scheme)
                        self.color_scheme = schemes[(idx + 1) % len(schemes)]
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos[0], event.pos[1])
                    elif event.button == 4:  # Scroll up
                        self.zoom *= 1.1
                    elif event.button == 5:  # Scroll down
                        self.zoom /= 1.1
                        
            # Update camera (follow center of mass)
            if self.sprites:
                avg_x = sum(s.x for s in self.sprites) / len(self.sprites)
                avg_y = sum(s.y for s in self.sprites) / len(self.sprites)
                self.camera_x += (avg_x - self.camera_x) * dt * 2
                self.camera_y += (avg_y - self.camera_y) * dt * 2
                
            # Update sprites
            for sprite in self.sprites:
                sprite.update(dt)
                sprite.glow = max(0, sprite.glow - dt)
                
            # Update particles
            self.update_particles(dt)
            
            # Level up
            if self.score > self.level * 1000:
                self.level += 1
                self.create_particle_burst(self.width // 2, self.height // 2,
                                         (255, 255, 0))
                
            # Drawing
            # Clear screen first
            self.screen.fill((135, 206, 235))  # Sky blue background
            
            # Background gradient
            for y in range(0, self.height, 20):
                progress = y / self.height
                # Calculate gradient from light to darker blue
                r = int(135 - progress * 50)
                g = int(206 - progress * 50) 
                b = int(235 - progress * 30)
                # Ensure values stay in valid range
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                pygame.draw.rect(self.screen, (r, g, b), (0, y, self.width, 20))
                
            # Draw connections
            self.draw_connections()
            
            # Draw particles
            self.draw_particles()
            
            # Draw sprites
            for sprite in sorted(self.sprites, key=lambda s: s.generation):
                self.draw_sprite_pygame(sprite)
                
            # Draw UI
            self.draw_ui()
            
            # Update display
            pygame.display.flip()
            
        pygame.quit()


def main():
    """Main entry point."""
    print("🌟 Welcome to Fractal Friends Game! 🌟")
    print("=" * 50)
    print("A fun, interactive fractal explorer for kids!")
    print("\nControls:")
    print("  🖱️ Click on creatures to make them grow friends!")
    print("  ⌨️ SPACE: Random growth")
    print("  ⌨️ R: Reset everything")
    print("  ⌨️ T: Change theme (animals, space, nature, etc.)")
    print("  ⌨️ C: Change color scheme")
    print("  🖱️ Scroll: Zoom in/out")
    print("=" * 50)
    
    if not PYGAME_AVAILABLE:
        print("\n⚠️ Pygame is not installed!")
        print("For the best experience, install pygame:")
        print("  pip install pygame")
        print("\nFalling back to basic mode...")
        
        if TKINTER_AVAILABLE:
            # Run tkinter version
            engine = KidFractalEngine()
            engine.root.mainloop()
        else:
            print("No GUI library available. Please install pygame or tkinter.")
    else:
        # Run pygame version
        engine = KidFractalEngine()
        engine.run()
        
    print("\nThanks for playing! 🎮✨")


if __name__ == '__main__':
    main()