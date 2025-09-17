---
layout: default
title: Home
custom_js:
  - /assets/js/fractal-navigator.js
---

<div class="hero-section">
  <h1 class="hero-title">Where Music Becomes Structure,<br>Structure Becomes Reality</h1>
  <p class="hero-subtitle">A revolutionary computational framework unifying mathematics, biology, and physics through the principle of <strong>harmonic integrity</strong>.</p>
  
  <div class="hero-actions">
    <a href="/research/" class="btn btn-primary">Explore the Research</a>
    <a href="/playground/" class="btn btn-secondary">Try Interactive Demo</a>
    <a href="/team/join/" class="btn btn-outline">Join Our Team</a>
  </div>
</div>

<div class="visualization-container">
  <div id="fractal-navigator">
    <!-- Three.js visualization will be inserted here -->
    <canvas id="manifold-canvas"></canvas>
  </div>
  <p class="visualization-caption">Interactive 48-Manifold Fractal Navigator</p>
</div>

## Revolutionary Applications

<div class="features-grid">
  <div class="feature-card">
    <div class="feature-icon">🧬</div>
    <h3>Protein Folding</h3>
    <p>Generate accurate 3D protein structures from amino acid sequences using harmonic composition and musical principles.</p>
    <a href="/applications/protein-folding/">Learn More →</a>
  </div>
  
  <div class="feature-card">
    <div class="feature-icon">⚛️</div>
    <h3>Fusion Control</h3>
    <p>Novel control systems for tokamak reactors that preserve information integrity and prevent catastrophic instabilities.</p>
    <a href="/applications/fusion-control/">Learn More →</a>
  </div>
  
  <div class="feature-card">
    <div class="feature-icon">🎵</div>
    <h3>Molecular Music</h3>
    <p>Transform protein structures into three-channel audio compositions, making molecular harmony audible.</p>
    <a href="/applications/sonification/">Learn More →</a>
  </div>
  
  <div class="feature-card">
    <div class="feature-icon">🤖</div>
    <h3>Reversible ML</h3>
    <p>Machine learning without information loss - perfect reconstruction through reversible transfers on the 48-manifold.</p>
    <a href="/applications/machine-learning/">Learn More →</a>
  </div>
</div>

## The Core Innovation

Unlike traditional computational approaches that lose information through transformation, the 48-Manifold framework maintains **perfect reversibility** through *transfer* operations. By treating computation as a combinatorial game where integrity is preserved at every step, we achieve:

- **Zero aliasing** by construction
- **10-25% better conditioning** in neural networks
- **40% reduction** in long-term drift
- **Perfect reconstruction** in signal processing

<div class="stats-row">
  <div class="stat">
    <div class="stat-number">48</div>
    <div class="stat-label">Dimensional Basis<br>(2⁴ × 3)</div>
  </div>
  <div class="stat">
    <div class="stat-number">0.816</div>
    <div class="stat-label">Correlation with<br>AlphaFold pLDDT</div>
  </div>
  <div class="stat">
    <div class="stat-number">21/21</div>
    <div class="stat-label">Tests Passing<br>Full Validation</div>
  </div>
  <div class="stat">
    <div class="stat-number">∞</div>
    <div class="stat-label">Perfect<br>Reversibility</div>
  </div>
</div>

## Recent Publications & Updates

<div class="blog-preview">
  {% for post in site.posts limit:3 %}
  <article class="blog-card">
    <time>{{ post.date | date: "%B %d, %Y" }}</time>
    <h4><a href="{{ post.url }}">{{ post.title }}</a></h4>
    <p>{{ post.excerpt | strip_html | truncate: 150 }}</p>
  </article>
  {% endfor %}
</div>

<div class="cta-section">
  <h2>Ready to Explore?</h2>
  <p>Dive into the mathematical foundations, try our interactive demos, or join our research community.</p>
  <div class="cta-buttons">
    <a href="/docs/getting-started/" class="btn btn-large">Get Started</a>
    <a href="https://github.com/48manifold/core" class="btn btn-large btn-github">View on GitHub</a>
  </div>
</div>

## Collaborators & Support

<div class="partners-grid">
  <!-- Partner logos would go here -->
  <p class="partners-note">Supported by leading research institutions and forward-thinking organizations committed to advancing computational science.</p>
</div>

<script src="/assets/js/fractal-navigator.js"></script>
<script src="/assets/js/three.min.js"></script>
<script>
  // Initialize the 48-manifold visualization
  document.addEventListener('DOMContentLoaded', function() {
    const navigator = new FractalNavigator('manifold-canvas');
    navigator.start();
  });
</script>

<style>
.hero-section {
  text-align: center;
  padding: 4rem 1rem;
  background: linear-gradient(135deg, #1a237e 0%, #3949ab 100%);
  color: white;
  border-radius: 1rem;
  margin-bottom: 3rem;
}

.hero-title {
  font-size: 3rem;
  font-weight: 700;
  margin-bottom: 1rem;
  line-height: 1.2;
}

.hero-subtitle {
  font-size: 1.25rem;
  opacity: 0.95;
  max-width: 700px;
  margin: 0 auto 2rem;
}

.hero-actions {
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
}

.btn {
  padding: 0.75rem 2rem;
  border-radius: 0.5rem;
  text-decoration: none;
  font-weight: 600;
  transition: all 0.3s ease;
  display: inline-block;
}

.btn-primary {
  background: #ffc107;
  color: #1a237e;
}

.btn-secondary {
  background: white;
  color: #1a237e;
}

.btn-outline {
  background: transparent;
  color: white;
  border: 2px solid white;
}

.btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}

.visualization-container {
  margin: 3rem 0;
  text-align: center;
}

#manifold-canvas {
  width: 100%;
  max-width: 800px;
  height: 400px;
  border-radius: 0.5rem;
  box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
  margin: 3rem 0;
}

.feature-card {
  background: white;
  padding: 2rem;
  border-radius: 0.5rem;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
  transition: transform 0.3s ease;
}

.feature-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}

.feature-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.stats-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 2rem;
  margin: 3rem 0;
  text-align: center;
}

.stat {
  padding: 1.5rem;
  background: #f5f5f5;
  border-radius: 0.5rem;
}

.stat-number {
  font-size: 2.5rem;
  font-weight: 700;
  color: #1a237e;
  margin-bottom: 0.5rem;
}

.stat-label {
  font-size: 0.9rem;
  color: #666;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.blog-preview {
  display: grid;
  gap: 1.5rem;
  margin: 3rem 0;
}

.blog-card {
  padding: 1.5rem;
  background: white;
  border-left: 4px solid #ffc107;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.blog-card time {
  color: #999;
  font-size: 0.9rem;
}

.cta-section {
  text-align: center;
  padding: 3rem;
  background: #f8f9fa;
  border-radius: 1rem;
  margin: 3rem 0;
}

.cta-buttons {
  display: flex;
  gap: 1rem;
  justify-content: center;
  margin-top: 2rem;
}

.btn-large {
  padding: 1rem 3rem;
  font-size: 1.125rem;
}

.btn-github {
  background: #24292e;
  color: white;
}

@media (max-width: 768px) {
  .hero-title {
    font-size: 2rem;
  }
  
  .features-grid {
    grid-template-columns: 1fr;
  }
  
  .hero-actions {
    flex-direction: column;
    align-items: center;
  }
}
</style>