/**
 * Main JavaScript for {wtf²B•2^4*3} Manifold Site
 * A-G-I.space
 */

// Get manifold configuration from data attributes or defaults
const MANIFOLD_CONFIG = {
    dimensions: 48,
    factorization: [3, 2, 2, 2, 2], // 48 = 3 * 2^4
    name: '{wtf²B•2^4*3} Manifold',
    domain: 'A-G-I.space'
};

// Utility functions
const utils = {
    // Throttle function calls
    throttle(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    // Smooth scroll to element
    smoothScroll(target) {
        const element = document.querySelector(target);
        if (element) {
            element.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    },
    
    // Copy text to clipboard
    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch (err) {
            console.error('Failed to copy:', err);
            return false;
        }
    }
};

// Fractal Navigator (placeholder for Three.js implementation)
class FractalNavigator {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        
        this.dimensions = MANIFOLD_CONFIG.dimensions;
        this.isAnimating = false;
        this.init();
    }
    
    init() {
        // Check if Three.js is loaded
        if (typeof THREE === 'undefined') {
            console.log('Three.js not loaded, skipping fractal navigator');
            return;
        }
        
        // Initialize Three.js scene
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(
            75,
            this.canvas.width / this.canvas.height,
            0.1,
            1000
        );
        
        this.renderer = new THREE.WebGLRenderer({ 
            canvas: this.canvas,
            antialias: true,
            alpha: true
        });
        
        this.renderer.setSize(this.canvas.width, this.canvas.height);
        this.camera.position.z = 5;
        
        this.createManifold();
        this.animate();
    }
    
    createManifold() {
        // Create a visual representation of the 48-manifold
        const geometry = new THREE.IcosahedronGeometry(2, 2);
        const material = new THREE.MeshBasicMaterial({
            color: 0x1a237e,
            wireframe: true,
            transparent: true,
            opacity: 0.8
        });
        
        this.manifold = new THREE.Mesh(geometry, material);
        this.scene.add(this.manifold);
        
        // Add particles for dimensions
        this.addParticles();
    }
    
    addParticles() {
        const particleCount = this.dimensions;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        
        for (let i = 0; i < particleCount; i++) {
            const theta = (i / particleCount) * Math.PI * 2;
            const phi = Math.acos(2 * (i / particleCount) - 1);
            const radius = 3;
            
            positions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
            positions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
            positions[i * 3 + 2] = radius * Math.cos(phi);
        }
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        const material = new THREE.PointsMaterial({
            color: 0xffc107,
            size: 0.1,
            transparent: true,
            opacity: 0.8
        });
        
        this.particles = new THREE.Points(geometry, material);
        this.scene.add(this.particles);
    }
    
    animate() {
        if (!this.isAnimating) {
            this.isAnimating = true;
            this.render();
        }
    }
    
    render() {
        requestAnimationFrame(() => this.render());
        
        if (this.manifold) {
            this.manifold.rotation.x += 0.005;
            this.manifold.rotation.y += 0.003;
        }
        
        if (this.particles) {
            this.particles.rotation.y -= 0.002;
        }
        
        this.renderer.render(this.scene, this.camera);
    }
    
    stop() {
        this.isAnimating = false;
    }
}

// Deepnote Embed Manager
class DeepnoteManager {
    constructor() {
        this.workspace = 'https://deepnote.com/@48-manifold';
        this.notebooks = {
            quickstart: 'quickstart-tutorial-uuid',
            protein: 'protein-composer-uuid',
            fractal: 'fractal-navigator-uuid',
            hand: 'hand-tensor-uuid'
        };
    }
    
    embedNotebook(containerId, notebookKey, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        const defaultOptions = {
            height: 600,
            hideCode: true,
            theme: 'light'
        };
        
        const settings = { ...defaultOptions, ...options };
        const notebookId = this.notebooks[notebookKey] || notebookKey;
        
        const iframe = document.createElement('iframe');
        iframe.src = `https://deepnote.com/embed/48-manifold/${notebookId}?hideCode=${settings.hideCode}`;
        iframe.width = '100%';
        iframe.height = settings.height;
        iframe.frameBorder = '0';
        iframe.style.border = 'none';
        iframe.style.borderRadius = '8px';
        
        // Add loading placeholder
        container.innerHTML = `
            <div class="deepnote-placeholder">
                <div class="spinner"></div>
                <p>Loading interactive notebook...</p>
            </div>
        `;
        
        iframe.onload = () => {
            container.innerHTML = '';
            container.appendChild(iframe);
        };
        
        // Load after a short delay to improve perceived performance
        setTimeout(() => {
            container.appendChild(iframe);
        }, 100);
    }
    
    loadWidget(containerId, notebookId, cellId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        const url = `https://deepnote.com/embed/48-manifold/${notebookId}?cellId=${cellId}&hideCode=true`;
        
        const iframe = document.createElement('iframe');
        iframe.src = url;
        iframe.width = '100%';
        iframe.height = '400';
        iframe.frameBorder = '0';
        
        container.appendChild(iframe);
    }
}

// Code copy functionality
function initCodeBlocks() {
    const codeBlocks = document.querySelectorAll('pre');
    
    codeBlocks.forEach(block => {
        // Create copy button
        const copyBtn = document.createElement('button');
        copyBtn.className = 'code-copy-btn';
        copyBtn.textContent = 'Copy';
        
        copyBtn.addEventListener('click', async () => {
            const code = block.textContent;
            const success = await utils.copyToClipboard(code);
            
            if (success) {
                copyBtn.textContent = 'Copied!';
                copyBtn.style.background = 'var(--color-success)';
                
                setTimeout(() => {
                    copyBtn.textContent = 'Copy';
                    copyBtn.style.background = '';
                }, 2000);
            }
        });
        
        // Wrap in container if not already wrapped
        if (!block.parentElement.classList.contains('code-block')) {
            const wrapper = document.createElement('div');
            wrapper.className = 'code-block';
            block.parentNode.insertBefore(wrapper, block);
            wrapper.appendChild(block);
            
            const header = document.createElement('div');
            header.className = 'code-block-header';
            header.appendChild(copyBtn);
            wrapper.insertBefore(header, block);
        }
    });
}

// Smooth scroll for anchor links
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = this.getAttribute('href');
            utils.smoothScroll(target);
        });
    });
}

// Playground tab functionality
function initPlaygroundTabs() {
    const tabs = document.querySelectorAll('.playground-tab');
    const contents = document.querySelectorAll('.playground-content');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.style.display = 'none');
            
            // Add active to clicked tab
            tab.classList.add('active');
            
            // Show corresponding content
            const target = tab.getAttribute('data-target');
            const content = document.getElementById(target);
            if (content) {
                content.style.display = 'block';
            }
        });
    });
}

// Stats counter animation
function animateStats() {
    const stats = document.querySelectorAll('.stat-number');
    
    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const stat = entry.target;
                const target = parseFloat(stat.getAttribute('data-target') || stat.textContent);
                const duration = 2000; // 2 seconds
                const start = performance.now();
                
                const updateCounter = (currentTime) => {
                    const elapsed = currentTime - start;
                    const progress = Math.min(elapsed / duration, 1);
                    
                    // Easing function
                    const easeOutQuart = 1 - Math.pow(1 - progress, 4);
                    const current = target * easeOutQuart;
                    
                    // Format based on whether it's an integer or float
                    if (Number.isInteger(target)) {
                        stat.textContent = Math.floor(current);
                    } else {
                        stat.textContent = current.toFixed(3);
                    }
                    
                    if (progress < 1) {
                        requestAnimationFrame(updateCounter);
                    } else {
                        stat.textContent = target;
                    }
                };
                
                requestAnimationFrame(updateCounter);
                observer.unobserve(stat);
            }
        });
    });
    
    stats.forEach(stat => observer.observe(stat));
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    console.log(`Initializing ${MANIFOLD_CONFIG.name} site...`);
    
    // Initialize components
    initCodeBlocks();
    initSmoothScroll();
    initPlaygroundTabs();
    animateStats();
    
    // Initialize fractal navigator if canvas exists
    const fractalCanvas = document.getElementById('fractal-canvas');
    if (fractalCanvas) {
        window.fractalNavigator = new FractalNavigator('fractal-canvas');
    }
    
    // Initialize Deepnote manager
    window.deepnote = new DeepnoteManager();
    
    // Load any Deepnote embeds marked for auto-load
    document.querySelectorAll('[data-deepnote-auto]').forEach(element => {
        const notebook = element.getAttribute('data-deepnote-notebook');
        const height = element.getAttribute('data-deepnote-height') || 600;
        window.deepnote.embedNotebook(element.id, notebook, { height: parseInt(height) });
    });
    
    // Add scroll-based animations
    const animatedElements = document.querySelectorAll('.animate-on-scroll');
    const animationObserver = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animated');
            }
        });
    }, { threshold: 0.1 });
    
    animatedElements.forEach(el => animationObserver.observe(el));
    
    console.log(`${MANIFOLD_CONFIG.name} site initialized successfully!`);
});

// Export for use in other scripts
window.ManifoldUtils = utils;
window.ManifoldConfig = MANIFOLD_CONFIG;