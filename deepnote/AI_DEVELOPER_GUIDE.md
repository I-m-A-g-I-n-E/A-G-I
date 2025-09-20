# AI Developer Guide for {wtf²B•2^4*3} Manifold Deepnote Notebooks

## 🎯 Mission
Transform the existing MPS/CPU codebase into CUDA-optimized Deepnote notebooks with interactive demonstrations.

---

## 📁 Repository Structure

```
deepnote/
├── cuda_devices.py          # Unified CUDA device manager
├── requirements.txt         # Deepnote environment setup
├── notebooks/
│   ├── 1_Memory_Performance.ipynb  # Memory & benchmarks
│   ├── 2_Molar_Protein.ipynb       # Protein folding
│   ├── 3_Motor_Hand.ipynb          # Hand visualizations
│   └── 4_Manifold_Playground.ipynb # Interactive playground
├── modules/                 # CUDA-optimized modules
│   ├── __init__.py
│   ├── composer_cuda.py     # Port of bio.composer
│   ├── conductor_cuda.py    # Port of bio.conductor
│   ├── sonifier_cuda.py     # Port of bio.sonifier
│   └── hand_cuda.py         # Port of hand tensor
├── tests/
│   ├── smoke_tests.py       # Basic functionality tests
│   └── integration_tests.py # End-to-end tests
└── results/                 # Benchmark results & outputs
    ├── memory/
    ├── protein/
    ├── hand/
    └── manifold/
```

---

## 🌿 Branch Strategy

### Main Branches
- `main` - Production-ready notebooks
- `develop` - Integration branch for features
- `deepnote-cuda` - CUDA optimization work

### Feature Branches
Each developer should work in isolated feature branches:

```bash
# Memory team
git checkout -b feature/memory-benchmarks
git checkout -b feature/memory-profiling

# Molar team  
git checkout -b feature/molar-composer
git checkout -b feature/molar-conductor
git checkout -b feature/molar-antibodies

# Motor team
git checkout -b feature/motor-gestures
git checkout -b feature/motor-visualization

# Manifold team
git checkout -b feature/manifold-fractal
git checkout -b feature/manifold-interactive
```

### Naming Conventions
- `feature/{module}-{feature}` - New features
- `fix/{module}-{issue}` - Bug fixes
- `optimize/{module}-{target}` - Performance improvements
- `docs/{module}-{topic}` - Documentation

### Merge Strategy
1. Feature branches → `develop` (via PR)
2. `develop` → `deepnote-cuda` (integration testing)
3. `deepnote-cuda` → `main` (after validation)

---

## 🚀 Development Workflow

### 1. Initial Setup
```bash
# Clone repository
git clone https://github.com/I-m-A-g-I-n-E/agi.git
cd agi

# Create your feature branch
git checkout -b feature/memory-benchmarks

# Install dependencies locally for testing
pip install -r deepnote/requirements.txt
```

### 2. Deepnote Environment Setup
```python
# In each notebook's first cell:
!pip install -r /work/deepnote/requirements.txt
!export PYTHONPATH=/work:$PYTHONPATH

# Verify CUDA
from deepnote.cuda_devices import cuda_manager
cuda_manager.ensure_cuda_available()
```

### 3. CUDA Conversion Guidelines

#### Device Management
```python
# OLD (MPS/CPU)
from bio.devices import get_device
device = get_device()  # Returns MPS on Mac

# NEW (CUDA)
from deepnote.cuda_devices import get_device, cuda_manager
device = get_device()  # Prioritizes CUDA

# Memory efficient operations
with cuda_manager.memory_efficient_mode(0.8):
    # Your memory-intensive operations
    pass
```

#### Tensor Operations
```python
# Always use non-blocking transfers
tensor = tensor.to(device, non_blocking=True)

# Clear cache regularly
cuda_manager.clear_cache()

# Monitor memory
mem_usage = cuda_manager.get_memory_summary()
```

#### Mixed Precision (for A100/V100)
```python
# Check if AMP is available
if cuda_manager.auto_mixed_precision_available():
    with torch.cuda.amp.autocast():
        # Your operations run in mixed precision
        output = model(input)
```

---

## 📋 Task Assignments

### Memory Team
**Lead: AI Dev 1**
- [ ] Port memory benchmarks to CUDA
- [ ] Implement allocation profiling
- [ ] Add multi-GPU scaling tests
- [ ] Create memory pressure tests
- [ ] Document optimization strategies

### Molar Team  
**Lead: AI Dev 2**
- [ ] Convert HarmonicPropagator to CUDA
- [ ] Port Conductor with GPU NeRF
- [ ] Implement batch processing
- [ ] Add AlphaFold comparison
- [ ] Create antibody analysis pipeline

### Motor Team
**Lead: AI Dev 3**
- [ ] Port hand tensor visualizations
- [ ] Implement gesture recognition
- [ ] Add real-time interaction
- [ ] Create haptic feedback simulation
- [ ] Build gesture library

### Manifold Team
**Lead: AI Dev 4**
- [ ] Create fractal navigator
- [ ] Build interactive playground
- [ ] Implement WebGL export
- [ ] Add parameter exploration
- [ ] Create visual presets

---

## 🧪 Testing Requirements

### Smoke Tests (Required for each PR)
```python
# Each notebook must pass:
def test_cuda_available():
    assert torch.cuda.is_available()

def test_module_import():
    from deepnote.modules import composer_cuda
    assert composer_cuda is not None

def test_basic_operation():
    x = torch.randn(48, 48, device='cuda')
    y = x @ x.T
    assert y.shape == (48, 48)

def test_memory_cleanup():
    initial = torch.cuda.memory_allocated()
    # Your operations
    torch.cuda.empty_cache()
    final = torch.cuda.memory_allocated()
    assert abs(final - initial) < 1e6  # Less than 1MB difference
```

### Integration Tests
```bash
# Run from repository root
python deepnote/tests/integration_tests.py
```

### Performance Benchmarks
- Each module must show >2x speedup vs CPU
- Memory usage must stay under 8GB for T4
- Batch operations must scale linearly

---

## 🔍 Code Review Checklist

- [ ] CUDA device handling with fallback
- [ ] Memory efficient operations
- [ ] Non-blocking transfers
- [ ] Proper cache clearing
- [ ] Type hints on all functions
- [ ] Docstrings with examples
- [ ] Smoke tests passing
- [ ] No hardcoded paths
- [ ] Results saved to appropriate directory

---

## 📊 Performance Targets

### Memory Notebook
- Allocation: <1ms for 48x48 tensors
- Transfer bandwidth: >10 GB/s
- Memory fragmentation: <10%

### Molar Notebook
- Composition: <100ms for 1000 residues
- Structure generation: <1s per protein
- Batch processing: 100 proteins/minute

### Motor Notebook
- Gesture recognition: 60 FPS
- Visualization: Real-time (>30 FPS)
- Interaction latency: <50ms

### Manifold Notebook
- Fractal rendering: 60 FPS
- Parameter updates: Real-time
- Export: <5s for 4K image

---

## 🚨 Common Issues & Solutions

### Out of Memory (OOM)
```python
# Solution 1: Clear cache
torch.cuda.empty_cache()

# Solution 2: Reduce batch size
batch_size = 16 if device.type == 'cuda' else 32

# Solution 3: Use memory efficient mode
with cuda_manager.memory_efficient_mode(0.5):
    # Operations use only 50% of GPU memory
    pass
```

### Slow Data Transfer
```python
# Use pinned memory for faster transfers
tensor = torch.randn(1000, 1000, pin_memory=True)
tensor = tensor.to(device, non_blocking=True)
```

### CUDA Version Mismatch
```bash
# Check versions
python -c "import torch; print(torch.version.cuda)"
nvcc --version

# Reinstall PyTorch for correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 📚 Resources

### Documentation
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/cuda.html)
- [NVIDIA Deep Learning Performance](https://docs.nvidia.com/deeplearning/performance/)
- [Deepnote GPU Guide](https://docs.deepnote.com/features/gpu)

### Project Specific
- Original codebase: `/bio/` directory
- Math foundations: `/docs/` directory
- Test data: `/outputs/` directory

### Communication
- Slack: #manifold-cuda channel
- Daily standup: 10 AM PST
- PR reviews: Within 24 hours

---

## ✅ Definition of Done

A feature is considered complete when:
1. **Code** is ported to CUDA with CPU fallback
2. **Tests** pass all smoke tests
3. **Performance** meets targets (>2x speedup)
4. **Documentation** is updated
5. **Notebook** runs in Deepnote without errors
6. **Review** approved by team lead
7. **Integration** tested with other modules

---

## 🎯 Quick Start Commands

```bash
# Setup
git checkout -b feature/your-feature
pip install -r deepnote/requirements.txt

# Development
jupyter lab deepnote/notebooks/

# Testing
python -m pytest deepnote/tests/

# Benchmarking
python deepnote/tests/benchmark.py --gpu

# Commit
git add .
git commit -m "feat(module): description"
git push origin feature/your-feature
```

---

## 🤝 Collaboration Protocol

1. **Claim your task** in the GitHub issue
2. **Create feature branch** following naming convention
3. **Update daily** in Slack standup
4. **Request review** when smoke tests pass
5. **Merge to develop** after approval
6. **Update documentation** if APIs change

---

## 📈 Success Metrics

- All 4 notebooks running on Deepnote
- >90% test coverage
- <100ms latency for interactions
- >10x speedup vs CPU for core operations
- Zero memory leaks
- Documentation complete

---

**Remember**: The goal is to create a robust, scalable foundation that other developers can build upon. Quality over speed, but both are important!

Good luck, and may your tensors be ever CUDA-aligned! 🚀