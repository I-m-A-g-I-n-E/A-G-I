# Deepnote Setup Instructions for {wtf²B•2^4*3} Manifold

Since the Deepnote API endpoints have changed, here's how to manually set up your Deepnote workspace with all the prepared notebooks:

## 📋 Quick Setup Steps

### 1. Create New Deepnote Project
1. Go to [Deepnote](https://deepnote.com)
2. Click "New Project"
3. Name it: **"48-Manifold"**
4. Select **GPU** machine (T4 or better)

### 2. Upload Files via Deepnote UI
In your Deepnote project, upload these files from `/deepnote/`:

#### Core Files (upload to root):
- `cuda_devices.py` - CUDA device manager
- `requirements.txt` - Dependencies
- `setup_deepnote.sh` - Setup script
- `smoke_tests.py` - Testing suite
- `AI_DEVELOPER_GUIDE.md` - Developer documentation

#### Notebooks (create `notebooks/` folder):
- `1_Memory_Performance.ipynb`
- `2_Molar_Protein.ipynb`
- `3_Motor_Hand.ipynb` (to be created)
- `4_Manifold_Playground.ipynb` (to be created)

### 3. Environment Setup
In a new notebook or terminal, run:

```bash
# Install requirements
pip install -r requirements.txt

# Make setup script executable and run it
chmod +x setup_deepnote.sh
./setup_deepnote.sh
```

### 4. Initialize GPU
In your first notebook cell:

```python
import sys
sys.path.append('/work')

from cuda_devices import cuda_manager, get_device
device = get_device()
print(f"Device: {device}")

if device.type == 'cuda':
    info = cuda_manager.get_device_info()
    print(f"GPU: {info.get('name')}")
    print(f"Memory: {info.get('total_memory_gb'):.1f} GB")
```

### 5. Verify Installation
Run the smoke tests:

```python
!python smoke_tests.py
```

## 🚀 Alternative: GitHub Integration

### Option A: Direct GitHub Import
1. In Deepnote, click "Import from GitHub"
2. Connect to repository: `https://github.com/I-m-A-g-I-n-E/agi`
3. Select branch: `gh-pages`
4. Navigate to `/deepnote/` folder
5. Import notebooks

### Option B: Git Clone in Deepnote
In Deepnote terminal:

```bash
# Clone the repository
git clone https://github.com/I-m-A-g-I-n-E/agi.git
cd agi
git checkout gh-pages

# Copy deepnote files to working directory
cp -r deepnote/* /work/

# Run setup
cd /work
chmod +x setup_deepnote.sh
./setup_deepnote.sh
```

## 📁 Final Structure
Your Deepnote project should look like:

```
/work/
├── cuda_devices.py
├── requirements.txt
├── setup_deepnote.sh
├── smoke_tests.py
├── AI_DEVELOPER_GUIDE.md
├── notebooks/
│   ├── 1_Memory_Performance.ipynb
│   ├── 2_Molar_Protein.ipynb
│   ├── 3_Motor_Hand.ipynb
│   └── 4_Manifold_Playground.ipynb
├── modules/           (to be created by developers)
├── tests/            (to be created by developers)
└── results/          (created by setup script)
    ├── memory/
    ├── protein/
    ├── hand/
    └── manifold/
```

## 🎯 For AI Developers

Each developer should:

1. **Duplicate the project** for their feature branch
2. **Follow the branch naming** in AI_DEVELOPER_GUIDE.md
3. **Run smoke tests** before any commits
4. **Save results** in the appropriate `/results/` subdirectory
5. **Document changes** in their notebook's markdown cells

## 💡 Tips

- **GPU Selection**: Choose T4 for development, V100/A100 for benchmarks
- **Memory Management**: Use `cuda_manager.clear_cache()` regularly
- **Collaboration**: Use Deepnote's real-time collaboration for pair programming
- **Version Control**: Commit notebooks with outputs cleared to avoid large diffs

## 🧪 Testing Checklist

- [ ] CUDA device detected
- [ ] All requirements installed
- [ ] Smoke tests pass
- [ ] Can import cuda_devices module
- [ ] Memory notebook runs
- [ ] Molar notebook runs
- [ ] No memory leaks detected

## 📞 Support

If you encounter issues:
1. Check the smoke test output
2. Verify GPU is enabled in Deepnote settings
3. Consult AI_DEVELOPER_GUIDE.md
4. Post in Slack #manifold-cuda channel

---

**Ready to start!** Open `notebooks/1_Memory_Performance.ipynb` to begin. 🚀