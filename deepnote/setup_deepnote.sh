#!/bin/bash

# Setup script for Deepnote environment
# Run this in Deepnote's terminal to configure the environment

echo "🚀 Setting up {wtf²B•2^4*3} Manifold Deepnote Environment"
echo "========================================================="

# Check Python version
echo "📌 Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
echo "   Python: $python_version"

# Check CUDA availability
echo "📌 Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "   ⚠️ CUDA not available (CPU mode)"
fi

# Install requirements
echo "📌 Installing requirements..."
pip install -q --upgrade pip
pip install -q -r /work/deepnote/requirements.txt

# Set Python path
echo "📌 Setting Python path..."
export PYTHONPATH=/work:$PYTHONPATH
echo "export PYTHONPATH=/work:\$PYTHONPATH" >> ~/.bashrc

# Create necessary directories
echo "📌 Creating directory structure..."
mkdir -p /work/deepnote/results/{memory,protein,hand,manifold}
mkdir -p /work/deepnote/modules
mkdir -p /work/deepnote/tests

# Run smoke tests
echo "📌 Running smoke tests..."
cd /work/deepnote
python3 smoke_tests.py

# Create initialization file for notebooks
cat > /work/deepnote/init_notebook.py << 'EOF'
"""
Initialize Deepnote notebook environment.
Run this in the first cell of each notebook.
"""

import sys
import os
import warnings

# Add project to path
sys.path.insert(0, '/work')
sys.path.insert(0, '/work/deepnote')

# Configure warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Import CUDA manager
try:
    from deepnote.cuda_devices import cuda_manager, get_device
    device = get_device()
    print(f"✅ Device initialized: {device}")
    
    if device.type == 'cuda':
        info = cuda_manager.get_device_info()
        print(f"🎮 GPU: {info.get('name', 'Unknown')}")
        print(f"💾 Memory: {info.get('total_memory_gb', 0):.1f} GB")
        print(f"🔧 CUDA: {info.get('cuda_version', 'Unknown')}")
except Exception as e:
    print(f"⚠️ Device initialization failed: {e}")
    print("   Falling back to CPU")

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

# Configure plotly
import plotly.io as pio
pio.templates.default = "plotly_dark"

print("\n🚀 Environment ready!")
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Open any notebook in /work/deepnote/notebooks/"
echo "   2. Run the first cell to initialize the environment"
echo "   3. Follow the TODOs in each notebook"
echo ""
echo "📚 Documentation: /work/deepnote/AI_DEVELOPER_GUIDE.md"
echo "🧪 Run tests: python3 /work/deepnote/smoke_tests.py"
echo ""
echo "Happy coding! 🎉"