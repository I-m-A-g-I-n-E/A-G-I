#!/usr/bin/env python3
"""
Smoke tests for all Deepnote notebooks.
Run this to ensure basic functionality before pushing.
"""

import sys
import torch
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_cuda_availability() -> Tuple[bool, str]:
    """Test if CUDA is available and properly configured."""
    try:
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        # Test basic CUDA operation
        x = torch.randn(48, 48, device='cuda')
        y = x @ x.T
        torch.cuda.synchronize()
        
        # Check CUDA version
        cuda_version = torch.version.cuda
        return True, f"CUDA {cuda_version} available"
    except Exception as e:
        return False, f"CUDA test failed: {e}"

def test_device_manager() -> Tuple[bool, str]:
    """Test the CUDA device manager."""
    try:
        from deepnote.cuda_devices import cuda_manager, get_device
        
        device = get_device()
        info = cuda_manager.get_device_info()
        
        if device.type not in ['cuda', 'cpu', 'mps']:
            return False, f"Unknown device type: {device.type}"
        
        # Test memory management
        if device.type == 'cuda':
            cuda_manager.clear_cache()
            mem_usage = cuda_manager.get_memory_summary()
            
        return True, f"Device manager working on {device}"
    except Exception as e:
        return False, f"Device manager failed: {e}"

def test_memory_allocation(size: int = 4800) -> Tuple[bool, str]:
    """Test memory allocation and deallocation."""
    try:
        from deepnote.cuda_devices import get_device, clear_gpu_cache
        
        device = get_device()
        
        # Allocate large tensor
        tensor = torch.zeros(size, size, device=device)
        memory_mb = tensor.element_size() * tensor.numel() / (1024**2)
        
        # Deallocate
        del tensor
        clear_gpu_cache()
        
        return True, f"Allocated and freed {memory_mb:.1f} MB"
    except Exception as e:
        return False, f"Memory allocation failed: {e}"

def test_tensor_operations() -> Tuple[bool, str]:
    """Test basic tensor operations on GPU."""
    try:
        from deepnote.cuda_devices import get_device
        
        device = get_device()
        
        # Test various operations
        x = torch.randn(48, 48, device=device)
        
        # Matrix multiplication
        y = torch.matmul(x, x.T)
        
        # Element-wise operations
        z = torch.sin(x) + torch.cos(x)
        
        # Reductions
        mean = x.mean()
        std = x.std()
        
        # Reshaping
        reshaped = x.reshape(3, 16, 48)
        
        # Permutation
        permuted = reshaped.permute(2, 0, 1)
        
        return True, f"Tensor operations successful on {device}"
    except Exception as e:
        return False, f"Tensor operations failed: {e}"

def test_mixed_precision() -> Tuple[bool, str]:
    """Test automatic mixed precision if available."""
    try:
        from deepnote.cuda_devices import cuda_manager, get_device
        
        device = get_device()
        
        if device.type != 'cuda':
            return True, "Mixed precision skipped (CPU)"
        
        if not cuda_manager.auto_mixed_precision_available():
            return True, "Mixed precision not available on this GPU"
        
        # Test AMP
        with torch.cuda.amp.autocast():
            x = torch.randn(1024, 1024, device=device)
            y = torch.matmul(x, x)
            
        return True, "Mixed precision working"
    except Exception as e:
        return False, f"Mixed precision failed: {e}"

def test_data_transfer() -> Tuple[bool, str]:
    """Test CPU <-> GPU data transfer."""
    try:
        from deepnote.cuda_devices import get_device
        
        device = get_device()
        
        if device.type == 'cpu':
            return True, "Transfer test skipped (CPU only)"
        
        # Create CPU tensor
        cpu_tensor = torch.randn(1000, 1000)
        
        # Transfer to GPU
        start = time.perf_counter()
        gpu_tensor = cpu_tensor.to(device, non_blocking=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        h2d_time = time.perf_counter() - start
        
        # Transfer back to CPU
        start = time.perf_counter()
        cpu_back = gpu_tensor.to('cpu')
        if device.type == 'cuda':
            torch.cuda.synchronize()
        d2h_time = time.perf_counter() - start
        
        return True, f"H2D: {h2d_time*1000:.1f}ms, D2H: {d2h_time*1000:.1f}ms"
    except Exception as e:
        return False, f"Data transfer failed: {e}"

def test_manifold_operations() -> Tuple[bool, str]:
    """Test 48-manifold specific operations."""
    try:
        from deepnote.cuda_devices import get_device
        
        device = get_device()
        dims = 48
        
        # Test factorization (48 = 2^4 × 3)
        x = torch.randn(32, dims, dims, dims, device=device)
        
        # Factor by 3
        if dims % 3 == 0:
            x_reshape = x.reshape(32, dims, dims//3, 3, dims//3, 3)
            x_permute = x_reshape.permute(0, 1, 3, 5, 2, 4)
            x_factor3 = x_permute.reshape(32, dims * 9, dims//3, dims//3)
        
        # Factor by 2
        if dims % 2 == 0:
            h, w = dims//3, dims//3
            if h % 2 == 0 and w % 2 == 0:
                y = x_factor3[:, :dims*4, :h, :w]
                y_reshape = y.reshape(32, -1, h//2, 2, w//2, 2)
                y_permute = y_reshape.permute(0, 1, 3, 5, 2, 4)
                y_factor2 = y_permute.reshape(32, -1, h//2, w//2)
        
        return True, "48-manifold operations working"
    except Exception as e:
        return False, f"Manifold operations failed: {e}"

def test_notebook_imports() -> Tuple[bool, str]:
    """Test that all required packages can be imported."""
    try:
        required_packages = [
            'torch',
            'numpy',
            'plotly',
            'ipywidgets',
            'pandas',
            'scipy',
            'matplotlib',
            'tqdm',
        ]
        
        failed = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                failed.append(package)
        
        if failed:
            return False, f"Missing packages: {', '.join(failed)}"
        
        return True, "All required packages available"
    except Exception as e:
        return False, f"Import test failed: {e}"

def test_file_structure() -> Tuple[bool, str]:
    """Test that expected file structure exists."""
    try:
        base_path = Path(__file__).parent
        
        expected_files = [
            'cuda_devices.py',
            'requirements.txt',
            'notebooks/1_Memory_Performance.ipynb',
            'notebooks/2_Molar_Protein.ipynb',
            'AI_DEVELOPER_GUIDE.md',
        ]
        
        missing = []
        for file in expected_files:
            if not (base_path / file).exists():
                missing.append(file)
        
        if missing:
            return False, f"Missing files: {', '.join(missing)}"
        
        return True, "File structure intact"
    except Exception as e:
        return False, f"File structure test failed: {e}"

def test_memory_cleanup() -> Tuple[bool, str]:
    """Test that memory is properly cleaned up."""
    try:
        from deepnote.cuda_devices import get_device, clear_gpu_cache, get_gpu_memory_usage
        
        device = get_device()
        
        if device.type != 'cuda':
            return True, "Memory cleanup skipped (non-CUDA)"
        
        # Get initial memory
        initial = get_gpu_memory_usage()
        
        # Allocate and deallocate
        for _ in range(10):
            x = torch.randn(1000, 1000, device=device)
            del x
        
        # Clear cache
        clear_gpu_cache()
        
        # Check final memory
        final = get_gpu_memory_usage()
        
        leaked = final['allocated'] - initial['allocated']
        if leaked > 0.01:  # More than 10MB leaked
            return False, f"Memory leak detected: {leaked:.2f} GB"
        
        return True, "Memory cleanup successful"
    except Exception as e:
        return False, f"Memory cleanup test failed: {e}"

def run_all_tests() -> Dict[str, Any]:
    """Run all smoke tests and return results."""
    tests = [
        ("CUDA Availability", test_cuda_availability),
        ("Device Manager", test_device_manager),
        ("Memory Allocation", test_memory_allocation),
        ("Tensor Operations", test_tensor_operations),
        ("Mixed Precision", test_mixed_precision),
        ("Data Transfer", test_data_transfer),
        ("Manifold Operations", test_manifold_operations),
        ("Package Imports", test_notebook_imports),
        ("File Structure", test_file_structure),
        ("Memory Cleanup", test_memory_cleanup),
    ]
    
    results = {
        "passed": [],
        "failed": [],
        "total": len(tests),
        "success_rate": 0,
        "details": {}
    }
    
    print("\n" + "="*60)
    print("🧪 RUNNING DEEPNOTE SMOKE TESTS")
    print("="*60 + "\n")
    
    for name, test_func in tests:
        try:
            passed, message = test_func()
            
            if passed:
                results["passed"].append(name)
                print(f"✅ {name}: {message}")
            else:
                results["failed"].append(name)
                print(f"❌ {name}: {message}")
            
            results["details"][name] = {
                "passed": passed,
                "message": message
            }
            
        except Exception as e:
            results["failed"].append(name)
            print(f"❌ {name}: Unexpected error: {e}")
            results["details"][name] = {
                "passed": False,
                "message": f"Unexpected error: {e}"
            }
    
    # Calculate success rate
    results["success_rate"] = len(results["passed"]) / results["total"] * 100
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"Passed: {len(results['passed'])}/{results['total']}")
    print(f"Failed: {len(results['failed'])}/{results['total']}")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    
    if results["failed"]:
        print(f"\n⚠️ Failed tests: {', '.join(results['failed'])}")
    
    if results["success_rate"] == 100:
        print("\n🎉 All tests passed! Ready for Deepnote deployment.")
    elif results["success_rate"] >= 80:
        print("\n⚡ Most tests passed. Review failures before deployment.")
    else:
        print("\n🚨 Multiple test failures. Environment needs configuration.")
    
    return results

if __name__ == "__main__":
    # Run tests
    results = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if results["success_rate"] == 100 else 1)