"""
CUDA-optimized device management for Deepnote environment.
Extends bio.devices with CUDA-specific optimizations and Deepnote compatibility.

This module ensures smooth operation across:
- Deepnote with CUDA (T4, V100, A100)
- Local development (CPU, MPS, CUDA)
- CI/CD environments (CPU only)
"""

import os
import torch
import warnings
from typing import Optional, Dict, Any, Tuple
from contextlib import contextmanager
import psutil
import GPUtil

class CUDADeviceManager:
    """
    Unified device manager optimized for Deepnote CUDA environments.
    Provides intelligent fallbacks and memory management.
    """
    
    _instance = None
    _device_cache = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._device_cache is None:
            self._device_cache = self._detect_device()
            self._setup_cuda_optimizations()
    
    def _detect_device(self) -> torch.device:
        """Detect best available device with Deepnote priority."""
        # Force CPU if requested
        if os.getenv("AGI_FORCE_CPU"):
            return torch.device("cpu")
        
        # Check for explicit device preference
        env_device = os.getenv("AGI_DEVICE", "").lower()
        
        # Deepnote typically has CUDA
        if env_device == "cuda" or torch.cuda.is_available():
            if torch.cuda.is_available():
                # Get GPU info for optimization
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    # Use first available GPU
                    device = torch.device("cuda:0")
                    self._log_gpu_info(device)
                    return device
        
        # Fallback for local dev
        if env_device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        
        # CPU fallback
        warnings.warn("No GPU detected, falling back to CPU. Performance will be limited.")
        return torch.device("cpu")
    
    def _setup_cuda_optimizations(self):
        """Configure CUDA-specific optimizations."""
        if self._device_cache.type == "cuda":
            # Enable TF32 for Ampere GPUs (A100)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable cudnn autotuner for conv operations
            torch.backends.cudnn.benchmark = True
            
            # Set memory fraction for better multi-notebook support
            if os.getenv("DEEPNOTE_MEMORY_FRACTION"):
                fraction = float(os.getenv("DEEPNOTE_MEMORY_FRACTION", "0.9"))
                torch.cuda.set_per_process_memory_fraction(fraction)
    
    def _log_gpu_info(self, device: torch.device):
        """Log GPU information for debugging."""
        if device.type == "cuda":
            idx = device.index or 0
            props = torch.cuda.get_device_properties(idx)
            print(f"🎮 GPU Detected: {props.name}")
            print(f"   Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
            print(f"   Multi-processors: {props.multi_processor_count}")
    
    @property
    def device(self) -> torch.device:
        """Get the cached device."""
        return self._device_cache
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        info = {
            "device": str(self._device_cache),
            "type": self._device_cache.type,
            "index": self._device_cache.index,
        }
        
        if self._device_cache.type == "cuda":
            idx = self._device_cache.index or 0
            props = torch.cuda.get_device_properties(idx)
            info.update({
                "name": props.name,
                "total_memory_gb": props.total_memory / 1024**3,
                "compute_capability": f"{props.major}.{props.minor}",
                "multiprocessors": props.multi_processor_count,
                "cuda_version": torch.version.cuda,
            })
            
            # Current memory usage
            if torch.cuda.is_available():
                info["allocated_memory_gb"] = torch.cuda.memory_allocated(idx) / 1024**3
                info["cached_memory_gb"] = torch.cuda.memory_reserved(idx) / 1024**3
        
        # System info
        info["cpu_count"] = psutil.cpu_count()
        info["ram_gb"] = psutil.virtual_memory().total / 1024**3
        
        return info
    
    @contextmanager
    def memory_efficient_mode(self, fraction: float = 0.8):
        """Context manager for memory-efficient operations."""
        if self._device_cache.type == "cuda":
            # Save current fraction
            old_fraction = torch.cuda.get_per_process_memory_fraction()
            
            try:
                # Set temporary fraction
                torch.cuda.set_per_process_memory_fraction(fraction)
                torch.cuda.empty_cache()
                yield
            finally:
                # Restore
                torch.cuda.set_per_process_memory_fraction(old_fraction)
                torch.cuda.empty_cache()
        else:
            yield
    
    def optimize_for_inference(self):
        """Optimize settings for inference (non-training) workloads."""
        if self._device_cache.type == "cuda":
            torch.backends.cudnn.benchmark = False  # Deterministic
            torch.cuda.empty_cache()
    
    def optimize_for_training(self):
        """Optimize settings for training workloads."""
        if self._device_cache.type == "cuda":
            torch.backends.cudnn.benchmark = True  # Autotuner
            torch.cuda.empty_cache()
    
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if self._device_cache.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_memory_summary(self) -> str:
        """Get a formatted memory summary."""
        if self._device_cache.type == "cuda":
            return torch.cuda.memory_summary()
        return "Memory tracking not available for CPU/MPS"
    
    def ensure_cuda_available(self) -> bool:
        """Check if CUDA is available, with helpful error messages."""
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available. Possible reasons:")
            print("   1. Running on CPU-only Deepnote instance")
            print("   2. CUDA drivers not installed")
            print("   3. PyTorch installed without CUDA support")
            print(f"   Current PyTorch: {torch.__version__}")
            print(f"   CUDA built: {torch.version.cuda}")
            return False
        return True
    
    def auto_mixed_precision_available(self) -> bool:
        """Check if automatic mixed precision is available."""
        if self._device_cache.type == "cuda":
            # Check compute capability (need >= 7.0 for good AMP)
            idx = self._device_cache.index or 0
            major, minor = torch.cuda.get_device_capability(idx)
            return major >= 7
        return False

# Global instance
cuda_manager = CUDADeviceManager()

# Convenience functions matching bio.devices API
def get_device() -> torch.device:
    """Get the best available device (CUDA prioritized for Deepnote)."""
    return cuda_manager.device

def to_device(x, device: Optional[torch.device] = None):
    """Move tensor(s) to device with proper handling."""
    if device is None:
        device = cuda_manager.device
    
    if hasattr(x, "to"):
        return x.to(device, non_blocking=True)
    
    # Handle collections
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(item, device) for item in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    
    return x

def clear_gpu_cache():
    """Clear GPU cache - useful in notebooks to prevent OOM."""
    cuda_manager.clear_cache()

def get_gpu_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage in GB."""
    if cuda_manager.device.type == "cuda":
        idx = cuda_manager.device.index or 0
        return {
            "allocated": torch.cuda.memory_allocated(idx) / 1024**3,
            "cached": torch.cuda.memory_reserved(idx) / 1024**3,
            "total": torch.cuda.get_device_properties(idx).total_memory / 1024**3,
        }
    return {"allocated": 0, "cached": 0, "total": 0}

# Export bio.devices compatible interface
__all__ = [
    "cuda_manager",
    "get_device",
    "to_device", 
    "clear_gpu_cache",
    "get_gpu_memory_usage",
    "CUDADeviceManager",
]