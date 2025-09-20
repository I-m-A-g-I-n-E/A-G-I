# Copilot Instructions for A-G-I Repository

## Repository Overview
This repository implements the Universal Semantic Kernel (USK) and Harmonic Integrity Protocol based on the 48-manifold (2^4 × 3) mathematical foundation. Our core principle is "Transfer, not Transform" — preserving informational integrity through reversible, on-grid operations.

## Core Principles

### 1. The 48-Manifold Foundation
- Always use 48 as the fundamental basis (2^4 × 3 = 48)
- Dimensions, tiles, and windows must be multiples of 48
- Use exact factorizations: {1, 2, 3, 4, 6, 8, 12, 16, 24, 48}
- Implement Chinese Remainder Theorem (CRT) indexing: `phi = (i mod 16) * 3 + (j mod 3)`

### 2. Transfer, Not Transform
- **NEVER** use lossy operations (pooling, strided convolutions, decimation)
- **ALWAYS** use reversible operations:
  - `space_to_depth_2` / `depth_to_space_2` for 2× scaling
  - `space_to_depth_3` / `depth_to_space_3` for 3× scaling
  - Unitary or orthogonal transformations
  - Integer-lifting operations (determinant = ±1)
- Single-shot measurement only at final output (measurement operator M)

### 3. Parity and Duality
- Maintain `keven` (even/symmetric) and `kodd` (odd/antisymmetric) separation
- Use parity masks: `p = (x + y) mod 2`
- Implement operations as `G = {L | R}` where:
  - `L` (Left) = cold, structural, resolving moves
  - `R` (Right) = hot, dynamic, tensive moves

## Code Conventions

### Naming Conventions
```python
# Preferred naming patterns
keven_channel    # Even/symmetric component
kodd_channel     # Odd/antisymmetric component  
space_to_depth_N # N ∈ {2, 3} permutation operations
measurement_M    # Final measurement operator
canonicalize     # Reduce to simplest form
```

### Required Validations
```python
# Always validate 48-alignment
assert width % 48 == 0, f"Width {width} must be multiple of 48"
assert height % 48 == 0, f"Height {height} must be multiple of 48"

# Verify reversibility
assert np.allclose(inverse_op(forward_op(x)), x), "Operation must be reversible"
```

### Error Handling
- Reject non-48-aligned inputs immediately with clear error messages
- Log all permutations and transformations for provenance
- Include measurement context in all outputs

## Module Structure

### Core Modules
- `manifold.py` - Core 48-manifold operations and permutations
- `kernels.py` - USK even/odd kernels and operators
- `measurement.py` - Measurement-first evaluation operators
- `provenance.py` - Tracking and logging all operations

### When Adding Features
1. Check 48-alignment compatibility first
2. Ensure all operations are reversible
3. Add parity-aware variants when applicable
4. Include provenance tracking
5. Write tests for round-trip exactness

## Common Patterns

### 48-Aligned Tiling
```python
def tile_48(image):
    """Tile image into 48x48 blocks"""
    assert image.shape[0] % 48 == 0
    assert image.shape[1] % 48 == 0
    # Use reshape and transpose, never slice with strides
    return image.reshape(H//48, 48, W//48, 48).transpose(0, 2, 1, 3)
```

### CRT Phase Indexing
```python
def crt_phase48(x, y):
    """Map coordinates to 48-phase index via CRT"""
    d = x % 16  # Dyadic component
    t = y % 3   # Triadic component
    phi = d * 3 + t  # Phase ∈ [0, 47]
    parity = (x + y) & 1  # 0=keven, 1=kodd
    return phi, parity
```

### Reversible Operations
```python
def apply_reversible_op(data, forward_fn, inverse_fn):
    """Apply operation with verification"""
    result = forward_fn(data)
    # Store provenance
    provenance = {"op": forward_fn.__name__, "inverse": inverse_fn.__name__}
    # Verify reversibility in debug mode
    if DEBUG:
        assert np.allclose(inverse_fn(result), data)
    return result, provenance
```

## Testing Requirements

### Must Test
- [ ] 48-alignment validation
- [ ] Reversibility of all operations (round-trip exactness)
- [ ] Parity mask correctness
- [ ] CRT indexing uniformity
- [ ] No aliasing in permutations
- [ ] Provenance completeness

### Performance Targets
- Operations should be `O(n)` with low constants
- Memory usage should scale linearly
- No temporary allocations larger than input
- Prefer in-place operations when possible

## Anti-Patterns to Avoid

### ❌ NEVER DO THIS:
```python
# Lossy downsampling
pooled = F.max_pool2d(x, 2)  # DESTROYS INFORMATION

# Non-reversible convolution  
out = F.conv2d(x, kernel, stride=2)  # CREATES ALIASES

# Ad-hoc resampling
resized = F.interpolate(x, scale_factor=0.5)  # DECIMATES

# Arbitrary normalization
normed = x / x.mean()  # NON-REVERSIBLE
```

### ✅ DO THIS INSTEAD:
```python
# Reversible permutation
downscaled = space_to_depth_2(x)  # PRESERVES ALL INFO

# Unitary mixing
out = unitary_conv(x, U)  # REVERSIBLE

# Exact permutation  
rearranged = space_to_depth_2(x)  # ON-GRID

# Orthogonal projection
projected = orthogonal_project(x, basis)  # PRESERVES NORM
```

## Domain-Specific Guidelines

### Fractal Generation (`fractal_48/`)
- Canvas dimensions ∈ 48ℕ
- Use CRT-based 48-phase scheduler
- Implement keven/kodd parity gating for color
- Export with complete provenance JSON

### Language Processing
- Vowel seeds as base energetic signatures
- 24-state alphabet mapping (B-Y interior states)
- Mirror pairs across M↔N hinge
- Forward operations with adjoint definitions

### Physics Simulations
- keven → structural/Poisson components
- kodd → dynamic/curl components  
- Smooth gates across six axes
- Measurement-first evaluation

## Integration Points

### With External Libraries
- Wrap PyTorch/JAX operations in reversibility checks
- Convert to 48-aligned before processing
- Log all external calls in provenance chain

### With Data Formats
- Prefer lossless formats (PNG, FLAC, exact JSON)
- Include metadata with 48-manifold parameters
- Support round-trip verification

## Documentation Standards

### Function Docstrings
```python
def operation(data: np.ndarray) -> np.ndarray:
    """One-line summary.
    
    Detailed description including 48-manifold relevance.
    
    Args:
        data: Must be 48-aligned array
        
    Returns:
        Transformed data maintaining reversibility
        
    Raises:
        ValueError: If not 48-aligned
        
    Notes:
        - Reversible via `inverse_operation`
        - Preserves keven/kodd separation
    """
```

## References
Key documents in repository:
- `USK_insights_2025-09-09.txt` - Foundational USK concepts
- `00_preamble.md` through `05_synthesis.md` - Theory documentation
- `IMMUNITY.md` - Final synthesis and implementation notes
- `manifold.py` - Reference implementation

## Questions to Ask When Coding
1. Is this operation reversible?
2. Are dimensions 48-aligned?
3. Is parity (keven/kodd) preserved?
4. Can I use a permutation instead of resampling?
5. Is provenance being tracked?
6. Will the inverse operation recover the input exactly?
7. Am I evaluating through the measurement operator M?

## Helpful Snippets

### Validate 48-Alignment
```python
def validate_48(shape):
    for i, dim in enumerate(shape):
        if dim % 48 != 0:
            raise ValueError(f"Dimension {i} ({dim}) not 48-aligned")
```

### Create 48-Aligned Canvas
```python
def make_canvas_48(target_w, target_h):
    w = ((target_w + 47) // 48) * 48
    h = ((target_h + 47) // 48) * 48
    return np.zeros((h, w, 3), dtype=np.float32)
```

### Apply Parity Mask
```python
def apply_parity_mask(data, even_fn, odd_fn):
    mask = get_parity_mask(data.shape)
    return np.where(mask == 0, even_fn(data), odd_fn(data))
```
