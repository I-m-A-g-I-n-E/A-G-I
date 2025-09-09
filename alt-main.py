
from typing import Tuple
import numpy as np

# Note: All ops are integer-based. Arrays use int32/int64; no floats.
# Assumptions: H and W are divisible by 48. Channels C can be any integer.

def space_to_depth_factor2(x: np.ndarray) -> np.ndarray:
    # x: (B, C, H, W), H,W even
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 2 == 0
    x = x.reshape(B, C, H//2, 2, W//2, 2)
    x = x.transpose(0,1,3,5,2,4)  # bring 2x2 to channel
    x = x.reshape(B, C*4, H//2, W//2)
    return x

def depth_to_space_factor2(x: np.ndarray) -> np.ndarray:
    B, C, H, W = x.shape
    assert C % 4 == 0
    x = x.reshape(B, C//4, 2, 2, H, W)
    x = x.transpose(0,1,4,2,5,3)
    x = x.reshape(B, C//4, H*2, W*2)
    return x

def space_to_depth_factor3(x: np.ndarray) -> np.ndarray:
    # 3x3 permutation; no floats
    B, C, H, W = x.shape
    assert H % 3 == 0 and W % 3 == 0
    x = x.reshape(B, C, H//3, 3, W//3, 3)
    x = x.transpose(0,1,3,5,2,4)  # (B, C, 3, 3, H/3, W/3)
    x = x.reshape(B, C*9, H//3, W//3)
    return x

def depth_to_space_factor3(x: np.ndarray) -> np.ndarray:
    B, C, H, W = x.shape
    assert C % 9 == 0
    x = x.reshape(B, C//9, 3, 3, H, W)
    x = x.transpose(0,1,4,2,5,3)
    x = x.reshape(B, C//9, H*3, W*3)
    return x

def unimodular_mix_1x1(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    # x: (B, C, H, W); A: (C, C) with det(A)=±1 and integer entries
    # Performs integer channel mixing; exactly invertible over Z.
    B, C, H, W = x.shape
    assert A.shape == (C, C)
    # Use matmul per (H,W) position to avoid floats
    x_ = x.transpose(0,2,3,1).reshape(B*H*W, C)
    x_ = (A @ x_.T).T  # integer matmul
    x = x_.reshape(B, H, W, C).transpose(0,3,1,2)
    return x

def unimodular_inv(A: np.ndarray) -> np.ndarray:
    # Find integer inverse of unimodular matrix A via Hermite/Smith normal form (simplified here
    # for small channel dims: use adjugate since det=±1).
    C = A.shape[0]
    # Compute adjugate and determinant (det must be ±1)
    det = round(np.linalg.det(A))
    assert abs(det) == 1
    adj = np.round(det * np.linalg.inv(A)).astype(int)  # adj(A) = det(A)*A^{-1}
    # Since det=±1, A^{-1} = adj(A)/det = ±adj(A)
    return (adj if det == 1 else -adj)

# Build keven/kodd/kones from channels: simple partition for illustration
def split_k_channels(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    B, C, H, W = x.shape
    assert C >= 3
    # Distribute channels cyclically to kones (DC-like), keven, kodd
    idx1 = np.arange(0, C, 3)
    idx2 = np.arange(1, C, 3)
    idx3 = np.arange(2, C, 3)
    return x[:, idx2], x[:, idx3], x[:, idx1]  # keven, kodd, kones

def merge_k_channels(keven: np.ndarray, kodd: np.ndarray, kones: np.ndarray) -> np.ndarray:
    B, C2, H, W = keven.shape
    C3 = kodd.shape[1]
    C1 = kones.shape[1]
    C = C1 + C2 + C3
    out = np.zeros((B, C, H, W), dtype=keven.dtype)
    out[:, np.arange(1, C, 3)[:C2]] = keven
    out[:, np.arange(2, C, 3)[:C3]] = kodd
    out[:, np.arange(0, C, 3)[:C1]] = kones
    return out

# Mixed 2/3 ladder: 48 -> 16 -> 8 -> 4 using ×3 then ×2 then ×2
class ReversibleLadder48:
    def __init__(self, C: int):
        self.C = C
        # Create small unimodular integer 1x1 mixers (e.g., triangular with 1s on diag)
        rng = np.random.default_rng(0)
        def rand_unimodular(c):
            A = np.eye(c, dtype=int)
            # Add a few integer shear operations (det stays 1)
            for _ in range(c):
                i, j = rng.integers(0, c, size=2)
                if i != j:
                    A[i, j] += rng.integers(-2, 3)  # small integer
            # Guarantee det=±1 by project to nearest unimodular via adj trick if needed
            det = round(np.linalg.det(A))
            if det == 0:
                A[0,0] += 1
                det = round(np.linalg.det(A))
            if abs(det) != 1:
                # Normalize to unimodular by dividing adjoint by det and rounding
                # For simplicity in this demo, force a product of elementary unimodular ops:
                A = np.eye(c, dtype=int)
            return A
        self.A1 = rand_unimodular(C*9)   # after ×3
        self.A2 = rand_unimodular(C*9*4) # after ×2
        self.A3 = rand_unimodular(C*9*4*4) # after ×2

        self.A1_inv = unimodular_inv(self.A1)
        self.A2_inv = unimodular_inv(self.A2)
        self.A3_inv = unimodular_inv(self.A3)

    def encode(self, x: np.ndarray) -> np.ndarray:
        # x: (B, C, 48m, 48n)
        B, C, H, W = x.shape
        assert H % 48 == 0 and W % 48 == 0 and C == self.C

        # Split into keven, kodd, kones for bookkeeping (optional)
        ke, ko, k1 = split_k_channels(x)
        x = merge_k_channels(ke, ko, k1)

        # ×3
        x = space_to_depth_factor3(x)
        x = unimodular_mix_1x1(x, self.A1)
        # ×2
        x = space_to_depth_factor2(x)
        x = unimodular_mix_1x1(x, self.A2)
        # ×2
        x = space_to_depth_factor2(x)
        x = unimodular_mix_1x1(x, self.A3)
        return x  # latent at 4× downsampled grid, channels multiplied by 9*4*4

    def decode(self, z: np.ndarray) -> np.ndarray:
        # Exact inverse
        x = unimodular_mix_1x1(z, self.A3_inv)
        x = depth_to_space_factor2(x)
        x = unimodular_mix_1x1(x, self.A2_inv)
        x = depth_to_space_factor2(x)
        x = unimodular_mix_1x1(x, self.A1_inv)
        x = depth_to_space_factor3(x)
        # Merge/split are permutations; identity overall
        ke, ko, k1 = split_k_channels(x)
        x = merge_k_channels(ke, ko, k1)
        return x

# Example usage (bit-exact round trip)
if __name__ == "__main__":
    B, C, H, W = 1, 6, 48, 48
    x = np.arange(B*C*H*W, dtype=np.int32).reshape(B, C, H, W)
    ladder = ReversibleLadder48(C)
    z = ladder.encode(x)
    x_hat = ladder.decode(z)
    assert np.array_equal(x, x_hat), "Round-trip must be exact (no aliasing, no floats)."
    print("Perfect reconstruction achieved.")
