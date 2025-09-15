#!/usr/bin/env python3
"""
Quick test to compare averaging vs best-of-N selection.
"""
import sys
import numpy as np
import torch
from bio import pipeline

def test_averaging_vs_best_of_n(sequence, seed=21, samples=8):
    """Test the difference between averaging samples vs selecting the best one."""
    
    print("="*60)
    print("TESTING: AVERAGING vs BEST-OF-N")
    print("="*60)
    
    # Generate samples individually instead of averaging
    all_compositions = []
    all_certainties = []
    
    with torch.no_grad():
        for i in range(samples):
            print(f"Generating sample {i+1}/{samples}...")
            
            # Generate single sample
            mean_comp, certainty = pipeline.compose_sequence(
                sequence,
                samples=1,  # Single sample
                variability=0.5,
                seed=seed + i,  # Different seed for each
                window_jitter=True,
            )
            all_compositions.append(mean_comp)
            all_certainties.append(certainty)
    
    # Compare averaging vs selection strategies
    print(f"\nGenerated {len(all_compositions)} samples")
    
    # Strategy 1: Current (averaging)
    min_windows = min(comp.shape[0] for comp in all_compositions)
    aligned_comps = [comp[:min_windows] for comp in all_compositions]
    ensemble = torch.stack(aligned_comps, dim=0)
    averaged_comp = ensemble.mean(dim=0)
    averaged_certainty = torch.stack([cert[:min_windows] for cert in all_certainties], dim=0).mean(dim=0)
    
    print(f"\nAVERAGED composition shape: {averaged_comp.shape}")
    print(f"AVERAGED certainty mean: {averaged_certainty.mean():.4f}")
    
    # Strategy 2: Select by highest certainty
    certainty_scores = [cert.mean().item() for cert in all_certainties]
    best_idx = np.argmax(certainty_scores)
    best_comp = all_compositions[best_idx][:min_windows]
    best_certainty = all_certainties[best_idx][:min_windows]
    
    print(f"\nBEST sample index: {best_idx}")  
    print(f"BEST composition shape: {best_comp.shape}")
    print(f"BEST certainty mean: {best_certainty.mean():.4f}")
    print(f"Certainty improvement: {best_certainty.mean() - averaged_certainty.mean():.4f}")
    
    # Save both for testing
    torch.save(averaged_comp, "outputs/test_averaged_mean.pt")
    torch.save(averaged_certainty, "outputs/test_averaged_certainty.pt")
    torch.save(best_comp, "outputs/test_best_mean.pt") 
    torch.save(best_certainty, "outputs/test_best_certainty.pt")
    
    print(f"\nSaved outputs/test_averaged_* and outputs/test_best_* for comparison")
    # No return values from tests; rely on printed summaries and saved artifacts

if __name__ == "__main__":
    sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"
    test_averaging_vs_best_of_n(sequence)
