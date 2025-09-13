#!/usr/bin/env python3
"""
Diagnostic script for the Sonification System.

Test 1: Checks if the input composition vector from the Composer is silent.
Test 2: Checks if the Sonifier can produce sound from a known, non-silent test signal.
"""
import torch
import numpy as np
import os
import sys

# Allow running as a script: `python bio/debug_sonifier.py`
# When executed this way, sys.path[0] is the bio/ directory, so `import bio.*` fails.
# Add the project root to sys.path so that `bio` is importable as a package.
if __package__ in (None, ""):
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)

from bio.sonifier import TrinitySonifier

def run_diagnostics():
    print("🔬 Conducting Sonification System Diagnostics...")
    
    # --- Test 1: Verify the Input Score from the Composer ---
    print("\n--- [Test 1: Analyzing Composer's Score] ---")
    input_prefix = "outputs/ubiquitin_ensemble"
    mean_comp_path = f"{input_prefix}_mean.npy"
    
    if not os.path.exists(mean_comp_path):
        print(f"❌ ERROR: Input file not found at {mean_comp_path}")
        print("   ACTION: Please run `compose_protein.py` first to generate the input.")
        return

    mean_composition = torch.from_numpy(np.load(mean_comp_path))
    
    # Check statistics of the input vector
    std_dev = torch.std(mean_composition).item()
    abs_max = torch.max(torch.abs(mean_composition)).item()

    print(f"   - Loaded mean composition vector with shape: {mean_composition.shape}")
    print(f"   - Signal Standard Deviation: {std_dev:.6f}")
    print(f"   - Signal Max Absolute Value: {abs_max:.6f}")

    is_input_silent = std_dev < 1e-6
    if is_input_silent:
        print("\n🔥 DIAGNOSIS: The Composer's score is SILENT.")
        print("   The root cause is likely in `bio/composer.py`.")
        print("   The Sonifier is likely not the problem.")
        return
    else:
        print("\n✅ VERDICT: The Composer's score is VALID and contains a signal.")
        print("   Proceeding to test the instrument...")

    # --- Test 2: Verify the Sonifier Instrument ---
    print("\n--- [Test 2: Testing Sonifier with a Perfect Note] ---")
    
    # Create a known, non-silent test signal (a simple sine wave's components)
    num_windows_test = mean_composition.shape[0]
    test_composition = torch.zeros(num_windows_test, 48)
    test_composition[:, 1] = 1.0  # Excite the first `kodd` harmonic
    
    # Create dummy inputs for the sonifier
    test_kore = torch.ones(48) / np.sqrt(48)
    test_certainty = torch.ones(num_windows_test) * 0.9
    test_dissonance = torch.zeros(num_windows_test)
    test_weights = {'kore': 1.0, 'cert': 1.0, 'diss': 1.0}
    
    sonifier = TrinitySonifier(bpm=96)
    test_waveform = sonifier.sonify_composition_3ch(
        test_composition, test_kore, test_certainty, test_dissonance, test_weights
    )

    # Check the output waveform directly
    output_max_amp = np.max(np.abs(test_waveform))
    print(f"   - Generated test waveform with max amplitude: {output_max_amp:.6f}")
    
    is_output_silent = output_max_amp < 1e-6
    if is_output_silent:
        print("\n🔥 DIAGNOSIS: The Sonifier instrument is BROKEN.")
        print("   It produced silence from a valid test signal.")
        print("   The root cause is in `bio/sonifier.py`.")
    else:
        print("\n✅ VERDICT: The Sonifier instrument is WORKING correctly.")
        # Save the audible test file for confirmation
        os.makedirs('outputs', exist_ok=True)
        sonifier.save_wav(test_waveform, "outputs/sonifier_diagnostic_output.wav")
        print("   An audible test file `sonifier_diagnostic_output.wav` has been saved.")
        print("   If the main output is silent, the issue may be in how the real data interacts with the Sonifier.")

if __name__ == "__main__":
    run_diagnostics()
