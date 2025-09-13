# A-G-I: 48-Manifold Research Playground

Practical demonstrations of the 48-basis manifold (48 = 2^4 × 3) applied to reversible computation, harmonic composition, immune-analog QA, structure generation, and sonification.

This README gives runnable examples for each major component. All examples reflect the current code in this repo.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- macOS with Apple Silicon is supported; several modules use PyTorch MPS if available.
- No external bioinformatics libraries are required. PDB parsing and feature extraction are implemented locally.

## Project layout (selected)

- `manifold.py` — Core 48-manifold primitives: `Fractal48Layer`, `KEven/KOdd/kull`, `SixAxisState`.
- `main.py` — Minimal algebraic demo (`Fractal48Transfer`) showing invertible steps and coupling.
- `fractal48_torch.py` — Full PyTorch autoencoder, benchmarks, and examples.
- `bio/composer.py` — `HarmonicPropagator`: map an AA sequence to windowed 48D Composition Vectors.
- `bio/conductor.py` — `Conductor`: map composition to torsions, build N–CA–C backbone, QC, and refinement.
- `bio/sonifier.py` — `TrinitySonifier`: 3-channel audio from composition (L/R parity, C center/kore).
- `compose_protein.py` — CLI for ensemble composition and saving mean/certainty.
- `generate_structure.py` — CLI for backbone generation (+ optional refinement + optional sonification).
- `live_audition.py` — Interactive loop to compose/conduct/refine/sonify.
- `immunity.py` — Immune-system analog, CLI demos for synthetic and real PDB QA.
- `tests/` — See `test_fractal48.py`, `test_immunity_unit.py`, `test_immunity_randomized.py` for properties and robustness.

## Quick starts

### 1) Compose → Structure → Sonify

Use the provided CLIs to go end-to-end. Ensure your sequence has length ≥ 48.

```bash
# Phase 2: Compose an AA sequence into windowed 48D Composition Vectors
python3 compose_protein.py \
  --sequence MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG \
  --samples 8 --variability 0.5 --seed 42 --window-jitter \
  --save-prefix outputs/ubiquitin_ensemble --save-format npy

# Phase 3: Generate a 3D backbone (N–CA–C) and QC; optionally refine
python3 generate_structure.py \
  --input-prefix outputs/ubiquitin_ensemble \
  --output-pdb outputs/ubiquitin_fold_full.pdb \
  --sequence MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRLRGG \
  --refine --refine-iters 300 --refine-step 2.0 --refine-seed 123

# Optional: 3-channel sonification directly from generate_structure
python3 generate_structure.py \
  --input-prefix outputs/ubiquitin_ensemble \
  --output-pdb outputs/ubiquitin_fold_full.pdb \
  --sequence MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRLRGG \
  --sonify-3ch --audio-wav outputs/ubiquitin.wav \
  --bpm 96 --stride-ticks 16 --amplify 1.0 --wc-kore 1.5 --wc-cert 1.0 --wc-diss 2.5

# Or use the helper to export separate L/C/R stems
python3 sonify_once.py outputs/ubiquitin_ensemble outputs/ubi_audio
```

Artifacts:

- `outputs/ubiquitin_ensemble_mean.npy` and `_certainty.npy` from composition.
- `outputs/ubiquitin_fold_full.pdb` (+ `_initial.pdb`) from conduction.
- QC JSON: `outputs/ubiquitin_fold_full_qc.json` (+ `_initial_qc.json`, and `_refined_qc.json` if refinement).
- Audio: `outputs/ubiquitin.wav` or `outputs/ubi_audio_{L,C,R}.wav`.

### 2) Live Audition (interactive)

```bash
python3 live_audition.py \
  --input-prefix outputs/ubiquitin_ensemble \
  --sequence-file path/to/seq.txt \
  --refine
```

Then interact with the prompt:

- `play` to render a performance (build PDB, run QC, export stems).
- `bpm=100`, `stride_ticks=12`, `clash=15`, etc., to tweak parameters.
- `exit` to quit.

### 3) Immune Analog + Folding QA (CLI)

Show help and run the synthetic QA demo using local adapters in `bio/protein_folding.py`:

```bash
python3 immunity.py --help
python3 immunity.py --no-demo --folding-demo --no-benchmark

# JSON/CSV summaries
python3 immunity.py --no-demo --folding-demo --json-summary - --no-benchmark
python3 immunity.py --no-demo --folding-demo --csv-summary demo.csv --no-benchmark
```

Real structures (RCSB PDB or local PDB path):

```bash
# Single structure from RCSB by PDB ID and chain
python3 immunity.py --no-demo --pdb-id 1MBN --chain A --stride 16

# Single structure from local file
python3 immunity.py --no-demo --pdb-path af_pdbs/AF-P69905-F1-model_v4.pdb --stride 16

# Batch mode with a list file (one entry per line: PDBID[,CHAIN] or /path/to/file.pdb[,CHAIN])
python3 immunity.py --no-demo \
  --pdb-list crystals_list.txt \
  --csv-summary crystal_batch.csv \
  --json-summary crystal_batch.json \
  --no-benchmark
```

Recommended flags when mapping experimental sources:

- AlphaFold/AFDB (pLDDT in B-factor column)
  - `--bmap plddt`
  - `--educate-first-n 1` (seed thymus with a self window)
  - Acceptance gate options:
    - Signature-only experiments: `--assume-folded`
    - Realistic gating: `--folding-threshold 0.7` (tune 0.6–0.75)

- Crystal structures (B-factors)
  - `--bmap b_factor`
  - `--educate-first-n 1`
  - Gate as above (`--assume-folded` or `--folding-threshold 0.7`)

### 4) 48-Manifold Demonstrations

Algebraic demo and bijective transfer (no floats):

```bash
python3 main.py
```

PyTorch encoder/decoder + benchmarks:

```bash
python3 fractal48_torch.py
```

Run a fast subset of tests that highlight exact transfer vs transform:

```bash
python3 test_fractal48.py quick
```

## Programmatic usage

Below are minimal Python snippets using the core APIs. You can run them in a notebook or a small script.

### Harmonic composition → Backbone → PDB

```python
from bio.composer import HarmonicPropagator
from bio.conductor import Conductor
import torch

seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRLRGG"

# 1) Compose to 48D windows
composer = HarmonicPropagator(n_layers=4, variability=0.3, seed=42, window_jitter=True)
comp = composer(seq)              # shape: (num_windows, 48)

# 2) Conduct to backbone (+ QC)
cond = Conductor()
backbone, phi, psi, modes = cond.build_backbone(comp, sequence=seq)
cond.save_to_pdb(backbone, "outputs/example.pdb")
qc = cond.quality_check(backbone, phi, psi, modes)
print(qc["summary"])  # basic sanity metrics
```

### Sonify 3 channels (L/R parity, C center/kore)

```python
from bio.sonifier import TrinitySonifier
from bio.key_estimator import estimate_key_and_modes

W = comp.shape[0]
kore, _ = estimate_key_and_modes(comp, seq)
certainty = torch.ones((W,), dtype=torch.float32)
dissonance = torch.zeros((W,), dtype=torch.float32)

son = TrinitySonifier(bpm=96.0, stride_ticks=16)
wave = son.sonify_composition_3ch(comp, kore, certainty, dissonance, {
    'kore': 1.5, 'cert': 1.0, 'diss': 2.5
})
son.save_wav(wave, "outputs/example_3ch.wav")
```

### Use `Fractal48Layer` in a PyTorch model

```python
import torch
import torch.nn as nn
from manifold import Fractal48Layer

class Tiny48Net(nn.Module):
    def __init__(self, channels=48):
        super().__init__()
        self.frac = Fractal48Layer(channels)
        self.act = nn.GELU()
        self.head = nn.Conv2d(channels, channels, 1)
    def forward(self, x):
        x = self.frac(x)
        x = self.act(x)
        return self.head(x)

x = torch.randn(2, 48, 1, 48)
net = Tiny48Net()
y = net(x)
print(y.shape)
```


## Tips and notes

- The system is strictly 48-aligned. Windowing in `HarmonicPropagator` uses size 48 with stride 16.
- `Conductor` exports either CA traces or full N–CA–C backbones depending on the input array shape.
- QC reports include simple geometry checks (bond lengths/angles, CA–CA distances, clash detection).
- Sonification can run without SciPy; a simple fallback is used when `scipy` is unavailable.
- All modules are torch/numpy-native; no heavy external dependencies.


## Troubleshooting

- If PyTorch MPS is unavailable, the code will fall back to CPU where applicable.
- Make sure your sequence length is ≥ 48; otherwise `HarmonicPropagator` will raise a ValueError.
- If batch QA cannot fetch a PDB ID, ensure network access or switch to `--pdb-path` for local files.


## License

Research code. No license specified; treat as all rights reserved unless otherwise stated.
