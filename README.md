Very, very, very early stage, generative intelligence

# Immune Analog + Folding QA

CLI entrypoint: `immunity.py`

## Quick start

```bash
# Show help
python3 immunity.py --help

# Synthetic folding QA (progress + summary)
python3 immunity.py --no-demo --folding-demo --no-benchmark

# JSON/CSV summaries
python3 immunity.py --no-demo --folding-demo --json-summary - --no-benchmark
python3 immunity.py --no-demo --folding-demo --csv-summary demo.csv --no-benchmark
```

## Real structures (PDB/AFDB)

Single structure:
```bash
# From RCSB by PDB ID and chain
python3 immunity.py --no-demo --pdb-id 1MBN --chain A --stride 16

# From local PDB path
python3 immunity.py --no-demo --pdb-path af_pdbs/AF-P69905-F1-model_v4.pdb --stride 16
```

Batch mode (`--pdb-list` file with one entry per line):
- Each line can be: `PDBID[,CHAIN]` or `/path/to/file.pdb[,CHAIN]`
- Lines starting with `#` are ignored.

```bash
python3 immunity.py --no-demo \
  --pdb-list crystals_list.txt \
  --csv-summary crystal_batch.csv \
  --json-summary crystal_batch.json \
  --no-benchmark
```

## Recommended flags: AF vs Crystal

- __AlphaFold/AFDB (pLDDT in B-factor column)__
  - Mapping: `--bmap plddt`
  - Education: `--educate-first-n 1` (optionally seed a self window)
  - Acceptance gate options:
    - For signature-only experiments: `--assume-folded`
    - For realistic gating: `--folding-threshold 0.7` (tune 0.6–0.75)

  Examples:
  ```bash
  # Batch with pLDDT, assume folded
  python3 immunity.py --no-demo --pdb-list af_list.txt \
    --bmap plddt --assume-folded --educate-first-n 1 \
    --csv-summary af_batch.csv --json-summary af_batch.json --no-benchmark

  # Batch with pLDDT and acceptance threshold
  python3 immunity.py --no-demo --pdb-list af_list.txt \
    --bmap plddt --educate-first-n 1 --folding-threshold 0.7 \
    --csv-summary af_batch_thr07.csv --json-summary af_batch_thr07.json --no-benchmark
  ```

- __Crystal structures (B-factors)__
  - Mapping: `--bmap b_factor`
  - Education: `--educate-first-n 1`
  - Acceptance gate: as above (`--assume-folded` or `--folding-threshold 0.7`)

  Example:
  ```bash
  python3 immunity.py --no-demo --pdb-list crystals_list.txt \
    --bmap b_factor --assume-folded --educate-first-n 1 \
    --csv-summary crystal_batch.csv --json-summary crystal_batch.json --no-benchmark
  ```

## Notes

- __Progress + summaries__: The CLI uses `click` progress bars and structured JSON/CSV outputs.
- __Education placement__: Thymus is educated before creating new cells so they inherit updated self signatures.
- __Threshold logic__: The folding acceptance gate defaults to 0.95; override via `--folding-threshold` or bypass with `--assume-folded` for signature-only experiments.
- __AF/Crystal mapping__: `--bmap` is propagated to both education and evaluation to keep signatures consistent (e.g., `plddt` for AF, `b_factor` for crystals).
