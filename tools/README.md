# Tools Directory

This directory contains utility tools for analyzing and working with the 48-Manifold system.

## Scripts

- **`analyze_sqlite_metrics.py`** - SQLite-based metrics analyzer for protein refinement batches
  - Provides insights and analysis of metrics stored in SQLite databases
  - Similar to `tools/analyze_metrics.py` but works directly with SQLite

## Usage

```bash
python tools/analyze_sqlite_metrics.py --db-path path/to/metrics.db
```

These tools complement the main workflow by providing analysis capabilities for batch processing and metrics evaluation.