#!/usr/bin/env bash
set -euo pipefail

# Paths
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/outputs"
INPUT_PREFIX="${OUT_DIR}/ubiq"
SEQ_FILE="${OUT_DIR}/ubiquitin.seq"
REF_PDB="${OUT_DIR}/1UBQ.pdb"

# Ensure outputs directory exists
mkdir -p "${OUT_DIR}"

echo "== Ensuring dependencies =="
python3 -m pip install -r "${ROOT_DIR}/requirements.txt"

echo "== Checking ensemble inputs =="
if [[ ! -f "${INPUT_PREFIX}_mean.npy" ]] || [[ ! -f "${INPUT_PREFIX}_certainty.npy" ]]; then
  echo "ERROR: Missing ensemble files:"
  echo "  - ${INPUT_PREFIX}_mean.npy"
  echo "  - ${INPUT_PREFIX}_certainty.npy"
  echo "Aborting."
  exit 1
fi

echo "== Checking sequence file =="
if [[ ! -f "${SEQ_FILE}" ]]; then
  echo "ERROR: Missing sequence file: ${SEQ_FILE}"
  echo "Aborting."
  exit 1
fi

echo "== Fetching reference PDB (if missing) =="
if [[ ! -f "${REF_PDB}" ]]; then
  curl -L -o "${REF_PDB}" "https://files.rcsb.org/download/1UBQ.pdb"
fi

echo "== Running QUICK3 refinement with stronger separation and no multiprocessing =="
cd "${ROOT_DIR}"

python3 agi.py structure \
  --input-prefix "${INPUT_PREFIX}" \
  --output-pdb "${OUT_DIR}/ubiq_quick3.pdb" \
  --sequence-file "${SEQ_FILE}" \
  --refine --refine-iters 900 --refine-seed 42 \
  --w-clash 16.0 --w-ca 2.0 --w-neighbor-ca 4.0 --w-nonadj-ca 5.0 --w-dihedral 1.5 \
  --num-workers 0 --eval-batch 128 \
  --critical-override-iters 120 --spacing-cross-mode \
  --spacing-max-attempts 1000 --spacing-top-bins 8 \
  --final-attempts 2000 \
  --neighbor-threshold 3.3 \
  --timeout-sec 900 \
  --sonify-3ch --audio-wav "${OUT_DIR}/ubiq_quick3.wav" \
  --reference-pdb "${REF_PDB}"

echo "== Done =="
echo "Check:"
echo "  PDB: ${OUT_DIR}/ubiq_quick3.pdb"
echo "  Refined PDB: ${OUT_DIR}/ubiq_quick3_refined.pdb (created by CLI)"
echo "  QC JSON: ${OUT_DIR}/ubiq_quick3_qc.json (and refined version)"
echo "  WAV: ${OUT_DIR}/ubiq_quick3_initial.wav and ${OUT_DIR}/ubiq_quick3_refined.wav (if refine enabled)"q