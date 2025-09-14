#!/bin/bash

SEQUENCE="MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"

for s in 1 11 21 31 41; do
  python3 compose_protein.py \
    --sequence "$SEQUENCE" \
    --samples 8 --variability 0.5 --seed $s --window-jitter \
    --save-prefix outputs/ubi_$s --save-format npy
  python3 generate_structure.py \
    --input-prefix outputs/ubi_$s \
    --output-pdb outputs/ubi_$s.pdb \
    --sequence "$SEQUENCE" \
    --refine --refine-iters 300 --refine-step 2.0 --refine-seed 123 \
    --ref-pdb af_pdds/AF-P69905-F1-model_v4.pdb --ref-chain A
done
