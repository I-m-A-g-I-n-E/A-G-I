#!/usr/bin/env python3
import sys
import torch
from bio.sonifier import TrinitySonifier
from bio.utils import load_ensemble

if __name__ == '__main__':
    prefix = sys.argv[1] if len(sys.argv) > 1 else 'outputs/ubiquitin_ensemble'
    out_base = sys.argv[2] if len(sys.argv) > 2 else 'outputs/ubi_audio'

    mean, cert = load_ensemble(prefix)
    # Ensure shapes: mean -> [W,48]; cert -> [W]
    if mean.ndim == 1 and mean.numel() == 48:
        mean = mean.unsqueeze(0)
    cert = cert.view(-1)

    son = TrinitySonifier()
    L, C, R = son.sonify_composition(mean, cert)
    son.save_wav(L, out_base + '_L.wav')
    son.save_wav(C, out_base + '_C.wav')
    son.save_wav(R, out_base + '_R.wav')
    print('Saved:', out_base + '_L.wav', out_base + '_C.wav', out_base + '_R.wav')
