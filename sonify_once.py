#!/usr/bin/env python3
import numpy as np
import torch
from bio.sonifier import TrinitySonifier
import sys

if __name__ == '__main__':
    prefix = sys.argv[1] if len(sys.argv) > 1 else 'outputs/ubiquitin_ensemble'
    out_base = sys.argv[2] if len(sys.argv) > 2 else 'outputs/ubi_audio'
    mean = torch.from_numpy(np.load(prefix + '_mean.npy')).to(torch.float32)
    if mean.ndim == 1 and mean.numel() == 48:
        mean = mean.unsqueeze(0)
    cert_path = prefix + '_certainty.npy'
    try:
        cert = torch.from_numpy(np.load(cert_path)).to(torch.float32)
    except Exception:
        cert = torch.ones((mean.shape[0],), dtype=torch.float32)
    son = TrinitySonifier()
    L, C, R = son.sonify_composition(mean, cert)
    son.save_wav(L, out_base + '_L.wav')
    son.save_wav(C, out_base + '_C.wav')
    son.save_wav(R, out_base + '_R.wav')
    print('Saved:', out_base + '_L.wav', out_base + '_C.wav', out_base + '_R.wav')
