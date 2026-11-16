"""Capture files -> per-matrix Hessian files for SPECPILOT_D3K_GPTQ (the data-aware Q copy's calibration).
(From the SpecPilot PEAR campaign's kgemm bench/hess.py, unchanged apart from this docstring; needs torch + one GPU.)
usage: calib_hess.py --calib <dir> --out <hdir> --heldout-from <file index> [--names <regex>]
<dir>/<module name>.<i>.pt (bf16 rows); indices >= heldout-from go to the held-out set (<hdir>/heldout/<name>.pt,
first 4096 rows); rows of all-zero files (startup warmup) are skipped."""
import argparse
import glob
import os
import re
from collections import defaultdict

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--heldout-from", type=int, required=True)
    ap.add_argument("--names", default=".")
    a = ap.parse_args()
    files = defaultdict(list)
    for p in glob.glob(os.path.join(a.calib, "*.pt")):
        name, i = p.rsplit("/", 1)[1].rsplit(".", 2)[0], int(p.rsplit(".", 2)[1])
        if re.search(a.names, name):
            files[name].append((i, p))
    os.makedirs(os.path.join(a.out, "heldout"), exist_ok=True)
    for name in sorted(files):
        H, n, held = None, 0, []
        for i, p in sorted(files[name]):
            x = torch.load(p).cuda()
            if x.shape[0] < 256 or x.float().abs().max() == 0:
                continue
            if i >= a.heldout_from:
                held.append(x)
                continue
            xf = x.float()
            h = xf.t() @ xf
            H = h if H is None else H + h
            n += x.shape[0]
        torch.save({"H": (H / n).cpu(), "n": n}, os.path.join(a.out, f"{name}.H.pt"))
        if held:
            torch.save(torch.cat(held)[:4096].cpu(), os.path.join(a.out, "heldout", f"{name}.pt"))
        print(f"{name}: n={n} heldout={sum(h.shape[0] for h in held)}", flush=True)
        del H
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
