#!/usr/bin/env python3
"""
Run WHAM on all completed silica_dense windows, then plot PMF.
Works on any machine — set WINS_DIR and GMX to local paths.
A window is complete if pullx.xvg has >= 19900 data lines.
"""

import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WINS_DIR = os.path.dirname(os.path.abspath(__file__))
WHAM_DIR = os.path.join(os.path.dirname(WINS_DIR), "wham")
GMX      = "/apps/codes/gromacs-2023/bin/gmx_mpi"   # change to "gmx_mpi" on remote machine

os.makedirs(WHAM_DIR, exist_ok=True)

# --- Collect completed windows ---
tpr_list, pullx_list = [], []
for win_dir in sorted(os.listdir(WINS_DIR)):
    if not win_dir.startswith("win_"):
        continue
    wd = os.path.join(WINS_DIR, win_dir)
    tpr   = os.path.join(wd, "window.tpr")
    pullx = os.path.join(wd, "pullx.xvg")
    if not (os.path.exists(tpr) and os.path.exists(pullx)):
        continue
    nframes = sum(1 for l in open(pullx) if not l.startswith(('#', '@')) and l.strip())
    if nframes < 19900:
        print(f"  SKIP {win_dir} — only {nframes} frames (incomplete)")
        continue
    tpr_list.append(tpr)
    pullx_list.append(pullx)

print(f"Running WHAM on {len(tpr_list)} complete windows...")

with open(os.path.join(WHAM_DIR, "tpr.dat"),   'w') as f: f.write('\n'.join(tpr_list)   + '\n')
with open(os.path.join(WHAM_DIR, "pullx.dat"), 'w') as f: f.write('\n'.join(pullx_list) + '\n')

cmd = (f"{GMX} wham "
       f"-it {WHAM_DIR}/tpr.dat "
       f"-ix {WHAM_DIR}/pullx.dat "
       f"-o {WHAM_DIR}/pmf.xvg "
       f"-hist {WHAM_DIR}/hist.xvg "
       f"-unit kJ -b 2000 -tol 1e-4 -nBootstrap 100 -bs-method b-hist "
       f"-min 2.4 -max 8.6")

r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
with open(os.path.join(WHAM_DIR, "wham.log"), 'w') as f:
    f.write(r.stdout + r.stderr)

if r.returncode != 0:
    print("WHAM failed. Check wham/wham.log")
    print(r.stderr[-500:])
    exit(1)

print("WHAM done. Plotting...")

# --- Read xvg ---
def read_xvg(path):
    x, cols = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or line.startswith('@'):
                continue
            parts = line.split()
            if parts:
                x.append(float(parts[0]))
                cols.append([float(v) for v in parts[1:]])
    return np.array(x), np.array(cols)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# PMF
r, pmf_cols = read_xvg(f"{WHAM_DIR}/pmf.xvg")
pmf = pmf_cols[:, 0]
# Exclude boundary zeros before normalizing
mask = pmf != 0.0
r_c = r[mask]; pmf_c = pmf[mask]
ref = pmf_c[-10:].mean()
pmf_norm = pmf_c - ref
ax1.plot(r_c, pmf_norm, 'steelblue', lw=2)
if pmf_cols.shape[1] > 1:
    err_c = pmf_cols[:, 1][mask]
    ax1.fill_between(r_c, pmf_norm - err_c, pmf_norm + err_c, alpha=0.3, color='steelblue', label='bootstrap error')
ax1.set_xlabel("Distance (nm)", fontsize=12)
ax1.set_ylabel("PMF (kJ/mol)", fontsize=12)
ax1.set_title(f"PMF — silica_dense ({len(tpr_list)} windows)", fontsize=11)
ax1.axhline(0, color='gray', lw=0.8, ls='--')
ax1.grid(alpha=0.3)
ax1.legend()

# Histograms
xi, hist_cols = read_xvg(f"{WHAM_DIR}/hist.xvg")
for col in range(hist_cols.shape[1]):
    y = hist_cols[:, col]
    if y.max() > 0:
        ax2.plot(xi, y / y.max(), alpha=0.4, lw=0.8)
ax2.set_xlabel("Distance (nm)", fontsize=12)
ax2.set_ylabel("Normalised count", fontsize=12)
ax2.set_title("Histograms — silica_dense", fontsize=11)
ax2.grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(WHAM_DIR, "pmf_plot.png")
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
