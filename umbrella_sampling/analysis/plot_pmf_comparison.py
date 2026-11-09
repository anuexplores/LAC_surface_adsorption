#!/usr/bin/env python3
"""
Comparison PMF plot for silica_dense, lys_silica_dense, aptes_silica_dense.

Set INDIVIDUAL_PANELS = True  → one subplot per system (each with its own axes).
Set INDIVIDUAL_PANELS = False → all systems on one shared axes.

Per-system settings in SYSTEMS dict:
  xlim      : (min, max) to fix axis view, or None to auto-scale.
  ylim      : (min, max) to fix axis view, or None to auto-scale.
  xmin_data : float — discard all data points with r < this value before plotting.
  xmax_data : float — discard all data points with r > this value before plotting.
  Set xmin_data / xmax_data to None to keep all data.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = "/Data/Anushka/Project4/ph5/orients/orient1/US"

L_Z_APTES = 20.67995   # nm — box z-length for aptes_silica_dense

# ── per-system settings ────────────────────────────────────────────────────────
# xlim / ylim : set to (min, max) to fix that axis for this system,
#               or None to auto-scale.
SYSTEMS = [
        {
        "name":       "silica_dense",
        "wham":       f"{BASE}/silica_dense/wham_orig",
        "color":      "steelblue",
        "label":      "Silica",
        "lz":         None,
        "xmin_data":  None,    # only plot data from 3.3 nm onwards
        "xmax_data":  None,
        "xlim":       None,
        "ylim":       None,
    },
    {
        "name":       "aptes_silica_dense",
        "wham":       f"{BASE}/aptes_silica_dense/wham_orig",
        "color":      "seagreen",
        "label":      "Lysine-functionalised silica",
        "lz":         L_Z_APTES,
        "xmin_data":  3.3,   # trim data below this r (nm); None = keep all
        "xmax_data":  None,   # trim data above this r (nm); None = keep all
        "xlim":       None,   # axis view limit
        "ylim":       None,
    },
    {
        "name":       "lys_silica_dense",
        "wham":       f"{BASE}/lys_silica_dense/wham_orig",
        "color":      "darkorange",
        "label":      "Aptes-functionalised silica",
        "lz":         None,
        "xmin_data":  None,
        "xmax_data":  None,
        "xlim":       None,
        "ylim":       None,
    },
]

# ── global plot settings ───────────────────────────────────────────────────────
INDIVIDUAL_PANELS = False   # True → one panel per system; False → combined plot

# Fallback limits for combined plot (used only when system's xlim/ylim is None)
XMIN, XMAX = 2.5, 11.0
YMIN, YMAX = None, None

SHOW_ERROR   = True
FIGSIZE_COMBINED   = (8, 5)
FIGSIZE_INDIVIDUAL = (6 * len(SYSTEMS), 5)
DPI          = 150
OUTFILE      = f"{BASE}/pmf_comparison.png"
# ─────────────────────────────────────────────────────────────────────────────


def read_xvg(path):
    x, cols = [], []
    with open(path) as f:
        for line in f:
            if line.startswith(('#', '@')):
                continue
            parts = line.split()
            if parts:
                try:
                    x.append(float(parts[0]))
                    cols.append([float(v) for v in parts[1:]])
                except ValueError:
                    pass
    return np.array(x), np.array(cols)


def load_pmf(s):
    pmf_file = f"{s['wham']}/pmf.xvg"
    r_raw, pmf_cols = read_xvg(pmf_file)
    r = r_raw + s["lz"] if s["lz"] else r_raw
    idx = np.argsort(r)
    r, pmf_cols = r[idx], pmf_cols[idx]
    # Exclude boundary zeros before normalising
    mask = pmf_cols[:, 0] != 0.0
    r, pmf_cols = r[mask], pmf_cols[mask]
    pmf = pmf_cols[:, 0]
    pmf -= pmf[-10:].mean()
    err = pmf_cols[:, 1] if pmf_cols.shape[1] > 1 else None
    # Per-system data trim
    keep = np.ones(len(r), dtype=bool)
    if s.get("xmin_data") is not None:
        keep &= r >= s["xmin_data"]
    if s.get("xmax_data") is not None:
        keep &= r <= s["xmax_data"]
    r, pmf = r[keep], pmf[keep]
    err = err[keep] if err is not None else None
    return r, pmf, err


def plot_on_ax(ax, s, r, pmf, err):
    ax.plot(r, pmf, '-', color=s["color"], lw=2, label=s["label"])
    if SHOW_ERROR and err is not None:
        ax.fill_between(r, pmf - err, pmf + err, alpha=0.15, color=s["color"])
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel("Protein–surface distance (nm)", fontsize=12)
    ax.set_ylabel("PMF (kJ/mol)", fontsize=12)
    ax.grid(alpha=0.3)


# ── load all PMFs ──────────────────────────────────────────────────────────────
pmf_data = []
for s in SYSTEMS:
    r, pmf, err = load_pmf(s)
    pmf_data.append((r, pmf, err))
    print(f"{s['label']:30s}  x: {r.min():.2f}–{r.max():.2f} nm  "
          f"min: {pmf.min():.1f} kJ/mol at {r[np.argmin(pmf)]:.2f} nm")

# ── plot ───────────────────────────────────────────────────────────────────────
if INDIVIDUAL_PANELS:
    fig, axes = plt.subplots(1, len(SYSTEMS), figsize=FIGSIZE_INDIVIDUAL, sharey=False)
    for ax, s, (r, pmf, err) in zip(axes, SYSTEMS, pmf_data):
        plot_on_ax(ax, s, r, pmf, err)
        ax.set_title(s["label"], fontsize=12)
        ax.legend(fontsize=10)
        # Per-system limits
        xl = s["xlim"] if s["xlim"] else (XMIN, XMAX)
        yl = s["ylim"] if s["ylim"] else (YMIN, YMAX)
        if xl[0] is not None or xl[1] is not None: ax.set_xlim(*xl)
        if yl[0] is not None or yl[1] is not None: ax.set_ylim(*yl)
else:
    fig, ax = plt.subplots(figsize=FIGSIZE_COMBINED)
    for s, (r, pmf, err) in zip(SYSTEMS, pmf_data):
        plot_on_ax(ax, s, r, pmf, err)
    ax.set_title("PMF comparison — silica vs lys-silica vs APTES-silica", fontsize=12)
    ax.legend(fontsize=11)
    # Global limits (per-system xlim/ylim ignored in combined mode)
    if XMIN is not None or XMAX is not None: ax.set_xlim(XMIN, XMAX)
    if YMIN is not None or YMAX is not None: ax.set_ylim(YMIN, YMAX)

plt.tight_layout()
plt.savefig(OUTFILE, dpi=DPI)
print(f"\nSaved: {OUTFILE}")
