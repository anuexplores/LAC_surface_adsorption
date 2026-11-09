#!/usr/bin/env python3
"""
MSM-reweighted TICA free energy landscape — silica system.
Uses deeptime (laccase-msm conda env). No pyemma needed.

Run:
    conda activate laccase-msm
    python3 plot_reweighted_tica.py

For each ligand, saves reweighted_figs/reweighted_tica_{ligand}.png
inside its msm-results directory.
"""

import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
BASE         = Path("/Data/Anushka/Project4/ph5/orients/orient1/tleap/silica")
RESULTS_BASE = BASE / "msm-results"
SYSTEM_LABEL = "Silica"

LIGANDS = [
    "aniline", "25xylidine", "benzene", "catechol", "hydroquinone",
    "hydroxyquinol", "phenol", "phloroglucinol", "pyrogallol", "resorcinol",
]

kT        = 2.479   # kJ/mol at 300 K
NBINS     = 80
NCONTOURS = 20
CMAP      = "nipy_spectral"
DPI       = 150

# ── font & tick customisation ─────────────────────────────────────────────────
XLABEL_FONTSIZE  = 22     # x-axis title font size
YLABEL_FONTSIZE  = 22     # y-axis title font size
TITLE_FONTSIZE   = 16     # subplot title font size
CBAR_FONTSIZE    = 16     # colorbar label font size
CBAR_TICK_SIZE   = 14     # colorbar tick label size

MAJOR_TICK_SIZE   = 10     # length of major ticks (pts)
MAJOR_TICK_WIDTH  = 1.3   # width of major ticks
MINOR_TICK_SIZE   = 5     # length of minor ticks (pts)
MINOR_TICK_WIDTH  = 0.9   # width of minor ticks
TICK_LABEL_SIZE   = 16    # axis tick label font size
# ─────────────────────────────────────────────────────────────────────────────


def split_by_lengths(x, lengths):
    cuts = np.cumsum(lengths)[:-1]
    return list(np.split(x, cuts))


def get_tica_coords(features_file, model_pkl):
    data    = np.load(features_file, allow_pickle=True)
    x       = np.asarray(data["features"], dtype=np.float64)
    lengths = np.asarray(data["lengths"], dtype=int)

    with open(model_pkl, "rb") as f:
        obj = pickle.load(f)
    tica_model = obj["tica_model"]
    mean       = obj["mean"]
    std        = obj["std"]
    keep       = obj["kept_feature_mask"]

    trajs  = split_by_lengths(x, lengths)
    trajs  = [tr[:, keep] for tr in trajs]
    ztrajs = [(tr - mean) / std for tr in trajs]
    ytrajs = [np.asarray(tica_model.transform(tr)) for tr in ztrajs]
    return np.concatenate(ytrajs)   # (n_frames, n_tica_dims)


def get_weights(msm_arrays_file, disc_traj_file):
    arr = np.load(msm_arrays_file, allow_pickle=True)
    pi  = np.asarray(arr["stationary_distribution"])

    disc      = np.load(disc_traj_file, allow_pickle=True)
    dtraj_all = np.asarray(disc["concatenated"], dtype=int)

    weights = np.where(dtraj_all >= 0,
                       pi[np.clip(dtraj_all, 0, len(pi) - 1)],
                       0.0)
    total = weights.sum()
    if total == 0:
        raise RuntimeError("All weights are zero — check MSM connectivity.")
    return weights / total


def plot_free_energy_2d(ax, x, y, weights, nbins, ncontours, cmap):
    """
    Weighted 2D free energy landscape: F = -kT * log(p), min set to 0.
    Matches pyemma plot_free_energy style.
    """
    h, xedges, yedges = np.histogram2d(x, y, bins=nbins, weights=weights)
    h = h.T   # rows = y, cols = x (matplotlib convention)

    # Free energy; bins with zero count → nan (white / outside contour)
    with np.errstate(divide='ignore', invalid='ignore'):
        F = np.where(h > 0, -kT * np.log(h), np.nan)
    F -= np.nanmin(F)   # shift minimum to 0

    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])

    vmax = np.nanpercentile(F, 99)   # clip extreme tails for color range
    cf = ax.contourf(xc, yc, F,
                     levels=ncontours,
                     vmin=0, vmax=vmax,
                     cmap=cmap)
    ax.contour(xc, yc, F,
               levels=ncontours,
               colors='k', linewidths=0.3, alpha=0.4)
    return cf, vmax


def plot_ligand(ligand, y_all, weights, outdir, label):
    fig, ax = plt.subplots(figsize=(8, 6))

    cf, vmax = plot_free_energy_2d(
        ax, y_all[:, 0], y_all[:, 1],
        weights, NBINS, NCONTOURS, CMAP
    )

    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label("Free Energy (kT)", fontsize=CBAR_FONTSIZE)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)

    ax.set_xlabel("TIC 1", fontsize=XLABEL_FONTSIZE, fontweight='bold', labelpad=10)
    ax.set_ylabel("TIC 2", fontsize=YLABEL_FONTSIZE, fontweight='bold', labelpad=10)
    #ax.set_title(f"{label}  —  {ligand}", fontsize=TITLE_FONTSIZE)

    ax.tick_params(axis='both', which='major',
                   labelsize=TICK_LABEL_SIZE,
                   length=MAJOR_TICK_SIZE, width=MAJOR_TICK_WIDTH)
    ax.tick_params(axis='both', which='minor',
                   length=MINOR_TICK_SIZE, width=MINOR_TICK_WIDTH)
    ax.minorticks_on()

    if XLIM is not None:
        ax.set_xlim(XLIM)
    if YLIM is not None:
        ax.set_ylim(YLIM)

    plt.tight_layout()
    out = outdir / f"reweighted_tica_{ligand}.png"
    plt.savefig(out, dpi=DPI)
    plt.close()
    print(f"  Saved: {out}")


def process_ligand(ligand):
    lig_dir = RESULTS_BASE / ligand
    msm_dir = lig_dir / "msm"
    out_dir = lig_dir / "reweighted_figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    features_file = lig_dir / "features.npz"
    model_pkl     = msm_dir / "model.pkl"
    msm_arrays    = msm_dir / "msm_arrays.npz"
    disc_traj     = msm_dir / "discrete_trajectories.npz"

    for f in [features_file, model_pkl, msm_arrays, disc_traj]:
        if not f.exists():
            print(f"  SKIP {ligand} — missing {f.name}")
            return False

    print(f"\n[{ligand}]")
    y_all   = get_tica_coords(features_file, model_pkl)
    weights = get_weights(msm_arrays, disc_traj)

    if y_all.shape[0] != len(weights):
        print(f"  SKIP {ligand} — frame count mismatch "
              f"(TICA {y_all.shape[0]} vs weights {len(weights)})")
        return False

    plot_ligand(ligand, y_all, weights, out_dir, SYSTEM_LABEL)
    return True


def main():
    failed = []
    for lig in LIGANDS:
        ok = process_ligand(lig)
        if not ok:
            failed.append(lig)
    print(f"\nDone: {len(LIGANDS)-len(failed)}/{len(LIGANDS)} ligands")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
