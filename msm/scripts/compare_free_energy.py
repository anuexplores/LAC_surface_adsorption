#!/usr/bin/env python3
"""
Plot all 10 ligand TICA free energy landscapes in a 2x5 grid.
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_BASE = Path("/Data/Anushka/Project4/ph5/orients/orient1/tleap/silica/msm-results")
LIGANDS = [
    "aniline", "25xylidine", "benzene", "catechol", "hydroquinone",
    "hydroxyquinol", "phenol", "phloroglucinol", "pyrogallol", "resorcinol",
]

kT   = 2.479   # kJ/mol at 300 K
BINS = 60


def tica_landscape(lig):
    msm_dir      = RESULTS_BASE / lig / "msm"
    features_file = RESULTS_BASE / lig / "features.npz"

    if not (msm_dir / "model.pkl").exists() or not features_file.exists():
        return None, None, None, None, None

    with open(msm_dir / "model.pkl", "rb") as f:
        obj = pickle.load(f)
    tica_model = obj["tica_model"]
    mean       = obj["mean"]
    std        = obj["std"]
    keep       = obj["kept_feature_mask"]

    data    = np.load(features_file, allow_pickle=True)
    x       = np.asarray(data["features"], dtype=np.float64)
    lengths = np.asarray(data["lengths"], dtype=int)

    cuts = np.cumsum(lengths)[:-1]
    trajs = list(np.split(x, cuts))
    trajs = [tr[:, keep] for tr in trajs]
    ztrajs = [(tr - mean) / std for tr in trajs]
    ytrajs = [np.asarray(tica_model.transform(tr)) for tr in ztrajs]
    y_all  = np.concatenate(ytrajs)

    tic1, tic2 = y_all[:, 0], y_all[:, 1]
    h, xe, ye = np.histogram2d(tic1, tic2, bins=BINS)
    h = h.T
    h = np.where(h > 0, h, np.nan)
    F = -kT * np.log(h / np.nanmax(h))

    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    return xc, yc, F, tic1, tic2


def main():
    outdir = RESULTS_BASE / "comparison"
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=False, sharey=False)
    axes_flat = axes.flatten()

    vmax_global = 12.0   # kJ/mol ceiling for colorscale

    for idx, lig in enumerate(LIGANDS):
        ax = axes_flat[idx]
        xc, yc, F, tic1, tic2 = tica_landscape(lig)

        if F is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(lig, fontsize=9)
            continue

        im = ax.contourf(xc, yc, F, levels=15,
                         cmap="RdYlBu_r", vmin=0, vmax=vmax_global)
        ax.set_title(lig, fontsize=10, fontweight="bold")
        ax.set_xlabel("TIC 1", fontsize=7)
        ax.set_ylabel("TIC 2", fontsize=7)
        ax.tick_params(labelsize=7)

    # shared colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Free energy (kJ/mol)", fontsize=10)

    fig.suptitle("TICA Free Energy Landscapes — Silica/Laccase (all ligands)",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    out_fig = outdir / "free_energy_comparison.png"
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_fig}")


if __name__ == "__main__":
    main()
