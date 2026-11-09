#!/usr/bin/env python3
"""
Plot TICA free energy landscape (TICA1 vs TICA2) and per-cluster free energy.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


kT = 2.479  # kJ/mol at 300 K


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--bins", type=int, default=80)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    with open(model_dir / "model.pkl", "rb") as f:
        obj = pickle.load(f)
    tica_model  = obj["tica_model"]
    mean        = obj["mean"]
    std         = obj["std"]
    keep        = obj["kept_feature_mask"]

    data = np.load(args.features, allow_pickle=True)
    x    = np.asarray(data["features"], dtype=np.float64)
    lengths = np.asarray(data["lengths"], dtype=int)

    # Standardize same way as build_msm.py
    cuts = np.cumsum(lengths)[:-1]
    trajectories = list(np.split(x, cuts))
    trajectories = [tr[:, keep] for tr in trajectories]
    ztrajs = [(tr - mean) / std for tr in trajectories]
    ytrajs = [np.asarray(tica_model.transform(tr)) for tr in ztrajs]
    y_all  = np.concatenate(ytrajs)

    # --- Plot 1: TICA1 vs TICA2 free energy landscape ---
    tic1, tic2 = y_all[:, 0], y_all[:, 1]
    h, xedge, yedge = np.histogram2d(tic1, tic2, bins=args.bins)
    h = h.T
    h = np.where(h > 0, h, np.nan)
    F = -kT * np.log(h / np.nanmax(h))   # relative free energy, min=0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.contourf(
        0.5*(xedge[:-1]+xedge[1:]),
        0.5*(yedge[:-1]+yedge[1:]),
        F, levels=20, cmap="RdYlBu_r"
    )
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Free energy (kJ/mol)")
    ax.set_xlabel("TIC 1")
    ax.set_ylabel("TIC 2")
    ax.set_title("TICA Free Energy Landscape — aniline/silica")
    plt.tight_layout()
    out1 = outdir / "tica_free_energy.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"Saved: {out1}")

    # --- Plot 2: Per-cluster free energy from stationary distribution ---
    arr = np.load(model_dir / "msm_arrays.npz", allow_pickle=True)
    pi  = np.asarray(arr["stationary_distribution"])
    F_cluster = -kT * np.log(pi / pi.max())

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(len(F_cluster)), np.sort(F_cluster), color="steelblue", width=0.8)
    ax.set_xlabel("Microstate (sorted by free energy)")
    ax.set_ylabel("Relative free energy (kJ/mol)")
    ax.set_title("Per-cluster Free Energy — aniline/silica")
    plt.tight_layout()
    out2 = outdir / "cluster_free_energy.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
