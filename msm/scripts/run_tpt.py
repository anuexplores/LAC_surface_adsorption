#!/usr/bin/env python3
"""
Define source/target microstates from cluster-level physical criteria and run TPT.

Default:
  A (unbound): mean minimum active-site distance >= unbound threshold
  B (bound):   mean minimum active-site distance <= bound threshold

For stronger definitions, edit classify_frames() to include ligand orientation,
specific contacts (e.g. H458), or distance to T1 copper.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text())

    with open(model_dir / "model.pkl", "rb") as f:
        obj = pickle.load(f)
    msm = obj["msm"]

    arr = np.load(model_dir / "msm_arrays.npz", allow_pickle=True)
    means = np.asarray(arr["cluster_feature_means"])
    names = [str(x) for x in arr["feature_names"]]
    pi = np.asarray(arr["stationary_distribution"])

    active_cols = [i for i, n in enumerate(names) if n.startswith("dmin_resid_") and n.endswith("_A")]
    if not active_cols:
        raise RuntimeError("No active-site distance columns found in feature_names.")
    dmin_active = means[:, active_cols].min(axis=1)

    bound_cut = float(cfg.get("bound_active_site_distance_A", cfg["tpt"]["bound_distance_A"]))
    unbound_cut = float(cfg.get("unbound_active_site_distance_A", cfg["tpt"]["unbound_distance_A"]))

    B = np.where(dmin_active <= bound_cut)[0]
    A = np.where(dmin_active >= unbound_cut)[0]

    if len(A) == 0 or len(B) == 0:
        raise RuntimeError(
            f"Empty source or target set: |A|={len(A)}, |B|={len(B)}. "
            "Inspect cluster_feature_means and adjust thresholds."
        )
    if np.intersect1d(A, B).size:
        raise RuntimeError("Source and target state sets overlap.")

    tpt = msm.reactive_flux(source_states=A, target_states=B)
    fraction = float(cfg.get("pathway_flux_fraction", 0.9))
    paths, capacities = tpt.pathways(fraction=fraction)

    np.savez_compressed(
        outdir / "tpt_arrays.npz",
        source_states=A,
        target_states=B,
        forward_committor=np.asarray(tpt.forward_committor),
        backward_committor=np.asarray(tpt.backward_committor),
        gross_flux=np.asarray(tpt.gross_flux),
        net_flux=np.asarray(tpt.net_flux),
        stationary_distribution=pi,
        total_flux=np.asarray(tpt.total_flux),
        rate=np.asarray(tpt.rate),
    )

    with open(outdir / "dominant_pathways.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "path_states", "capacity", "fraction_of_total_flux"])
        total = float(np.sum(capacities)) if len(capacities) else np.nan
        for rank, (path, cap) in enumerate(zip(paths, capacities), start=1):
            writer.writerow([
                rank,
                " -> ".join(map(str, path)),
                float(cap),
                float(cap / total) if total > 0 else np.nan,
            ])

    with open(outdir / "state_sets.txt", "w") as f:
        f.write(f"Source A states (unbound): {A.tolist()}\n")
        f.write(f"Target B states (bound): {B.tolist()}\n")
        f.write(f"Total reactive flux: {float(tpt.total_flux)}\n")
        f.write(f"Rate per MSM step: {float(tpt.rate)}\n")

    print(f"A states: {A.tolist()}")
    print(f"B states: {B.tolist()}")
    print(f"Total reactive flux: {float(tpt.total_flux):.6g}")
    print(f"Rate per MSM step: {float(tpt.rate):.6g}")
    print(f"Wrote TPT results to {outdir}")


if __name__ == "__main__":
    main()
