#!/usr/bin/env python3
"""
Build TICA -> k-means -> reversible maximum-likelihood MSM with deeptime.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
from deeptime.clustering import KMeans
from deeptime.decomposition import TICA
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM


def split_by_lengths(x: np.ndarray, lengths: np.ndarray) -> List[np.ndarray]:
    cuts = np.cumsum(lengths)[:-1]
    return list(np.split(x, cuts))


def implied_timescales(dtrajs, lagtimes, n_timescales=10):
    out = {}
    for lag in lagtimes:
        counts = TransitionCountEstimator(
            lagtime=int(lag), count_mode="sliding"
        ).fit_fetch(dtrajs)
        counts = counts.submodel_largest()
        msm = MaximumLikelihoodMSM(
            reversible=True, lagtime=int(lag)
        ).fit_fetch(counts)
        vals = np.asarray(msm.timescales(k=n_timescales), dtype=float)
        out[int(lag)] = vals
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument(
        "--its-lags",
        default="10,20,50,100,200,500",
        help="Comma-separated lag times in saved-frame units",
    )
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text())

    data = np.load(args.features, allow_pickle=True)
    x = np.asarray(data["features"], dtype=np.float64)
    lengths = np.asarray(data["lengths"], dtype=int)
    feature_names = list(data["feature_names"])
    metadata_json = str(data["metadata_json"].item())
    trajectories = split_by_lengths(x, lengths)

    # Remove unusable columns and standardize.
    finite_cols = np.all(np.isfinite(x), axis=0)
    nonconstant_cols = np.nanstd(x, axis=0) > 1e-12
    keep = finite_cols & nonconstant_cols
    if not np.any(keep):
        raise RuntimeError("No finite, nonconstant feature columns remain.")
    trajectories = [tr[:, keep] for tr in trajectories]
    kept_names = [str(n) for n, flag in zip(feature_names, keep) if flag]

    pooled = np.concatenate(trajectories)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0)
    std[std < 1e-12] = 1.0
    ztrajs = [(tr - mean) / std for tr in trajectories]

    tica_lag = int(cfg["tica_lag_frames"])
    tica_dim = int(cfg["tica_dim"])
    tica_model = TICA(
        lagtime=tica_lag,
        dim=tica_dim,
        scaling="kinetic_map",
    ).fit_fetch(ztrajs)
    ytrajs = [np.asarray(tica_model.transform(tr)) for tr in ztrajs]

    n_clusters = int(cfg["n_clusters"])
    seed = int(cfg.get("random_seed", 42))
    cluster_model = KMeans(
        n_clusters=n_clusters,
        init_strategy="kmeans++",
        max_iter=500,
        tolerance=1e-6,
        fixed_seed=seed,
        n_jobs=None,
    ).fit_fetch(ytrajs)
    dtrajs_full = [np.asarray(cluster_model.transform(y)).reshape(-1).astype(int)
                   for y in ytrajs]

    msm_lag = int(cfg["msm_lag_frames"])
    counts_full = TransitionCountEstimator(
        lagtime=msm_lag,
        count_mode="sliding",
        n_states=n_clusters,
    ).fit_fetch(dtrajs_full)
    counts = counts_full.submodel_largest()

    msm = MaximumLikelihoodMSM(
        reversible=True,
        lagtime=msm_lag,
    ).fit_fetch(counts)

    # Convert trajectories to largest connected submodel; frames outside get -1.
    dtrajs = [
        np.asarray(counts.transform_discrete_trajectories_to_submodel(d)).astype(int)
        for d in dtrajs_full
    ]

    # Cluster-level means in original, unstandardized feature space.
    raw_trajs = split_by_lengths(x, lengths)
    n_active = msm.transition_matrix.shape[0]
    cluster_feature_sum = np.zeros((n_active, x.shape[1]), dtype=float)
    cluster_counts = np.zeros(n_active, dtype=np.int64)
    for raw, d in zip(raw_trajs, dtrajs):
        valid = d >= 0
        for state in np.unique(d[valid]):
            mask = d == state
            cluster_feature_sum[state] += raw[mask].sum(axis=0)
            cluster_counts[state] += int(mask.sum())
    cluster_feature_means = np.divide(
        cluster_feature_sum,
        cluster_counts[:, None],
        out=np.full_like(cluster_feature_sum, np.nan),
        where=cluster_counts[:, None] > 0,
    )

    with open(outdir / "model.pkl", "wb") as f:
        pickle.dump(
            {
                "tica_model": tica_model,
                "cluster_model": cluster_model,
                "count_model": counts,
                "msm": msm,
                "mean": mean,
                "std": std,
                "kept_feature_mask": keep,
                "feature_names": feature_names,
                "kept_feature_names": kept_names,
                "metadata_json": metadata_json,
            },
            f,
        )

    np.savez_compressed(
        outdir / "discrete_trajectories.npz",
        concatenated=np.concatenate(dtrajs),
        lengths=lengths,
    )
    np.savez_compressed(
        outdir / "msm_arrays.npz",
        transition_matrix=np.asarray(msm.transition_matrix),
        stationary_distribution=np.asarray(msm.stationary_distribution),
        cluster_feature_means=cluster_feature_means,
        cluster_counts=cluster_counts,
        feature_names=np.asarray(feature_names, dtype=object),
    )

    lagtimes = [int(v) for v in args.its_lags.split(",") if v.strip()]
    its = implied_timescales(dtrajs_full, lagtimes)
    np.savez_compressed(
        outdir / "implied_timescales.npz",
        lagtimes=np.asarray(list(its.keys())),
        timescales=np.asarray(
            [np.pad(v, (0, max(0, 10-len(v))), constant_values=np.nan)[:10]
             for v in its.values()]
        ),
    )

    print(f"Active microstates: {n_active}")
    print(f"MSM lag: {msm_lag} saved frames")
    print(f"Outputs written to {outdir}")


if __name__ == "__main__":
    main()
