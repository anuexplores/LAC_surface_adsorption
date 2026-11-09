#!/usr/bin/env python3
"""
Compute MSM kinetics: MFPT (binding/unbinding) and rates.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np


def mfpt_to_target(T, target_states):
    """Mean first passage time (in lag steps) to reach any state in target_states."""
    n = T.shape[0]
    target_set = set(int(s) for s in target_states)
    non_target = [i for i in range(n) if i not in target_set]

    T_sub = T[np.ix_(non_target, non_target)]
    A_mat = np.eye(len(non_target)) - T_sub
    b = np.ones(len(non_target))
    mfpt_sub = np.linalg.solve(A_mat, b)

    mfpt = np.full(n, np.nan)
    for s in target_states:
        mfpt[int(s)] = 0.0
    for idx, state in enumerate(non_target):
        mfpt[state] = mfpt_sub[idx]
    return mfpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--tpt-dir", required=True)
    p.add_argument("--dt-ns", required=True, type=float,
                   help="Time per saved frame in ns")
    p.add_argument("--msm-lag-frames", required=True, type=int,
                   help="MSM lag in saved frames")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    tpt_dir = Path(args.tpt_dir)
    lag_ns = args.msm_lag_frames * args.dt_ns

    with open(model_dir / "model.pkl", "rb") as f:
        obj = pickle.load(f)
    msm = obj["msm"]
    T = np.asarray(msm.transition_matrix)
    pi = np.asarray(msm.stationary_distribution)
    n = T.shape[0]

    tpt_data = np.load(tpt_dir / "tpt_arrays.npz", allow_pickle=True)
    A = tpt_data["source_states"]   # unbound
    B = tpt_data["target_states"]   # bound
    total_flux = float(tpt_data["total_flux"])
    tpt_rate = float(tpt_data["rate"])

    # MFPT: unbound → bound (binding)
    mfpt_bind = mfpt_to_target(T, B)
    pi_A = pi[A] / pi[A].sum()
    mfpt_bind_ns = float(np.dot(pi_A, mfpt_bind[A])) * lag_ns

    # MFPT: bound → unbound (unbinding)
    mfpt_unbind = mfpt_to_target(T, A)
    pi_B = pi[B] / pi[B].sum()
    mfpt_unbind_ns = float(np.dot(pi_B, mfpt_unbind[B])) * lag_ns

    # Rates from MFPT
    k_on_ns = 1.0 / mfpt_bind_ns      # ns^-1
    k_off_ns = 1.0 / mfpt_unbind_ns   # ns^-1

    # TPT rate (in per lag step) → convert to ns^-1
    tpt_rate_ns = tpt_rate / lag_ns

    print("=" * 50)
    print("  MSM KINETICS SUMMARY")
    print("=" * 50)
    print(f"  MSM lag time         : {lag_ns:.1f} ns ({args.msm_lag_frames} frames)")
    print(f"  Active microstates   : {n}")
    print(f"  Unbound states (A)   : {len(A)}")
    print(f"  Bound states (B)     : {len(B)}")
    print()
    print(f"  MFPT (unbound→bound) : {mfpt_bind_ns:.1f} ns")
    print(f"  MFPT (bound→unbound) : {mfpt_unbind_ns:.1f} ns")
    print()
    print(f"  k_on  (from MFPT)    : {k_on_ns*1000:.4f} µs⁻¹")
    print(f"  k_off (from MFPT)    : {k_off_ns*1000:.4f} µs⁻¹")
    print()
    print(f"  TPT total flux       : {total_flux:.4g}")
    print(f"  TPT rate             : {tpt_rate_ns*1000:.4f} µs⁻¹")
    print("=" * 50)


if __name__ == "__main__":
    main()
