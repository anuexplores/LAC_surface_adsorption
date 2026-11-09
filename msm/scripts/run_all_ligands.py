#!/usr/bin/env python3
"""
Run the full MSM pipeline for all 9 remaining silica ligands.
For each ligand:
  1. Generate config JSON
  2. extract_features.py
  3. build_msm.py
  4. Auto-detect bound/unbound thresholds from cluster data
  5. run_tpt.py
  6. compute_kinetics.py
  7. plot_free_energy.py
  8. plot_implied_timescales.py (update dashed line)
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
SILICA_BASE = Path("/Data/Anushka/Project4/ph5/orients/orient1/tleap/silica")
SCRIPTS_DIR = SILICA_BASE / "msm-scripts"
RESULTS_BASE = SILICA_BASE / "msm-results"

LIGANDS = [
    "25xylidine", "benzene", "catechol", "hydroquinone",
    "hydroxyquinol", "phenol", "phloroglucinol", "pyrogallol", "resorcinol",
]

# ── shared MSM parameters (same as aniline) ──────────────────────────────────
CONFIG_TEMPLATE = {
    "protein_selection": "(protein or resname HE2) and resid 1 to 499",
    "active_site_resids": [150, 161, 162, 163, 164, 206, 392, 458],
    "t1_copper_selection": "resname CU1 and resid 500",
    "loop_gate_definition": {
        "type": "COM_distance",
        "group1": [163],
        "group2": [458],
    },
    "stride": 1,
    "tica_lag_frames": 100,
    "tica_dim": 5,
    "n_clusters": 100,
    "random_seed": 42,
    "msm_lag_frames": 200,
    "pathway_flux_fraction": 0.9,
    "tica": {"lag_frames": 100, "dimensions": 5, "kinetic_map": True},
    "clustering": {"method": "kmeans", "clusters": 100, "random_seed": 42, "max_iterations": 500},
    "msm": {"lag_frames": 200, "reversible": True, "count_mode": "sliding"},
    "validation": {
        "implied_timescales": [5, 10, 20, 40, 60, 80, 100, 150, 200],
        "chapman_kolmogorov": True,
    },
    "tpt": {"bound_distance_A": 20.0, "unbound_distance_A": 50.0, "flux_fraction": 0.9},
}


def run(cmd, label):
    print(f"\n>>> {label}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        print(f"ERROR in {label}:\n{result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def auto_thresholds(msm_dir):
    """Compute bound/unbound distance thresholds from cluster feature means."""
    arr = np.load(msm_dir / "msm_arrays.npz", allow_pickle=True)
    means = np.asarray(arr["cluster_feature_means"])
    names = [str(x) for x in arr["feature_names"]]
    active_cols = [
        i for i, n in enumerate(names)
        if n.startswith("dmin_resid_") and n.endswith("_A")
    ]
    dmin = means[:, active_cols].min(axis=1)
    bound_cut = float(np.min(dmin)) + 10.0          # min + 10 Å
    unbound_cut = float(np.percentile(dmin, 70))    # 70th percentile
    print(f"    dmin range: {dmin.min():.1f} – {dmin.max():.1f} Å")
    print(f"    bound_cut={bound_cut:.1f} Å  unbound_cut={unbound_cut:.1f} Å")
    return bound_cut, unbound_cut


def process_ligand(ligand):
    print(f"\n{'='*60}")
    print(f"  LIGAND: {ligand}")
    print(f"{'='*60}")

    lig_dir   = SILICA_BASE / f"silica_{ligand}" / "MD"
    out_dir   = RESULTS_BASE / ligand
    msm_dir   = out_dir / "msm"
    tpt_dir   = out_dir / "tpt"
    figs_dir  = out_dir / "figs"
    cfg_path  = SCRIPTS_DIR / f"config_{ligand}.json"

    for d in [out_dir, msm_dir, tpt_dir, figs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Generate config ────────────────────────────────────────────────
    cfg = dict(CONFIG_TEMPLATE)
    cfg["ligands"] = {
        ligand: {
            "resname": "UNK",
            "replicates": [
                {
                    "topology":   str(lig_dir / "200ns.tpr"),
                    "trajectory": str(lig_dir / "200ns_noPBC.xtc"),
                },
                {
                    "topology":   str(lig_dir / "rep2" / "200ns.tpr"),
                    "trajectory": str(lig_dir / "rep2" / "200ns_noPBC.xtc"),
                },
                {
                    "topology":   str(lig_dir / "rep3" / "200ns.tpr"),
                    "trajectory": str(lig_dir / "rep3" / "200ns_noPBC.xtc"),
                },
            ],
        }
    }
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"Config written: {cfg_path}")

    features_file = out_dir / "features.npz"

    # ── 2. Extract features ───────────────────────────────────────────────
    if features_file.exists():
        print(f"\n>>> extract_features  [SKIP — {features_file.name} exists]")
    else:
        ok = run(
            ["python", str(SCRIPTS_DIR / "extract_features.py"),
             "--ligand", ligand,
             "--config", str(cfg_path),
             "--out", str(features_file)],
            "extract_features"
        )
        if not ok:
            return False

    # ── 3. Build MSM ──────────────────────────────────────────────────────
    if (msm_dir / "model.pkl").exists():
        print(f"\n>>> build_msm  [SKIP — model.pkl exists]")
    else:
        ok = run(
            ["python", str(SCRIPTS_DIR / "build_msm.py"),
             "--features", str(features_file),
             "--config", str(cfg_path),
             "--outdir", str(msm_dir)],
            "build_msm"
        )
        if not ok:
            return False

    # ── 4. Auto-detect thresholds and update config ───────────────────────
    bound_cut, unbound_cut = auto_thresholds(msm_dir)
    cfg["bound_active_site_distance_A"]   = bound_cut
    cfg["unbound_active_site_distance_A"] = unbound_cut
    cfg_path.write_text(json.dumps(cfg, indent=2))

    # ── 5. TPT ────────────────────────────────────────────────────────────
    if (tpt_dir / "tpt_arrays.npz").exists():
        print(f"\n>>> run_tpt  [SKIP — tpt_arrays.npz exists]")
    else:
        ok = run(
            ["python", str(SCRIPTS_DIR / "run_tpt.py"),
             "--model-dir", str(msm_dir),
             "--config", str(cfg_path),
             "--outdir", str(tpt_dir)],
            "run_tpt"
        )
        if not ok:
            return False

    # ── 6. Kinetics ───────────────────────────────────────────────────────
    if (tpt_dir / "state_sets.txt").exists():
        print(f"\n>>> compute_kinetics  [SKIP — state_sets.txt exists]")
    else:
        run(
            ["python", str(SCRIPTS_DIR / "compute_kinetics.py"),
             "--model-dir", str(msm_dir),
             "--tpt-dir", str(tpt_dir),
             "--dt-ns", "0.1",
             "--msm-lag-frames", "200"],
            "compute_kinetics"
        )

    # ── 7. Free energy plots ──────────────────────────────────────────────
    if (figs_dir / "tica_free_energy.png").exists():
        print(f"\n>>> plot_free_energy  [SKIP — tica_free_energy.png exists]")
    else:
        run(
            ["python", str(SCRIPTS_DIR / "plot_free_energy.py"),
             "--model-dir", str(msm_dir),
             "--features", str(features_file),
             "--outdir", str(figs_dir)],
            "plot_free_energy"
        )

    # ── 8. ITS plot ───────────────────────────────────────────────────────
    if (msm_dir / "implied_timescales.png").exists():
        print(f"\n>>> plot_diagnostics  [SKIP — implied_timescales.png exists]")
    else:
        run(
            ["python", str(SCRIPTS_DIR / "plot_diagnostics.py"),
             "--model-dir", str(msm_dir),
             "--dt-ns", "0.1"],
            "plot_diagnostics"
        )

    print(f"\n  Done: {ligand}  →  {out_dir}")
    return True


def main():
    failed = []
    for lig in LIGANDS:
        ok = process_ligand(lig)
        if not ok:
            failed.append(lig)

    print(f"\n{'='*60}")
    print(f"Completed: {len(LIGANDS) - len(failed)}/{len(LIGANDS)} ligands")
    if failed:
        print(f"Failed: {failed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
