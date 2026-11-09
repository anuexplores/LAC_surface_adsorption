#!/usr/bin/env python3
"""
Analyse H-bond occupancy (N1_lig → T180-OG1 and G175-O),
ligand–residue distance distributions, local RMSF (residues 170-185),
and S8-like configuration prevalence — for xylidine and aniline
across bulk, silica, lysine, APTES.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = Path("/Data/Anushka/Project4/ph5/orients/orient1/tleap/energy_landscapes")

# ── system definitions ────────────────────────────────────────────────────────
SYSTEMS = {
    "xylidine": {
        "bulk":   ("/Data/Anushka/Project4/anushka@10.0.66.94/xylidine_wsAB/packed/MD/RC/2000ns.gro",
                   "/Data/Anushka/Project4/anushka@10.0.66.94/xylidine_wsAB/packed/MD/RC/2000ns_noPBC.xtc"),
        "silica": ("/Data/Anushka/Project4/ph5/orients/orient1/tleap/silica/silica_25xylidine/MD/silica_25xylidine.gro",
                   "/Data/Anushka/Project4/ph5/orients/orient1/tleap/silica/silica_25xylidine/MD/200ns_noPBC.xtc"),
        "lysine": ("/Data/Anushka/Project4/anushka@10.0.66.94/P4/orient1/ph5/lysine/lys_silica_25xylidine/MD/lys_silica_25xylidine.gro",
                   "/Data/Anushka/Project4/anushka@10.0.66.94/P4/orient1/ph5/lysine/lys_silica_25xylidine/MD/200ns_noPBC.xtc"),
        "aptes":  ("/Data/Anushka/Project4/anushka@10.0.66.94/P4/orient1/ph5/aptes/aptes_25xylidine/MD/rep5/200ns.gro",
                   "/Data/Anushka/Project4/anushka@10.0.66.94/P4/orient1/ph5/aptes/aptes_25xylidine/MD/rep5/200ns_noPBC.xtc"),
    },
    "aniline": {
        "bulk":   ("/media/anushka/One Touch/ws_AB4/Anushka/MD-docked/aniline/MD/set1/RC/1000ns.gro",
                   "/media/anushka/One Touch/ws_AB4/Anushka/MD-docked/aniline/MD/set1/RC/1000ns_noPBC.xtc"),
        "silica": ("/Data/Anushka/Project4/ph5/orients/orient1/tleap/silica/silica_aniline/MD/silica_aniline.gro",
                   "/Data/Anushka/Project4/ph5/orients/orient1/tleap/silica/silica_aniline/MD/200ns_noPBC.xtc"),
        "lysine": ("/Data/Anushka/Project4/anushka@10.0.66.94/P4/orient1/ph5/lysine/lys_silica_aniline/MD/lys_silica_aniline.gro",
                   "/Data/Anushka/Project4/anushka@10.0.66.94/P4/orient1/ph5/lysine/lys_silica_aniline/MD/200ns_noPBC.xtc"),
        "aptes":  ("/Data/Anushka/Project4/anushka@10.0.66.94/P4/orient1/ph5/aptes/aptes_aniline/MD/rep5/rep5_aniline_start.gro",
                   "/Data/Anushka/Project4/anushka@10.0.66.94/P4/orient1/ph5/aptes/aptes_aniline/MD/rep5/200ns_noPBC.xtc"),
    },
}

SURF_ORDER = ["bulk", "silica", "lysine", "aptes"]
SURF_COLORS = {"bulk": "#2171B5", "silica": "#CB181D", "lysine": "#984EA3", "aptes": "#238B45"}
SURF_LABELS = {"bulk": "Bulk", "silica": "Silica", "lysine": "Lysine", "aptes": "APTES"}
STRIDE = 10   # every 10th frame to keep runtime manageable
HBOND_CUTOFF = 3.5   # Å, N–O distance threshold
RMSF_RESIDS = list(range(168, 188))   # loop around G175 and T180

def min_dist_per_frame(u, sel1, sel2):
    """Per-frame minimum distance between two atom groups (handles multiple ligand copies)."""
    dists = []
    for ts in u.trajectory[::STRIDE]:
        d = np.min(np.linalg.norm(
            sel1.positions[:, None, :] - sel2.positions[None, :, :], axis=2))
        dists.append(d)
    return np.array(dists)

def compute_rmsf(u, resids):
    """
    Cα RMSF for given resids, after aligning each frame to a reference
    using backbone Cα of stable regions (excluding the loop of interest).
    """
    loop_lo, loop_hi = min(resids), max(resids)
    # alignment selection: all protein Cα OUTSIDE the loop of interest
    align_sel = u.select_atoms(
        f"name CA and protein and not (resid {loop_lo}-{loop_hi})")
    # target selection: Cα of the loop
    loop_sel = u.select_atoms(
        f"name CA and protein and resid {loop_lo}-{loop_hi}")
    if len(loop_sel) == 0 or len(align_sel) < 5:
        return None, None

    # build reference positions from first frame
    u.trajectory[0]
    ref_align = align_sel.positions.copy()

    loop_positions = []
    for ts in u.trajectory[::STRIDE]:
        # superpose alignment atoms onto reference
        mob = align_sel.positions.copy()
        # centre both sets
        ref_c = ref_align.mean(axis=0)
        mob_c = mob.mean(axis=0)
        ref_n = ref_align - ref_c
        mob_n = mob - mob_c
        # SVD rotation
        H = mob_n.T @ ref_n
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1, 1, d])
        R = Vt.T @ D @ U.T
        # apply rotation + translation to loop atoms
        loop_pos = (loop_sel.positions - mob_c) @ R.T + ref_c
        loop_positions.append(loop_pos)

    loop_positions = np.array(loop_positions)
    mean_pos = loop_positions.mean(axis=0)
    rmsf_vals = np.sqrt(((loop_positions - mean_pos)**2).sum(axis=2).mean(axis=0))
    return loop_sel.resids, rmsf_vals

# ── collect data ──────────────────────────────────────────────────────────────
results = {}

for lig_name, sys_dict in SYSTEMS.items():
    results[lig_name] = {}
    for surf, (gro, xtc) in sys_dict.items():
        if not Path(xtc).exists():
            print(f"  SKIP {lig_name}/{surf}: {xtc} not found")
            continue
        print(f"  {lig_name}/{surf}...", flush=True)
        try:
            u = mda.Universe(str(gro), str(xtc))
            # Select all N1 atoms of all ligand copies
            n1  = u.select_atoms("resname UNK and name N1")
            og1 = u.select_atoms("resid 180 and name OG1")
            go  = u.select_atoms("resid 175 and name O")

            if len(n1) == 0 or len(og1) == 0 or len(go) == 0:
                print(f"    WARNING: missing atoms (N1={len(n1)}, OG1={len(og1)}, GO={len(go)})")
                continue

            d_t180 = min_dist_per_frame(u, n1, og1)
            d_g175 = min_dist_per_frame(u, n1, go)

            occ_t180 = (d_t180 < HBOND_CUTOFF).mean() * 100
            occ_g175 = (d_g175 < HBOND_CUTOFF).mean() * 100
            # S8-like: both H-bonds simultaneously
            s8_like  = ((d_t180 < HBOND_CUTOFF) & (d_g175 < HBOND_CUTOFF)).mean() * 100

            rids, rmsf_vals = compute_rmsf(u, RMSF_RESIDS)

            results[lig_name][surf] = {
                "d_t180": d_t180, "d_g175": d_g175,
                "occ_t180": occ_t180, "occ_g175": occ_g175,
                "s8_like": s8_like,
                "rmsf_resids": rids, "rmsf_vals": rmsf_vals,
                "n_frames": len(d_t180),
            }
            print(f"    occ_T180={occ_t180:.1f}%  occ_G175={occ_g175:.1f}%  "
                  f"S8-like={s8_like:.1f}%  ({len(d_t180)} frames)")
        except Exception as e:
            print(f"    ERROR: {e}")

# ── plotting ──────────────────────────────────────────────────────────────────
for lig_name in SYSTEMS:
    if not results.get(lig_name):
        continue

    surfs = [s for s in SURF_ORDER if s in results[lig_name]]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # Panel 1: distance distribution N1–OG1(T180)
    ax = axes[0]
    for s in surfs:
        d = results[lig_name][s]["d_t180"]
        ax.hist(d, bins=60, range=(0, 12), density=True, histtype="step",
                color=SURF_COLORS[s], label=SURF_LABELS[s], lw=2)
    ax.axvline(HBOND_CUTOFF, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("d(N₁ – OG1 T180) (Å)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"T180 distance", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 2: distance distribution N1–O(G175)
    ax = axes[1]
    for s in surfs:
        d = results[lig_name][s]["d_g175"]
        ax.hist(d, bins=60, range=(0, 12), density=True, histtype="step",
                color=SURF_COLORS[s], label=SURF_LABELS[s], lw=2)
    ax.axvline(HBOND_CUTOFF, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("d(N₁ – O G175) (Å)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("G175 distance", fontsize=10, fontweight="bold")

    # Panel 3: H-bond occupancy bar chart
    ax = axes[2]
    x = np.arange(len(surfs))
    w = 0.35
    occ_t = [results[lig_name][s]["occ_t180"] for s in surfs]
    occ_g = [results[lig_name][s]["occ_g175"] for s in surfs]
    s8_v  = [results[lig_name][s]["s8_like"]  for s in surfs]
    colors_s = [SURF_COLORS[s] for s in surfs]
    bars1 = ax.bar(x - w/2, occ_t, w, color=colors_s, label="T180(OG1)", alpha=0.85)
    bars2 = ax.bar(x + w/2, occ_g, w, color=colors_s, label="G175(O)",   alpha=0.45, hatch="//")
    ax.plot(x, s8_v, "kD", ms=7, label="S8-like (both)", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([SURF_LABELS[s] for s in surfs], fontsize=9)
    ax.set_ylabel("H-bond occupancy (%)", fontsize=10)
    ax.set_title("H-bond occupancy", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)

    # Panel 4: RMSF of loop 168-187
    ax = axes[3]
    for s in surfs:
        rids = results[lig_name][s].get("rmsf_resids")
        rmsf = results[lig_name][s].get("rmsf_vals")
        if rids is not None and rmsf is not None:
            ax.plot(rids, rmsf, color=SURF_COLORS[s], label=SURF_LABELS[s], lw=2, marker="o", ms=3)
    ax.axvline(175, color="grey", ls=":", lw=1)
    ax.axvline(180, color="grey", ls=":", lw=1)
    ax.text(175.1, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] > 0 else 1, "G175", fontsize=7, va="top")
    ax.text(180.1, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] > 0 else 1, "T180", fontsize=7, va="top")
    ax.set_xlabel("Residue", fontsize=10)
    ax.set_ylabel("RMSF (Å)", fontsize=10)
    ax.set_title("Loop RMSF (168–187)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle(f"{lig_name} — H-bond & RMSF analysis", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / f"hbond_rmsf_{lig_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")

import csv

# ── summary table ─────────────────────────────────────────────────────────────
print("\n=== H-BOND OCCUPANCY SUMMARY ===")
for lig_name in SYSTEMS:
    print(f"\n{lig_name.upper()}")
    print(f"  {'Surface':8s}  {'T180-OG1 (%)':14s}  {'G175-O (%)':12s}  {'S8-like (%)':12s}  frames")
    for s in SURF_ORDER:
        if s not in results.get(lig_name, {}):
            continue
        r = results[lig_name][s]
        print(f"  {SURF_LABELS[s]:8s}  {r['occ_t180']:14.1f}  {r['occ_g175']:12.1f}  {r['s8_like']:12.1f}  {r['n_frames']}")

# ── CSV export ────────────────────────────────────────────────────────────────

# CSV 1: H-bond occupancy
with open(OUT_DIR / "hbond_occupancy.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Ligand", "Surface", "T180_OG1_occ_%", "G175_O_occ_%",
                "S8_like_occ_%", "n_frames", "HBond_cutoff_A"])
    for lig_name in SYSTEMS:
        for s in SURF_ORDER:
            if s not in results.get(lig_name, {}):
                continue
            r = results[lig_name][s]
            w.writerow([lig_name, SURF_LABELS[s],
                        f"{r['occ_t180']:.2f}", f"{r['occ_g175']:.2f}",
                        f"{r['s8_like']:.2f}", r['n_frames'], HBOND_CUTOFF])
print(f"\nSaved: {OUT_DIR/'hbond_occupancy.csv'}")

# CSV 2: RMSF per residue
with open(OUT_DIR / "loop_rmsf.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Ligand", "Surface", "Residue", "RMSF_A", "alignment"])
    for lig_name in SYSTEMS:
        for s in SURF_ORDER:
            if s not in results.get(lig_name, {}):
                continue
            r = results[lig_name][s]
            if r["rmsf_resids"] is None:
                continue
            for res, val in zip(r["rmsf_resids"], r["rmsf_vals"]):
                w.writerow([lig_name, SURF_LABELS[s], int(res), f"{val:.4f}",
                            "backbone_Calpha_excl_loop168-187"])
print(f"Saved: {OUT_DIR/'loop_rmsf.csv'}")

# CSV 3: raw distance time series (N1 to T180-OG1 and G175-O)
with open(OUT_DIR / "lig_residue_distances.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Ligand", "Surface", "Frame", "d_N1_T180OG1_A", "d_N1_G175O_A"])
    for lig_name in SYSTEMS:
        for s in SURF_ORDER:
            if s not in results.get(lig_name, {}):
                continue
            r = results[lig_name][s]
            for i, (d1, d2) in enumerate(zip(r["d_t180"], r["d_g175"])):
                w.writerow([lig_name, SURF_LABELS[s], i, f"{d1:.4f}", f"{d2:.4f}"])
print(f"Saved: {OUT_DIR/'lig_residue_distances.csv'}")
