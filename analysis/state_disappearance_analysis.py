#!/usr/bin/env python3
"""
Mechanistic analysis: S4 (phenol), S3 (catechol/pyrogallol), S5 (resorcinol).
Loop-1 = residues 160–164 (corrected).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import MDAnalysis as mda
import warnings, csv
warnings.filterwarnings("ignore")

OUT  = Path("/Data/Anushka/Project4/ph5/orients/orient1/tleap/energy_landscapes")
STRIDE = 10

SURF_COLORS = {"bulk":"#2171B5","silica":"#CB181D","lysine":"#984EA3","aptes":"#238B45"}
SURF_LABELS = {"bulk":"Bulk","silica":"Silica","lysine":"Lysine","aptes":"APTES"}
SURF_ORDER  = ["bulk","silica","lysine","aptes"]

MD  = Path("/media/anushka/One Touch/ws_AB4/Anushka/MD-docked")
TRI = Path("/Data/Anushka/Project4/anushka@10.0.66.94/Trihydric")
SIL = Path("/Data/Anushka/Project4/ph5/orients/orient1/tleap/silica")
LYS = Path("/Data/Anushka/Project4/anushka@10.0.66.94/P4/orient1/ph5/lysine")
APT = Path("/Data/Anushka/Project4/anushka@10.0.66.94/P4/orient1/ph5/aptes")

def ap(lig):
    d = APT / f"aptes_{lig}/MD/rep5"
    gro = sorted(d.glob("*.gro"))[0] if d.exists() and list(d.glob("*.gro")) else None
    return gro, d/"200ns_noPBC.xtc"

SYSTEMS = {
    "phenol": {
        "bulk":   (MD/"Phenol/MD/set1/RC/1000ns.gro",     MD/"Phenol/MD/set1/RC/1000ns_noPBC.xtc"),
        "silica": (SIL/"silica_phenol/MD/silica_phenol.gro", SIL/"silica_phenol/MD/200ns_noPBC.xtc"),
        "lysine": (LYS/"lys_silica_phenol/MD/lys_silica_phenol.gro", LYS/"lys_silica_phenol/MD/200ns_noPBC.xtc"),
        "aptes":  ap("phenol"),
    },
    "catechol": {
        "bulk":   (MD/"Catechol/Random/packed/MD/allostery/site-3/2000ns.gro",
                   MD/"Catechol/Random/packed/MD/allostery/site-3/2000ns_noPBC.xtc"),
        "silica": (SIL/"silica_catechol/MD/silica_catechol.gro", SIL/"silica_catechol/MD/200ns_noPBC.xtc"),
        "lysine": (LYS/"lys_silica_catechol/MD/lys_silica_catechol.gro", LYS/"lys_silica_catechol/MD/200ns_noPBC.xtc"),
        "aptes":  ap("catechol"),
    },
    "pyrogallol": {
        "bulk":   (TRI/"Pyrogallol/MD/set3/RC/2000ns.gro",  TRI/"Pyrogallol/MD/set3/RC/2000ns_noPBC.xtc"),
        "silica": (SIL/"silica_pyrogallol/MD/silica_pyrogallol.gro", SIL/"silica_pyrogallol/MD/200ns_noPBC.xtc"),
        "lysine": (LYS/"lys_silica_pyrogallol/MD/lys_silica_pyrogallol.gro", LYS/"lys_silica_pyrogallol/MD/200ns_noPBC.xtc"),
        "aptes":  ap("pyrogallol"),
    },
    "resorcinol": {
        "bulk":   (MD/"pyemma/dihydric/resorcinol/resorcinol_3000ns.gro",
                   MD/"pyemma/dihydric/resorcinol/resorcinol_2700ns_noPBC.xtc"),
        "silica": (SIL/"silica_resorcinol/MD/silica_resorcinol.gro", SIL/"silica_resorcinol/MD/200ns_noPBC.xtc"),
        "lysine": (LYS/"lys_silica_resorcinol/MD/lys_silica_resorcinol.gro", LYS/"lys_silica_resorcinol/MD/200ns_noPBC.xtc"),
        "aptes":  ap("resorcinol"),
    },
}

# Loop-1 = 160-164 (corrected)
LOOP1_LO, LOOP1_HI = 160, 164

STATE_RESIDUES = {
    "phenol":     {"label":"S4",
                   "rmsf_range":(155,175),       # captures Loop-1 (160-164) and H458 approach
                   "key_resids":[160,161,162,163,164],
                   "dist_resid":458,"dist_name":"CA",
                   "state_name":"S4 (Loop-1/H458 cavity)"},
    "catechol":   {"label":"S3","rmsf_range":(330,360),
                   "key_resids":[343,344,345],
                   "dist_resid":343,"dist_name":"CA",
                   "state_name":"S3 (S343/F344/T345 patch)"},
    "pyrogallol": {"label":"S3","rmsf_range":(330,360),
                   "key_resids":[343,344,345],
                   "dist_resid":343,"dist_name":"CA",
                   "state_name":"S3 (S343/F344/T345 patch)"},
    "resorcinol": {"label":"S5","rmsf_range":(215,430),
                   "key_resids":[224,422],
                   "dist_resid":224,"dist_name":"CG",
                   "state_name":"S5 (F422/D224 trap)"},
}

def get_rmsf_and_dist(gro, xtc, rmsf_range, dist_resid, dist_name, stride=STRIDE):
    u = mda.Universe(str(gro), str(xtc))
    # align on backbone excluding Loop-1 (160-164) and active site (168-187)
    align_sel = u.select_atoms(
        f"name CA and protein and not resid {LOOP1_LO}-{LOOP1_HI} and not resid 168-187")
    u.trajectory[0]
    ref_pos = align_sel.positions.copy()

    r0, r1  = rmsf_range
    ca_roi  = u.select_atoms(f"name CA and protein and resid {r0}-{r1}")
    target  = u.select_atoms(f"resid {dist_resid} and name {dist_name}")
    ligand  = u.select_atoms("resname UNK")

    ca_pos_list, dist_list = [], []
    for ts in u.trajectory[::stride]:
        mob_c = align_sel.positions.mean(0); ref_c = ref_pos.mean(0)
        H = (align_sel.positions - mob_c).T @ (ref_pos - ref_c)
        U_, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U_.T)
        R = Vt.T @ np.diag([1,1,d]) @ U_.T
        roi_aligned = (ca_roi.positions - mob_c) @ R.T + ref_c
        ca_pos_list.append(roi_aligned.copy())
        if len(target) >= 1 and len(ligand) >= 1:
            tgt = (target.positions - mob_c) @ R.T + ref_c
            lig = (ligand.positions - mob_c) @ R.T + ref_c
            dists = np.linalg.norm(lig[:,None,:] - tgt[None,:,:], axis=2)
            dist_list.append(dists.min())

    ca_pos = np.array(ca_pos_list)
    rmsf   = np.sqrt(((ca_pos - ca_pos.mean(0))**2).sum(2).mean(0))
    return ca_roi.resids, rmsf, np.array(dist_list)

results = {}
for lig, sys_dict in SYSTEMS.items():
    results[lig] = {}
    meta = STATE_RESIDUES[lig]
    print(f"\n{'='*50}\n{lig.upper()}")
    for surf, (gro, xtc) in sys_dict.items():
        if gro is None or not Path(str(xtc)).exists():
            print(f"  SKIP {surf}"); continue
        print(f"  {surf}...", flush=True)
        try:
            resids, rmsf, dists = get_rmsf_and_dist(
                gro, xtc, meta["rmsf_range"], meta["dist_resid"], meta["dist_name"])
            print(f"    n={len(dists)}  resids={resids[0]}-{resids[-1]}  "
                  f"mean_dist={np.mean(dists):.2f} Å")
            results[lig][surf] = {"resids": resids, "rmsf": rmsf, "dists": dists}
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"    ERROR: {e}")

# ── Per-ligand 3-panel figure ─────────────────────────────────────────────────
def make_figure(lig):
    meta  = STATE_RESIDUES[lig]
    surfs = [s for s in SURF_ORDER if s in results[lig]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{lig.capitalize()} — {meta['state_name']} disappears upon immobilization",
                 fontsize=12, fontweight="bold", y=1.01)

    # Panel A: RMSF profile
    ax = axes[0]
    for s in surfs:
        r = results[lig][s]
        ax.plot(r["resids"], r["rmsf"], color=SURF_COLORS[s], label=SURF_LABELS[s], lw=1.6)
    for kres in meta["key_resids"]:
        ax.axvline(kres, color="grey", lw=0.8, ls="--", alpha=0.6)
    if lig == "phenol":
        ax.axvspan(LOOP1_LO, LOOP1_HI, alpha=0.10, color="orange", label="Loop-1 (160–164)")
    r0, r1 = meta["rmsf_range"]
    ax.set_xlim(r0, r1); ax.set_xlabel("Residue", fontsize=11); ax.set_ylabel("RMSF (Å)", fontsize=11)
    ax.set_title(f"Backbone RMSF — res {r0}–{r1}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    for kres in meta["key_resids"]:
        ax.text(kres, ax.get_ylim()[1]*0.97, str(kres), ha="center",
                va="top", fontsize=7, color="grey", rotation=90)

    # Panel B: Bar chart at key residues
    ax = axes[1]
    key_res = meta["key_resids"]
    x = np.arange(len(key_res)); bar_w = 0.2
    for i, s in enumerate(surfs):
        r = results[lig][s]
        vals = [r["rmsf"][np.where(r["resids"]==kr)[0][0]]
                if len(np.where(r["resids"]==kr)[0]) else np.nan
                for kr in key_res]
        ax.bar(x + i*bar_w, vals, bar_w, label=SURF_LABELS[s],
               color=SURF_COLORS[s], alpha=0.85)
    ax.set_xticks(x + bar_w*(len(surfs)-1)/2)
    ax.set_xticklabels([f"Res {kr}" for kr in key_res], fontsize=9)
    ax.set_ylabel("RMSF (Å)", fontsize=11)
    ax.set_title(f"RMSF at key {meta['label']} residues", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # Panel C: Distance distribution
    ax = axes[2]
    for s in surfs:
        d = results[lig][s]["dists"]
        if len(d):
            ax.hist(d, bins=60, density=True, histtype="step",
                    color=SURF_COLORS[s], label=SURF_LABELS[s], lw=2)
    ax.axvline(5.0, color="black", ls="--", lw=1, label="5 Å contact")
    ax.set_xlabel(f"Min dist: ligand → {meta['dist_name']}(res {meta['dist_resid']}) (Å)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Ligand proximity to {meta['label']} anchor (res {meta['dist_resid']})",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = OUT / f"state_loss_{lig}_v2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

for lig in ["phenol","catechol","pyrogallol","resorcinol"]:
    if results.get(lig):
        make_figure(lig)

# ── 4-panel RMSF summary ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("RMSF at State-Defining Residues — Effect of Immobilization",
             fontsize=13, fontweight="bold")

configs = [
    ("phenol",    axes[0,0], [160,161,162,163,164], "S4 — Loop-1 (160–164)\n(Phenol)"),
    ("catechol",  axes[0,1], [343,344,345],           "S3 — S343/F344/T345\n(Catechol)"),
    ("pyrogallol",axes[1,0], [343,344,345],           "S3 — S343/F344/T345\n(Pyrogallol)"),
    ("resorcinol",axes[1,1], [224,422],               "S5 — D224/F422\n(Resorcinol)"),
]

for lig, ax, key_res, title in configs:
    if not results.get(lig): ax.set_visible(False); continue
    surfs = [s for s in SURF_ORDER if s in results[lig]]
    x = np.arange(len(key_res)); bar_w = 0.18
    for i, s in enumerate(surfs):
        r = results[lig][s]
        vals = [r["rmsf"][np.where(r["resids"]==kr)[0][0]]
                if len(np.where(r["resids"]==kr)[0]) else np.nan
                for kr in key_res]
        ax.bar(x + i*bar_w, vals, bar_w, label=SURF_LABELS[s],
               color=SURF_COLORS[s], alpha=0.87)
    ax.set_xticks(x + bar_w*(len(surfs)-1)/2)
    ax.set_xticklabels([f"Res {kr}" for kr in key_res], fontsize=10)
    ax.set_ylabel("RMSF (Å)", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

fig.tight_layout()
out = OUT / "state_loss_rmsf_summary_v2.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ── CSV export ────────────────────────────────────────────────────────────────
with open(OUT/"state_rmsf_keyresidues_v2.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["Ligand","Surface","Residue","RMSF_A"])
    for lig in results:
        for s in SURF_ORDER:
            if s not in results[lig]: continue
            r = results[lig][s]
            for res, val in zip(r["resids"], r["rmsf"]):
                w.writerow([lig, SURF_LABELS[s], int(res), f"{val:.4f}"])
print(f"Saved: {OUT/'state_rmsf_keyresidues_v2.csv'}")

print("\nAll done.")
