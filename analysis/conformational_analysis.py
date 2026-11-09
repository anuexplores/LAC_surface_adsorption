#!/usr/bin/env python3
"""
Conformational change analysis across bulk / silica / lysine / APTES:
  1. T180 chi1 dihedral distribution (N-CA-CB-OG1)
  2. Full-protein backbone RMSF (aligned)
  3. Active-site inter-residue distances (T180-OG1 to G175-O, T180-CB to G175-CA)
  4. Protein-surface contact map (residues within 5 Å of surface atoms)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import MDAnalysis as mda
import warnings
warnings.filterwarnings("ignore")
import csv

OUT_DIR = Path("/Data/Anushka/Project4/ph5/orients/orient1/tleap/energy_landscapes")
STRIDE  = 10

SURF_COLORS = {"bulk":"#2171B5","silica":"#CB181D","lysine":"#984EA3","aptes":"#238B45"}
SURF_LABELS = {"bulk":"Bulk","silica":"Silica","lysine":"Lysine","aptes":"APTES"}
SURF_ORDER  = ["bulk","silica","lysine","aptes"]

# ── trajectories ──────────────────────────────────────────────────────────────
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

# ── helpers ───────────────────────────────────────────────────────────────────
def dihedral_angle(p1, p2, p3, p4):
    """Praxitelous dihedral for arrays of shape (N,3)."""
    b1 = p2 - p1;  b2 = p3 - p2;  b3 = p4 - p3
    n1 = np.cross(b1, b2);  n2 = np.cross(b2, b3)
    n1 /= (np.linalg.norm(n1, axis=-1, keepdims=True) + 1e-12)
    n2 /= (np.linalg.norm(n2, axis=-1, keepdims=True) + 1e-12)
    m1  = np.cross(n1, b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12))
    x   = np.sum(n1 * n2, axis=-1)
    y   = np.sum(m1 * n2, axis=-1)
    return np.degrees(np.arctan2(y, x))

def svd_align(mob_pos, ref_pos):
    """Return rotation matrix R and translation t to align mob onto ref."""
    ref_c = ref_pos.mean(0);  mob_c = mob_pos.mean(0)
    H  = (mob_pos - mob_c).T @ (ref_pos - ref_c)
    U, _, Vt = np.linalg.svd(H)
    d  = np.linalg.det(Vt.T @ U.T)
    R  = Vt.T @ np.diag([1,1,d]) @ U.T
    return R, ref_c - mob_c @ R.T

def aligned_pos(sel_align, sel_target, ref_align_pos, ref_target_pos=None):
    """Apply backbone alignment and return transformed target positions."""
    mob_c  = sel_align.positions.mean(0)
    ref_c  = ref_align_pos.mean(0)
    H = (sel_align.positions - mob_c).T @ (ref_align_pos - ref_c)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1,1,d]) @ U.T
    return (sel_target.positions - mob_c) @ R.T + ref_c

# ── main collection ───────────────────────────────────────────────────────────
results = {}

for lig_name, sys_dict in SYSTEMS.items():
    results[lig_name] = {}
    for surf, (gro, xtc) in sys_dict.items():
        if not Path(xtc).exists():
            print(f"  SKIP {lig_name}/{surf}"); continue
        print(f"  {lig_name}/{surf}...", flush=True)
        try:
            u = mda.Universe(str(gro), str(xtc))
            protein = u.select_atoms("protein")

            # --- alignment reference (backbone excluding loop 168-187) ---
            align_ref_sel = u.select_atoms(
                "name CA and protein and not resid 168-187")
            u.trajectory[0]
            ref_align_pos = align_ref_sel.positions.copy()

            # --- selections ---
            ca_all  = u.select_atoms("name CA and protein")
            t180_N  = u.select_atoms("resid 180 and name N")
            t180_CA = u.select_atoms("resid 180 and name CA")
            t180_CB = u.select_atoms("resid 180 and name CB")
            t180_OG = u.select_atoms("resid 180 and name OG1")
            g175_CA = u.select_atoms("resid 175 and name CA")
            g175_O  = u.select_atoms("resid 175 and name O")
            # surface atoms (SIO2 / SI / OB or surface residue names)
            surf_atoms = u.select_atoms(
                "not protein and not resname SOL WAT HOH CL NA K UNK")

            chi1_ts, d_t180_g175, d_cb_ca, surf_contacts = [], [], [], []
            ca_positions = []

            for ts in u.trajectory[::STRIDE]:
                # 1. T180 chi1 dihedral
                if all(len(s)==1 for s in [t180_N,t180_CA,t180_CB,t180_OG]):
                    p = np.stack([t180_N.positions[0], t180_CA.positions[0],
                                  t180_CB.positions[0], t180_OG.positions[0]])
                    chi1_ts.append(dihedral_angle(
                        p[None,0], p[None,1], p[None,2], p[None,3])[0])

                # 2. Active-site distances (OG1 to G175-O backbone)
                if len(t180_OG)==1 and len(g175_O)==1:
                    d_t180_g175.append(
                        np.linalg.norm(t180_OG.positions[0]-g175_O.positions[0]))
                if len(t180_CB)==1 and len(g175_CA)==1:
                    d_cb_ca.append(
                        np.linalg.norm(t180_CB.positions[0]-g175_CA.positions[0]))

                # 3. Aligned Cα positions for RMSF
                if len(align_ref_sel) >= 5:
                    ap = aligned_pos(align_ref_sel, ca_all, ref_align_pos)
                    ca_positions.append(ap.copy())

                # 4. Surface contacts: protein residues within 5 Å of surface
                if len(surf_atoms) > 0:
                    prot_pos = protein.positions
                    surf_pos = surf_atoms.positions
                    dists = np.linalg.norm(
                        prot_pos[:,None,:] - surf_pos[None,:,:], axis=2)
                    close_mask = (dists < 5.0).any(axis=1)
                    surf_contacts.append(protein.resids[close_mask])

            # compute per-residue RMSF
            ca_positions = np.array(ca_positions)
            mean_pos = ca_positions.mean(0)
            rmsf_all = np.sqrt(((ca_positions-mean_pos)**2).sum(2).mean(0))

            # most-contacted surface residues (mode across frames)
            if surf_contacts:
                from collections import Counter
                all_contacts = [r for frame in surf_contacts for r in frame]
                contact_counts = Counter(all_contacts)
                total_frames   = len(surf_contacts)
                contact_freq   = {r: c/total_frames for r,c in contact_counts.items()}
            else:
                contact_freq = {}

            results[lig_name][surf] = {
                "chi1":          np.array(chi1_ts),
                "d_OG1_G175O":   np.array(d_t180_g175),
                "d_CB_G175CA":   np.array(d_cb_ca),
                "rmsf_resids":   ca_all.resids,
                "rmsf_vals":     rmsf_all,
                "contact_freq":  contact_freq,
                "n_frames":      len(chi1_ts),
            }
            chi1_mean = np.mean(chi1_ts) if chi1_ts else float('nan')
            print(f"    χ1_mean={chi1_mean:.1f}°  "
                  f"d_OG1–G175O_mean={np.mean(d_t180_g175):.2f} Å  "
                  f"({len(chi1_ts)} frames)")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"    ERROR: {e}")

# ── plotting ──────────────────────────────────────────────────────────────────
for lig_name in SYSTEMS:
    if not results.get(lig_name): continue
    surfs = [s for s in SURF_ORDER if s in results[lig_name]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{lig_name} — Conformational Analysis", fontsize=13, fontweight="bold")

    # ── panel A: T180 chi1 distribution ──────────────────────────────────────
    ax = axes[0,0]
    for s in surfs:
        chi1 = results[lig_name][s]["chi1"]
        if len(chi1): ax.hist(chi1, bins=72, range=(-180,180), density=True,
                histtype="step", color=SURF_COLORS[s], label=SURF_LABELS[s], lw=2)
    ax.set_xlabel("T180 χ₁ dihedral (°)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("T180 χ₁ — OG1 orientation", fontsize=11, fontweight="bold")
    ax.set_xlim(-180,180)
    ax.axvspan(-180,-60, alpha=0.07, color="green", label="OG1 toward pocket")
    ax.axvspan(60, 180, alpha=0.07, color="red",   label="OG1 away from pocket")
    ax.legend(fontsize=8)

    # ── panel B: OG1(T180) – O(G175) distance distribution ──────────────────
    ax = axes[0,1]
    for s in surfs:
        d = results[lig_name][s]["d_OG1_G175O"]
        if len(d): ax.hist(d, bins=60, range=(3,18), density=True,
                histtype="step", color=SURF_COLORS[s], label=SURF_LABELS[s], lw=2)
    ax.set_xlabel("d(OG1 T180 – O G175) (Å)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Active-site pocket distance", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # ── panel C: full-protein RMSF profile ───────────────────────────────────
    ax = axes[1,0]
    for s in surfs:
        rids = results[lig_name][s]["rmsf_resids"]
        rmsf = results[lig_name][s]["rmsf_vals"]
        ax.plot(rids, rmsf, color=SURF_COLORS[s], label=SURF_LABELS[s], lw=1.2, alpha=0.85)
    ax.axvspan(168,187, alpha=0.12, color="gold", label="G175–T180 loop")
    ax.set_xlabel("Residue", fontsize=11)
    ax.set_ylabel("RMSF (Å)", fontsize=11)
    ax.set_title("Full-protein RMSF (backbone-aligned)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    # ── panel D: surface contact frequency heatmap ───────────────────────────
    ax = axes[1,1]
    surf_surfs = [s for s in surfs if s != "bulk"]
    all_res = sorted(set(
        r for s in surf_surfs for r in results[lig_name][s]["contact_freq"]))
    if all_res and surf_surfs:
        mat = np.zeros((len(surf_surfs), len(all_res)))
        for i, s in enumerate(surf_surfs):
            cf = results[lig_name][s]["contact_freq"]
            for j, r in enumerate(all_res):
                mat[i,j] = cf.get(r, 0.0)
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=1, origin="upper")
        ax.set_yticks(range(len(surf_surfs)))
        ax.set_yticklabels([SURF_LABELS[s] for s in surf_surfs], fontsize=9)
        ax.set_xlabel("Residue", fontsize=11)
        # x-ticks every 20 residues
        tick_idx = list(range(0, len(all_res), max(1,len(all_res)//20)))
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([all_res[i] for i in tick_idx], fontsize=7, rotation=45)
        plt.colorbar(im, ax=ax, label="Contact frequency", shrink=0.8)
        ax.set_title("Protein–surface contact frequency", fontsize=11, fontweight="bold")
        # mark G175/T180 on heatmap
        for res_mark in [175, 180]:
            if res_mark in all_res:
                xi = all_res.index(res_mark)
                ax.axvline(xi, color="blue", lw=1.5, ls="--", alpha=0.7)
    else:
        ax.text(0.5,0.5,"No surface atoms found\n(bulk only)", ha="center",
                va="center", transform=ax.transAxes, color="grey")
        ax.set_title("Protein–surface contact frequency", fontsize=11, fontweight="bold")

    fig.tight_layout()
    out = OUT_DIR / f"conformational_{lig_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")

# ── CSV export ────────────────────────────────────────────────────────────────

# chi1 time series
with open(OUT_DIR/"T180_chi1.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["Ligand","Surface","Frame","chi1_deg"])
    for lig_name in SYSTEMS:
        for s in SURF_ORDER:
            if s not in results.get(lig_name,{}): continue
            for i,v in enumerate(results[lig_name][s]["chi1"]):
                w.writerow([lig_name, SURF_LABELS[s], i, f"{v:.3f}"])
print(f"Saved: {OUT_DIR/'T180_chi1.csv'}")

# active-site distances
with open(OUT_DIR/"activesite_distances.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["Ligand","Surface","Frame","d_OG1_T180_to_O_G175_A","d_CB_T180_to_CA_G175_A"])
    for lig_name in SYSTEMS:
        for s in SURF_ORDER:
            if s not in results.get(lig_name,{}): continue
            r = results[lig_name][s]
            for i,(d1,d2) in enumerate(zip(r["d_OG1_G175O"],r["d_CB_G175CA"])):
                w.writerow([lig_name, SURF_LABELS[s], i, f"{d1:.4f}", f"{d2:.4f}"])
print(f"Saved: {OUT_DIR/'activesite_distances.csv'}")

# full protein RMSF
with open(OUT_DIR/"fullprotein_rmsf.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["Ligand","Surface","Residue","RMSF_A"])
    for lig_name in SYSTEMS:
        for s in SURF_ORDER:
            if s not in results.get(lig_name,{}): continue
            r = results[lig_name][s]
            for res,val in zip(r["rmsf_resids"],r["rmsf_vals"]):
                w.writerow([lig_name, SURF_LABELS[s], int(res), f"{val:.4f}"])
print(f"Saved: {OUT_DIR/'fullprotein_rmsf.csv'}")

# surface contact frequency
with open(OUT_DIR/"surface_contacts.csv","w",newline="") as f:
    w = csv.writer(f)
    w.writerow(["Ligand","Surface","Residue","Contact_frequency"])
    for lig_name in SYSTEMS:
        for s in SURF_ORDER:
            if s not in results.get(lig_name,{}): continue
            cf = results[lig_name][s]["contact_freq"]
            for res,freq in sorted(cf.items()):
                w.writerow([lig_name, SURF_LABELS[s], int(res), f"{freq:.4f}"])
print(f"Saved: {OUT_DIR/'surface_contacts.csv'}")

print("\nAll done.")
