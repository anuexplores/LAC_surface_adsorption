#!/usr/bin/env python3
"""
Compute MSD for ligands in bulk (enzyme in solution) and on surfaces
(silica, APTES). Plots <r²> vs time for all systems.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import MDAnalysis as mda
import warnings
warnings.filterwarnings("ignore")

# ── directory roots ─────────────────────────────────────────────────────────
MD_DOCKED = Path("/media/anushka/One Touch/ws_AB4/Anushka/MD-docked")
TRIHYDRIC  = Path("/Data/Anushka/Project4/anushka@10.0.66.94/Trihydric")
SILICA_BASE = Path("/Data/Anushka/Project4/ph5/orients/orient1/tleap/silica")
APTES_BASE  = Path("/Data/Anushka/Project4/anushka@10.0.66.94/P4/orient1/ph5/aptes")
LYSINE_BASE = Path("/Data/Anushka/Project4/anushka@10.0.66.94/P4/orient1/ph5/lysine")
OUT_DIR    = Path("/Data/Anushka/Project4/ph5/orients/orient1/tleap/energy_landscapes")

LIGANDS = ["25xylidine","aniline","catechol","hydroquinone",
           "hydroxyquinol","phenol","phloroglucinol","pyrogallol","resorcinol"]

# ── trajectory maps ──────────────────────────────────────────────────────────
XYLIDINE_BULK = Path("/Data/Anushka/Project4/anushka@10.0.66.94/xylidine_wsAB/packed/MD/RC")

BULK_MAP = {
    "25xylidine":    (XYLIDINE_BULK/"2000ns.gro",
                      XYLIDINE_BULK/"2000ns_noPBC.xtc"),
    "aniline":       (MD_DOCKED/"aniline/MD/set1/RC/1000ns.gro",
                      MD_DOCKED/"aniline/MD/set1/RC/1000ns_noPBC.xtc"),
    "phenol":        (MD_DOCKED/"Phenol/MD/set1/RC/1000ns.gro",
                      MD_DOCKED/"Phenol/MD/set1/RC/1000ns_noPBC.xtc"),
    "pyrogallol":    (TRIHYDRIC/"Pyrogallol/MD/set1/RC/1000ns.gro",
                      TRIHYDRIC/"Pyrogallol/MD/set1/RC/1000ns_noPBC.xtc"),
    "phloroglucinol":(TRIHYDRIC/"Phloroglucinol/MD/set1/1000ns.gro",
                      TRIHYDRIC/"Phloroglucinol/MD/set1/1000ns_noPBC.xtc"),
    "hydroxyquinol": (TRIHYDRIC/"Hydroxyquinol/MD/set1/1000ns.gro",
                      TRIHYDRIC/"Hydroxyquinol/MD/set1/RC/2000ns_noPBC.xtc"),
}

def silica_paths(lig):
    d = SILICA_BASE / f"silica_{lig}/MD"
    return d/f"silica_{lig}.gro", d/"200ns_noPBC.xtc"

def aptes_paths(lig):
    d = APTES_BASE / f"aptes_{lig}/MD/rep5"
    gros = sorted(d.glob("*.gro")) if d.exists() else []
    gro = gros[0] if gros else None
    xtc = d/"200ns_noPBC.xtc"
    return gro, xtc

def lysine_paths(lig):
    d = LYSINE_BASE / f"lys_silica_{lig}/MD"
    gros = sorted(d.glob(f"lys_silica_{lig}.gro")) if d.exists() else []
    gro = gros[0] if gros else None
    xtc = d/"200ns_noPBC.xtc"
    return gro, xtc

# ── MSD computation ──────────────────────────────────────────────────────────
def get_com_positions(gro, xtc, stride=10):
    """Return (times_ns, COM_positions_array) for ligand UNK."""
    u  = mda.Universe(str(gro), str(xtc))
    ag = u.select_atoms("resname UNK")
    if len(ag) == 0:
        return None, None
    masses = ag.masses
    total_m = masses.sum()
    positions = []
    times     = []
    for ts in u.trajectory[::stride]:
        com = (ag.positions * masses[:,None]).sum(axis=0) / total_m
        positions.append(com.copy())
        times.append(ts.time / 1000.0)   # ps → ns
    return np.array(times), np.array(positions)

def msd_from_positions(pos):
    """Windowed MSD: MSD[tau] = <|r(t+tau)-r(t)|^2>_t  in Å²."""
    n = len(pos)
    max_lag = n // 2
    msd = np.zeros(max_lag)
    for lag in range(1, max_lag):
        diff = pos[lag:] - pos[:-lag]
        msd[lag] = np.mean(np.einsum('ij,ij->i', diff, diff))
    return msd

def diffusion_coeff(times_ns, msd_A2, fit_frac=(0.05, 0.25)):
    """Fit linear region of MSD to get D in cm²/s."""
    t = np.array(times_ns)
    m = np.array(msd_A2)
    lo = int(len(t) * fit_frac[0])
    hi = int(len(t) * fit_frac[1])
    if hi <= lo + 2: hi = lo + max(3, len(t)//5)
    hi = min(hi, len(t))
    slope, intercept = np.polyfit(t[lo:hi], m[lo:hi], 1)   # Å²/ns
    D_A2_per_ns = slope / 6.0                               # 3D: MSD = 6Dt
    D_cm2_per_s = D_A2_per_ns * 1e-16 / 1e-9               # Å²/ns → cm²/s
    return max(D_cm2_per_s, 0)

# ── collect data ─────────────────────────────────────────────────────────────
STRIDE_BULK = 20   # every 20th frame of 1000 ns traj
STRIDE_SURF = 5    # every 5th frame of 200 ns traj

results = {lig: {} for lig in LIGANDS}

print("Computing MSD...")

# --- bulk ---
for lig, (gro, xtc) in BULK_MAP.items():
    if not Path(xtc).exists():
        print(f"  SKIP bulk {lig}: {xtc} not found")
        continue
    print(f"  bulk {lig}...", flush=True)
    try:
        t, pos = get_com_positions(gro, xtc, stride=STRIDE_BULK)
        if pos is not None and len(pos) > 10:
            msd = msd_from_positions(pos)
            lag_t = t[:len(msd)] - t[0]
            results[lig]["bulk"] = (lag_t, msd)
            D = diffusion_coeff(lag_t, msd)
            print(f"    D_bulk = {D:.3e} cm²/s  ({len(pos)} frames)")
    except Exception as e:
        print(f"    ERROR: {e}")

# --- silica ---
for lig in LIGANDS:
    gro, xtc = silica_paths(lig)
    if not xtc.exists():
        print(f"  SKIP silica {lig}: no xtc")
        continue
    print(f"  silica {lig}...", flush=True)
    try:
        t, pos = get_com_positions(gro, xtc, stride=STRIDE_SURF)
        if pos is not None and len(pos) > 10:
            msd = msd_from_positions(pos)
            lag_t = t[:len(msd)] - t[0]
            results[lig]["silica"] = (lag_t, msd)
            D = diffusion_coeff(lag_t, msd)
            print(f"    D_silica = {D:.3e} cm²/s  ({len(pos)} frames)")
    except Exception as e:
        print(f"    ERROR: {e}")

# --- aptes ---
for lig in LIGANDS:
    gro, xtc = aptes_paths(lig)
    if gro is None or not xtc.exists():
        print(f"  SKIP aptes {lig}: missing files")
        continue
    print(f"  aptes {lig}...", flush=True)
    try:
        t, pos = get_com_positions(gro, xtc, stride=STRIDE_SURF)
        if pos is not None and len(pos) > 10:
            msd = msd_from_positions(pos)
            lag_t = t[:len(msd)] - t[0]
            results[lig]["aptes"] = (lag_t, msd)
            D = diffusion_coeff(lag_t, msd)
            print(f"    D_aptes  = {D:.3e} cm²/s  ({len(pos)} frames)")
    except Exception as e:
        print(f"    ERROR: {e}")

# --- lysine ---
for lig in LIGANDS:
    gro, xtc = lysine_paths(lig)
    if gro is None or not xtc.exists():
        print(f"  SKIP lysine {lig}: missing files")
        continue
    print(f"  lysine {lig}...", flush=True)
    try:
        t, pos = get_com_positions(gro, xtc, stride=STRIDE_SURF)
        if pos is not None and len(pos) > 10:
            msd = msd_from_positions(pos)
            lag_t = t[:len(msd)] - t[0]
            results[lig]["lysine"] = (lag_t, msd)
            D = diffusion_coeff(lag_t, msd)
            print(f"    D_lysine = {D:.3e} cm²/s  ({len(pos)} frames)")
    except Exception as e:
        print(f"    ERROR: {e}")

# ── plotting ─────────────────────────────────────────────────────────────────
SYSTEM_COLORS = {"bulk": "#2171B5", "silica": "#CB181D", "aptes": "#238B45", "lysine": "#984EA3"}
SYSTEM_LABELS = {"bulk": "Bulk (in solution)", "silica": "Silica", "aptes": "APTES", "lysine": "Lysine"}

# --- Fig 1: one subplot per ligand ---
ligs_with_data = [l for l in LIGANDS if results[l]]
ncols = 5
nrows = (len(ligs_with_data) + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.5 * nrows),
                         sharex=False, sharey=False)
axes = axes.flatten()

for ax_i, lig in enumerate(ligs_with_data):
    ax = axes[ax_i]
    # find common max lag time across all available systems for this ligand
    max_lags = [results[lig][s][0][-1] for s in ["bulk","silica","lysine","aptes"]
                if s in results[lig]]
    t_max = min(max_lags) if max_lags else None
    for sys_name in ["bulk", "silica", "lysine", "aptes"]:
        if sys_name not in results[lig]:
            continue
        lag_t, msd = results[lig][sys_name]
        mask = lag_t <= t_max
        ax.plot(lag_t[mask], msd[mask], color=SYSTEM_COLORS[sys_name],
                label=SYSTEM_LABELS[sys_name], lw=1.8)
    ax.set_title(lig, fontsize=10, fontweight="bold")
    ax.set_xlabel("lag time (ns)", fontsize=8)
    ax.set_ylabel(r"$\langle r^2 \rangle$ (Å²)", fontsize=8)
    ax.tick_params(labelsize=7)
    if t_max is not None:
        ax.set_xlim(0, t_max)
    if ax_i == 0:
        ax.legend(fontsize=7)

for ax_i in range(len(ligs_with_data), len(axes)):
    axes[ax_i].set_visible(False)

fig.suptitle("Mean Square Displacement — Bulk vs Immobilised", fontsize=13, y=1.01)
fig.tight_layout()
out1 = OUT_DIR / "msd_per_ligand.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out1}")

# --- Fig 2: one panel per surface, all ligands ---
lig_colours = plt.cm.tab10(np.linspace(0, 1, 10))
fig2, axes2 = plt.subplots(1, 4, figsize=(22, 5))
panels = [("bulk", axes2[0]), ("silica", axes2[1]), ("lysine", axes2[2]), ("aptes", axes2[3])]

for sys_name, ax in panels:
    plotted = 0
    for k, lig in enumerate(LIGANDS):
        if sys_name not in results[lig]:
            continue
        lag_t, msd = results[lig][sys_name]
        ax.plot(lag_t, msd, color=lig_colours[k], label=lig, lw=1.8)
        plotted += 1
    ax.set_title(SYSTEM_LABELS[sys_name], fontsize=11, fontweight="bold")
    ax.set_xlabel("lag time (ns)", fontsize=10)
    ax.set_ylabel(r"$\langle r^2 \rangle$ (Å²)", fontsize=10)
    if plotted > 0:
        ax.legend(fontsize=7, ncol=2)

fig2.suptitle("MSD per surface — all ligands", fontsize=13)
fig2.tight_layout()
out2 = OUT_DIR / "msd_by_surface.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2}")

# ── diffusion coefficient summary ────────────────────────────────────────────
import csv

print("\n=== Diffusion Coefficients (cm²/s) ===")
print(f"{'Ligand':15s}  {'Bulk':12s}  {'Silica':12s}  {'Lysine':12s}  {'APTES':12s}")

# CSV 1: D coefficients
with open(OUT_DIR / "diffusion_coefficients.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Ligand", "Bulk_D_cm2_per_s", "Silica_D_cm2_per_s",
                "Lysine_D_cm2_per_s", "APTES_D_cm2_per_s"])
    for lig in LIGANDS:
        row_print = []
        row_csv   = [lig]
        for sys in ["bulk", "silica", "lysine", "aptes"]:
            if sys in results[lig]:
                lag_t, msd = results[lig][sys]
                D = diffusion_coeff(lag_t, msd)
                row_print.append(f"{D:.3e}")
                row_csv.append(f"{D:.6e}")
            else:
                row_print.append("   --   ")
                row_csv.append("")
        print(f"{lig:15s}  {'  '.join(row_print)}")
        w.writerow(row_csv)
print(f"Saved: {OUT_DIR/'diffusion_coefficients.csv'}")

# CSV 2: MSD curves per ligand per surface
with open(OUT_DIR / "msd_curves.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Ligand", "Surface", "lag_time_ns", "MSD_A2"])
    for lig in LIGANDS:
        for sys in ["bulk", "silica", "lysine", "aptes"]:
            if sys not in results[lig]:
                continue
            lag_t, msd = results[lig][sys]
            # clip to common time
            max_lags = [results[lig][s][0][-1] for s in ["bulk","silica","lysine","aptes"]
                        if s in results[lig]]
            t_max = min(max_lags)
            mask  = lag_t <= t_max
            for t, m in zip(lag_t[mask], msd[mask]):
                w.writerow([lig, sys, f"{t:.4f}", f"{m:.4f}"])
print(f"Saved: {OUT_DIR/'msd_curves.csv'}")
