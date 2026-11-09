#!/usr/bin/env python3
"""
Binding events and residence time analysis.
Measures minimum distance between each UNK molecule and T1 Cu (resid 500)
over the trajectory. Defines binding event as distance < CUTOFF Angstroms.
"""

import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── Configuration ──────────────────────────────────────────────────────────────
CUTOFF   = 5.0   # Å — distance threshold for binding
N_MOLS   = 6     # ligand molecules per system
MOL_ATOMS = {'catechol': 14, 'hydroquinone': 14}

SYSTEMS = [
    {
        'name':      'catechol_rep1',
        'ligand':    'catechol',
        'gro':       '/Data/Anushka/Project4/ph5/orients/orient1/tleap/aptes_catechol/MD/1000ns_posres.gro',
        'xtc':       '/Data/Anushka/Project4/ph5/orients/orient1/tleap/aptes_catechol/MD/1000ns_posres_noPBC.xtc',
    },
    {
        'name':      'hydroquinone_rep1',
        'ligand':    'hydroquinone',
        'gro':       '/Data/Anushka/Project4/ph5/orients/orient1/tleap/aptes_hydroquinone/MD/rep1/200ns_posres.gro',
        'xtc':       '/Data/Anushka/Project4/ph5/orients/orient1/tleap/aptes_hydroquinone/MD/rep1/200ns_posres_noPBC.xtc',
    },
]

OUT_DIR = '/Data/Anushka/Project4/ph5/orients/orient1/tleap/binding_analysis'
os.makedirs(OUT_DIR, exist_ok=True)


def analyse_system(sys):
    name    = sys['name']
    ligand  = sys['ligand']
    mol_atm = MOL_ATOMS[ligand]

    print(f"\n{'='*60}")
    print(f"System: {name}")
    print(f"{'='*60}")

    u = mda.Universe(sys['gro'], sys['xtc'])
    n_frames = len(u.trajectory)
    dt = u.trajectory.dt          # ps per frame
    total_time_ns = n_frames * dt / 1000

    print(f"  Frames: {n_frames}  |  dt: {dt:.0f} ps  |  Total: {total_time_ns:.0f} ns")

    # Select T1 Cu (resid 500 — first Cu, the mononuclear blue copper)
    t1_cu = u.select_atoms('resid 500 and name CU')
    if len(t1_cu) == 0:
        # fallback: first Cu atom
        t1_cu = u.select_atoms('name CU')[0:1]
    print(f"  T1 Cu: {len(t1_cu)} atom(s) — resid {t1_cu.resids}")

    # Select UNK atoms and split into molecules
    unk_all = u.select_atoms('resname UNK')
    print(f"  UNK atoms: {len(unk_all)}  ({len(unk_all)//mol_atm} molecules × {mol_atm} atoms)")

    # Split UNK into individual molecules by resid
    unk_resids = sorted(set(unk_all.resids))
    unk_mols   = [u.select_atoms(f'resname UNK and resid {r}') for r in unk_resids]

    # ── Compute min distance: each molecule → T1 Cu, every frame ──────────────
    n_mols = len(unk_mols)
    min_dist = np.zeros((n_frames, n_mols))  # Å

    for fi, ts in enumerate(u.trajectory):
        cu_pos = t1_cu.positions  # (1, 3)
        for mi, mol in enumerate(unk_mols):
            dists = distances.distance_array(mol.positions, cu_pos)
            min_dist[fi, mi] = dists.min()

    time_ns = np.arange(n_frames) * dt / 1000  # ns

    # ── Binding analysis per molecule ─────────────────────────────────────────
    print(f"\n  Cutoff: {CUTOFF} Å")
    print(f"  {'Mol':>4}  {'Events':>7}  {'Mean res (ns)':>14}  {'Max res (ns)':>13}  "
          f"{'Total bound (ns)':>17}  {'% bound':>8}")

    all_events = []
    all_residence = []

    for mi in range(n_mols):
        d = min_dist[:, mi]
        bound = d < CUTOFF  # boolean array

        # Find binding events (runs of True)
        events = []
        in_event = False
        start = 0
        for fi in range(n_frames):
            if bound[fi] and not in_event:
                in_event = True
                start = fi
            elif not bound[fi] and in_event:
                in_event = False
                duration_ns = (fi - start) * dt / 1000
                events.append(duration_ns)
        if in_event:  # still bound at end
            duration_ns = (n_frames - start) * dt / 1000
            events.append(duration_ns)

        n_events      = len(events)
        mean_res_ns   = np.mean(events)  if events else 0.0
        max_res_ns    = np.max(events)   if events else 0.0
        total_bound   = sum(events)
        pct_bound     = 100 * total_bound / total_time_ns

        print(f"  {unk_resids[mi]:>4}  {n_events:>7}  {mean_res_ns:>14.3f}  "
              f"{max_res_ns:>13.3f}  {total_bound:>17.3f}  {pct_bound:>7.1f}%")

        all_events.extend(events)
        all_residence.extend(events)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n  --- Summary across all {n_mols} molecules ---")
    print(f"  Total binding events : {len(all_events)}")
    print(f"  Mean residence time  : {np.mean(all_residence):.3f} ns" if all_residence else "  No events")
    print(f"  Median residence time: {np.median(all_residence):.3f} ns" if all_residence else "")
    print(f"  Max residence time   : {np.max(all_residence):.3f} ns"  if all_residence else "")

    # ── Save distance vs time data ─────────────────────────────────────────────
    np.save(os.path.join(OUT_DIR, f'{name}_mindist.npy'), min_dist)
    np.save(os.path.join(OUT_DIR, f'{name}_time.npy'),    time_ns)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(n_mols, 1, figsize=(12, 2*n_mols), sharex=True)
    if n_mols == 1: axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, n_mols))

    for mi in range(n_mols):
        ax = axes[mi]
        ax.plot(time_ns, min_dist[:, mi], color=colors[mi], lw=0.5, alpha=0.8)
        ax.axhline(CUTOFF, color='red', ls='--', lw=1, label=f'{CUTOFF} Å cutoff')
        ax.fill_between(time_ns, 0, CUTOFF,
                        where=min_dist[:, mi] < CUTOFF,
                        alpha=0.3, color=colors[mi], label='bound')
        ax.set_ylabel(f'Mol {unk_resids[mi]}\ndist (Å)', fontsize=8)
        ax.set_ylim(0, 30)
        ax.legend(fontsize=7, loc='upper right')

    axes[-1].set_xlabel('Time (ns)', fontsize=10)
    fig.suptitle(f'{name} — UNK min distance to T1 Cu', fontsize=11, y=1.01)
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, f'{name}_binding.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Plot saved: {plot_path}")

    return {
        'name':          name,
        'n_events':      len(all_events),
        'mean_res_ns':   np.mean(all_residence) if all_residence else 0,
        'median_res_ns': np.median(all_residence) if all_residence else 0,
        'max_res_ns':    np.max(all_residence) if all_residence else 0,
        'min_dist':      min_dist,
        'time_ns':       time_ns,
    }


# ── Run analysis ──────────────────────────────────────────────────────────────
results = []
for sys in SYSTEMS:
    r = analyse_system(sys)
    results.append(r)

# ── Comparison summary ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("COMPARISON SUMMARY")
print(f"{'='*60}")
print(f"{'System':<25} {'Events':>7} {'Mean res (ns)':>14} {'Max res (ns)':>13}")
print("-"*60)
for r in results:
    print(f"{r['name']:<25} {r['n_events']:>7} {r['mean_res_ns']:>14.3f} {r['max_res_ns']:>13.3f}")
