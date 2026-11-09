#!/usr/bin/env python3
"""
Contact analysis: which enzyme residues interact with the poly-LYS surface (SLY)?
Cutoff: 4.5 Å, any heavy atom of enzyme residue vs any heavy atom of SLY chains.
Outputs: polylys_contacts.csv and polylys_contacts.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import capped_distance
import warnings, csv
warnings.filterwarnings("ignore")

MD   = "/Data/Anushka/Project4/anushka@10.0.66.94/P4/orient1/ph5/lysine/polyLysine/MD"
GRO  = f"{MD}/100ns.gro"
XTC  = f"{MD}/100ns_noPBC.xtc"
OUT  = "/Data/Anushka/Project4/ph5/orients/orient1/tleap/energy_landscapes"
CUTOFF = 4.5   # Å
STRIDE = 10

u = mda.Universe(GRO, XTC)

# Enzyme heavy atoms (protein, not SLY)
enzyme = u.select_atoms("protein and not resname SLY and not name H*")
sly    = u.select_atoms("resname SLY and not name H*")

print(f"Enzyme heavy atoms : {len(enzyme)}")
print(f"SLY heavy atoms    : {len(sly)}")
print(f"Trajectory frames  : {len(u.trajectory)}")

# Per-residue contact count
resids   = np.unique(enzyme.resids)
res_hits = {r: 0 for r in resids}
res_name = {a.resid: a.resname for a in enzyme}
n_frames = 0

for ts in u.trajectory[::STRIDE]:
    pairs, _ = capped_distance(enzyme.positions, sly.positions,
                                max_cutoff=CUTOFF, box=ts.dimensions)
    if len(pairs):
        hit_resids = np.unique(enzyme[pairs[:, 0]].resids)
        for r in hit_resids:
            res_hits[r] += 1
    n_frames += 1
    if n_frames % 50 == 0:
        print(f"  frame {n_frames*STRIDE}…", flush=True)

print(f"Done. {n_frames} frames analysed.")

# Contact frequency (%)
rows = []
for r in resids:
    freq = res_hits[r] / n_frames * 100
    rows.append({"Residue": int(r), "Resname": res_name[r],
                 "ContactFreq_pct": round(freq, 2)})

df = pd.DataFrame(rows).sort_values("ContactFreq_pct", ascending=False)
csv_out = f"{OUT}/polylys_contacts.csv"
df.to_csv(csv_out, index=False)
print(f"Saved: {csv_out}")

# Print top contacts
print("\nTop 20 contacting residues:")
print(df.head(20).to_string(index=False))

# ── Plot ──────────────────────────────────────────────────────────────────────
# Show all residues with >5% contact frequency
sig = df[df["ContactFreq_pct"] > 5].sort_values("Residue")
fig, ax = plt.subplots(figsize=(max(10, len(sig)*0.4), 5))
colors = ["#CB181D" if f > 50 else "#2171B5" if f > 20 else "#6BAED6"
          for f in sig["ContactFreq_pct"]]
ax.bar(range(len(sig)), sig["ContactFreq_pct"], color=colors)
ax.set_xticks(range(len(sig)))
ax.set_xticklabels([f"{r}\n{n}" for r, n in zip(sig["Residue"], sig["Resname"])],
                   fontsize=8, rotation=45, ha="right")
ax.set_ylabel("Contact frequency (%)", fontsize=13, fontweight="bold")
ax.set_xlabel("Residue", fontsize=13, fontweight="bold")
ax.set_title("Enzyme residues contacting poly-LYS surface (>5% frames, cutoff 4.5 Å)",
             fontsize=11, fontweight="bold")
ax.axhline(50, color="grey", ls="--", lw=0.8, alpha=0.6)
ax.axhline(20, color="grey", ls=":", lw=0.8, alpha=0.6)
ax.set_ylim(0, 105)
fig.tight_layout()
png_out = f"{OUT}/polylys_contacts.png"
fig.savefig(png_out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {png_out}")
