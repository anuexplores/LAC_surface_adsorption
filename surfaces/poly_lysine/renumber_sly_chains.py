#!/usr/bin/env python3
"""
Renumber SLY so each chain gets one resid (1-49).
All 176 atoms of chain 1 → resid 1, chain 2 → resid 2, etc.
Chain boundary detected when old resid resets from 8 → 1.
LYS (protein) resids untouched.
"""
from pathlib import Path

GRO = Path("/Data/Anushka/Project4/anushka@10.0.66.94/P4/orient1/ph5/lysine/polyLysine/MD/100ns.gro")

with open(GRO) as f:
    lines = f.readlines()

out = [lines[0], lines[1]]

chain_id   = 0
prev_resid = None

for line in lines[2:-1]:
    if len(line) < 15:
        out.append(line); continue

    resname = line[5:8]

    if resname == 'SLY':
        old_resid = int(line[0:5])
        # new chain when resid resets (8 → 1) or very first SLY atom
        if prev_resid is None or (old_resid == 1 and prev_resid != 1):
            chain_id += 1
        prev_resid = old_resid
        line = f"{chain_id % 100000:5d}" + line[5:]

    out.append(line)

out.append(lines[-1])

out_path = GRO.parent / "100ns_chainid.gro"
with open(out_path, 'w') as f:
    f.writelines(out)

# Verify
sly_resids = set()
for l in out[2:-1]:
    if len(l) > 8 and l[5:8] == 'SLY':
        sly_resids.add(int(l[0:5]))
print(f"Written: {out_path}")
print(f"SLY chain resids: {sorted(sly_resids)}")
print(f"Total chains: {len(sly_resids)}")
