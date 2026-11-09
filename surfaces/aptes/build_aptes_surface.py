#!/usr/bin/env python3
"""
Build APTES-functionalized silica surface for orient1 pH5.

Strategy:
- Take orient1_surface.mol2 (9072 atoms, SI/OB types, net charge 0)
- Select 20 top-surface OB atoms (those at highest Z, spaced evenly)
- Above each selected OB, build: APTES-Si(OH)2-(CH2)3-NH3+
- APTES-Si uses CHARMM-INTERFACE SI parameters
- OA (bridging O, APTES-Si to surface) and OH groups use OB/oh types
- Organic chain (C/N/H) uses GAFF2 types from butylammonium parameterization
- Net charge of APTES groups: +20 total (20 x NH3+, each +1)
- Output: orient1_aptes_surface.mol2 + orient1_aptes_surface.frcmod
"""

import math, re, os

TLEAP = "/Data/Anushka/enzyme/laccase/OUT_CU_lc/test/again/pH5/ChainA/tleap"
SRC_MOL2 = f"{TLEAP}/orient1_surface.mol2"
OUT_MOL2 = f"{TLEAP}/orient1_aptes_surface.mol2"
OUT_FRCMOD = f"{TLEAP}/orient1_aptes_surface.frcmod"

N_APTES = 20  # number of APTES groups (~2/nm² for 10.7 nm² surface)

# ── GAFF2 charges from butylammonium parameterization ────────────────────────
# C1(methyl cap) → replaced by Si, so we use C2-C4 charges for our C1-C3
# Adjusted to keep propylamine chain charge near +1
# From antechamber: C(adj to Si)=-0.105, CH2=-0.113, CH2(adj N)=+0.105, N=-0.844, 3xHN=+0.462
# For Si-C bond: C gets slightly more negative (like C adj to heteroatom)
C1_Q  = -0.108  # C bonded to Si (c3, like C2 in BAM)
H_C1  =  0.078  # hc on C1 (2H)
C2_Q  = -0.113  # middle CH2 (c3)
H_C2  =  0.088  # hc on C2 (2H)
C3_Q  =  0.105  # CH2 adjacent to N (c3)
H_C3  =  0.112  # hx on C3 (2H)
N_Q   = -0.844  # nz
HN_Q  =  0.462  # hn (3H)

# APTES-Si parameters (from CHARMM-INTERFACE Si, slightly adjusted for -4 coordination)
SI2_Q  =  0.718  # APTES Si atom (SI type) — adjusted so each APTES group = +1.000 exactly
OH_Q   = -0.550  # OH oxygen on APTES Si (use OB type, adjusted)
HO_Q   =  0.200  # H on OH of APTES Si

# Check charge per APTES group
aptes_charge = (SI2_Q
    + 2*(OH_Q + HO_Q)             # 2 x Si-OH
    + C1_Q + 2*H_C1               # C1H2
    + C2_Q + 2*H_C2               # C2H2
    + C3_Q + 2*H_C3               # C3H2
    + N_Q  + 3*HN_Q)              # NH3+
print(f"Charge per APTES group: {aptes_charge:+.6f}")

# ── Read source mol2 ──────────────────────────────────────────────────────────
with open(SRC_MOL2) as f:
    content = f.read()

sections = {}
current = None
for line in content.split('\n'):
    if line.startswith('@<TRIPOS>'):
        current = line.strip()
        sections[current] = []
    elif current:
        sections[current].append(line)

# Parse atoms
atom_lines = sections.get('@<TRIPOS>ATOM', [])
atoms = []
for line in atom_lines:
    if not line.strip(): continue
    p = line.split()
    atoms.append({
        'id': int(p[0]), 'name': p[1],
        'x': float(p[2]), 'y': float(p[3]), 'z': float(p[4]),
        'type': p[5], 'res_id': p[6], 'res_name': p[7], 'charge': float(p[8])
    })

# Parse bonds
bond_lines = sections.get('@<TRIPOS>BOND', [])
bonds = []
for line in bond_lines:
    if not line.strip(): continue
    p = line.split()
    bonds.append({'id': int(p[0]), 'a1': int(p[1]), 'a2': int(p[2]), 'type': p[3]})

# ── Find top-surface OB atoms ─────────────────────────────────────────────────
ob_atoms = [a for a in atoms if a['type'] == 'OB']
max_z = max(a['z'] for a in atoms)
# Top OB atoms: those within 3 Å of max Z
top_ob = sorted([a for a in ob_atoms if a['z'] > max_z - 3.0],
                key=lambda a: a['z'], reverse=True)

print(f"Total OB atoms: {len(ob_atoms)}")
print(f"Top OB atoms (z > {max_z-3.0:.1f}): {len(top_ob)}")

# Select N_APTES evenly spaced OB atoms across the surface XY plane
# Grid-based selection to ensure even coverage
x_vals = sorted(set(round(a['x'], 0) for a in top_ob))
y_vals = sorted(set(round(a['y'], 0) for a in top_ob))

# Simple approach: sort by (x,y) and pick evenly spaced
top_ob_sorted = sorted(top_ob, key=lambda a: (round(a['x']/10), round(a['y']/10)))

# Pick every k-th atom
step = max(1, len(top_ob_sorted) // N_APTES)
selected_ob = top_ob_sorted[::step][:N_APTES]

# Ensure minimum spacing of 8 Å between selected sites
final_selected = []
for a in selected_ob:
    too_close = False
    for b in final_selected:
        dist = math.sqrt((a['x']-b['x'])**2 + (a['y']-b['y'])**2)
        if dist < 8.0:
            too_close = True; break
    if not too_close:
        final_selected.append(a)
    if len(final_selected) >= N_APTES:
        break

# If we didn't get enough, relax spacing
if len(final_selected) < N_APTES:
    for a in top_ob_sorted:
        if a in final_selected: continue
        too_close = any(math.sqrt((a['x']-b['x'])**2+(a['y']-b['y'])**2) < 5.0
                        for b in final_selected)
        if not too_close:
            final_selected.append(a)
        if len(final_selected) >= N_APTES:
            break

print(f"Selected {len(final_selected)} APTES sites")
print(f"Net charge from APTES: {len(final_selected) * aptes_charge:+.4f}")

# ── Build new atoms and bonds ─────────────────────────────────────────────────
new_atoms = list(atoms)  # start with existing surface atoms
new_bonds = list(bonds)
next_atom_id = max(a['id'] for a in atoms) + 1
next_bond_id = max(b['id'] for b in bonds) + 1
next_res_id  = max(int(a['res_id']) for a in atoms) + 1

aptes_res_id = next_res_id

for site_ob in final_selected:
    # APTES geometry: build chain pointing straight up (+Z)
    ox, oy, oz = site_ob['x'], site_ob['y'], site_ob['z']

    # APTES-Si: 1.68 Å above the surface OB (Si-O bond length)
    si_x, si_y, si_z = ox, oy, oz + 1.68

    # Two OH groups on APTES-Si (tetrahedral, pointing out ~109.5°)
    oh1_x, oh1_y, oh1_z = si_x + 1.40, si_y,       si_z + 0.20
    h1_x,  h1_y,  h1_z  = si_x + 2.35, si_y,       si_z + 0.20
    oh2_x, oh2_y, oh2_z = si_x - 0.70, si_y + 1.21, si_z + 0.20
    h2_x,  h2_y,  h2_z  = si_x - 0.70, si_y + 2.16, si_z + 0.20

    # Propylamine chain: Si-C1-C2-C3-N pointing up
    c1_x, c1_y, c1_z = si_x,       si_y,       si_z + 1.90
    c1h1x, c1h1y, c1h1z = c1_x+1.03, c1_y,     c1_z
    c1h2x, c1h2y, c1h2z = c1_x-0.51, c1_y+0.89, c1_z

    c2_x, c2_y, c2_z = si_x,       si_y,       si_z + 3.42
    c2h1x, c2h1y, c2h1z = c2_x+1.03, c2_y,     c2_z
    c2h2x, c2h2y, c2h2z = c2_x-0.51, c2_y+0.89, c2_z

    c3_x, c3_y, c3_z = si_x,       si_y,       si_z + 4.94
    c3h1x, c3h1y, c3h1z = c3_x+1.03, c3_y,     c3_z
    c3h2x, c3h2y, c3h2z = c3_x-0.51, c3_y+0.89, c3_z

    n_x, n_y, n_z = si_x,       si_y,       si_z + 6.41
    nh1x, nh1y, nh1z = n_x+1.02, n_y,       n_z
    nh2x, nh2y, nh2z = n_x-0.51, n_y+0.88,  n_z
    nh3x, nh3y, nh3z = n_x-0.51, n_y-0.88,  n_z

    res = str(aptes_res_id)
    rn  = "APT"

    new_atom_list = [
        # id  name    x       y       z       type  res  resname  charge
        (next_atom_id+0,  'SI',  si_x,  si_y,  si_z,  'SI',  res, rn,  SI2_Q),
        (next_atom_id+1,  'O1',  oh1_x, oh1_y, oh1_z, 'OB',  res, rn,  OH_Q),
        (next_atom_id+2,  'H1',  h1_x,  h1_y,  h1_z,  'ho',  res, rn,  HO_Q),
        (next_atom_id+3,  'O2',  oh2_x, oh2_y, oh2_z, 'OB',  res, rn,  OH_Q),
        (next_atom_id+4,  'H2',  h2_x,  h2_y,  h2_z,  'ho',  res, rn,  HO_Q),
        (next_atom_id+5,  'C1',  c1_x,  c1_y,  c1_z,  'c3',  res, rn,  C1_Q),
        (next_atom_id+6,  'HC1', c1h1x, c1h1y, c1h1z,'hc',   res, rn,  H_C1),
        (next_atom_id+7,  'HC2', c1h2x, c1h2y, c1h2z,'hc',   res, rn,  H_C1),
        (next_atom_id+8,  'C2',  c2_x,  c2_y,  c2_z,  'c3',  res, rn,  C2_Q),
        (next_atom_id+9,  'HC3', c2h1x, c2h1y, c2h1z,'hc',   res, rn,  H_C2),
        (next_atom_id+10, 'HC4', c2h2x, c2h2y, c2h2z,'hc',   res, rn,  H_C2),
        (next_atom_id+11, 'C3',  c3_x,  c3_y,  c3_z,  'c3',  res, rn,  C3_Q),
        (next_atom_id+12, 'HC5', c3h1x, c3h1y, c3h1z,'hx',   res, rn,  H_C3),
        (next_atom_id+13, 'HC6', c3h2x, c3h2y, c3h2z,'hx',   res, rn,  H_C3),
        (next_atom_id+14, 'N1',  n_x,   n_y,   n_z,   'nz',  res, rn,  N_Q),
        (next_atom_id+15, 'HN1', nh1x,  nh1y,  nh1z,  'hn',  res, rn,  HN_Q),
        (next_atom_id+16, 'HN2', nh2x,  nh2y,  nh2z,  'hn',  res, rn,  HN_Q),
        (next_atom_id+17, 'HN3', nh3x,  nh3y,  nh3z,  'hn',  res, rn,  HN_Q),
    ]

    for row in new_atom_list:
        new_atoms.append({
            'id': row[0], 'name': row[1],
            'x': row[2], 'y': row[3], 'z': row[4],
            'type': row[5], 'res_id': row[6], 'res_name': row[7], 'charge': row[8]
        })

    # Bonds: OB(surface) - SI, SI-O1, O1-H1, SI-O2, O2-H2, SI-C1, C1-C2, C2-C3, C3-N
    si_id = next_atom_id+0
    bond_defs = [
        (site_ob['id'], si_id),           # surface OB → APTES Si
        (si_id, next_atom_id+1),          # Si-O1
        (next_atom_id+1, next_atom_id+2), # O1-H1
        (si_id, next_atom_id+3),          # Si-O2
        (next_atom_id+3, next_atom_id+4), # O2-H2
        (si_id, next_atom_id+5),          # Si-C1
        (next_atom_id+5, next_atom_id+6), # C1-HC1
        (next_atom_id+5, next_atom_id+7), # C1-HC2
        (next_atom_id+5, next_atom_id+8), # C1-C2
        (next_atom_id+8, next_atom_id+9), # C2-HC3
        (next_atom_id+8, next_atom_id+10),# C2-HC4
        (next_atom_id+8, next_atom_id+11),# C2-C3
        (next_atom_id+11, next_atom_id+12),# C3-HC5
        (next_atom_id+11, next_atom_id+13),# C3-HC6
        (next_atom_id+11, next_atom_id+14),# C3-N
        (next_atom_id+14, next_atom_id+15),# N-HN1
        (next_atom_id+14, next_atom_id+16),# N-HN2
        (next_atom_id+14, next_atom_id+17),# N-HN3
    ]

    for a1, a2 in bond_defs:
        new_bonds.append({'id': next_bond_id, 'a1': a1, 'a2': a2, 'type': '1'})
        next_bond_id += 1

    next_atom_id += 18
    aptes_res_id += 1

# ── Verify total charge ───────────────────────────────────────────────────────
total_charge = sum(a['charge'] for a in new_atoms)
print(f"\nTotal atoms: {len(new_atoms)}")
print(f"Total bonds: {len(new_bonds)}")
print(f"Net charge of full surface: {total_charge:+.4f}")

# ── Write mol2 ────────────────────────────────────────────────────────────────
# Count residues
res_ids = sorted(set(int(a['res_id']) for a in new_atoms))
n_residues = len(res_ids)

with open(OUT_MOL2, 'w') as f:
    f.write("@<TRIPOS>MOLECULE\n")
    f.write("orient1_aptes_surface\n")
    f.write(f"  {len(new_atoms)}  {len(new_bonds)}  {n_residues}  0  0\n")
    f.write("SMALL\n")
    f.write("USER_CHARGES\n\n\n")

    f.write("@<TRIPOS>ATOM\n")
    for a in new_atoms:
        f.write(f"{a['id']:>7} {a['name']:<8} {a['x']:>10.4f} {a['y']:>10.4f} {a['z']:>10.4f} "
                f"{a['type']:<8} {a['res_id']:<4} {a['res_name']:<8} {a['charge']:>10.6f}\n")

    f.write("@<TRIPOS>BOND\n")
    for b in new_bonds:
        f.write(f"{b['id']:>6} {b['a1']:>6} {b['a2']:>6} {b['type']}\n")

    f.write("@<TRIPOS>SUBSTRUCTURE\n")
    for rid in res_ids:
        res_atoms = [a for a in new_atoms if int(a['res_id']) == rid]
        f.write(f"{rid:>6} {res_atoms[0]['res_name']:<8} {res_atoms[0]['id']:>6} TEMP  0 ****  ****  0 ROOT\n")

print(f"\nWrote: {OUT_MOL2}")

# ── Write frcmod ──────────────────────────────────────────────────────────────
# Combines CHARMM-INTERFACE SI/OB with GAFF2 organic chain params
frcmod_content = """INTERFACE FF + GAFF2 parameters for APTES-functionalized cristobalite silica
MASS
SI  28.086
OB  15.999
c3  12.011
hc   1.008
hx   1.008
ho   1.008
nz  14.007
hn   1.008

BOND
SI-OB   285.000   1.680
SI-c3   160.000   1.870
OB-ho   367.600   0.960
c3-c3   310.000   1.526
c3-nz   367.000   1.471
c3-hc   337.000   1.090
c3-hx   337.000   1.090
nz-hn   434.000   1.013

ANGLE
OB-SI-OB    100.000  109.500
SI-OB-SI    100.000  149.000
SI-OB-ho    60.000   115.000
c3-SI-OB    60.000   109.500
OB-c3-hc    50.000   109.500
OB-c3-c3    63.000   111.100
c3-c3-c3    40.000   111.100
c3-c3-hc    50.000   109.500
c3-c3-hx    50.000   109.500
c3-c3-nz    67.700   111.200
c3-nz-hn    50.000   109.500
hn-nz-hn    43.600   107.700
c3-c3-SI    60.000   111.100
hc-c3-hc    39.400   107.800
hx-c3-hx    39.400   107.800
hx-c3-nz    50.000   109.500
hc-c3-SI    50.000   109.500

DIHE
X-OB-SI-X    1     0.00000000    0.000   2.0
X-SI-c3-X    1     0.00000000    0.000   2.0
X-c3-c3-X    1     1.40000000    0.000   3.0
X-c3-nz-X    1     0.00000000    0.000   2.0

IMPROPER

NONB
SI    2.07500000  -0.09300000
OB    1.73500000  -0.05400000
c3    1.90800000  -0.10940000
hc    1.48700000  -0.01570000
hx    1.35900000  -0.01570000
ho    0.30000000  -0.00030000
nz    1.82400000  -0.17000000
hn    0.60000000  -0.01570000

"""

with open(OUT_FRCMOD, 'w') as f:
    f.write(frcmod_content)
print(f"Wrote: {OUT_FRCMOD}")
