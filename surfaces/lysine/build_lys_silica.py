#!/usr/bin/env python3
"""
Setup 2: Silica surface (original Si/O partial charges) + lysine monolayer + laccase.
Uses orient1_surface_original.mol2 (no APTES groups) with full Si(+1.1)/O(-0.55)
partial charges. Lysine 11x11 grid (9 Å spacing) covers the surface. No ligands.
EM + 500 ns production MDP generated.
"""

import os, subprocess, shutil, re
import numpy as np
import parmed as pmd

TLEAP_DIR  = "/Data/Anushka/enzyme/laccase/OUT_CU_lc/test/again/pH5/ChainA/tleap"
OUT_DIR    = "/Data/Anushka/Project4/ph5/orients/orient1/tleap/lys_silica_surface"
GMX        = "/apps/codes/gromacs-2023/bin/gmx_mpi"
TLEAP_BIN  = "/apps/codes/anaconda3/bin/tleap"

SURF_MOL2_ORIG = os.path.join(TLEAP_DIR, "orient1_surface_original.mol2")
SURF_MOL2  = os.path.join(OUT_DIR, "silica_surface_SUR.mol2")   # residues renamed to SUR
SURF_FRCM  = os.path.join(TLEAP_DIR, "orient1_surface_original.frcmod")
ENZY_PDB   = os.path.join(TLEAP_DIR, "orient1_combined.pdb")

PRMTOP     = os.path.join(OUT_DIR, "lys_silica.prmtop")
INPCRD     = os.path.join(OUT_DIR, "lys_silica.inpcrd")
GRO        = os.path.join(OUT_DIR, "lys_silica.gro")
TOP        = os.path.join(OUT_DIR, "lys_silica.top")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "MD"), exist_ok=True)


def rename_sur(text):
    """Residues already named SUR from mol2; kept as safety net for any R-2 remnants."""
    out = []
    for line in text.split('\n'):
        if len(line) > 10 and line[5:8] == 'R-2':
            line = line[:5] + 'SUR' + line[8:]
        out.append(line)
    return '\n'.join(out)


def count_mol2_atoms(mol2_path):
    n = 0
    in_atom = False
    with open(mol2_path) as f:
        for line in f:
            if '@<TRIPOS>ATOM' in line: in_atom = True; continue
            if '@<TRIPOS>' in line and in_atom: break
            if in_atom and line.strip() and line.split()[0].isdigit(): n += 1
    return n


def rename_surface_lys_to_sly(gro_path, *top_paths):
    """Rename surface LYS molecule type to SLY in GRO and topology files.
    Surface LYS molecules are the standalone ones (not part of enzyme system1).
    Atom range is determined from the topology [ molecules ] section."""

    # Read first topology to determine LYS atom range
    with open(top_paths[0]) as f:
        top_txt = f.read()

    # Parse atom count per molecule type from [ atoms ] sections
    mol_atom_counts = {}
    cur_mol = None; in_atoms_sec = False; max_i = 0
    for line in top_txt.split('\n'):
        s = line.strip()
        if '[ moleculetype ]' in line:
            if cur_mol and max_i: mol_atom_counts[cur_mol] = max_i
            cur_mol = None; in_atoms_sec = False; max_i = 0
        elif cur_mol is None and s and not s.startswith(';'):
            cur_mol = s.split()[0]
        elif cur_mol and '[ atoms ]' in line:
            in_atoms_sec = True
        elif in_atoms_sec and s.startswith('[') and 'atoms' not in s:
            in_atoms_sec = False
        elif in_atoms_sec and re.match(r'^\d+\s+\w', s):
            max_i = int(s.split()[0])
    if cur_mol and max_i:
        mol_atom_counts[cur_mol] = max_i

    # Parse [ molecules ] section for molecule order and counts
    mol_order = []
    in_mols = False
    for line in top_txt.split('\n'):
        s = line.strip()
        if '[ molecules ]' in line: in_mols = True; continue
        if in_mols and s.startswith('[') and 'molecules' not in s: in_mols = False
        if in_mols and s and not s.startswith(';'):
            parts = s.split()
            if len(parts) == 2 and parts[1].isdigit():
                mol_order.append((parts[0], int(parts[1])))

    # Find atom range of surface LYS
    offset = 0
    lys_start = lys_end = None
    for mol_name, mol_count in mol_order:
        n = mol_atom_counts.get(mol_name, 0)
        if mol_name == 'LYS':
            lys_start = offset + 1
            lys_end   = offset + mol_count * n
            break
        offset += mol_count * n

    if lys_start is None:
        print("  WARNING: standalone LYS not found in topology — skipping SLY rename"); return

    # Rename in GRO: residues in [lys_start, lys_end] from LYS to SLY
    with open(gro_path) as f:
        gro_lines = f.readlines()
    out = []
    for i, line in enumerate(gro_lines):
        # Line i=0: title, i=1: natoms, i=2..n-1: atoms (1-based), last: box
        if 2 <= i < len(gro_lines) - 1:
            atom_i = i - 1  # 1-based atom index
            if lys_start <= atom_i <= lys_end and len(line) > 10 and line[5:8] == 'LYS':
                line = line[:5] + 'SLY' + line[8:]
        out.append(line)
    with open(gro_path, 'w') as f: f.writelines(out)

    # Rename in each topology: moleculetype name + [ atoms ] residue column + [ molecules ]
    for top_path in top_paths:
        with open(top_path) as f: lines = top_txt.splitlines(keepends=True) if top_path == top_paths[0] else f.readlines()
        # Re-read file if not first
        if top_path != top_paths[0]:
            with open(top_path) as f: lines = f.readlines()
        else:
            lines = top_txt.splitlines(keepends=True)

        in_lys_mol = False; past_name = False; in_lys_atoms = False
        in_mols_sec = False
        for i, line in enumerate(lines):
            s = line.strip()
            # Track moleculetype sections
            if '[ moleculetype ]' in line:
                in_lys_mol = False; past_name = False; in_lys_atoms = False
            elif not past_name and s and not s.startswith(';'):
                past_name = True
                if s.split()[0] == 'LYS':
                    in_lys_mol = True
                    lines[i] = re.sub(r'\bLYS\b', 'SLY', line, count=1)
            # Track [ atoms ] within LYS moleculetype
            if in_lys_mol and '[ atoms ]' in line:
                in_lys_atoms = True
            elif in_lys_atoms and s.startswith('[') and 'atoms' not in s:
                in_lys_atoms = False
            elif in_lys_atoms and re.match(r'^\s*\d+\s+\w', line):
                # Column 4 (0-indexed) in [ atoms ] is residue name
                parts = line.split()
                if len(parts) > 3 and parts[3] == 'LYS':
                    parts[3] = 'SLY'
                    lines[i] = '  '.join(parts) + '\n'
            # Track [ molecules ] section
            if '[ molecules ]' in line: in_mols_sec = True
            elif in_mols_sec and s.startswith('[') and 'molecules' not in s: in_mols_sec = False
            elif in_mols_sec and re.match(r'^\s*LYS\b', line):
                lines[i] = re.sub(r'\bLYS\b', 'SLY', line, count=1)

        with open(top_path, 'w') as f: f.writelines(lines)

    print(f"  Surface LYS → SLY in GRO (atoms {lys_start}–{lys_end}) and {len(top_paths)} topology file(s)")


def rename_mol2_residues(src, dst, old='SUR-2', new='SUR'):
    """Replace residue/substructure name in mol2 ATOM and SUBSTRUCTURE sections."""
    with open(src) as f:
        txt = f.read()
    # Replace in ATOM section subst_name column (whole-word match to avoid partial hits)
    txt = txt.replace(old, new)
    with open(dst, 'w') as f:
        f.write(txt)


# ── STEP 0: Create mol2 with surface residues named 'SUR' ────────────────────
print("=== STEP 0: Rename surface residues SUR-2 → SUR in mol2 ===")
rename_mol2_residues(SURF_MOL2_ORIG, SURF_MOL2)
n_surf_atoms = count_mol2_atoms(SURF_MOL2)
print(f"  silica_surface_SUR.mol2: {n_surf_atoms} atoms, residues named SUR, original charges")


# ── STEP 1: Generate single LYS mol2 via tleap ───────────────────────────────
print("\n=== STEP 1: Generate single LYS mol2 ===")
lys_single_mol2 = os.path.join(OUT_DIR, "_lys_single.mol2")
lys_gen_in      = os.path.join(OUT_DIR, "_tleap_gen_lys.in")
with open(lys_gen_in, 'w') as f:
    f.write(f"""source leaprc.protein.ff19SB
lys = sequence {{ LYS }}
savemol2 lys {lys_single_mol2} 0
quit
""")
result = subprocess.run([TLEAP_BIN, '-f', lys_gen_in], capture_output=True, text=True, cwd=OUT_DIR)
if not os.path.exists(lys_single_mol2) or os.path.getsize(lys_single_mol2) == 0:
    print("  ERROR: tleap failed"); print(result.stdout[-500:]); exit(1)
print(f"  OK: {lys_single_mol2}")


# ── STEP 2: Parse LYS mol2 and orient CA→NZ along +Z ────────────────────────
print("\n=== STEP 2: Orient LYS (CA at origin, NZ pointing +Z) ===")
# Correct AMBER ff19SB atom types for LYS (savemol2 outputs SYBYL types)
LYS_TYPES = {'N':'N','H':'H','CA':'XC','HA':'H1','CB':'C8','HB2':'HC','HB3':'HC',
             'CG':'C8','HG2':'HC','HG3':'HC','CD':'C8','HD2':'HC','HD3':'HC',
             'CE':'C8','HE2':'HP','HE3':'HP','NZ':'N3','HZ1':'H','HZ2':'H',
             'HZ3':'H','C':'C','O':'O'}
lys_atoms, lys_bonds = [], []
in_atom = in_bond = False
with open(lys_single_mol2) as f:
    for line in f:
        if '@<TRIPOS>ATOM' in line:  in_atom = True;  in_bond = False; continue
        if '@<TRIPOS>BOND' in line:  in_bond = True;  in_atom = False; continue
        if '@<TRIPOS>' in line:      in_atom = in_bond = False
        if in_atom and line.strip():
            p = line.split()
            name = p[1]
            lys_atoms.append({'id': int(p[0]), 'name': name,
                              'x': float(p[2]), 'y': float(p[3]), 'z': float(p[4]),
                              'type': LYS_TYPES.get(name, p[5]),
                              'charge': float(p[8]) if len(p) > 8 else 0.0})
        if in_bond and line.strip():
            p = line.split()
            lys_bonds.append({'a1': int(p[1]), 'a2': int(p[2]), 'btype': p[3]})

ca = next(a for a in lys_atoms if a['name'] == 'CA')
nz = next(a for a in lys_atoms if a['name'] == 'NZ')
ca_pos = np.array([ca['x'], ca['y'], ca['z']])
nz_pos = np.array([nz['x'], nz['y'], nz['z']])
v = (nz_pos - ca_pos) / np.linalg.norm(nz_pos - ca_pos)
z_axis = np.array([0., 0., 1.])
axis = np.cross(v, z_axis)
sin_a = np.linalg.norm(axis)
cos_a = float(np.dot(v, z_axis))
if sin_a < 1e-6:
    R = np.eye(3) if cos_a > 0 else np.diag([1., -1., -1.])
else:
    axis /= sin_a
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)
for a in lys_atoms:
    pos = R @ (np.array([a['x'], a['y'], a['z']]) - ca_pos)
    a['x'], a['y'], a['z'] = float(pos[0]), float(pos[1]), float(pos[2])
nz_z = next(a for a in lys_atoms if a['name'] == 'NZ')['z']
print(f"  {len(lys_atoms)} LYS atoms, {len(lys_bonds)} bonds: CA at z=0, NZ at z={nz_z:.2f} Å")


# ── STEP 3: Build lysine monolayer mol2 (11×11 grid at 9 Å spacing) ──────────
print("\n=== STEP 3: Build lysine monolayer mol2 ===")
CA_Z    = 10.0
GRID_XS = np.arange(-45, 46, 9)
GRID_YS = np.arange(-45, 46, 9)
n_lys   = len(GRID_XS) * len(GRID_YS)
n_atm   = len(lys_atoms)
n_bnd   = len(lys_bonds)
mono_mol2 = os.path.join(OUT_DIR, "lysine_monolayer.mol2")
with open(mono_mol2, 'w') as f:
    f.write('@<TRIPOS>MOLECULE\nlysine_monolayer\n')
    f.write(f'{n_lys*n_atm:6d} {n_lys*n_bnd:6d} {n_lys:6d}     0     0\n')
    f.write('SMALL\nUSER_CHARGES\n\n')
    f.write('@<TRIPOS>ATOM\n')
    aid, rid = 1, 1
    for xi in GRID_XS:
        for yi in GRID_YS:
            for a in lys_atoms:
                f.write(f'{aid:6d} {a["name"]:<4s} {a["x"]+xi:10.4f} {a["y"]+yi:10.4f} '
                        f'{a["z"]+CA_Z:10.4f} {a["type"]:<6s} {rid:4d} LYS  {a["charge"]:8.4f}\n')
                aid += 1
            rid += 1
    f.write('@<TRIPOS>BOND\n')
    bid = 1
    for ri in range(n_lys):
        off = ri * n_atm
        for b in lys_bonds:
            f.write(f'{bid:6d} {b["a1"]+off:6d} {b["a2"]+off:6d} {b["btype"]}\n')
            bid += 1
    f.write('@<TRIPOS>SUBSTRUCTURE\n')
    for ri in range(n_lys):
        root = ri * n_atm + 1
        f.write(f'{ri+1:6d} LYS  {root:6d} TEMP 0 **** **** 0\n')
nz_z = next(a for a in lys_atoms if a['name'] == 'NZ')['z']
print(f"  {n_lys} LYS residues ({n_atm} atoms each), CA at Z={CA_Z} Å, NZ at Z≈{CA_Z+nz_z:.1f} Å")
print(f"  Gap to enzyme bottom (Z≈46 Å): ≈{46-(CA_Z+nz_z):.1f} Å ({(46-(CA_Z+nz_z))/10:.1f} nm)")


# ── STEP 4: Write tleap script ────────────────────────────────────────────────
print("\n=== STEP 4: Write tleap script ===")
tleap_in = os.path.join(OUT_DIR, "tleap_lys_silica.in")
with open(tleap_in, 'w') as f:
    f.write(f"""source leaprc.protein.ff19SB
source leaprc.gaff2
source leaprc.water.tip3p

addAtomTypes {{
        {{ "M1"  "Cu" "sp3" }}
        {{ "M2"  "Cu" "sp3" }}
        {{ "M3"  "Cu" "sp3" }}
        {{ "M4"  "Cu" "sp3" }}
        {{ "Y1"  "N" "sp3" }}
        {{ "Z1"  "S" "sp3" }}
        {{ "Y3"  "N" "sp3" }}
        {{ "Y4"  "N" "sp3" }}
        {{ "Y5"  "N" "sp3" }}
        {{ "Y6"  "N" "sp3" }}
        {{ "Y7"  "N" "sp3" }}
        {{ "Y8"  "N" "sp3" }}
        {{ "Y9"  "N" "sp3" }}
        {{ "Y2"  "N" "sp3" }}
        {{ "Y10"  "N" "sp3" }}
        {{ "SI" "Si" "sp3" }}
        {{ "OB" "O"  "sp3" }}
}}

HD1 = loadmol2 {TLEAP_DIR}/HD1.mol2
HD2 = loadmol2 {TLEAP_DIR}/HD2.mol2
HD3 = loadmol2 {TLEAP_DIR}/HD3.mol2
HD4 = loadmol2 {TLEAP_DIR}/HD4.mol2
HD5 = loadmol2 {TLEAP_DIR}/HD5.mol2
HD6 = loadmol2 {TLEAP_DIR}/HD6.mol2
HD7 = loadmol2 {TLEAP_DIR}/HD7.mol2
HE1 = loadmol2 {TLEAP_DIR}/HE1.mol2
HE2 = loadmol2 {TLEAP_DIR}/HE2.mol2
HE3 = loadmol2 {TLEAP_DIR}/HE3.mol2
CM1 = loadmol2 {TLEAP_DIR}/CM1.mol2
CU1 = loadmol2 {TLEAP_DIR}/CU1_modified.mol2
CU2 = loadmol2 {TLEAP_DIR}/CU2_modified.mol2
CU3 = loadmol2 {TLEAP_DIR}/CU3_modified.mol2
CU4 = loadmol2 {TLEAP_DIR}/CU4_modified.mol2

loadamberparams frcmod.ions1lm_126_tip3p
loadamberparams {TLEAP_DIR}/CU1_mcpbpy.frcmod
loadamberparams {TLEAP_DIR}/CU2_mcpbpy.frcmod
loadamberparams {TLEAP_DIR}/CU3_mcpbpy.frcmod
loadamberparams {TLEAP_DIR}/CU4_mcpbpy.frcmod

loadamberparams {SURF_FRCM}
surf = loadmol2 {SURF_MOL2}

LYS_mono = loadmol2 {mono_mol2}

mol = loadpdb {ENZY_PDB}

bond mol.85.SG mol.488.SG
bond mol.117.SG mol.205.SG
bond mol.395.ND1 mol.500.CU
bond mol.453.SG mol.500.CU
bond mol.458.ND1 mol.500.CU
bond mol.394.C mol.395.N
bond mol.395.C mol.396.N
bond mol.452.C mol.453.N
bond mol.453.C mol.454.N
bond mol.457.C mol.458.N
bond mol.458.C mol.459.N
bond mol.111.NE2 mol.501.CU
bond mol.400.NE2 mol.501.CU
bond mol.452.NE2 mol.501.CU
bond mol.110.C mol.111.N
bond mol.111.C mol.112.N
bond mol.399.C mol.400.N
bond mol.400.C mol.401.N
bond mol.451.C mol.452.N
bond mol.66.ND1 mol.502.CU
bond mol.109.NE2 mol.502.CU
bond mol.454.NE2 mol.502.CU
bond mol.65.C mol.66.N
bond mol.66.C mol.67.N
bond mol.108.C mol.109.N
bond mol.109.C mol.110.N
bond mol.454.C mol.455.N
bond mol.64.NE2 mol.503.CU
bond mol.398.NE2 mol.503.CU
bond mol.63.C mol.64.N
bond mol.64.C mol.65.N
bond mol.397.C mol.398.N
bond mol.398.C mol.399.N

COM = combine {{mol LYS_mono surf}}
solvatebox COM TIP3PBOX 10.0
addions COM Cl- 0
addions COM Na+ 0
saveamberparm COM {PRMTOP} {INPCRD}
quit
""")
print(f"  Wrote: {tleap_in}")


# ── STEP 5: Run tleap ─────────────────────────────────────────────────────────
print("\n=== STEP 5: Run tleap ===")
tleap_out = tleap_in.replace('.in', '.out')
result = subprocess.run([TLEAP_BIN, "-f", tleap_in], capture_output=True, text=True, cwd=OUT_DIR)
with open(tleap_out, 'w') as f:
    f.write(result.stdout + (result.stderr or ''))
if not os.path.exists(PRMTOP) or os.path.getsize(PRMTOP) == 0:
    print("  ERROR: tleap failed")
    errs = [l for l in result.stdout.split('\n') if 'error' in l.lower() or 'FATAL' in l]
    for e in errs[:10]: print(f"    {e}")
    print(f"  See: {tleap_out}"); exit(1)
print(f"  OK: prmtop ({os.path.getsize(PRMTOP)//1024} KB)")
for line in result.stdout.split('\n'):
    if 'Added' in line and 'ions' in line.lower():
        print(f"    {line.strip()}")


# ── STEP 6: parmed → GROMACS ──────────────────────────────────────────────────
print("\n=== STEP 6: parmed → GROMACS ===")
amber = pmd.load_file(PRMTOP, INPCRD)
amber.save(GRO, format='gro', overwrite=True)
amber.save(TOP, format='gromacs', overwrite=True)
for path in [GRO, TOP]:
    with open(path) as f: txt = f.read()
    with open(path, 'w') as f: f.write(rename_sur(txt))
with open(GRO) as f: lines_gro = f.readlines()
n_atoms = int(lines_gro[1].strip())
print(f"  Total atoms: {n_atoms},  Box: {lines_gro[-1].strip()}")


# Rename surface LYS → SLY in GRO and topology
print("\n=== STEP 7: Rename surface LYS → SLY ===")
rename_surface_lys_to_sly(GRO, TOP)


# ── STEP 8: Patch POSRES_SUR (before Cl- → in surface section) ───────────────
print("\n=== STEP 8: Patch POSRES_SUR ===")
INSERT = '\n#ifdef POSRES_SUR\n#include "posres_SUR.itp"\n#endif\n'

def patch_posres(top_path):
    with open(top_path) as f: txt = f.read()
    for tgt in ['[ moleculetype ]\n; Name            nrexcl\nCl-',
                '[ moleculetype ]\n; Name            nrexcl\nNa+',
                '[ moleculetype ]\n; Name            nrexcl\nWAT']:
        if tgt in txt:
            txt = txt.replace(tgt, INSERT + tgt, 1); break
    with open(top_path, 'w') as f: f.write(txt)

patch_posres(TOP)
print("  POSRES_SUR inserted before Cl-")


# ── STEP 9: Generate posres_SUR.itp ──────────────────────────────────────────
print("\n=== STEP 9: posres_SUR.itp ===")
posres_path = os.path.join(OUT_DIR, "posres_SUR.itp")
with open(posres_path, 'w') as f:
    f.write('; position restraints for SUR (plain silica surface)\n\n'
            '[ position_restraints ]\n;  i funct       fcx        fcy        fcz\n')
    for i in range(1, n_surf_atoms + 1):
        f.write(f'{i}  1  1000  1000  1000\n')
print(f"  posres_SUR.itp: {n_surf_atoms} surface atoms")


# ── STEP 10: Write MDP files ──────────────────────────────────────────────────
print("\n=== STEP 10: Write MDP files ===")

minim_mdp_content = """; Energy minimization with POSRES_SUR on surface
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = 50000
constraints     = none
nstlist         = 1
cutoff-scheme   = Verlet
ns_type         = grid
coulombtype     = PME
rcoulomb        = 1.0
rvdw            = 1.0
pbc             = xyz
define          = -DPOSRES_SUR
refcoord-scaling = com
"""

prod_mdp_content = """; 500 ns production MD — dt=2 fs, POSRES_SUR (surface fixed, enzyme free to adsorb)
integrator              = md
nsteps                  = 250000000
dt                      = 0.002

nstxout                 = 30000
nstvout                 = 30000
nstfout                 = 30000
nstenergy               = 30000
nstlog                  = 30000
nstxout-compressed      = 30000
compressed-x-grps       = System

continuation            = no
constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 1
lincs_order             = 4

cutoff-scheme           = Verlet
ns_type                 = grid
nstlist                 = 10
rcoulomb                = 1.0
rvdw                    = 1.0
coulombtype             = PME
pme_order               = 4
fourierspacing          = 0.16

define                  = -DPOSRES_SUR
refcoord-scaling        = com

tcoupl                  = V-rescale
tc-grps                 = non-Water Water
tau_t                   = 0.1     0.1
ref_t                   = 300     300

pcoupl                  = C-rescale
pcoupltype              = isotropic
tau_p                   = 2.0
ref_p                   = 1.0
compressibility         = 4.5e-5

pbc                     = xyz

gen_vel                 = yes
gen_temp                = 300
gen_seed                = 2222
"""

for fname, content in [("minim.mdp", minim_mdp_content), ("500ns.mdp", prod_mdp_content)]:
    with open(os.path.join(OUT_DIR, fname), 'w') as f: f.write(content)
print("  Written: minim.mdp, 500ns.mdp")


# ── STEP 11: Copy to MD dir, grompp + EM ─────────────────────────────────────
print("\n=== STEP 11: grompp + EM ===")
md_dir = os.path.join(OUT_DIR, "MD")
for fname in ["lys_silica.gro", "lys_silica.top"]:
    shutil.copy(os.path.join(OUT_DIR, fname), md_dir)
shutil.copy(posres_path, md_dir)
for fname in ["minim.mdp", "500ns.mdp"]:
    shutil.copy(os.path.join(OUT_DIR, fname), md_dir)

grompp_cmd = [GMX, "grompp", "-f", "minim.mdp", "-c", "lys_silica.gro",
              "-r", "lys_silica.gro",
              "-p", "lys_silica.top", "-o", "em.tpr", "-maxwarn", "5"]
result = subprocess.run(grompp_cmd, capture_output=True, text=True, cwd=md_dir)
with open(os.path.join(md_dir, "grompp_em.log"), 'w') as f: f.write(result.stdout + result.stderr)
if not os.path.exists(os.path.join(md_dir, "em.tpr")):
    print("  ERROR: grompp failed")
    for l in (result.stdout + result.stderr).split('\n'):
        if 'error' in l.lower() or 'ERROR' in l: print(f"    {l}")
    print(f"  See: {md_dir}/grompp_em.log"); exit(1)
print("  grompp OK → em.tpr")

mdrun_cmd = ["mpirun", "-np", "4", GMX, "mdrun", "-v", "-s", "em.tpr",
             "-deffnm", "em", "-pin", "on"]
print("  Running EM (4 ranks)...")
result = subprocess.run(mdrun_cmd, capture_output=True, text=True, cwd=md_dir)
with open(os.path.join(md_dir, "em_run.log"), 'w') as f: f.write(result.stdout + result.stderr)

em_gro = os.path.join(md_dir, "em.gro")
if os.path.exists(em_gro) and os.path.getsize(em_gro) > 0:
    log_txt = open(os.path.join(md_dir, "em.log")).read() if os.path.exists(os.path.join(md_dir,"em.log")) else result.stdout
    for line in log_txt.split('\n'):
        if 'converged' in line.lower() or 'Potential Energy' in line:
            print(f"  {line.strip()}")
    print(f"  EM DONE → {em_gro}")
else:
    print("  WARNING: EM may not have completed — check MD/em.log")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"""
{'='*65}
SETUP 2 — SILICA SURFACE + LYSINE MONOLAYER + LACCASE
  Directory: {OUT_DIR}
  Surface:   Plain silica (Si/O charges: +1.1/-0.55, {n_surf_atoms} atoms)
  SLY:       {n_lys} residues (11x11, 9 Å spacing, Cα Z={CA_Z} Å)
  Total atoms: {n_atoms}

Production run (from {md_dir}):
  {GMX} grompp -f 500ns.mdp -c em.gro -p lys_silica.top -o prod.tpr -maxwarn 5
  mpirun -np 4 {GMX} mdrun -v -s prod.tpr -deffnm prod -pin on
{'='*65}
""")
