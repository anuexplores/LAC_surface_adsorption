#!/usr/bin/env python3
"""
Build lysine-monolayer + laccase adsorption system.
APTES surface covered with dense LYS monolayer (11x11 grid, 9 Å spacing).
Laccase enzyme placed ~2 nm above the lysine layer. No ligands.
Runs EM and generates MDP files for NVT/NPT/500 ns production.
"""

import os, subprocess, shutil, re
import numpy as np
import parmed as pmd

TLEAP_DIR = "/Data/Anushka/enzyme/laccase/OUT_CU_lc/test/again/pH5/ChainA/tleap"
OUT_DIR   = "/Data/Anushka/Project4/ph5/orients/orient1/tleap/lys_surface"
GMX       = "/apps/codes/gromacs-2023/bin/gmx_mpi"
TLEAP_BIN = "/apps/codes/anaconda3/bin/tleap"

SURF_MOL2 = os.path.join(TLEAP_DIR, "orient1_aptes_surface_original.mol2")
SURF_FRCM = os.path.join(TLEAP_DIR, "orient1_aptes_surface_original.frcmod")
ENZY_PDB  = os.path.join(TLEAP_DIR, "orient1_combined.pdb")

PRMTOP    = os.path.join(OUT_DIR, "lys_surface.prmtop")
INPCRD    = os.path.join(OUT_DIR, "lys_surface.inpcrd")
GRO       = os.path.join(OUT_DIR, "lys_surface.gro")
TOP       = os.path.join(OUT_DIR, "lys_surface.top")
TOP_HMR   = os.path.join(OUT_DIR, "lys_surface_hmr.top")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "MD"), exist_ok=True)


def rename_sur(text):
    out = []
    for line in text.split('\n'):
        if len(line) > 10 and line[5:8] == 'R-2':
            line = line[:5] + 'SUR' + line[8:]
        out.append(line)
    return '\n'.join(out)


# ── STEP 1: Generate single LYS residue PDB via tleap ────────────────────────
print("=== STEP 1: Generate single LYS PDB ===")
lys_single_pdb = os.path.join(OUT_DIR, "_lys_single.pdb")
lys_gen_in     = os.path.join(OUT_DIR, "_tleap_gen_lys.in")
with open(lys_gen_in, 'w') as f:
    f.write(f"""source leaprc.protein.ff19SB
lys = sequence {{ LYS }}
savepdb lys {lys_single_pdb}
quit
""")
result = subprocess.run([TLEAP_BIN, '-f', lys_gen_in], capture_output=True, text=True, cwd=OUT_DIR)
if not os.path.exists(lys_single_pdb) or os.path.getsize(lys_single_pdb) == 0:
    print("  ERROR: tleap failed to generate LYS PDB")
    print(result.stdout[-500:])
    exit(1)
print(f"  OK: {lys_single_pdb}")


# ── STEP 2: Parse LYS atoms and orient CA→NZ along +Z ────────────────────────
print("\n=== STEP 2: Orient LYS (NZ pointing +Z) ===")
raw_atoms = []
with open(lys_single_pdb) as f:
    for line in f:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            raw_atoms.append({
                'record':  line[:6],
                'name':    line[12:16].strip(),
                'resname': line[17:20].strip(),
                'x': float(line[30:38]),
                'y': float(line[38:46]),
                'z': float(line[46:54]),
            })

# Keep only LYS atoms (skip ACE/NME caps if generated)
lys_atoms = [a for a in raw_atoms if a['resname'] == 'LYS'] or raw_atoms

ca = next(a for a in lys_atoms if a['name'] == 'CA')
nz = next(a for a in lys_atoms if a['name'] == 'NZ')
ca_pos = np.array([ca['x'], ca['y'], ca['z']])
nz_pos = np.array([nz['x'], nz['y'], nz['z']])

# Rotation matrix: CA→NZ vector → +Z axis
v = (nz_pos - ca_pos)
v = v / np.linalg.norm(v)
z_axis = np.array([0., 0., 1.])
axis = np.cross(v, z_axis)
sin_a = np.linalg.norm(axis)
cos_a = float(np.dot(v, z_axis))
if sin_a < 1e-6:
    R = np.eye(3) if cos_a > 0 else np.diag([1., -1., -1.])
else:
    axis = axis / sin_a
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)

# Apply rotation, center at CA
for a in lys_atoms:
    pos = R @ (np.array([a['x'], a['y'], a['z']]) - ca_pos)
    a['x'], a['y'], a['z'] = float(pos[0]), float(pos[1]), float(pos[2])

nz_after = next(a for a in lys_atoms if a['name'] == 'NZ')
print(f"  LYS atoms: {len(lys_atoms)}")
print(f"  CA at z=0, NZ at z={nz_after['z']:.2f} Å (should be ~5-6 Å)")


# ── STEP 3: Build lysine monolayer PDB (11×11 grid) ───────────────────────────
print("\n=== STEP 3: Build lysine monolayer ===")
# APTES surface: X ~[-51.3, 52.0], Y ~[-51.95, 53.23], APT tips at Z=13.43 Å
# Place Cα at Z=17 Å, giving ~3.6 Å clearance above APTES tips
# NZ ends up at Z~22-23 Å; enzyme bottom at Z~46 Å → gap ~2.3 nm ✓
CA_Z     = 17.0
GRID_XS  = np.arange(-45, 46, 9)   # 11 points: -45,-36,...,36,45
GRID_YS  = np.arange(-45, 46, 9)

mono_pdb = os.path.join(OUT_DIR, "lysine_monolayer.pdb")
with open(mono_pdb, 'w') as f:
    atom_n = 1
    res_n  = 1
    for xi in GRID_XS:
        for yi in GRID_YS:
            for a in lys_atoms:
                x = a['x'] + xi
                y = a['y'] + yi
                z = a['z'] + CA_Z
                f.write(f"ATOM  {atom_n:5d}  {a['name']:<3s} LYS A{res_n:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
                atom_n += 1
            res_n += 1
    f.write("END\n")

n_lys = len(GRID_XS) * len(GRID_YS)
print(f"  {n_lys} LYS residues ({len(GRID_XS)}x{len(GRID_YS)} grid, 9 Å spacing)")
print(f"  XY: [{GRID_XS[0]:.0f},{GRID_XS[-1]:.0f}] x [{GRID_YS[0]:.0f},{GRID_YS[-1]:.0f}] Å")
print(f"  Cα at Z={CA_Z} Å, NZ at Z≈{CA_Z + nz_after['z']:.1f} Å")
print(f"  Enzyme bottom at Z≈46 Å → gap ≈{46 - (CA_Z + nz_after['z']):.1f} Å")


# ── STEP 4: Write tleap script (enzyme + LYS monolayer + APTES surface) ───────
print("\n=== STEP 4: Write tleap script ===")
tleap_in = os.path.join(OUT_DIR, "tleap_lys_surface.in")
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

LYS_mono = loadpdb {mono_pdb}

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
print("\n=== STEP 5: Run tleap (this may take a few minutes) ===")
tleap_out = os.path.join(OUT_DIR, "tleap_lys_surface.out")
result = subprocess.run([TLEAP_BIN, "-f", tleap_in], capture_output=True, text=True, cwd=OUT_DIR)
with open(tleap_out, 'w') as f:
    f.write(result.stdout + (result.stderr or ''))
if not os.path.exists(PRMTOP) or os.path.getsize(PRMTOP) == 0:
    print("  ERROR: tleap failed")
    errs = [l for l in result.stdout.split('\n') if 'error' in l.lower() or 'FATAL' in l]
    for e in errs[:10]: print(f"    {e}")
    print(f"  Full log: {tleap_out}")
    exit(1)
print(f"  OK: prmtop ({os.path.getsize(PRMTOP)//1024} KB)")
# Report charge/ions from tleap log
for line in result.stdout.split('\n'):
    if 'Added' in line or 'Total unperturbed charge' in line or 'ions to' in line.lower():
        print(f"    {line.strip()}")


# ── STEP 6: parmed → GROMACS ──────────────────────────────────────────────────
print("\n=== STEP 6: parmed → GROMACS ===")
amber = pmd.load_file(PRMTOP, INPCRD)
amber.save(GRO, format='gro', overwrite=True)
amber.save(TOP, format='gromacs', overwrite=True)
for path in [GRO, TOP]:
    with open(path) as f: txt = f.read()
    with open(path, 'w') as f: f.write(rename_sur(txt))
print(f"  GRO: {GRO}")
print(f"  TOP: {TOP}")

with open(GRO) as f: lines = f.readlines()
n_atoms = int(lines[1].strip())
print(f"  Total atoms: {n_atoms}")
print(f"  Box: {lines[-1].strip()}")


# ── STEP 7: Apply HMR (for production at 4 fs) ────────────────────────────────
print("\n=== STEP 7: Apply HMR topology ===")
amber_hmr = pmd.load_file(PRMTOP, INPCRD)
pmd.tools.HMassRepartition(amber_hmr).execute()
amber_hmr.save(TOP_HMR, format='gromacs', overwrite=True)
with open(TOP_HMR) as f: txt = f.read()
with open(TOP_HMR, 'w') as f: f.write(rename_sur(txt))
print(f"  HMR topology: {TOP_HMR}")


# ── STEP 8: Patch POSRES_SUR into both topologies ────────────────────────────
print("\n=== STEP 8: Patch POSRES_SUR ===")
INSERT = '\n#ifdef POSRES_SUR\n#include "posres_SUR.itp"\n#endif\n'

def patch_posres(top_path):
    with open(top_path) as f: txt = f.read()
    UNK_TARGET = '[ moleculetype ]\n; Name            nrexcl\nUNK'
    if UNK_TARGET in txt:
        txt = txt.replace(UNK_TARGET, INSERT + UNK_TARGET, 1)
    else:
        for tgt in ['[ moleculetype ]\n; Name            nrexcl\nNa+',
                    '[ moleculetype ]\n; Name            nrexcl\nWAT',
                    '[ moleculetype ]\n; Name            nrexcl\nCl-']:
            if tgt in txt:
                txt = txt.replace(tgt, INSERT + tgt, 1); break
    with open(top_path, 'w') as f: f.write(txt)

patch_posres(TOP)
patch_posres(TOP_HMR)
print("  POSRES_SUR block inserted in both topologies")


# ── STEP 9: Generate posres_SUR.itp ──────────────────────────────────────────
print("\n=== STEP 9: posres_SUR.itp ===")
n_sur = 0
in_s1 = False; in_s1_next = False; in_atoms_sec = False
with open(TOP) as f:
    for line in f:
        if '[ moleculetype ]' in line:
            in_atoms_sec = False; in_s1_next = True
        elif in_s1_next and line.strip().startswith('system1'):
            in_s1 = True; in_s1_next = False
        elif in_s1_next and line.strip() and not line.strip().startswith(';'):
            in_s1_next = False
        elif in_s1 and '[ atoms ]' in line:
            in_atoms_sec = True
        elif in_s1 and in_atoms_sec and '[ ' in line and 'atoms' not in line:
            break
        elif in_s1 and in_atoms_sec and re.match(r'^\s*\d+\s+\w', line):
            n_sur = int(line.split()[0])

posres_path = os.path.join(OUT_DIR, "posres_SUR.itp")
with open(posres_path, 'w') as f:
    f.write('; position restraints for SUR (APTES surface)\n\n'
            '[ position_restraints ]\n;  i funct       fcx        fcy        fcz\n')
    for i in range(1, n_sur + 1):
        f.write(f'{i}  1  1000  1000  1000\n')
print(f"  posres_SUR.itp: {n_sur} surface atoms")
if n_sur == 0:
    print("  WARNING: surface not found as 'system1' — check topology manually")


# ── STEP 10: Write MDP files ──────────────────────────────────────────────────
print("\n=== STEP 10: Write MDP files ===")

minim_mdp = os.path.join(OUT_DIR, "minim.mdp")
with open(minim_mdp, 'w') as f:
    f.write("""; Energy minimization with surface position restraints
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
""")

nvt_mdp = os.path.join(OUT_DIR, "nvt_equil.mdp")
with open(nvt_mdp, 'w') as f:
    f.write("""; NVT equilibration — 500 ps, dt=2 fs, POSRES_SUR
integrator              = md
nsteps                  = 250000
dt                      = 0.002

nstxout                 = 0
nstvout                 = 0
nstfout                 = 0
nstenergy               = 5000
nstlog                  = 5000
nstxout-compressed      = 5000
compressed-x-grps       = System

continuation            = no
constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 2
lincs_order             = 4

cutoff-scheme           = Verlet
ns_type                 = grid
nstlist                 = 20
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

pcoupl                  = no
pbc                     = xyz

gen_vel                 = yes
gen_temp                = 300
gen_seed                = -1
""")

npt_mdp = os.path.join(OUT_DIR, "npt_equil.mdp")
with open(npt_mdp, 'w') as f:
    f.write("""; NPT equilibration — 1 ns, dt=2 fs, POSRES_SUR
integrator              = md
nsteps                  = 500000
dt                      = 0.002

nstxout                 = 0
nstvout                 = 0
nstfout                 = 0
nstenergy               = 5000
nstlog                  = 5000
nstxout-compressed      = 5000
compressed-x-grps       = System

continuation            = yes
constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 2
lincs_order             = 4

cutoff-scheme           = Verlet
ns_type                 = grid
nstlist                 = 20
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

gen_vel                 = no
""")

prod_mdp = os.path.join(OUT_DIR, "prod_500ns.mdp")
with open(prod_mdp, 'w') as f:
    f.write("""; 500 ns production — HMR dt=4 fs, POSRES_SUR, enzyme free to adsorb
integrator              = md
nsteps                  = 125000000
dt                      = 0.004

nstxout                 = 0
nstvout                 = 0
nstfout                 = 0
nstenergy               = 2500
nstlog                  = 2500
nstxout-compressed      = 2500
compressed-x-grps       = System

continuation            = yes
constraint_algorithm    = lincs
constraints             = h-bonds
lincs_iter              = 1
lincs_order             = 4

cutoff-scheme           = Verlet
ns_type                 = grid
nstlist                 = 100
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

gen_vel                 = no
""")

print(f"  minim.mdp, nvt_equil.mdp, npt_equil.mdp, prod_500ns.mdp written")


# ── STEP 11: grompp + EM ──────────────────────────────────────────────────────
print("\n=== STEP 11: grompp + EM ===")
md_dir = os.path.join(OUT_DIR, "MD")
shutil.copy(GRO, os.path.join(md_dir, "lys_surface.gro"))
shutil.copy(TOP, os.path.join(md_dir, "lys_surface.top"))
shutil.copy(TOP_HMR, os.path.join(md_dir, "lys_surface_hmr.top"))
shutil.copy(posres_path, os.path.join(md_dir, "posres_SUR.itp"))
for mdp in [minim_mdp, nvt_mdp, npt_mdp, prod_mdp]:
    shutil.copy(mdp, os.path.join(md_dir, os.path.basename(mdp)))

grompp_cmd = [GMX, "grompp",
              "-f", "minim.mdp",
              "-c", "lys_surface.gro",
              "-p", "lys_surface.top",
              "-o", "em.tpr",
              "-maxwarn", "5"]
result = subprocess.run(grompp_cmd, capture_output=True, text=True, cwd=md_dir)
with open(os.path.join(md_dir, "grompp_em.log"), 'w') as f:
    f.write(result.stdout + result.stderr)
if not os.path.exists(os.path.join(md_dir, "em.tpr")):
    print("  ERROR: grompp failed")
    for line in (result.stdout + result.stderr).split('\n'):
        if 'error' in line.lower() or 'ERROR' in line:
            print(f"    {line}")
    print(f"  See: {md_dir}/grompp_em.log")
    exit(1)
print("  grompp OK → em.tpr")

mdrun_cmd = ["mpirun", "-np", "4", GMX, "mdrun",
             "-v", "-s", "em.tpr", "-deffnm", "em", "-pin", "on"]
print(f"  Running EM (4 ranks)...")
result = subprocess.run(mdrun_cmd, capture_output=True, text=True, cwd=md_dir)
with open(os.path.join(md_dir, "em.log"), 'w') as f:
    f.write(result.stdout + result.stderr)

em_gro = os.path.join(md_dir, "em.gro")
if os.path.exists(em_gro) and os.path.getsize(em_gro) > 0:
    # Extract final energy and steps from log
    log_lines = open(os.path.join(md_dir, "em.log")).read()
    for line in log_lines.split('\n'):
        if 'converged' in line.lower() or 'Potential Energy' in line:
            print(f"  {line.strip()}")
    print(f"  EM DONE: {em_gro}")
else:
    print("  ERROR: EM may have failed — check MD/em.log")


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"""
{'='*60}
System built: {OUT_DIR}
  Total atoms:   {n_atoms}
  LYS residues:  {n_lys} (11x11 grid, 9 Å spacing)
  Surface atoms: {n_sur}

Next steps (run from {md_dir}):

  # NVT equilibration (500 ps, 2 fs)
  {GMX} grompp -f nvt_equil.mdp -c em.gro -p lys_surface_hmr.top -o nvt.tpr -maxwarn 5
  mpirun -np 4 {GMX} mdrun -v -s nvt.tpr -deffnm nvt -pin on

  # NPT equilibration (1 ns, 2 fs)
  {GMX} grompp -f npt_equil.mdp -c nvt.gro -p lys_surface_hmr.top -t nvt.cpt -o npt.tpr -maxwarn 5
  mpirun -np 4 {GMX} mdrun -v -s npt.tpr -deffnm npt -pin on

  # Production (500 ns, 4 fs HMR)
  {GMX} grompp -f prod_500ns.mdp -c npt.gro -p lys_surface_hmr.top -t npt.cpt -o prod.tpr -maxwarn 5
  mpirun -np 4 {GMX} mdrun -v -s prod.tpr -deffnm prod -pin on
{'='*60}
""")
