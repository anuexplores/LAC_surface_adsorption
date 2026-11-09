#!/usr/bin/env python3
"""
Create GRO + TOP files for hydroquinone, resorcinol, pyrogallol,
phloroglucinol, and phenol on both plain silica (original face) and
APTES (original face).

Output:
  <lig>/MD/sim_start.gro + sim.top          (plain silica, 1 sim)
  aptes_<lig>/MD/repN/repN_<lig>_start.gro  (APTES, rep1–rep5)
  aptes_<lig>/MD/repN/repN_<lig>.top
"""

import os, re, math, subprocess, shutil
import parmed as pmd

# ── Paths ──────────────────────────────────────────────────────────────────────
TLEAP_DIR  = "/Data/Anushka/enzyme/laccase/OUT_CU_lc/test/again/pH5/ChainA/tleap"
MD_BASE    = "/Data/Anushka/Project4/ph5/orients/orient1/tleap"
TLEAP_BIN  = "/apps/codes/anaconda3/bin/tleap"
PACKMOL    = "/apps/codes/anaconda3/bin/packmol"

APTES_BASE_GRO     = os.path.join(MD_BASE, "aptes_catechol", "MD", "1000ns_posres.gro")
APTES_REP_BASE_TOP = os.path.join(MD_BASE, "aptes_catechol", "MD", "_rep_base.top")

# Translation: original-MD-frame → orient1 space
TX, TY, TZ = -37.270, -24.801, 57.918

N_MOLS       = 6
SPHERE_INNER = 37.0
SPHERE_OUTER = 55.0
REPLICAS     = [1, 2, 3, 4, 5]
INSERT_SEEDS = {1: 1001, 2: 2002, 3: 3003, 4: 4004, 5: 5005}

# Enzyme COM in 1000ns_posres.gro
APTES_COM_X, APTES_COM_Y, APTES_COM_Z = 7.9, 85.5, 53.1
APTES_ZMIN  = 34.0   # Å — above APTES tips

_APT = f"{TLEAP_DIR}"
LIGANDS = [
    {
        'name':          'resorcinol',
        'mol_atoms':     14,
        'acpype_mol2':   f"{_APT}/Resorcinol/acpype_ligands_OB/ligands_OB.mol2",
        'acpype_frcmod': f"{_APT}/Resorcinol/acpype_ligands_OB/ligands_OB_AC.frcmod",
        'n_in_src':      6,
    },
    {
        'name':          'pyrogallol',
        'mol_atoms':     15,
        'acpype_mol2':   f"{_APT}/Pyrogallol/acpype_ligands_OB/ligands_OB.mol2",
        'acpype_frcmod': f"{_APT}/Pyrogallol/acpype_ligands_OB/ligands_OB_AC.frcmod",
        'n_in_src':      1,   # single molecule → duplicated to 6
    },
]

NON_ENZ = {'SUR', 'WAT', 'Na+', 'Cl-', 'SOL', 'UNK', 'APT'}

TLEAP_HEADER = """\
source leaprc.protein.ff19SB
source leaprc.gaff2
source leaprc.water.tip3p

addAtomTypes {
        { "M1"  "Cu" "sp3" }
        { "M2"  "Cu" "sp3" }
        { "M3"  "Cu" "sp3" }
        { "M4"  "Cu" "sp3" }
        { "Y1"  "N" "sp3" }
        { "Z1"  "S" "sp3" }
        { "Y3"  "N" "sp3" }
        { "Y4"  "N" "sp3" }
        { "Y5"  "N" "sp3" }
        { "Y6"  "N" "sp3" }
        { "Y7"  "N" "sp3" }
        { "Y8"  "N" "sp3" }
        { "Y9"  "N" "sp3" }
        { "Y2"  "N" "sp3" }
        { "Y10"  "N" "sp3" }
        { "SI" "Si" "sp3" }
        { "OB" "O"  "sp3" }
}

HD1 = loadmol2 HD1.mol2
HD2 = loadmol2 HD2.mol2
HD3 = loadmol2 HD3.mol2
HD4 = loadmol2 HD4.mol2
HD5 = loadmol2 HD5.mol2
HD6 = loadmol2 HD6.mol2
HD7 = loadmol2 HD7.mol2
HE1 = loadmol2 HE1.mol2
HE2 = loadmol2 HE2.mol2
HE3 = loadmol2 HE3.mol2
CM1 = loadmol2 CM1.mol2
CU1 = loadmol2 CU1_modified.mol2
CU2 = loadmol2 CU2_modified.mol2
CU3 = loadmol2 CU3_modified.mol2
CU4 = loadmol2 CU4_modified.mol2

loadamberparams frcmod.ions1lm_126_tip3p
loadamberparams CU1_mcpbpy.frcmod
loadamberparams CU2_mcpbpy.frcmod
loadamberparams CU3_mcpbpy.frcmod
loadamberparams CU4_mcpbpy.frcmod

"""

TLEAP_BONDS = """\
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
"""


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS  (identical to setup_new_ligands.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_orient1_mol2(lig):
    src        = lig['acpype_mol2']
    n_src      = lig['n_in_src']
    mol_atoms  = lig['mol_atoms']
    name       = lig['name']
    out_mol2   = os.path.join(TLEAP_DIR, f"orient1_{name}.mol2")

    with open(src) as f:
        content = f.read()

    sections = {}
    current = None
    for line in content.split('\n'):
        if line.startswith('@<TRIPOS>'):
            current = line.strip()
            sections[current] = []
        elif current is not None and line.strip():
            sections[current].append(line)

    atoms = []
    for line in sections.get('@<TRIPOS>ATOM', []):
        p = line.split()
        if len(p) >= 9:
            atoms.append({'name': p[1], 'x': float(p[2]), 'y': float(p[3]),
                          'z': float(p[4]), 'type': p[5],
                          'res_id': p[6], 'res_name': p[7], 'charge': p[8]})

    bonds = []
    for line in sections.get('@<TRIPOS>BOND', []):
        p = line.split()
        if len(p) >= 4:
            bonds.append((int(p[1]), int(p[2]), p[3]))

    bonds_per_mol = len(bonds) // n_src

    if n_src < N_MOLS:
        base_atoms = atoms[:]
        base_bonds = bonds[:]
        for dup in range(N_MOLS - n_src):
            src_mol = dup % n_src
            a_start = src_mol * mol_atoms
            mol = base_atoms[a_start:a_start + mol_atoms]
            x_offset = (n_src + dup) * 10.0
            new_res = str(504 + n_src + dup)
            for a in mol:
                atoms.append({'name': a['name'], 'x': a['x'] + x_offset,
                              'y': a['y'], 'z': a['z'], 'type': a['type'],
                              'res_id': new_res, 'res_name': a['res_name'],
                              'charge': a['charge']})
            b_start      = src_mol * bonds_per_mol
            target_start = (n_src + dup) * mol_atoms
            source_start =  src_mol      * mol_atoms
            atom_offset  = target_start - source_start
            for (a1, a2, bt) in base_bonds[b_start:b_start + bonds_per_mol]:
                bonds.append((a1 + atom_offset, a2 + atom_offset, bt))

    assert len(atoms) == N_MOLS * mol_atoms, \
        f"{name}: expected {N_MOLS*mol_atoms} atoms, got {len(atoms)}"

    out_atom_lines = []
    for i, a in enumerate(atoms):
        res_id = str(504 + i // mol_atoms)
        out_atom_lines.append(
            f"{i+1:>7} {a['name']:<8} {a['x']+TX:10.4f} {a['y']+TY:10.4f} "
            f"{a['z']+TZ:10.4f} {a['type']:<8} {res_id:<4} {a['res_name']:<8} {a['charge']}\n")

    out_bond_lines = []
    for i, (a1, a2, bt) in enumerate(bonds):
        out_bond_lines.append(f"{i+1:>6} {a1:>5} {a2:>5} {bt}\n")

    mol_header = sections.get('@<TRIPOS>MOLECULE', [])
    out_lines = ['@<TRIPOS>MOLECULE\n']
    for j, line in enumerate(mol_header):
        if j == 1 and re.match(r'^\s*\d+\s+\d+', line):
            p = line.split()
            p[0] = str(N_MOLS * mol_atoms)
            p[1] = str(len(bonds))
            out_lines.append('   ' + '    '.join(p) + '\n')
        else:
            out_lines.append(line + '\n')

    out_lines.append('@<TRIPOS>ATOM\n')
    out_lines.extend(out_atom_lines)
    out_lines.append('@<TRIPOS>BOND\n')
    out_lines.extend(out_bond_lines)

    if '@<TRIPOS>SUBSTRUCTURE' in sections:
        out_lines.append('@<TRIPOS>SUBSTRUCTURE\n')
        for line in sections['@<TRIPOS>SUBSTRUCTURE']:
            out_lines.append(line + '\n')

    with open(out_mol2, 'w') as f:
        f.writelines(out_lines)
    print(f"    orient1_{name}.mol2: {N_MOLS} mols × {mol_atoms} atoms, {len(bonds)} bonds")
    return out_mol2


def run_tleap(script, tleap_in, expected_prmtop):
    with open(tleap_in, 'w') as f:
        f.write(script)
    result = subprocess.run([TLEAP_BIN, "-f", os.path.basename(tleap_in)],
                            cwd=TLEAP_DIR, capture_output=True, text=True)
    with open(tleap_in.replace('.in', '.out'), 'w') as f:
        f.write(result.stdout + ('\nSTDERR:\n' + result.stderr if result.stderr else ''))
    ok = os.path.exists(expected_prmtop) and os.path.getsize(expected_prmtop) > 0
    if ok:
        print(f"    tleap OK  ({os.path.getsize(expected_prmtop)//1024} KB)")
    else:
        errs = [l for l in result.stdout.split('\n') if 'FATAL' in l or 'error' in l.lower()]
        print(f"    tleap FAILED"); [print(f"      {e}") for e in errs[:5]]
    return ok


def parmed_convert(prmtop, inpcrd, gro_out, top_out):
    amber = pmd.load_file(prmtop, inpcrd)
    amber.save(gro_out, format='gro',     overwrite=True)
    amber.save(top_out, format='gromacs', overwrite=True)
    for path in [gro_out, top_out]:
        with open(path) as f: txt = f.read()
        txt = txt.replace('R-2', 'SUR')
        if path.endswith('.top'):
            block = '\n#ifdef POSRES_SUR\n#include "posres_SUR.itp"\n#endif\n'
            if '#ifdef POSRES_SUR' not in txt:
                for tgt in ['\n[ moleculetype ]\n; Name            nrexcl\nNa+',
                             '\n[ moleculetype ]\n; Name            nrexcl\nCl-',
                             '\n[ moleculetype ]\n; Name            nrexcl\nWAT']:
                    if tgt in txt:
                        txt = txt.replace(tgt, block + tgt, 1); break
        with open(path, 'w') as f: f.write(txt)
    with open(gro_out) as f: n = int(f.readlines()[1].strip())
    print(f"    parmed: {n} atoms")


def extract_unk_section(top_path):
    with open(top_path) as f: lines = f.readlines()
    start = end = None
    for i, l in enumerate(lines):
        if '[ moleculetype ]' in l:
            for j in range(i+1, min(i+6, len(lines))):
                s = lines[j].strip()
                if s and not s.startswith(';'):
                    if s.startswith('UNK') and start is None:
                        start = i
                    elif start is not None:
                        end = i
                    break
        if start is not None and end is not None:
            break
    if start is None:
        raise RuntimeError(f"UNK moleculetype not found in {top_path}")
    return ''.join(lines[start:end])


def make_rep_top(unk_block, out_path):
    with open(APTES_REP_BASE_TOP) as f: lines = f.readlines()
    start = end = None
    for i, l in enumerate(lines):
        if '[ moleculetype ]' in l:
            for j in range(i+1, min(i+6, len(lines))):
                s = lines[j].strip()
                if s and not s.startswith(';'):
                    if s.startswith('UNK') and start is None:
                        start = i
                    elif start is not None:
                        end = i
                    break
        if start is not None and end is not None:
            break
    with open(out_path, 'w') as f:
        f.writelines(lines[:start])
        f.write(unk_block)
        f.writelines(lines[end:])


def fix_unk_charge(top_path, mol_atoms):
    """Adjust last UNK atom charge so per-molecule charge sums to exactly 0."""
    with open(top_path) as f: lines = f.readlines()
    in_unk = False; in_atoms = False; atom_indices = []
    for i, l in enumerate(lines):
        s = l.strip()
        if '[ moleculetype ]' in s: in_unk = False; in_atoms = False
        if s == 'UNK          3': in_unk = True
        if in_unk and s == '[ atoms ]': in_atoms = True; continue
        if in_atoms and s.startswith('['): in_atoms = False
        if in_atoms and s and not s.startswith(';'):
            if len(s.split()) >= 7: atom_indices.append(i)
    if len(atom_indices) != mol_atoms: return
    charges = [float(lines[i].split()[6]) for i in atom_indices]
    excess = sum(charges)
    if abs(excess) < 1e-8: return
    last_i = atom_indices[-1]
    parts = lines[last_i].split()
    old_q = float(parts[6]); new_q = old_q - excess
    lines[last_i] = lines[last_i].replace(f" {parts[6]} ", f" {new_q:.8f} ", 1)
    with open(top_path, 'w') as f: f.writelines(lines)


def packmol_place(enzy_pdb, lig_single_pdb, pack_out, pack_inp,
                  com_x, com_y, com_z, zmin, seed, work_dir, mol_atoms):
    with open(pack_inp, 'w') as f:
        f.write(f"tolerance 2.0\nfiletype pdb\noutput {pack_out}\nseed {seed}\n\n")
        f.write(f"structure {enzy_pdb}\n  number 1\n  fixed 0. 0. 0. 0. 0. 0.\nend structure\n\n")
        f.write(f"structure {lig_single_pdb}\n  number {N_MOLS}\n")
        f.write(f"  inside sphere  {com_x:.2f} {com_y:.2f} {com_z:.2f} {SPHERE_OUTER}\n")
        f.write(f"  outside sphere {com_x:.2f} {com_y:.2f} {com_z:.2f} {SPHERE_INNER}\n")
        f.write(f"  above plane 0. 0. 1. {zmin:.2f}\n")
        f.write(f"end structure\n")
    result = subprocess.run(PACKMOL, stdin=open(pack_inp),
                            capture_output=True, text=True, cwd=work_dir)
    if not os.path.exists(pack_out):
        print("    PACKMOL failed"); print(result.stdout[-400:]); return None
    with open(pack_out) as f:
        placed = [l for l in f if l.startswith(("ATOM","HETATM")) and "UNK" in l]
    n_exp = N_MOLS * mol_atoms
    status = "OK" if len(placed) == n_exp else f"WARNING: got {len(placed)}, expected {n_exp}"
    print(f"    PACKMOL: {len(placed)} atoms  {status}")
    return placed


def placed_to_gro(placed, mol_atoms):
    lines = []
    for idx, pline in enumerate(placed):
        mol_idx = idx // mol_atoms + 1
        aname = pline[12:16].strip()
        x = float(pline[30:38])/10; y = float(pline[38:46])/10; z = float(pline[46:54])/10
        lines.append(f"{mol_idx:5d}{'UNK':<5s}{aname:>5s}{idx+1:5d}"
                     f"{x:8.3f}{y:8.3f}{z:8.3f}\n")
    return lines


def write_gro(base_lines, unk_lines, title, box, out_path):
    total = len(base_lines) + len(unk_lines)
    with open(out_path, 'w') as f:
        f.write(title)
        f.write(f"{total}\n")
        for i, l in enumerate(base_lines):
            f.write(l[:15] + f"{(i+1)%100000:5d}" + l[20:])
        for l in unk_lines:
            f.write(l)
        f.write(box)
    print(f"    GRO: {os.path.basename(out_path)}  ({total} atoms)")


def single_mol_pdb(unk_gro_lines, mol_atoms, out_path):
    with open(out_path, 'w') as f:
        for i, line in enumerate(unk_gro_lines[:mol_atoms]):
            aname = line[10:15].strip()
            x = float(line[20:28])*10; y = float(line[28:36])*10; z = float(line[36:44])*10
            f.write(f"HETATM{i+1:5d} {aname:<4s} UNK A{1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
        f.write("END\n")


def enzy_surface_pdb(atom_lines, excl_res, out_path):
    non_solv = [l for l in atom_lines if l[5:10].strip() not in excl_res]
    with open(out_path, 'w') as f:
        for i, line in enumerate(non_solv):
            aname = line[10:15].strip(); resnam = line[5:10].strip()
            resnum = int(line[0:5].strip())
            x = float(line[20:28])*10; y = float(line[28:36])*10; z = float(line[36:44])*10
            f.write(f"ATOM  {i+1:5d} {aname:<4s} {resnam:<3s} A{resnum%9999:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
        f.write("END\n")
    return non_solv


def get_missing_atomtypes(rep_top_path, mol_atoms):
    """Return list of (name, definition) for atomtypes used in UNK but missing from [ atomtypes ]."""
    with open(rep_top_path) as f: lines = f.readlines()
    # Collect defined atomtypes
    in_at = False; defined = set()
    for l in lines:
        s = l.strip()
        if '[ atomtypes ]' in s: in_at = True; continue
        if in_at and s.startswith('['): in_at = False
        if in_at and s and not s.startswith(';'):
            defined.add(s.split()[0])
    # Collect UNK atom types
    in_unk = False; in_atoms = False; used = set()
    for l in lines:
        s = l.strip()
        if '[ moleculetype ]' in s: in_unk = False; in_atoms = False
        if s == 'UNK          3': in_unk = True
        if in_unk and s == '[ atoms ]': in_atoms = True; continue
        if in_atoms and s.startswith('['): in_atoms = False
        if in_atoms and s and not s.startswith(';') and len(s.split()) >= 2:
            used.add(s.split()[1])
    return sorted(used - defined)


def add_missing_atomtypes(top_path, missing_defs):
    """Insert missing atomtype lines before [ cmaptypes ]."""
    with open(top_path) as f: txt = f.read()
    insert = ''.join(f"{d}\n" for d in missing_defs)
    old = "HW             1   1.008000  0.00000000  A              0              0\n\n[ cmaptypes ]"
    new = "HW             1   1.008000  0.00000000  A              0              0\n" + insert + "\n[ cmaptypes ]"
    if old in txt:
        with open(top_path, 'w') as f: f.write(txt.replace(old, new, 1))


# Atomtype definitions from GAFF2 (for any that may be missing from catechol base)
GAFF2_ATOMTYPE_DEFS = {
    'nv': 'nv             7  14.010000  0.00000000  A     0.33681317       0.468608',
    'nh': 'nh             7  14.010000  0.00000000  A      0.3189952        0.89956',
    'oh': 'oh             8  16.000000  0.00000000  A     0.32428713       0.389112',
    'ho': 'ho             1   1.008000  0.00000000  A    0.053453923      0.0012552',
    'ca': 'ca             6  12.010000  0.00000000  A     0.33152123      0.4133792',
    'ha': 'ha             1   1.008000  0.00000000  A     0.26254785      0.0673624',
    'c3': 'c3             6  12.011000  0.00000000  A     0.33996695      0.4577296',
    'hc': 'hc             1   1.008000  0.00000000  A     0.26495328      0.0656888',
}


# ── Parse APTES base GRO once ──────────────────────────────────────────────────
print("=" * 65)
print("Parsing APTES base GRO (1000ns_posres.gro) ...")
with open(APTES_BASE_GRO) as f: ag = f.readlines()
aptes_n     = int(ag[1].strip())
aptes_box   = ag[-1]
aptes_title = ag[0]
aptes_atoms = ag[2:2+aptes_n]
aptes_base  = [l for l in aptes_atoms if l[5:10].strip() != 'UNK']
print(f"  Total: {aptes_n}  Base (no UNK): {len(aptes_base)}")

enzy_sur_aptes_pdb = os.path.join(MD_BASE, "aptes_catechol", "MD", "_enzy_sur_aptes_base.pdb")
enzy_surface_pdb(aptes_base, {'WAT','Cl-','SOL','Na+'}, enzy_sur_aptes_pdb)
print(f"  Wrote shared APTES enzyme+surface PDB")


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════
for lig in LIGANDS:
    name      = lig['name']
    mol_atoms = lig['mol_atoms']
    frcmod    = lig['acpype_frcmod']

    print(f"\n{'='*65}")
    print(f"LIGAND: {name}  ({mol_atoms} atoms/mol × {N_MOLS})")

    # Check parameters exist
    if not os.path.exists(lig['acpype_mol2']):
        print(f"  SKIP: mol2 not found: {lig['acpype_mol2']}"); continue
    if not os.path.exists(frcmod):
        print(f"  SKIP: frcmod not found: {frcmod}"); continue

    lig_mol2 = build_orient1_mol2(lig)

    plain_md = os.path.join(MD_BASE, name,            "MD")
    aptes_md = os.path.join(MD_BASE, f"aptes_{name}", "MD")
    os.makedirs(plain_md, exist_ok=True)
    os.makedirs(aptes_md, exist_ok=True)

    # ── A) Plain silica ─────────────────────────────────────────────────────
    print(f"\n  [A] Plain silica ...")
    prmtop_p = os.path.join(TLEAP_DIR, f"orient1_{name}_surface_complex_solv.prmtop")
    inpcrd_p = os.path.join(TLEAP_DIR, f"orient1_{name}_surface_complex_solv.inpcrd")
    gro_p    = os.path.join(TLEAP_DIR, f"orient1_{name}_surface_complex_solv.gro")
    top_p    = os.path.join(TLEAP_DIR, f"orient1_{name}_surface_complex_solv.top")

    tleap_p = (TLEAP_HEADER
        + f"LIG = loadmol2 orient1_{name}.mol2\n"
        + f"loadamberparams {frcmod}\n\n"
        + "loadamberparams orient1_surface_original.frcmod\n"
        + "surf = loadmol2 orient1_surface_original.mol2\n\n"
        + "mol = loadpdb orient1_combined.pdb\n\n"
        + TLEAP_BONDS + "\n"
        + "COM = combine {mol LIG surf}\n"
        + "solvatebox COM TIP3PBOX 10.0\n"
        + "addions COM Na+ 9\naddions COM Cl- 0\n"
        + f"saveamberparm COM orient1_{name}_surface_complex_solv.prmtop "
        + f"orient1_{name}_surface_complex_solv.inpcrd\n\nquit\n")

    tleap_in_p = os.path.join(TLEAP_DIR, f"tleap_orient1_{name}_original.in")
    if not run_tleap(tleap_p, tleap_in_p, prmtop_p):
        print(f"  SKIP {name} plain"); continue
    parmed_convert(prmtop_p, inpcrd_p, gro_p, top_p)

    # PACKMOL for plain silica
    with open(gro_p) as f: pg = f.readlines()
    pn = int(pg[1].strip())
    p_base  = [l for l in pg[2:2+pn] if l[5:10].strip() != 'UNK']
    p_unk   = [l for l in pg[2:2+pn] if l[5:10].strip() == 'UNK']
    enzy_pdb_p = os.path.join(plain_md, f"_enzy_sur_{name}.pdb")
    enzy_surface_pdb(p_base, {'WAT','Na+','Cl-','SOL'}, enzy_pdb_p)

    lig_pdb_p = os.path.join(plain_md, f"_{name}_single.pdb")
    single_mol_pdb(p_unk, mol_atoms, lig_pdb_p)

    # Plain silica COM (enzyme COM from plain tleap GRO)
    enz_lines = [l for l in p_base if l[5:10].strip() not in NON_ENZ]
    if enz_lines:
        px = sum(float(l[20:28]) for l in enz_lines)/len(enz_lines)*10
        py = sum(float(l[28:36]) for l in enz_lines)/len(enz_lines)*10
        pz = sum(float(l[36:44]) for l in enz_lines)/len(enz_lines)*10
    else:
        px, py, pz = APTES_COM_X, APTES_COM_Y, APTES_COM_Z

    # ZMIN: max Z of SUR atoms + 5 Å
    sur_lines = [l for l in p_base if l[5:10].strip() == 'SUR']
    zmin_p = max(float(l[36:44])*10 for l in sur_lines) + 5.0 if sur_lines else 15.0

    pack_out_p = os.path.join(plain_md, f"_{name}_placed.pdb")
    pack_inp_p = os.path.join(plain_md, f"_packmol_{name}.inp")
    placed_p = packmol_place(enzy_pdb_p, lig_pdb_p, pack_out_p, pack_inp_p,
                             px, py, pz, zmin_p, 1001, plain_md, mol_atoms)
    if placed_p is None: continue

    gro_sim = os.path.join(plain_md, "sim_start.gro")
    write_gro(p_base, placed_to_gro(placed_p, mol_atoms), pg[0], pg[-1], gro_sim)

    # sim.top: insert UNK 6 after Na+ and WAT
    with open(top_p) as f: top_txt = f.read()
    sim_top = re.sub(
        r"(Na\+\s+9\r?\nWAT\s+\d+)",
        lambda m: m.group(1) + "\nUNK              6", top_txt)
    if sim_top == top_txt:
        sim_top = re.sub(
            r"(Na\+\s+9\r?\nWAT\s+\d+)",
            lambda m: m.group(1) + "\nUNK              6", top_txt)
    with open(os.path.join(plain_md, "sim.top"), 'w') as f: f.write(sim_top)
    print(f"    Wrote sim_start.gro + sim.top")

    # ── B) APTES replicas ───────────────────────────────────────────────────
    print(f"\n  [B] APTES replicas ...")
    prmtop_a = os.path.join(TLEAP_DIR, f"orient1_aptes_{name}_solv.prmtop")
    inpcrd_a = os.path.join(TLEAP_DIR, f"orient1_aptes_{name}_solv.inpcrd")
    gro_a    = os.path.join(TLEAP_DIR, f"orient1_aptes_{name}_solv.gro")
    top_a    = os.path.join(TLEAP_DIR, f"orient1_aptes_{name}_solv.top")

    tleap_a = (TLEAP_HEADER
        + f"LIG = loadmol2 orient1_{name}.mol2\n"
        + f"loadamberparams {frcmod}\n\n"
        + "loadamberparams orient1_aptes_surface_original.frcmod\n"
        + "surf = loadmol2 orient1_aptes_surface_original.mol2\n\n"
        + "mol = loadpdb orient1_combined.pdb\n\n"
        + TLEAP_BONDS + "\n"
        + "COM = combine {mol LIG surf}\n"
        + "solvatebox COM TIP3PBOX 10.0\n"
        + "addions COM Cl- 11\naddions COM Na+ 0\n"
        + f"saveamberparm COM orient1_aptes_{name}_solv.prmtop "
        + f"orient1_aptes_{name}_solv.inpcrd\n\nquit\n")

    tleap_in_a = os.path.join(TLEAP_DIR, f"tleap_orient1_aptes_{name}_original.in")
    if not run_tleap(tleap_a, tleap_in_a, prmtop_a):
        print(f"  SKIP {name} APTES"); continue
    parmed_convert(prmtop_a, inpcrd_a, gro_a, top_a)

    # Extract UNK block and build _rep_base top
    unk_block = extract_unk_section(top_a)
    rep_base_top = os.path.join(aptes_md, f"_rep_base_{name}.top")
    make_rep_top(unk_block, rep_base_top)
    print(f"    _rep_base_{name}.top ready")

    # Check and fix missing atomtypes in rep_base
    missing = get_missing_atomtypes(rep_base_top, mol_atoms)
    if missing:
        defs = [GAFF2_ATOMTYPE_DEFS[t] for t in missing if t in GAFF2_ATOMTYPE_DEFS]
        unknown = [t for t in missing if t not in GAFF2_ATOMTYPE_DEFS]
        if unknown: print(f"    WARNING: unknown atomtypes {unknown} — add manually")
        if defs:
            add_missing_atomtypes(rep_base_top, defs)
            print(f"    Added atomtypes to rep_base: {missing}")

    # Fix charge
    fix_unk_charge(rep_base_top, mol_atoms)

    # Single-mol PDB from APTES tleap GRO
    with open(gro_a) as f: ag2 = f.readlines()
    an2 = int(ag2[1].strip())
    ag2_unk = [l for l in ag2[2:2+an2] if l[5:10].strip() == 'UNK']
    lig_pdb_a = os.path.join(aptes_md, f"_{name}_single.pdb")
    single_mol_pdb(ag2_unk, mol_atoms, lig_pdb_a)

    # Per-replica setup
    for rep in REPLICAS:
        rep_dir = os.path.join(aptes_md, f"rep{rep}")
        os.makedirs(rep_dir, exist_ok=True)

        pack_out_r = os.path.join(rep_dir, f"{name}_placed.pdb")
        pack_inp_r = os.path.join(rep_dir, f"packmol_{name}.inp")
        placed_r = packmol_place(
            enzy_sur_aptes_pdb, lig_pdb_a, pack_out_r, pack_inp_r,
            APTES_COM_X, APTES_COM_Y, APTES_COM_Z,
            APTES_ZMIN, INSERT_SEEDS[rep], rep_dir, mol_atoms)
        if placed_r is None: continue

        rep_gro = os.path.join(rep_dir, f"rep{rep}_{name}_start.gro")
        write_gro(aptes_base, placed_to_gro(placed_r, mol_atoms),
                  aptes_title, aptes_box, rep_gro)

        rep_top = os.path.join(rep_dir, f"rep{rep}_{name}.top")
        shutil.copy(rep_base_top, rep_top)
        print(f"    rep{rep}: {rep_gro.split('/')[-1]} + {rep_top.split('/')[-1]}")

print(f"\n{'='*65}\nALL DONE\n{'='*65}")
for lig in LIGANDS:
    n = lig['name']
    print(f"\n  {n}:")
    print(f"    plain:  {MD_BASE}/{n}/MD/sim_start.gro + sim.top")
    for rep in REPLICAS:
        print(f"    APTES rep{rep}: {MD_BASE}/aptes_{n}/MD/rep{rep}/rep{rep}_{n}_start.gro")
