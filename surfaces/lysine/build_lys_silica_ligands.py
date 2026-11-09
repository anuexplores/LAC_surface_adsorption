"""
Build lys_silica + 6-ligand systems for all 10 ligands from frame 1823.pdb.
Extracts enzyme + surface conformation from 1823.pdb, places 6 ligand copies
in solution, solvates, and converts to GROMACS format.
"""
import os, subprocess, shutil, re

PDB1823 = "/Data/Anushka/Project4/ph5/orients/orient1/tleap/lysine/lys_silica_surface/MD/1823.pdb"
BASE_TLEAP = "/Data/Anushka/enzyme/laccase/OUT_CU_lc/test/again/pH5/ChainA/tleap"
LYS_DIR   = "/Data/Anushka/Project4/ph5/orients/orient1/tleap/lysine/lys_silica_surface"
OUT_BASE  = "/Data/Anushka/Project4/ph5/orients/orient1/tleap/lysine"

SUR_MOL2 = os.path.join(LYS_DIR, "silica_surface_SUR.mol2")
SLY_MOL2 = os.path.join(LYS_DIR, "lysine_monolayer.mol2")
SUR_FRCMOD = os.path.join(BASE_TLEAP, "orient1_surface_original.frcmod")

# Residue types that belong to the enzyme (not surface/solvent)
ENZYME_RES = {
    'ALA','ARG','ASH','ASN','ASP','CYX','GLN','GLU','GLY','HID','HIE',
    'HIP','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR',
    'VAL','CM1','CU1','CU2','CU3','CU4','HD1','HD2','HD3','HD4','HD5',
    'HD6','HD7','HE1','HE2','HE3'
}

LIGANDS = {
    'catechol':      {'frcmod': f'{BASE_TLEAP}/Catechol/acpype_ligands_OB/ligand_OB_AC.frcmod',
                      'mol2':   f'{BASE_TLEAP}/Catechol/acpype_ligands_OB/ligand_OB.mol2'},
    'phenol':        {'frcmod': f'{BASE_TLEAP}/Phenol/acpype_ligands_OB/ligand_OB_AC.frcmod',
                      'mol2':   f'{BASE_TLEAP}/Phenol/acpype_ligands_OB/ligand_OB.mol2'},
    'hydroquinone':  {'frcmod': f'{BASE_TLEAP}/Hydroquinone/acpype_ligands_OB/ligands_OB_AC.frcmod',
                      'mol2':   f'{BASE_TLEAP}/Hydroquinone/acpype_ligands_OB/ligands_OB.mol2'},
    'hydroxyquinol': {'frcmod': f'{BASE_TLEAP}/Hydroxyquinol/acpype_ligands_OB/ligand_OB_AC.frcmod',
                      'mol2':   f'{BASE_TLEAP}/Hydroxyquinol/acpype_ligands_OB/ligand_OB.mol2'},
    'phloroglucinol':{'frcmod': f'{BASE_TLEAP}/Phloroglucinol/acpype_ligands_OB/ligand_OB_AC.frcmod',
                      'mol2':   f'{BASE_TLEAP}/Phloroglucinol/acpype_ligands_OB/ligand_OB.mol2'},
    'pyrogallol':    {'frcmod': f'{BASE_TLEAP}/Pyrogallol/acpype/ligand_OB_AC.frcmod',
                      'mol2':   f'{BASE_TLEAP}/Pyrogallol/acpype/ligand_OB.mol2'},
    'resorcinol':    {'frcmod': f'{BASE_TLEAP}/Resorcinol/acpype_ligands_OB (1)/ligand_OB_AC.frcmod',
                      'mol2':   f'{BASE_TLEAP}/Resorcinol/acpype_ligands_OB (1)/ligand_OB.mol2'},
    'aniline':       {'frcmod': f'{BASE_TLEAP}/Aniline/acpype_ligands_OB/ligands_OB_AC.frcmod',
                      'mol2':   f'{BASE_TLEAP}/Aniline/acpype_ligands_OB/ligands_OB.mol2'},
    '25xylidine':    {'frcmod': f'{BASE_TLEAP}/Cat+Xyl/acpype_ligands2_OB/ligands2_OB_AC.frcmod',
                      'mol2':   f'{BASE_TLEAP}/Cat+Xyl/acpype_ligands2_OB/ligands2_OB.mol2'},
    'benzene':       {'frcmod': f'{BASE_TLEAP}/Benzene/acpype_ligands_OB (1)/ligands_OB_AC.frcmod',
                      'mol2':   f'{BASE_TLEAP}/Benzene/acpype_ligands_OB (1)/ligands_OB.mol2'},
}

# Target center for the 6-molecule cluster (already contained in one mol2)
# Place cluster in solution above the enzyme: center of box XY, mid-solution Z
TARGET_CENTER = (62.0, 62.0, 110.0)

AMBER_ENV = "AmberTools23"


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_mol2_atoms(path):
    """Return list of raw atom lines from @<TRIPOS>ATOM block."""
    lines, in_a = [], False
    for l in open(path):
        if l.strip() == '@<TRIPOS>ATOM':
            in_a = True; continue
        if l.startswith('@<TRIPOS>') and in_a:
            break
        if in_a:
            lines.append(l)
    return lines


def mol2_center(path):
    xs, ys, zs = [], [], []
    for l in parse_mol2_atoms(path):
        p = l.split()
        if len(p) >= 5:
            try:
                xs.append(float(p[2])); ys.append(float(p[3])); zs.append(float(p[4]))
            except ValueError:
                pass
    return (min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2


def update_mol2_coords(mol2_src, pdb_lines, res_name, out_path):
    """Replace XYZ in mol2_src with coords of res_name atoms from pdb_lines (in order)."""
    pdb_coords = []
    for l in pdb_lines:
        if l.startswith(('ATOM', 'HETATM')) and l[17:20].strip() == res_name:
            x = float(l[30:38]); y = float(l[38:46]); z = float(l[46:54])
            pdb_coords.append((x, y, z))

    mol2_atom_lines = parse_mol2_atoms(mol2_src)
    if len(mol2_atom_lines) != len(pdb_coords):
        raise ValueError(f"{res_name}: mol2 has {len(mol2_atom_lines)} atoms, PDB has {len(pdb_coords)}")

    new_mol2 = []
    in_a = False
    atom_idx = 0
    for l in open(mol2_src):
        if l.strip() == '@<TRIPOS>ATOM':
            in_a = True
            new_mol2.append(l)
            continue
        if l.startswith('@<TRIPOS>') and in_a:
            in_a = False
        if in_a:
            p = l.split()
            if len(p) >= 5:
                x, y, z = pdb_coords[atom_idx]; atom_idx += 1
                # Rebuild line preserving atom_id, name, type, resid, resname, charge
                new_line = f"{p[0]:>6s} {p[1]:<8s} {x:>10.4f} {y:>10.4f} {z:>10.4f}"
                if len(p) >= 9:
                    new_line += f"  {p[5]:<5s} {p[6]:>3s}  {p[7]:<8s} {p[8]}\n"
                else:
                    new_line += "\n"
                new_mol2.append(new_line)
            else:
                new_mol2.append(l)
        else:
            new_mol2.append(l)
    with open(out_path, 'w') as f:
        f.writelines(new_mol2)
    print(f"  Updated mol2: {out_path} ({atom_idx} atoms)")


def extract_enzyme_pdb(pdb_lines, out_path):
    """Write enzyme-only PDB (protein + Cu cofactors) from 1823.pdb.
    Strips OXT (C-terminal O added by GROMACS) and N-terminal H1/H2/H3,
    as tleap adds terminal atoms itself from the force field.
    """
    with open(out_path, 'w') as f:
        for l in pdb_lines:
            if l.startswith('CRYST1'):
                f.write(l); break
        for l in pdb_lines:
            if not l.startswith(('ATOM', 'HETATM')):
                continue
            res  = l[17:20].strip()
            atom = l[12:16].strip()
            if res not in ENZYME_RES:
                continue
            if atom == 'OXT':
                continue
            # Strip N-terminal extra protons added by GROMACS (H1 H2 H3 on first residue)
            if atom in ('H1', 'H2', 'H3') and res == 'GLY':
                resseq = l[22:26].strip()
                if resseq == '1':
                    continue
            f.write(l)
        f.write("END\n")
    n = sum(1 for l in open(out_path) if l.startswith(('ATOM','HETATM')))
    print(f"  Enzyme PDB: {out_path} ({n} atoms)")


def write_tleap(lig_name, lig_mol2, lig_frcmod, translate_vec,
                enzyme_pdb, sur_mol2, sly_mol2, out_dir):
    """Write tleap input for lys_silica + 6 ligands (mol2 already has 6 molecules)."""
    sys_name = f"lys_silica_{lig_name}"
    tleap_path = os.path.join(out_dir, f"tleap_{sys_name}.in")
    prmtop = os.path.join(out_dir, f"{sys_name}.prmtop")
    inpcrd = os.path.join(out_dir, f"{sys_name}.inpcrd")

    # Quote paths that contain spaces
    def q(p): return f'"{p}"' if ' ' in p else p

    tx, ty, tz = translate_vec
    lig_loads = f"LIG = loadmol2 {q(lig_mol2)}\ntranslate LIG {{{tx:.3f} {ty:.3f} {tz:.3f}}}\n"
    lig_combines = " LIG"

    content = f"""source leaprc.protein.ff19SB
source leaprc.gaff2
source leaprc.water.tip3p

addAtomTypes {{
    {{ "M1"  "Cu" "sp3" }}
    {{ "M2"  "Cu" "sp3" }}
    {{ "M3"  "Cu" "sp3" }}
    {{ "M4"  "Cu" "sp3" }}
    {{ "Y1"  "N"  "sp3" }}
    {{ "Z1"  "S"  "sp3" }}
    {{ "Y3"  "N"  "sp3" }}
    {{ "Y4"  "N"  "sp3" }}
    {{ "Y5"  "N"  "sp3" }}
    {{ "Y6"  "N"  "sp3" }}
    {{ "Y7"  "N"  "sp3" }}
    {{ "Y8"  "N"  "sp3" }}
    {{ "Y9"  "N"  "sp3" }}
    {{ "Y2"  "N"  "sp3" }}
    {{ "Y10" "N"  "sp3" }}
    {{ "SI"  "Si" "sp3" }}
    {{ "OB"  "O"  "sp3" }}
}}

HD1 = loadmol2 {BASE_TLEAP}/HD1.mol2
HD2 = loadmol2 {BASE_TLEAP}/HD2.mol2
HD3 = loadmol2 {BASE_TLEAP}/HD3.mol2
HD4 = loadmol2 {BASE_TLEAP}/HD4.mol2
HD5 = loadmol2 {BASE_TLEAP}/HD5.mol2
HD6 = loadmol2 {BASE_TLEAP}/HD6.mol2
HD7 = loadmol2 {BASE_TLEAP}/HD7.mol2
HE1 = loadmol2 {BASE_TLEAP}/HE1.mol2
HE2 = loadmol2 {BASE_TLEAP}/HE2.mol2
HE3 = loadmol2 {BASE_TLEAP}/HE3.mol2
CM1 = loadmol2 /Data/Anushka/Project4/ph5/orients/orient1/tleap/lysine/lys_silica_shared/CM1_corrected.mol2
CU1 = loadmol2 {BASE_TLEAP}/CU1_modified.mol2
CU2 = loadmol2 {BASE_TLEAP}/CU2_modified.mol2
CU3 = loadmol2 {BASE_TLEAP}/CU3_modified.mol2
CU4 = loadmol2 {BASE_TLEAP}/CU4_modified.mol2

loadamberparams /apps/scratch/amber22/dat/leap/parm/frcmod.ions1lm_126_tip3p
loadamberparams {BASE_TLEAP}/CU1_mcpbpy.frcmod
loadamberparams {BASE_TLEAP}/CU2_mcpbpy.frcmod
loadamberparams {BASE_TLEAP}/CU3_mcpbpy.frcmod
loadamberparams {BASE_TLEAP}/CU4_mcpbpy.frcmod

loadamberparams {q(lig_frcmod)}
{lig_loads}
loadamberparams {SUR_FRCMOD}
surf = loadmol2 {sur_mol2}
SLY_mono = loadmol2 {sly_mol2}

mol = loadpdb {enzyme_pdb}

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

COM = combine {{mol{lig_combines} SLY_mono surf}}
solvatebox COM TIP3PBOX 10.0
addions COM Na+ 0
addions COM Cl- 0
saveamberparm COM {prmtop} {inpcrd}

quit
"""
    with open(tleap_path, 'w') as f:
        f.write(content)
    return tleap_path, prmtop, inpcrd


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    pdb_lines = open(PDB1823).readlines()

    # Step 1: Extract enzyme PDB
    shared_dir = os.path.join(OUT_BASE, "lys_silica_shared")
    os.makedirs(shared_dir, exist_ok=True)
    enzyme_pdb = os.path.join(shared_dir, "enzyme_1823.pdb")
    print("Extracting enzyme from 1823.pdb...")
    extract_enzyme_pdb(pdb_lines, enzyme_pdb)

    # Step 2: Update SUR and SLY mol2 with 1823.pdb coordinates
    sur_1823 = os.path.join(shared_dir, "sur_1823.mol2")
    sly_1823 = os.path.join(shared_dir, "sly_1823.mol2")
    print("Updating SUR mol2...")
    update_mol2_coords(SUR_MOL2, pdb_lines, 'SUR', sur_1823)
    print("Updating SLY mol2...")
    update_mol2_coords(SLY_MOL2, pdb_lines, 'SLY', sly_1823)

    # Step 3: Build each ligand system
    for lig_name, lig_info in LIGANDS.items():
        print(f"\n{'='*60}")
        print(f"Building: lys_silica_{lig_name}")
        print('='*60)

        lig_mol2   = lig_info['mol2']
        lig_frcmod = lig_info['frcmod']

        # Compute single translate vector to move 6-molecule cluster to target center
        cx, cy, cz = mol2_center(lig_mol2)
        print(f"  Ligand mol2 center: ({cx:.2f}, {cy:.2f}, {cz:.2f})")
        tx0, ty0, tz0 = TARGET_CENTER
        translate_vec = (tx0-cx, ty0-cy, tz0-cz)

        # Create output directory
        out_dir = os.path.join(OUT_BASE, f"lys_silica_{lig_name}")
        os.makedirs(out_dir, exist_ok=True)

        # Write tleap input
        tleap_in, prmtop, inpcrd = write_tleap(
            lig_name, lig_mol2, lig_frcmod, translate_vec,
            enzyme_pdb, sur_1823, sly_1823, out_dir
        )
        tleap_out = tleap_in.replace('.in', '.out')

        # Run tleap
        print(f"  Running tleap...")
        cmd = f"conda run -n {AMBER_ENV} tleap -f {tleap_in}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        with open(tleap_out, 'w') as f:
            f.write(result.stdout + result.stderr)

        if not os.path.exists(prmtop):
            print(f"  ERROR: tleap failed. Check {tleap_out}")
            continue
        print(f"  tleap OK")

        # Convert to GROMACS using a temp script (avoids shell quoting issues)
        print(f"  Converting to GROMACS format...")
        gro  = prmtop.replace('.prmtop', '.gro')
        top  = prmtop.replace('.prmtop', '.top')
        conv_script = os.path.join(out_dir, "_convert.py")
        with open(conv_script, 'w') as cs:
            cs.write(f"""import parmed as pmd
a = pmd.load_file('{prmtop}', '{inpcrd}')
a.save('{top}', overwrite=True)
a.save('{gro}', overwrite=True)
print('atoms:', len(a.atoms))
""")
        conv = subprocess.run(
            f"conda run -n {AMBER_ENV} python3 {conv_script}",
            shell=True, capture_output=True, text=True
        )
        print(f"  {conv.stdout.strip()}")
        if conv.returncode != 0:
            print(f"  parmed error: {conv.stderr[:300]}")
            continue

        # Rename R-2 → SUR in top and gro
        for fpath in [gro, top]:
            if os.path.exists(fpath):
                with open(fpath) as f: content = f.read()
                content = content.replace('R-2', 'SUR')
                with open(fpath, 'w') as f: f.write(content)

        # Rename LYS → SLY for the lysine monolayer
        rename_sly_in_gro(gro)
        rename_sly_in_top(top)

        # Add POSRES blocks to topology
        add_posres_blocks(top)

        # Copy posres itp files
        src_posres_sur = os.path.join(LYS_DIR, "posres_SUR.itp")
        src_posres_sly = os.path.join(LYS_DIR, "posres_SLY.itp")
        shutil.copy(src_posres_sur, os.path.join(out_dir, "posres_SUR.itp"))
        shutil.copy(src_posres_sly, os.path.join(out_dir, "posres_SLY.itp"))
        print(f"  Copied posres itp files")

        # Write mdp files and run minimization
        md_dir = os.path.join(out_dir, "MD")
        os.makedirs(md_dir, exist_ok=True)
        write_mdp_files(md_dir)
        run_minimization(sys_name=f"lys_silica_{lig_name}", gro=gro, top=top,
                         out_dir=out_dir, md_dir=md_dir)

        print(f"  Done: {out_dir}")

    print("\n\nAll ligand systems built.")


def rename_sly_in_gro(gro_path):
    """Rename LYS→SLY for the lysine monolayer atoms in the .gro file.
    Monolayer LYS residues appear after UNK (ligand) and before SUR.
    """
    with open(gro_path) as f:
        lines = f.readlines()
    natoms = int(lines[1].strip())
    atom_lines = lines[2:2+natoms]

    unk_last = -1
    sur_first = natoms
    for i, line in enumerate(atom_lines):
        resname = line[5:10].strip()
        if resname == 'UNK':
            unk_last = i
        if resname == 'SUR' and sur_first == natoms:
            sur_first = i

    result = [lines[0], lines[1]]
    for i, line in enumerate(atom_lines):
        if line[5:10].strip() == 'LYS' and i > unk_last and i < sur_first:
            line = line[:5] + 'SLY  ' + line[10:]
        result.append(line)
    result.append(lines[-1])
    with open(gro_path, 'w') as f:
        f.writelines(result)
    print(f"  Renamed LYS→SLY in {gro_path}")


def rename_sly_in_top(top_path):
    """Rename LYS→SLY in the monolayer moleculetype block and [ molecules ] section."""
    with open(top_path) as f:
        lines = f.readlines()

    in_lys_moltype = False
    in_molecules = False
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect [ moleculetype ] sections
        if re.match(r'^\s*\[\s*moleculetype\s*\]', line):
            # Peek at next non-comment line for the name
            j = i + 1
            while j < len(lines) and lines[j].startswith(';'):
                j += 1
            if j < len(lines) and re.match(r'^LYS\s', lines[j].strip()):
                in_lys_moltype = True
            else:
                in_lys_moltype = False
        if re.match(r'^\s*\[\s*molecules\s*\]', line):
            in_molecules = True
            in_lys_moltype = False
        if in_lys_moltype:
            line = re.sub(r'\bLYS\b', 'SLY', line)
        elif in_molecules:
            if re.match(r'^LYS\s', line):
                line = 'SLY' + line[3:]
        result.append(line)
        i += 1

    with open(top_path, 'w') as f:
        f.writelines(result)
    print(f"  Renamed LYS→SLY in {top_path}")


def add_posres_blocks(top_path):
    """Insert #ifdef POSRES_SLY and #ifdef POSRES_SUR blocks into topology."""
    with open(top_path) as f:
        content = f.read()

    # Insert POSRES_SLY before '[ moleculetype ]\n; Name\nsystem2'
    sly_block = "#ifdef POSRES_SLY\n#include \"posres_SLY.itp\"\n#endif\n"
    marker_sys2 = "[ moleculetype ]\n; Name            nrexcl\nsystem2"
    if sly_block not in content and marker_sys2 in content:
        content = content.replace(marker_sys2, sly_block + marker_sys2)

    # Insert POSRES_SUR before '[ moleculetype ]\n; Name\nCl-'
    sur_block = "#ifdef POSRES_SUR\n#include \"posres_SUR.itp\"\n#endif\n"
    marker_cl = "[ moleculetype ]\n; Name            nrexcl\nCl-"
    if sur_block not in content and marker_cl in content:
        content = content.replace(marker_cl, sur_block + marker_cl)

    with open(top_path, 'w') as f:
        f.write(content)
    print(f"  Added POSRES blocks to topology")


def write_mdp_files(md_dir):
    """Write minim.mdp and 200ns.mdp into md_dir."""
    minim = """; Energy minimization with POSRES_SUR and POSRES_SLY
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
define          = -DPOSRES_SUR -DPOSRES_SLY
refcoord-scaling = com
"""
    mdp200 = """; 200 ns production MD — dt=2 fs, POSRES_SUR POSRES_SLY
integrator              = md
nsteps                  = 100000000
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

define                  = -DPOSRES_SUR -DPOSRES_SLY
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
    with open(os.path.join(md_dir, "minim.mdp"), 'w') as f:
        f.write(minim)
    with open(os.path.join(md_dir, "200ns.mdp"), 'w') as f:
        f.write(mdp200)
    print(f"  Wrote mdp files to {md_dir}")


def run_minimization(sys_name, gro, top, out_dir, md_dir):
    """Run grompp + mdrun for energy minimization."""
    tpr = os.path.join(md_dir, "em.tpr")
    gmx = "bash -c 'module load gromacs-2023 && gmx_mpi"

    # grompp
    grompp_cmd = (
        f"bash -c 'module load gromacs-2023 && gmx_mpi grompp "
        f"-f {md_dir}/minim.mdp -c {gro} -r {gro} -p {top} "
        f"-o {tpr} -maxwarn 5 2>&1'"
    )
    print(f"  Running grompp...")
    r = subprocess.run(grompp_cmd, shell=True, capture_output=True, text=True)
    grompp_out = r.stdout + r.stderr
    with open(os.path.join(md_dir, "grompp.log"), 'w') as f:
        f.write(grompp_out)
    if not os.path.exists(tpr):
        print(f"  ERROR: grompp failed. Check {md_dir}/grompp.log")
        return
    print(f"  grompp OK")

    # mdrun
    em_log = os.path.join(md_dir, "em_run.log")
    mdrun_cmd = (
        f"bash -c 'module load gromacs-2023 && gmx_mpi mdrun "
        f"-v -deffnm {md_dir}/em 2>&1 | tee {em_log}'"
    )
    print(f"  Running mdrun (minimization)...")
    r = subprocess.run(mdrun_cmd, shell=True, capture_output=True, text=True, timeout=3600)
    # Extract convergence line
    for line in (r.stdout + r.stderr).split('\n'):
        if 'converged' in line.lower() or 'Potential Energy' in line or 'Fmax' in line:
            print(f"  {line.strip()}")
    if os.path.exists(os.path.join(md_dir, "em.gro")):
        print(f"  Minimization complete: {md_dir}/em.gro")
    else:
        print(f"  WARNING: em.gro not found — check {em_log}")


if __name__ == "__main__":
    main()
