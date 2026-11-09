#!/usr/bin/env python3
"""
Rename polylysine (system2) residues from LYS → SLY in:
  - polylys_silica.gro   (polyLysine/ and MD/)
  - polylys_silica.top   (polyLysine/ and MD/)
  - em.gro, 100ns.gro    (MD/)
  - index.ndx            (regenerated after rename)

system2 atoms: 7360–15983  (49 chains × 176 atoms)
"""
import re, subprocess
from pathlib import Path

BASE = Path("/Data/Anushka/Project4/anushka@10.0.66.94/P4/orient1/ph5/lysine/polyLysine")
MD   = BASE / "MD"
GMX  = "/apps/codes/gromacs-2024.3-plumed/bin/gmx_mpi"

SLY_START = 7360   # 1-based atom index (first atom of system2)
SLY_END   = 15983  # last atom of system2

# ── GRO renamer ──────────────────────────────────────────────────────────────
def rename_gro(path):
    path = Path(path)
    if not path.exists():
        print(f"  SKIP (not found): {path}"); return
    with open(path) as f: lines = f.readlines()
    changed = 0
    for i, line in enumerate(lines):
        if i < 2 or i >= len(lines) - 1: continue  # skip header, natoms, box
        atom_i = i - 1  # 1-based
        if SLY_START <= atom_i <= SLY_END and len(line) > 10 and line[5:8] == 'LYS':
            lines[i] = line[:5] + 'SLY' + line[8:]
            changed += 1
    with open(path, 'w') as f: f.writelines(lines)
    print(f"  {path.name}: {changed} atoms renamed LYS→SLY")

# ── TOP renamer ───────────────────────────────────────────────────────────────
def rename_top(path):
    path = Path(path)
    if not path.exists():
        print(f"  SKIP (not found): {path}"); return
    with open(path) as f: lines = f.readlines()

    in_sys2_mol  = False   # inside system2 moleculetype block
    past_molname = False
    in_atoms_sec = False
    in_mols_sec  = False

    for i, line in enumerate(lines):
        s = line.strip()
        # track moleculetype sections
        if '[ moleculetype ]' in line:
            in_sys2_mol = False; past_molname = False; in_atoms_sec = False
        elif not past_molname and s and not s.startswith(';'):
            past_molname = True
            in_sys2_mol = s.split()[0] == 'system2'
        # track [ atoms ] inside system2
        if in_sys2_mol and '[ atoms ]' in line:
            in_atoms_sec = True
        elif in_atoms_sec and s.startswith('[') and 'atoms' not in s:
            in_atoms_sec = False
        elif in_atoms_sec and re.match(r'^\s*\d+\s+\w', line):
            # column 4 (0-indexed after split) is residue name
            parts = line.split()
            if len(parts) > 3 and parts[3] == 'LYS':
                parts[3] = 'SLY'
                lines[i] = '  '.join(parts) + '\n'
        # [ molecules ] — no rename needed (system2 name stays)
    with open(path, 'w') as f: f.writelines(lines)
    print(f"  {path.name}: system2 residues renamed LYS→SLY")

# ── Run ───────────────────────────────────────────────────────────────────────
print("=== Renaming LYS → SLY in GRO files ===")
for gro in [BASE/"polylys_silica.gro", MD/"polylys_silica.gro",
            MD/"em.gro", MD/"100ns.gro"]:
    rename_gro(gro)

print("\n=== Renaming LYS → SLY in TOP files ===")
for top in [BASE/"polylys_silica.top", MD/"polylys_silica.top"]:
    rename_top(top)

print("\n=== Regenerating index.ndx ===")
result = subprocess.run(
    [GMX, "make_ndx", "-f", str(MD/"em.gro"), "-o", str(MD/"index.ndx")],
    input="q\n", capture_output=True, text=True, cwd=str(MD))
groups = [l for l in result.stdout.split('\n') if l.strip().startswith('[') or
          (result.stderr and '[' in l)]
# parse from stderr (make_ndx prints to stderr)
groups = [l for l in result.stderr.split('\n') if '[ ' in l]
print("  Groups:", groups[-5:] if len(groups) > 5 else groups)
print(f"  Written: {MD/'index.ndx'}")

print("\nDone. Verify with:")
print(f"  grep SLY {MD/'em.gro'} | head -3")
print(f"  grep 'SLY' {MD/'index.ndx'}")
