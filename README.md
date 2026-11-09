# LAC_surface_adsorption

**Understanding the Role of Surface Chemistry in Substrate-Binding Pathways and Kinetics of Adsorbed Fungal Laccase**

Anushka Biswas† and Mithun Radhakrishna\*,‡,¶

†Department of Chemical Engineering, IIT Gandhinagar, Palaj, Gujarat 382355, India  
‡Department of Chemical Engineering and Center for Biomedical Engineering, IIT Gandhinagar  
¶Center for Biomedical Engineering, IIT Gandhinagar  
E-mail: mithunr@iitgn.ac.in

---

## Overview

This repository contains simulation data, analysis scripts, and force-field parameters supporting the manuscript above. We investigate how three chemically distinct silica surface functionalizations — bare silica, 3-aminopropyltriethoxysilane (APTES), and poly(L-lysine) — alter the adsorption orientation, structural dynamics, and substrate-binding kinetics of *Trametes versicolor* laccase (TvL) at pH 5.

**Key findings:**
- APTES-silica gives the deepest adsorption free-energy minimum (~−427 kJ mol⁻¹)
- The active-site gating mechanism of TvL is preserved after adsorption on all three surfaces
- APTES yields the shortest binding and longest unbinding times for all 9 substrates that bound
- For 4 phenolic substrates, binding times fell by factors of 8–13 relative to solution
- Catechol MFPT: 0.49 µs (solution), 0.90 µs (APTES), 0.23 µs (poly-L-lysine)

Nine substrates studied: catechol, pyrogallol, hydroxyquinol, hydroquinone, resorcinol, phloroglucinol, phenol, aniline, 2,5-xylidine.

---

## Repository Structure

```
LAC_surface_adsorption/
├── surfaces/
│   ├── silica/                    # Bare silica surface (CHARMM-GUI + parmed pipeline)
│   ├── aptes/
│   │   ├── build_aptes_surface.py     # Full APTES monolayer construction pipeline
│   │   └── parameters/
│   │       ├── butylammonium.mol2     # AMBER GAFF2 parameters for APTES
│   │       └── butylammonium.frcmod
│   ├── lysine/
│   │   ├── build_lys_silica.py        # L-lysine monolayer on silica
│   │   └── build_lys_silica_ligands.py
│   └── poly_lysine/
│       ├── polylys_contacts.py        # Contact analysis: enzyme residues vs. SLY brush
│       ├── rename_lys_to_sly.py       # Rename grafted LYS → SLY in GRO/TOP
│       └── renumber_sly_chains.py     # Renumber SLY chains 1–49 for VMD selection
├── md_scripts/
│   ├── mdp/
│   │   ├── minim.mdp                  # Energy minimization (steepest descent, POSRES_SUR)
│   │   └── prod_200ns.mdp             # Production NPT, 200 ns, V-rescale + C-rescale
│   └── setup/
│       └── orient.py                  # Laccase orientation analysis and placement
├── msm/
│   └── scripts/
│       ├── extract_features.py        # TICA/torsion feature extraction
│       ├── build_msm.py               # MSM construction (deeptime)
│       ├── compute_kinetics.py        # MFPT, rate constants from MSM
│       ├── run_tpt.py                 # Transition path theory analysis
│       ├── plot_free_energy.py        # Free-energy landscape from stationary distribution
│       ├── plot_reweighted_tica.py    # Reweighted TICA projections
│       ├── compare_free_energy.py     # Cross-surface free-energy comparison
│       └── run_all_ligands.py         # Batch driver across all substrates
├── umbrella_sampling/
│   ├── setup_windows.py               # Window generation along enzyme–surface distance
│   └── analysis/
│       ├── run_wham_and_plot_pmf.py   # WHAM + PMF plotting
│       └── plot_pmf_comparison.py     # Cross-surface PMF overlay
└── analysis/
    ├── conformational_analysis.py     # S3-loop RMSF and conformation classification
    ├── hbond_rmsf_analysis.py         # H-bond occupancy + RMSF correlation
    ├── compute_msd.py                 # Ligand mean-squared displacement
    └── state_disappearance_analysis.py   # MSM state occupancy vs. surface
```

---

## Simulation Systems

| Surface | Functionalization | Total charge (pH 5) |
|---|---|---|
| Bare silica | –OH / –O⁻ | neutral |
| APTES-silica | –(CH₂)₃–NH₃⁺ | +102 *e* (102 chains) |
| Poly-Lys-silica | 8-mer poly-L-lysine brush (SLY) | +392 *e* (49 chains) |

**Enzyme:** *Trametes versicolor* laccase (PDB: 1GYC), placed in orient1 (copper T1 site facing away from surface).  
**pH:** 5 (ionizable residues assigned with PROPKA 3.5; copper centres parameterized with MCPB.py).  
**Force fields:** AMBER ff19SB (protein/poly-Lys), GAFF2 (ligands, APTES), SPC/E (water).

---

## Software

| Task | Software |
|---|---|
| Surface & system building | CHARMM-GUI, AmberTools 23 (tleap, MCPB.py, antechamber), PACKMOL |
| MD simulations | GROMACS 2024.3-plumed (`gmx_mpi`) |
| MSM construction | deeptime 0.4, MDAnalysis 2.x, Python 3.10 |
| Umbrella sampling / PMF | GROMACS pull code + WHAM |
| Visualisation | VMD 1.9.4, UCSF Chimera |

---

## Usage

### 1. Build surfaces

```bash
conda activate AmberTools23

# APTES monolayer
python surfaces/aptes/build_aptes_surface.py

# L-lysine monolayer
python surfaces/lysine/build_lys_silica.py
```

### 2. MD simulation

```bash
# Energy minimization
gmx_mpi grompp -f md_scripts/mdp/minim.mdp -c system.gro -p topol.top -o em.tpr -maxwarn 1
gmx_mpi mdrun -v -s em.tpr -deffnm em

# Production (200 ns NPT)
gmx_mpi grompp -f md_scripts/mdp/prod_200ns.mdp -c em.gro -p topol.top -o prod.tpr -maxwarn 1
gmx_mpi mdrun -v -s prod.tpr -deffnm prod
```

### 3. Build MSMs and compute kinetics

```bash
conda activate laccase-msm

# Extract TICA features, build MSM, compute MFPT for one surface/ligand
python msm/scripts/extract_features.py --surface aptes --ligand catechol
python msm/scripts/build_msm.py --surface aptes --ligand catechol
python msm/scripts/compute_kinetics.py --surface aptes --ligand catechol

# Run for all substrates across all surfaces
python msm/scripts/run_all_ligands.py
```

### 4. Umbrella sampling (PMF)

```bash
python umbrella_sampling/setup_windows.py
python umbrella_sampling/analysis/run_wham_and_plot_pmf.py
python umbrella_sampling/analysis/plot_pmf_comparison.py
```

---

## Data Availability

Trajectory files (`.xtc`, `.trr`) and large binary files are not tracked in this repository. Representative structures and analysis output CSVs are available from the authors on request.

---

## Citation

If you use these scripts or data, please cite:

> Biswas A. and Radhakrishna M. *Understanding the Role of Surface Chemistry in Substrate-Binding Pathways and Kinetics of Adsorbed Fungal Laccase.* (manuscript in preparation, 2025)

---

## Contact

Mithun Radhakrishna — mithunr@iitgn.ac.in  
Anushka Biswas — anushkabiswas@iitgn.ac.in  
IIT Gandhinagar, Palaj, Gujarat 382355, India
