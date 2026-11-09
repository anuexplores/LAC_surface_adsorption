#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Tuple
import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import distance_array

def min_distance(a, b, box):
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.min(distance_array(a.positions, b.positions, box=box)))

def com_distance(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.linalg.norm(a.center_of_mass() - b.center_of_mass()))

def ligand_instances(u, resname):
    lig = u.select_atoms(f"resname {resname} and not name H*")
    residues = list(lig.residues)
    if not residues:
        raise ValueError(f"No ligand residues found for resname={resname!r}")
    return [r.atoms.select_atoms("not name H*") for r in residues]

def protein_residue_group(u, protein_selection, resid):
    g = u.select_atoms(f"({protein_selection}) and resid {resid} and not name H*")
    if len(g) == 0:
        raise ValueError(f"Residue {resid} returned zero atoms")
    return g

def residue_list_group(u, protein_selection, resids):
    s = " ".join(map(str, resids))
    g = u.select_atoms(f"({protein_selection}) and resid {s} and not name H*")
    if len(g) == 0:
        raise ValueError(f"Residue group {resids} returned zero atoms")
    return g

def extract_one(topology, trajectory, ligand_resname, protein_selection,
                active_site_resids, t1_copper_selection,
                loop_gate_group1, loop_gate_group2, stride):
    if not Path(topology).is_file():
        raise FileNotFoundError(topology)
    if not Path(trajectory).is_file():
        raise FileNotFoundError(trajectory)

    u = mda.Universe(topology, trajectory)

    residue_groups = [
        protein_residue_group(u, protein_selection, r)
        for r in active_site_resids
    ]

    t1 = u.select_atoms(t1_copper_selection)
    if len(t1) != 1:
        raise ValueError(
            f"T1 copper selection {t1_copper_selection!r} returned {len(t1)} atoms; "
            "it must select exactly one T1 Cu atom."
        )

    gate1 = residue_list_group(u, protein_selection, loop_gate_group1)
    gate2 = residue_list_group(u, protein_selection, loop_gate_group2)
    ligands = ligand_instances(u, ligand_resname)

    feature_names = (
        [f"dmin_resid_{r}_A" for r in active_site_resids]
        + ["dmin_T1Cu_A", "loop_gate_COM_A"]
    )

    rows = [[] for _ in ligands]
    frame_indices = []

    for ts in u.trajectory[::stride]:
        frame_indices.append(int(ts.frame))
        gate = com_distance(gate1, gate2)
        for i, lig in enumerate(ligands):
            vals = [min_distance(lig, rg, ts.dimensions) for rg in residue_groups]
            vals.append(min_distance(lig, t1, ts.dimensions))
            vals.append(gate)
            rows[i].append(vals)

    arrays = [np.asarray(x, dtype=np.float32) for x in rows]
    meta = []
    for i, lig in enumerate(ligands, start=1):
        r = lig.residues[0]
        meta.append({
            "ligand_index_within_replica": i,
            "ligand_resname": ligand_resname,
            "ligand_resid": int(r.resid),
            "ligand_segid": str(r.segid),
            "topology": str(topology),
            "trajectory": str(trajectory),
            "n_saved_frames": len(frame_indices)
        })
    return arrays, meta, feature_names

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--system", default="unknown")
    p.add_argument("--ligand", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    if args.ligand not in cfg["ligands"]:
        raise KeyError(f"{args.ligand!r} not found in config['ligands']")
    ligcfg = cfg["ligands"][args.ligand]

    loop = cfg["loop_gate_definition"]
    if loop.get("type") != "COM_distance":
        raise ValueError("Only COM_distance loop gate is supported")

    all_features, all_meta = [], []
    names = None

    for rep_idx, rep in enumerate(ligcfg["replicates"], start=1):
        feats, meta, names = extract_one(
            rep["topology"], rep["trajectory"], ligcfg["resname"],
            cfg["protein_selection"], cfg["active_site_resids"],
            cfg["t1_copper_selection"], loop["group1"], loop["group2"],
            int(cfg.get("stride", 1))
        )
        for m in meta:
            m["system"] = args.system
            m["ligand_name"] = args.ligand
            m["physical_replicate"] = rep_idx
        all_features.extend(feats)
        all_meta.extend(meta)

    lengths = np.asarray([len(x) for x in all_features], dtype=np.int64)
    X = np.concatenate(all_features, axis=0)
    if not np.all(np.isfinite(X)):
        raise RuntimeError("Non-finite values detected in features")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        features=X,
        lengths=lengths,
        feature_names=np.asarray(names, dtype=object),
        metadata_json=np.asarray(json.dumps(all_meta), dtype=object)
    )

    print(f"Wrote: {out}")
    print(f"Physical replicates: {len(ligcfg['replicates'])}")
    print(f"Ligand trajectories: {len(all_features)}")
    print(f"Total ligand-frames: {len(X)}")
    print(f"Features per frame: {len(names)}")
    for i, n in enumerate(names, 1):
        print(f"{i:2d}. {n}")

if __name__ == "__main__":
    main()
