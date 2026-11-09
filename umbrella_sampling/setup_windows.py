#!/usr/bin/env python3
"""
Extract umbrella sampling window configurations from pull trajectories
and set up per-window directories with grompp input.

Usage: python3 setup_windows.py [silica|lys_silica|aptes_silica|all]
"""

import os
import sys
import shutil
import subprocess
import numpy as np

US_BASE = "/Data/Anushka/Project4/ph5/orients/orient1/US"

SYSTEMS = {
    "silica": {
        "top":      "laccase_flipped_silica.top",
        "posres":   ["posres_SUR.itp"],
        "index":    "index.ndx",
        "win_min":  1.0,    # nm — Interface COM near surface
        "win_max":  5.5,    # nm — Interface COM in bulk
        "win_step": 0.25,   # nm — 2.5 Å spacing
        "init_z":   1.561,  # initial Interface_COM - SUR_COM (nm)
        "rate":     0.001,  # nm/ps, positive = pulling away
    },
    "lys_silica": {
        "top":      "lys_silica.top",
        "posres":   ["posres_SUR.itp", "posres_SLY.itp"],
        "index":    "index.ndx",
        "win_min":  3.5,
        "win_max":  8.5,
        "win_step": 0.25,   # 2.5 Å spacing for histogram overlap
        "init_z":   3.87,
        "rate":     0.001,
    },
    "aptes_silica": {
        "top":      "laccase_aptes.top",
        "posres":   ["posres_SUR.itp"],
        "index":    "index.ndx",
        "win_min":  3.5,
        "win_max":  10.5,
        "win_step": 0.25,   # 2.5 Å spacing for histogram overlap
        "init_z":   10.57,
        "rate":     -0.001,
    },
}


def run(cmd, cwd=None, check=True, capture=True):
    r = subprocess.run(cmd, shell=True, cwd=cwd,
                       capture_output=capture, text=True,
                       executable='/bin/bash')
    if check and r.returncode != 0:
        print(f"ERROR: {cmd}")
        print(r.stderr[-1000:])
        raise RuntimeError(f"Command failed (rc={r.returncode})")
    return r


def read_pullx(pullx_file):
    """Read pullx.xvg and return (time_ps, coord_nm) arrays."""
    times, coords = [], []
    with open(pullx_file) as f:
        for line in f:
            if line.startswith('#') or line.startswith('@'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                times.append(float(parts[0]))
                coords.append(float(parts[1]))
    return np.array(times), np.array(coords)


def find_frame_time(times, coords, target_coord):
    """Find time (ps) closest to target_coord in pullx."""
    idx = np.argmin(np.abs(coords - target_coord))
    return times[idx], coords[idx]


GMX = "/apps/codes/gromacs-2023/bin/gmx_mpi"


def extract_frame(traj, tpr, out_gro, time_ps, cwd):
    """Extract single frame from trajectory at time_ps."""
    run(f'echo "0" | {GMX} trjconv '
        f'-f {traj} -s {tpr} -o {out_gro} -dump {time_ps:.3f} -pbc whole',
        cwd=cwd)


def setup_window(surf, win_idx, target_dist, frame_gro, cfg):
    """Create window directory and grompp input."""
    surf_dir  = os.path.join(US_BASE, surf)
    win_dir   = os.path.join(surf_dir, "windows", f"win_{win_idx:03d}")
    os.makedirs(win_dir, exist_ok=True)

    # Copy start.gro for this window
    shutil.copy2(frame_gro, os.path.join(win_dir, "conf.gro"))

    # Copy topology and posres
    for f in [cfg["top"]] + cfg["posres"]:
        src = os.path.join(surf_dir, "input", f)
        shutil.copy2(src, os.path.join(win_dir, f))

    # Copy and patch window.mdp: replace WINDOW_DIST with actual target distance
    mdp_src = os.path.join(surf_dir, "windows", "window.mdp")
    mdp_dst = os.path.join(win_dir, "window.mdp")
    with open(mdp_src) as f:
        content = f.read()
    content = content.replace("WINDOW_DIST", f"{target_dist:.4f}")
    with open(mdp_dst, 'w') as f:
        f.write(content)

    # Grompp
    index = os.path.join(surf_dir, "input", cfg["index"])
    top   = os.path.join(win_dir, cfg["top"])
    cmd = (
        f'{GMX} grompp '
        f'-f window.mdp -c conf.gro -r conf.gro '
        f'-p {cfg["top"]} -n {index} '
        f'-o window.tpr -maxwarn 1'
    )
    r = run(cmd, cwd=win_dir, check=False)
    if r.returncode != 0:
        print(f"  WARN: grompp failed for win_{win_idx:03d} ({target_dist:.2f} nm)")
        return False

    # Write per-window run script
    with open(os.path.join(win_dir, "run_window.sh"), 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=us_{surf}_w{win_idx:03d}
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=16:00:00

module load gromacs-2023
cd {win_dir}
mpirun -np 16 gmx_mpi mdrun \\
    -v -deffnm window \\
    -s window.tpr \\
    -px pullx.xvg \\
    -pf pullf.xvg 2>&1 | tee mdrun.log
echo "Window {win_idx} done: $(date)"
""")
    os.chmod(os.path.join(win_dir, "run_window.sh"), 0o755)
    return True


def process_surface(surf):
    cfg = SYSTEMS[surf]
    surf_dir  = os.path.join(US_BASE, surf)
    pull_dir  = os.path.join(surf_dir, "pull")
    wins_dir  = os.path.join(surf_dir, "windows")

    pullx_file = os.path.join(pull_dir, "pullx.xvg")
    pull_tpr   = os.path.join(pull_dir, "pull.tpr")
    pull_traj  = os.path.join(pull_dir, "pull.xtc")
    frames_dir = os.path.join(wins_dir, "_frames")
    os.makedirs(frames_dir, exist_ok=True)

    if not os.path.exists(pullx_file):
        print(f"[{surf}] pullx.xvg not found — run pull simulation first")
        return

    print(f"\n[{surf}] Reading pull coordinates...")
    times, coords = read_pullx(pullx_file)
    print(f"  Pull trajectory: {times[0]:.0f} to {times[-1]:.0f} ps, "
          f"coord: {coords[0]:.3f} to {coords[-1]:.3f} nm")

    targets = np.arange(cfg["win_min"], cfg["win_max"] + 0.001, cfg["win_step"])
    print(f"  Target windows: {targets[0]:.2f} to {targets[-1]:.2f} nm "
          f"({len(targets)} windows, spacing {cfg['win_step']} nm)")

    results = []
    for i, tgt in enumerate(targets):
        t, actual = find_frame_time(times, coords, tgt)
        frame_gro = os.path.join(frames_dir, f"frame_{i:03d}.gro")

        print(f"  Window {i:03d}: target={tgt:.2f} nm, "
              f"frame at t={t:.0f} ps (actual coord={actual:.3f} nm)")

        if abs(actual - tgt) > 0.15:
            print(f"  WARN: frame coord {actual:.3f} is >0.15 nm from target {tgt:.2f}")

        # Skip window if already completed (pullf.xvg has full data)
        win_dir = os.path.join(wins_dir, f"win_{i:03d}")
        pullf_done = os.path.join(win_dir, "pullf.xvg")
        if os.path.exists(pullf_done):
            nframes = sum(1 for l in open(pullf_done) if not l.startswith(('#','@')))
            if nframes >= 19900:
                print(f"  SKIP win_{i:03d} — already complete ({nframes} frames)")
                results.append((i, tgt, True))
                continue

        # Extract frame
        if not os.path.exists(frame_gro):
            extract_frame(pull_traj, pull_tpr, frame_gro, t, pull_dir)

        # Set up window directory
        ok = setup_window(surf, i, tgt, frame_gro, cfg)
        results.append((i, tgt, ok))

    print(f"\n[{surf}] Window setup summary:")
    for i, tgt, ok in results:
        print(f"  {'OK' if ok else 'ERR'}  win_{i:03d}  {tgt:.2f} nm")

    # Write batch submit script
    submit_script = os.path.join(wins_dir, "submit_all_windows.sh")
    with open(submit_script, 'w') as f:
        f.write("#!/bin/bash\n# Submit all US window jobs\n")
        for i, tgt, ok in results:
            if ok:
                win_dir = os.path.join(wins_dir, f"win_{i:03d}")
                f.write(f"sbatch {win_dir}/run_window.sh\n")
    os.chmod(submit_script, 0o755)
    print(f"  Submit all: {submit_script}")


if __name__ == "__main__":
    to_run = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    if "all" in to_run:
        to_run = list(SYSTEMS.keys())

    for surf in to_run:
        if surf not in SYSTEMS:
            print(f"Unknown surface: {surf}. Options: {list(SYSTEMS.keys())}")
        else:
            process_surface(surf)
