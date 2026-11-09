import numpy as np
import os
import subprocess

protein_pdb = "LAC_mcpbpy.pdb"
surface_pdb = "silica_flipped.pdb"

gap = 20.0            # protein distance above surface
surface_top = 5.6    # adjust after checking slab height
z_protein = surface_top + gap

output_dir = "systems"
packmol_bin = "packmol"

# (θx, θy, name)
angles = [
    (0, 0, "initial"),
    (90, 0, "tilt_x90"),
    (180, 0, "tilt_x180"),
    (270, 0, "tilt_x270"),
    (0, 90, "tilt_y90"),
    (0, 270, "tilt_y270")
]

def rotate_pdb(pdbfile, tilt_x, tilt_y, outfile):
    """Rotate PDB around its centroid."""

    θx = np.radians(tilt_x)
    θy = np.radians(tilt_y)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(θx), -np.sin(θx)],
                   [0, np.sin(θx),  np.cos(θx)]])

    Ry = np.array([[ np.cos(θy), 0, np.sin(θy)],
                   [0, 1, 0],
                   [-np.sin(θy), 0, np.cos(θy)]])

    R = Ry @ Rx

    coords = []

    with open(pdbfile) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x,y,z])

    coords = np.array(coords)
    centroid = coords.mean(axis=0)

    with open(pdbfile) as f, open(outfile,"w") as out:
        for line in f:
            if line.startswith(("ATOM","HETATM")):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                shifted = np.array([x,y,z]) - centroid
                rotated = R @ shifted + centroid

                line = f"{line[:30]}{rotated[0]:8.3f}{rotated[1]:8.3f}{rotated[2]:8.3f}{line[54:]}"
            out.write(line)

def write_packmol_input(protein_file, output_file, packmol_inp):

    with open(packmol_inp,"w") as f:
        f.write(f"""tolerance 2.0
filetype pdb
output {output_file}
seed 12345

# Protein placed above silica surface
structure {protein_file}
  number 1
  fixed 1. 0. {z_protein} 0. 0. 0.
end structure

# Silica surface (single layer)
structure {surface_pdb}
  number 1
  fixed 0. 0. 0. 0. 0. 0.
end structure
""")

def main():

    os.makedirs(output_dir,exist_ok=True)

    for i,(θx,θy,name) in enumerate(angles,start=1):

        rot_file = f"{output_dir}/LAC_rot{i}.pdb"
        system_file = f"{output_dir}/system{i}.pdb"
        packmol_inp = f"{output_dir}/packmol{i}.inp"

        print(f"[INFO] Rotating protein {name} (x={θx}, y={θy})")

        rotate_pdb(protein_pdb,θx,θy,rot_file)

        print(f"[INFO] Writing Packmol input")

        write_packmol_input(rot_file,system_file,packmol_inp)

        print(f"[INFO] Running Packmol")

        subprocess.run(
            [packmol_bin],
            input=open(packmol_inp).read(),
            text=True,
            capture_output=True
        )

    print(f"[DONE] Generated {len(angles)} systems in '{output_dir}'")

if __name__ == "__main__":
    main()
