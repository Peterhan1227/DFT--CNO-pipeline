"""
combine_cno_cubes.py — post-processing visualization helper.

Combines already-exported individual CNO density cube files into a single
VESTA-readable cube file. It does NOT read WAVECAR, does NOT load
cno_orbitals.npy, and does NOT recompute CNOs. It only reads existing .cube
files and sums their scalar volumetric data. This is useful when VESTA
multi-phase overlay of many separate cube files is inconvenient.
"""
import numpy as np
from pathlib import Path
from config import MATERIAL, OUTPUT_SUBDIR

# ── paths ─────────────────────────────────────────────────────────────────────
base_dir = Path(__file__).resolve().parent
cube_dir = base_dir / "Data" / MATERIAL / "output" / OUTPUT_SUBDIR / "cno_cubes"

# ── user settings ─────────────────────────────────────────────────────────────
# Number of CNO cube files to combine (cno_000 through cno_{n_combine-1}).
n_combine = 7

# If True, normalize each input cube so its scalar grid sums to 1 before
# summing. Since individual CNO cubes from export_cno_cubes.py are already
# normalized, False is the correct choice under normal use.
normalize_each_input = False


# ── cube reader ───────────────────────────────────────────────────────────────
def read_cube(path):
    """Read a Gaussian .cube file.

    Returns
    -------
    header_lines : list of str — all lines from comment 1 through the last
                   atom line (each string includes its trailing newline).
    grid_shape   : (Nx, Ny, Nz) tuple of int
    data_grid    : ndarray shape (Nx, Ny, Nz), dtype float64
    """
    with open(path) as fh:
        lines = fh.readlines()

    # line 2: natoms (may be negative — sign encodes units, abs gives atom count)
    natoms      = int(lines[2].split()[0])
    n_atom_lines = abs(natoms)

    # lines 3-5: grid dimensions (first token is the count)
    Nx = int(lines[3].split()[0])
    Ny = int(lines[4].split()[0])
    Nz = int(lines[5].split()[0])

    # header = comments + origin + 3 grid-axis lines + atom lines
    n_header    = 6 + n_atom_lines
    header_lines = lines[:n_header]

    # parse volumetric data from the remaining lines
    values = []
    for line in lines[n_header:]:
        values.extend(float(x) for x in line.split())

    values   = np.array(values, dtype=float)
    expected = Nx * Ny * Nz
    if len(values) != expected:
        raise ValueError(
            f"Expected {expected} values ({Nx}×{Ny}×{Nz}) in '{path.name}', "
            f"but found {len(values)}."
        )

    data_grid = values.reshape((Nx, Ny, Nz))
    return header_lines, (Nx, Ny, Nz), data_grid


# ── cube writer ───────────────────────────────────────────────────────────────
def write_cube(path, header_lines, data_grid, comment1=None, comment2=None):
    """Write a Gaussian .cube file.

    Parameters
    ----------
    header_lines : list of str from read_cube (includes trailing newlines).
    data_grid    : ndarray shape (Nx, Ny, Nz).
    comment1     : if not None, replaces header line 0.
    comment2     : if not None, replaces header line 1.
    """
    Nx, Ny, Nz = data_grid.shape

    hdr = list(header_lines)
    if comment1 is not None:
        hdr[0] = comment1 + "\n"
    if comment2 is not None:
        hdr[1] = comment2 + "\n"

    with open(path, "w") as fh:
        for line in hdr:
            fh.write(line)

        # volumetric data — 6 values per line, scientific notation
        count = 0
        for ix in range(Nx):
            for iy in range(Ny):
                for iz in range(Nz):
                    fh.write(f"  {data_grid[ix, iy, iz]:13.6e}")
                    count += 1
                    if count % 6 == 0:
                        fh.write("\n")
        if count % 6 != 0:
            fh.write("\n")


# ── main logic ────────────────────────────────────────────────────────────────
cube_paths = [
    cube_dir / f"cno_{i:03d}_density.cube"
    for i in range(n_combine)
]

# verify all input files exist before doing any work
missing = [p for p in cube_paths if not p.exists()]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} input cube file(s):\n"
        + "\n".join(f"  {p}" for p in missing)
        + f"\nRun export_cubes.py first, or reduce n_combine (currently {n_combine})."
    )

print(f"Combining top {n_combine} CNO cube files:")
for p in cube_paths:
    print(f"  {p.name}")

# read the first cube to get header and reference shape
header0, shape0, grid0 = read_cube(cube_paths[0])
combined = np.zeros_like(grid0)

for p in cube_paths:
    header, shape, grid = read_cube(p)
    if shape != shape0:
        raise ValueError(
            f"Grid shape mismatch: '{cube_paths[0].name}' has {shape0} "
            f"but '{p.name}' has {shape}."
        )
    if normalize_each_input:
        s = grid.sum()
        if s <= 0:
            raise ValueError(
                f"Input cube '{p.name}' has non-positive sum ({s}); "
                "cannot normalize."
            )
        grid = grid / s
    combined += grid

print(f"Grid shape: {shape0}")
print(f"Combined density sum before normalization: {combined.sum():.8f}")

total = combined.sum()
if total <= 0:
    raise ValueError("Combined density has non-positive sum; cannot normalize.")
combined = combined / total

# ── write output cube ─────────────────────────────────────────────────────────
out_path = cube_dir / f"combined_top_{n_combine:03d}_cno_density.cube"
write_cube(
    out_path,
    header0,
    combined,
    comment1=f"Combined density of top {n_combine} CNOs",
    comment2="Scalar field = normalized sum_i |CNO_i(r)|^2",
)
print(f"Wrote combined cube to: {out_path}")

# ── metadata ──────────────────────────────────────────────────────────────────
meta_path = cube_dir / f"combined_top_{n_combine:03d}_cno_density_metadata.txt"
with open(meta_path, "w") as fh:
    fh.write(f"n_combine                 : {n_combine}\n")
    fh.write(f"normalize_each_input      : {normalize_each_input}\n")
    fh.write(f"grid shape (Nx, Ny, Nz)  : {shape0}\n")
    fh.write(f"output cube               : {out_path}\n")
    fh.write("input cube files          :\n")
    for p in cube_paths:
        fh.write(f"  {p.name}\n")
    fh.write(
        "NOTE: this is a combined scalar density, not separately colored CNOs\n"
    )
    fh.write(
        "NOTE: VESTA will show one isosurface representing the spatial "
        "support of the selected CNOs\n"
    )
print(f"Wrote metadata to: {meta_path}")
