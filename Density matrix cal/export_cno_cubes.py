import numpy as np
from pathlib import Path
from config import MATERIAL

# ── element → atomic number lookup ────────────────────────────────────────────
_ATOMIC_NUMBERS = {
    "H": 1,  "He": 2,  "Li": 3,  "Be": 4,  "B": 5,   "C": 6,   "N": 7,
    "O": 8,  "F": 9,   "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14,
    "P": 15, "S": 16,  "Cl": 17, "Ar": 18, "K": 19,  "Ca": 20, "Cu": 29,
    "Sb": 51,"Pt": 78, "Te": 52, "Co": 27, "Sn": 50
}


def read_poscar_structure(poscar_path):
    """Parse a VASP POSCAR and return structure data.

    Returns
    -------
    latvec      : (3, 3) float array — lattice vectors in Angstrom
    species     : list of element symbol strings
    counts      : list of int atom counts per species
    atom_symbols: flat list of element symbol per atom
    atom_numbers: flat list of atomic number per atom
    cart_coords : (natoms, 3) float array — Cartesian coordinates in Angstrom
    """
    with open(poscar_path) as fh:
        lines = fh.readlines()

    scale  = float(lines[1])
    latvec = scale * np.array([
        [float(x) for x in lines[2].split()],
        [float(x) for x in lines[3].split()],
        [float(x) for x in lines[4].split()],
    ])

    species = lines[5].split()
    counts  = [int(x) for x in lines[6].split()]
    natoms  = sum(counts)

    atom_symbols = []
    for sym, cnt in zip(species, counts):
        atom_symbols.extend([sym] * cnt)

    atom_numbers = []
    for sym in atom_symbols:
        if sym not in _ATOMIC_NUMBERS:
            raise ValueError(
                f"Unknown element symbol '{sym}' in POSCAR. "
                "Add it to _ATOMIC_NUMBERS in this script."
            )
        atom_numbers.append(_ATOMIC_NUMBERS[sym])

    # Line 7 is optional Selective dynamics; skip it if present
    coord_line_idx = 7
    if lines[7].strip().lower().startswith("s"):
        coord_line_idx = 8

    coord_mode  = lines[coord_line_idx].strip().lower()
    coord_lines = lines[coord_line_idx + 1 : coord_line_idx + 1 + natoms]
    raw_coords  = np.array([[float(x) for x in ln.split()[:3]] for ln in coord_lines])

    if coord_mode.startswith("d"):          # Direct / fractional
        cart_coords = raw_coords @ latvec
    else:                                   # Cartesian
        cart_coords = scale * raw_coords

    return latvec, species, counts, atom_symbols, atom_numbers, cart_coords


# ── paths ─────────────────────────────────────────────────────────────────────
base_dir    = Path(__file__).resolve().parent
data_dir    = base_dir / "Data" / MATERIAL
output_dir  = data_dir / "output"
poscar_path = data_dir / "POSCAR"
occ_file        = output_dir / "cno_occupations.npy"
orb_file        = output_dir / "cno_orbitals.npy"
grid_shape_file = output_dir / "fft_grid_shape.npy"
cube_dir        = output_dir / "cno_cubes"
cube_dir.mkdir(parents=True, exist_ok=True)

# Emergency/debug override. Set to (Nx, Ny, Nz) only if fft_grid_shape.npy is
# unavailable. Under normal use this should remain None — the exact grid shape
# is read from fft_grid_shape.npy produced by the main CNO script.
manual_grid_shape = None

# ── error checks ──────────────────────────────────────────────────────────────
for p, label in [
    (poscar_path, "POSCAR"),
    (occ_file,    "cno_occupations.npy"),
    (orb_file,    "cno_orbitals.npy"),
]:
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p}  [{label}]")

if manual_grid_shape is None and not grid_shape_file.exists():
    raise FileNotFoundError(
        f"Required FFT grid shape file not found: {grid_shape_file}. "
        "Run the main CNO script again after adding fft_grid_shape.npy output, "
        "or set manual_grid_shape = (Nx, Ny, Nz) near the top of this script."
    )

# ── load CNO data ─────────────────────────────────────────────────────────────
eigvals = np.load(occ_file)
eigvecs = np.load(orb_file)

print(f"Loaded CNO occupations from : {occ_file}")
print(f"Loaded CNO orbitals from    : {orb_file}")

if eigvecs.ndim != 2:
    raise ValueError(
        f"cno_orbitals.npy must be 2-D (Nr, n_cno), got shape {eigvecs.shape}"
    )

Nr              = eigvecs.shape[0]
n_cno_available = eigvecs.shape[1]
# Export the first/top CNOs by occupation.
# The main CNO script already sorted eigvals/eigvecs from largest to smallest.
# This does not recompute CNOs; it only converts saved CNOs into VESTA-readable cube files.
n_export_cno    = min(10, n_cno_available)

if len(eigvals) < n_cno_available:
    raise ValueError(
        f"eigvals has {len(eigvals)} entries but eigvecs has {n_cno_available} columns; "
        "lengths must match"
    )
if len(eigvals) < n_export_cno:
    raise ValueError(
        f"eigvals has only {len(eigvals)} entries, need at least {n_export_cno}"
    )

# ── FFT grid shape ─────────────────────────────────────────────────────────────
# Reads the exact FFT grid shape produced by the main CNO script.
# manual_grid_shape is an emergency override only — leave as None for normal use.
if manual_grid_shape is not None:
    Nx, Ny, Nz = manual_grid_shape
    grid_source = "manual override"
else:
    Nx, Ny, Nz = np.load(grid_shape_file).astype(int)
    grid_source = "fft_grid_shape.npy"

if Nx * Ny * Nz != Nr:
    raise ValueError(
        f"Grid shape ({Nx}, {Ny}, {Nz}) gives {Nx * Ny * Nz} points, "
        f"but cno_orbitals.npy has Nr={Nr}."
    )

print(f"Using grid shape ({grid_source}): ({Nx}, {Ny}, {Nz})   Nr = {Nr}")

# ── read POSCAR structure ─────────────────────────────────────────────────────
latvec, species, counts, atom_symbols, atom_numbers, cart_coords = \
    read_poscar_structure(poscar_path)
spec_str = " ".join(f"{s}({c})" for s, c in zip(species, counts))
print(f"Loaded {len(atom_numbers)} atoms from POSCAR: {spec_str}")
print(f"Lattice vectors (Å):\n{latvec}")


# ── cube-file writer ──────────────────────────────────────────────────────────
def write_cube_scalar(path, scalar_grid, latvec, atom_numbers, cart_coords,
                      comment1="", comment2=""):
    """Write a Gaussian .cube file with real atoms from POSCAR.

    Coordinates and grid vectors are written in Angstrom.
    scalar_grid  : ndarray shape (Nx, Ny, Nz), real-valued.
    latvec       : (3, 3) array of lattice vectors in Angstrom.
    atom_numbers : list of int atomic numbers, one per atom.
    cart_coords  : (natoms, 3) array of Cartesian coordinates in Angstrom.
    comment1     : first comment line (arbitrary string).
    comment2     : accepted for API compatibility; line 2 is always the
                   fixed Angstrom note required for VESTA.
    """
    gx, gy, gz = scalar_grid.shape
    natoms = len(atom_numbers)
    origin = [0.0, 0.0, 0.0]
    dv1 = latvec[0] / gx
    dv2 = latvec[1] / gy
    dv3 = latvec[2] / gz

    with open(path, "w") as f:
        # two comment lines
        f.write(f"{comment1}\n")
        f.write("Coordinates/vectors written in Angstrom for VESTA visualization\n")

        # natoms and origin
        f.write(f"{natoms:5d}  {origin[0]:12.6f}  {origin[1]:12.6f}  {origin[2]:12.6f}\n")

        # grid axis lines: count then step vector
        f.write(f"{gx:5d}  {dv1[0]:12.6f}  {dv1[1]:12.6f}  {dv1[2]:12.6f}\n")
        f.write(f"{gy:5d}  {dv2[0]:12.6f}  {dv2[1]:12.6f}  {dv2[2]:12.6f}\n")
        f.write(f"{gz:5d}  {dv3[0]:12.6f}  {dv3[1]:12.6f}  {dv3[2]:12.6f}\n")

        # real atoms: atomic number, charge 0.0, Cartesian position
        for Z, xyz in zip(atom_numbers, cart_coords):
            f.write(f"{Z:5d}   0.000000  {xyz[0]:12.6f}  {xyz[1]:12.6f}  {xyz[2]:12.6f}\n")

        # volumetric data — 6 values per line
        count = 0
        for ix in range(gx):
            for iy in range(gy):
                for iz in range(gz):
                    f.write(f"  {scalar_grid[ix, iy, iz]:12.6e}")
                    count += 1
                    if count % 6 == 0:
                        f.write("\n")
        if count % 6 != 0:
            f.write("\n")


# ── export top CNOs ───────────────────────────────────────────────────────────
print(f"Exporting top {n_export_cno} CNOs to: {cube_dir}")

for i in range(n_export_cno):
    cno     = eigvecs[:, i].reshape(Nx, Ny, Nz)
    density = np.abs(cno) ** 2
    density = density / density.sum()

    cube_path = cube_dir / f"cno_{i:03d}_density.cube"
    write_cube_scalar(
        cube_path,
        density,
        latvec,
        atom_numbers,
        cart_coords,
        comment1=f"CNO {i} density |phi(r)|^2",
        comment2=f"occupation = {eigvals[i]:.10e}; normalized discrete sum = 1",
    )

    np.save(cube_dir / f"cno_{i:03d}_complex_grid.npy", cno)

    print(f"  Exported: {cube_path.name}  (occupation = {eigvals[i]:.6e})")

print("Done.")

# ── metadata ──────────────────────────────────────────────────────────────────
meta_path = output_dir / "cno_cube_export_metadata.txt"
with open(meta_path, "w") as f:
    f.write(f"source cno_occupations      : {occ_file}\n")
    f.write(f"source cno_orbitals         : {orb_file}\n")
    f.write(f"POSCAR path                 : {poscar_path}\n")
    f.write(f"grid shape (Nx, Ny, Nz)     : ({Nx}, {Ny}, {Nz})\n")
    f.write(f"grid shape source          : {grid_source}\n")
    f.write(f"FFT grid shape file        : {grid_shape_file}\n")
    f.write(f"Nr (flat grid points)       : {Nr}\n")
    f.write(f"number of exported CNOs     : {n_export_cno}\n")
    f.write(f"cube output directory       : {cube_dir}\n")
    f.write(
        "NOTE: cube files contain normalized |CNO_i(r)|^2, "
        "not the complex CNO itself\n"
    )
    f.write(
        "NOTE: complex CNO grids are saved separately as .npy files "
        "in the same cube directory\n"
    )
    f.write("\nexported CNO occupations:\n")
    for i in range(n_export_cno):
        f.write(f"  CNO {i:3d} : {eigvals[i]:.10e}\n")
print(f"Wrote cube export metadata to: {meta_path}")
