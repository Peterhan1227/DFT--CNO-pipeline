import sys
import numpy as np
from pathlib import Path
from config import MATERIAL, OUTPUT_SUBDIR

sys.path.insert(0, str(Path(__file__).resolve().parent / "helper functions"))
from ws_cell import read_poscar_structure


# ── paths ─────────────────────────────────────────────────────────────────────
base_dir    = Path(__file__).resolve().parent
data_dir    = base_dir / "Data" / MATERIAL
output_dir  = data_dir / "output" / OUTPUT_SUBDIR
poscar_path = data_dir / "POSCAR"
occ_file        = output_dir / "cno_occupations.npy"
orb_file        = output_dir / "cno_orbitals.npy"
grid_shape_file = output_dir / "fft_grid_shape.npy"
ws_enabled_file = output_dir / "ws_enabled.npy"
cube_dir        = output_dir / "cno_cubes"
cube_dir.mkdir(parents=True, exist_ok=True)

# Tile the output cube into an N×N×N supercell so VESTA shows the full crystal
# without needing to change boundary settings.  Set to (1,1,1) for primitive only.
CUBE_SUPERCELL = (3, 3, 3)

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
        "Run the main CNO script or set manual_grid_shape = (Nx, Ny, Nz)."
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
n_export_cno    = min(10, n_cno_available)

if len(eigvals) < n_export_cno:
    raise ValueError(f"eigvals has only {len(eigvals)} entries, need {n_export_cno}")

# ── FFT grid shape ─────────────────────────────────────────────────────────────
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
latvec, species, counts, atom_symbols, atom_numbers, frac_coords, cart_coords = \
    read_poscar_structure(poscar_path)
spec_str = " ".join(f"{s}({c})" for s, c in zip(species, counts))
print(f"Loaded {len(atom_numbers)} atoms from POSCAR: {spec_str}")

# ── detect WS mode ────────────────────────────────────────────────────────────
ws_mode = ws_enabled_file.exists() and bool(np.load(ws_enabled_file))
print(f"WS mode: {ws_mode}")

if ws_mode:
    _ws_files = {
        "ws_points_cart.npy": output_dir / "ws_points_cart.npy",
    }
    for fname, p in _ws_files.items():
        if not p.exists():
            raise FileNotFoundError(
                f"WS mode is active but {fname} not found: {p}\n"
                "Rerun Wavecar_to_Coeff.py to regenerate WS map files."
            )
    ws_points_cart = np.load(output_dir / "ws_points_cart.npy")
    if len(ws_points_cart) != Nr:
        raise ValueError(
            f"ws_points_cart has {len(ws_points_cart)} points but Nr={Nr}."
        )
    print(f"Loaded ws_points_cart: shape={ws_points_cart.shape}")
    _ws_base_path = output_dir / "ws_base_indices.npy"
    ws_base_indices = np.load(_ws_base_path) if _ws_base_path.exists() else None


# The Gaussian .cube format standard is BOHR — VESTA ignores any comment about
# Angstrom and always reads raw numbers as Bohr.  Convert everything here.
_ANG2BOHR = 1.8897259886


# ── cube-file writer ──────────────────────────────────────────────────────────
def write_cube_scalar(path, scalar_grid, latvec, atom_numbers, cart_coords,
                      comment1="", comment2="", supercell=(1, 1, 1)):
    """Write a Gaussian .cube file (BOHR units), optionally tiled into a supercell.

    supercell=(sx, sy, sz) repeats the primitive cell sx×sy×sz times along the
    three lattice directions.  Atoms and volumetric data are both replicated.
    """
    gx, gy, gz = scalar_grid.shape
    sx, sy, sz = supercell

    # Convert Angstrom → Bohr for all spatial quantities
    dv1 = latvec[0] / gx * _ANG2BOHR
    dv2 = latvec[1] / gy * _ANG2BOHR
    dv3 = latvec[2] / gz * _ANG2BOHR

    # Tile volumetric data
    tiled = np.tile(scalar_grid, (sx, sy, sz))
    Gx, Gy, Gz = tiled.shape   # sx*gx, sy*gy, sz*gz

    # Build supercell atom list (positions in Bohr)
    super_Z, super_xyz = [], []
    for nx in range(sx):
        for ny in range(sy):
            for nz in range(sz):
                shift = (nx * latvec[0] + ny * latvec[1] + nz * latvec[2]) * _ANG2BOHR
                for Z, xyz in zip(atom_numbers, cart_coords):
                    super_Z.append(Z)
                    super_xyz.append(xyz * _ANG2BOHR + shift)
    natoms_super = len(super_Z)

    with open(path, "w") as f:
        f.write(f"{comment1}\n")
        sc_tag = f"{sx}x{sy}x{sz} supercell; " if supercell != (1, 1, 1) else ""
        f.write(f"{sc_tag}all coordinates in Bohr (Gaussian cube standard)\n")
        f.write(f"{natoms_super:5d}   0.000000   0.000000   0.000000\n")
        f.write(f"{Gx:5d}  {dv1[0]:12.6f}  {dv1[1]:12.6f}  {dv1[2]:12.6f}\n")
        f.write(f"{Gy:5d}  {dv2[0]:12.6f}  {dv2[1]:12.6f}  {dv2[2]:12.6f}\n")
        f.write(f"{Gz:5d}  {dv3[0]:12.6f}  {dv3[1]:12.6f}  {dv3[2]:12.6f}\n")
        for Z, xyz in zip(super_Z, super_xyz):
            f.write(f"{Z:5d}   0.000000  {xyz[0]:12.6f}  {xyz[1]:12.6f}  {xyz[2]:12.6f}\n")
        count = 0
        for ix in range(Gx):
            for iy in range(Gy):
                for iz in range(Gz):
                    f.write(f"  {tiled[ix, iy, iz]:12.6e}")
                    count += 1
                    if count % 6 == 0:
                        f.write("\n")
        if count % 6 != 0:
            f.write("\n")


# ── export top CNOs ───────────────────────────────────────────────────────────
print(f"\nExporting top {n_export_cno} CNOs to: {cube_dir}")

for i in range(n_export_cno):
    cno     = eigvecs[:, i]
    density = np.abs(cno) ** 2
    density = density / density.sum()

    if ws_mode:
        # Save WS point cloud for any further analysis
        np.savez(
            cube_dir / f"cno_{i:03d}_ws_points.npz",
            points_cart    = ws_points_cart,
            density        = density,
            complex_values = cno,
            occupation     = np.array(eigvals[i]),
        )
        # Back-map density to the primitive cell grid.
        # |CNO|^2 has full crystal periodicity, so the value at the WS
        # representative r_ws equals the value at the primitive grid point —
        # the back-mapped cube is the correct density for VESTA visualisation.
        if ws_base_indices is None:
            print(f"  WARNING: ws_base_indices missing, cannot write cube for CNO {i}")
            continue
        grid_3d = np.zeros((Nx, Ny, Nz), dtype=float)
        grid_3d[ws_base_indices[:, 0],
                ws_base_indices[:, 1],
                ws_base_indices[:, 2]] = density
    else:
        grid_3d = density.reshape(Nx, Ny, Nz)
        np.save(cube_dir / f"cno_{i:03d}_complex_grid.npy", cno.reshape(Nx, Ny, Nz))

    sx, sy, sz = CUBE_SUPERCELL
    cube_path = cube_dir / f"cno_{i:03d}_density.cube"
    write_cube_scalar(
        cube_path,
        grid_3d,
        latvec,
        atom_numbers,
        cart_coords,
        comment1=f"CNO {i} density |phi(r)|^2",
        comment2=f"occupation = {eigvals[i]:.10e}; normalized discrete sum = 1",
        supercell=CUBE_SUPERCELL,
    )
    print(f"  Exported: {cube_path.name}  (occupation = {eigvals[i]:.6e})  [{sx}×{sy}×{sz} supercell]")

print("Done.")

# ── metadata ──────────────────────────────────────────────────────────────────
meta_path = output_dir / "cno_cube_export_metadata.txt"
with open(meta_path, "w") as f:
    f.write(f"source cno_occupations      : {occ_file}\n")
    f.write(f"source cno_orbitals         : {orb_file}\n")
    f.write(f"POSCAR path                 : {poscar_path}\n")
    f.write(f"grid shape (Nx, Ny, Nz)     : ({Nx}, {Ny}, {Nz})\n")
    f.write(f"grid shape source           : {grid_source}\n")
    f.write(f"Nr (flat grid points)       : {Nr}\n")
    f.write(f"ws_mode                     : {ws_mode}\n")
    f.write(f"number of exported CNOs     : {n_export_cno}\n")
    f.write(f"cube output directory       : {cube_dir}\n")
    if ws_mode:
        f.write("NOTE: WS mode — density back-mapped to primitive grid for cube export.\n")
        f.write("NOTE: WS point clouds also saved as .npz per CNO.\n")
    else:
        f.write("NOTE: cube files contain normalized |CNO_i(r)|^2, not the complex CNO.\n")
        f.write("NOTE: complex CNO grids saved separately as .npy files in cube directory.\n")
    f.write("\nexported CNO occupations:\n")
    for i in range(n_export_cno):
        f.write(f"  CNO {i:3d} : {eigvals[i]:.10e}\n")
print(f"Wrote cube export metadata to: {meta_path}")
