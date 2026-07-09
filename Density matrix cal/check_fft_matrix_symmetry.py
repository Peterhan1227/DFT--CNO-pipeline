"""
check_fft_matrix_symmetry.py — same S_ij = exp(2*pi*i * G_i . r_j) check as
check_fft_matrix.py, but with r_j replaced by the SYMMETRY-TRANSFORMED points
r_src that symmetry_adapt_cnos.py actually interpolates from:

    r_src[n] = (positions[n] - center) @ R_inv.T + center

positions/center/R_inv are computed exactly as in symmetry_adapt_cnos.py (same
config, same R_CART table) -- this file does not import that script (it has
heavy module-level side effects requiring cno_orbitals.npy), it duplicates
only the small geometric piece.

Unlike the plain WS grid (which is a lattice-translated relabelling of the
primitive grid, hence exactly on-grid), r_src is generally OFF the k/N grid
whenever the WS center is not commensurate with the FFT grid spacing -- that
mismatch is exactly why symmetry_adapt_cnos.py needs trilinear interpolation
(map_coordinates) instead of an exact permutation. This script quantifies
that mismatch algebraically:
  - rank(S) / distinctness of r_src rows: a true bijection must keep all Nr
    points distinct; collapsed points would mean a bug in R_inv or center.
  - |det(S/sqrt(Nr))|: equals 1 (like the plain grid) only if r_src happens
    to coincide with the original grid mod a lattice translation (i.e. the
    operation is grid-exact, no real interpolation needed); it deviates from
    1 in proportion to how far off-grid the symmetry map lands.
"""

import numpy as np
from pathlib import Path
from config import MATERIAL, OUTPUT_SUBDIR, WS_CENTER_COORD_TYPE, WS_CENTER
from ws_cell import read_poscar_structure

# ── which operation to check -- must match a key in R_CART below ─────────────
OPERATION = "swap_xy"     # "swap_xy" | "c3_111" | "inversion"

data_dir   = Path(__file__).resolve().parent / "Data" / MATERIAL
output_dir = data_dir / "output" / OUTPUT_SUBDIR

Nx, Ny, Nz = (int(v) for v in np.load(output_dir / "fft_grid_shape.npy"))
Nr = Nx * Ny * Nz
print(f"FFT grid : ({Nx}, {Ny}, {Nz})   Nr={Nr}")

positions = np.load(output_dir / "ws_points_frac_cont.npy")   # (Nr,3), WS-folded fractional
assert positions.shape == (Nr, 3)

if WS_CENTER_COORD_TYPE == "fractional":
    center = np.array(WS_CENTER, dtype=float)
else:
    center = np.load(output_dir / "ws_center_frac_wrapped.npy").astype(float)
print(f"center   : {center.tolist()} (fractional)")

# ── R_inv: identical construction to symmetry_adapt_cnos.py ───────────────────
_latvec = read_poscar_structure(data_dir / "POSCAR")[0]

R_CART = {
    "inversion": -np.eye(3, dtype=float),
    "c3_111":    np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float),
    "swap_xy":   np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float),
}
_S_cart = R_CART[OPERATION]
_R_frac = np.linalg.inv(_latvec.T) @ _S_cart @ _latvec.T
R_inv = np.round(_R_frac)
if not np.allclose(_R_frac, R_inv, atol=1e-6):
    raise ValueError(f"Operation '{OPERATION}' is not a symmetry of this lattice.")
print(f"operation: {OPERATION}   R_inv (fractional) =\n{R_inv.astype(int)}")

# ── the points we actually interpolate ────────────────────────────────────────
r_src = (positions - center) @ R_inv.T + center     # (Nr, 3), fractional, generally off-grid

# how far off the k/N grid does the symmetry map land?
r_src_rounded = np.round(r_src * np.array([Nx, Ny, Nz])) / np.array([Nx, Ny, Nz])
off_grid_err  = np.abs(r_src - r_src_rounded)
print(f"\nmax distance of r_src from nearest k/N grid point (fractional): {off_grid_err.max():.6f}")
print(f"mean distance                                                : {off_grid_err.mean():.6f}")

# ── G grid: same dense integer grid as check_fft_matrix.py ───────────────────
gx, gy, gz = (a.ravel() for a in np.mgrid[0:Nx, 0:Ny, 0:Nz])
G = np.column_stack([gx, gy, gz]).astype(float)

print(f"\nBuilding S ({Nr}x{Nr}) from G and r_src ...")
S = np.exp(2j * np.pi * (G @ r_src.T))

rank = np.linalg.matrix_rank(S)
print(f"rank(S)  : {rank} / {Nr}   ({'FULL RANK' if rank == Nr else 'RANK DEFICIENT !!'})")

sign, logdet = np.linalg.slogdet(S / np.sqrt(Nr))
mag = np.exp(logdet)
print(f"det(S/sqrt(Nr)) : sign={sign:.6f}   |det|={mag:.10f}")
print("  (|det|=1 would mean r_src coincides with the grid mod a lattice "
      "translation -- i.e. exact, no real interpolation; deviation from 1 "
      "quantifies how far off-grid this operation's samples actually are)")
