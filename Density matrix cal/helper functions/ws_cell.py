"""
ws_cell.py — Wigner-Seitz cell utilities for the CNO pipeline.

Lattice convention (shared with all other scripts):
  latvec rows are lattice vectors a1, a2, a3.
  Fractional to Cartesian : r_cart = r_frac @ latvec
  Cartesian to fractional : r_frac = r_cart @ inv(latvec)
"""

import numpy as np

_ATOMIC_NUMBERS = {
    "H": 1,   "He": 2,  "Li": 3,  "Be": 4,  "B": 5,   "C": 6,   "N": 7,
    "O": 8,   "F": 9,   "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14,
    "P": 15,  "S": 16,  "Cl": 17, "Ar": 18, "K": 19,  "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23,  "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27,
    "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39,  "Zr": 40, "Nb": 41,
    "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48,
    "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53,  "Xe": 54,
    "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
    "Hf": 72, "Ta": 73, "W": 74,  "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83,
}

# Grid points to process per iteration in build_ws_grid_map.
# Each iteration allocates (CHUNK, ntrans, 3) float64.
# For nmax=2: ntrans=125, so CHUNK=4096 costs ~12 MB — safe on any machine.
_CHUNK = 4096


def read_poscar_structure(poscar_path):
    """Parse a VASP POSCAR.

    Handles Direct/Cartesian coordinates and optional Selective Dynamics line.

    Returns
    -------
    latvec      : (3, 3) lattice vectors in Angstrom (rows = a1, a2, a3)
    species     : list of element symbols per species type
    counts      : list of atom counts per species type
    atom_symbols: flat list of element symbol for each atom
    atom_numbers: flat list of atomic number for each atom
    frac_coords : (natoms, 3) fractional coordinates
    cart_coords : (natoms, 3) Cartesian coordinates in Angstrom
    """
    with open(poscar_path) as fh:
        lines = fh.readlines()

    scale  = float(lines[1])
    latvec = scale * np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)])

    species = lines[5].split()
    counts  = [int(x) for x in lines[6].split()]
    natoms  = sum(counts)

    atom_symbols = [sym for sym, n in zip(species, counts) for _ in range(n)]
    atom_numbers = []
    for sym in atom_symbols:
        if sym not in _ATOMIC_NUMBERS:
            raise ValueError(f"Unknown element '{sym}'. Add it to _ATOMIC_NUMBERS in ws_cell.py.")
        atom_numbers.append(_ATOMIC_NUMBERS[sym])

    # Line 7 is optional Selective Dynamics; skip it when present
    coord_line = 7 if not lines[7].strip().lower().startswith("s") else 8
    coord_mode = lines[coord_line].strip().lower()
    raw = np.array([[float(x) for x in lines[coord_line + 1 + i].split()[:3]]
                    for i in range(natoms)])

    if coord_mode.startswith("d"):
        frac_coords = raw
        cart_coords = raw @ latvec
    else:
        cart_coords = scale * raw
        frac_coords = cart_coords @ np.linalg.inv(latvec)

    return latvec, species, counts, atom_symbols, atom_numbers, frac_coords, cart_coords


def parse_ws_center(center, coord_type, latvec):
    """Convert a WS center specification into Cartesian and fractional forms.

    Parameters
    ----------
    center     : (3,) — coordinates of the desired WS center
    coord_type : "fractional" or "cartesian"
    latvec     : (3, 3) — lattice vectors (rows)

    Returns
    -------
    center_cart         : (3,) Cartesian Angstrom
    center_frac_cont    : (3,) fractional (same value as input, not wrapped)
    center_frac_wrapped : (3,) fractional wrapped to [0, 1)
    """
    c = np.asarray(center, dtype=float)
    if coord_type.lower().startswith("f"):
        center_frac_cont = c
        center_cart      = c @ latvec
    else:
        center_cart      = c
        center_frac_cont = c @ np.linalg.inv(latvec)
    return center_cart, center_frac_cont, center_frac_cont % 1.0


def build_ws_grid_map(latvec, grid_shape, center_cart, nmax=2, tie_tol=1e-12):
    """Map each primitive FFT grid point to its periodic image nearest to center_cart.

    Algorithm
    ---------
    Every real-space FFT grid point r sits at fractional coordinate
    (ix/Nx, iy/Ny, iz/Nz) in the primitive cell, i.e.

        r_prim = [ix/Nx, iy/Ny, iz/Nz] @ latvec.

    Its periodic images are  r_prim + n @ latvec  for any integer vector n.
    We pick the image closest to center_cart, searching n in [-nmax, nmax]^3.
    Ties (within tie_tol Å) are broken by the lexicographically smallest n.

    This is a one-to-one relabelling of the Nr = Nx*Ny*Nz grid points.
    The WS cell has the same volume as the primitive cell.

    Returns
    -------
    r_ws_cart      : (Nr, 3) Cartesian coordinates of WS grid points
    r_ws_frac_cont : (Nr, 3) fractional coords of WS points — NOT wrapped,
                     because the Bloch phase exp(2πi k·r) must use the
                     actual (possibly >1) fractional coordinate.
    prim_indices   : (Nr, 3) int — original FFT indices (ix, iy, iz) for each point
    translations   : (Nr, 3) int — integer n that was applied to each point
    """
    Nx, Ny, Nz = grid_shape

    # Primitive FFT grid: r_prim[p] = (ix/Nx, iy/Ny, iz/Nz) @ latvec
    ix, iy, iz  = [a.ravel() for a in np.mgrid[0:Nx, 0:Ny, 0:Nz]]
    r_prim_frac = np.column_stack([ix / Nx, iy / Ny, iz / Nz])  # (Nr, 3)
    r_prim_cart = r_prim_frac @ latvec                           # (Nr, 3)
    Nr = len(ix)

    # All candidate translations n in [-nmax, nmax]^3.
    # Pre-sort lexicographically: argmax on the tie mask then gives the
    # lex-smallest n among those that achieve the minimum distance.
    ns = np.arange(-nmax, nmax + 1)
    n1, n2, n3 = [a.ravel() for a in np.meshgrid(ns, ns, ns, indexing="ij")]
    all_n        = np.column_stack([n1, n2, n3])                  # (ntrans, 3)
    lex_order    = np.lexsort((all_n[:, 2], all_n[:, 1], all_n[:, 0]))
    all_n        = all_n[lex_order]
    all_n_cart   = all_n @ latvec                                  # (ntrans, 3)

    # For each grid point find the best translation.
    # Memory note: allocating (Nr, ntrans, 3) at once would be ~650 MB for a
    # 60^3 grid with nmax=2. We process _CHUNK points at a time instead.
    best_n_idx = np.empty(Nr, dtype=np.intp)
    for start in range(0, Nr, _CHUNK):
        sl = slice(start, min(start + _CHUNK, Nr))
        # displaced[p, t] = r_prim[p] + n_cart[t] - center  → (chunk, ntrans, 3)
        displaced = r_prim_cart[sl, None, :] + all_n_cart[None, :, :] - center_cart
        dist      = np.sqrt(np.einsum("pti,pti->pt", displaced, displaced))
        min_d     = dist.min(axis=1, keepdims=True)
        # argmax on bool finds the first True in lex order = lex-smallest n at min dist
        best_n_idx[sl] = np.argmax(dist <= min_d + tie_tol, axis=1)

    best_n = all_n[best_n_idx]  # (Nr, 3) integer translations applied

    r_ws_cart      = r_prim_cart + best_n @ latvec
    r_ws_frac_cont = r_prim_frac + best_n     # continuous, not wrapped

    return r_ws_cart, r_ws_frac_cont, np.column_stack([ix, iy, iz]), best_n
