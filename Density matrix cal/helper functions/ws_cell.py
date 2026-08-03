"""
ws_cell.py — Wigner-Seitz cell utilities for the CNO pipeline.

Lattice convention (shared with all other scripts):
  latvec rows are lattice vectors a1, a2, a3.
  Fractional to Cartesian : r_cart = r_frac @ latvec
  Cartesian to fractional : r_frac = r_cart @ inv(latvec)
"""

from itertools import combinations

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

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

# Grid points to process per iteration in the WS-map builders.
# Each iteration allocates (CHUNK, ntrans, 3) float64.
# For nmax=2: ntrans=125, so CHUNK=4096 costs ~12 MB — safe on any machine.
_CHUNK = 4096

# Squared-distance comparison parameters.  A relative tolerance scaled to the
# local squared distance is essential: comparing distances after ``sqrt`` can
# turn an exact Voronoi tie into an order-dependent near-tie.
_TIE_ATOL2 = 1.0e-11      # Angstrom^2
_TIE_RTOL = 256 * np.finfo(float).eps
_GEOM_TOL = 2.0e-10       # fractional/half-space arithmetic tolerance


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


def _primitive_grid(grid_shape):
    """Return integer indices and fractional locations of FFT nodes."""
    Nx, Ny, Nz = (int(v) for v in grid_shape)
    ix, iy, iz = [a.ravel() for a in np.mgrid[0:Nx, 0:Ny, 0:Nz]]
    indices = np.column_stack([ix, iy, iz]).astype(int)
    return indices, indices / np.array([Nx, Ny, Nz], dtype=float)[None, :]


def _translation_grid(nmax):
    ns = np.arange(-int(nmax), int(nmax) + 1)
    n1, n2, n3 = [a.ravel() for a in np.meshgrid(ns, ns, ns, indexing="ij")]
    return np.column_stack([n1, n2, n3]).astype(int)


def _tie_mask_squared(distance2, *, atol2=_TIE_ATOL2, rtol=_TIE_RTOL):
    """Mask all translations tied to the minimum squared distance per row."""
    minimum = distance2.min(axis=1, keepdims=True)
    scale = np.maximum(1.0, np.maximum(np.abs(minimum), np.max(np.abs(distance2), axis=1, keepdims=True)))
    return distance2 <= minimum + atol2 + rtol * scale


def _ws_halfspaces_fractional(latvec, center_cart, nmax=3):
    """Half-spaces ``A @ f <= b`` for the centered WS/Voronoi polyhedron.

    Coordinates are continuous fractional row coordinates.  The lattice
    vectors remain rows, so ``f @ latvec`` is Cartesian.  The construction is
    the exact bisector condition used by :func:`direct_fourier.ws_membership`.
    """
    latvec = np.asarray(latvec, dtype=float)
    center_cart = np.asarray(center_cart, dtype=float)
    translations = _translation_grid(nmax)
    translations = translations[np.any(translations != 0, axis=1)]
    R_cart = translations @ latvec
    R2 = np.einsum("ij,ij->i", R_cart, R_cart)
    # 2 * (f @ latvec - center) dot R <= |R|^2.
    A = 2.0 * (latvec @ R_cart.T).T
    b = R2 + 2.0 * (R_cart @ center_cart)
    return A, b


def ws_polyhedron(latvec, center_cart, nmax=3):
    """Return irredundant fractional WS half-spaces and Cartesian vertices.

    ``nmax`` is only the finite neighbour list used to find the Voronoi cell;
    the returned active half-spaces are the actual WS facets.  A caller can
    use the vertices for conservative PAW-site discovery and the facets for
    exact voxel clipping.
    """
    latvec = np.asarray(latvec, dtype=float)
    A_all, b_all = _ws_halfspaces_fractional(latvec, center_cart, nmax=nmax)
    center_frac = np.asarray(center_cart, dtype=float) @ np.linalg.inv(latvec)
    halfspaces = np.column_stack([A_all, -b_all])
    try:
        intersections = HalfspaceIntersection(halfspaces, center_frac).intersections
    except Exception as exc:  # pragma: no cover - qhull message is valuable to caller
        raise RuntimeError("Could not construct the Wigner-Seitz polyhedron") from exc
    # Keep actual two-dimensional facets only.  A redundant long-lattice
    # vector can touch one WS vertex (or an edge) without bounding a face;
    # retaining it would make every voxel near that vertex look like a
    # boundary voxel and defeat the finite-volume fast path.
    scale = max(1.0, float(np.max(np.abs(b_all))))
    active = np.zeros(len(b_all), dtype=bool)
    face_tol = 1.0e-8 * scale
    for i, (normal, bound) in enumerate(zip(A_all, b_all)):
        points = intersections[np.abs(intersections @ normal - bound) <= face_tol]
        if len(points) >= 3 and np.linalg.matrix_rank(points[1:] - points[0], tol=1.0e-9) >= 2:
            active[i] = True
    A, b = A_all[active], b_all[active]
    return A, b, intersections @ latvec


def _voronoi_facets_cart(lattice, nmax=3):
    """Irredundant Cartesian half-spaces of the WS cell of ``lattice``.

    The result is ``normal @ r <= bound`` about the origin.  It is used for
    the *sampling-lattice* Voronoi voxel, whose shape (unlike a fractional
    parallelepiped) inherits every point symmetry of an invariant FFT mesh.
    """
    lattice = np.asarray(lattice, dtype=float)
    translations = _translation_grid(nmax)
    translations = translations[np.any(translations != 0, axis=1)]
    R = translations @ lattice
    normal_all = 2.0 * R
    bound_all = np.einsum("ij,ij->i", R, R)
    halfspaces = np.column_stack([normal_all, -bound_all])
    vertices = HalfspaceIntersection(halfspaces, np.zeros(3)).intersections
    scale = max(1.0, float(np.max(np.abs(bound_all))))
    active = np.zeros(len(bound_all), dtype=bool)
    for i, (normal, bound) in enumerate(zip(normal_all, bound_all)):
        points = vertices[np.abs(vertices @ normal - bound) <= 1.0e-8 * scale]
        if len(points) >= 3 and np.linalg.matrix_rank(points[1:] - points[0], tol=1.0e-10) >= 2:
            active[i] = True
    return normal_all[active], bound_all[active], vertices


def build_ws_grid_map(latvec, grid_shape, center_cart, nmax=2, tie_tol=1e-12):
    """Map each primitive FFT grid point to its periodic image nearest to center_cart.

    Algorithm
    ---------
    Every real-space FFT grid point r sits at fractional coordinate
    (ix/Nx, iy/Ny, iz/Nz) in the primitive cell, i.e.

        r_prim = [ix/Nx, iy/Ny, iz/Nz] @ latvec.

    Its periodic images are  r_prim + n @ latvec  for any integer vector n.
    We pick the image closest to center_cart, searching n in [-nmax, nmax]^3.
    Ties are intentionally retained only by the newer
    :func:`build_ws_weighted_tie_map` and
    :func:`build_ws_finite_volume_map`.  This legacy compatibility function
    still chooses one representative so existing output directories and old
    downstream scripts remain readable.  New regional-PAW construction must
    not use it.

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
    indices, r_prim_frac = _primitive_grid(grid_shape)
    # Primitive FFT grid: r_prim[p] = (ix/Nx, iy/Ny, iz/Nz) @ latvec
    r_prim_cart = r_prim_frac @ latvec                           # (Nr, 3)
    Nr = len(indices)

    # All candidate translations n in [-nmax, nmax]^3.
    # Pre-sort lexicographically: argmax on the tie mask then gives the
    # lex-smallest n among those that achieve the minimum distance.
    all_n = _translation_grid(nmax)
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
        dist2     = np.einsum("pti,pti->pt", displaced, displaced)
        min_d2    = dist2.min(axis=1, keepdims=True)
        # argmax on bool finds the first True in lex order = lex-smallest n at min dist
        best_n_idx[sl] = np.argmax(dist2 <= min_d2 + tie_tol, axis=1)

    best_n = all_n[best_n_idx]  # (Nr, 3) integer translations applied

    r_ws_cart      = r_prim_cart + best_n @ latvec
    r_ws_frac_cont = r_prim_frac + best_n     # continuous, not wrapped

    return r_ws_cart, r_ws_frac_cont, indices, best_n


def build_ws_weighted_tie_map(latvec, grid_shape, center_cart, nmax=3,
                              tie_atol2=_TIE_ATOL2, tie_rtol=_TIE_RTOL):
    """Map every FFT node to *all* exactly nearest periodic WS images.

    For a native node with ``m`` equally-nearest images, the result has ``m``
    rows with equal ``weight=1/m``.  The total weight per native FFT node is
    exactly one.  This is the narrow replacement for the old arbitrary
    boundary ownership rule and is useful as a diagnostic quadrature path.

    Returns
    -------
    r_cart, r_frac, base_indices, translations, weights, tie_count
        ``tie_count`` has one entry per native FFT node; all other arrays are
        expanded sample arrays and can therefore be longer than ``prod(grid)``.
    """
    latvec = np.asarray(latvec, dtype=float)
    center_cart = np.asarray(center_cart, dtype=float)
    base, primitive_frac = _primitive_grid(grid_shape)
    primitive_cart = primitive_frac @ latvec
    translations_all = _translation_grid(nmax)
    translation_cart = translations_all @ latvec

    out_base, out_translation, out_weight, tie_count = [], [], [], []
    for start in range(0, len(base), _CHUNK):
        sl = slice(start, min(start + _CHUNK, len(base)))
        displaced = (primitive_cart[sl, None, :] + translation_cart[None, :, :]
                     - center_cart[None, None, :])
        dist2 = np.einsum("pti,pti->pt", displaced, displaced)
        tied = _tie_mask_squared(dist2, atol2=tie_atol2, rtol=tie_rtol)
        local_rows, translation_rows = np.nonzero(tied)
        counts = np.bincount(local_rows, minlength=len(base[sl]))
        if np.any(counts == 0):
            raise RuntimeError("A native FFT node has no nearest WS image")
        out_base.append(base[sl][local_rows])
        out_translation.append(translations_all[translation_rows])
        out_weight.append(1.0 / counts[local_rows])
        tie_count.append(counts)

    base_out = np.concatenate(out_base, axis=0)
    trans_out = np.concatenate(out_translation, axis=0)
    weights = np.concatenate(out_weight, axis=0)
    counts = np.concatenate(tie_count, axis=0)
    frac_out = base_out / np.asarray(grid_shape, dtype=float)[None, :] + trans_out
    if not np.isclose(weights.sum(), len(base), rtol=0.0, atol=1.0e-10):
        raise RuntimeError("Weighted tie map does not preserve the total native-grid weight")
    return frac_out @ latvec, frac_out, base_out, trans_out, weights, counts


def _clip_polyhedron_volume(A, b, latvec, tol=_GEOM_TOL):
    """Exact Cartesian volume of the convex polyhedron ``A @ f <= b``."""
    vertices = []
    for choice in combinations(range(len(b)), 3):
        mat = A[list(choice)]
        if abs(np.linalg.det(mat)) < 1.0e-13:
            continue
        point = np.linalg.solve(mat, b[list(choice)])
        if np.all(A @ point <= b + tol):
            vertices.append(point)
    if len(vertices) < 4:
        return 0.0
    vertices = np.asarray(vertices)
    # Qhull requires unique vertices; removing duplicate plane intersections
    # also avoids spurious zero-volume hull failures at a WS edge/corner.
    rounded = np.round(vertices / 1.0e-10).astype(np.int64)
    _, unique = np.unique(rounded, axis=0, return_index=True)
    vertices = vertices[np.sort(unique)]
    if len(vertices) < 4:
        return 0.0
    try:
        return float(ConvexHull(vertices @ latvec).volume)
    except Exception:
        return 0.0


def build_ws_finite_volume_map(latvec, grid_shape, center_cart, nmax=3):
    """Finite-volume quadrature map for a centered Wigner-Seitz cell.

    Each native FFT node is the centre of its periodic *sampling-lattice
    Voronoi voxel*.  This is the true dual voxel of the FFT-node lattice; it
    tiles space and, when the mesh is symmetry-compatible, transforms into
    itself under the point group.  A fractional-coordinate parallelepiped is
    not suitable here: in a hexagonal cell C3 shears it into a different
    parallelepiped even when the FFT nodes themselves map exactly.

    For every periodic image of that voxel which intersects the WS polyhedron, a
    sample row is returned with weight

    ``volume(voxel image intersect WS) / volume(native voxel)``.

    Weights therefore sum to ``prod(grid_shape)`` and are not ownership
    labels.  The wavefunction is still evaluated from the original FFT/plane
    wave representation at the node; only the regional integral changes.
    Boundary voxels are clipped against the actual Voronoi half-spaces, not
    approximated by a point-in-cell test.
    """
    latvec = np.asarray(latvec, dtype=float)
    grid = np.asarray(grid_shape, dtype=int)
    base, primitive_frac = _primitive_grid(grid)
    ws_A, ws_b, _ = ws_polyhedron(latvec, center_cart, nmax=nmax)
    # The Voronoi cell of the sampling lattice is the finite-volume voxel.
    # Convert its Cartesian facets and vertices into the global fractional
    # coordinate system used by the regional WS polyhedron.
    sample_lattice = latvec / grid[:, None]
    voxel_normal_cart, voxel_bound0, voxel_vertices_cart = _voronoi_facets_cart(sample_lattice, nmax=3)
    voxel_A = voxel_normal_cart @ latvec.T
    voxel_vertices_frac = voxel_vertices_cart @ np.linalg.inv(latvec)
    voxel_min_offset = np.min(voxel_vertices_frac @ ws_A.T, axis=0)
    voxel_max_offset = np.max(voxel_vertices_frac @ ws_A.T, axis=0)
    translations_all = _translation_grid(nmax)
    voxel_volume = abs(float(np.linalg.det(latvec))) / int(np.prod(grid))

    base_rows, translation_rows, weight_rows = [], [], []
    # The inexpensive half-space bound rejects almost all periodic images.
    # Exact convex clipping is done only for voxels crossing a WS facet.
    chunk = min(_CHUNK, 1024)
    for start in range(0, len(base), chunk):
        sl = slice(start, min(start + chunk, len(base)))
        centers = primitive_frac[sl, None, :] + translations_all[None, :, :]
        slack = centers @ ws_A.T - ws_b[None, None, :]
        possible = np.all(slack + voxel_min_offset[None, None, :] <= _GEOM_TOL, axis=2)
        full = possible & np.all(slack + voxel_max_offset[None, None, :] <= _GEOM_TOL, axis=2)

        local, trans = np.nonzero(full)
        if len(local):
            base_rows.append(base[sl][local])
            translation_rows.append(translations_all[trans])
            weight_rows.append(np.ones(len(local)))

        boundary = np.argwhere(possible & ~full)
        for local_i, trans_i in boundary:
            center = centers[local_i, trans_i]
            voxel_b = voxel_bound0 + center @ voxel_A.T
            volume = _clip_polyhedron_volume(
                np.vstack([ws_A, voxel_A]), np.concatenate([ws_b, voxel_b]), latvec)
            if volume > voxel_volume * 1.0e-12:
                base_rows.append(base[sl][local_i:local_i + 1])
                translation_rows.append(translations_all[trans_i:trans_i + 1])
                weight_rows.append(np.array([volume / voxel_volume]))

    base_out = np.concatenate(base_rows, axis=0)
    trans_out = np.concatenate(translation_rows, axis=0)
    weights = np.concatenate(weight_rows, axis=0)
    frac_out = base_out / grid.astype(float)[None, :] + trans_out
    expected = int(np.prod(grid))
    if not np.isclose(weights.sum(), expected, rtol=0.0, atol=2.0e-7 * expected):
        raise RuntimeError(
            "Finite-volume WS weights do not sum to the WS volume: "
            f"sum={weights.sum():.12f}, expected={expected}. Increase nmax or inspect geometry."
        )
    return frac_out @ latvec, frac_out, base_out, trans_out, weights
