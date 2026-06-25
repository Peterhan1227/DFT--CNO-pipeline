"""Crystal context — primitive-cell edges, atoms, bonds, and periodic replicas.

Atom colors and covalent radii come from ASE so the viewer does not maintain its
own element table; ASE is already a runtime dependency for the package.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from ase.data import chemical_symbols, covalent_radii
from ase.data.colors import jmol_colors


def _resolve_atomic_number(symbol_or_number) -> int:
    if isinstance(symbol_or_number, (int, np.integer)):
        return int(symbol_or_number)
    s = str(symbol_or_number).strip()
    try:
        return chemical_symbols.index(s)
    except ValueError:
        return 0


def atom_color(symbol_or_number) -> Tuple[float, float, float]:
    Z = _resolve_atomic_number(symbol_or_number)
    if 0 <= Z < len(jmol_colors):
        rgb = jmol_colors[Z]
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    return (0.7, 0.7, 0.7)


def atom_radius(symbol_or_number, scale: float = 0.4) -> float:
    """Display radius — ``covalent_radius * scale``.

    The default ``scale = 0.4`` keeps spheres small enough not to swallow the
    orbital lobe while still being clearly visible at typical isovalues.
    """
    Z = _resolve_atomic_number(symbol_or_number)
    if 0 <= Z < len(covalent_radii):
        return float(covalent_radii[Z]) * float(scale)
    return 0.4


# --------------------------------------------------------------------- geometry
def replication_translations(
    lattice: np.ndarray,
    replication: Sequence[int],
) -> np.ndarray:
    """Return the Cartesian translations for an ``(Nx, Ny, Nz)`` replication.

    Always includes the zero translation as row 0; otherwise rows are sorted by
    ``(n1, n2, n3)`` lexicographically so caller-side iteration order is stable.
    """
    lattice = np.asarray(lattice, dtype=np.float64)
    nx, ny, nz = (max(1, int(n)) for n in replication)
    triples = list(product(range(nx), range(ny), range(nz)))
    triples.sort()
    if (0, 0, 0) in triples:
        triples.remove((0, 0, 0))
        triples.insert(0, (0, 0, 0))
    n = np.asarray(triples, dtype=np.int64)
    return n.astype(np.float64) @ lattice


def primitive_cell_edges(lattice: np.ndarray, origin: np.ndarray | None = None):
    """Return a :class:`pyvista.PolyData` with the 12 edges of the primitive cell."""
    import pyvista as pv

    lattice = np.asarray(lattice, dtype=np.float64)
    origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=np.float64)
    a1, a2, a3 = lattice[0], lattice[1], lattice[2]
    corners = np.array(
        [
            origin,
            origin + a1,
            origin + a2,
            origin + a3,
            origin + a1 + a2,
            origin + a1 + a3,
            origin + a2 + a3,
            origin + a1 + a2 + a3,
        ]
    )
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]
    lines = np.empty((len(edges), 3), dtype=np.int64)
    lines[:, 0] = 2
    for i, (a, b) in enumerate(edges):
        lines[i, 1] = a
        lines[i, 2] = b
    return pv.PolyData(corners, lines=lines.ravel())


def replicated_cell_edges(
    lattice: np.ndarray,
    replication: Sequence[int],
    origin: np.ndarray | None = None,
):
    """Stack of primitive-cell edge polydatas, one per translation."""
    import pyvista as pv

    shifts = replication_translations(lattice, replication)
    blocks = []
    base_origin = np.zeros(3) if origin is None else np.asarray(origin, dtype=np.float64)
    for shift in shifts:
        blocks.append(primitive_cell_edges(lattice, base_origin + shift))
    return pv.merge(blocks) if blocks else primitive_cell_edges(lattice, base_origin)


def replicated_atoms(
    symbols: Sequence,
    cart_positions: np.ndarray,
    lattice: np.ndarray,
    replication: Sequence[int],
):
    """Return ``(positions, symbols)`` for all atoms in the replicated cells."""
    cart_positions = np.asarray(cart_positions, dtype=np.float64)
    shifts = replication_translations(lattice, replication)
    all_pos = []
    all_sym: List = []
    syms = [str(s) for s in symbols]
    for shift in shifts:
        all_pos.append(cart_positions + shift[None, :])
        all_sym.extend(syms)
    pos = np.concatenate(all_pos, axis=0) if all_pos else cart_positions
    return pos, all_sym


def find_bonds(
    positions: np.ndarray,
    symbols: Sequence,
    bond_scale: float = 1.15,
) -> List[Tuple[int, int]]:
    """Distance-cutoff bond detection.

    A bond is drawn iff ``d_ij < bond_scale * (r_i + r_j)`` where ``r_i`` is the
    ASE covalent radius of atom ``i``.  Self-bonds, zero-length, and reversed
    duplicates are excluded.
    """
    positions = np.asarray(positions, dtype=np.float64)
    n = positions.shape[0]
    if n < 2:
        return []
    radii = np.array(
        [
            covalent_radii[_resolve_atomic_number(s)]
            if 0 <= _resolve_atomic_number(s) < len(covalent_radii)
            else 0.7
            for s in symbols
        ]
    )
    bonds: List[Tuple[int, int]] = []
    # Vectorized pair distance — fine for the ~thousands of atoms typical here.
    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    cutoff = bond_scale * (radii[:, None] + radii[None, :])
    iu = np.triu_indices(n, k=1)
    mask = (d[iu] > 1e-6) & (d[iu] < cutoff[iu])
    for a, b in zip(iu[0][mask], iu[1][mask]):
        bonds.append((int(a), int(b)))
    return bonds


def build_bonded_crystal(
    symbols: Sequence,
    cart_positions: np.ndarray,
    lattice: np.ndarray,
    replication: Sequence[int],
    bond_scale: float = 1.15,
    halo: int = 1,
):
    """Return ``(draw_positions, draw_symbols, bonds)`` for a fully-bonded supercell.

    The displayed atoms are those in the ``replication`` cells, but bonds are
    searched over a one-cell ``halo`` on every side so atoms at the supercell
    boundary keep their neighbours (in the diamond structure the bond partner of an
    atom always lives in an adjacent cell).  A bond is kept if **either** endpoint
    is a displayed atom; any halo atom touched by a kept bond is also returned so it
    can be drawn (no dangling half-bonds).  ``bonds`` index into ``draw_positions``.
    """
    cart_positions = np.asarray(cart_positions, dtype=np.float64)
    lattice = np.asarray(lattice, dtype=np.float64)
    nx, ny, nz = (max(1, int(n)) for n in replication)
    syms = [str(s) for s in symbols]

    positions: List[np.ndarray] = []
    all_sym: List[str] = []
    displayed: List[bool] = []
    for a in range(-halo, nx + halo):
        for b in range(-halo, ny + halo):
            for c in range(-halo, nz + halo):
                shift = a * lattice[0] + b * lattice[1] + c * lattice[2]
                is_disp = (0 <= a < nx) and (0 <= b < ny) and (0 <= c < nz)
                for s, p in zip(syms, cart_positions):
                    positions.append(p + shift)
                    all_sym.append(s)
                    displayed.append(is_disp)
    pos = np.asarray(positions, dtype=np.float64)
    disp = np.asarray(displayed, dtype=bool)

    all_bonds = find_bonds(pos, all_sym, bond_scale=bond_scale)
    kept = [(i, j) for (i, j) in all_bonds if disp[i] or disp[j]]

    keep_idx = set(int(i) for i in np.flatnonzero(disp))
    for i, j in kept:
        keep_idx.add(int(i))
        keep_idx.add(int(j))
    order = sorted(keep_idx)
    remap = {old: new for new, old in enumerate(order)}
    draw_pos = pos[order] if order else pos[:0]
    draw_sym = [all_sym[k] for k in order]
    draw_bonds = [(remap[i], remap[j]) for (i, j) in kept]
    return draw_pos, draw_sym, draw_bonds


def build_atom_glyphs(
    positions: np.ndarray,
    symbols: Sequence,
    radius_scale: float = 0.4,
    theta_resolution: int = 16,
    phi_resolution: int = 16,
) -> Tuple[object, np.ndarray]:
    """Return ``(merged_mesh, rgb_per_point)``.

    Spheres are merged into a single :class:`pyvista.PolyData` for efficient
    rendering; the per-cell RGB color array is returned separately so the caller
    can pass it directly to ``add_mesh(..., scalars=colors, rgb=True)``.
    """
    import pyvista as pv

    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape[0] == 0:
        return pv.PolyData(), np.zeros((0, 3), dtype=np.float64)

    meshes = []
    per_face_colors = []
    for pos, sym in zip(positions, symbols):
        r = atom_radius(sym, scale=radius_scale)
        sphere = pv.Sphere(
            radius=r,
            center=pos,
            theta_resolution=theta_resolution,
            phi_resolution=phi_resolution,
        )
        meshes.append(sphere)
        per_face_colors.append(
            np.tile(np.array(atom_color(sym), dtype=np.float64), (sphere.n_cells, 1))
        )
    merged = pv.merge(meshes)
    colors = (
        np.concatenate(per_face_colors, axis=0)
        if per_face_colors
        else np.zeros((0, 3), dtype=np.float64)
    )
    return merged, colors


def build_bond_glyphs(
    positions: np.ndarray,
    bonds: Iterable[Tuple[int, int]],
    radius: float = 0.08,
    resolution: int = 12,
):
    """Return a merged :class:`pyvista.PolyData` of cylindrical bonds."""
    import pyvista as pv

    bonds = list(bonds)
    if not bonds:
        return pv.PolyData()
    tubes = []
    for a, b in bonds:
        p0 = positions[a]
        p1 = positions[b]
        center = 0.5 * (p0 + p1)
        direction = p1 - p0
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            continue
        cyl = pv.Cylinder(
            center=center,
            direction=direction / length,
            radius=radius,
            height=length,
            resolution=resolution,
            capping=True,
        )
        tubes.append(cyl)
    return pv.merge(tubes) if tubes else pv.PolyData()


__all__ = [
    "atom_color",
    "atom_radius",
    "replication_translations",
    "primitive_cell_edges",
    "replicated_cell_edges",
    "replicated_atoms",
    "find_bonds",
    "build_bonded_crystal",
    "build_atom_glyphs",
    "build_bond_glyphs",
]
