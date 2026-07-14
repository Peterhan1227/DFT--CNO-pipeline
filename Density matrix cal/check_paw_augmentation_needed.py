"""
check_paw_augmentation_needed.py -- fast pre-check for whether a material's
WS-cell choice actually needs paw_augmentation/paw_regional_cno.py's
expensive region-intersected quadrature, or whether the WS cell happens to
never split an atom's augmentation sphere -- in which case Q_A degenerates
everywhere to either the full atomic Qij or exactly zero, and the ordinary
(non-regional) reciprocal PAW treatment already used elsewhere in this
pipeline (paw_augmentation/paw_overlap.py) is EXACT for this material/
WS-center choice, with no need to run the regional pipeline at all.

Needs only POSCAR + POTCAR + the WS_CENTER settings from config.py -- no
WAVECAR, no k-point/band loop. Runs in well under a second.

Test (pure geometry): for each (atom, image) site close enough to the WS
center to matter, sample the WS/Voronoi membership test
(helper functions/direct_fourier.py's ws_membership) on the projector
sphere's OUTER SURFACE (r = rmax_eff) at a moderately dense set of angular
directions. The WS cell is convex (a Voronoi cell always is), and a ball is
the convex hull of its bounding sphere, so: if every sampled surface point
has the SAME membership as the atom, the entire solid sphere -- not just
the sampled shell -- is guaranteed to lie on that one side of the WS
boundary. If sampled points disagree, the sphere straddles the boundary
and region-splitting genuinely matters for that site.

This is deliberately coarser than paw_regional_cno.py's own quadrature
(no radial integration, no integration weights, no attempt to size the
"how much" of a straddle) -- it only answers the binary question "does
region-splitting matter here at all", which is what determines whether the
expensive pipeline needs to run in the first place.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "helper functions"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "paw_augmentation"))

from ws_cell import read_poscar_structure, parse_ws_center  # noqa: E402
from direct_fourier import ws_membership  # noqa: E402
from paw_overlap import load_pawpp  # noqa: E402

import config  # noqa: E402  -- read-only

N_THETA = 24                 # coarse angular sampling of the sphere SURFACE only --
N_PHI = 48                    # no radial integration needed, see module docstring
SITE_SEARCH_NMAX = 4          # atom-image search range (matches paw_regional_cno.py's convention)
DIST_PRUNE = 16.0             # Angstrom, candidate-image prefilter


def _surface_directions(n_theta=N_THETA, n_phi=N_PHI):
    """Unit vectors on a theta/phi grid -- membership sampling only, no
    integration weights needed (see module docstring)."""
    theta = np.arccos(np.linspace(1.0, -1.0, n_theta))
    phi = np.arange(n_phi) * (2 * np.pi / n_phi)
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    return np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)], axis=-1).reshape(-1, 3)


def check_paw_augmentation_needed(material=None, ws_center=None, ws_center_coord_type=None, ws_nmax=None,
                                   verbose=True):
    """Returns a report dict with needs_regional_augmentation (bool) and the
    list of straddling / fully-inside / fully-outside sites."""
    material = material if material is not None else config.MATERIAL
    ws_center = ws_center if ws_center is not None else config.WS_CENTER
    ws_center_coord_type = ws_center_coord_type if ws_center_coord_type is not None \
        else config.WS_CENTER_COORD_TYPE
    ws_nmax = ws_nmax if ws_nmax is not None else config.WS_TRANSLATION_SEARCH_RANGE

    data_dir = Path(__file__).resolve().parent / "Data" / material
    latvec, species, counts, atom_symbols, atom_numbers, frac_coords, cart_coords = \
        read_poscar_structure(data_dir / "POSCAR")
    pawpp = load_pawpp(data_dir / "POTCAR")
    pawpp_elements = [pp.element.split("_")[0] for pp in pawpp]
    elements_idx = [pawpp_elements.index(s) for s in atom_symbols]

    center_cart, _, _ = parse_ws_center(ws_center, ws_center_coord_type, latvec)
    directions = _surface_directions()

    ns = np.arange(-SITE_SEARCH_NMAX, SITE_SEARCH_NMAX + 1)
    n1, n2, n3 = [a.ravel() for a in np.meshgrid(ns, ns, ns, indexing="ij")]
    all_n = np.column_stack([n1, n2, n3])
    all_n_cart = all_n @ latvec

    fully_inside, fully_outside, straddling = [], [], []
    for iatom, ei in enumerate(elements_idx):
        pp = pawpp[ei]
        rmax_eff = pp.proj_rmax * (pp.NPSRNL - 1) / pp.NPSRNL
        images_cart = cart_coords[iatom] + all_n_cart
        d_center = np.linalg.norm(images_cart - center_cart[None, :], axis=1)
        candidates = np.where(d_center < DIST_PRUNE)[0]
        for ii in candidates:
            Rimg = images_cart[ii]
            surface_pts = Rimg[None, :] + rmax_eff * directions
            inside = ws_membership(surface_pts, center_cart, latvec, nmax=ws_nmax)
            frac_inside = float(inside.mean())
            entry = dict(iatom=iatom, element=pp.element, image=tuple(int(x) for x in all_n[ii]),
                         rmax_eff=float(rmax_eff), frac_surface_inside=frac_inside)
            if inside.all():
                fully_inside.append(entry)
            elif not inside.any():
                fully_outside.append(entry)
            else:
                straddling.append(entry)

    needs_regional = len(straddling) > 0

    if verbose:
        print(f"=== PAW augmentation region-splitting pre-check: {material} ===")
        print(f"WS center: {ws_center} ({ws_center_coord_type})  ->  {np.round(center_cart, 4)} Ang")
        print(f"contributing sites within {DIST_PRUNE} Ang: "
              f"{len(fully_inside)} fully inside, {len(fully_outside)} fully outside, "
              f"{len(straddling)} STRADDLING\n")
        if straddling:
            print("straddling sites (region-splitting matters here):")
            for e in straddling:
                print(f"  atom {e['iatom']:3d} ({e['element']})  image={e['image']}  "
                      f"rmax_eff={e['rmax_eff']:.3f} Ang  frac_surface_inside={e['frac_surface_inside']:.2f}")
            print(f"\nVERDICT: paw_regional_cno.py's region-intersected quadrature IS needed "
                  f"-- {len(straddling)} augmentation sphere(s) straddle the WS boundary.")
        else:
            print("VERDICT: every augmentation sphere is entirely inside or entirely outside the "
                  "WS cell -- Q_A is exactly the full atomic Qij or exactly zero everywhere. The "
                  "expensive region-intersected quadrature in paw_regional_cno.py is NOT needed for "
                  "this material/WS-center choice; the ordinary reciprocal PAW treatment "
                  "(paw_augmentation/paw_overlap.py) is already exact here.")

    return dict(material=material, ws_center=ws_center, ws_center_coord_type=ws_center_coord_type,
                needs_regional_augmentation=needs_regional,
                fully_inside=fully_inside, fully_outside=fully_outside, straddling=straddling)


if __name__ == "__main__":
    check_paw_augmentation_needed()
