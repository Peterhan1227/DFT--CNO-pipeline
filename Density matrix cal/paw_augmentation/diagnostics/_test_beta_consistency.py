"""
Cross-check: beta_i = <p~_i | psi~_n> computed via
  (a) the already-validated reciprocal-space route (paw.nonlq.proj), vs
  (b) a direct real-space sum using the same radial splines/Qij used in
      build_real_space_S, at the SAME (unwrapped) Cartesian grid points
      used for D, to pin down the correct normalization factor before
      trusting the full S-matrix construction.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "helper functions"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for paw_overlap (paw_augmentation/)
from ws_cell import read_poscar_structure

from vaspwfc import vaspwfc
from paw import nonlq
from sph_harm import sph_r
from paw_overlap import load_pawpp

# Frozen snapshot, not live Data/ -- see RESULTS.md "data integrity incident"
data_dir = Path(__file__).resolve().parent / "data_snapshot" / "WSe2_mono"
latvec, species, counts, atom_symbols, atom_numbers, frac_coords, cart_coords = \
    read_poscar_structure(data_dir / "POSCAR")
pawpp = load_pawpp(data_dir / "POTCAR")
pawpp_elements = [pp.element.split('_')[0] for pp in pawpp]
elements_idx = [pawpp_elements.index(s) for s in atom_symbols]

wfc = vaspwfc(str(data_dir / "WAVECAR"), lsorbit=False)
Nx, Ny, Nz = wfc._ngrid
Nr = Nx * Ny * Nz
ik = 1
ib = 2  # a W 5p semicore band, strongly affected by augmentation
kvec = wfc._kvecs[ik - 1]

Cg = wfc.readBandCoeff(ispin=1, ikpt=ik, iband=ib, norm=False)

# (a) reciprocal-space beta, validated route
from ase.io import read
atoms = read(str(data_dir / "POSCAR"))
proj = nonlq(atoms, wfc._encut, pawpp, k=kvec, lgam=wfc._lgam, gamma_half=wfc._gam_half)
beta_recip = proj.proj(Cg)
print("beta_recip[:5] =", beta_recip[:5])
print("proj.element_idx =", proj.element_idx)

# (b) real-space route: c_n(r) via our sqrt(Nr)-normalized IFFT convention
gvec = wfc.gvectors(ik)
gx, gy, gz = gvec[:, 0] % Nx, gvec[:, 1] % Ny, gvec[:, 2] % Nz
cg_grid = np.zeros((Nx, Ny, Nz), dtype=np.complex128)
cg_grid[gx, gy, gz] = Cg
c_r = np.fft.ifftn(cg_grid) * np.sqrt(Nr)   # matches main.py / paw_density_matrix.py convention

ix, iy, iz = [a.ravel() for a in np.mgrid[0:Nx, 0:Ny, 0:Nz]]
r_box_frac = np.column_stack([ix / Nx, iy / Ny, iz / Nz])
r_box_cart = r_box_frac @ latvec
c_r_flat = c_r[ix, iy, iz]

# Build real-space projector values P_i(r) for atom 0 (W), including nearby
# periodic images for wraparound, EXACTLY as build_real_space_S does but for
# a single atom, and try both candidate normalizations.
ns = np.arange(-2, 3)
n1, n2, n3 = [a.ravel() for a in np.meshgrid(ns, ns, ns, indexing='ij')]
all_n = np.column_stack([n1, n2, n3])
all_n_cart = all_n @ latvec

iatom = 0
pp = pawpp[elements_idx[iatom]]
rmax_eff = pp.proj_rmax * (pp.NPSRNL - 1) / pp.NPSRNL
lmmax = pp.lmmax

beta_realspace_noNr = np.zeros(lmmax, dtype=np.complex128)
images_cart = cart_coords[iatom] + all_n_cart
for Rimg in images_cart:
    disp = r_box_cart - Rimg[None, :]
    dist = np.linalg.norm(disp, axis=1)
    mask = dist <= rmax_eff
    if not mask.any():
        continue
    disp_m = disp[mask]
    dist_m = dist[mask]
    c_m = c_r_flat[mask]

    Bblock = np.zeros((lmmax, mask.sum()), dtype=np.float64)
    rproj_ylm = [sph_r(disp_m, l).T for l in range(pp.proj_l.max() + 1)]
    iL = 0
    for l, spl_r in zip(pp.proj_l, pp.spl_rproj):
        TLP1 = 2 * l + 1
        rad = spl_r(dist_m)
        Bblock[iL:iL + TLP1, :] = rad * rproj_ylm[l]
        iL += TLP1
    Bblock *= np.sqrt(np.linalg.det(latvec))

    beta_realspace_noNr += Bblock @ c_m

print("\nbeta_realspace / sqrt(Nr) [candidate normalization] first 5:")
print((beta_realspace_noNr / np.sqrt(Nr))[:5])
print("\nbeta_realspace / Nr [alt candidate] first 5:")
print((beta_realspace_noNr / Nr)[:5])
print("\nbeta_realspace raw (no Nr factor) first 5:")
print(beta_realspace_noNr[:5])

print("\nratio (beta_recip / (beta_realspace_noNr/sqrt(Nr)))[:5] =")
ratio = beta_recip[:lmmax] / (beta_realspace_noNr / np.sqrt(Nr))
print(ratio)
