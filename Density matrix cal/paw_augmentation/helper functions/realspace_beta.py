"""
Zero-padded band-limited real-space reconstruction of Bloch states, and
direct real-space projector-overlap (beta) computation for PAW cross-checks.

Extracted from the original quadrature_convergence_check.py convergence
study (kept in full, unmodified, at
paw_augmentation/diagnostics/quadrature_convergence_check.py for historical
reference) so that paw_regional_cno.py does not need to import a whole
superseded convergence-study script just to reuse these two general-purpose
functions for its real-space cross-check validation.
"""
import numpy as np
from sph_harm import sph_r
from plane_wave import zero_pad_ifft

DIST_PRUNE = 16.0  # Angstrom, same default the original study used


def real_space_beta_for_bands(pawpp, elements_idx, atom_cart, latvec, r_grid_cart,
                               psi_r_bands, Nr, nmax, k_frac, dist_prune=DIST_PRUNE):
    """
    beta_n,i = <p~_i|psi_n> for a batch of BLOCH states via a direct
    real-space sum. Returns only the small (nb, n_proj_total)
    projector-overlap array. Never allocates an Nr x Nr matrix regardless of
    how large Nr is.

      psi_r_bands : (nb, Nr) complex -- the FULL Bloch wavefunction values
                    psi_n(r) = exp(2*pi*i*k_frac.r_frac) * u_n(r) at
                    r_grid_cart. NOT the bare cell-periodic u_n(r).
      k_frac      : (3,) fractional k-point (same convention as r_frac
                    elsewhere in this codebase) -- used for the per-atom-
                    image translation phase exp(-2*pi*i*k_frac.n_image)
                    applied to each image's contribution before summing.
      Nr          : the grid THIS call is using (must be recomputed per grid
                    factor by the caller, never reused from a different grid).
    """
    natoms = atom_cart.shape[0]
    nb = psi_r_bands.shape[0]
    ns = np.arange(-nmax, nmax + 1)
    n1, n2, n3 = [a.ravel() for a in np.meshgrid(ns, ns, ns, indexing='ij')]
    all_n = np.column_stack([n1, n2, n3])              # integer lattice translations
    all_n_cart = all_n @ latvec
    image_phase = np.exp(-2j * np.pi * (all_n @ np.asarray(k_frac)))  # (ntrans,)
    centroid = r_grid_cart.mean(axis=0)

    beta_blocks = []
    for iatom in range(natoms):
        pp = pawpp[elements_idx[iatom]]
        rmax_eff = pp.proj_rmax * (pp.NPSRNL - 1) / pp.NPSRNL
        lmmax = pp.lmmax
        beta_atom = np.zeros((lmmax, nb), dtype=np.complex128)

        images_cart = atom_cart[iatom] + all_n_cart
        d_centroid = np.linalg.norm(images_cart - centroid[None, :], axis=1)
        candidate_idx = np.where(d_centroid < dist_prune)[0]

        for ii in candidate_idx:
            Rimg = images_cart[ii]
            disp = r_grid_cart - Rimg[None, :]
            dist = np.linalg.norm(disp, axis=1)
            mask = dist <= rmax_eff
            if not mask.any():
                continue
            disp_m = disp[mask]
            dist_m = dist[mask]
            c_m = psi_r_bands[:, mask]  # (nb, npts)

            Bblock = np.zeros((lmmax, mask.sum()), dtype=np.complex128)
            rproj_ylm = [sph_r(disp_m, l).T for l in range(pp.proj_l.max() + 1)]
            iL = 0
            for l, spl_r in zip(pp.proj_l, pp.spl_rproj):
                TLP1 = 2 * l + 1
                rad = spl_r(dist_m)
                Bblock[iL:iL + TLP1, :] = rad * rproj_ylm[l]
                iL += TLP1
            Bblock *= np.sqrt(np.linalg.det(latvec)) * image_phase[ii]

            beta_atom += Bblock @ c_m.T  # (lmmax, npts) @ (npts, nb) -> (lmmax, nb)

        beta_blocks.append(beta_atom / np.sqrt(Nr))

    return np.concatenate(beta_blocks, axis=0).T  # (nb, n_proj_total)
