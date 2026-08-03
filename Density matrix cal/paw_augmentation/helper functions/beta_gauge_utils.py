"""
Gauge correction for reciprocal-space PAW projector overlaps, and EIGENVAL
k-weight parsing.

Extracted from the original paw_lowrank_cno.py low-rank-CNO experiment (kept
in full, unmodified, at paw_augmentation/diagnostics/paw_lowrank_cno.py for
historical reference) so that paw_regional_cno.py does not need to import a
whole superseded pipeline just to reuse these two general-purpose functions.
"""
import numpy as np

OCC_TOL = 1e-6


def read_eigenval_kweights(path, nkpts_expected, nbands_expected):
    with open(path) as fh:
        lines = fh.readlines()
    nkpts = int(lines[5].split()[1])
    nbands = int(lines[5].split()[2])
    if nkpts != nkpts_expected or nbands != nbands_expected:
        raise ValueError("EIGENVAL/WAVECAR dimension mismatch")
    kweights = np.zeros(nkpts)
    idx = 6
    for ik in range(nkpts):
        while not lines[idx].split():
            idx += 1
        kweights[ik] = float(lines[idx].split()[3])
        idx += 1 + nbands_expected
    kweights /= kweights.sum()
    return kweights


def read_eigenval_energies(path, nkpts_expected, nbands_expected):
    """Parse per-band energies from a BZ-mesh EIGENVAL: (nkpts, nbands),
    spin-up (column 1) only -- same file, same layout read_eigenval_kweights
    reads for k-weights, just also keeping the per-band energy column this
    time. Used only for RESTRICT_TO_FERMI_WINDOW-style band selection
    (main.py's _read_eigenval does the equivalent parse); raises ValueError
    on a dimension mismatch, same convention as read_eigenval_kweights."""
    with open(path) as fh:
        lines = fh.readlines()
    nkpts = int(lines[5].split()[1])
    nbands = int(lines[5].split()[2])
    if nkpts != nkpts_expected or nbands != nbands_expected:
        raise ValueError("EIGENVAL/WAVECAR dimension mismatch")
    energies = np.zeros((nkpts, nbands))
    idx = 6
    for ik in range(nkpts):
        while not lines[idx].split():
            idx += 1
        idx += 1
        for ib in range(nbands):
            energies[ik, ib] = float(lines[idx].split()[1])
            idx += 1
    return energies


def gauge_correct_beta(beta_recip, k_frac, elements_idx, pawpp, frac_coords):
    """Multiply each atom's projector-channel block of a paw.nonlq.proj()
    beta array by exp(2*pi*i * k_frac . tau_atom_frac), converting
    nonlq.proj()'s "bare-G atom-position phase" gauge into the direct
    real-space-integral gauge that G_ps (and the true S operator) use. A
    no-op for same-k pairing (the added phase cancels there); required for
    cross-k pairing.

    beta_recip : (nb, n_proj_total) -- one k-point's worth of states (or a
                 single state), atom-then-lm-channel ordered.
    k_frac     : (3,) fractional k-point of THESE states.
    """
    beta_out = beta_recip.copy()
    off = 0
    for iatom, ei in enumerate(elements_idx):
        lm = pawpp[ei].lmmax
        phase = np.exp(2j * np.pi * np.dot(k_frac, frac_coords[iatom]))
        beta_out[:, off:off + lm] *= phase
        off += lm
    return beta_out
