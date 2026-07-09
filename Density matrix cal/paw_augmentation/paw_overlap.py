"""
PAW-augmentation-corrected band-pair overlaps for the CNO density-matrix pipeline.

Implements, using the existing (validated) PAW machinery in
VaspBandUnfolding/paw.py (pawpotcar, nonlq) rather than re-deriving the
spherical-Bessel-transform / projector math from scratch:

    <psi_m|S|psi_n> = <psi~_m|psi~_n> + beta_m^dagger . Qij . beta_n

where
    <psi~_m|psi~_n>  = sum_G Cg_m(G)^* Cg_n(G)      (plain PW overlap, raw/
                                                       un-renormalized coeffs)
    beta_n,i         = <p~_i | psi~_n>              (nonlq.proj, reciprocal-
                                                       space projector overlap)
    Qij              = <phi_i^AE|phi_j^AE> - <phi~_i^PS|phi~_j^PS>
                                                     (pawpotcar.get_Qij(),
                                                       block-diagonal over atoms)

This is exactly the formula in TASK_BRIEF.md section 2, but reusing the
existing tested reciprocal-space projector transform (nonlq.calc_qproj) and
POTCAR radial-dataset parser (pawpotcar) instead of re-implementing the
spherical Bessel transform by hand.

Does not touch pysbt / real-space AE reconstruction (aewfc.py) at all -- not
needed for overlap matrix elements, only for pointwise real-space AE
wavefunctions, which the brief explicitly says to avoid in favor of the
generalized-eigenvalue approach.
"""
import numpy as np
from scipy.sparse import block_diag as sp_block_diag

from vaspwfc import vaspwfc
from paw import pawpotcar, nonlq
from ase.io import read


def load_pawpp(potcar_path):
    """Parse all element PAW datasets from a POTCAR file."""
    text = open(potcar_path).read()
    chunks = text.split('End of Dataset')[:-1]
    return [pawpotcar(c) for c in chunks]


def build_qij_block(pawpp, element_idx):
    """
    Block-diagonal Qij matrix stacked in the same (atom, then lm-channel)
    order that nonlq.proj() returns beta_njk in.
    """
    blocks = [pawpp[element_idx[iatom]].get_Qij() for iatom in range(len(element_idx))]
    return sp_block_diag(blocks).toarray()


class PawOverlapCorrector(object):
    """
    Computes both the uncorrected plane-wave-only overlap and the PAW
    augmentation-corrected overlap for occupied bands at a given k-point.
    """

    def __init__(self, wavecar, poscar, potcar):
        self.wfc = vaspwfc(wavecar, lsorbit=False)
        self.atoms = read(poscar)
        self.pawpp = load_pawpp(potcar)
        self._encut = self.wfc._encut
        self._lgam = self.wfc._lgam
        self._gam_half = getattr(self.wfc, '_gam_half', 'x')
        self._proj_cache = {}

    def _get_projector(self, ik):
        if ik not in self._proj_cache:
            kvec = self.wfc._kvecs[ik - 1]
            proj = nonlq(
                self.atoms, self._encut, self.pawpp, k=kvec,
                lgam=self._lgam, gamma_half=self._gam_half,
            )
            self._qij_block = build_qij_block(proj.pawpp, proj.element_idx)
            self._proj_cache[ik] = proj
        return self._proj_cache[ik]

    def occupied_bands(self, ik, ispin=1, occ_thresh=1e-6):
        occ_all = self.wfc._occs[ispin - 1, ik - 1, :]
        return np.where(occ_all > occ_thresh)[0] + 1

    def overlaps(self, ik, ispin=1, bands=None):
        """
        Returns (bands, S_uncorrected, S_corrected) for the given k-point.

        S_uncorrected: plain PW overlap of RAW (un-renormalized) coefficients,
                       i.e. <psi~_m|psi~_n>.
        S_corrected:   <psi~_m|psi~_n> + beta_m^dagger Qij beta_n.

        (Note: this uncorrected matrix built from raw coefficients is NOT
        identical to the brief's "baseline" diagnostic, which uses
        readBandCoeff(norm=True) -- see paw_augmentation/diagnostics.py for
        the exact-brief-recipe baseline. Here we use raw coefficients for
        both so the "correction added" is isolated cleanly.)
        """
        if bands is None:
            bands = self.occupied_bands(ik, ispin)

        proj = self._get_projector(ik)

        Cg_list = []
        beta_list = []
        for ib in bands:
            Cg = self.wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=int(ib), norm=False)
            Cg_list.append(Cg)
            beta_list.append(proj.proj(Cg))

        Ck = np.stack(Cg_list)          # (nbands, nplw)
        Bk = np.stack(beta_list)        # (nbands, nproj_total)

        S_ps = Ck.conj() @ Ck.T
        S_aug = Bk.conj() @ self._qij_block @ Bk.T
        S_corrected = S_ps + S_aug

        return bands, S_ps, S_corrected


def offdiag_maxabs(S):
    n = S.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return np.max(np.abs(S[mask])) if n > 1 else 0.0


def diag_stats(S):
    d = np.diag(S).real
    return d.min(), d.max(), np.mean(np.abs(d - 1.0))


if __name__ == '__main__':
    import sys
    # See diagnostics.py header note: reading from the frozen data snapshot,
    # not Data/WSe2_mono directly, because that file was overwritten mid-task.
    corr = PawOverlapCorrector(
        'data_snapshot/WSe2_mono/WAVECAR',
        'data_snapshot/WSe2_mono/POSCAR',
        'data_snapshot/WSe2_mono/POTCAR',
    )
    for ik in [1, 2, 50, 150, 300]:
        bands, S_ps, S_corr = corr.overlaps(ik)
        od_before = offdiag_maxabs(S_ps)
        od_after = offdiag_maxabs(S_corr)
        dmin, dmax, dmad = diag_stats(S_corr)
        print(f"ik={ik:4d}  nbands={len(bands):3d}  "
              f"max|offdiag| raw-PW={od_before:.3e}  corrected={od_after:.3e}  "
              f"corrected diag in [{dmin:.4f},{dmax:.4f}] (mean|d-1|={dmad:.2e})")
