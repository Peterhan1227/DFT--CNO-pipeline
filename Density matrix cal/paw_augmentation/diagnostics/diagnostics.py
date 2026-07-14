"""
Before/after diagnostic table, reproducing the exact baseline recipe from
TASK_BRIEF.md (readBandCoeff(norm=True), plain PW overlap) side-by-side with
the PAW-augmentation-corrected overlap from paw_overlap.py, across the same
5 k-points for WSe2_mono, plus the Si regression check.

Run:
    PYTHONPATH=<repo_root>/VaspBandUnfolding PYTHONIOENCODING=utf-8 \
        <python> diagnostics.py
"""
import json
import numpy as np
from vaspwfc import vaspwfc
from paw_overlap import PawOverlapCorrector, offdiag_maxabs, diag_stats

KPOINTS = [1, 2, 50, 150, 300]

# NOTE: Data/WSe2_mono/WAVECAR was overwritten on disk mid-task (observed
# 2026-07-09, nbands 17->15) -- almost certainly the user's own VASP work on
# the parallel "switch POTCAR" track mentioned in TASK_BRIEF.md, running
# concurrently with this overnight job. To keep all of this task's results
# self-consistent (and not racing a live-changing file), everything below
# reads from a frozen snapshot taken at 2026-07-09T14:20:28Z, saved under
# paw_augmentation/data_snapshot/. See RESULTS.md for full detail.
WSE2_DIR = 'data_snapshot/WSe2_mono'
SI_DIR = 'data_snapshot/Si'


def brief_baseline_overlap(wfc, ik, ispin=1):
    """Exact recipe from TASK_BRIEF.md's diagnostics snippet."""
    occ_all = wfc._occs[ispin - 1, ik - 1, :]
    bands = np.where(occ_all > 1e-6)[0] + 1
    Ck = np.stack([
        wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=int(ib), norm=True)
        for ib in bands
    ])
    overlap = Ck.conj() @ Ck.T
    return bands, overlap


def run_wse2():
    print("=" * 78)
    print("WSe2_mono (W_sv hard/semicore potential) -- PAW correction target")
    print("=" * 78)

    wfc = vaspwfc(f'{WSE2_DIR}/WAVECAR', lsorbit=False)
    corr = PawOverlapCorrector(
        f'{WSE2_DIR}/WAVECAR',
        f'{WSE2_DIR}/POSCAR',
        f'{WSE2_DIR}/POTCAR',
    )

    rows = []
    print(f"{'ik':>5s} {'nbands':>7s} {'baseline(norm=True) max|offdiag|':>34s} "
          f"{'corrected max|offdiag|':>24s} {'corrected diag range':>24s}")
    for ik in KPOINTS:
        bands_b, S_baseline = brief_baseline_overlap(wfc, ik)
        bands_c, S_ps, S_corr = corr.overlaps(ik)
        assert np.array_equal(bands_b, bands_c), "band index mismatch"

        od_before = offdiag_maxabs(S_baseline)
        od_after = offdiag_maxabs(S_corr)
        dmin, dmax, dmad = diag_stats(S_corr)

        print(f"{ik:5d} {len(bands_b):7d} {od_before:34.4e} {od_after:24.4e} "
              f"    [{dmin:.4f},{dmax:.4f}]")

        rows.append(dict(
            ik=ik, nbands=int(len(bands_b)),
            offdiag_baseline=float(od_before),
            offdiag_corrected=float(od_after),
            diag_min=float(dmin), diag_max=float(dmax),
            diag_mean_abs_dev=float(dmad),
        ))
    return rows


def run_si():
    print()
    print("=" * 78)
    print("Si (soft potential, no POTCAR present) -- regression sanity check")
    print("=" * 78)
    print("No POTCAR available for Si in this dataset; PAW correction not")
    print("computed (not needed -- soft potential, baseline already ~1e-8).")
    print("Reproducing brief's exact baseline diagnostic only.")

    wfc = vaspwfc(f'{SI_DIR}/WAVECAR', lsorbit=False)
    rows = []
    print(f"{'ik':>5s} {'nbands':>7s} {'baseline(norm=True) max|offdiag|':>34s}")
    for ik in KPOINTS:
        bands_b, S_baseline = brief_baseline_overlap(wfc, ik)
        od_before = offdiag_maxabs(S_baseline)
        print(f"{ik:5d} {len(bands_b):7d} {od_before:34.4e}")
        rows.append(dict(ik=ik, nbands=int(len(bands_b)), offdiag_baseline=float(od_before)))
    return rows


if __name__ == '__main__':
    wse2_rows = run_wse2()
    si_rows = run_si()

    with open('diagnostic_results.json', 'w') as f:
        json.dump(dict(wse2=wse2_rows, si=si_rows), f, indent=2)
    print("\nWrote diagnostic_results.json")
