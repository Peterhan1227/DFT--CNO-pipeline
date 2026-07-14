"""
diagnostics/test_fixed_k_gram.py -- band-pair Gram-matrix diagnostics at a
handful of representative k-points, for both non-SOC and SOC (spinor)
WAVECARs.

At a fixed k-point, occupied Kohn-Sham states must be mutually orthonormal:
<psi_m|psi_n> = delta_mn. This script builds that (small, nb x nb) Gram
matrix three different ways and compares them:

  1. raw coefficients      -- readBandCoeff(norm=False), plain sum_G C*C
  2. norm=True coefficients -- readBandCoeff(norm=True) (main.py's own
     convention), plain sum_G C*C
  3. PAW-corrected          -- raw coefficients plus the existing reciprocal-
     space augmentation correction from paw_augmentation/paw_overlap.py
     (<psi_m|S|psi_n> = <psi~_m|psi~_n> + beta_m^H Qij beta_n), only when a
     POTCAR is available for this material

For SOC (LSORBIT) datasets, readBandCoeff returns the up- and down-spinor
coefficients concatenated; the Gram matrix is built by summing the up and
down channels' contributions separately (both plain and PAW-corrected),
matching main.py's own SOC accumulation `rho += wk*(psi_up^H psi_up +
psi_dn^H psi_dn)` (main.py:266-267).

Everything here is an nb x nb band-pair matrix (nb = number of occupied
bands at that k) -- never an Nr x Nr real-space projector.

Occupied bands are additionally split into 'binary' (occ ~= 1) and
'fractional' (0 << occ << 1, i.e. smeared/metallic) groups, and Gram-matrix
stats are reported for each group separately, since main.py's SOC branch
currently omits the `occ` weight entirely (flagged, not fixed, at the bottom
of this report) -- exactly the fractional-occupation bands are where that
omission would matter.

Read-only; writes diagnostics/output/test_fixed_k_gram__<material>.json.

Run directly:
    PYTHONPATH=<repo_root>/VaspBandUnfolding PYTHONIOENCODING=utf-8 \
        <python> test_fixed_k_gram.py [MATERIAL ...]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def _sub_gram_stats(G, mask, tol):
    """Gram-matrix stats restricted to the bands selected by `mask`."""
    if mask.sum() == 0:
        return dict(n=0)
    sub = G[np.ix_(mask, mask)]
    stats = C.gram_stats(sub, tol=tol)
    stats["n"] = int(mask.sum())
    return stats


def _band_pair_gram_nonsoc(wfc, ik, ispin, bands):
    Ck_raw = np.stack([wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=int(ib), norm=False)
                        for ib in bands])
    Ck_norm = np.stack([wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=int(ib), norm=True)
                         for ib in bands])
    return C.gram_from_coeffs(Ck_raw), C.gram_from_coeffs(Ck_norm)


def _band_pair_gram_soc(wfc, ik, ispin, bands):
    gvec = C.safe_gvectors(wfc, ik)
    nG = gvec.shape[0]

    def _gram(norm):
        Ck = np.stack([wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=int(ib), norm=norm)
                        for ib in bands])
        up, dn = Ck[:, :nG], Ck[:, nG:]
        return C.gram_from_coeffs(up) + C.gram_from_coeffs(dn)

    return _gram(False), _gram(True)


def check_material(material, ispin=1, occ_tol=1e-6):
    data_dir = C.DATA_ROOT / material
    wavecar_path = data_dir / "WAVECAR"
    report = dict(material=material, data_dir=str(data_dir), tol=C.TOL)

    if not wavecar_path.exists():
        report["status"] = "SKIPPED"
        report["skip_reason"] = "no WAVECAR"
        return report

    try:
        lsorbit, wfc = C.detect_lsorbit(wavecar_path)
    except Exception as e:
        report["status"] = "FAIL"
        report["error"] = f"could not open WAVECAR: {e}"
        return report

    report["lsorbit"] = lsorbit
    has_potcar = C.paw_correction_available(data_dir)
    report["potcar_available"] = has_potcar
    if not has_potcar:
        report["paw_correction_note"] = "skipped: no POTCAR for this material"

    reps, n_frac_at_best = C.pick_representative_kpoints(wfc, ispin, occ_tol=occ_tol)
    report["representative_kpoints"] = reps
    report["max_fractional_bands_at_any_k"] = n_frac_at_best

    kpoint_reports = []
    for ik in reps:
        sel = C.occupied_bands_split(wfc, ik, ispin, occ_tol=occ_tol)
        bands, occ = sel["bands"], sel["occ"]
        entry = dict(
            ik=int(ik), k_frac=wfc._kvecs[ik - 1].tolist(),
            n_occupied=int(len(bands)),
            n_binary=int(sel["binary_mask"].sum()),
            n_fractional=int(sel["fractional_mask"].sum()),
            halved=sel["halved"],
        )
        if len(bands) == 0:
            entry["status"] = "SKIPPED"
            entry["skip_reason"] = "no occupied bands at this k"
            kpoint_reports.append(entry)
            continue

        try:
            if lsorbit:
                G_raw, G_norm = _band_pair_gram_soc(wfc, ik, ispin, bands)
            else:
                G_raw, G_norm = _band_pair_gram_nonsoc(wfc, ik, ispin, bands)
        except C.GammaOnlyUnsupported as e:
            entry["status"] = "SKIPPED"
            entry["skip_reason"] = str(e)
            kpoint_reports.append(entry)
            continue

        entry["gram_raw"] = C.gram_stats(G_raw, tol=C.TOL["gram_corrected"])
        entry["gram_norm_true"] = C.gram_stats(G_norm, tol=C.TOL["gram_corrected"])
        entry["gram_raw_binary_subset"] = _sub_gram_stats(
            G_raw, sel["binary_mask"], C.TOL["gram_corrected"])
        entry["gram_raw_fractional_subset"] = _sub_gram_stats(
            G_raw, sel["fractional_mask"], C.TOL["gram_corrected"])

        if has_potcar:
            try:
                if lsorbit:
                    S_ps, S_corr = C.build_soc_paw_gram(data_dir, ik, bands, ispin=ispin)
                else:
                    S_ps, S_corr = C.build_nonsoc_paw_gram(data_dir, ik, ispin, bands)
                # Sanity: the corrector's own "plain PW" overlap of raw
                # coefficients must agree with our independently-built G_raw
                # (both are <psi~_m|psi~_n> from the same raw coefficients).
                consistency = float(np.max(np.abs(S_ps - G_raw)))
                entry["paw_ps_vs_gram_raw_consistency"] = consistency
                entry["gram_paw_corrected"] = C.gram_stats(S_corr, tol=C.TOL["gram_corrected"])
                entry["gram_paw_corrected_binary_subset"] = _sub_gram_stats(
                    S_corr, sel["binary_mask"], C.TOL["gram_corrected"])
                entry["gram_paw_corrected_fractional_subset"] = _sub_gram_stats(
                    S_corr, sel["fractional_mask"], C.TOL["gram_corrected"])
            except Exception as e:
                entry["paw_correction_error"] = str(e)

        entry["status"] = "OK"
        kpoint_reports.append(entry)

    report["kpoints"] = kpoint_reports

    # Overall pass/fail is judged only on the PAW-corrected Gram (the raw
    # and norm=True Grams are "before" reference measurements, not something
    # with a pass criterion -- a large raw off-diagonal on a hard/semicore
    # potential is the expected, correct finding, not a bug).
    corrected_flags = [
        kp["gram_paw_corrected"]["passed"]
        for kp in kpoint_reports
        if kp.get("status") == "OK" and "gram_paw_corrected" in kp
    ]
    report["status"] = C.overall_status(*corrected_flags) if has_potcar else "SKIPPED"
    return report


def main(materials=None):
    materials = materials or C.available_materials()
    if not materials:
        print("No materials found under Data/.")
        return 1

    for material in materials:
        print(f"=== test_fixed_k_gram: {material} ===")
        report = check_material(material)
        path = C.write_report("test_fixed_k_gram", report, material=material)
        print(f"  lsorbit={report.get('lsorbit')}  potcar_available={report.get('potcar_available')}")
        for kp in report.get("kpoints", []):
            if kp.get("status") != "OK":
                print(f"    ik={kp['ik']:5d}  SKIPPED ({kp.get('skip_reason')})")
                continue
            gr, gn = kp["gram_raw"], kp["gram_norm_true"]
            line = (f"    ik={kp['ik']:5d}  n_occ={kp['n_occupied']:3d} "
                    f"(binary={kp['n_binary']}, frac={kp['n_fractional']})  "
                    f"raw: max_offdiag={gr['max_offdiag']:.2e} "
                    f"diag_err={gr['max_diag_err']:.2e}  "
                    f"norm=True: max_offdiag={gn['max_offdiag']:.2e}")
            if "gram_paw_corrected" in kp:
                gc = kp["gram_paw_corrected"]
                line += (f"  corrected: max_offdiag={gc['max_offdiag']:.2e} "
                         f"({'OK' if gc['passed'] else 'FAIL'})")
            print(line)
        print(f"  status={report['status']}  -> {path}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
