"""
diagnostics/test_input_consistency.py -- cross-checks WAVECAR / POTCAR /
POSCAR / EIGENVAL / KPOINTS for one material *before* trusting anything the
rest of the pipeline computes from them.

What this checks
-----------------
1. Identity: SHA256 + size for each of the 5 standard input files. POTCAR is
   hashed like the others but its text is never parsed or copied into the
   report (only the digest/size), per the diagnostics contract.
2. Lattice: WAVECAR's real-space cell (self._Acell) vs POSCAR's lattice
   vectors.
3. k-grid: EIGENVAL's fractional k-coordinates, per-(k,band) energies, and
   occupations vs the same quantities stored directly in the WAVECAR
   (self._kvecs, self._bands, self._occs) -- these come from two different
   records in two different files and are NOT guaranteed to agree if the
   files were regenerated at different times (exactly the failure mode that
   bit paw_augmentation/, see paw_augmentation/RESULTS.md's "data integrity
   incident").
4. nkpts / nbands consistency across WAVECAR, EIGENVAL, and (best-effort)
   KPOINTS.
5. Weighted occupation count (the electron count main.py's Tr(rho) is
   supposed to reproduce) and an explicit statement of the occupation-number
   spin convention actually observed (see _common.spin_convention_report).

Read-only: never writes into Data/<material>/output/. Writes its own JSON
report to diagnostics/output/test_input_consistency__<material>.json.

Run directly:
    PYTHONPATH=<repo_root>/VaspBandUnfolding PYTHONIOENCODING=utf-8 \
        <python> test_input_consistency.py [MATERIAL ...]
(no arguments -> run against every material under Data/)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def check_material(material, ispin=1, occ_tol=1e-6):
    data_dir = C.DATA_ROOT / material
    report = dict(material=material, data_dir=str(data_dir), tol=C.TOL)

    # 1. Identity ------------------------------------------------------------
    report["hashes"] = C.dataset_hashes(data_dir)

    wavecar_path = data_dir / "WAVECAR"
    if not report["hashes"]["WAVECAR"]["present"]:
        report["status"] = "SKIPPED"
        report["skip_reason"] = "no WAVECAR"
        return report

    lsorbit, wfc = C.detect_lsorbit(wavecar_path)
    report["lsorbit_detected"] = lsorbit
    report["wavecar"] = dict(
        nkpts=int(wfc._nkpts), nbands=int(wfc._nbands), nspin=int(wfc._nspin),
        encut_ev=float(wfc._encut), ngrid=[int(x) for x in wfc._ngrid],
    )

    checks = {}

    # 2. Lattice ---------------------------------------------------------------
    if report["hashes"]["POSCAR"]["present"]:
        latvec_poscar = C.read_poscar_structure(data_dir / "POSCAR")[0]
        checks["lattice"] = C.compare_lattices(wfc._Acell, latvec_poscar)
    else:
        checks["lattice"] = dict(skipped=True, reason="no POSCAR")

    # 3+4. EIGENVAL cross-check --------------------------------------------
    if report["hashes"]["EIGENVAL"]["present"]:
        ev = C.read_eigenval_full(data_dir / "EIGENVAL")

        dims_ok = (ev["nkpts"] == wfc._nkpts and ev["nbands"] == wfc._nbands)
        checks["dims"] = dict(
            eigenval_nkpts=ev["nkpts"], wavecar_nkpts=int(wfc._nkpts),
            eigenval_nbands=ev["nbands"], wavecar_nbands=int(wfc._nbands),
            passed=bool(dims_ok),
        )

        if dims_ok:
            kdiff = np.max(np.abs(ev["kfrac"] - wfc._kvecs))
            checks["kcoord"] = dict(
                max_abs_diff=float(kdiff), tol=C.TOL["kfrac"],
                passed=bool(kdiff < C.TOL["kfrac"]),
            )

            # WAVECAR's own stored eigenvalues (self._bands) vs EIGENVAL,
            # for whichever spin channel EIGENVAL actually has (min of the
            # two ispin extents, since a mismatched ISPIN header would
            # otherwise index out of range).
            ns_cmp = min(ev["ispin_header"], wfc._nspin)
            ediff = np.max(np.abs(ev["energies"][:ns_cmp] - wfc._bands[:ns_cmp]))
            checks["band_energies"] = dict(
                max_abs_diff_ev=float(ediff), tol=C.TOL["energy_ev"],
                passed=bool(ediff < C.TOL["energy_ev"]),
                ispin_compared=int(ns_cmp),
            )

            odiff = np.max(np.abs(ev["occupations"][:ns_cmp] - wfc._occs[:ns_cmp]))
            checks["occupations"] = dict(
                max_abs_diff=float(odiff), tol=C.TOL["occupation"],
                passed=bool(odiff < C.TOL["occupation"]),
            )
        else:
            checks["kcoord"] = dict(skipped=True, reason="dimension mismatch")
            checks["band_energies"] = dict(skipped=True, reason="dimension mismatch")
            checks["occupations"] = dict(skipped=True, reason="dimension mismatch")

        # weighted occupation count + spin convention, from the WAVECAR's
        # own occupations (independent of whether EIGENVAL matched)
        total_occ, per_k, any_halved = C.weighted_occupation_count(
            wfc, ispin, ev["kweights"], occ_tol=occ_tol
        )
        report["weighted_occupation_count"] = dict(
            ispin=ispin, total=total_occ, halving_triggered=any_halved,
            per_k_min=float(per_k.min()), per_k_max=float(per_k.max()),
        )
        report["spin_convention"] = C.spin_convention_report(wfc, ev, ispin_config=ispin)
    else:
        checks["dims"] = dict(skipped=True, reason="no EIGENVAL")
        checks["kcoord"] = dict(skipped=True, reason="no EIGENVAL")
        checks["band_energies"] = dict(skipped=True, reason="no EIGENVAL")
        checks["occupations"] = dict(skipped=True, reason="no EIGENVAL")
        report["spin_convention"] = C.spin_convention_report(wfc, {}, ispin_config=ispin)

    # KPOINTS: best-effort nkpts cross-check
    kp = C.read_kpoints_header(data_dir / "KPOINTS") if report["hashes"]["KPOINTS"]["present"] \
        else dict(present=False)
    if kp.get("nkpts_listed") is not None:
        kp["matches_wavecar_nkpts"] = bool(kp["nkpts_listed"] == wfc._nkpts)
    report["kpoints"] = kp

    report["checks"] = checks
    passed_flags = [v.get("passed") for v in checks.values() if isinstance(v, dict)]
    report["status"] = C.overall_status(*passed_flags)
    return report


def main(materials=None):
    materials = materials or C.available_materials()
    if not materials:
        print("No materials found under Data/.")
        return 1

    overall_ok = True
    for material in materials:
        print(f"=== test_input_consistency: {material} ===")
        report = check_material(material)
        path = C.write_report("test_input_consistency", report, material=material)
        status = report["status"]
        overall_ok = overall_ok and (status != "FAIL")
        print(f"  status={status}")
        for name, chk in report.get("checks", {}).items():
            if not isinstance(chk, dict):
                continue
            if chk.get("skipped"):
                print(f"    {name:18s} SKIPPED ({chk.get('reason')})")
            elif "passed" in chk:
                mark = "OK" if chk["passed"] else "FAIL"
                extra = {k: v for k, v in chk.items() if k not in ("passed",)}
                print(f"    {name:18s} {mark}  {extra}")
        if "weighted_occupation_count" in report:
            print(f"    weighted_occupation_count = "
                  f"{report['weighted_occupation_count']['total']:.6f}")
        print(f"  -> {path}")
        print()

    print("OVERALL:", "PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
