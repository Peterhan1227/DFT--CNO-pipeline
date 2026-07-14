"""
diagnostics/test_fft_ws_invariance.py -- checks that main.py's chain of
representations (plane-wave coefficients -> primitive FFT grid -> WS-cell
grid, all Bloch-phased) is internally consistent at a handful of
representative k-points.

Checks, per representative k-point and per material:

1. Gram-matrix agreement across representations: the same band-pair Gram
   matrix <psi_m|psi_n>, built four independent ways --
     - coefficient space (sum_G C*C)
     - primitive FFT grid, cell-periodic u_nk (no phase)
     - primitive FFT grid, full Bloch psi_nk (with phase)
     - WS-cell grid, reindexed + Bloch-phased
   must agree to numerical precision (Parseval unitarity of ifftn/fftn, and
   invariance of a Gram matrix under any common per-point phase or
   permutation of its arguments).
2. FFT round trip: fftn(ifftn(cg)) recovers the original scattered
   coefficient array to machine precision.
3. Duplicate modulo-grid G indices: no two distinct G-vectors alias to the
   same (gx%Nx, gy%Ny, gz%Nz) FFT grid point (would silently corrupt cg).
4. Direct-Fourier agreement: the FFT-based psi matches an independent direct
   plane-wave sum (helper functions/direct_fourier.py), the same check
   main_v2.py already performs, reused here as a regression diagnostic.
5. Bloch translation covariance: for the WS-cell grid, whose points are
   images of the primitive grid shifted by an integer lattice translation n
   (ws_cell.build_ws_grid_map), psi_ws(p) must equal
   psi_prim(base_index(p)) * exp(2*pi*i * k . n(p)) exactly.
6. WS mapping permutation: ws_cell's (ix,iy,iz) base-index map must be an
   exact bijection onto the primitive FFT grid (every grid point hit exactly
   once) -- not just "distinct after rounding".
7. Gamma-only WAVECARs are explicitly detected (via
   _common.safe_gvectors -> GammaOnlyUnsupported) and reported as
   unsupported rather than silently mishandled.

For SOC datasets, all Gram matrices are the up+down spinor total, matching
main.py's own SOC accumulation convention (main.py:266-267).

The WS-cell grid used here is a self-contained default (center at the
fractional origin, nmax=2) independent of config.py's WS_CENTER, so this
diagnostic doesn't depend on -- or get invalidated by -- config.py being
pointed at a specific material's chosen center.

Read-only; writes diagnostics/output/test_fft_ws_invariance__<material>.json.

Run directly:
    PYTHONPATH=<repo_root>/VaspBandUnfolding PYTHONIOENCODING=utf-8 \
        <python> test_fft_ws_invariance.py [MATERIAL ...]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402
sys.path.insert(0, str(C.PIPELINE_DIR / "helper functions"))
from direct_fourier import fourier_eval_bands  # noqa: E402

WS_CENTER_DEFAULT = [0.0, 0.0, 0.0]
WS_NMAX_DEFAULT = 2
MAX_DIAG_BANDS = 4       # bands used for the (cheap) direct-Fourier check


def _coeffs_nonsoc(wfc, ik, ispin, bands):
    return np.stack([wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=int(ib), norm=True)
                      for ib in bands])


def _coeffs_soc(wfc, ik, ispin, bands, nG):
    Ck = np.stack([wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=int(ib), norm=True)
                   for ib in bands])
    return Ck[:, :nG], Ck[:, nG:]


def _fft_roundtrip_error(Ck, gvec, ngrid):
    Nx, Ny, Nz = ngrid
    Nr = Nx * Ny * Nz
    nb = Ck.shape[0]
    gx, gy, gz = gvec[:, 0] % Nx, gvec[:, 1] % Ny, gvec[:, 2] % Nz
    cg = np.zeros((nb, Nx, Ny, Nz), dtype=np.complex128)
    cg[:, gx, gy, gz] = Ck
    u = np.fft.ifftn(cg, axes=(1, 2, 3))
    cg2 = np.fft.fftn(u, axes=(1, 2, 3))
    return float(np.max(np.abs(cg2 - cg)))


def check_material(material, ispin=1, occ_tol=1e-6):
    data_dir = C.DATA_ROOT / material
    wavecar_path = data_dir / "WAVECAR"
    report = dict(material=material, data_dir=str(data_dir), tol=C.TOL)

    if not (wavecar_path.exists() and (data_dir / "POSCAR").exists()):
        report["status"] = "SKIPPED"
        report["skip_reason"] = "missing WAVECAR or POSCAR"
        return report

    lsorbit, wfc = C.detect_lsorbit(wavecar_path)
    report["lsorbit"] = lsorbit
    Nx, Ny, Nz = (int(x) for x in wfc._ngrid)
    Nr = Nx * Ny * Nz
    ngrid = (Nx, Ny, Nz)

    latvec = C.read_poscar_structure(data_dir / "POSCAR")[0]
    center_cart, _, _ = C.parse_ws_center(WS_CENTER_DEFAULT, "fractional", latvec)
    r_ws_cart, r_ws_frac_cont, prim_indices, translations_all = C.build_ws_grid_map(
        latvec, ngrid, center_cart, nmax=WS_NMAX_DEFAULT
    )
    ix, iy, iz = [a.ravel() for a in np.mgrid[0:Nx, 0:Ny, 0:Nz]]
    r_prim_frac = np.column_stack([ix / Nx, iy / Ny, iz / Nz])

    # 6. WS mapping permutation: exact bijection on integer (ix,iy,iz) triples.
    flat_base = (prim_indices[:, 0].astype(np.int64) * Ny + prim_indices[:, 1]) * Nz \
        + prim_indices[:, 2]
    n_unique_base = len(np.unique(flat_base))
    report["ws_permutation"] = dict(
        Nr=Nr, n_unique_base_indices=int(n_unique_base),
        passed=bool(n_unique_base == Nr),
    )

    reps, n_frac = C.pick_representative_kpoints(wfc, ispin, occ_tol=occ_tol)
    report["representative_kpoints"] = reps

    kpoint_reports = []
    for ik in reps:
        entry = dict(ik=int(ik), k_frac=wfc._kvecs[ik - 1].tolist())
        k_frac = wfc._kvecs[ik - 1]

        try:
            gvec = C.safe_gvectors(wfc, ik)
        except C.GammaOnlyUnsupported as e:
            entry["status"] = "SKIPPED"
            entry["skip_reason"] = str(e)
            kpoint_reports.append(entry)
            continue

        entry["duplicate_grid_indices"] = C.duplicate_grid_indices(gvec, ngrid)

        sel = C.occupied_bands_split(wfc, ik, ispin, occ_tol=occ_tol)
        bands = sel["bands"]
        if len(bands) == 0:
            entry["status"] = "SKIPPED"
            entry["skip_reason"] = "no occupied bands at this k"
            kpoint_reports.append(entry)
            continue
        nG = gvec.shape[0]

        if lsorbit:
            Ck_up, Ck_dn = _coeffs_soc(wfc, ik, ispin, bands, nG)
            channels = [Ck_up, Ck_dn]
        else:
            channels = [_coeffs_nonsoc(wfc, ik, ispin, bands)]

        # 2. FFT round trip (each channel independently).
        roundtrip_err = max(_fft_roundtrip_error(Ck, gvec, ngrid) for Ck in channels)
        entry["fft_roundtrip_max_err"] = roundtrip_err
        entry["fft_roundtrip_passed"] = bool(roundtrip_err < C.TOL["fft_roundtrip"])

        # 1. Gram-matrix agreement across representations (summed over
        #    spinor channels for SOC, single channel otherwise).
        G_coef = np.zeros((len(bands), len(bands)), dtype=np.complex128)
        G_u = np.zeros_like(G_coef)
        G_psi_prim = np.zeros_like(G_coef)
        G_psi_ws = np.zeros_like(G_coef)
        # ifft_bands returns (nb, Nr) flat in the same C-order as
        # mgrid[0:Nx,0:Ny,0:Nz].ravel(), so the WS map's (ix,iy,iz) base
        # index for each WS point converts to a flat index the same way.
        base_flat = (prim_indices[:, 0].astype(np.int64) * Ny
                     + prim_indices[:, 1]) * Nz + prim_indices[:, 2]

        psi_ws_by_channel = []
        psi_prim_by_channel = []
        for Ck in channels:
            G_coef += C.gram_from_coeffs(Ck)
            u_fft = C.ifft_bands(Ck, gvec, ngrid)                       # (nb,Nr) no phase
            psi_prim = u_fft * np.exp(2j * np.pi * (r_prim_frac @ k_frac))[None, :]
            u_ws = u_fft[:, base_flat]
            psi_ws = u_ws * np.exp(2j * np.pi * (r_ws_frac_cont @ k_frac))[None, :]
            G_u += C.gram_from_grid(u_fft)
            G_psi_prim += C.gram_from_grid(psi_prim)
            G_psi_ws += C.gram_from_grid(psi_ws)
            psi_ws_by_channel.append(psi_ws)
            psi_prim_by_channel.append(psi_prim)

        max_diff_coef_vs_u = float(np.max(np.abs(G_coef - G_u)))
        max_diff_u_vs_psiprim = float(np.max(np.abs(G_u - G_psi_prim)))
        max_diff_psiprim_vs_ws = float(np.max(np.abs(G_psi_prim - G_psi_ws)))
        entry["gram_agreement"] = dict(
            coef_vs_primitive_u=max_diff_coef_vs_u,
            primitive_u_vs_primitive_psi=max_diff_u_vs_psiprim,
            primitive_psi_vs_ws_psi=max_diff_psiprim_vs_ws,
            tol=C.TOL["fft_roundtrip"],
            passed=bool(max(max_diff_coef_vs_u, max_diff_u_vs_psiprim,
                             max_diff_psiprim_vs_ws) < 1e-6),
        )

        # 5. Bloch translation covariance on the WS grid:
        #    psi_ws(p) == psi_prim(base_index(p)) * exp(2*pi*i * k.n(p))
        n = translations_all
        phase_pred = np.exp(2j * np.pi * (n @ k_frac))               # (Nr,)
        cov_err = 0.0
        for psi_prim, psi_ws in zip(psi_prim_by_channel, psi_ws_by_channel):
            predicted = psi_prim[:, base_flat] * phase_pred[None, :]
            cov_err = max(cov_err, float(np.max(np.abs(predicted - psi_ws))))
        entry["bloch_translation_covariance"] = dict(
            max_err=cov_err, tol=C.TOL["bloch_covariance"],
            passed=bool(cov_err < C.TOL["bloch_covariance"]),
        )

        # 4. Direct-Fourier agreement (cheap subset: first MAX_DIAG_BANDS
        #    bands, non-SOC channel or SOC up-channel, primitive grid).
        nb_diag = min(MAX_DIAG_BANDS, len(bands))
        Ck_diag = channels[0][:nb_diag]
        psi_fft_diag = (C.ifft_bands(Ck_diag, gvec, ngrid)
                         * np.exp(2j * np.pi * (r_prim_frac @ k_frac))[None, :])
        psi_dir_diag = fourier_eval_bands(
            Ck_diag, gvec, k_frac, r_prim_frac, mode='psi',
            norm_factor=1.0 / np.sqrt(Nr), chunk_size=4096, verbose=False,
        )
        direct_err = float(np.max(np.abs(psi_dir_diag - psi_fft_diag)))
        entry["direct_fourier_agreement"] = dict(
            n_bands_checked=nb_diag, max_err=direct_err,
            tol=C.TOL["direct_fourier"], passed=bool(direct_err < C.TOL["direct_fourier"]),
        )

        entry["status"] = "OK"
        kpoint_reports.append(entry)

    report["kpoints"] = kpoint_reports

    flags = [report["ws_permutation"]["passed"]]
    for kp in kpoint_reports:
        if kp.get("status") != "OK":
            continue
        flags += [
            kp["fft_roundtrip_passed"],
            kp["gram_agreement"]["passed"],
            kp["bloch_translation_covariance"]["passed"],
            kp["direct_fourier_agreement"]["passed"],
            kp["duplicate_grid_indices"]["n_duplicates"] == 0,
        ]
    report["status"] = C.overall_status(*flags)
    return report


def main(materials=None):
    materials = materials or C.available_materials()
    if not materials:
        print("No materials found under Data/.")
        return 1

    overall_ok = True
    for material in materials:
        print(f"=== test_fft_ws_invariance: {material} ===")
        report = check_material(material)
        path = C.write_report("test_fft_ws_invariance", report, material=material)
        overall_ok = overall_ok and (report["status"] != "FAIL")
        wsperm = report.get("ws_permutation", {})
        print(f"  ws_permutation: {wsperm.get('n_unique_base_indices')}/{wsperm.get('Nr')} "
              f"({'OK' if wsperm.get('passed') else 'FAIL'})")
        for kp in report.get("kpoints", []):
            if kp.get("status") != "OK":
                print(f"    ik={kp['ik']:5d}  SKIPPED ({kp.get('skip_reason')})")
                continue
            ga = kp["gram_agreement"]
            print(f"    ik={kp['ik']:5d}  roundtrip={kp['fft_roundtrip_max_err']:.2e}  "
                  f"gram_agree(max)={max(ga['coef_vs_primitive_u'], ga['primitive_u_vs_primitive_psi'], ga['primitive_psi_vs_ws_psi']):.2e}  "
                  f"bloch_cov={kp['bloch_translation_covariance']['max_err']:.2e}  "
                  f"direct_fourier={kp['direct_fourier_agreement']['max_err']:.2e}  "
                  f"dup_G={kp['duplicate_grid_indices']['n_duplicates']}")
        print(f"  status={report['status']}  -> {path}")
        print()

    print("OVERALL:", "PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
