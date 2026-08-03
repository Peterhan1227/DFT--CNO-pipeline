"""
psd_grid_sweep_extended.py -- follow-up to psd_convergence_diagnostic.py:
the angular (Q_A) sweep already showed no convergence trend (flat/
oscillating), so this script skips it and goes straight to the finest
angular level, then extends the FFT/WS-grid sweep to f=4,5 to see whether
min_eig(G_A) actually converges to PSD or keeps drifting. Diagnostic only.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "helper functions"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helper functions"))

import paw_regional_cno as prc  # noqa: E402
from ws_cell import build_ws_grid_map  # noqa: E402
from realspace_beta import zero_pad_ifft  # noqa: E402
from psd_convergence_diagnostic import (  # noqa: E402
    MATERIAL, WS_CENTER, WS_CENTER_COORD_TYPE, TRIAGE_LEVEL,
    enumerate_candidate_sites_geometric, triage_sites, spectral_norm, mineig,
    rayleigh_decomposition, build_G_A, build_K_A,
)

FINEST_ANGULAR = (192, 384)
GRID_FACTORS = [1, 2, 3, 4, 5]

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)


def main():
    t0 = time.time()
    sys_ = prc.load_system(material=MATERIAL, ws_center=WS_CENTER, ws_center_coord_type=WS_CENTER_COORD_TYPE)
    kpoints = sys_["reduced_reference_kpoints"]
    print(f"=== Extended grid-factor sweep: {MATERIAL}, WS_CENTER={WS_CENTER} ===\n")

    candidates = enumerate_candidate_sites_geometric(
        sys_["pawpp"], sys_["elements_idx"], sys_["cart_coords"], sys_["latvec"], sys_["center_cart"],
        sys_["r_ws_cart"])
    sites = triage_sites(sys_["pawpp"], candidates, sys_["center_cart"], sys_["latvec"], triage_level=TRIAGE_LEVEL)
    print(f"  {len(sites)} sites retained after triage")

    Psi_native, beta_by_site, p, states = prc.build_state_list_and_beta(sys_, sites, kpoints)
    print(f"  {len(states)} states")

    nt, npi = FINEST_ANGULAR
    print(f"\n  building Q_A at finest angular level ({nt}x{npi}) only (angular sweep already answered)...")
    Q_finest = {}
    for s in sites:
        pp = sys_["pawpp"][s["pp_idx"]]
        Q_finest[(s["iatom"], s["image"])] = prc.build_regional_Qij_site(
            pp, s["image_cart"], sys_["center_cart"], sys_["latvec"], n_theta=nt, n_phi=npi)

    G_ps_A_native = Psi_native.conj().T @ Psi_native
    G_aug_A_native, G_A_native = build_G_A(G_ps_A_native, beta_by_site, sites, Q_finest)
    eigvals_native, eigvecs_native = np.linalg.eigh(0.5 * (G_A_native + G_A_native.conj().T))
    i_min = int(np.argmin(eigvals_native))
    v = eigvecs_native[:, i_min]
    v = v / np.linalg.norm(v)
    print(f"  frozen negative eigenvector (native grid, finest angular): eigval={eigvals_native[i_min]:.6f}")

    Nx0, Ny0, Nz0 = sys_["Nx"], sys_["Ny"], sys_["Nz"]
    wfc = sys_["wfc"]
    ispin = sys_["ispin"]

    results = dict(material=MATERIAL, ws_center=WS_CENTER, finest_angular=FINEST_ANGULAR,
                    frozen_eigval_native=float(eigvals_native[i_min]), grid_sweep=[])
    prev = None
    for f in GRID_FACTORS:
        Nxf, Nyf, Nzf = Nx0 * f, Ny0 * f, Nz0 * f
        t_g0 = time.time()
        r_ws_cart_f, r_ws_frac_cont_f, prim_indices_f, _ = build_ws_grid_map(
            sys_["latvec"], (Nxf, Nyf, Nzf), sys_["center_cart"], nmax=sys_["ws_nmax"])
        base_flat_f = (prim_indices_f[:, 0].astype(np.int64) * Nyf + prim_indices_f[:, 1]) * Nzf \
            + prim_indices_f[:, 2]

        Psi_f = np.empty((Nxf * Nyf * Nzf, len(states)), dtype=np.complex128)
        by_k = {}
        for idx, st in enumerate(states):
            by_k.setdefault(st["ik"], []).append((idx, st["band"]))
        for ik, idx_band_pairs in by_k.items():
            kvec = wfc._kvecs[ik - 1]
            bands = [b for _, b in idx_band_pairs]
            idxs = [i for i, _ in idx_band_pairs]
            gvec = wfc.gvectors(ik)
            Ck = np.stack([wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=int(ib), norm=False)
                           for ib in bands])
            u_bands_f, _, _ = zero_pad_ifft(Ck, gvec, f, (Nx0, Ny0, Nz0))
            u_ws_f = u_bands_f[:, base_flat_f]
            psi_ws_f = u_ws_f * np.exp(2j * np.pi * (r_ws_frac_cont_f @ kvec))[None, :]
            Psi_f[:, idxs] = psi_ws_f.T

        G_ps_A_f = Psi_f.conj().T @ Psi_f
        G_aug_A_f, G_A_f = build_G_A(G_ps_A_f, beta_by_site, sites, Q_finest)
        K_A_f = build_K_A(G_A_f, p)
        t_g = time.time() - t_g0

        row = dict(grid_factor=f, Nx=Nxf, Ny=Nyf, Nz=Nzf,
                   min_eig_G_A=mineig(G_A_f), min_eig_K_A=mineig(K_A_f),
                   trace_G_A=float(np.trace(G_A_f).real), elapsed_s=t_g)
        if prev is not None:
            row["spectral_norm_delta_G_ps"] = spectral_norm(G_ps_A_f - prev[0])
            row["spectral_norm_delta_G_A"] = spectral_norm(G_A_f - prev[1])
        else:
            row["spectral_norm_delta_G_ps"] = None
            row["spectral_norm_delta_G_A"] = None
        decomp = rayleigh_decomposition(v, G_ps_A_f, Q_finest, beta_by_site, sites)
        row["v_G_ps_A_v"] = decomp["v_G_ps_A_v"]
        row["v_G_aug_A_v"] = decomp["v_G_aug_A_v"]
        row["v_G_A_v"] = decomp["v_G_A_v"]
        results["grid_sweep"].append(row)
        print(f"  f={f}  ({Nxf}x{Nyf}x{Nzf})  min_eig(G_A)={row['min_eig_G_A']: .6f}  "
              f"v.G_A.v={row['v_G_A_v']: .6f}  dSpecG_A={row['spectral_norm_delta_G_A']}  ({t_g:.1f}s)")
        prev = (G_ps_A_f, G_A_f)

    results["total_runtime_s"] = time.time() - t0
    with open(OUT / "psd_grid_sweep_extended.json", "w") as fjson:
        json.dump(results, fjson, indent=2, default=lambda o: float(o) if isinstance(o, np.floating)
                   else int(o) if isinstance(o, np.integer) else list(o) if isinstance(o, (tuple, np.ndarray))
                   else str(o))
    print(f"\nSaved -> {OUT / 'psd_grid_sweep_extended.json'}")
    print(f"Total runtime: {results['total_runtime_s']:.1f}s")


if __name__ == "__main__":
    main()
