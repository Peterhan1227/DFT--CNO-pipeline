"""
psd_convergence_diagnostic.py -- rigorous, real-geometry convergence
diagnostic for the G_A_psd violation found at WSe2_mono's "Center 2"
(Se-Se hollow site) WS center, min_eig(G_A) ~= -0.021 on the 5-k-point
reduced reference. DIAGNOSTIC ONLY: does not change any production code,
does not clip eigenvalues, does not adjust tolerances, does not authorize
a new physical approximation. See paw_regional_cno.py and RESULTS.md.

G_A = G_ps_A + G_aug_A mixes two INDEPENDENT spatial quadratures:
  1. G_ps_A = Psi^H Psi on the native FFT/WS grid (smooth/pseudo part).
  2. G_aug_A = sum_site beta_site^H Q_A_site beta_site, where Q_A_site
     comes from radial x angular integration with a WS/Voronoi membership
     test (the augmentation part).
This script refines EACH quadrature independently (holding the other
fixed) to determine which one (if either) the -0.021 eigenvalue is
sensitive to, using spectral (operator) norms throughout -- an elementwise
per-Q_A tolerance of 1e-4 can produce a much larger eigenvalue shift after
contraction over many states, so max-abs alone is not trusted here.

Sections, matching the task spec:
  F. geometry-only candidate site enumeration + triage (no dependence on
     whether a native FFT grid point happens to land near an image).
  A. build states/Psi/beta ONCE (neither depends on angular resolution).
  B. angular (Q_A) resolution sweep on the REAL WS geometry.
  C. Rayleigh decomposition of the most-negative G_A eigenvector, at every
     angular level.
  D. FFT/WS-grid resolution sweep (Q_A frozen at the finest angular
     level), rebuilding the WS grid map fresh at each grid factor.
  E. explicit convergence+PSD gate (reported only -- this script never
     calls the production main(), so it cannot itself unblock a run).
  G. results saved to JSON + printed as a compact table.
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

MATERIAL = "WSe2_mono"
WS_CENTER = [2 / 3, 1 / 3, 0.5]
WS_CENTER_COORD_TYPE = "fractional"

ANGULAR_LEVELS = [(64, 128), (96, 192), (128, 256), (192, 384)]
GRID_FACTORS = [1, 2, 3]
TRIAGE_LEVEL = (64, 128)
TRIAGE_NORM_TOL = 1e-8   # spectral-norm floor to call a candidate site "negligible"

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)


def spectral_norm(M):
    return float(np.linalg.norm(np.atleast_2d(M), ord=2))


def mineig(M):
    return float(np.linalg.eigvalsh(0.5 * (M + M.conj().T)).min())


# ── F: geometry-only candidate enumeration (no native-FFT-point dependence) ──

def enumerate_candidate_sites_geometric(pawpp, elements_idx, atom_cart, latvec, center_cart,
                                         r_ws_cart, nmax=prc.SITE_SEARCH_NMAX):
    """Every (atom, image) whose sphere could plausibly reach the WS cell,
    kept by a pure geometric distance bound -- NOT by whether a native FFT
    grid POINT happens to land nearby (that is what find_contributing_sites
    does, which item F says this diagnostic must not rely on). The bound
    itself is the actual spatial reach of the WS cell (max distance from
    center_cart to any of its own r_ws_cart points, i.e. the cell's own
    geometric extent -- a property of the cell's shape, not of grid
    density) plus a margin, rather than an isotropic bound on the largest
    lattice vector: for a slab with a large vacuum direction, that
    isotropic bound is enormous and keeps hundreds of physically-irrelevant
    candidates (whole vacuum-separated periodic images) that a WS-cell-
    shaped bound correctly excludes.
    """
    cell_reach = float(np.max(np.linalg.norm(r_ws_cart - center_cart[None, :], axis=1)))
    safety_bound = 1.5 * cell_reach
    ns = np.arange(-nmax, nmax + 1)
    n1, n2, n3 = [a.ravel() for a in np.meshgrid(ns, ns, ns, indexing='ij')]
    all_n = np.column_stack([n1, n2, n3])
    all_n_cart = all_n @ latvec

    candidates = []
    for iatom, ei in enumerate(elements_idx):
        pp = pawpp[ei]
        rmax_eff = pp.proj_rmax * (pp.NPSRNL - 1) / pp.NPSRNL
        images_cart = atom_cart[iatom] + all_n_cart
        d = np.linalg.norm(images_cart - center_cart[None, :], axis=1)
        keep = np.where(d < rmax_eff + safety_bound)[0]
        for ii in keep:
            candidates.append(dict(iatom=int(iatom), element=pp.element, pp_idx=int(ei),
                                    image=tuple(int(x) for x in all_n[ii]),
                                    image_cart=images_cart[ii], rmax_eff=rmax_eff))
    print(f"  geometric candidate search: nmax={nmax}, cell_reach={cell_reach:.3f} A, "
          f"safety_bound={safety_bound:.3f} A -> {len(candidates)} raw candidates")
    return candidates


def triage_sites(pawpp, candidates, center_cart, latvec, triage_level=TRIAGE_LEVEL, tol=TRIAGE_NORM_TOL):
    """Compute Q_A at a cheap triage resolution for every candidate and
    keep only those with non-negligible spectral norm -- the site list
    this diagnostic actually uses everywhere below."""
    nt, npi = triage_level
    kept = []
    for c in candidates:
        pp = pawpp[c["pp_idx"]]
        Q_A = prc.build_regional_Qij_site(pp, c["image_cart"], center_cart, latvec,
                                           n_theta=nt, n_phi=npi)
        norm = spectral_norm(Q_A)
        if norm > tol:
            c = dict(c)
            c["triage_norm"] = norm
            kept.append(c)
    return kept


def verify_partition(pawpp, sites, finest_Q_by_site):
    """sum_images Q_A(atom,image) == Q_full(atom), per atom, using the
    FINEST-level Q_A -- a post-hoc check that the geometric+triage site
    list found above is actually complete (item F, final sentence)."""
    by_atom = {}
    for s in sites:
        by_atom.setdefault(s["iatom"], []).append(s)
    print("  partition check (finest angular level, geometry-only site list):")
    all_ok = True
    rows = []
    for iatom, atom_sites in by_atom.items():
        pp = pawpp[atom_sites[0]["pp_idx"]]
        full_Q = pp.get_Qij()
        Q_sum = sum(finest_Q_by_site[(s["iatom"], s["image"])] for s in atom_sites)
        err = float(np.max(np.abs(Q_sum - full_Q)))
        ok = bool(err < 1e-3)
        all_ok = all_ok and ok
        rows.append(dict(iatom=iatom, element=pp.element, n_images=len(atom_sites), max_abs_err=err, passed=ok))
        print(f"    atom {iatom} ({pp.element}): {len(atom_sites)} images, max|err|={err:.4e}  "
              f"{'OK' if ok else 'FAIL'}")
    return dict(rows=rows, passed=all_ok)


# ── C: Rayleigh decomposition of a fixed vector against G_ps/G_aug/G_A ──

def rayleigh_decomposition(v, G_ps_A, G_aug_by_site, beta_by_site, sites):
    """real(v^H G_ps_A v), real(v^H G_aug_A v), real(v^H G_A v), and each
    site's own real(v^H beta_site^H Q_A_site beta_site v), sorted by
    |contribution| descending."""
    vps = complex(np.vdot(v, G_ps_A @ v))
    site_contribs = []
    total_aug = 0.0 + 0.0j
    for i, s in enumerate(sites):
        Q_A = G_aug_by_site[(s["iatom"], s["image"])]
        w = beta_by_site[i].T @ v            # (lmmax,)
        contrib = complex(np.vdot(w, Q_A @ w))
        total_aug += contrib
        site_contribs.append(dict(iatom=s["iatom"], element=s["element"], image=s["image"],
                                   contribution=float(contrib.real)))
    site_contribs.sort(key=lambda d: -abs(d["contribution"]))
    return dict(
        v_G_ps_A_v=float(vps.real), v_G_aug_A_v=float(total_aug.real),
        v_G_A_v=float(vps.real + total_aug.real), site_contributions=site_contribs,
    )


def build_G_A(G_ps_A, beta_by_site, sites, Q_by_site):
    G_aug_A = np.zeros_like(G_ps_A)
    for i, s in enumerate(sites):
        Q_A = Q_by_site[(s["iatom"], s["image"])]
        G_aug_A += beta_by_site[i].conj() @ Q_A @ beta_by_site[i].T
    return G_aug_A, G_ps_A + G_aug_A


def build_K_A(G_A, p):
    sqrtP = np.sqrt(p)
    K_A = (sqrtP[:, None] * G_A) * sqrtP[None, :]
    return 0.5 * (K_A + K_A.conj().T)


def main():
    t0 = time.time()
    results = dict(material=MATERIAL, ws_center=WS_CENTER, ws_center_coord_type=WS_CENTER_COORD_TYPE)

    sys_ = prc.load_system(material=MATERIAL, ws_center=WS_CENTER,
                            ws_center_coord_type=WS_CENTER_COORD_TYPE)
    kpoints = sys_["reduced_reference_kpoints"]
    print(f"=== PSD convergence diagnostic: {MATERIAL}, WS_CENTER={WS_CENTER}, "
          f"{len(kpoints)} reduced-reference k-points ===\n")

    # ── F: geometry-only site enumeration + triage ──
    print("--- F. Geometry-only candidate site enumeration (no native-FFT-point dependence) ---")
    candidates = enumerate_candidate_sites_geometric(
        sys_["pawpp"], sys_["elements_idx"], sys_["cart_coords"], sys_["latvec"], sys_["center_cart"],
        sys_["r_ws_cart"])
    sites = triage_sites(sys_["pawpp"], candidates, sys_["center_cart"], sys_["latvec"])
    print(f"  {len(sites)} sites retained after triage (spectral norm > {TRIAGE_NORM_TOL:.0e}):")
    for s in sites:
        print(f"    atom {s['iatom']} ({s['element']}) image={s['image']} "
              f"triage_norm={s['triage_norm']:.4e}")
    results["sites"] = [dict(iatom=s["iatom"], element=s["element"], image=s["image"],
                              triage_norm=s["triage_norm"]) for s in sites]

    # ── A: build states / Psi / beta ONCE (angular-resolution independent) ──
    print("\n--- A. Building states, Psi (native grid), beta_by_site (once) ---")
    Psi_native, beta_by_site, p, states = prc.build_state_list_and_beta(sys_, sites, kpoints)
    print(f"  {len(states)} states, Psi_native shape={Psi_native.shape}")
    G_ps_A_native = Psi_native.conj().T @ Psi_native

    # ── B: angular (Q_A) resolution sweep on the REAL geometry ──
    print("\n--- B. Angular (Q_A) resolution sweep, real WS geometry ---")
    Q_by_level = {}       # level -> {(iatom,image): Q_A}
    prev_level = None
    for level in ANGULAR_LEVELS:
        nt, npi = level
        Q_here = {}
        for s in sites:
            pp = sys_["pawpp"][s["pp_idx"]]
            Q_here[(s["iatom"], s["image"])] = prc.build_regional_Qij_site(
                pp, s["image_cart"], sys_["center_cart"], sys_["latvec"], n_theta=nt, n_phi=npi)
        Q_by_level[level] = Q_here

        G_aug_A, G_A = build_G_A(G_ps_A_native, beta_by_site, sites, Q_here)
        K_A = build_K_A(G_A, p)

        row = dict(n_theta=nt, n_phi=npi,
                   min_eig_G_A=mineig(G_A), min_eig_K_A=mineig(K_A),
                   trace_G_A=float(np.trace(G_A).real), trace_K_A=float(np.trace(K_A).real))
        if prev_level is not None:
            Q_prev, G_aug_prev, G_A_prev = prev_level
            max_site_spec = max(spectral_norm(Q_here[k] - Q_prev[k]) for k in Q_here)
            max_site_maxabs = max(float(np.max(np.abs(Q_here[k] - Q_prev[k]))) for k in Q_here)
            row["max_site_spectral_norm_delta_Q"] = max_site_spec
            row["max_site_maxabs_delta_Q"] = max_site_maxabs
            row["spectral_norm_delta_G_aug"] = spectral_norm(G_aug_A - G_aug_prev)
            row["spectral_norm_delta_G_A"] = spectral_norm(G_A - G_A_prev)
        else:
            row["max_site_spectral_norm_delta_Q"] = None
            row["max_site_maxabs_delta_Q"] = None
            row["spectral_norm_delta_G_aug"] = None
            row["spectral_norm_delta_G_A"] = None
        results["angular_sweep"] = results.get("angular_sweep", []) + [row]
        print(f"  ({nt:3d}x{npi:3d})  min_eig(G_A)={row['min_eig_G_A']: .6f}  "
              f"min_eig(K_A)={row['min_eig_K_A']: .6f}  trace(G_A)={row['trace_G_A']:.6f}  "
              f"dSpecQ={row['max_site_spectral_norm_delta_Q']}  dSpecG_A={row['spectral_norm_delta_G_A']}")

        prev_level = (Q_here, G_aug_A, G_A)

    # freeze the most-negative eigenvector of the BASE (64x128) level's G_A
    Q_base = Q_by_level[ANGULAR_LEVELS[0]]
    G_aug_base, G_A_base = build_G_A(G_ps_A_native, beta_by_site, sites, Q_base)
    eigvals_base, eigvecs_base = np.linalg.eigh(0.5 * (G_A_base + G_A_base.conj().T))
    i_min = int(np.argmin(eigvals_base))
    v = eigvecs_base[:, i_min]
    v = v / np.linalg.norm(v)
    print(f"\n  frozen negative eigenvector: level={ANGULAR_LEVELS[0]}, "
          f"eigval={eigvals_base[i_min]:.6f}, |v|={np.linalg.norm(v):.6f}")
    results["frozen_eigval_base_level"] = float(eigvals_base[i_min])

    # ── C: Rayleigh decomposition at every angular level ──
    print("\n--- C. Rayleigh decomposition of the frozen negative eigenvector, per angular level ---")
    for level in ANGULAR_LEVELS:
        decomp = rayleigh_decomposition(v, G_ps_A_native, Q_by_level[level], beta_by_site, sites)
        results["negative_vector_angular"] = results.get("negative_vector_angular", []) + \
            [dict(n_theta=level[0], n_phi=level[1], **decomp)]
        print(f"  ({level[0]:3d}x{level[1]:3d})  v.G_ps_A.v={decomp['v_G_ps_A_v']: .6f}  "
              f"v.G_aug_A.v={decomp['v_G_aug_A_v']: .6f}  v.G_A.v={decomp['v_G_A_v']: .6f}")
        for sc in decomp["site_contributions"]:
            print(f"      atom {sc['iatom']} ({sc['element']}) image={sc['image']}: "
                  f"{sc['contribution']: .6f}")

    # ── D: FFT/WS-grid resolution sweep, Q_A frozen at the finest angular level ──
    print("\n--- D. FFT/WS-grid resolution sweep (Q_A frozen at finest angular level) ---")
    Q_finest = Q_by_level[ANGULAR_LEVELS[-1]]
    partition_report = verify_partition(sys_["pawpp"], sites, Q_finest)
    results["partition_check_finest_level"] = partition_report

    Nx0, Ny0, Nz0 = sys_["Nx"], sys_["Ny"], sys_["Nz"]
    wfc = sys_["wfc"]
    ispin = sys_["ispin"]
    channel_ranges = sys_["channel_ranges"]

    prev_grid = None
    for f in GRID_FACTORS:
        Nxf, Nyf, Nzf = Nx0 * f, Ny0 * f, Nz0 * f
        t_grid0 = time.time()
        r_ws_cart_f, r_ws_frac_cont_f, prim_indices_f, _ = build_ws_grid_map(
            sys_["latvec"], (Nxf, Nyf, Nzf), sys_["center_cart"], nmax=sys_["ws_nmax"])
        base_flat_f = (prim_indices_f[:, 0].astype(np.int64) * Nyf + prim_indices_f[:, 1]) * Nzf \
            + prim_indices_f[:, 2]

        # rebuild Psi at this grid factor for the SAME states (same k/band
        # order as `states`/`p` from section A) -- beta is grid-independent
        # and is NOT rebuilt.
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
        t_grid = time.time() - t_grid0

        row = dict(grid_factor=f, Nx=Nxf, Ny=Nyf, Nz=Nzf,
                   min_eig_G_A=mineig(G_A_f), min_eig_K_A=mineig(K_A_f),
                   trace_G_A=float(np.trace(G_A_f).real), trace_K_A=float(np.trace(K_A_f).real),
                   elapsed_s=t_grid)
        if prev_grid is not None:
            G_ps_prev, G_A_prev = prev_grid
            row["spectral_norm_delta_G_ps"] = spectral_norm(G_ps_A_f - G_ps_prev)
            row["spectral_norm_delta_G_A"] = spectral_norm(G_A_f - G_A_prev)
        else:
            row["spectral_norm_delta_G_ps"] = None
            row["spectral_norm_delta_G_A"] = None
        results["grid_sweep"] = results.get("grid_sweep", []) + [row]
        print(f"  f={f}  ({Nxf}x{Nyf}x{Nzf})  min_eig(G_A)={row['min_eig_G_A']: .6f}  "
              f"min_eig(K_A)={row['min_eig_K_A']: .6f}  dSpecG_ps={row['spectral_norm_delta_G_ps']}  "
              f"dSpecG_A={row['spectral_norm_delta_G_A']}  ({t_grid:.1f}s)")

        decomp = rayleigh_decomposition(v, G_ps_A_f, Q_finest, beta_by_site, sites)
        results["negative_vector_grid"] = results.get("negative_vector_grid", []) + \
            [dict(grid_factor=f, **decomp)]
        print(f"      v.G_ps_A.v={decomp['v_G_ps_A_v']: .6f}  v.G_aug_A.v={decomp['v_G_aug_A_v']: .6f}  "
              f"v.G_A.v={decomp['v_G_A_v']: .6f}")

        prev_grid = (G_ps_A_f, G_A_f)

    # ── E: explicit convergence + PSD gate (reported only) ──
    last_angular = results["angular_sweep"][-1]
    last_grid = results["grid_sweep"][-1]
    angular_converged = (last_angular["spectral_norm_delta_G_A"] is not None
                          and last_angular["spectral_norm_delta_G_A"] < 1e-3)
    grid_converged = (last_grid["spectral_norm_delta_G_A"] is not None
                       and last_grid["spectral_norm_delta_G_A"] < 1e-3)
    is_psd = last_grid["min_eig_G_A"] > -1e-3
    gate_passed = angular_converged and grid_converged and is_psd
    results["gate"] = dict(angular_converged=angular_converged, grid_converged=grid_converged,
                            is_psd=is_psd, passed=gate_passed)
    print("\n--- E. Real-geometry convergence + PSD gate (report only; this script "
          "cannot itself authorize the full 324-k build) ---")
    print(f"  angular_converged (dSpecG_A<1e-3 at finest level): {angular_converged}")
    print(f"  grid_converged    (dSpecG_A<1e-3 at finest level): {grid_converged}")
    print(f"  is_psd            (min_eig(G_A)>-1e-3 at finest levels): {is_psd}")
    print(f"  GATE: {'PASS' if gate_passed else 'FAIL'}")

    results["total_runtime_s"] = time.time() - t0
    with open(OUT / "psd_convergence_diagnostic.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating)
                   else int(o) if isinstance(o, np.integer) else list(o) if isinstance(o, (tuple, np.ndarray))
                   else str(o))
    print(f"\nSaved -> {OUT / 'psd_convergence_diagnostic.json'}")
    print(f"Total runtime: {results['total_runtime_s']:.1f}s")


if __name__ == "__main__":
    main()
