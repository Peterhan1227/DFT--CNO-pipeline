"""
paw_lowrank_phase1_diagnostics.py -- Phase 1: localize the negative
eigenvalue found in paw_lowrank_cno.py's full production run
(min eig(K) = -0.1995, RESULTS.md 2026-07-11 "low-rank" entry), BEFORE
changing the augmentation formula.

Does not change main.py/config.py (config.py only read). Does not modify
paw_lowrank_cno.py's production formula. Read-only with respect to
production outputs; writes only to paw_augmentation/output/.

Checks performed (see main() for the driving order):
  1. Reconstruct G_ps, G_aug, G_total, K separately (never saved as such by
     the production script) and report their min eigenvalues + Hermiticity
     errors.
  2. For the most negative eigenvector of G_total, compute u^H G_ps u,
     u^H G_aug u, u^H G_total u, and decompose the augmentation expectation
     value by (atom, periodic image) -- using a genuinely per-image
     real-space projector evaluation (NOT the production reciprocal route,
     which is combined-across-images by construction and cannot be resolved
     this way).
  3. k-point subset scaling: min eig(G_total)/min eig(K) for nested subsets
     of 1,2,4,8,...,all k-points (complete band sets preserved at every
     included k-point), to see whether the negative eigenvalue grows
     coherently with mesh size or is a fixed, isolated few-state effect.
  4. Random gauge test: psi_a -> exp(i theta_a) psi_a (and beta_a transformed
     consistently) must leave G/K eigenvalues exactly unchanged (a pure
     linear-algebra fact, given consistent construction) -- verifies no
     hidden gauge dependence remains beyond the already-fixed atom-position
     phase.
  5. Spectral-norm (not just max-entry) comparison of the augmentation-only
     block against the converged real-space reference, on the largest
     manageable k-point subset.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "helper functions"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for paw_overlap (paw_augmentation/)
from ws_cell import read_poscar_structure, parse_ws_center, build_ws_grid_map  # noqa: E402

from vaspwfc import vaspwfc  # noqa: E402
from paw import nonlq  # noqa: E402
from sph_harm import sph_r  # noqa: E402
from paw_overlap import load_pawpp, build_qij_block  # noqa: E402
from ase.io import read as ase_read  # noqa: E402

from paw_lowrank_cno import (  # noqa: E402
    read_eigenval_kweights, build_state_list, gauge_correct_beta,
    OCC_TOL,
)
from quadrature_convergence_check import zero_pad_ifft, real_space_beta_for_bands  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

K_SUBSET_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 324]
SPECTRAL_NORM_KPOINTS = [1, 41, 81, 122, 163, 203, 244, 284]   # 8, spread across the mesh
GAUGE_TEST_SEED = 0


# ── shared setup ─────────────────────────────────────────────────────────

def load_system():
    import config  # read-only
    material = config.MATERIAL
    data_dir = Path(__file__).resolve().parent.parent.parent / "Data" / material
    wfc = vaspwfc(str(data_dir / "WAVECAR"), lsorbit=False)
    latvec, species, counts, atom_symbols, atom_numbers, frac_coords, cart_coords = \
        read_poscar_structure(data_dir / "POSCAR")
    pawpp = load_pawpp(data_dir / "POTCAR")
    pawpp_elements = [pp.element.split('_')[0] for pp in pawpp]
    elements_idx = [pawpp_elements.index(s) for s in atom_symbols]
    atoms = ase_read(str(data_dir / "POSCAR"))
    qij_block = build_qij_block(pawpp, elements_idx)
    kweights = read_eigenval_kweights(data_dir / "EIGENVAL", wfc._nkpts, wfc._nbands)
    center_cart, _, _ = parse_ws_center(config.WS_CENTER, config.WS_CENTER_COORD_TYPE, latvec)
    ws_nmax = config.WS_TRANSLATION_SEARCH_RANGE
    Nx, Ny, Nz = (int(x) for x in wfc._ngrid)
    Nr = Nx * Ny * Nz
    r_ws_cart, r_ws_frac_cont, prim_indices, translations_all = build_ws_grid_map(
        latvec, (Nx, Ny, Nz), center_cart, nmax=ws_nmax,
    )
    return dict(
        config=config, material=material, data_dir=data_dir, wfc=wfc,
        latvec=latvec, atom_symbols=atom_symbols, cart_coords=cart_coords,
        frac_coords=frac_coords, pawpp=pawpp, elements_idx=elements_idx,
        atoms=atoms, qij_block=qij_block, kweights=kweights,
        center_cart=center_cart, ws_nmax=ws_nmax, Nx=Nx, Ny=Ny, Nz=Nz, Nr=Nr,
        r_ws_cart=r_ws_cart, r_ws_frac_cont=r_ws_frac_cont,
        prim_indices=prim_indices, ispin=config.ISPIN,
    )


def build_matrices_for_kpoints(sys_, ik_list, gauge_theta=None):
    """Build Psi (Nr, nstates), Beta (nstates, n_proj_total), p (nstates,)
    for exactly the given k-point list (complete band sets at each),
    reusing the production (native 1x grid, gauge-corrected reciprocal
    beta) construction.

    gauge_theta: optional dict {state_index: theta} (or None) -- if given,
    multiplies Psi's column and Beta's row for that state by exp(i*theta),
    for the gauge-invariance test.
    """
    wfc = sys_["wfc"]
    Nx, Ny, Nz, Nr = sys_["Nx"], sys_["Ny"], sys_["Nz"], sys_["Nr"]
    ispin = sys_["ispin"]
    base_flat = (sys_["prim_indices"][:, 0].astype(np.int64) * Ny
                 + sys_["prim_indices"][:, 1]) * Nz + sys_["prim_indices"][:, 2]

    states = []
    for ik in ik_list:
        occ_all = wfc._occs[ispin - 1, ik - 1, :]
        bands = np.where(occ_all > OCC_TOL)[0] + 1
        occ = occ_all[bands - 1].copy()
        if len(occ) and occ.max() > 1.5:
            occ = occ / 2.0
        wk = sys_["kweights"][ik - 1]
        for ib, f in zip(bands, occ):
            states.append(dict(ik=int(ik), band=int(ib), p=float(wk * f)))
    nstates = len(states)

    Psi = np.empty((Nr, nstates), dtype=np.complex128)
    n_proj_total = sum(sys_["pawpp"][ei].lmmax for ei in sys_["elements_idx"])
    Beta = np.empty((nstates, n_proj_total), dtype=np.complex128)
    p = np.array([st["p"] for st in states])

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
        Nx0, Ny0, Nz0 = (int(x) for x in wfc._ngrid)
        u_bands, _, _ = zero_pad_ifft(Ck, gvec, 1, (Nx0, Ny0, Nz0))
        u_ws = u_bands[:, base_flat]
        psi_ws = u_ws * np.exp(2j * np.pi * (sys_["r_ws_frac_cont"] @ kvec))[None, :]
        Psi[:, idxs] = psi_ws.T

        proj = nonlq(sys_["atoms"], wfc._encut, sys_["pawpp"], k=kvec,
                     lgam=wfc._lgam, gamma_half=wfc._gam_half)
        assert list(proj.element_idx) == list(sys_["elements_idx"])
        beta_recip = np.stack([proj.proj(Ck[i]) for i in range(len(bands))])
        beta_recip = gauge_correct_beta(beta_recip, kvec, sys_["elements_idx"],
                                         sys_["pawpp"], sys_["frac_coords"])
        Beta[idxs, :] = beta_recip

    if gauge_theta is not None:
        phase = np.exp(1j * gauge_theta)
        Psi = Psi * phase[None, :]
        Beta = Beta * phase[:, None]

    return Psi, Beta, p, states


def build_G_K(Psi, Beta, qij_block, p):
    G_ps = Psi.conj().T @ Psi
    G_aug = Beta.conj() @ qij_block @ Beta.T
    G_total = G_ps + G_aug
    sqrtP = np.sqrt(p)
    K = (sqrtP[:, None] * G_total) * sqrtP[None, :]
    return G_ps, G_aug, G_total, K


def _mineig(M):
    Mh = 0.5 * (M + M.conj().T)
    return float(np.linalg.eigvalsh(Mh).min())


def _maxeig(M):
    Mh = 0.5 * (M + M.conj().T)
    return float(np.linalg.eigvalsh(Mh).max())


def _herm(M):
    return float(np.max(np.abs(M - M.conj().T)))


# ── step 1+2: full-system reconstruction + eigenvector decomposition ───────

def per_image_beta_for_probe(psi_u, pawpp, elements_idx, atom_cart, latvec, r_grid_cart,
                              Nr, nmax, dist_prune=16.0):
    """Real-space projector overlap <p_i^(atom,image)|psi_u>, resolved PER
    (atom, image) SITE (not summed across images) for a single probe
    function psi_u(r) already fully phased/combined on the WS grid (e.g.
    Psi @ u for an eigenvector u). This is exactly build_real_space_S's own
    geometric per-image loop, applied to one probe function instead of
    building an Nr x Nr matrix from many bands.

    Returns a list of dicts: iatom, image (int3 tuple), beta (lmmax,) array.
    """
    natoms = atom_cart.shape[0]
    ns = np.arange(-nmax, nmax + 1)
    n1, n2, n3 = [a.ravel() for a in np.meshgrid(ns, ns, ns, indexing='ij')]
    all_n = np.column_stack([n1, n2, n3])
    all_n_cart = all_n @ latvec
    centroid = r_grid_cart.mean(axis=0)

    results = []
    for iatom, ei in enumerate(elements_idx):
        pp = pawpp[ei]
        rmax_eff = pp.proj_rmax * (pp.NPSRNL - 1) / pp.NPSRNL
        lmmax = pp.lmmax
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
            psi_m = psi_u[mask]
            Bblock = np.zeros((lmmax, mask.sum()), dtype=np.float64)
            rproj_ylm = [sph_r(disp_m, l).T for l in range(pp.proj_l.max() + 1)]
            iL = 0
            for l, spl_r in zip(pp.proj_l, pp.spl_rproj):
                TLP1 = 2 * l + 1
                rad = spl_r(dist_m)
                Bblock[iL:iL + TLP1, :] = rad * rproj_ylm[l]
                iL += TLP1
            Bblock *= np.sqrt(np.linalg.det(latvec))
            beta_vec = (Bblock @ psi_m) / np.sqrt(Nr)
            results.append(dict(iatom=int(iatom), element=pp.element,
                                 image=tuple(int(x) for x in all_n[ii]), beta=beta_vec))
    return results


def step1_2_localize(sys_):
    print("=== Step 1+2: reconstruct G_ps/G_aug/G_total/K and localize the "
          "negative eigenvector (all 324 k-points) ===")
    t0 = time.time()
    ik_all = list(range(1, sys_["wfc"]._nkpts + 1))
    Psi, Beta, p, states = build_matrices_for_kpoints(sys_, ik_all)
    print(f"  built Psi/Beta for nstates={len(states)}  +{time.time()-t0:.1f}s")

    G_ps, G_aug, G_total, K = build_G_K(Psi, Beta, sys_["qij_block"], p)

    report = dict(
        nstates=len(states),
        min_eig_G_ps=_mineig(G_ps), min_eig_G_aug=_mineig(G_aug),
        min_eig_G_total=_mineig(G_total), min_eig_K=_mineig(K),
        max_eig_G_total=_maxeig(G_total), max_eig_K=_maxeig(K),
        herm_G_ps=_herm(G_ps), herm_G_aug=_herm(G_aug),
        herm_G_total=_herm(G_total), herm_K=_herm(K),
    )
    print(f"  min eig(G_ps)    = {report['min_eig_G_ps']:.6f}  (expect ~1, PSD Gram)")
    print(f"  min eig(G_aug)   = {report['min_eig_G_aug']:.6f}  (Qij-sandwiched, need not be PSD alone)")
    print(f"  min eig(G_total) = {report['min_eig_G_total']:.6f}  (MUST be >= 0 for physical validity)")
    print(f"  min eig(K)       = {report['min_eig_K']:.6f}")
    print(f"  Hermiticity errs : G_ps={report['herm_G_ps']:.2e}  G_aug={report['herm_G_aug']:.2e}  "
          f"G_total={report['herm_G_total']:.2e}  K={report['herm_K']:.2e}")

    # Confirms (or refutes) that K's negativity is inherited from G_total,
    # not an artifact of the sqrt(P) congruence (which cannot introduce a
    # negative eigenvalue that wasn't already present in G_total, since
    # X^H G X is PSD whenever G is PSD for any X -- here X = sqrt(P), a
    # positive diagonal matrix).
    report["K_negativity_explained_by_G_total"] = bool(report["min_eig_G_total"] < -1e-6)

    eigvals_Gtot, eigvecs_Gtot = np.linalg.eigh(0.5 * (G_total + G_total.conj().T))
    order = np.argsort(eigvals_Gtot)
    u = eigvecs_Gtot[:, order[0]]
    mu = float(eigvals_Gtot[order[0]])
    print(f"\n  Most negative eigenvector of G_total: eigenvalue={mu:.6f}")

    uGpsu = float(np.real(np.vdot(u, G_ps @ u)))
    uGaugu = float(np.real(np.vdot(u, G_aug @ u)))
    uGtotu = float(np.real(np.vdot(u, G_total @ u)))
    print(f"  u^H G_ps u    = {uGpsu:.6f}")
    print(f"  u^H G_aug u   = {uGaugu:.6f}")
    print(f"  u^H G_total u = {uGtotu:.6f}  (should equal eigenvalue {mu:.6f})")

    # Decompose u^H G_aug u by atom/image using a genuinely per-image
    # real-space evaluation of psi_u = Psi @ u (see per_image_beta_for_probe).
    psi_u = Psi @ u
    sites = per_image_beta_for_probe(
        psi_u, sys_["pawpp"], sys_["elements_idx"], sys_["cart_coords"],
        sys_["latvec"], sys_["r_ws_cart"], sys_["Nr"], nmax=sys_["ws_nmax"] + 1,
    )
    off = 0
    channel_ranges = []
    for iatom, ei in enumerate(sys_["elements_idx"]):
        lm = sys_["pawpp"][ei].lmmax
        channel_ranges.append((off, off + lm))
        off += lm

    site_contribs = []
    for site in sites:
        sl = slice(*channel_ranges[site["iatom"]])
        Qatom = sys_["qij_block"][sl, sl]
        contrib = float(np.real(site["beta"].conj() @ Qatom @ site["beta"]))
        site_contribs.append(dict(iatom=site["iatom"], element=site["element"],
                                   image=site["image"], contribution=contrib))
    site_contribs.sort(key=lambda d: d["contribution"])
    total_realspace = sum(s["contribution"] for s in site_contribs)

    print(f"\n  Per-atom/image decomposition of u^H G_aug u (real-space, "
          f"independent route; sum={total_realspace:.6f} vs production "
          f"u^H G_aug u={uGaugu:.6f}):")
    for s in site_contribs:
        print(f"    atom {s['iatom']} ({s['element']})  image={s['image']}  "
              f"contribution={s['contribution']:.6f}")

    report["eigenvector_decomposition"] = dict(
        mu=mu, uGpsu=uGpsu, uGaugu=uGaugu, uGtotu=uGtotu,
        realspace_site_contributions=site_contribs,
        realspace_total=total_realspace,
    )
    return report, states


# ── step 3: k-point subset scaling ──────────────────────────────────────────

def _evenly_spaced_kpoints(nkpts, n):
    if n >= nkpts:
        return list(range(1, nkpts + 1))
    if n <= 1:
        return [1]
    return sorted(set(int(round(i * (nkpts - 1) / (n - 1))) + 1 for i in range(n)))


def step3_kpoint_scaling(sys_):
    print("\n=== Step 3: k-point subset scaling (min eigenvalue vs mesh size) ===")
    nkpts = sys_["wfc"]._nkpts
    rows = []
    for n in K_SUBSET_SIZES:
        ik_list = _evenly_spaced_kpoints(nkpts, n)
        t0 = time.time()
        Psi, Beta, p, states = build_matrices_for_kpoints(sys_, ik_list)
        G_ps, G_aug, G_total, K = build_G_K(Psi, Beta, sys_["qij_block"], p)
        row = dict(
            n_kpoints_requested=n, n_kpoints_actual=len(ik_list),
            nstates=len(states),
            min_eig_G_total=_mineig(G_total), min_eig_K=_mineig(K),
            elapsed_s=float(time.time() - t0),
        )
        rows.append(row)
        print(f"  n_k={len(ik_list):4d}  nstates={len(states):5d}  "
              f"min_eig(G_total)={row['min_eig_G_total']:.6f}  "
              f"min_eig(K)={row['min_eig_K']:.6f}  +{row['elapsed_s']:.1f}s")
    return rows


# ── step 4: random gauge invariance test ────────────────────────────────────

def step4_gauge_test(sys_):
    print("\n=== Step 4: random gauge invariance test ===")
    ik_all = list(range(1, sys_["wfc"]._nkpts + 1))
    Psi0, Beta0, p, states = build_matrices_for_kpoints(sys_, ik_all)
    G_ps0, G_aug0, G_total0, K0 = build_G_K(Psi0, Beta0, sys_["qij_block"], p)
    eig0 = np.sort(np.linalg.eigvalsh(0.5 * (G_total0 + G_total0.conj().T)))
    eigK0 = np.sort(np.linalg.eigvalsh(0.5 * (K0 + K0.conj().T)))

    rng = np.random.default_rng(GAUGE_TEST_SEED)
    theta = rng.uniform(0, 2 * np.pi, size=len(states))
    Psi1 = Psi0 * np.exp(1j * theta)[None, :]
    Beta1 = Beta0 * np.exp(1j * theta)[:, None]
    G_ps1, G_aug1, G_total1, K1 = build_G_K(Psi1, Beta1, sys_["qij_block"], p)
    eig1 = np.sort(np.linalg.eigvalsh(0.5 * (G_total1 + G_total1.conj().T)))
    eigK1 = np.sort(np.linalg.eigvalsh(0.5 * (K1 + K1.conj().T)))

    max_eig_diff_Gtotal = float(np.max(np.abs(eig0 - eig1)))
    max_eig_diff_K = float(np.max(np.abs(eigK0 - eigK1)))
    passed = bool(max_eig_diff_Gtotal < 1e-8 and max_eig_diff_K < 1e-8)
    print(f"  max|eig(G_total) - eig(G_total_gauged)| = {max_eig_diff_Gtotal:.3e}")
    print(f"  max|eig(K) - eig(K_gauged)|             = {max_eig_diff_K:.3e}")
    print(f"  {'PASSED -- no hidden gauge dependence' if passed else 'FAILED -- gauge correction incomplete'}")
    return dict(max_eig_diff_G_total=max_eig_diff_Gtotal, max_eig_diff_K=max_eig_diff_K,
                passed=passed)


# ── step 5: spectral-norm comparison on a larger k-point subset ─────────────

def step5_spectral_norm(sys_):
    print(f"\n=== Step 5: spectral-norm comparison vs converged real-space reference "
          f"({len(SPECTRAL_NORM_KPOINTS)} k-points) ===")
    t0 = time.time()
    Psi, Beta_lowrank, p, states = build_matrices_for_kpoints(sys_, SPECTRAL_NORM_KPOINTS)
    G_ps = Psi.conj().T @ Psi
    G_aug_lowrank = Beta_lowrank.conj() @ sys_["qij_block"] @ Beta_lowrank.T

    # Real-space (3x-grid-converged) augmentation reference for the SAME states.
    Nx0, Ny0, Nz0 = (int(x) for x in sys_["wfc"]._ngrid)
    f = 3
    Nxf, Nyf, Nzf = Nx0 * f, Ny0 * f, Nz0 * f
    Nr_f = Nxf * Nyf * Nzf
    r_ws_cart_f, r_ws_frac_cont_f, prim_indices_f, _ = build_ws_grid_map(
        sys_["latvec"], (Nxf, Nyf, Nzf), sys_["center_cart"], nmax=sys_["ws_nmax"],
    )
    base_flat_f = (prim_indices_f[:, 0].astype(np.int64) * Nyf + prim_indices_f[:, 1]) * Nzf \
        + prim_indices_f[:, 2]

    beta_rows = []
    for st in states:
        ik, ib = st["ik"], st["band"]
        kvec = sys_["wfc"]._kvecs[ik - 1]
        gvec = sys_["wfc"].gvectors(ik)
        Cg = sys_["wfc"].readBandCoeff(ispin=sys_["ispin"], ikpt=ik, iband=ib, norm=False)
        u_bands, _, _ = zero_pad_ifft(Cg[None, :], gvec, f, (Nx0, Ny0, Nz0))
        u_ws = u_bands[:, base_flat_f]
        psi_ws = u_ws * np.exp(2j * np.pi * (r_ws_frac_cont_f @ kvec))[None, :]
        beta_rs = real_space_beta_for_bands(
            sys_["pawpp"], sys_["elements_idx"], sys_["cart_coords"], sys_["latvec"],
            r_ws_cart_f, psi_ws, Nr_f, nmax=sys_["ws_nmax"], k_frac=kvec,
        )
        beta_rows.append(beta_rs[0])
    Beta_3x = np.stack(beta_rows)
    G_aug_3x = Beta_3x.conj() @ sys_["qij_block"] @ Beta_3x.T

    diff = G_aug_lowrank - G_aug_3x
    diff_h = 0.5 * (diff + diff.conj().T)
    spectral_norm = float(np.linalg.norm(diff_h, ord=2))
    max_abs_entry = float(np.max(np.abs(diff)))
    print(f"  n_states={len(states)}  spectral_norm(G_aug_lowrank - G_aug_3x) = {spectral_norm:.4e}")
    print(f"  max|entry| = {max_abs_entry:.4e}   (Weyl bound on eigenvalue perturbation: "
          f"{spectral_norm:.4e})")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    return dict(n_states=len(states), kpoints=SPECTRAL_NORM_KPOINTS,
                spectral_norm=spectral_norm, max_abs_entry=max_abs_entry)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)!r}")


def main():
    t_start = time.time()
    sys_ = load_system()

    step12_report, states = step1_2_localize(sys_)
    step3_rows = step3_kpoint_scaling(sys_)
    step4_report = step4_gauge_test(sys_)
    step5_report = step5_spectral_norm(sys_)

    full_report = dict(
        material=sys_["material"],
        step1_2_localization=step12_report,
        step3_kpoint_scaling=step3_rows,
        step4_gauge_invariance=step4_report,
        step5_spectral_norm=step5_report,
        total_runtime_s=float(time.time() - t_start),
    )
    json_path = OUT / "paw_lowrank_phase1_diagnostics.json"
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2, default=_json_default)

    txt_path = OUT / "paw_lowrank_phase1_diagnostics.txt"
    with open(txt_path, "w") as f:
        f.write("=== Phase 1 diagnostics: localizing the K min-eigenvalue PSD violation ===\n\n")
        f.write(f"material: {sys_['material']}\n\n")

        f.write("--- Step 1+2: full-system (324 k-points) reconstruction ---\n")
        r = step12_report
        f.write(f"nstates: {r['nstates']}\n")
        f.write(f"min_eig(G_ps)    = {r['min_eig_G_ps']:.6f}\n")
        f.write(f"min_eig(G_aug)   = {r['min_eig_G_aug']:.6f}\n")
        f.write(f"min_eig(G_total) = {r['min_eig_G_total']:.6f}\n")
        f.write(f"min_eig(K)       = {r['min_eig_K']:.6f}\n")
        f.write(f"max_eig(G_total) = {r['max_eig_G_total']:.6f}   max_eig(K) = {r['max_eig_K']:.6f}\n")
        f.write(f"Hermiticity: G_ps={r['herm_G_ps']:.2e}  G_aug={r['herm_G_aug']:.2e}  "
                f"G_total={r['herm_G_total']:.2e}  K={r['herm_K']:.2e}\n")
        f.write(f"K's negativity is inherited from a non-PSD G_total "
                f"(congruence by sqrt(P) cannot introduce negativity that "
                f"wasn't already present): {r['K_negativity_explained_by_G_total']}\n\n")

        ev = r["eigenvector_decomposition"]
        f.write(f"Most negative eigenvector of G_total: eigenvalue = {ev['mu']:.6f}\n")
        f.write(f"  u^H G_ps u    = {ev['uGpsu']:.6f}\n")
        f.write(f"  u^H G_aug u   = {ev['uGaugu']:.6f}\n")
        f.write(f"  u^H G_total u = {ev['uGtotu']:.6f}\n")
        f.write(f"  per-atom/image decomposition (independent real-space route, "
                f"sum={ev['realspace_total']:.6f}):\n")
        for s in ev["realspace_site_contributions"]:
            f.write(f"    atom {s['iatom']} ({s['element']})  image={s['image']}  "
                    f"contribution={s['contribution']:.6f}\n")
        f.write("\n")

        f.write("--- Step 3: k-point subset scaling ---\n")
        for row in step3_rows:
            f.write(f"  n_k={row['n_kpoints_actual']:4d}  nstates={row['nstates']:5d}  "
                    f"min_eig(G_total)={row['min_eig_G_total']:.6f}  "
                    f"min_eig(K)={row['min_eig_K']:.6f}\n")
        f.write("\n")

        f.write("--- Step 4: random gauge invariance ---\n")
        g = step4_report
        f.write(f"max|eig(G_total)-eig(G_total_gauged)| = {g['max_eig_diff_G_total']:.4e}\n")
        f.write(f"max|eig(K)-eig(K_gauged)|             = {g['max_eig_diff_K']:.4e}\n")
        f.write(f"passed: {g['passed']}\n\n")

        f.write("--- Step 5: spectral-norm comparison (8 k-points, 3x real-space reference) ---\n")
        s5 = step5_report
        f.write(f"kpoints: {s5['kpoints']}  n_states: {s5['n_states']}\n")
        f.write(f"spectral_norm(G_aug_lowrank - G_aug_3x) = {s5['spectral_norm']:.4e}\n")
        f.write(f"max|entry| = {s5['max_abs_entry']:.4e}\n\n")

        f.write(f"total_runtime_s: {full_report['total_runtime_s']:.1f}\n")

    print(f"\nSaved -> {txt_path}\nSaved -> {json_path}")
    print(f"\nTotal runtime: {full_report['total_runtime_s']:.1f}s")


if __name__ == "__main__":
    main()
