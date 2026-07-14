"""
diagnostics/test_bvk_reduction_toy.py -- validates the REDUCTION ALGORITHM
main.py uses to build a one-cell density matrix from a Born-von Karman (BvK)
k-mesh sum, on a small synthetic system where an independent ground truth is
available by direct construction. No WAVECAR/POSCAR/POTCAR involved -- this
is pure linear algebra, decoupled from the FFT/WAVECAR-parsing checks in the
other three scripts.

Setup (1D BvK chain; the algebra is dimension-independent, so this
generalizes directly to main.py's real 3D case)
--------------------------------------------------------------------
- N unit cells under periodic boundary conditions ("cells" = k-mesh size),
  each with Nr_cell abstract real-space grid points at intra-cell fractional
  positions i/Nr_cell (i = 0..Nr_cell-1) -- the toy analog of main.py's Nr
  WS/primitive-cell grid points.
- N Bloch k-points k_p = p/N (p = 0..N-1), the BvK-mesh sampling of the
  primitive cell's BZ, with uniform weight w_p = 1/N.
- For each k_p, a random Nr_cell x Nr_cell unitary U(k_p) (via QR of a
  complex Ginibre matrix) supplies Nr_cell mutually orthonormal
  "cell-periodic" band vectors u_{n,p}[i] = U(k_p)[i, n].
- Bloch wavefunction restricted to the origin cell:
      psi_{n,p}[i] = exp(2*pi*i * (p/N) * (i/Nr_cell)) * u_{n,p}[i]
  -- exactly main.py's convention psi_nk(r) = exp(2*pi*i k.r_frac) u_nk(r)
  (main.py:220-226) with r_frac = i/Nr_cell (a primitive/WS-cell point) and
  k_frac = p/N.

Two independent constructions of the reduced one-cell density matrix
----------------------------------------------------------------------
(A) "weighted one-cell formula used by main.py" (rho_main):
      rho_main[i,i'] = sum_p w_p sum_n f_np * psi_np[i] * conj(psi_np[i'])
    -- literally main.py's rho accumulation loop (main.py:271-273),
    reimplemented at toy scale.

(B) "explicit P_A P P_A" (P_AA_explicit): build the FULL N*Nr_cell x
    N*Nr_cell crystal density matrix P_full = sum_{n,p} f_np |Psi_np><Psi_np|
    from the EXPLICIT full-crystal Bloch vectors
      Psi_np(R,i) = (1/sqrt(N)) * exp(2*pi*i*(p/N)*R) * psi_np[i],
    then extract the (R=0, R'=0) block -- i.e. P_A P_full P_A restricted to
    the origin cell's Nr_cell coordinates, P_A being the diagonal 0/1
    real-space restriction operator onto that cell. This is a genuinely
    independent computational path (full (N*Nr_cell)^2 outer-product sum and
    submatrix extraction, no k-space reduction identity assumed).

Mathematically (A) == (B) always, by Bloch's theorem for a finite periodic
lattice -- the point of the test is to catch bugs in main.py's specific
matrix-construction ALGORITHM (indexing, phase convention, normalization),
not to discover new physics.

Checks
------
- ||P_AA_explicit - rho_main||_max < bvk_identity tolerance.
- trace(rho_main) * N ~= trace(P_full) = sum_np f_np (both a self-consistency
  identity and a check that the two per-cell/whole-crystal electron counts
  agree).
- rho_main is Hermitian and PSD (min eigenvalue >= -tol).
- rho_main's eigenvalues lie in [0, 1] (+tol slack) -- the exact physical
  bound whose violation motivated this entire diagnostics package (see
  paw_augmentation/TASK_BRIEF.md).

Run for both a binary-occupation regime (f in {0,1}, main.py's common case)
and a fractional-occupation regime (f ~ Uniform[0,1], the smeared-metal
case, and the case where the SOC `occ`-omission bug -- flagged, not fixed,
below -- would actually change the answer).

No production data is read or written; writes
diagnostics/output/test_bvk_reduction_toy.json.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def _random_unitary(n, rng):
    """Haar-ish random unitary via QR of a complex Ginibre matrix, with the
    usual sign-fix so Q is genuinely uniform (not just "some" unitary)."""
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)
    phases = np.diag(R) / np.abs(np.diag(R))
    return Q * phases[np.newaxis, :]


def build_bvk_system(N, Nr_cell, occ_mode, rng):
    """Returns dict with u (N,Nr_cell,Nr_cell) [u[p,:,n]=u_{n,p}], f (N,Nr_cell)
    occupations, and the derived psi_np / Psi_np arrays."""
    u = np.stack([_random_unitary(Nr_cell, rng) for _ in range(N)])  # (N, i, n)

    if occ_mode == "binary":
        # Assign a random per-(n,p) "energy" and occupy the lowest half of
        # all N*Nr_cell states -- a generic binary-occupation configuration,
        # not tied to any particular band ordering.
        energies = rng.normal(size=(N, Nr_cell))
        flat = energies.ravel()
        n_occ = (N * Nr_cell) // 2
        thresh = np.sort(flat)[n_occ - 1] if n_occ > 0 else -np.inf
        f = (energies <= thresh).astype(float)
    elif occ_mode == "fractional":
        f = rng.uniform(0.0, 1.0, size=(N, Nr_cell))
    else:
        raise ValueError(occ_mode)

    i_idx = np.arange(Nr_cell) / Nr_cell            # (Nr_cell,)
    p_idx = np.arange(N)                             # (N,)
    k_frac = p_idx / N                                # (N,)

    # psi_np[p, i, n] = exp(2*pi*i * k_p * i/Nr_cell) * u[p, i, n]
    phase_intracell = np.exp(2j * np.pi * np.outer(k_frac, i_idx))    # (N, Nr_cell)
    psi = u * phase_intracell[:, :, None]             # (N, Nr_cell, Nr_cell) = (p, i, n)

    return dict(u=u, f=f, psi=psi, k_frac=k_frac, i_idx=i_idx)


def rho_main_formula(sysdict, N, Nr_cell):
    """(A) main.py's own weighted one-cell formula:
    rho[i,i'] = sum_p w_p sum_n f_np psi_np[i] conj(psi_np[i'])."""
    psi, f = sysdict["psi"], sysdict["f"]              # (N,Nr_cell,Nr_cell), (N,Nr_cell)
    w_p = 1.0 / N
    rho = np.zeros((Nr_cell, Nr_cell), dtype=np.complex128)
    for p in range(N):
        Psi_p = psi[p]                                  # (i, n)
        fw = f[p]                                        # (n,)
        rho += w_p * (Psi_p * fw[None, :]) @ Psi_p.conj().T
    return rho


def explicit_full_crystal_block(sysdict, N, Nr_cell):
    """(B) explicit P_A P_full P_A: build the full N*Nr_cell x N*Nr_cell
    crystal density matrix from explicit Bloch vectors and extract the
    R=0 block."""
    psi, f, k_frac = sysdict["psi"], sysdict["f"], sysdict["k_frac"]
    Ntot = N * Nr_cell
    P_full = np.zeros((Ntot, Ntot), dtype=np.complex128)
    R = np.arange(N)

    for p in range(N):
        cell_phase = np.exp(2j * np.pi * k_frac[p] * R) / np.sqrt(N)   # (N,) over R
        Psi_p_cell = psi[p]                                             # (i, n)
        for n in range(Nr_cell):
            fpn = f[p, n]
            if fpn == 0.0:
                continue
            # Big vector Psi_{n,p}(R,i) = cell_phase[R] * Psi_p_cell[i, n]
            big_vec = (cell_phase[:, None] * Psi_p_cell[None, :, n]).reshape(Ntot)
            P_full += fpn * np.outer(big_vec, big_vec.conj())

    trace_full = float(np.trace(P_full).real)
    P_AA = P_full[:Nr_cell, :Nr_cell]
    return P_AA, trace_full


def run_one_case(N, Nr_cell, occ_mode, seed, tol):
    rng = np.random.default_rng(seed)
    sysdict = build_bvk_system(N, Nr_cell, occ_mode, rng)

    rho_main = rho_main_formula(sysdict, N, Nr_cell)
    P_AA_explicit, trace_full = explicit_full_crystal_block(sysdict, N, Nr_cell)

    identity_err = float(np.max(np.abs(P_AA_explicit - rho_main)))

    herm_err = float(np.max(np.abs(rho_main - rho_main.conj().T)))
    eigvals = np.linalg.eigvalsh(0.5 * (rho_main + rho_main.conj().T))
    eig_min, eig_max = float(eigvals.min()), float(eigvals.max())

    tr_rho_main = float(np.trace(rho_main).real)
    trace_consistency_err = float(abs(N * tr_rho_main - trace_full))

    return dict(
        N=N, Nr_cell=Nr_cell, occ_mode=occ_mode, seed=seed,
        explicit_vs_formula_max_err=identity_err,
        explicit_vs_formula_tol=tol["bvk_identity"],
        explicit_vs_formula_passed=bool(identity_err < tol["bvk_identity"]),
        herm_err=herm_err,
        trace_rho_main=tr_rho_main,
        trace_full_crystal=trace_full,
        trace_consistency_err=trace_consistency_err,
        trace_consistency_passed=bool(trace_consistency_err < tol["bvk_identity"] * N),
        eig_min=eig_min, eig_max=eig_max,
        psd_tol=tol["bvk_bound"],
        psd_passed=bool(eig_min > -tol["bvk_bound"]),
        bound_01_tol=tol["bvk_bound"],
        bound_01_passed=bool(eig_max < 1.0 + tol["bvk_bound"]),
    )


def main():
    tol = C.TOL
    cases = [
        dict(N=6, Nr_cell=4, occ_mode="binary", seed=0),
        dict(N=6, Nr_cell=4, occ_mode="fractional", seed=1),
        dict(N=9, Nr_cell=3, occ_mode="binary", seed=2),
        dict(N=9, Nr_cell=3, occ_mode="fractional", seed=3),
    ]

    print("=== test_bvk_reduction_toy ===\n")
    results = []
    all_ok = True
    for case in cases:
        r = run_one_case(tol=tol, **case)
        results.append(r)
        ok = (r["explicit_vs_formula_passed"] and r["trace_consistency_passed"]
              and r["psd_passed"] and r["bound_01_passed"])
        all_ok = all_ok and ok
        print(f"N={r['N']:2d} Nr_cell={r['Nr_cell']:2d} occ={r['occ_mode']:11s} "
              f"seed={r['seed']}: "
              f"explicit-vs-formula={r['explicit_vs_formula_max_err']:.2e}  "
              f"trace_consistency={r['trace_consistency_err']:.2e}  "
              f"eig=[{r['eig_min']:.2e}, {r['eig_max']:.6f}]  "
              f"{'OK' if ok else 'FAIL'}")

    report = dict(tol=tol, cases=results,
                  status="PASS" if all_ok else "FAIL")
    path = C.write_report("test_bvk_reduction_toy", report)
    print(f"\nstatus={report['status']}  -> {path}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
