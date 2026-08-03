"""
direct_fourier.py — Vectorized direct plane-wave Fourier evaluator.

Convention (identical to main.py):
  Lattice:        r = s1*a1 + s2*a2 + s3*a3   (s = fractional coord)
  k-point:        k = kappa1*b1 + kappa2*b2 + kappa3*b3
  G-vector:       G = m1*b1 + m2*b2 + m3*b3   (m = integer indices from gvectors())
  Dot product:    G.r = m1*s1 + m2*s2 + m3*s3  (since b_i.a_j = delta_ij)

Cell-periodic part:
  u_nk(s)   = (norm_factor) * sum_G C_G * exp(2pi i m . s)

Full Bloch wavefunction (mode='psi'):
  psi_nk(s) = (norm_factor) * sum_G C_G * exp(2pi i (m + kappa) . s)

Normalization used by main.py:
  norm_factor = 1 / sqrt(Nr)    where Nr = Nx * Ny * Nz
  -> sum_j |psi_nk(s_j)|^2 = 1  (discrete, same as ifftn * sqrt(Nr))

Physical normalization (used by the archived ``main_v3.py`` diagnostic):
  norm_factor = 1 / sqrt(Omega)  where Omega = unit-cell volume in Ang^3
  -> sum_j |psi_nk(r_j)|^2 * dV ≈ 1  (continuous-integral approximation)
"""

import numpy as np
import time


# ─────────────────────────────────────────────────────────── evaluators ───────

def fourier_eval_bands(
    Ck_batch,           # (n_bands, n_G) complex128  [VASP-normalized: sum|C|^2 = 1]
    gvec_int,           # (n_G, 3) int  — actual signed G indices (m1, m2, m3)
    k_frac,             # (3,)  float  — k-point fractional reciprocal coordinates
    s_points,           # (n_pts, 3) float — evaluation points, fractional real-space
    mode='psi',         # 'psi' = full Bloch (k+G phase); 'u' = cell-periodic (G only)
    norm_factor=None,   # float — applied uniformly; default = 1/sqrt(n_pts)
    chunk_size=4096,    # int — real-space points processed per iteration
    verbose=True,
):
    """
    Evaluate plane-wave expansion at arbitrary real-space fractional coordinates.

    For fixed k, the phase matrix P[j, G] = exp(2πi (m_G + κ) . s_j)
    is shared by all bands.  All bands are evaluated in one batch per chunk:

        Psi[j, b] = sum_G P[j, G] * C[b, G]   (matrix multiply, shape chunk × nb)

    Returns
    -------
    psi : (n_bands, n_pts) complex128
    """
    Ck  = np.asarray(Ck_batch, dtype=np.complex128)   # (nb, nG)
    G   = np.asarray(gvec_int,  dtype=np.float64)      # (nG, 3)
    k   = np.asarray(k_frac,    dtype=np.float64)      # (3,)
    s   = np.asarray(s_points,  dtype=np.float64)      # (npts, 3)

    nb, nG = Ck.shape
    npts   = s.shape[0]

    if norm_factor is None:
        norm_factor = 1.0 / np.sqrt(npts)

    # Phase vector argument: (k+G) or just G
    kpG = G + k[np.newaxis, :] if mode == 'psi' else G   # (nG, 3)

    # Memory estimate for full matrix (informational only)
    mem_gb = npts * nG * 16 / 1e9
    if verbose:
        print(f"    direct_fourier: npts={npts:,}  nG={nG}  nb={nb}  mode={mode!r}"
              f"  chunk={chunk_size}  full_P={mem_gb:.2f} GB")

    psi = np.zeros((nb, npts), dtype=np.complex128)

    for start in range(0, npts, chunk_size):
        end = min(start + chunk_size, npts)
        s_c = s[start:end]                               # (chunk, 3)
        dot = s_c @ kpG.T                                # (chunk, nG)
        P   = np.exp(2j * np.pi * dot)                   # (chunk, nG)
        # (chunk, nG) @ (nG, nb) = (chunk, nb)  -> transpose -> (nb, chunk)
        psi[:, start:end] = (P @ Ck.T).T

    psi *= norm_factor
    return psi


# ─────────────────────────────────────────────────── Wigner-Seitz membership ──

def ws_membership(
    r_cart,         # (n_pts, 3) Cartesian positions to test
    center_cart,    # (3,) Cartesian WS center
    latvec,         # (3, 3) lattice vectors (rows = a1, a2, a3)
    nmax=2,         # search range for lattice translations
    tol=1e-10,      # geometric tolerance (Å^2 in the bisector condition)
):
    """
    Return boolean mask: True if r_cart[i] is inside the WS cell at center_cart.

    Test: point d = r - center is inside the Voronoi cell iff
          2 * d . R <= |R|^2   for all non-zero lattice vectors R = n @ latvec.
    """
    ns  = np.arange(-nmax, nmax + 1)
    n1, n2, n3 = np.meshgrid(ns, ns, ns, indexing='ij')
    all_n = np.column_stack([n1.ravel(), n2.ravel(), n3.ravel()])
    all_n = all_n[np.any(all_n != 0, axis=1)]        # remove (0,0,0)
    R_cart = all_n @ latvec                            # (ntrans, 3)
    R2     = np.einsum('ti,ti->t', R_cart, R_cart)    # (ntrans,)

    d   = np.asarray(r_cart, dtype=float) - center_cart   # (n_pts, 3)
    dot = d @ R_cart.T                                      # (n_pts, ntrans)

    inside = np.all(2.0 * dot <= R2[None, :] + tol, axis=1)
    return inside


# ──────────────────────────────────────────────────────────── benchmark ────────

def benchmark_one_kpoint(Ck_batch, gvec_int, k_frac, n_pts_list, chunk_size=4096):
    """
    Time the direct Fourier evaluation for several grid sizes.

    Returns dict mapping n_pts -> time_sec.
    """
    G  = np.asarray(gvec_int, dtype=np.float64)
    k  = np.asarray(k_frac,   dtype=np.float64)
    nb = Ck_batch.shape[0]

    print(f"  Benchmark: nG={G.shape[0]}  nb={nb}  chunk={chunk_size}")
    results = {}
    for n in n_pts_list:
        rng = np.random.default_rng(42)
        s   = rng.uniform(0, 1, size=(n, 3))
        t0  = time.perf_counter()
        fourier_eval_bands(Ck_batch, gvec_int, k_frac, s,
                           norm_factor=1.0, chunk_size=chunk_size, verbose=False)
        dt = time.perf_counter() - t0
        print(f"    n_pts={n:8,}  time={dt:.3f} s  ({n/dt/1e6:.2f} M pts/s)")
        results[n] = dt
    return results
