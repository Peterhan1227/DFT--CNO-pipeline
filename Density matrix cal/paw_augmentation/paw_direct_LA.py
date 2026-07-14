"""
paw_direct_LA.py -- exact alternative to paw_regional_cno.py's state-space
K_A route (build_G_A_K_A + eigh(K_A), cost ~ O(Nr*nstates^2 + nstates^3))
for getting the regional CNO spectrum, cheaper when the WS-cell grid Nr is
SMALLER than nstates (many k-points/bands, modest real-space grid) -- the
opposite regime from WSe2_mono, where Nr >> nstates and the K_A route wins.
Selected via the USE_DIRECT_L_A toggle at the top of paw_regional_cno.py;
both routes read the same Psi/beta_by_site/sites/p and are meant to agree
to numerical precision (see _self_test() below).

Derivation
----------
G_A[a,b] = <psi~_a|P_A|psi~_b> + sum_site beta_a(site)^H Q_A(site) beta_b(site)

Q_A(site) is a difference of AE and PS partial-wave overlaps restricted to
region A -- unlike the full atomic Qij it is NOT guaranteed positive
semidefinite, so it cannot be folded into a real "append sqrt(Q_A)*beta as
extra grid points" embedding without tracking sign. Eigendecompose each
site's Q_A = V diag(d) V^H (d real, any sign) and define the per-site
channel amplitude c = V^H @ beta_site^T and weight w = sqrt(|d|); then

    beta_a^H Q_A beta_b = sum_ch sign(d_ch) * conj(w_ch c_a,ch) * (w_ch c_b,ch)

so stacking B = w[:,None]*c (lmmax, nstates) over every site into one
(n_channels, nstates) block and Phi = vstack[Psi; B] (M, nstates) with
M = Nr + n_channels, and j = [+1]*Nr ++ [sign(d) per channel] (M,):

    G_A = Phi^H diag(j) Phi         K_A = Phi_p^H diag(j) Phi_p,  Phi_p = Phi * sqrt(p)

K_A (nstates x nstates) has rank <= M. A thin QR of Phi_p^H (nstates x M,
tall when M < nstates) gives Phi_p^H = Q2 R2 with Q2 (nstates, M)
orthonormal columns (Q2^H Q2 = I_M exactly) and R2 (M, M) upper
triangular. Then

    K_A = Q2 (R2 diag(j) R2^H) Q2^H = Q2 K_small Q2^H

so K_A's nonzero eigenvalues/eigenvectors are EXACTLY K_small's eigenvalues
and Q2 @ (K_small eigenvectors) -- K_small is only M x M (Hermitian, plain
eigh, no signature issues left to handle numerically) and the most
expensive step (the QR) costs O(nstates * M^2), the same scaling as
forming K_A directly costs O(Nr*nstates^2) with the roles of Nr and
nstates swapped -- i.e. cheaper exactly when M < nstates. No new physics
or approximation is introduced; this is the standard tall-QR Gram-matrix
trick applied to an indefinite (signed) Gram form instead of a plain PSD
one.
"""
import numpy as np


def solve_natural_orbitals_direct(Psi, beta_by_site, sites, p, occ_tol=1e-6):
    """Exact alternative to build_G_A_K_A(...) + np.linalg.eigh(K_A).

    Parameters mirror paw_regional_cno.py's build_state_list_and_beta output:
    Psi (Nr, nstates) complex, beta_by_site (list of (nstates, lmmax_site)
    complex, parallel to sites), sites (list of dicts with a "Q_A" key,
    (lmmax_site, lmmax_site) real symmetric), p (nstates,) real weights.

    Returns
    -------
    lam_sel : (n_sel,) eigenvalues > occ_tol, descending -- same convention
              and same numerical values as filtering np.linalg.eigh(K_A).
    X       : (Nr, n_sel) real-space pseudo orbitals, X = Psi @ Y with
              Y = sqrt(p)[:,None]*U_sel/sqrt(lam_sel)[None,:] -- identical
              formula/contract to the K_A route.
    eigvals : (M,) the full small-eigenproblem spectrum (M = Nr +
              n_channels), i.e. every nonzero eigenvalue of K_A (the
              remaining nstates-M eigenvalues of K_A are exactly zero).
    diagnostics : dict with M, n_channels, herm_K_small, min_eig,
              trace_K_small -- cheap (M x M) equivalents of the K_A route's
              _herm(K_A)/_mineig(K_A)/trace(K_A) checks (exactly equal to
              those quantities, not approximations -- see module docstring).
    """
    Nr, nstates = Psi.shape
    sqrtp = np.sqrt(p)

    B_blocks = []
    signs = []
    for beta_site, s in zip(beta_by_site, sites):
        Q_A = s["Q_A"]
        d, V = np.linalg.eigh(0.5 * (Q_A + Q_A.T))
        c = V.T @ beta_site.T                     # (lmmax_site, nstates)
        w = np.sqrt(np.abs(d))
        B_blocks.append(w[:, None] * c)
        signs.append(np.sign(d))

    if B_blocks:
        B = np.concatenate(B_blocks, axis=0)
        j_aug = np.concatenate(signs, axis=0)
    else:
        B = np.zeros((0, nstates), dtype=Psi.dtype)
        j_aug = np.zeros(0)

    Phi = np.concatenate([Psi, B], axis=0)         # (M, nstates)
    j = np.concatenate([np.ones(Nr), j_aug])       # (M,)
    Phi_p = Phi * sqrtp[None, :]

    Q2, R2 = np.linalg.qr(Phi_p.conj().T, mode="reduced")   # Q2:(nstates,M) R2:(M,M)
    K_small = R2 @ (j[:, None] * R2.conj().T)
    K_small = 0.5 * (K_small + K_small.conj().T)

    eigvals, V_small = np.linalg.eigh(K_small)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    V_small = V_small[:, order]
    U = Q2 @ V_small                                # (nstates, M)

    sel = eigvals > occ_tol
    n_sel = int(sel.sum())
    lam_sel = eigvals[sel]
    U_sel = U[:, sel]
    Y = (sqrtp[:, None] * U_sel) / np.sqrt(lam_sel)[None, :]
    X = Psi @ Y

    # Y^H G_A Y should equal I exactly (same identity the K_A route checks
    # via Y.conj().T @ G_A @ Y) -- computed here in the cheap M-dimensional
    # space instead of ever forming G_A: Phi_p @ U_sel = R2^H @ V_sel (from
    # Phi_p^H = Q2 R2, Q2^H Q2 = I, U_sel = Q2 V_sel), so
    # Y^H G_A Y = (R2^H V_sel / sqrt(lam))^H diag(j) (R2^H V_sel / sqrt(lam)).
    V_sel = V_small[:, sel]
    Z = (R2.conj().T @ V_sel) / np.sqrt(lam_sel)[None, :]
    ortho_err = float(np.max(np.abs((Z.conj().T * j[None, :]) @ Z - np.eye(n_sel)))) if n_sel else 0.0

    diagnostics = dict(
        M=int(Phi.shape[0]), n_channels=int(B.shape[0]),
        herm_K_small=float(np.max(np.abs(K_small - K_small.conj().T))),
        min_eig=float(eigvals.min()) if len(eigvals) else 0.0,
        trace_K_small=float(np.trace(K_small).real),
        n_selected=n_sel, ortho_err=ortho_err,
    )
    return lam_sel, X, eigvals, diagnostics


def _self_test(seed=0, Nr=25, nstates=9, nsites=3, lmmax=4, verbose=True):
    """Pure-numpy correctness check against the plain state-space K_A route
    on random synthetic data (no physics, no file I/O -- safe to run any
    time). Confirms the QR/signed-channel construction above reproduces the
    same eigenvalues and the same gauge-invariant reconstructed operator
    Psi @ diag(lam) @ Psi^H as directly building and diagonalizing K_A."""
    rng = np.random.default_rng(seed)

    Psi = rng.normal(size=(Nr, nstates)) + 1j * rng.normal(size=(Nr, nstates))
    p = rng.uniform(0.1, 1.0, size=nstates)

    sites = []
    beta_by_site = []
    for _ in range(nsites):
        A = rng.normal(size=(lmmax, lmmax)) + 1j * rng.normal(size=(lmmax, lmmax))
        Q_A = np.real(A + A.conj().T)          # Hermitian, real, indefinite by construction
        sites.append(dict(Q_A=Q_A))
        beta_by_site.append(rng.normal(size=(nstates, lmmax)) + 1j * rng.normal(size=(nstates, lmmax)))

    # reference: plain state-space route (mirrors build_G_A_K_A)
    G_ps_A = Psi.conj().T @ Psi
    G_aug_A = np.zeros_like(G_ps_A)
    for beta_site, s in zip(beta_by_site, sites):
        G_aug_A += beta_site.conj() @ s["Q_A"] @ beta_site.T
    G_A = G_ps_A + G_aug_A
    sqrtp = np.sqrt(p)
    K_A = (sqrtp[:, None] * G_A) * sqrtp[None, :]
    K_A = 0.5 * (K_A + K_A.conj().T)
    eigvals_ref, U_ref = np.linalg.eigh(K_A)
    order = np.argsort(eigvals_ref)[::-1]
    eigvals_ref = eigvals_ref[order]
    U_ref = U_ref[:, order]
    sel_ref = eigvals_ref > 1e-10
    lam_ref = eigvals_ref[sel_ref]
    Y_ref = (sqrtp[:, None] * U_ref[:, sel_ref]) / np.sqrt(lam_ref)[None, :]
    X_ref = Psi @ Y_ref
    D_ref = X_ref @ np.diag(lam_ref) @ X_ref.conj().T

    # direct route
    lam_sel, X, eigvals, diag = solve_natural_orbitals_direct(Psi, beta_by_site, sites, p, occ_tol=1e-10)
    D_direct = X @ np.diag(lam_sel) @ X.conj().T

    eig_err = float(np.max(np.abs(np.sort(lam_sel)[::-1] - np.sort(lam_ref)[::-1])))
    op_err = float(np.max(np.abs(D_direct - D_ref)))
    trace_err = float(abs(np.trace(K_A).real - diag["trace_K_small"]))

    if verbose:
        print(f"eigenvalue max|diff| (direct vs K_A route): {eig_err:.3e}")
        print(f"reconstructed-operator max|diff| (gauge-invariant): {op_err:.3e}")
        print(f"trace(K_A) vs trace(K_small): diff={trace_err:.3e}")
        print(f"direct route ortho_err (Y^H G_A Y - I): {diag['ortho_err']:.3e}")
        print(f"M={diag['M']}  n_channels={diag['n_channels']}  n_selected={diag['n_selected']}"
              f"  (nstates={nstates}, Nr={Nr})")

    ok = eig_err < 1e-8 and op_err < 1e-8 and trace_err < 1e-8
    print("SELF-TEST " + ("PASSED" if ok else "FAILED"))
    return ok


if __name__ == "__main__":
    _self_test()
