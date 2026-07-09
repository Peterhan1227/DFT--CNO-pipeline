"""
induce_band_rep.py — Stage 2 of the CNO irrep pipeline: induce the real-space
site-symmetry representation rho(h) (Stage 1, site_symmetry.py) into
momentum-space band-representation matrices D_k(g) and characters chi_k(g).

Run with the 'irrep' conda environment (has spglib + numpy + scipy).

Physics summary
----------------
Stage 1 found the site-symmetry group G_q (order Ng) that fixes a WS center q
and computed rho(h)_ba = <phi_b|U_h|phi_a> for every CNO h in G_q, on the full
included-CNO space. This script never recomputes rho(h): it is read verbatim
from the Stage-1 .npz.

Given a user-chosen local CNO subspace {a} (a subset of the Stage-1 included
CNOs, possibly a union of several candidate blocks), we build the induced band
representation of the full space group on the orbit of q:

  1. The full space group {R_g|t_g} (mod lattice translations) is regenerated
     from spglib (this is abstract group algebra, not a CNO symmetry action).
  2. The orbit q_alpha = g_alpha q (mod lattice) is enumerated; one coset
     representative g_alpha = {R_alpha|t_alpha} is fixed per orbit point, with
     t_alpha adjusted so g_alpha maps q to q_alpha EXACTLY (not just mod
     lattice).
  3. For every (g, alpha), the coset decomposition
         g g_alpha = {E|L_{beta alpha}(g)} g_beta h_{beta alpha}(g)
     is solved by pure affine algebra: the rotation part fixes R_h uniquely,
     the requirement that h stabilizes q exactly (using the Stage-1 t_eff
     convention) fixes h's translation, and matching the resulting point to
     the stored orbit points fixes beta and L. h is matched to a Stage-1 site
     operation by (R, t_eff), never by list position.
  4. The sewing matrix B_g(k) is assembled from L_{beta alpha}(g) (a pure
     translation phase) and rho(h_{beta alpha}(g)) (read from Stage 1). No
     other phase is added.
  5. At each configured high-symmetry k, the little group G_k is found and
     D_k(g) = B_g(k) is evaluated; characters and group-law/unitarity errors
     are reported.

Because every rho(h) is the raw (possibly imperfect) Stage-1 matrix, all
induced quantities inherit that same error -- this script performs no
renormalization or unitarization anywhere. The additional error introduced by
induction itself (coset-decomposition reconstruction, sewing/little-group
composition laws) should be at machine precision; if it is not, the coset
algebra or phase convention has a bug.
"""

import io as _io
import json
import sys
from pathlib import Path

import numpy as np
import spglib

# ── paths ─────────────────────────────────────────────────────────────────────
HERE          = Path(__file__).resolve().parent
REPO_ROOT     = HERE.parent
STAGE1_DIR    = REPO_ROOT / "CNO Symmetry Analysis"
PIPELINE_DIR  = REPO_ROOT / "Density matrix cal"
IRREP_DIR     = REPO_ROOT / "Irrep"

sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(IRREP_DIR / "scripts"))

from ws_cell import read_poscar_structure               # noqa: E402
from build_irrep_kpoints import parse_kpoints            # noqa: E402

# ── configuration ─────────────────────────────────────────────────────────────
MATERIAL      = "Si"
STAGE1_NPZ    = STAGE1_DIR / "output" / MATERIAL / f"site_symmetry_{MATERIAL}_verify.npz"
POSCAR        = PIPELINE_DIR / "Data" / MATERIAL / "POSCAR"

# One or more local CNO subspaces, given as ORIGINAL GLOBAL CNO indices (the
# same indexing as Stage 1's cno_occupations.npy). Concatenated (order
# preserved) into a single local basis of dimension n_local; the induced band
# representation has dimension N_orbit * n_local.
SELECTED_LOCAL_CNOS = [[0], [2, 3], [5, 6]]

# High-symmetry k-points: reuse the IrRep workflow's own list so results are
# directly comparable (same file format as build_irrep_kpoints.py expects).
KPOINTS_SOURCE = IRREP_DIR / "output" / "high_symmetry_points.out"

# Optional comparison to the existing DFT IrRep result (Section 9).
ENABLE_DFT_COMPARISON = True
DFT_SYMMETRY_JSON     = IRREP_DIR / "output" / "symmetry_matrices.json"

SYMPREC        = 1e-5     # must match Stage 1's spglib tolerance
FRAC_TOL       = 1e-6     # tolerance for "differs by an integer lattice vector"
WARN_LEAKAGE   = 5e-2
WARN_UNITARY   = 5e-2
WARN_GROUPLAW  = 5e-2

OUTPUT_DIR = HERE / "output" / MATERIAL

sep = "-" * 70


# ═══════════════════════════════════════════════════════════════════════════
# 0. Load Stage-1 output (never recompute CNO symmetry actions)
# ═══════════════════════════════════════════════════════════════════════════

d1 = np.load(STAGE1_NPZ, allow_pickle=True)

q               = np.asarray(d1['q'], dtype=float)          # (3,) fractional, exact stored center
latvec          = np.asarray(d1['latvec'], dtype=float)     # (3,3)
sg_label        = str(d1['spacegroup_label'])

R_site          = d1['R_ops']                 # (Ng,3,3) int   -- site-symmetry rotations
t_eff_site      = d1['t_eff_ops']             # (Ng,3) float   -- exact q-fixing translations
op_product_site = d1['operation_product_table']   # (Ng,Ng) int
Ng              = len(R_site)

D_all                 = d1['D_all']                  # (Ng, n_inc, n_inc) complex, RAW rho(h)
transformed_norm2_all = d1['transformed_norm2_all']  # (Ng, n_inc) float
cno_indices_all       = d1['cno_indices']            # (n_inc,) global CNO indices included in Stage 1
cno_occ_all           = d1['cno_occupations']        # (n_inc,) occupations, same order as cno_indices_all

n_inc  = len(cno_indices_all)
pos_of = {int(idx): k for k, idx in enumerate(cno_indices_all)}

print(f"Loaded Stage-1 data: material={MATERIAL}  space group={sg_label}")
print(f"  q (fractional) = {q}")
print(f"  site-symmetry order Ng = {Ng}")
print(f"  Stage-1 included CNOs (n_inc) = {n_inc}: {cno_indices_all.tolist()}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. User-selected local CNO subspace and its Stage-1 quality
# ═══════════════════════════════════════════════════════════════════════════

local_cno_indices = [int(i) for grp in SELECTED_LOCAL_CNOS for i in grp]
n_local = len(local_cno_indices)

_missing = [i for i in local_cno_indices if i not in pos_of]
if _missing:
    raise ValueError(
        f"Selected local CNO indices {_missing} are not among Stage-1 included "
        f"CNOs {cno_indices_all.tolist()}. Choose indices from that list "
        f"(occupation cutoff was already applied in Stage 1).")
if len(set(local_cno_indices)) != len(local_cno_indices):
    raise ValueError(f"Duplicate index in SELECTED_LOCAL_CNOS: {local_cno_indices}")

local_pos        = [pos_of[i] for i in local_cno_indices]
local_occ        = cno_occ_all[local_pos]
local_site_representation = D_all[:, local_pos, :][:, :, local_pos]   # (Ng, n_local, n_local), RAW

print(f"\nLocal CNO subspace selected: groups={SELECTED_LOCAL_CNOS}")
print(f"  local_cno_indices = {local_cno_indices}  (n_local={n_local})")
print(f"  local_occupations = {local_occ}")


def _block_diagnostics(D_block, transformed_norm2_sel, op_product_table):
    """Same three checks as Stage 1's block_diagnostics, recomputed for an
    arbitrary (possibly newly combined) selection directly from the raw
    D_all / transformed_norm2_all -- no renormalization, no unitarization."""
    Ng_, dim = D_block.shape[0], D_block.shape[1]
    unitary_err = np.array([
        float(np.linalg.norm(D_block[g].conj().T @ D_block[g] - np.eye(dim)))
        for g in range(Ng_)
    ])
    gl_err = np.full((Ng_, Ng_), np.nan)
    for i in range(Ng_):
        for j in range(Ng_):
            k_ = op_product_table[i, j]
            if k_ >= 0:
                gl_err[i, j] = np.linalg.norm(D_block[i] @ D_block[j] - D_block[k_])
    block_retained = np.sum(np.abs(D_block) ** 2, axis=1)     # (Ng_,dim)
    with np.errstate(divide='ignore', invalid='ignore'):
        leak = 1.0 - block_retained / transformed_norm2_sel
    leak = np.where(transformed_norm2_sel > 1e-12, leak, np.nan)
    return unitary_err, gl_err, leak


_transformed_norm2_sel = transformed_norm2_all[:, local_pos]     # (Ng, n_local)
_unitary_err, _gl_err, _leak = _block_diagnostics(
    local_site_representation, _transformed_norm2_sel, op_product_site)

_max_unitary = float(np.max(_unitary_err))
_max_gl      = float(np.nanmax(_gl_err)) if np.isfinite(_gl_err).any() else float('nan')
_max_leak    = float(np.nanmax(_leak)) if np.isfinite(_leak).any() else float('nan')

print(f"  Stage-1 input quality for this selection (raw, unmodified):")
print(f"    max unitarity error  = {_max_unitary:.3e}")
print(f"    max group-law error  = {_max_gl:.3e}")
print(f"    max leakage          = {_max_leak:.3e}")
if _max_unitary > WARN_UNITARY:
    print(f"  WARNING: large unitarity error ({_max_unitary:.3e} > {WARN_UNITARY}) "
          f"in the selected local subspace -- rho(h) is used raw regardless.")
if _max_gl == _max_gl and _max_gl > WARN_GROUPLAW:
    print(f"  WARNING: large group-law error ({_max_gl:.3e} > {WARN_GROUPLAW}) "
          f"in the selected local subspace -- rho(h) is used raw regardless.")
if _max_leak == _max_leak and _max_leak > WARN_LEAKAGE:
    print(f"  WARNING: large leakage ({_max_leak:.3e} > {WARN_LEAKAGE}) in the "
          f"selected local subspace -- rho(h) is used raw regardless.")


# ═══════════════════════════════════════════════════════════════════════════
# helpers: integer-lattice-vector arithmetic shared by every step below
# ═══════════════════════════════════════════════════════════════════════════

def wrap_frac(v, tol=FRAC_TOL):
    """Wrap fractional coordinates into [0,1), snapping points within `tol`
    of an integer to exactly 0 so that boundary points cluster consistently."""
    w = np.mod(v, 1.0)
    w = np.where(np.abs(w - 1.0) < tol, 0.0, w)
    w = np.where(np.abs(w) < tol, 0.0, w)
    return w


def is_integer_vec(v, tol=FRAC_TOL):
    r = np.round(v)
    return bool(np.max(np.abs(v - r)) < tol), r


def is_integer_mat(M, tol=1e-6):
    r = np.round(M)
    return bool(np.max(np.abs(M - r)) < tol), r


def op_key(R, t, tol=6):
    """Canonical hashable key for a (R, t) space-group operation, translation
    taken mod 1 lattice vector (matches Stage 1's _op_key convention)."""
    t_w = tuple(np.round(((np.asarray(t) + 0.5) % 1.0 - 0.5), tol).tolist())
    return (tuple(int(x) for x in np.asarray(R).ravel()), t_w)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Full space group (spglib) and the real-space orbit of q
# ═══════════════════════════════════════════════════════════════════════════

latvec2, species, counts, atom_syms, atom_nums, frac_coords, cart_coords = \
    read_poscar_structure(POSCAR)

if not np.allclose(latvec2, latvec, atol=1e-6):
    raise ValueError("POSCAR lattice does not match the lattice stored in the "
                      "Stage-1 .npz -- Stage 1 and Stage 2 must use the same crystal.")

cell = (latvec2, frac_coords, atom_nums)
sg_label_check = spglib.get_spacegroup(cell, symprec=SYMPREC)
if sg_label_check != sg_label:
    raise ValueError(f"Space group mismatch: Stage 1 saved {sg_label!r}, "
                      f"recomputation here gives {sg_label_check!r}.")

sym_full = spglib.get_symmetry(cell, symprec=SYMPREC)
R_full = sym_full['rotations']         # (Nop,3,3) int
t_full = sym_full['translations']      # (Nop,3) float, fractional
Nop_total = len(R_full)

R_full_inv = np.zeros_like(R_full)
for g in range(Nop_total):
    ok, Rinv = is_integer_mat(np.linalg.inv(R_full[g].astype(float)))
    if not ok:
        raise RuntimeError(f"Full-group op {g}: R^-1 is not integer -- R=\n{R_full[g]}")
    R_full_inv[g] = Rinv.astype(int)

# full-group product table (needed for little-group closure and the general
# sewing-matrix composition-law check in Section 7)
full_op_index = {op_key(R_full[g], t_full[g]): g for g in range(Nop_total)}
op_product_full = np.full((Nop_total, Nop_total), -1, dtype=int)
for i in range(Nop_total):
    for j in range(Nop_total):
        R_ij = R_full[i] @ R_full[j]
        t_ij = R_full[i].astype(float) @ t_full[j] + t_full[i]
        op_product_full[i, j] = full_op_index.get(op_key(R_ij, t_ij), -1)
n_missing_full_products = int(np.sum(op_product_full < 0))
if n_missing_full_products:
    raise RuntimeError(f"{n_missing_full_products} operation products not found in the "
                        f"full space group -- spglib's operation list is not closed mod lattice.")

print(f"\n{sep}\nFULL SPACE GROUP AND ORBIT\n{sep}")
print(f"  |G/T| (spglib operations mod lattice) = {Nop_total}")

# --- orbit of q: cluster all Nop_total operations by where they send q ------
q_cont_all = np.einsum('gij,j->gi', R_full.astype(float), q) + t_full   # (Nop,3)
q_wrap_all = wrap_frac(q_cont_all)

cluster_of = -np.ones(Nop_total, dtype=int)
cluster_reps = []       # list of canonical wrapped points, one per cluster
cluster_members = []    # list of lists of global op indices

for g in range(Nop_total):
    w = q_wrap_all[g]
    found = -1
    for c, rep in enumerate(cluster_reps):
        diff = w - rep
        ok, _ = is_integer_vec(diff)
        if ok:
            found = c
            break
    if found < 0:
        found = len(cluster_reps)
        cluster_reps.append(w)
        cluster_members.append([])
    cluster_of[g] = found
    cluster_members[found].append(g)

N_orbit = len(cluster_reps)

# identity is always its own operation and always fixes q exactly -> its
# cluster is, by definition, the orbit point q_0 = q itself.
identity_g = next(g for g in range(Nop_total)
                   if np.array_equal(R_full[g], np.eye(3, dtype=int))
                   and np.allclose(t_full[g], 0.0, atol=1e-8))
c0 = cluster_of[identity_g]

# reorder clusters so alpha=0 is the identity's cluster (original-site orbit
# point), preserving the relative order of the rest.
order = [c0] + [c for c in range(N_orbit) if c != c0]
cluster_members = [cluster_members[c] for c in order]
remap = {old: new for new, old in enumerate(order)}
cluster_of = np.array([remap[c] for c in cluster_of])

if Nop_total != N_orbit * Ng:
    raise RuntimeError(
        f"Orbit-stabilizer relation FAILED: N_orbit*|G_q| = {N_orbit}*{Ng} = "
        f"{N_orbit * Ng} != |G/T| = {Nop_total}.")
_cluster_sizes = [len(m) for m in cluster_members]
if any(s != Ng for s in _cluster_sizes):
    raise RuntimeError(f"Orbit clusters have unequal size {_cluster_sizes}; "
                        f"expected every coset to have |G_q|={Ng} elements.")

print(f"  Orbit size N_orbit = {N_orbit}   (orbit-stabilizer: {N_orbit}*{Ng} = {Nop_total} OK)")

# --- choose one coset representative g_alpha per orbit point ----------------
orbit_points = np.zeros((N_orbit, 3))   # stored q_alpha (reference-cell point)
coset_R      = np.zeros((N_orbit, 3, 3), dtype=int)
coset_t      = np.zeros((N_orbit, 3))
coset_rep_g  = np.zeros(N_orbit, dtype=int)     # which full-group index was used

for alpha in range(N_orbit):
    if alpha == 0:
        g_rep = identity_g
    else:
        g_rep = min(cluster_members[alpha])     # deterministic tie-break
    coset_rep_g[alpha] = g_rep
    q_alpha_ref = wrap_frac(q_cont_all[g_rep]) if alpha != 0 else q.copy()
    L0 = np.round(q_cont_all[g_rep] - q_alpha_ref)
    resid = float(np.max(np.abs(q_cont_all[g_rep] - q_alpha_ref - L0)))
    if resid > 1e-6:
        raise RuntimeError(f"Orbit point {alpha}: representative adjustment residual "
                            f"{resid:.3e} too large.")
    orbit_points[alpha] = q_alpha_ref
    coset_R[alpha] = R_full[g_rep]
    coset_t[alpha] = t_full[g_rep] - L0
    # exactness check: R_alpha q + t_alpha == q_alpha exactly
    check = coset_R[alpha].astype(float) @ q + coset_t[alpha] - orbit_points[alpha]
    if np.max(np.abs(check)) > 1e-8:
        raise RuntimeError(f"Orbit point {alpha}: g_alpha does not map q to q_alpha exactly "
                            f"(residual {np.max(np.abs(check)):.3e}).")

print(f"  Orbit points q_alpha (fractional):")
for alpha in range(N_orbit):
    print(f"    alpha={alpha}: {orbit_points[alpha].round(6)}  "
          f"(rep g={coset_rep_g[alpha]}, R=\n{coset_R[alpha]})")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Bloch basis / induced dimension (purely algebraic bookkeeping -- no FFT)
# ═══════════════════════════════════════════════════════════════════════════

N_induced = N_orbit * n_local
print(f"\nInduced band-representation dimension = N_orbit * n_local = "
      f"{N_orbit} * {n_local} = {N_induced}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Coset decomposition: g g_alpha = {E|L_{beta alpha}(g)} g_beta h_{beta alpha}(g)
# ═══════════════════════════════════════════════════════════════════════════
#
# Derivation (see module docstring): using the Stage-1 t_eff convention for h
# (so h(q) = q EXACTLY, not just mod lattice), and g_alpha/g_beta adjusted so
# g_alpha(q)=q_alpha, g_beta(q)=q_beta exactly (Section 2), applying both
# sides of the coset relation to q collapses the translation part to
#     Q := g(g_alpha(q)) = q_beta + L        (L integer -- this fixes L)
# while the rotation part fixes R_h uniquely:
#     R_g R_alpha = R_beta R_h   =>   R_h = R_beta^{-1} (R_g R_alpha)
# and h's own stabilizing translation is then forced by definition:
#     t_eff_h = q - R_h q
# which is matched against the Stage-1 site-operation list by (R, t_eff) --
# never by list position, per the task specification.
#
# IMPORTANT: this decomposition is written as a function of the RAW (Rg, tg)
# values, not of a catalogued operation index. spglib's Nop_total operations
# are only defined mod an overall lattice translation, so the "product" of
# two catalogued operations (Rg Rh, Rg th + tg) generally differs from the
# catalogued representative stored at that rotation by an integer lattice
# vector. That integer is invisible to the rotation/closure bookkeeping but
# NOT to the Bloch phase (nonsymmorphic groups do not admit a translation-
# consistent set of coset representatives). Any composition-law check must
# therefore decompose the EXACT composed operation on the fly rather than
# looking up a canonically-reduced "gh" from a product table -- otherwise a
# spurious phase (e.g. exactly -1 at a zone-boundary point) appears that has
# nothing to do with the induction itself.

coset_R_inv = np.zeros((N_orbit, 3, 3), dtype=int)
for alpha in range(N_orbit):
    ok, Rinv = is_integer_mat(np.linalg.inv(coset_R[alpha].astype(float)))
    if not ok:
        raise RuntimeError(f"Orbit rep {alpha}: R_alpha^-1 not integer.")
    coset_R_inv[alpha] = Rinv.astype(int)


def decompose_coset(Rg, tg, alpha):
    """Solve g g_alpha = {E|L} g_beta h for an ARBITRARY affine (Rg,tg) (not
    necessarily one of the catalogued Nop_total operations). Returns
    (beta, L, m_site_op, recon_rot_err, recon_trans_err)."""
    R_prod = Rg @ coset_R[alpha]
    t_prod = Rg.astype(float) @ coset_t[alpha] + tg
    Q = R_prod.astype(float) @ q + t_prod   # == g(g_alpha(q)), continuous

    beta, L = -1, None
    for c in range(N_orbit):
        diff = Q - orbit_points[c]
        ok, Lc = is_integer_vec(diff)
        if ok:
            beta, L = c, Lc
            break
    if beta < 0:
        raise RuntimeError(f"alpha={alpha}: Q={Q} matches no stored orbit point "
                            f"modulo a lattice vector.")

    R_h_float = coset_R_inv[beta].astype(float) @ R_prod
    ok, R_h = is_integer_mat(R_h_float)
    if not ok:
        raise RuntimeError(f"alpha={alpha}: R_beta^-1 (R_g R_alpha) is not "
                            f"integer -- coset algebra is inconsistent.")
    R_h = R_h.astype(int)

    t_eff_h_pred = q - R_h.astype(float) @ q
    m_match = -1
    for m in range(Ng):
        if np.array_equal(R_site[m], R_h) and \
           np.max(np.abs(t_eff_site[m] - t_eff_h_pred)) < 1e-6:
            m_match = m
            break
    if m_match < 0:
        raise RuntimeError(f"alpha={alpha}: no Stage-1 site operation matches "
                            f"(R_h={R_h.tolist()}, t_eff={t_eff_h_pred.round(6).tolist()}).")

    R_recon = coset_R[beta] @ R_site[m_match]
    t_recon = coset_R[beta].astype(float) @ t_eff_site[m_match] + coset_t[beta] + L
    recon_rot_err   = float(np.max(np.abs(R_recon - R_prod)))
    recon_trans_err = float(np.max(np.abs(t_recon - t_prod)))
    return beta, L, m_match, recon_rot_err, recon_trans_err


beta_of          = np.full((Nop_total, N_orbit), -1, dtype=int)
L_of             = np.zeros((Nop_total, N_orbit, 3))
site_op_index_of = np.full((Nop_total, N_orbit), -1, dtype=int)
recon_rot_err    = np.zeros((Nop_total, N_orbit))
recon_trans_err  = np.zeros((Nop_total, N_orbit))

for g in range(Nop_total):
    for alpha in range(N_orbit):
        beta, L, m_match, rre, rte = decompose_coset(R_full[g], t_full[g], alpha)
        beta_of[g, alpha]          = beta
        L_of[g, alpha]             = L
        site_op_index_of[g, alpha] = m_match
        recon_rot_err[g, alpha]    = rre
        recon_trans_err[g, alpha]  = rte

_max_recon_rot   = float(np.max(recon_rot_err))
_max_recon_trans = float(np.max(recon_trans_err))
print(f"\nCoset decomposition: {Nop_total}*{N_orbit} = {Nop_total * N_orbit} (g,alpha) pairs solved")
print(f"  max reconstruction error (rotation)    = {_max_recon_rot:.3e}  (must be exactly 0, integer matrices)")
print(f"  max reconstruction error (translation) = {_max_recon_trans:.3e}  (should be at Stage-1/machine precision)")
if _max_recon_rot > 0:
    raise RuntimeError("Coset-decomposition rotation reconstruction failed exactly -- bug in the algebra.")
if _max_recon_trans > 1e-6:
    print(f"  WARNING: translation reconstruction error {_max_recon_trans:.3e} exceeds 1e-6 -- "
          f"check the phase convention.")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Induced sewing matrix B_g(k)
# ═══════════════════════════════════════════════════════════════════════════
#
# [B_g(k)]_{beta b, alpha a} = e^{-2pi i k'.L_{beta alpha}(g)} rho(h_{beta alpha}(g))_{ba}
# with k' = R_g^{-T} k. Only the (beta<-alpha) block is nonzero. No further
# phase (e.g. involving t_g or q_alpha) is added -- those effects already sit
# inside L_{beta alpha}(g) from the coset decomposition above.

def _assemble_B(Rg, beta_arr, L_arr, m_arr, k):
    k = np.asarray(k, dtype=float)
    ok, Rg_inv = is_integer_mat(np.linalg.inv(Rg.astype(float)))
    if not ok:
        raise RuntimeError(f"R^-1 not integer for Rg=\n{Rg}")
    k_prime = Rg_inv.T.astype(float) @ k
    B = np.zeros((N_orbit, n_local, N_orbit, n_local), dtype=complex)
    for alpha in range(N_orbit):
        beta  = int(beta_arr[alpha])
        L     = L_arr[alpha]
        m     = int(m_arr[alpha])
        phase = np.exp(-2j * np.pi * np.dot(k_prime, L))
        B[beta, :, alpha, :] = phase * local_site_representation[m]
    return B.reshape(N_induced, N_induced)


def build_sewing_matrix(g, k):
    """Sewing matrix for a CATALOGUED operation g (0..Nop_total-1), using the
    precomputed tables -- this is what is actually reported as D_k(g)."""
    return _assemble_B(R_full[g], beta_of[g], L_of[g], site_op_index_of[g], k)


def build_sewing_matrix_raw(Rg, tg, k):
    """Sewing matrix for an ARBITRARY (not necessarily catalogued) affine
    operation (Rg,tg), decomposed on the fly. Used only for composition-law
    verification against the EXACT (unreduced) product of two operations."""
    beta_arr = np.zeros(N_orbit, dtype=int)
    L_arr    = np.zeros((N_orbit, 3))
    m_arr    = np.zeros(N_orbit, dtype=int)
    for alpha in range(N_orbit):
        beta, L, m_match, _, _ = decompose_coset(Rg, tg, alpha)
        beta_arr[alpha], L_arr[alpha], m_arr[alpha] = beta, L, m_match
    return _assemble_B(Rg, beta_arr, L_arr, m_arr, k)


# ═══════════════════════════════════════════════════════════════════════════
# 6-7. High-symmetry k-points, little groups, D_k(g), characters, and checks
# ═══════════════════════════════════════════════════════════════════════════

kpoint_entries = parse_kpoints(str(KPOINTS_SOURCE))   # [(label, [kx,ky,kz]), ...]
kpoint_labels = [lbl for lbl, _ in kpoint_entries]
kpoints_arr   = np.array([xyz for _, xyz in kpoint_entries], dtype=float)
Nk = len(kpoint_entries)

print(f"\n{sep}\nHIGH-SYMMETRY K-POINTS ({KPOINTS_SOURCE.name})\n{sep}")

little_group_indices_list = []      # per k: list of full-group op indices g
reciprocal_fold_vectors_list = []   # per k: (n_lg,3) integer G_g
induced_matrices_list = []          # per k: (n_lg, N_induced, N_induced) complex
characters_list = []                # per k: (n_lg,) complex
unitarity_errors_list = []          # per k: (n_lg,) float
little_group_law_errors_list = []   # per k: (n_lg,n_lg) float (nan off-closure)
little_group_closure_ok_list = []   # per k: bool
sewing_general_max_error_list = []  # per k: float (worst over ALL g,h pairs)
sewing_general_worst_pair_list = [] # per k: (g,h) worst pair

for ik, (label, k) in enumerate(kpoint_entries):
    k = np.asarray(k, dtype=float)

    little_group = []
    fold_vectors = []
    for g in range(Nop_total):
        k_prime = R_full_inv[g].T.astype(float) @ k
        ok, G = is_integer_vec(k_prime - k)
        if ok:
            little_group.append(g)
            fold_vectors.append(G)
    little_group = np.array(little_group, dtype=int)
    fold_vectors = np.array(fold_vectors, dtype=float)
    n_lg = len(little_group)

    D_k = np.array([build_sewing_matrix(g, k) for g in little_group])   # (n_lg,N,N)
    chi_k = np.array([np.trace(D) for D in D_k])

    unit_err = np.array([
        float(np.linalg.norm(D_k[i].conj().T @ D_k[i] - np.eye(N_induced)))
        for i in range(n_lg)
    ])

    # D_k(g) D_k(h) = D_k(gh) for g,h in the little group. "gh" here means the
    # EXACT composed operation (Rg Rh, Rg th + tg), decomposed on the fly --
    # NOT the canonically mod-lattice-reduced catalogued operation, which can
    # differ from the exact product by an integer lattice vector invisible to
    # rotation/closure bookkeeping but not to the Bloch phase (see Section 4).
    # Closure (whether gh's COSET is itself in the little group) is still
    # checked via op_product_full, since that only concerns which rotation
    # class the product belongs to.
    gl_err = np.full((n_lg, n_lg), np.nan)
    closure_ok = True
    for i, g in enumerate(little_group):
        for j, h in enumerate(little_group):
            gh_cat = op_product_full[g, h]
            if gh_cat not in little_group:
                closure_ok = False
                continue
            R_gh_raw = R_full[g] @ R_full[h]
            t_gh_raw = R_full[g].astype(float) @ t_full[h] + t_full[g]
            D_gh_raw = build_sewing_matrix_raw(R_gh_raw, t_gh_raw, k)
            gl_err[i, j] = np.linalg.norm(D_k[i] @ D_k[j] - D_gh_raw)

    # general sewing composition law B_g(hk) B_h(k) = B_{gh}(k), h and g
    # ranging over the FULL space group (not just the little group at k) --
    # a strictly more general check than little-group closure above. Again
    # "gh" means the exact composed (unreduced) operation.
    worst_err, worst_pair = 0.0, (None, None)
    B_h_k_cache = {h: build_sewing_matrix(h, k) for h in range(Nop_total)}
    for h in range(Nop_total):
        hk = R_full_inv[h].T.astype(float) @ k
        B_h_k = B_h_k_cache[h]
        for g in range(Nop_total):
            R_gh_raw = R_full[g] @ R_full[h]
            t_gh_raw = R_full[g].astype(float) @ t_full[h] + t_full[g]
            B_g_hk = build_sewing_matrix(g, hk)
            B_gh_k = build_sewing_matrix_raw(R_gh_raw, t_gh_raw, k)
            err = float(np.linalg.norm(B_g_hk @ B_h_k - B_gh_k))
            if err > worst_err:
                worst_err, worst_pair = err, (g, h)

    little_group_indices_list.append(little_group)
    reciprocal_fold_vectors_list.append(fold_vectors)
    induced_matrices_list.append(D_k)
    characters_list.append(chi_k)
    unitarity_errors_list.append(unit_err)
    little_group_law_errors_list.append(gl_err)
    little_group_closure_ok_list.append(closure_ok)
    sewing_general_max_error_list.append(worst_err)
    sewing_general_worst_pair_list.append(worst_pair)

    print(f"\n  [{ik}] {label}  k={k.tolist()}")
    print(f"    |little group| = {n_lg}")
    print(f"    little-group closure                 : {'OK' if closure_ok else 'FAILED'}")
    print(f"    max induced-matrix unitarity error    : {float(np.max(unit_err)):.3e}")
    _gl_finite = gl_err[np.isfinite(gl_err)]
    print(f"    max little-group group-law error      : "
          f"{float(np.max(_gl_finite)) if _gl_finite.size else float('nan'):.3e}")
    print(f"    max general sewing composition error  : {worst_err:.3e}  (pair g,h={worst_pair})")
    print(f"    characters:")
    for i, g in enumerate(little_group):
        print(f"      g={g:2d}  R={R_full[g].tolist()}  t={np.round(t_full[g],4).tolist()}  "
              f"chi={chi_k[i]:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
# 9. Optional comparison to the existing DFT IrRep result
# ═══════════════════════════════════════════════════════════════════════════

dft_comparison = []   # list of dicts, one per matched k-point

if ENABLE_DFT_COMPARISON and DFT_SYMMETRY_JSON.exists():
    print(f"\n{sep}\nOPTIONAL COMPARISON TO DFT IRREP RESULT ({DFT_SYMMETRY_JSON.name})\n{sep}")
    with open(DFT_SYMMETRY_JSON, encoding="utf-8") as f:
        dft = json.load(f)

    occ_by_label = {o["kp_label"]: o for o in dft.get("occupied_subspace", []) if o}
    kp_by_label  = {kp["label"]: kp for kp in dft.get("kpoints", [])}

    for ik, label in enumerate(kpoint_labels):
        if label not in occ_by_label or label not in kp_by_label:
            print(f"  [{label}] not present in DFT JSON -- skipped")
            continue
        occ = occ_by_label[label]
        # ind -> (R,t) lookup, from any block's symmetry_operations at this k
        blocks = kp_by_label[label]["blocks"]
        if not blocks:
            continue
        ind_to_Rt = {
            so["ind"]: (np.array(so["rotation"], dtype=int), np.array(so["translation"], dtype=float))
            for so in blocks[0]["symmetry_operations"]
        }

        our_little_group = little_group_indices_list[ik]
        our_chi = characters_list[ik]

        matched = []   # (our_g, dft_ind, our_chi, dft_chi)
        for i_g, g in enumerate(our_little_group):
            for ind, (Rd, td) in ind_to_Rt.items():
                if np.array_equal(R_full[g], Rd) and \
                   np.max(np.abs(wrap_frac(t_full[g]) - wrap_frac(td))) < 1e-4:
                    if str(ind) in occ["D_raw"]:
                        raw = np.array(occ["D_raw"][str(ind)], dtype=float)   # (dim,dim,2) [re,im]
                        dft_D = raw[..., 0] + 1j * raw[..., 1]
                        matched.append((g, ind, our_chi[i_g], complex(np.trace(dft_D))))
                    break

        dim_ours = N_induced
        dim_dft  = occ["dimension"]
        chi_ours = np.array([m[2] for m in matched])
        chi_dft  = np.array([m[3] for m in matched])
        norm_diff = float(np.linalg.norm(chi_ours - chi_dft)) if len(matched) else float('nan')

        print(f"\n  [{label}]  matched {len(matched)}/{len(our_little_group)} little-group operations")
        print(f"    dimension: ours={dim_ours}  DFT occupied subspace={dim_dft}")
        print(f"    character-vector norm difference (matched ops) = {norm_diff:.4f}")
        for g, ind, c_ours, c_dft in matched:
            print(f"      g={g:2d} (DFT ind={ind:2d})  chi_ours={c_ours:.4f}  chi_DFT={c_dft:.4f}  "
                  f"diff={abs(c_ours - c_dft):.4f}")

        dft_comparison.append({
            "label": label, "dimension_ours": dim_ours, "dimension_dft": dim_dft,
            "n_matched": len(matched), "n_little_group": len(our_little_group),
            "chi_ours": chi_ours, "chi_dft": chi_dft, "norm_diff": norm_diff,
            "matched_g": np.array([m[0] for m in matched]),
            "matched_dft_ind": np.array([m[1] for m in matched]),
        })
elif ENABLE_DFT_COMPARISON:
    print(f"\n(DFT comparison enabled but {DFT_SYMMETRY_JSON} not found -- skipped)")


# ═══════════════════════════════════════════════════════════════════════════
# 8. Save outputs: .npz + readable text report
# ═══════════════════════════════════════════════════════════════════════════

def _obj_array(items):
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.savez(OUTPUT_DIR / f"induced_band_rep_{MATERIAL}.npz",
    material                 = MATERIAL,
    spacegroup_label          = sg_label,
    q                         = q,
    local_cno_indices         = np.array(local_cno_indices),
    local_occupations         = local_occ,
    local_site_representation = local_site_representation,
    local_subspace_groups     = _obj_array([np.array(g) for g in SELECTED_LOCAL_CNOS]),
    local_unitarity_error     = _unitary_err,
    local_group_law_error     = _gl_err,
    local_leakage             = _leak,
    spacegroup_R              = R_full,
    spacegroup_t              = t_full,
    operation_product_table   = op_product_full,
    N_orbit                   = N_orbit,
    Ng_site                   = Ng,
    orbit_points              = orbit_points,
    coset_R                   = coset_R,
    coset_t                   = coset_t,
    orbit_action_beta         = beta_of,
    orbit_translation_L       = L_of,
    orbit_site_op_index       = site_op_index_of,
    coset_recon_rot_error_max   = _max_recon_rot,
    coset_recon_trans_error_max = _max_recon_trans,
    kpoints                   = kpoints_arr,
    kpoint_labels             = np.array(kpoint_labels),
    little_group_indices      = _obj_array(little_group_indices_list),
    reciprocal_fold_vectors   = _obj_array(reciprocal_fold_vectors_list),
    induced_matrices          = _obj_array(induced_matrices_list),
    characters                = _obj_array(characters_list),
    unitarity_errors          = _obj_array(unitarity_errors_list),
    group_law_errors          = _obj_array(little_group_law_errors_list),
    little_group_closure_ok   = np.array(little_group_closure_ok_list),
    sewing_general_max_error  = np.array(sewing_general_max_error_list),
    sewing_general_worst_pair = _obj_array(sewing_general_worst_pair_list),
    dft_comparison            = _obj_array(dft_comparison),
)

# ── readable text report, organized by k-point ──────────────────────────────
_r = _io.StringIO()
def w(s=""): _r.write(s + "\n")

w(sep); w("INDUCED BAND REPRESENTATION -- STAGE 2 REPORT"); w(sep)
w(f"Material               : {MATERIAL}   space group: {sg_label}")
w(f"WS center q (fractional): {q.tolist()}")
w(f"Site-symmetry order Ng = {Ng}   |G/T| = {Nop_total}   N_orbit = {N_orbit}")
w(f"Selected local CNO subspace groups : {SELECTED_LOCAL_CNOS}")
w(f"local_cno_indices (flattened)      : {local_cno_indices}   n_local = {n_local}")
w(f"local_occupations                  : {local_occ.tolist()}")
w(f"Stage-1 input quality for this selection (raw, unmodified):")
w(f"  max unitarity error = {_max_unitary:.3e}   max group-law error = {_max_gl:.3e}   "
  f"max leakage = {_max_leak:.3e}")
w(f"Induced band-representation dimension = N_orbit * n_local = {N_orbit} * {n_local} = {N_induced}")
w()
w(f"Orbit points q_alpha (fractional):")
for alpha in range(N_orbit):
    w(f"  alpha={alpha}: {orbit_points[alpha].round(6).tolist()}  (rep g={coset_rep_g[alpha]})")
w()
w(f"Coset-decomposition reconstruction error: rotation max = {_max_recon_rot:.3e}  "
  f"translation max = {_max_recon_trans:.3e}")

for ik, label in enumerate(kpoint_labels):
    little_group = little_group_indices_list[ik]
    D_k          = induced_matrices_list[ik]
    chi_k        = characters_list[ik]
    unit_err     = unitarity_errors_list[ik]
    gl_err       = little_group_law_errors_list[ik]
    k            = kpoints_arr[ik]

    w(); w(sep); w(f"K-POINT [{ik}] {label}   k (fractional) = {k.tolist()}"); w(sep)
    w(f"1. Orbit size N_orbit={N_orbit}, n_local={n_local}, induced dimension={N_induced}")
    w(f"2. Selected local CNO subspace: {SELECTED_LOCAL_CNOS} (indices {local_cno_indices})")
    w(f"   Stage-1 quality: unitarity={_max_unitary:.3e}  group-law={_max_gl:.3e}  leakage={_max_leak:.3e}")
    w(f"3. Little-group operation indices ({len(little_group)}): {little_group.tolist()}")
    w(f"   little-group closure: {'OK' if little_group_closure_ok_list[ik] else 'FAILED'}")
    w(f"4. D_k(g) matrices and characters:")
    for i, g in enumerate(little_group):
        w(f"   g={g:2d}  R={R_full[g].tolist()}  t={np.round(t_full[g], 4).tolist()}  "
          f"chi={chi_k[i]:.6f}")
        for row in D_k[i]:
            w("     [" + ", ".join(f"{v.real:+.4f}{v.imag:+.4f}j" for v in row) + "]")
    _gl_fin = gl_err[np.isfinite(gl_err)]
    w(f"5. Max unitarity error = {float(np.max(unit_err)):.3e}   "
      f"max little-group group-law error = {float(np.max(_gl_fin)) if _gl_fin.size else float('nan'):.3e}   "
      f"max general sewing composition error = {sewing_general_max_error_list[ik]:.3e}")

if dft_comparison:
    w(); w(sep); w("COMPARISON TO DFT IRREP RESULT"); w(sep)
    for c in dft_comparison:
        w(f"  [{c['label']}]  matched {c['n_matched']}/{c['n_little_group']} ops   "
          f"dim ours={c['dimension_ours']}  dim DFT={c['dimension_dft']}   "
          f"character-vector norm diff={c['norm_diff']:.4f}")

txt_path = OUTPUT_DIR / f"induced_band_rep_{MATERIAL}.txt"
txt_path.write_text(_r.getvalue(), encoding="utf-8")

print(f"\nReport : {txt_path}")
print(f"Data   : {OUTPUT_DIR / f'induced_band_rep_{MATERIAL}.npz'}")

