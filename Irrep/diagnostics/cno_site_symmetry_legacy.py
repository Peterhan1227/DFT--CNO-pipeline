"""
site_symmetry.py — find the site-symmetry group at a WS center and verify how
the CNO space transforms under it, with the numerical accuracy of every check
traced back to its actual source (exact permutation vs. interpolation error,
subspace incompleteness, or genuine representation non-closure).

Run with the 'irrep' conda environment (has spglib + numpy + scipy).

Deprecated: this historical exploration assumes the former one-to-one WS
sample map.  It is retained for reproducing its old Si report only.  Use
``Density matrix cal/symmetry/cno_symmetry.py`` for current finite-volume CNO
outputs, which may contain an expanded weighted WS sample map.

Physics summary
---------------
For a chosen WS center q, spglib gives the full crystal symmetry (R, t) in
fractional coordinates. The subset that fixes q up to a lattice translation,

    R q + t = q + T          (T = integer lattice vector),

is the site-symmetry group. Each such op is reported only by its raw data:
operation index, R, spglib translation t, integer translation T, and the
center-fixing translation t_eff = t - T (so that R q + t_eff = q exactly).
No point-group class, Bilbao label, or irrep is assigned here.

U_g acts on a CNO as (U_g phi)(r) = phi(g^{-1} r), with g the center-fixing
operation (R, t_eff). Since R q + t_eff = q,

    g^{-1}(r) = R^{-1}(r - t_eff) = R^{-1}(r - q) + q = (r - q) R^{-T} + q

(fractional coordinates), independent of t_eff itself.

Two regimes, handled by genuinely different (and separately verified) code
paths:

  EXACT ops   -- g^{-1} maps the unfolded WS grid onto itself as an integer
                 permutation. The CNO data is periodic on the primitive FFT
                 grid (the WS relabeling is bookkeeping for *position* only,
                 not for the stored *value*), so the correct exact lookup key
                 is the primitive FFT index (ix,iy,iz) mod N, not the raw
                 unfolded actual_idx. Using the unfolded index directly turns
                 out to be a common bug: a handful of WS-boundary points are
                 tied between periodic images by the (lexicographic, not
                 rotation-symmetric) tie-break in build_ws_grid_map, so their
                 *recorded* unfolded coordinate is not what a naive exact
                 rotation would predict -- even though the underlying data
                 value is completely well defined via the primitive index.
                 Before this permutation is ever applied to real CNO data, it
                 is checked purely geometrically: bijectivity, p_g^-1 . p_g
                 = I, and p_g . p_h = p_gh (for all triples where g, h and gh
                 are all exact). Any failure aborts with an error -- this is
                 exactly the kind of bug that otherwise hides as a few-percent
                 "exact-operation" group-law error.

  INTERPOLATED ops -- mask-normalized trilinear interpolation on an unfolded
                 supercell embedding (no primitive-cell folding, no
                 mode='wrap'), reusing the method from
                 `Density matrix cal/symmetry_adapt_cnos.py`. The supercell
                 is now sized automatically from the actual data footprint and
                 every operation's source-coordinate range (plus a stencil
                 pad), rather than assuming a fixed 3x3x3 box is large enough.

For every operation and every included CNO, three logically distinct checks
are kept separate (never averaged together):

  1. transformation accuracy : ||U_g phi_a||^2 vs 1, and the round-trip
                                ||U_(g^-1) U_g phi_a - phi_a|| / ||phi_a||.
  2. subspace closure        : leakage = 1 - ||P_B U_g phi_a||^2/||U_g phi_a||^2
                                for whichever candidate block B contains a.
  3. representation consistency : ||D^dagger(g) D(g) - I|| and
                                ||D(g) D(h) - D(gh)|| on D_ba(g) = <phi_b|U_g|phi_a>.

D_all(g) = Phi_all^dagger U_g Phi_all is built, UNMODIFIED (no renormalizing
transformed CNOs, no polar-unitarizing D), for every included CNO and every
site-symmetry operation; this raw matrix is the primary saved output.
Candidate degenerate blocks are proposed from occupations using a full-window
spread test (max-min < atol+rtol*mean), not chained adjacent gaps, so a slow
drift through several near-equal eigenvalues cannot silently merge unrelated
states. Manual blocks may be supplied instead.
"""

import sys
import numpy as np
from pathlib import Path
from scipy.ndimage import map_coordinates

# ── paths ─────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).resolve().parent
PIPELINE_DIR = HERE.parents[1] / "Density matrix cal"
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(PIPELINE_DIR / "helper functions"))

from ws_cell import read_poscar_structure, parse_ws_center
import spglib

# ── configuration ─────────────────────────────────────────────────────────────
MATERIAL      = "Si"
OUTPUT_SUBDIR = "full_occ_mirror_sym"
CNO_SOURCE    = "cno_orbitals_pre_sym_adapt.npy"   # raw CNOs, before any sym-adapt rotation

# WS center — must match Density matrix cal/config.py
WS_CENTER            = [0.875, 0.875, 0.875]
WS_CENTER_COORD_TYPE = "fractional"

OCC_CUTOFF = 0.03    # include every CNO with occupation above this

# Candidate degenerate blocks: full-window spread test, not chained gaps.
BLOCK_ATOL = 1e-4    # absolute occupation tolerance
BLOCK_RTOL = 0.02    # relative occupation tolerance (fraction of window mean)
MANUAL_BLOCKS = None # e.g. [[2, 3], [5, 6]] to override automatic proposal;
                      # any included CNO not listed is tested as a singleton.

GRID_EXACT_TOL      = 1e-6         # voxel-coordinate tolerance for "maps exactly onto the grid"
SYMPREC             = 1e-5          # spglib tolerance (Angstrom)
COVERAGE_THRESHOLDS = (0.5, 0.9, 0.999)  # interpolation-coverage reporting thresholds
INTERP_PAD          = 2             # voxels of padding added beyond the data/target bounding box

OUTPUT_DIR = HERE / "output" / MATERIAL

# ── load saved CNO / WS-grid data ──────────────────────────────────────────────
data_dir = PIPELINE_DIR / "Data" / MATERIAL
cno_dir  = data_dir / "output" / OUTPUT_SUBDIR
poscar   = data_dir / "POSCAR"

latvec, species, counts, atom_syms, atom_nums, frac_coords, cart_coords = \
    read_poscar_structure(poscar)
volume = abs(np.dot(latvec[0], np.cross(latvec[1], latvec[2])))

cno_occ  = np.load(cno_dir / "cno_occupations.npy")
cno_orb  = np.load(cno_dir / CNO_SOURCE)                          # (Nr, n_cno) complex
ws_frac  = np.load(cno_dir / "ws_points_frac_cont.npy")           # (Nr,3) actual, NOT wrapped
ws_base  = np.load(cno_dir / "ws_base_indices.npy").astype(int)   # (Nr,3) FFT primitive indices
ws_trans = np.load(cno_dir / "ws_translation_int.npy").astype(int)  # (Nr,3) integer n
fft_shape = tuple(np.load(cno_dir / "fft_grid_shape.npy").astype(int).tolist())

Nr         = len(ws_frac)
Nx, Ny, Nz = fft_shape
Nvec       = np.array([Nx, Ny, Nz], dtype=int)

if Nr != Nx * Ny * Nz:
    raise ValueError(f"Nr={Nr} does not match Nx*Ny*Nz={Nx*Ny*Nz} -- the WS grid is not "
                      f"a full relabeling of the primitive FFT grid; the exact-permutation "
                      f"path below assumes it is.")

# actual (non-primitive-cell-folded) integer grid coordinates, exactly as in
# symmetry_adapt_cnos.py: actual_idx = ws_base_indices + N * ws_translation_int
actual_idx = ws_base + Nvec * ws_trans   # (Nr, 3)

_pos_check = actual_idx / Nvec.astype(float)
_max_coord_err = float(np.max(np.abs(ws_frac - _pos_check)))
if _max_coord_err > 1e-6:
    raise ValueError(
        f"ws_points_frac_cont and actual_idx disagree by {_max_coord_err:.3e} — check .npy files.")

# primitive-FFT-index -> row lookup. The CNO data is periodic on this grid: the
# WS "unfolded" position is bookkeeping for where a point sits in space, not
# for which value is stored there. ws_base is a bijection onto all Nx*Ny*Nz
# primitive combinations (asserted above via Nr == Nx*Ny*Nz), so this array is
# total.
prim_row_of = -np.ones(fft_shape, dtype=int)
prim_row_of[ws_base[:, 0], ws_base[:, 1], ws_base[:, 2]] = np.arange(Nr)
if np.any(prim_row_of < 0):
    raise ValueError("ws_base_indices does not cover every primitive FFT grid point exactly once.")

# ── crystal symmetry via spglib ───────────────────────────────────────────────
cell = (latvec, frac_coords, atom_nums)
sg_label = spglib.get_spacegroup(cell, symprec=SYMPREC)
sym = spglib.get_symmetry(cell, symprec=SYMPREC)
rotations    = sym['rotations']     # (Nop, 3, 3) int
translations = sym['translations']  # (Nop, 3) float, fractional
Nop_total = len(rotations)

# ── site-symmetry filter: R q + t = q + T ────────────────────────────────────
center_cart, q, q_wrap = parse_ws_center(WS_CENTER, WS_CENTER_COORD_TYPE, latvec)

site_ops = []
for i, (R, t) in enumerate(zip(rotations, translations)):
    Rq_t = R.astype(float) @ q + t
    T    = np.round(Rq_t - q).astype(int)
    err  = np.linalg.norm(Rq_t - q - T)
    if err < 1e-5:
        R_inv_f   = np.linalg.inv(R.astype(float))
        R_inv_int = np.round(R_inv_f).astype(int)
        if not np.allclose(R_inv_f, R_inv_int, atol=1e-6):
            raise ValueError(f"R^-1 is not integer for global op {i}: R=\n{R}")
        site_ops.append({
            'R': R.astype(int), 't': t, 'T': T, 't_eff': t - T,
            'R_inv': R_inv_int, 'global_idx': i,
        })

Ng = len(site_ops)

# ── operation product / inverse tables: purely from (R,t) algebra ────────────
def _op_key(R, t):
    t_w = tuple(np.round(((t + 0.5) % 1.0 - 0.5), 6).tolist())
    return (tuple(int(x) for x in R.ravel()), t_w)

op_index = {_op_key(op['R'], op['t']): k for k, op in enumerate(site_ops)}

operation_product_table = -np.ones((Ng, Ng), dtype=int)
for i, gi in enumerate(site_ops):
    for j, hj in enumerate(site_ops):
        R_gh = gi['R'] @ hj['R']
        t_gh = gi['R'].astype(float) @ hj['t'] + gi['t']
        operation_product_table[i, j] = op_index.get(_op_key(R_gh, t_gh), -1)

n_missing_products = int(np.sum(operation_product_table < 0))

identity_idx = next(i for i, o in enumerate(site_ops)
                    if np.array_equal(o['R'], np.eye(3, dtype=int)) and np.allclose(o['t'], 0.0, atol=1e-8))
operation_inverse_table = np.array(
    [int(np.where(operation_product_table[i] == identity_idx)[0][0]) for i in range(Ng)])

# ── per-op source coordinates and grid-exactness ─────────────────────────────
# g^{-1}(r) = (r - q) R^{-T} + q  — no primitive-cell folding, no mode='wrap'.
for op in site_ops:
    r_src = (ws_frac - q[None, :]) @ op['R_inv'].T + q[None, :]   # (Nr, 3) fractional
    voxel = r_src * Nvec[None, :]                                  # (Nr, 3) voxel units, unwrapped
    dev   = np.abs(voxel - np.round(voxel))
    op['grid_max_dev'] = float(np.max(dev))
    op['grid_exact']   = op['grid_max_dev'] < GRID_EXACT_TOL
    op['_voxel']       = voxel

# ── exact ops: build + geometrically verify the permutation BEFORE touching CNOs
for op in site_ops:
    if not op['grid_exact']:
        continue
    target_prim = np.round(op['_voxel']).astype(int) % Nvec[None, :]   # always in [0,N)
    perm = prim_row_of[target_prim[:, 0], target_prim[:, 1], target_prim[:, 2]]
    if np.any(perm < 0) or len(np.unique(perm)) != Nr:
        raise RuntimeError(
            f"Op global_idx={op['global_idx']}: grid-exact but the induced map is not a "
            f"bijection over the WS grid ({np.sum(perm < 0)} unmatched, "
            f"{Nr - len(np.unique(perm))} collisions). The grid-exactness tolerance is "
            f"inconsistent with the actual data -- fix this before trusting any result.")
    op['perm'] = perm

_perm_inverse_checks = 0
_perm_product_checks = 0
for i, gi in enumerate(site_ops):
    if not gi['grid_exact']:
        continue
    j = int(operation_inverse_table[i])
    hj = site_ops[j]
    if hj['grid_exact']:
        composed = gi['perm'][hj['perm']]
        if not np.array_equal(composed, np.arange(Nr)):
            n_bad = int(np.sum(composed != np.arange(Nr)))
            raise RuntimeError(
                f"Permutation inverse-consistency failed: op {i} (global_idx="
                f"{gi['global_idx']}) composed with its inverse op {j} (global_idx="
                f"{hj['global_idx']}) is not the identity permutation ({n_bad}/{Nr} rows differ).")
        _perm_inverse_checks += 1
    for k_j, hj2 in enumerate(site_ops):
        if not hj2['grid_exact']:
            continue
        k = int(operation_product_table[i, k_j])
        if k < 0 or not site_ops[k]['grid_exact']:
            continue
        composed = hj2['perm'][gi['perm']]
        if not np.array_equal(composed, site_ops[k]['perm']):
            n_bad = int(np.sum(composed != site_ops[k]['perm']))
            raise RuntimeError(
                f"Permutation product-consistency failed: op {i} (global_idx={gi['global_idx']}) "
                f"composed with op {k_j} (global_idx={hj2['global_idx']}) does not match op {k} "
                f"(global_idx={site_ops[k]['global_idx']}) ({n_bad}/{Nr} rows differ).")
        _perm_product_checks += 1

n_exact_ops = sum(1 for op in site_ops if op['grid_exact'])

# ── dynamically-sized supercell embedding for interpolated ops ───────────────
# Sized from the actual data footprint AND every operation's source-coordinate
# range, not assumed to be 3x3x3 -- material-independent.
_all_voxel_min = np.min([op['_voxel'].min(axis=0) for op in site_ops], axis=0)
_all_voxel_max = np.max([op['_voxel'].max(axis=0) for op in site_ops], axis=0)
_bbox_min = np.minimum(actual_idx.min(axis=0), np.floor(_all_voxel_min).astype(int)) - INTERP_PAD
_bbox_max = np.maximum(actual_idx.max(axis=0), np.ceil(_all_voxel_max).astype(int)) + INTERP_PAD

offset      = -_bbox_min                                    # (3,)
super_shape = tuple((_bbox_max - _bbox_min + 1).astype(int).tolist())
IJK_super   = actual_idx + offset[None, :]                   # (Nr, 3) fixed placement

mask_super = np.zeros(super_shape, dtype=float)
mask_super[IJK_super[:, 0], IJK_super[:, 1], IJK_super[:, 2]] = 1.0

def embed_columns(columns):
    """Embed (Nr,k) column data into the supercell frame, once per column."""
    k = columns.shape[1]
    re_stack = np.zeros((k,) + super_shape)
    im_stack = np.zeros((k,) + super_shape)
    for a in range(k):
        re_stack[a][IJK_super[:, 0], IJK_super[:, 1], IJK_super[:, 2]] = columns[:, a].real
        im_stack[a][IJK_super[:, 0], IJK_super[:, 1], IJK_super[:, 2]] = columns[:, a].imag
    return re_stack, im_stack

def interpolate_from_embedding(re_stack, im_stack, voxel):
    """Mask-normalized trilinear interpolation at unfolded `voxel` coordinates."""
    coords_T = (voxel + offset[None, :]).T
    den  = map_coordinates(mask_super, coords_T, order=1, mode="constant", cval=0.0)
    safe = den > 1e-12
    den_safe = np.where(safe, den, 1.0)
    k = re_stack.shape[0]
    out = np.zeros((coords_T.shape[1], k), dtype=complex)
    for a in range(k):
        num_re = map_coordinates(re_stack[a] * mask_super, coords_T, order=1, mode="constant", cval=0.0)
        num_im = map_coordinates(im_stack[a] * mask_super, coords_T, order=1, mode="constant", cval=0.0)
        out[:, a] = np.where(safe, (num_re + 1j * num_im) / den_safe, 0.0)
    return out, den

def transform_columns(op, columns, precomputed=None):
    """Apply U_g to (Nr,k) column data: exact permutation, or fresh interpolation."""
    if op['grid_exact']:
        return columns[op['perm'], :]
    if precomputed is not None:
        re_stack, im_stack = precomputed
    else:
        re_stack, im_stack = embed_columns(columns)
    out, _ = interpolate_from_embedding(re_stack, im_stack, op['_voxel'])
    return out

def op_coverage(op):
    if op['grid_exact']:
        return {'exact': True, 'min_den': 1.0, 'mean_den': 1.0,
                'frac_below': {thr: 0.0 for thr in COVERAGE_THRESHOLDS}, 'n_zero': 0}
    coords_T = (op['_voxel'] + offset[None, :]).T
    den = map_coordinates(mask_super, coords_T, order=1, mode="constant", cval=0.0)
    return {'exact': False, 'min_den': float(den.min()), 'mean_den': float(den.mean()),
            'frac_below': {thr: float((den < thr).mean()) for thr in COVERAGE_THRESHOLDS},
            'n_zero': int(np.sum(den <= 1e-12))}

# ── included CNOs (occupation cutoff) ──────────────────────────────────────────
cno_indices = np.where(cno_occ > OCC_CUTOFF)[0]
n_inc = len(cno_indices)
Phi   = cno_orb[:, cno_indices]                 # (Nr, n_inc), RAW basis -- never rotated
pos_of = {int(idx): k for k, idx in enumerate(cno_indices)}

input_overlap_matrix = Phi.conj().T @ Phi        # (n_inc, n_inc), ideally ~ identity

Phi_super_re, Phi_super_im = embed_columns(Phi)  # precomputed ONCE, reused across all Ng ops

# ── candidate blocks: full-window spread test, not chained adjacent gaps ─────
def propose_blocks(indices, occ, atol, rtol):
    blocks = []
    i, n = 0, len(indices)
    while i < n:
        j = i
        while j + 1 < n:
            window = occ[indices[i:j + 2]]
            spread = float(window.max() - window.min())
            mean_ = float(np.mean(np.abs(window)))
            if spread < atol + rtol * mean_:
                j += 1
            else:
                break
        blocks.append([int(x) for x in indices[i:j + 1]])
        i = j + 1
    return blocks

if MANUAL_BLOCKS is not None:
    manual_set = {i for b in MANUAL_BLOCKS for i in b}
    if not manual_set.issubset(set(cno_indices.tolist())):
        raise ValueError("MANUAL_BLOCKS references a CNO index not above OCC_CUTOFF.")
    candidate_blocks = [list(b) for b in MANUAL_BLOCKS]
    candidate_blocks += [[int(i)] for i in cno_indices if int(i) not in manual_set]
else:
    candidate_blocks = propose_blocks(cno_indices, cno_occ, BLOCK_ATOL, BLOCK_RTOL)

singleton_idx     = [g[0] for g in candidate_blocks if len(g) == 1]
degenerate_blocks = [g for g in candidate_blocks if len(g) > 1]

# ── per-operation representation on the full included-CNO space ─────────────
op_results = []
for op in site_ops:
    phi_t = transform_columns(op, Phi, precomputed=(Phi_super_re, Phi_super_im))
    transformed_norm2 = np.sum(np.abs(phi_t) ** 2, axis=0)    # (n_inc,) ||U_g phi_a||^2
    D = Phi.conj().T @ phi_t                                   # (n_inc, n_inc) — RAW, not renormalized
    retained_weight = np.sum(np.abs(D) ** 2, axis=0)           # (n_inc,) ||D[:,a]||^2
    op_results.append({'D': D, 'transformed_norm2': transformed_norm2,
                        'retained_weight': retained_weight, 'coverage': op_coverage(op),
                        'phi_t': phi_t})

D_all                 = np.array([r['D'] for r in op_results])                    # (Ng,n_inc,n_inc)
characters_all         = np.array([np.trace(r['D']) for r in op_results])          # (Ng,)
transformed_norm2_all   = np.array([r['transformed_norm2'] for r in op_results])    # (Ng,n_inc)
retained_weight_all     = np.array([r['retained_weight'] for r in op_results])      # (Ng,n_inc)

# ── inverse-consistency: ||U_(g^-1) U_g phi_a - phi_a|| / ||phi_a|| ──────────
inverse_consistency_all = np.zeros((Ng, n_inc))
_phi_norms = np.linalg.norm(Phi, axis=0)
for g, op in enumerate(site_ops):
    op_inv = site_ops[int(operation_inverse_table[g])]
    phi_tt = transform_columns(op_inv, op_results[g]['phi_t'])
    diffs  = np.linalg.norm(phi_tt - Phi, axis=0)
    inverse_consistency_all[g, :] = np.where(_phi_norms > 1e-12, diffs / _phi_norms, np.nan)

for r in op_results:
    del r['phi_t']

# ── group law on the raw D_all ────────────────────────────────────────────────
group_law_error_all = np.full((Ng, Ng), np.nan)
for i in range(Ng):
    for j in range(Ng):
        k = operation_product_table[i, j]
        if k >= 0:
            group_law_error_all[i, j] = np.linalg.norm(D_all[i] @ D_all[j] - D_all[k])

# ── exact / interpolated split helpers ───────────────────────────────────────
exact_mask = np.array([op['grid_exact'] for op in site_ops])

def scalar_split(values):
    """values: (Ng,) array (nan allowed). max + worst op index, for exact / interp / all."""
    idx_all = np.arange(len(values))
    def stat(mask):
        idx = idx_all[mask]
        if len(idx) == 0:
            return {'max': float('nan'), 'worst_op': None}
        vals = values[idx]
        finite = np.isfinite(vals)
        if not finite.any():
            return {'max': float('nan'), 'worst_op': None}
        best = idx[finite][int(np.argmax(vals[finite]))]
        return {'max': float(values[best]), 'worst_op': int(best)}
    return {'exact': stat(exact_mask), 'interp': stat(~exact_mask), 'all': stat(np.ones_like(exact_mask))}

def pair_split(mat):
    """mat: (Ng,Ng) array (nan allowed). 'exact' requires BOTH ops exact; 'interp' means
    at least one op is interpolated."""
    ii, jj = np.meshgrid(np.arange(Ng), np.arange(Ng), indexing='ij')
    both_exact = exact_mask[ii] & exact_mask[jj]
    def stat(mask):
        vals = np.where(mask & np.isfinite(mat), mat, -np.inf)
        flat = int(np.argmax(vals))
        if vals.flat[flat] == -np.inf:
            return {'max': float('nan'), 'worst_pair': None}
        pair = np.unravel_index(flat, mat.shape)
        return {'max': float(mat[pair]), 'worst_pair': (int(pair[0]), int(pair[1]))}
    return {'exact': stat(both_exact), 'interp': stat(~both_exact), 'all': stat(np.ones_like(mat, dtype=bool))}

# ── unified per-block diagnostics (also used for the full included space) ───
def block_diagnostics(block):
    block = [int(i) for i in block]
    pos = [pos_of[i] for i in block]
    dim = len(block)
    D_block = D_all[:, pos, :][:, :, pos]                      # (Ng,dim,dim)

    unitary_err = np.array([float(np.linalg.norm(D_block[g].conj().T @ D_block[g] - np.eye(dim)))
                             for g in range(Ng)])
    gl_err = np.full((Ng, Ng), np.nan)
    for i in range(Ng):
        for j in range(Ng):
            k = operation_product_table[i, j]
            if k >= 0:
                gl_err[i, j] = np.linalg.norm(D_block[i] @ D_block[j] - D_block[k])

    block_retained = np.sum(np.abs(D_block) ** 2, axis=1)       # (Ng,dim) weight retained IN this block
    full_transf    = transformed_norm2_all[:, pos]              # (Ng,dim) ||U_g phi_a||^2
    with np.errstate(divide='ignore', invalid='ignore'):
        leak = 1.0 - block_retained / full_transf
    leak = np.where(full_transf > 1e-12, leak, np.nan)

    norm_err = np.abs(full_transf - 1.0)                        # (Ng,dim)
    inv_cons = inverse_consistency_all[:, pos]                  # (Ng,dim)

    return {
        'block': block, 'dim': dim, 'D_block': D_block,
        'unitary_err': unitary_err, 'gl_err': gl_err, 'leakage': leak,
        'norm_err': norm_err, 'inv_cons': inv_cons,
        'unitary_split':  scalar_split(unitary_err),
        'gl_split':       pair_split(gl_err),
        'leak_split':     scalar_split(np.nanmax(leak, axis=1)),
        'norm_split':     scalar_split(np.nanmax(norm_err, axis=1)),
        'invcons_split':  scalar_split(np.nanmax(inv_cons, axis=1)),
    }

block_diag = {tuple(b): block_diagnostics(b) for b in candidate_blocks}
full_diag  = block_diagnostics(list(cno_indices))

leakage_all         = full_diag['leakage']
unitarity_error_all = full_diag['unitary_err']

# ── build report ───────────────────────────────────────────────────────────────
sep = "-" * 70
import io as _io
_r = _io.StringIO()
def w(s=""): _r.write(s + "\n")

def fmt_scalar(sp):
    def f(s):
        return "n/a" if s['max'] != s['max'] else f"{s['max']:.3e} (op#{s['worst_op']})"
    return f"exact: {f(sp['exact']):<20} interp: {f(sp['interp']):<20} all: {f(sp['all'])}"

def fmt_pair(sp):
    def f(s):
        return "n/a" if s['max'] != s['max'] else f"{s['max']:.3e} (pair {s['worst_pair']})"
    return f"exact: {f(sp['exact']):<24} interp: {f(sp['interp']):<24} all: {f(sp['all'])}"

# 1. run configuration ────────────────────────────────────────────────────────
w(sep); w("1. RUN CONFIGURATION"); w(sep)
w(f"  Material               : {MATERIAL}")
w(f"  Space group (spglib)   : {sg_label}")
w(f"  CNO source             : {CNO_SOURCE}")
w(f"  WS center q (frac)     : {q}")
w(f"  WS center q (Cartesian): {center_cart.round(6)} Ang")
w(f"  Occupation cutoff      : {OCC_CUTOFF}")
w(f"  Block tolerance        : atol={BLOCK_ATOL}, rtol={BLOCK_RTOL}  (full-window spread test)")
w(f"  Manual blocks          : {MANUAL_BLOCKS}")
w(f"  Interpolation          : dynamically-sized supercell (shape {super_shape}), "
  f"mask-normalized trilinear;")
w(f"                           exact ops use a verified integer permutation (no interpolation)")
w(f"  CNOs included          : {n_inc}  (of {len(cno_occ)} total)")
w(f"  Site-symmetry order    : {Ng}  (of {Nop_total} total crystal ops); {n_exact_ops} exact, "
  f"{Ng - n_exact_ops} interpolated")
w(f"  Max |input overlap - I|: {float(np.max(np.abs(input_overlap_matrix - np.eye(n_inc)))):.3e}")

# 2. site-symmetry operations ─────────────────────────────────────────────────
w(); w(sep); w("2. SITE-SYMMETRY OPERATIONS"); w(sep)
w(f"  Each op identified only by (R, t, T, t_eff) — no class label assigned.")
w()
for i, op in enumerate(site_ops):
    w(f"  # {i}   global_idx={op['global_idx']}")
    w(f"    R      = {op['R'].tolist()}")
    w(f"    t      = {op['t'].round(6).tolist()}")
    w(f"    T      = {op['T'].tolist()}")
    w(f"    t_eff  = {op['t_eff'].round(6).tolist()}")
    w(f"    mapping = {'EXACT (verified permutation)' if op['grid_exact'] else 'INTERPOLATED'}"
      f"   max_dev = {op['grid_max_dev']:.3e} voxel")
    w()

# 3. exact-permutation and interpolation-quality checks ──────────────────────
w(sep); w("3. EXACT-PERMUTATION AND INTERPOLATION-QUALITY CHECKS"); w(sep)
w(f"  Exact ops verified bijective            : {n_exact_ops}/{n_exact_ops}")
w(f"  Inverse-consistency checks (p_g^-1 p_g=I): {_perm_inverse_checks} performed, all exact")
w(f"  Product-consistency checks (p_g p_h=p_gh): {_perm_product_checks} performed, all exact")
w(f"  (any failure of the above raises an error before CNO data is touched)")
w()
thr_hdr = "  ".join(f"<{t:g}" for t in COVERAGE_THRESHOLDS)
w(f"  Interpolated-op coverage (denominator statistics):")
w(f"  {'#':>3}  {'min_den':>9}  {'mean_den':>9}  frac_below[{thr_hdr}]  {'n_zero':>7}  "
  f"{'max_inv_cons':>13}")
for i, op in enumerate(site_ops):
    if op['grid_exact']:
        continue
    cov = op_results[i]['coverage']
    fb  = "  ".join(f"{cov['frac_below'][t]:.4f}" for t in COVERAGE_THRESHOLDS)
    max_ic = float(np.nanmax(inverse_consistency_all[i, :]))
    w(f"  {i:3d}  {cov['min_den']:9.4f}  {cov['mean_den']:9.4f}  [{fb}]  {cov['n_zero']:7d}  {max_ic:13.4e}")
if n_exact_ops:
    max_ic_exact = float(np.nanmax(inverse_consistency_all[exact_mask, :])) if n_exact_ops else float('nan')
    w(f"  (exact ops: max inverse-consistency error = {max_ic_exact:.3e} -- should be at machine precision)")

# 4. included CNOs and proposed grouping ─────────────────────────────────────
w(); w(sep); w("4. CNOs INCLUDED (occupation cutoff = %.4g)" % OCC_CUTOFF); w(sep)
for grp in candidate_blocks:
    occ_str = ", ".join(f"{cno_occ[i]:.6f}" for i in grp)
    tag = "singleton" if len(grp) == 1 else f"candidate degenerate block (dim={len(grp)})"
    w(f"  {grp}  occ=[{occ_str}]  -- {tag}")
w()
w(f"  NOTE: grouping requires the FULL window spread max(occ)-min(occ) < "
  f"{BLOCK_ATOL} + {BLOCK_RTOL}*mean(|occ|);")
w(f"  it is a hint for what to test, not proof that a block is closed under the group.")

# 5. singleton summary ────────────────────────────────────────────────────────
w(); w(sep); w("5. SINGLETON SUMMARY"); w(sep)
singleton_blocks = [g for g in candidate_blocks if len(g) == 1]
if singleton_blocks:
    for grp in singleton_blocks:
        bd = block_diag[tuple(grp)]
        idx = grp[0]
        eta = bd['D_block'][:, 0, 0]
        w(f"  CNO {idx}  (occ={cno_occ[idx]:.6f})   min|eta|={np.min(np.abs(eta)):.6f}")
        w(f"    transformation accuracy (norm err)  : {fmt_scalar(bd['norm_split'])}")
        w(f"    transformation accuracy (inv. cons.): {fmt_scalar(bd['invcons_split'])}")
        w(f"    subspace closure (leakage)          : {fmt_scalar(bd['leak_split'])}")
        w(f"    representation consistency (unitary): {fmt_scalar(bd['unitary_split'])}")
        w(f"    representation consistency (gl err) : {fmt_pair(bd['gl_split'])}")
        w()
else:
    w("  (no singletons above the occupation cutoff)")

# 6. candidate-block summary ─────────────────────────────────────────────────
w(sep); w("6. CANDIDATE-BLOCK SUMMARY"); w(sep)
if degenerate_blocks:
    for grp in degenerate_blocks:
        bd = block_diag[tuple(grp)]
        occ_str = ", ".join(f"{cno_occ[i]:.6f}" for i in grp)
        w(f"  Block {grp}  (dim={bd['dim']})")
        w(f"    Occupations                         : [{occ_str}]")
        w(f"    transformation accuracy (norm err)  : {fmt_scalar(bd['norm_split'])}")
        w(f"    transformation accuracy (inv. cons.): {fmt_scalar(bd['invcons_split'])}")
        w(f"    subspace closure (leakage)          : {fmt_scalar(bd['leak_split'])}")
        w(f"    representation consistency (unitary): {fmt_scalar(bd['unitary_split'])}")
        w(f"    representation consistency (gl err) : {fmt_pair(bd['gl_split'])}")
        w()
    w("  DETAIL (appendix): D(g) matrices for each degenerate block")
    w("  " + "-" * 66)
    for grp in degenerate_blocks:
        bd = block_diag[tuple(grp)]
        w(f"  Block {grp}:")
        for g in range(Ng):
            w(f"    op {g}: D =")
            for row in bd['D_block'][g]:
                w("      [" + ", ".join(f"{v.real:+.4f}{v.imag:+.4f}j" for v in row) + "]")
        w()
else:
    w("  (no candidate degenerate blocks above the occupation cutoff)")

# 7. full included-space summary ─────────────────────────────────────────────
w(sep); w("7. FULL INCLUDED CNO-SPACE SUMMARY (D_all)"); w(sep)
w(f"  CNOs included                        : {n_inc}")
w(f"  transformation accuracy (norm err)   : {fmt_scalar(full_diag['norm_split'])}")
w(f"  transformation accuracy (inv. cons.) : {fmt_scalar(full_diag['invcons_split'])}")
w(f"  subspace closure (leakage)           : {fmt_scalar(full_diag['leak_split'])}")
w(f"  representation consistency (unitary) : {fmt_scalar(full_diag['unitary_split'])}")
w(f"  representation consistency (gl err)  : {fmt_pair(full_diag['gl_split'])}")
if n_missing_products:
    w(f"  WARNING: {n_missing_products} operation product(s) not found within the site-symmetry group")

# 8. concise final assessment ─────────────────────────────────────────────────
w(); w(sep); w("8. FINAL ASSESSMENT"); w(sep)
_all_norm    = full_diag['norm_split']['all']['max']
_all_unitary = full_diag['unitary_split']['all']['max']
_all_gl      = full_diag['gl_split']['all']['max']
_verdict = "PASS" if all(x == x and x < 5e-2 for x in (_all_norm, _all_unitary, _all_gl)) else "WARNING"
w(f"  {_verdict}")
w(f"  Exact-permutation geometry             : verified ({n_exact_ops} ops, "
  f"{_perm_inverse_checks} inverse + {_perm_product_checks} product checks, all exact)")
w(f"  Full-space max transformation-norm err : {_all_norm:.3e}")
w(f"  Full-space max unitarity error         : {_all_unitary:.3e}")
w(f"  Full-space max group-law error         : {_all_gl:.3e}")
w(f"  (raw errors as computed; CNOs never renormalized, D(g) never polar-unitarized)")

# ── save .npz and report ───────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
tag = f"{MATERIAL}_verify"

def _obj_array(items):
    arr = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        arr[i] = item
    return arr

interpolation_coverage = _obj_array([op_results[g]['coverage'] for g in range(Ng)])
candidate_block_diagnostics = _obj_array([block_diag[tuple(b)] for b in candidate_blocks])

np.savez(OUTPUT_DIR / f"site_symmetry_{tag}.npz",
    material      = MATERIAL,
    spacegroup_label = sg_label,
    symprec       = SYMPREC,
    latvec        = latvec,
    q             = q,
    q_wrap        = q_wrap,
    q_cart        = center_cart,
    R_ops         = np.array([op['R']     for op in site_ops]),
    t_ops         = np.array([op['t']     for op in site_ops]),
    T_ops         = np.array([op['T']     for op in site_ops]),
    t_eff_ops     = np.array([op['t_eff'] for op in site_ops]),
    operation_product_table = operation_product_table,
    operation_inverse_table = operation_inverse_table,
    grid_exact    = np.array([op['grid_exact']   for op in site_ops]),
    grid_max_dev  = np.array([op['grid_max_dev'] for op in site_ops]),
    interpolation_coverage = interpolation_coverage,
    cno_indices   = cno_indices,
    cno_occupations = cno_occ[cno_indices],
    input_overlap_matrix = input_overlap_matrix,
    D_all                   = D_all,
    characters_all          = characters_all,
    transformed_norm2_all   = transformed_norm2_all,
    inverse_consistency_all = inverse_consistency_all,
    retained_weight_all     = retained_weight_all,
    leakage_all             = leakage_all,
    unitarity_error_all     = unitarity_error_all,
    group_law_error_all     = group_law_error_all,
    candidate_blocks             = _obj_array(candidate_blocks),
    candidate_block_diagnostics  = candidate_block_diagnostics,
)

txt_path = OUTPUT_DIR / f"site_symmetry_{tag}.txt"
txt_path.write_text(_r.getvalue(), encoding="utf-8")

print(f"Report : {txt_path}")
print(f"Data   : {OUTPUT_DIR / f'site_symmetry_{tag}.npz'}")
