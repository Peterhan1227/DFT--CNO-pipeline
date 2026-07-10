"""
symmetry_adapt_cnos.py — symmetry-adapt a nearly-degenerate CNO subspace.

D_ij = <phi_i | R | phi_j>  evaluated by direct interpolation in WS coordinates:

    r_src[n] = (positions[n] - center) @ R_inv.T + center
    (R phi_j)[n] = phi_j(r_src[n])   via scipy.ndimage.map_coordinates

Diagonalising D gives symmetry-adapted combinations: phi_new_a = sum_j phi_j * U[j,a].
"""
import sys
import numpy as np
from pathlib import Path
from scipy.ndimage import map_coordinates
from config import MATERIAL, OUTPUT_SUBDIR, WS_CENTER_COORD_TYPE, WS_CENTER

sys.path.insert(0, str(Path(__file__).resolve().parent / "helper functions"))

# ── settings ──────────────────────────────────────────────────────────────────
index_groups     = [[2, 3], [4, 5], [8, 9]]     # list of degenerate subspaces to symmetry-adapt, e.g. [[2,3],[4,5,6]]
operation        = "c3_z"       # Si (diamond, D3d):   "swap_xy" | "c3_111" | "inversion"
                                 # WSe2_mono (1H, D3h): "c3_z" | "sigma_h" | "sigma_v"
                                 # (D3h has NO inversion center -- see R_CART guard below)
out_name         = "cnos_sym_adapted.npy"
export_cubes     = True         # True: swap cno_orbitals.npy -> out_name, run export_cubes.py, restore
check_occ_cutoff = 0.03         # check non-selected CNOs with occupation above this value

# Diagnostic tolerances — tune to match interpolation quality for your grid.
weight_leakage_warn_tol      = 0.02   # subspace leakage threshold for "consider expanding"
density_residual_warn_tol    = 0.08   # ||R(rho)-rho||/||rho|| threshold
eigenstate_residual_warn_tol = 0.12   # ||R(phi)-eta*phi||/||phi|| threshold
singleton_overlap_warn_tol   = 0.98   # warn when |<phi|R phi>| < this
group_unitarity_warn_tol     = 0.05   # norm(D^HD-I) threshold for degenerate groups

# Center for the symmetry operation in fractional coordinates.
# None → use WS_CENTER from config.py.
# Override only if you want a different center than the WS cell center.
op_center = None

# ── paths and data ─────────────────────────────────────────────────────────────
output_dir = Path(__file__).resolve().parent / "Data" / MATERIAL / "output" / OUTPUT_SUBDIR

cno_orbs = np.load(output_dir / "cno_orbitals.npy")      # (Nr, n_cno), complex128
cno_occ  = np.load(output_dir / "cno_occupations.npy")   # (n_cno,)

Nx, Ny, Nz = np.load(output_dir / "fft_grid_shape.npy").astype(int)
Nvec       = np.array([Nx, Ny, Nz], dtype=int)
Nr         = Nx * Ny * Nz

# ── WS grid: actual integer coordinates ───────────────────────────────────────
ws_base_file  = output_dir / "ws_base_indices.npy"
ws_trans_file = output_dir / "ws_translation_int.npy"
ws_frac_file  = output_dir / "ws_points_frac_cont.npy"

ws_base_indices = (np.load(ws_base_file).astype(int) if ws_base_file.exists()
                   else np.stack(np.unravel_index(np.arange(Nr), (Nx, Ny, Nz)), axis=1))

ws_translation_int = (np.load(ws_trans_file).astype(int) if ws_trans_file.exists()
                      else np.zeros((Nr, 3), dtype=int))

# True grid coordinates before any primitive-cell folding.
# actual_idx[n] in [-Nvec, 2*Nvec) typically for a WS cell.
actual_idx = ws_base_indices + Nvec * ws_translation_int   # (Nr, 3)

positions = (np.load(ws_frac_file) if ws_frac_file.exists()
             else actual_idx / Nvec.astype(float))

# ── coordinate consistency check ───────────────────────────────────────────────
positions_from_actual = actual_idx / Nvec.astype(float)
max_err = np.max(np.abs(positions - positions_from_actual))
print(f"Coord consistency error (max |positions - actual_idx/Nvec|): {max_err:.3e}")
if max_err > 1e-6:
    raise ValueError(
        f"positions and actual_idx disagree by {max_err:.3e} — check .npy files.")

# ── symmetry center and R^{-1} matrix ─────────────────────────────────────────
if op_center is not None:
    center = np.array(op_center, dtype=float)
elif WS_CENTER_COORD_TYPE == "fractional":
    center = np.array(WS_CENTER, dtype=float)
else:
    center = np.load(output_dir / "ws_center_frac_wrapped.npy").astype(float)

# R_inv acts on FRACTIONAL coordinates: r_src = R_inv @ (r - c) + c
#
# Operations are written as CARTESIAN matrices S (acting on Cartesian xyz) and then
# converted to the fractional frame in which they are applied:
#     R_inv = inv(Lᵀ) @ S @ Lᵀ        (L rows = lattice vectors a_i)
# For a non-orthogonal cell, permuting fractional axes is NOT the same Cartesian
# operation, so this conversion is essential.  (Bug fixed: previously S was applied
# directly to fractional coords, so on this FCC lattice "swap_xy" was physically a
# y<->z mirror, not x<->y.)  A crystal symmetry maps the lattice onto itself, so
# R_inv must come out integer — we assert that.
from ws_cell import read_poscar_structure

_latvec = read_poscar_structure(
    Path(__file__).resolve().parent / "Data" / MATERIAL / "POSCAR")[0]   # (3,3) rows a_i

R_CART = {
    # Si (diamond, D3d bond-center site symmetry):
    "inversion": -np.eye(3, dtype=float),                                   # r -> -r about center
    "c3_111":    np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float),  # 120° about cubic [111]
    "swap_xy":   np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float),  # Cartesian x<->y mirror
    # WSe2_mono (1H, D3h point group -- verified against Data/WSe2_mono/POSCAR's
    # hexagonal lattice vectors, integer fractional matrices confirmed):
    "c3_z":      np.array([[-0.5, -np.sqrt(3) / 2, 0],
                            [np.sqrt(3) / 2, -0.5, 0],
                            [0, 0, 1]], dtype=float),                       # 120° rotation about z
    "sigma_h":   np.diag([1.0, 1.0, -1.0]),                                 # horizontal mirror z -> -z
    "sigma_v":   np.diag([1.0, -1.0, 1.0]),                                 # a vertical mirror plane
}

# D3h (WSe2_mono's point group) has NO inversion center -- unlike Si's D3d, "-I"
# is not one of its 12 elements. The integer-fractional-matrix check below would
# NOT catch this (-I is trivially an integer lattice operation for ANY Bravais
# lattice, valid or not as a true site symmetry), so it's guarded explicitly here
# instead of relying on that check to fail.
if operation == "inversion" and MATERIAL == "WSe2_mono":
    raise ValueError(
        "'inversion' is not a symmetry of WSe2_mono's point group (D3h has no "
        "inversion center) -- use 'c3_z', 'sigma_h', or 'sigma_v' instead."
    )

_S_cart = R_CART[operation]
_R_frac = np.linalg.inv(_latvec.T) @ _S_cart @ _latvec.T
R_inv = np.round(_R_frac)
if not np.allclose(_R_frac, R_inv, atol=1e-6):
    raise ValueError(
        f"Operation '{operation}' is not a symmetry of this lattice; "
        f"its fractional matrix is not integer:\n{_R_frac}")
print(f"Operation '{operation}': Cartesian S ->\n{_S_cart.astype(int)}")
print(f"applied as fractional R_inv ->\n{R_inv.astype(int)}")

# ── supercell embedding ────────────────────────────────────────────────────────
# Embed WS data into a 3x supercell so r_src values near the WS boundary
# can be interpolated without wrapping artifacts.
# actual_idx[n] + Nvec maps [-Nvec, 2*Nvec) -> [0, 3*Nvec), always in-bounds.
offset      = Nvec                          # shape (3,)
super_shape = tuple(3 * Nvec)              # (3Nx, 3Ny, 3Nz)
IJK_super   = actual_idx + offset          # (Nr, 3), integer voxel indices in supercell

# Build mask once — marks which supercell voxels carry real data.
mask_super = np.zeros(super_shape, dtype=float)
mask_super[IJK_super[:, 0], IJK_super[:, 1], IJK_super[:, 2]] = 1.0

# ── source positions in supercell voxel coordinates ───────────────────────────
# No modular wrap — r_src is evaluated in the actual WS/supercell coordinate frame.
r_src     = (positions - center) @ R_inv.T + center    # (Nr, 3), fractional
src_coord = r_src * Nvec + offset                       # (Nr, 3), voxel coords in supercell

# ── precompute interpolation denominator (same for every CNO) ─────────────────
_coords_T = src_coord.T                                 # (3, Nr) for map_coordinates
den  = map_coordinates(mask_super, _coords_T, order=1, mode="constant", cval=0.0)
safe = den > 1e-12

# ── interpolation diagnostics ──────────────────────────────────────────────────
print("Interpolation coverage (denominator):")
print(f"  min(den)          = {den.min():.4f}")
print(f"  mean(den)         = {den.mean():.4f}")
print(f"  frac(den < 0.999) = {(den < 0.999).mean():.4f}")
print(f"  frac(den < 0.5)   = {(den < 0.5).mean():.4f}")
if (den < 0.5).mean() > 0.01:
    print("WARNING: many points have low denominator — "
          "symmetry op samples outside known WS data; result may be unreliable.")


def interp_R_phi(phi_col):
    """Evaluate (R phi)[n] = phi(r_src[n]) by trilinear interpolation."""
    phi_super = np.zeros(super_shape, dtype=complex)
    phi_super[IJK_super[:, 0], IJK_super[:, 1], IJK_super[:, 2]] = phi_col

    num_real = map_coordinates(phi_super.real * mask_super, _coords_T,
                               order=1, mode="constant", cval=0.0)
    num_imag = map_coordinates(phi_super.imag * mask_super, _coords_T,
                               order=1, mode="constant", cval=0.0)

    return np.where(safe, (num_real + 1j * num_imag) / np.where(safe, den, 1.0), 0.0)


# ── flat list of all adapted indices ──────────────────────────────────────────
all_adapted_flat = [i for grp in index_groups for i in grp]
all_adapted_set  = set(all_adapted_flat)

# ── per-group D-matrix, diagonalisation, and CNO rotation ─────────────────────
np.set_printoptions(precision=6, suppress=True, linewidth=100)
print(f"Operation    : {operation}   center = {center}")

cno_out      = cno_orbs.copy()
_grp_D       = []   # D matrix per group
_grp_eta     = []   # eigenvalues per group
_grp_n       = []   # group size per group
_grp_basis   = []   # original phi basis per group
_grp_Rcache  = {}   # orbital index -> R_phi, accumulated across groups

for indices in index_groups:
    n = len(indices)
    D = np.zeros((n, n), dtype=complex)
    for b, jb in enumerate(indices):
        R_phi_b = interp_R_phi(cno_orbs[:, jb])
        _grp_Rcache[jb] = R_phi_b
        for a, ja in enumerate(indices):
            D[a, b] = cno_orbs[:, ja].conj() @ R_phi_b

    print(f"\nCNO indices  : {indices}")
    for idx in indices:
        print(f"  CNO {idx}  occ = {cno_occ[idx]:.6f}")
    eta, U = np.linalg.eig(D)
    order = np.argsort(-eta.real)
    eta, U = eta[order], U[:, order]

    phi_new = cno_orbs[:, indices] @ U
    phi_new /= np.sqrt(np.sum(np.abs(phi_new) ** 2, axis=0))
    for a, idx in enumerate(indices):
        cno_out[:, idx] = phi_new[:, a]

    _grp_D.append(D)
    _grp_eta.append(eta)
    _grp_n.append(n)
    _grp_basis.append(cno_orbs[:, indices])

out_path = output_dir / out_name
np.save(out_path, cno_out)
print(f"\nSaved -> {out_path}")
print(f"  Modified CNOs : {all_adapted_flat}   (all others unchanged)")

# ── verification (informational; written to metadata) ─────────────────────────
import io as _io
_v = _io.StringIO()
_v.write("\n=== symmetry_adapt_cnos (interpolation mode) ===\n")
_v.write(f"operation     : {operation}   center = {center}\n")
_v.write(f"index_groups  : {index_groups}\n")

_max_weight_leakage = 0.0
_unitary_error_max  = 0.0

for indices, D, eta, n, _phi_basis in zip(index_groups, _grp_D, _grp_eta, _grp_n, _grp_basis):
    _unitary_error = np.linalg.norm(D.conj().T @ D - np.eye(n))
    _unitary_error_max = max(_unitary_error_max, _unitary_error)

    _v.write(f"\n--- indices {indices} ---\n")
    _v.write(f"D:\n{D}\nD^HD:\n{D.conj().T @ D}\n")
    _v.write(f"norm(D^HD - I) : {_unitary_error:.4e}\n")
    _v.write(f"eigenvalues    : {eta}\n")
    _v.write(f"|eigenvalues|   : {np.abs(eta)}\n")

    _v.write("\nLeakage of R|phi_j> outside selected subspace:\n")
    for _jb in indices:
        _R_phi        = _grp_Rcache[_jb]
        _proj_coeffs  = _phi_basis.conj().T @ _R_phi
        _total_w      = float(np.real(_R_phi.conj() @ _R_phi))
        _inside_w     = float(np.real(_proj_coeffs.conj() @ _proj_coeffs))
        _weight_leak  = 1.0 - _inside_w / _total_w
        _residual_norm = np.sqrt(max(_weight_leak, 0.0))
        _max_weight_leakage = max(_max_weight_leakage, _weight_leak)
        _warn = "  <-- consider expanding indices" if _weight_leak > weight_leakage_warn_tol else ""
        _v.write(f"  CNO {_jb}:"
                 f"  inside_weight={_inside_w / _total_w:.6f}"
                 f"  weight_leakage={_weight_leak:.4e}"
                 f"  residual_norm=sqrt(weight_leakage)={_residual_norm:.4e}"
                 + _warn + "\n")

    _v.write("\nDensity symmetry  ||R(rho) - rho|| / ||rho||:\n")
    for _a, _idx in enumerate(indices):
        _rho   = np.abs(cno_out[:, _idx]) ** 2
        _R_rho = interp_R_phi(_rho.astype(complex)).real
        _err   = np.linalg.norm(_R_rho - _rho) / np.linalg.norm(_rho)
        _warn  = "  <-- not symmetric" if _err > density_residual_warn_tol else ""
        _v.write(f"  CNO {_idx}: {_err:.4e}" + _warn + "\n")

    _v.write("\nEigenstate residual  ||R(phi) - eta*phi|| / ||phi||:\n")
    for _a, _idx in enumerate(indices):
        _phi       = cno_out[:, _idx]
        _R_phi_out = interp_R_phi(_phi)
        _res       = np.linalg.norm(_R_phi_out - eta[_a] * _phi) / np.linalg.norm(_phi)
        _warn      = "  <-- not clean eigenstate" if _res > eigenstate_residual_warn_tol else ""
        _v.write(f"  CNO {_idx}  eta={eta[_a]:.4f}: {_res:.4e}" + _warn + "\n")

# Group-based R-invariance check for all CNOs above check_occ_cutoff.
# Degenerate groups identified by gaps in occupation (gap > 0.01 = new group).
# Singletons: |<phi|R phi>|.  Multi-CNO groups: D-matrix unitarity + weight leakage.
_checked = [i for i, o in enumerate(cno_occ) if o >= check_occ_cutoff]
_groups, _grp = [], [_checked[0]] if _checked else []
for _i in _checked[1:]:
    if abs(cno_occ[_i] - cno_occ[_i - 1]) < 0.01:
        _grp.append(_i)
    else:
        _groups.append(_grp); _grp = [_i]
if _grp:
    _groups.append(_grp)

_v.write(f"\nR-invariance by degenerate group (occ >= {check_occ_cutoff}):\n")
_max_singleton_dev   = 0.0
_min_singleton_overlap = 1.0
for _grp in _groups:
    _occ_str = ", ".join(f"{cno_occ[i]:.4f}" for i in _grp)
    _is_adapted = any(set(_grp) == set(grp) for grp in index_groups)
    _tag = "  [selected -- see above]" if _is_adapted else ""
    _v.write(f"  group {_grp}  occ=[{_occ_str}]{_tag}\n")
    if _is_adapted:
        continue
    if len(_grp) == 1:
        _i       = _grp[0]
        _phi_i   = cno_orbs[:, _i]
        _R_phi_i = interp_R_phi(_phi_i)
        _overlap = abs(_phi_i.conj() @ _R_phi_i)
        _min_singleton_overlap = min(_min_singleton_overlap, _overlap)
        _max_singleton_dev     = max(_max_singleton_dev, 1.0 - _overlap)
        _warn = ("  <-- possible symmetry/interpolation issue"
                 if _overlap < singleton_overlap_warn_tol else "")
        _v.write(f"    |<phi|R phi>| = {_overlap:.6f}" + _warn + "\n")
    else:
        _ng = len(_grp)
        _D_g = np.zeros((_ng, _ng), dtype=complex)
        _R_phi_grp = {}
        for _b2, _jb2 in enumerate(_grp):
            _Rphi2 = interp_R_phi(cno_orbs[:, _jb2])
            _R_phi_grp[_jb2] = _Rphi2
            for _a2, _ja2 in enumerate(_grp):
                _D_g[_a2, _b2] = cno_orbs[:, _ja2].conj() @ _Rphi2
        _unit_err_g  = np.linalg.norm(_D_g.conj().T @ _D_g - np.eye(_ng))
        _phi_g       = cno_orbs[:, _grp]
        _wt_leaks_g  = []
        _res_norms_g = []
        for _jb2 in _grp:
            _Rphi2      = _R_phi_grp[_jb2]
            _total_w_g  = float(np.real(_Rphi2.conj() @ _Rphi2))
            _proj_g     = _phi_g @ (_phi_g.conj().T @ _Rphi2)
            _inside_w_g = float(np.real(_proj_g.conj() @ _proj_g))
            _wt_leak_g  = 1.0 - _inside_w_g / _total_w_g
            _wt_leaks_g.append(_wt_leak_g)
            _res_norms_g.append(np.sqrt(max(_wt_leak_g, 0.0)))
        _max_wt_g  = max(_wt_leaks_g)
        _max_res_g = max(_res_norms_g)
        _warn = ("  <-- not R-invariant; expand indices"
                 if _max_wt_g > weight_leakage_warn_tol or _unit_err_g > group_unitarity_warn_tol
                 else "")
        _v.write(f"    max_weight_leakage={_max_wt_g:.4e}"
                 f"  max_residual_norm={_max_res_g:.4e}"
                 f"  norm(D^HD-I)={_unit_err_g:.4e}"
                 + _warn + "\n")

# ── final summary ──────────────────────────────────────────────────────────────
if _max_weight_leakage < 0.01 and _unitary_error_max < 0.03:
    _result = "PASS"
elif _max_weight_leakage < weight_leakage_warn_tol and _unitary_error_max < group_unitarity_warn_tol:
    _result = "PASS WITH SMALL INTERPOLATION ERROR"
else:
    _result = "WARNING"

_v.write("\nSymmetry diagnostic summary:\n")
_v.write(f"  selected-subspace maximum weight leakage   : {_max_weight_leakage:.4e}\n")
_v.write(f"  maximum singleton overlap deviation from 1 : {_max_singleton_dev:.4e}\n")
_v.write(f"  D-matrix unitarity error (max across groups): {_unitary_error_max:.4e}\n")
_v.write(f"  result: {_result}\n")

_ver_text = _v.getvalue()
print(f"Diagnostic result: {_result}"
      f"  (weight_leakage={_max_weight_leakage:.4e}"
      f"  unitarity_err={_unitary_error_max:.4e})")

if _max_weight_leakage > weight_leakage_warn_tol:
    print(f"WARNING: subspace weight leakage {_max_weight_leakage:.4e} > {weight_leakage_warn_tol}"
          f" -- try expanding one or more index groups")
if _min_singleton_overlap < singleton_overlap_warn_tol:
    print(f"WARNING: singleton |<phi|R phi>| = {_min_singleton_overlap:.6f}"
          f" < {singleton_overlap_warn_tol} -- possible symmetry/interpolation issue")

if export_cubes:
    import shutil, subprocess
    orb_file = output_dir / "cno_orbitals.npy"
    backup   = output_dir / "cno_orbitals_pre_sym_adapt.npy"
    shutil.copy2(orb_file, backup)
    shutil.copy2(out_path, orb_file)
    try:
        subprocess.run([__import__("sys").executable,
                        str(Path(__file__).resolve().parent / "export_cubes.py")],
                       check=True)
    finally:
        shutil.copy2(backup, orb_file)
    meta_path = output_dir / "cno_cube_export_metadata.txt"
    with open(meta_path, "a") as _mf:
        _mf.write(_ver_text)
    print(f"Verification appended to: {meta_path.name}")
else:
    ver_path = output_dir / "sym_adapt_verification.txt"
    with open(ver_path, "w") as _vf:
        _vf.write(_ver_text)
    print(f"Verification written to: {ver_path.name}")
