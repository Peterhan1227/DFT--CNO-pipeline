"""
symmetry_adapt_cnos.py — symmetry-adapt a nearly-degenerate CNO subspace.

D_ij = <phi_i | R | phi_j> = sum_k  conj(phi_i[k]) * phi_j[src_map[k]]

src_map[k] = k' is the WS grid index whose position equals R^{-1}(r_k - c) + c.
Diagonalising D gives symmetry-adapted combinations: phi_new_a = sum_j phi_j * U[j,a].
"""
import numpy as np
from pathlib import Path
from config import MATERIAL, OUTPUT_SUBDIR, WS_CENTER_COORD_TYPE, WS_CENTER

# ── settings ──────────────────────────────────────────────────────────────────
indices          = [2, 3]       # CNO subspace to symmetry-adapt
operation        = "swap_xy"    # "swap_xy" | "c3_111" | "inversion"
out_name         = "cnos_sym_adapted.npy"
export_cubes     = True         # True: swap cno_orbitals.npy -> out_name, run export_cubes.py, restore
check_occ_cutoff = 0.03         # check non-selected CNOs with occupation above this value

# Center for the symmetry operation in fractional coordinates.
# None → use WS_CENTER from config.py.
# Override only if you want a different center than the WS cell center.
# For Si with atoms at (0,0,0) and (0.75,0.75,0.75):
#   Inversion through (0.875,0.875,0.875) maps (0,0,0) <-> (0.75,0.75,0.75) ✓
#   so op_center = None is correct for all three operations.
op_center = None

# ── paths and data ─────────────────────────────────────────────────────────────
output_dir = Path(__file__).resolve().parent / "Data" / "Si" / "output" / OUTPUT_SUBDIR

cno_orbs = np.load(output_dir / "cno_orbitals.npy")      # (Nr, n_cno), complex128
cno_occ  = np.load(output_dir / "cno_occupations.npy")   # (n_cno,)
Nx, Ny, Nz = np.load(output_dir / "fft_grid_shape.npy").astype(int)
Nr = Nx * Ny * Nz

# ── WS grid positions and primitive-cell index for each point ──────────────────
# positions[k]: fractional coord actually used in the density matrix (WS or primitive)
# ws_base[k]:   primitive-cell (ix,iy,iz) for the reverse lookup table
ws_frac_file = output_dir / "ws_points_frac_cont.npy"
ws_base_file = output_dir / "ws_base_indices.npy"

positions = (np.load(ws_frac_file) if ws_frac_file.exists()
             else np.stack(np.unravel_index(np.arange(Nr), (Nx, Ny, Nz)), axis=1)
                  / np.array([Nx, Ny, Nz], dtype=float))

ws_base = (np.load(ws_base_file).astype(int) if ws_base_file.exists()
           else np.stack(np.unravel_index(np.arange(Nr), (Nx, Ny, Nz)), axis=1))

# ── symmetry center and R^{-1} matrix ─────────────────────────────────────────
if op_center is not None:
    center = np.array(op_center, dtype=float)
elif WS_CENTER_COORD_TYPE == "fractional":
    center = np.array(WS_CENTER, dtype=float)
else:
    center = np.load(output_dir / "ws_center_frac_wrapped.npy").astype(float)

# R^{-1} acts on fractional coordinates: r_src = R^{-1}(r - c) + c
# inversion:  r -> 2c - r
# c3_111:     forward (x,y,z)->(y,z,x),  inverse (x,y,z)->(z,x,y)
# swap_xy:    (x,y,z)->(y,x,z), self-inverse
R_INV = {
    "inversion": -np.eye(3, dtype=float),
    "c3_111":    np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float),
    "swap_xy":   np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float),
}
R_inv = R_INV[operation]

# ── build src_map: src_map[k] = k' such that (R phi)[k] = phi[k'] ─────────────
r_src   = (positions - center) @ R_inv.T + center            # (Nr, 3)
idx_src = (np.round(r_src % 1.0 * np.array([Nx, Ny, Nz]))
           .astype(int) % np.array([Nx, Ny, Nz]))             # (Nr, 3)

rev = np.full((Nx, Ny, Nz), -1, dtype=int)
rev[ws_base[:, 0], ws_base[:, 1], ws_base[:, 2]] = np.arange(Nr)
src_map = rev[idx_src[:, 0], idx_src[:, 1], idx_src[:, 2]]

if np.any(src_map < 0):
    raise ValueError("Some source grid points have no WS entry — "
                     "operation is not a symmetry of this grid.")

# ── symmetry matrix D_ij = <phi_i | R | phi_j> ────────────────────────────────
n = len(indices)
D = np.zeros((n, n), dtype=complex)
for b, jb in enumerate(indices):
    R_phi_b = cno_orbs[src_map, jb]            # (R phi_b)[k] = phi_b[src_map[k]]
    for a, ja in enumerate(indices):
        D[a, b] = cno_orbs[:, ja].conj() @ R_phi_b

# ── diagnostics ────────────────────────────────────────────────────────────────
np.set_printoptions(precision=6, suppress=True, linewidth=100)
print(f"CNO indices  : {indices}")
print(f"Operation    : {operation}   center = {center}")
for idx in indices:
    print(f"  CNO {idx}  occ = {cno_occ[idx]:.6f}")
eta, U = np.linalg.eig(D)
order = np.argsort(-eta.real)
eta, U = eta[order], U[:, order]

# ── rotate CNOs: phi_new_a = sum_j phi_j * U[j, a] ────────────────────────────
phi_new = cno_orbs[:, indices] @ U
phi_new /= np.sqrt(np.sum(np.abs(phi_new) ** 2, axis=0))

cno_out = cno_orbs.copy()
for a, idx in enumerate(indices):
    cno_out[:, idx] = phi_new[:, a]

out_path = output_dir / out_name
np.save(out_path, cno_out)
print(f"\nSaved -> {out_path}")
print(f"  Modified CNOs : {indices}   (all others unchanged)")

# ── verification (critical checks raise; informational written to metadata) ────
n_unique = len(np.unique(src_map))
if n_unique != Nr:
    raise ValueError(f"src_map not a bijection: {n_unique}/{Nr} unique — operation not 1-to-1")
if operation in ("inversion", "swap_xy"):
    n_fail = int(np.sum(src_map[src_map] != np.arange(Nr)))
    if n_fail > 0:
        raise ValueError(f"R² ≠ identity on {n_fail} grid points — bad symmetry center?")

import io as _io
_v = _io.StringIO()
_v.write("\n=== symmetry_adapt_cnos ===\n")
_v.write(f"operation     : {operation}   center = {center}\n")
_v.write(f"indices       : {indices}\n")
_v.write(f"\nD:\n{D}\nD^HD:\n{D.conj().T @ D}\n")
_v.write(f"norm(D^HD - I) : {np.linalg.norm(D.conj().T @ D - np.eye(n)):.4e}\n")
_v.write(f"eigenvalues   : {eta}\n")
_v.write(f"|eigenvalues|  : {np.abs(eta)}\n")

_v.write("\nLeakage of R|phi_j> outside subspace:\n")
_phi_basis = cno_orbs[:, indices]
_max_leak = 0.0
for _, _jb in enumerate(indices):
    _R_phi = cno_orbs[src_map, _jb]
    _proj  = _phi_basis @ (_phi_basis.conj().T @ _R_phi)
    _leak  = np.linalg.norm(_R_phi - _proj) / np.linalg.norm(_R_phi)
    _max_leak = max(_max_leak, _leak)
    _v.write(f"  CNO {_jb}: {_leak:.4e}"
             + ("  <-- expand indices\n" if _leak > 0.05 else "\n"))

_v.write("\nDensity symmetry  ||rho[src_map] - rho|| / ||rho||:\n")
for _a, _idx in enumerate(indices):
    _rho = np.abs(cno_out[:, _idx]) ** 2
    _err = np.linalg.norm(_rho[src_map] - _rho) / np.linalg.norm(_rho)
    _v.write(f"  CNO {_idx}: {_err:.4e}" + ("  <-- not symmetric\n" if _err > 0.02 else "\n"))

_v.write("\nEigenstate residual  ||phi[src_map] - eta*phi|| / ||phi||:\n")
for _a, _idx in enumerate(indices):
    _phi = cno_out[:, _idx]
    _res = np.linalg.norm(_phi[src_map] - eta[_a] * _phi) / np.linalg.norm(_phi)
    _v.write(f"  CNO {_idx}  eta={eta[_a]:.4f}: {_res:.4e}"
             + ("  <-- not clean eigenstate\n" if _res > 0.05 else "\n"))

# Group-based R-invariance check for all CNOs above check_occ_cutoff.
# Degenerate groups are identified by gaps in occupation (gap > 0.01 = new group).
# Singletons: eigenstate residual.  Multi-CNO groups: D-matrix unitarity + leakage.
# A large value for a singleton means [rho, R] != 0 (likely ISYM > 0 in VASP).
# A large value for a degenerate group means you need to expand `indices`.
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
_max_nonsel_res = 0.0
for _grp in _groups:
    _occ_str = ", ".join(f"{cno_occ[i]:.4f}" for i in _grp)
    _tag = "  [selected — see above]" if set(_grp) == set(indices) else ""
    _v.write(f"  group {_grp}  occ=[{_occ_str}]{_tag}\n")
    if set(_grp) == set(indices):
        continue
    if len(_grp) == 1:
        # For a non-degenerate CNO: R must map its 1D eigenspace to itself,
        # so R|phi⟩ = eta|phi⟩ exactly iff [rho,R]=0.
        # Since src_map is a bijection, ||R phi|| = ||phi|| = 1, so
        # |<phi|R phi>| = 1 iff R phi is parallel to phi (no leakage).
        _i = _grp[0]
        _phi_i = cno_orbs[:, _i]
        _overlap = abs(_phi_i.conj() @ _phi_i[src_map])   # |<phi|R phi>|, should be 1
        _max_nonsel_res = max(_max_nonsel_res, 1.0 - _overlap)
        _v.write(f"    |<phi|R phi>| = {_overlap:.6f}"
                 + ("  <-- [rho,R]!=0, rerun VASP with ISYM=0\n" if _overlap < 0.95 else "\n"))
    else:
        _ng = len(_grp)
        _D_g = np.zeros((_ng, _ng), dtype=complex)
        for _b2, _jb2 in enumerate(_grp):
            _Rphi2 = cno_orbs[src_map, _jb2]
            for _a2, _ja2 in enumerate(_grp):
                _D_g[_a2, _b2] = cno_orbs[:, _ja2].conj() @ _Rphi2
        _unit_err = np.linalg.norm(_D_g.conj().T @ _D_g - np.eye(_ng))
        _phi_g = cno_orbs[:, _grp]
        _lk_g = max(
            np.linalg.norm(cno_orbs[src_map, _jb2] -
                           _phi_g @ (_phi_g.conj().T @ cno_orbs[src_map, _jb2])) /
            np.linalg.norm(cno_orbs[src_map, _jb2])
            for _jb2 in _grp
        )
        _max_nonsel_res = max(_max_nonsel_res, _unit_err, _lk_g)
        _v.write(f"    norm(D^HD-I)={_unit_err:.4e}  leakage={_lk_g:.4e}"
                 + ("  <-- not R-invariant; expand indices\n" if _unit_err > 0.05 or _lk_g > 0.05
                    else "\n"))

_ver_text = _v.getvalue()

if _max_leak > 0.05:
    print(f"WARNING: leakage {_max_leak:.1%} — try expanding indices "
          f"(e.g. {list(range(indices[0], indices[-1] + 3))})")
if _max_nonsel_res > 0.05:
    print(f"WARNING: non-selected singleton CNOs have |<phi|R phi>| < 0.95 "
          f"— R may not be a symmetry of this density matrix (rerun VASP with ISYM=0?)")

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
