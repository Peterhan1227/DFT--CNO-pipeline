"""
diagnostics/_common.py -- shared read-only parsing/FFT/reporting helpers for
the "Density matrix cal" diagnostics package.

Everything here is read-only with respect to production data and production
outputs: WAVECAR/POSCAR/POTCAR/EIGENVAL/KPOINTS are only ever opened for
reading, and nothing under Data/<material>/output/<OUTPUT_SUBDIR>/ is ever
written. Each diagnostic script writes its own JSON report under
diagnostics/output/.

Import convention matches the rest of this repo (paw_augmentation/, Backup
script/, etc.): flat sys.path insertion, not package-relative imports, so
every test_*.py here can also be run directly with `python test_foo.py`.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

DIAG_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = DIAG_DIR.parent                       # "Density matrix cal"
REPO_ROOT = PIPELINE_DIR.parent
OUTPUT_DIR = DIAG_DIR / "output"

for _p in (
    str(PIPELINE_DIR),
    str(PIPELINE_DIR / "helper functions"),
    str(PIPELINE_DIR / "paw_augmentation"),
    str(REPO_ROOT / "VaspBandUnfolding"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vaspwfc import vaspwfc                  # noqa: E402
from ws_cell import (                        # noqa: E402
    read_poscar_structure, parse_ws_center, build_ws_grid_map,
)

DATA_ROOT = PIPELINE_DIR / "Data"

# ── shared tolerances (documented once, reused by every report) ────────────
TOL = dict(
    lattice_ang=1e-4,          # WAVECAR vs POSCAR real-space cell, Angstrom
    kfrac=1e-6,                # EIGENVAL vs WAVECAR fractional k-coordinates
    energy_ev=1e-3,            # EIGENVAL vs WAVECAR band energies, eV
    occupation=1e-6,           # EIGENVAL vs WAVECAR occupations
    gram_corrected=1e-4,       # max|offdiag| / |diag-1| for the PAW-corrected Gram
    fft_roundtrip=1e-9,        # forward/inverse FFT round-trip residual
    direct_fourier=1e-8,       # direct-Fourier vs ifftn agreement
    bloch_covariance=1e-8,     # psi(r+R) = e^{2pi i k.R} psi(r)
    ws_permutation=0,          # WS grid map must be an EXACT bijection
    bvk_identity=1e-10,        # explicit P_A P P_A vs weighted one-cell formula
    bvk_bound=1e-9,            # [0,1] / PSD slack for the BvK toy
)


# ── dataset discovery ───────────────────────────────────────────────────────

def available_materials():
    """Every subdirectory of Data/ that has a WAVECAR."""
    if not DATA_ROOT.is_dir():
        return []
    return sorted(p.name for p in DATA_ROOT.iterdir()
                  if p.is_dir() and (p / "WAVECAR").exists())


def detect_lsorbit(wavecar_path):
    """Probe whether a WAVECAR needs lsorbit=True.

    Tries lsorbit=False first; vaspwfc.gvectors() itself raises a ValueError
    with 'NONCOLLINEAR' in the message when that guess is wrong (see
    VaspBandUnfolding/vaspwfc.py:392-401), which we translate into a second,
    correctly-configured open rather than guessing from file naming.

    Returns (lsorbit: bool, wfc: vaspwfc) using whichever setting worked.
    """
    wfc = vaspwfc(str(wavecar_path), lsorbit=False)
    try:
        wfc.gvectors(1)
        return False, wfc
    except ValueError as e:
        if "NONCOLLINEAR" in str(e).upper():
            return True, vaspwfc(str(wavecar_path), lsorbit=True)
        raise


class GammaOnlyUnsupported(RuntimeError):
    """Raised when a WAVECAR is Gamma-only but was not opened with the
    lgamma=True half-grid reconstruction this diagnostics package expects
    callers to provide explicitly."""


def safe_gvectors(wfc, ik):
    """wfc.gvectors(ik), turning vaspwfc's own Gamma-only ValueError into a
    clear, typed, unsupported-dataset signal instead of a generic traceback
    or (worse) silently wrong G-vectors."""
    try:
        return wfc.gvectors(ik)
    except ValueError as e:
        msg = str(e)
        if "GAMMA-ONLY" in msg.upper():
            raise GammaOnlyUnsupported(
                "Detected a Gamma-only WAVECAR (vaspwfc raised: "
                f"{msg.strip()}). This diagnostics package does not "
                "implement the lgamma=True half-grid reconstruction path; "
                "re-run with a vaspwfc opened as lgamma=True (and the "
                "correct gamma_half) to support it, or treat this dataset "
                "as unsupported."
            ) from e
        raise


# ── file identity / hashing ─────────────────────────────────────────────────

DATASET_FILES = ["WAVECAR", "POTCAR", "POSCAR", "EIGENVAL", "KPOINTS"]


def sha256_and_size(path, chunk_size=1 << 20):
    """(hex_digest, size_bytes) for a file, or (None, None) if it's absent.

    Streams the file through the hash in chunks -- the bytes are never held
    in full or copied into any report; only the digest and size are kept.
    """
    path = Path(path)
    if not path.exists():
        return None, None
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def dataset_hashes(data_dir):
    """SHA256 + size for each standard dataset file. Missing files are
    reported as present=False rather than raising."""
    data_dir = Path(data_dir)
    out = {}
    for name in DATASET_FILES:
        digest, size = sha256_and_size(data_dir / name)
        out[name] = dict(present=digest is not None, sha256=digest, size_bytes=size)
    return out


# ── EIGENVAL parsing (superset of main.py's _read_eigenval) ────────────────

def read_eigenval_full(path):
    """Parse a BZ-mesh EIGENVAL, including occupations and the ISPIN header
    field that main.py's own _read_eigenval (main.py:18-48) silently ignores.

    Returns a dict:
      nkpts, nbands, ispin_header,
      kfrac (nkpts,3), kweights_raw (nkpts,), kweights (nkpts,, normalized),
      energies (ispin_header, nkpts, nbands),
      occupations (ispin_header, nkpts, nbands)
    """
    with open(path) as fh:
        lines = fh.readlines()
    ispin_header = int(lines[0].split()[3])
    nkpts = int(lines[5].split()[1])
    nbands = int(lines[5].split()[2])

    kfrac = np.zeros((nkpts, 3))
    kweights = np.zeros(nkpts)
    energies = np.zeros((ispin_header, nkpts, nbands))
    occupations = np.zeros((ispin_header, nkpts, nbands))

    idx = 6
    for ik in range(nkpts):
        while not lines[idx].split():
            idx += 1
        kline = lines[idx].split()
        kfrac[ik] = [float(x) for x in kline[:3]]
        kweights[ik] = float(kline[3])
        idx += 1
        for ib in range(nbands):
            cols = lines[idx].split()
            if ispin_header == 1:
                energies[0, ik, ib] = float(cols[1])
                occupations[0, ik, ib] = float(cols[2])
            else:
                energies[0, ik, ib] = float(cols[1])
                energies[1, ik, ib] = float(cols[2])
                occupations[0, ik, ib] = float(cols[3])
                occupations[1, ik, ib] = float(cols[4])
            idx += 1

    wsum = kweights.sum()
    kweights_norm = kweights / wsum if wsum else kweights.copy()
    return dict(
        nkpts=nkpts, nbands=nbands, ispin_header=ispin_header,
        kfrac=kfrac, kweights_raw=kweights, kweights=kweights_norm,
        energies=energies, occupations=occupations,
    )


def read_kpoints_header(path):
    """Best-effort KPOINTS summary: explicit-list vs mesh-generator mode and,
    for explicit lists, the k-point count (for cross-checking against
    WAVECAR/EIGENVAL nkpts). Returns None fields when the mode can't be
    determined cheaply (mesh-generator KPOINTS specify a generator, not a
    literal count, so no count check is attempted there)."""
    path = Path(path)
    if not path.exists():
        return dict(present=False)
    lines = path.read_text().splitlines()
    if len(lines) < 3:
        return dict(present=True, mode="unknown", nkpts_listed=None)
    try:
        nk_field = int(lines[1].split()[0])
    except (ValueError, IndexError):
        return dict(present=True, mode="unknown", nkpts_listed=None)
    mode_line = lines[2].strip().lower() if len(lines) > 2 else ""
    if nk_field == 0:
        return dict(present=True, mode=f"mesh-generator ({mode_line or '?'})",
                     nkpts_listed=None)
    return dict(present=True, mode="explicit-list", nkpts_listed=nk_field)


# ── lattice comparison ──────────────────────────────────────────────────────

def compare_lattices(latvec_wavecar, latvec_poscar, tol=None):
    tol = TOL["lattice_ang"] if tol is None else tol
    diff = np.asarray(latvec_wavecar) - np.asarray(latvec_poscar)
    max_abs = float(np.max(np.abs(diff)))
    return dict(max_abs_diff_ang=max_abs, tol_ang=tol, passed=bool(max_abs < tol))


# ── occupation handling (mirrors main.py's selection/halving convention) ───

def occupied_bands_split(wfc, ik, ispin, occ_tol=1e-6, frac_tol=1e-3):
    """Reproduce main.py's occupied-band selection and spin-degeneracy
    halving (main.py:244-249), then split into 'binary' (occ within frac_tol
    of 1) and 'fractional' (0 << occ << 1) groups.

    The fractional group is exactly where a fractional `occ` weight actually
    matters in main.py's rho accumulation -- and therefore exactly where the
    SOC branch's omission of `occ` (main.py:266-267, `psi_up.T @ psi_up.conj()`
    with no occupation weighting at all) would produce a quantifiably wrong
    rho, since every occupied band there is implicitly weighted as if fully
    occupied regardless of its true (possibly fractional) occupation.
    """
    occ_all = wfc._occs[ispin - 1, ik - 1, :]
    bands = np.where(occ_all > occ_tol)[0] + 1
    occ = occ_all[bands - 1].copy()
    halved = False
    if len(occ) and np.max(occ) > 1.5:
        occ = occ / 2.0
        halved = True
    binary_mask = occ >= (1.0 - frac_tol)
    fractional_mask = ~binary_mask
    return dict(bands=bands, occ=occ, halved=halved,
                binary_mask=binary_mask, fractional_mask=fractional_mask)


def weighted_occupation_count(wfc, ispin, kweights, occ_tol=1e-6):
    """sum_k w_k * sum_n occ_nk, post main.py's halving convention -- the
    total electron count main.py's rho trace is supposed to reproduce."""
    total = 0.0
    per_k = np.zeros(wfc._nkpts)
    any_halved = False
    for ik in range(1, wfc._nkpts + 1):
        sel = occupied_bands_split(wfc, ik, ispin, occ_tol=occ_tol)
        per_k[ik - 1] = float(sel["occ"].sum())
        any_halved = any_halved or sel["halved"]
        total += kweights[ik - 1] * per_k[ik - 1]
    return float(total), per_k, any_halved


def pick_representative_kpoints(wfc, ispin, occ_tol=1e-6, frac_tol=1e-3):
    """Deterministic, small set of k-points to spot-check: first, middle,
    last (structural coverage of the mesh), plus whichever k-point has the
    most fractionally-occupied bands (if any -- metals/semimetals under
    smearing), so fractional-occupation physics is always exercised when the
    dataset actually has any, not just the binary-occupation common case."""
    nk = wfc._nkpts
    iks = {1, max(1, nk // 2), nk}

    best_ik, best_n_frac = None, 0
    for ik in range(1, nk + 1):
        sel = occupied_bands_split(wfc, ik, ispin, occ_tol=occ_tol, frac_tol=frac_tol)
        n_frac = int(sel["fractional_mask"].sum())
        if n_frac > best_n_frac:
            best_n_frac, best_ik = n_frac, ik
    if best_ik is not None:
        iks.add(best_ik)

    return sorted(iks), best_n_frac


def spin_convention_report(wfc, eigenval, ispin_config):
    """Self-documenting summary of the occupation-number convention actually
    observed in this dataset, rather than assuming VASP's textbook f<=2
    convention. main.py only halves when max(occ) > 1.5 (main.py:248); if a
    dataset never trips that branch, main.py is implicitly treating `occ` as
    already-per-spin-orbital."""
    occ_max = float(wfc._occs.max())
    return dict(
        nspin_wavecar=int(wfc._nspin),
        ispin_eigenval_header=eigenval.get("ispin_header") if eigenval else None,
        ispin_config=ispin_config,
        lsorbit=bool(wfc._lsoc),
        occ_max_observed=occ_max,
        halving_would_trigger=bool(occ_max > 1.5),
        interpretation=(
            "occ_max <= 1.5: occupations are already per-spin-orbital; "
            "main.py's '>1.5 -> halve' branch (main.py:248-249) is a no-op "
            "for this dataset."
            if occ_max <= 1.5 else
            "occ_max > 1.5: main.py halves occupations, assuming "
            "spin-degenerate f<=2 storage (valid only for spin-restricted, "
            "non-SOC calculations)."
        ),
    )


# ── FFT / Gram-matrix helpers (band-pair, never Nr x Nr) ───────────────────

def ifft_bands(Ck, gvec, ngrid):
    """(nb, nG) plane-wave coeffs + integer G-vectors -> (nb, Nr) cell-
    periodic u_nk on the primitive FFT grid, main.py's exact convention:
    u = ifftn(cg, axes=(1,2,3)) * sqrt(Nr) (main.py:264-272), reshaped flat.
    """
    Nx, Ny, Nz = ngrid
    Nr = Nx * Ny * Nz
    Ck = np.asarray(Ck)
    nb = Ck.shape[0]
    gx, gy, gz = gvec[:, 0] % Nx, gvec[:, 1] % Ny, gvec[:, 2] % Nz
    cg = np.zeros((nb, Nx, Ny, Nz), dtype=np.complex128)
    cg[:, gx, gy, gz] = Ck
    u = np.fft.ifftn(cg, axes=(1, 2, 3)) * np.sqrt(Nr)
    return u.reshape(nb, Nr)


def duplicate_grid_indices(gvec, ngrid):
    """Count duplicate (gx%Nx, gy%Ny, gz%Nz) triples. A duplicate means two
    distinct G-vectors alias to the same FFT grid point -- cg[:, gx,gy,gz]=Ck
    would silently overwrite one with the other, corrupting the IFFT."""
    Nx, Ny, Nz = ngrid
    gx, gy, gz = gvec[:, 0] % Nx, gvec[:, 1] % Ny, gvec[:, 2] % Nz
    flat = (gx.astype(np.int64) * Ny + gy) * Nz + gz
    n_unique = len(np.unique(flat))
    return dict(nG=int(len(flat)), n_unique=int(n_unique),
                n_duplicates=int(len(flat) - n_unique))


def gram_from_coeffs(Ck):
    """<psi_m|psi_n> = sum_G Cm(G)^* Cn(G) for a (nb, nG) coefficient array
    -- the band-pair Gram matrix built directly in coefficient space, never
    as an Nr x Nr real-space projector."""
    Ck = np.asarray(Ck)
    return Ck.conj() @ Ck.T


def gram_from_grid(u_or_psi):
    """Same Gram matrix, built from (nb, Nr) real-space samples instead of
    (nb, nG) coefficients: sum_r conj(f_m(r)) f_n(r)."""
    A = np.asarray(u_or_psi)
    return A.conj() @ A.T


def gram_stats(G, tol=1e-6):
    """Diagonal / off-diagonal / eigenvalue / operator-norm diagnostics for
    a band-pair Gram matrix G (nb x nb), expected close to the identity for
    mutually orthonormal bands."""
    nb = G.shape[0]
    if nb == 0:
        return dict(nb=0, passed=True)
    diag = np.diag(G).real
    herm_err = float(np.max(np.abs(G - G.conj().T)))
    max_diag_err = float(np.max(np.abs(diag - 1.0)))
    if nb > 1:
        off_mask = ~np.eye(nb, dtype=bool)
        max_offdiag = float(np.max(np.abs(G[off_mask])))
    else:
        max_offdiag = 0.0
    eigvals = np.linalg.eigvalsh(0.5 * (G + G.conj().T))
    norm_G_minus_I = float(np.linalg.norm(G - np.eye(nb), ord=2))
    return dict(
        nb=int(nb),
        max_diag_err=max_diag_err,
        max_offdiag=max_offdiag,
        herm_err=herm_err,
        eig_min=float(eigvals.min()),
        eig_max=float(eigvals.max()),
        norm_G_minus_I=norm_G_minus_I,
        tol=tol,
        passed=bool(max_diag_err < tol and max_offdiag < tol),
    )


# ── PAW augmentation correction wrappers ────────────────────────────────────

def paw_correction_available(data_dir):
    return (Path(data_dir) / "POTCAR").exists()


def build_nonsoc_paw_gram(data_dir, ik, ispin, bands):
    """Non-SOC band-pair Gram with the existing reciprocal-space PAW
    correction, via paw_augmentation/paw_overlap.py's PawOverlapCorrector
    (reused, not re-derived). Returns (S_ps, S_corrected)."""
    from paw_overlap import PawOverlapCorrector  # noqa: E402 (paw_augmentation on sys.path)
    corr = PawOverlapCorrector(
        str(Path(data_dir) / "WAVECAR"),
        str(Path(data_dir) / "POSCAR"),
        str(Path(data_dir) / "POTCAR"),
    )
    b, S_ps, S_corr = corr.overlaps(ik, ispin=ispin, bands=bands)
    assert np.array_equal(b, np.asarray(bands)), "band index mismatch"
    return S_ps, S_corr


def build_soc_paw_gram(data_dir, ik, bands, ispin=1):
    """SOC-spinor analog of PawOverlapCorrector.overlaps(): sums the PAW
    augmentation correction over the up and down spinor channels, matching
    main.py's SOC accumulation rho += wk*(psi_up^H psi_up + psi_dn^H psi_dn)
    (main.py:266-267). The reciprocal-space projector (paw.nonlq) is built
    from a spatial-only Qij (spin-independent in the standard scalar-
    relativistic PAW operator used here), so each spinor channel is
    projected with the SAME projector object and the two channels' overlap
    contributions are added -- there is no cross (up-down) augmentation term.

    Requires a POTCAR to be present; raises FileNotFoundError otherwise
    (callers should check paw_correction_available() first and report
    'skipped: no POTCAR' rather than calling this blind).
    """
    from paw import nonlq                              # noqa: E402
    from paw_overlap import load_pawpp, build_qij_block  # noqa: E402
    from ase.io import read as ase_read                  # noqa: E402

    data_dir = Path(data_dir)
    potcar = data_dir / "POTCAR"
    if not potcar.exists():
        raise FileNotFoundError(f"No POTCAR at {potcar}; cannot build SOC PAW correction.")

    wfc = vaspwfc(str(data_dir / "WAVECAR"), lsorbit=True)
    atoms = ase_read(str(data_dir / "POSCAR"))
    pawpp = load_pawpp(potcar)

    kvec = wfc._kvecs[ik - 1]
    proj = nonlq(atoms, wfc._encut, pawpp, k=kvec, lgam=False, gamma_half='x')
    qij_block = build_qij_block(proj.pawpp, proj.element_idx)

    gvec = wfc.gvectors(ik)
    nG = gvec.shape[0]

    Cg_up, Cg_dn, beta_up, beta_dn = [], [], [], []
    for ib in bands:
        Cg = wfc.readBandCoeff(ispin=ispin, ikpt=ik, iband=int(ib), norm=False)
        up, dn = Cg[:nG], Cg[nG:]
        Cg_up.append(up)
        Cg_dn.append(dn)
        beta_up.append(proj.proj(up))
        beta_dn.append(proj.proj(dn))

    Ck_up, Ck_dn = np.stack(Cg_up), np.stack(Cg_dn)
    Bk_up, Bk_dn = np.stack(beta_up), np.stack(beta_dn)

    S_ps = Ck_up.conj() @ Ck_up.T + Ck_dn.conj() @ Ck_dn.T
    S_aug = Bk_up.conj() @ qij_block @ Bk_up.T + Bk_dn.conj() @ qij_block @ Bk_dn.T
    return S_ps, S_ps + S_aug


# ── JSON report I/O ─────────────────────────────────────────────────────────

def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)!r}")


def write_report(name, report, material=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{name}__{material}.json" if material else f"{name}.json"
    path = OUTPUT_DIR / fname
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    return path


def overall_status(*passed_flags):
    flags = [bool(p) for p in passed_flags if p is not None]
    return "PASS" if flags and all(flags) else ("FAIL" if flags else "SKIPPED")
