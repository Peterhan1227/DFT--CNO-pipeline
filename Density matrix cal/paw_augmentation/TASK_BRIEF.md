# Task: PAW augmentation correction for the CNO density-matrix pipeline

You are starting cold with no memory of the conversation that produced this brief. Read this whole
file before doing anything. This is exploratory research code — work in this folder only, overnight,
autonomously, using your own judgment on subproblems not explicitly specified below. Do not ask the
user questions; if you hit a fork in the road, pick the more conservative/verifiable option, document
the choice and why in RESULTS.md, and keep going.

## Hard constraint

**Do not modify any existing file.** Not `main.py`, not `config.py`, not `ws_cell.py`, not anything
already in `Density matrix cal/` outside this `paw_augmentation/` folder, and not any file under
`Data/*/output/`. All new code and outputs go in `Density matrix cal/paw_augmentation/`. This is
intentional: the existing pipeline must stay untouched and working while this is explored separately.

## Background: what problem this solves

`Density matrix cal/main.py` builds a real-space one-body density matrix from a VASP WAVECAR:

```
rho(r,r') = sum_k w_k * sum_n f_nk * psi_nk(r) * psi_nk*(r')
```

where `psi_nk(r)` is reconstructed by IFFT of the plane-wave coefficients stored in WAVECAR (read via
`VaspBandUnfolding/vaspwfc.py`, a third-party WAVECAR parser at the repo root — add it to
`PYTHONPATH`). Diagonalizing `rho` (restricted to a chosen real-space Wigner-Seitz cell) gives "CNOs"
(natural orbitals) whose occupation-number eigenvalues are physically required to lie in `[0, 1]`
for any valid closed-shell system, because `rho = sum_k w_k P_k` is a convex combination
(`sum_k w_k = 1`) of genuine orthogonal projectors `P_k` — and for any such combination, `<psi|rho|psi>
<= sum_k w_k = 1` for every unit vector `|psi>`, forcing every eigenvalue of `rho` below 1.

### The bug this was chasing

For `Data/WSe2_mono` (a 1H-WSe2 monolayer), CNO occupations were coming out above 1 (up to ~1.17).
After ruling out several other explanations (k-mesh symmetry reduction, WS-cell grid folding bugs,
G-vector aliasing on the FFT grid, ALGO/ISMEAR/NBANDS choice — none of these changed the result), the
root cause was pinned down definitively:

**`Data/WSe2_mono/POTCAR`'s tungsten potential is `W_sv`** (`TITEL = PAW_PBE W_sv 04Sep2015`,
`ZVAL=14`, `RCORE=2.5`) — a "hard", semicore potential that explicitly includes the W 5p/5d/6s shells
as valence. VASP's PAW method only guarantees `<psi_m|S|psi_n> = delta_mn` with the full PAW overlap
operator `S = 1 + sum_i |p_i>(<phi_i|phi_j> - <phi~_i|phi~_j>)<p_i|` (the augmentation correction).
`vaspwfc`/WAVECAR only ever gives you the smooth **pseudo**-wavefunction `psi~` — the plain PW
coefficients — never the augmentation piece. For soft potentials (Si, Se) that omission is invisible
(measured band-pair overlap error ~1e-8). For `W_sv` it is not: measured off-diagonal overlap between
clearly non-degenerate occupied bands at the same k-point reaches **0.14-0.18**, and this was confirmed
two independent ways (via `vaspwfc.readBandCoeff` and a from-scratch manual WAVECAR byte parser
reading the same file, bypassing the library entirely — identical result to machine precision).

The clinching, quantitative piece of evidence: reading each occupied band's *raw* (un-renormalized)
plane-wave norm at k-point 1 (`readBandCoeff(..., norm=False)`) gives:

```
band  1  E= -76.38 eV   raw ||psi~||^2 = 0.956
band  2  E= -40.61 eV   raw ||psi~||^2 = 0.449   <- W 5p semicore
band  3  E= -40.51 eV   raw ||psi~||^2 = 0.439   <- W 5p semicore
band  4  E= -40.51 eV   raw ||psi~||^2 = 0.439   <- W 5p semicore
band  5  E= -17.12 eV   raw ||psi~||^2 = 1.281
band  6  E= -15.82 eV   raw ||psi~||^2 = 1.330
band  7  E=  -8.53 eV   raw ||psi~||^2 = 1.047
band  8  E=  -4.79 eV   raw ||psi~||^2 = 0.926
band  9  E=  -4.79 eV   raw ||psi~||^2 = 0.926
band 10  E=  -4.46 eV   raw ||psi~||^2 = 1.064
band 11  E=  -3.66 eV   raw ||psi~||^2 = 0.937
band 12  E=  -3.66 eV   raw ||psi~||^2 = 0.937
band 13  E=  -2.64 eV   raw ||psi~||^2 = 0.887
```

The W 5p semicore bands (2,3,4) have only ~44% of their true norm in the smooth pseudo part — the
other 56% of that electron lives inside the PAW augmentation sphere and is simply absent from
WAVECAR. `readBandCoeff(norm=True)` (and `main.py`) then force-renormalize every band back to exactly
1, which papers over a different amount of missing/excess charge per band and is exactly what turns
into the observed non-orthogonality once you build `rho` from these renormalized states.

**The actual fix (this task) is being tried in parallel on a separate track**: switching to a
non-semicore `W` POTCAR (removing the 5p/5s semicore shell from valence) to sidestep the issue
entirely for the immediate WSe2 project. **This task is the harder, more general fix**: correctly
account for PAW augmentation in the density-matrix construction itself, because future materials may
require a hard/semicore potential where switching POTCARs isn't an option (rare-earth elements, other
heavy transition metals, etc.).

## Goal

Produce a new, self-contained script (or small set of scripts) under this `paw_augmentation/` folder
that constructs band-pair overlaps (and, if time allows, a full density matrix) **with the PAW
augmentation correction included**, verified against `Data/WSe2_mono` (the `W_sv` case) such that:

(a) Off-diagonal overlap between occupied bands at a given k-point drops from ~0.15-0.18 to something
    close to the Si baseline of ~1e-8 (a few orders of magnitude improvement is a meaningful partial
    win even if not perfect — report exactly what you achieve).
(b) If (a) is achieved and time remains: extend to a full corrected density matrix / CNO occupation
    calculation for the WSe2 `W_center` case (see `Density matrix cal/config.py` for how WS-cell
    centers are specified; do not run or modify `main.py` itself, replicate only what you need in your
    own script) and confirm resulting occupation eigenvalues stay within `[0, 1]` (allow ~1e-3
    tolerance for numerical/truncation error).

## Recommended strategy, in order

### 1. Try an existing, tested implementation first

Prefer reusing validated PAW machinery over re-deriving it from scratch — the projector/augmentation
math (spherical Bessel transforms, log-radial-grid integration, PAW sign/normalization conventions) is
easy to get subtly wrong in ways that look plausible but are not exactly right.

Check whether **`pawpyseed`** (https://github.com/kylebystrom/pawpyseed — a Python/C library built
specifically to reconstruct VASP PAW all-electron wavefunctions and proper overlaps from
WAVECAR+POTCAR) can be installed in the `irrep` conda env:

```
C:\Users\hanziruopeter\miniconda3\envs\irrep\python.exe -m pip install pawpyseed
```

This may need a C compiler on Windows and may not have been updated for newer VASP WAVECAR tags —
if installation or basic usage fails, spend at most ~30-45 minutes on it, document exactly what went
wrong, and move to option 2. Don't silently give up without a clear note of the failure mode (compile
error, API mismatch, wrong results, etc.) in RESULTS.md.

If it does work: use it to get properly-corrected overlaps or AE-reconstructed projections for the
occupied bands of `Data/WSe2_mono/WAVECAR` at several k-points, and run the verification diagnostics
below.

### 2. If no existing library works, implement the correction directly

This requires, per element in `Data/WSe2_mono/POTCAR`:

- Parsing the PAW radial dataset: the radial grid, projector functions `p~_i(r)`, AE partial waves
  `phi_i(r)`, PS partial waves `phi~_i(r)` (look for sections in the POTCAR text such as "PAW radial
  sets", "grid", "aepotential", "pseudo wavefunction", "ae wavefunction" — grep the file to find the
  exact tags used by this POTCAR's format/version).
- Computing augmentation overlap integrals per (l, i, j):
  `Q_ij = integral [phi_i(r) phi_j(r) - phi~_i(r) phi~_j(r)] r^2 dr`
- Computing projector overlaps `<p~_i | psi~_nk>` for each occupied band by transforming
  `p~_i(r) * Y_lm(theta,phi)` to reciprocal space at each `(k+G)` actually used for that WAVECAR
  k-point entry — standard formula:
  `p~_i(k+G) = 4*pi * i^l * Y_lm(direction of k+G) * integral p~_l(r) j_l(|k+G| r) r^2 dr`
  (spherical Bessel transform on the radial grid), then dotting with the actual plane-wave
  coefficients `C_n(G)` read from WAVECAR (same `gvectors()`/`readBandCoeff` calls `main.py` uses —
  reuse `vaspwfc.py`, don't re-implement WAVECAR parsing).
- Assembling the corrected overlap:
  `<psi_m|psi_n> = <psi~_m|psi~_n> + sum_{i,j} <psi~_m|p~_i> Q_ij <p~_j|psi~_n>`
  and verifying this is close to `delta_mn` for the occupied bands (see diagnostics below).
- **Only if that overlap check passes**, consider extending to a full real-space density matrix.
  Note: augmentation charge is localized in atomic spheres, not naturally represented on the FFT
  grid used elsewhere in this pipeline — attempting a literal real-space AE wavefunction on the grid
  is likely much harder than needed. A more tractable path: solve the CNO problem as a **generalized
  eigenvalue problem** `rho_corrected v = lambda * S v` in a band/plane-wave basis using the corrected
  overlap matrix `S` (built from the pieces above) instead of the current plain `rho v = lambda v` —
  this only needs the overlap matrix elements, not pointwise real-space AE reconstruction. Prefer this
  approach over literal real-space reconstruction if you get this far.

A secondary fallback if the from-scratch Bessel-transform route proves too error-prone: check whether
VASP's `LOCPROJ` output (if you have access to regenerate a WAVECAR with `LOCPROJ` set, or if a
`vasprun.xml`/`PROCAR` from a comparable run is available anywhere in the repo) could supply usable
projections instead of deriving them yourself — note this is an approximation (different convention
than the exact PAW augmentation projector) and clearly label it as such if used.

## Diagnostics to run (reuse this exact pattern so results are directly comparable to what was already measured)

```python
import numpy as np
from vaspwfc import vaspwfc   # PYTHONPATH must include VaspBandUnfolding/

wfc = vaspwfc('Data/WSe2_mono/WAVECAR', lsorbit=False)
for ik in [1, 2, 50, 150, 300]:
    occ_all = wfc._occs[0, ik-1, :]
    bands = np.where(occ_all > 1e-6)[0] + 1
    Ck = np.stack([wfc.readBandCoeff(ispin=1, ikpt=ik, iband=int(ib), norm=True) for ib in bands])
    overlap = Ck.conj() @ Ck.T   # <- this is the UNCORRECTED baseline (should show ~0.15-0.18 max offdiag)
    # your corrected overlap should replace/augment this and should show much smaller max offdiag
```

Report a clear before/after table across at least the same 5 k-points (`ik = 1, 2, 50, 150, 300`) in
`RESULTS.md`. Also re-run the identical check against `Data/Si/WAVECAR` as a regression sanity check —
Si should already show ~1e-8 and your correction must not make it worse.

## Environment notes

- Python executable (has numpy/scipy/spglib; install any new deps like `pawpyseed` here too):
  `C:\Users\hanziruopeter\miniconda3\envs\irrep\python.exe`
- `vaspwfc.py` lives in `VaspBandUnfolding/` at the repo root (sibling of `Density matrix cal/`) — add
  it to `PYTHONPATH` when running scripts, e.g.:
  `PYTHONPATH="<repo_root>/VaspBandUnfolding" PYTHONIOENCODING=utf-8 <python> your_script.py`
  (`PYTHONIOENCODING=utf-8` is needed because the Windows console codepage otherwise breaks on the
  Angstrom symbol some scripts print).
- Data available: `Density matrix cal/Data/WSe2_mono/{WAVECAR,POSCAR,POTCAR,EIGENVAL}` (the `W_sv`
  case this is all about) and `Density matrix cal/Data/Si/{WAVECAR,POSCAR}` (soft-potential regression
  reference, no POTCAR present but not needed since Si isn't the augmentation target here).
- This is a Windows/Git-Bash environment. Bash tool paths need forward slashes and `/c/Users/...`
  style; PowerShell needs its own syntax if you use it instead.

## Deliverables

1. New script(s) under `Density matrix cal/paw_augmentation/` implementing the above.
2. `Density matrix cal/paw_augmentation/RESULTS.md` documenting:
   - Which strategy was used (`pawpyseed` vs manual) and why.
   - The before/after overlap table across the 5 k-points, for both WSe2 and the Si regression check.
   - If you attempted the full corrected density matrix / CNO occupations: whether eigenvalues now
     stay within `[0, 1]`, and for which WS center / config.
   - Anything that didn't work and why, stated plainly — a well-documented partial result or negative
     result is a fine outcome for an overnight exploratory run. Do not report success unless the
     verification diagnostics above actually confirm it on real data.
3. Do not touch `main.py`, `config.py`, or anything outside this folder. Do not commit changes to git
   (leave that for the user to review).
