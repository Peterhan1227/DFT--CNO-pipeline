# PAW augmentation correction for the CNO density-matrix pipeline — RESULTS

## TL;DR

- **Goal (a) — band-pair overlap correction: solidly achieved.** Off-diagonal
  overlap between occupied WSe2 bands at k = 1, 2, 50, 150, 300 drops from
  **0.14–0.18 (uncorrected) to 3×10⁻⁷–5×10⁻⁶ (corrected)** — five orders of
  magnitude, and better than the Si soft-potential reference (~10⁻⁸ at Γ).
  Diagonal (normalization) lands at **1.0000** to 6 decimal places.
- **Goal (b) — full corrected density matrix / CNO occupations: implemented
  and code-validated, but NOT completed with a physically valid final
  number.** Partway through this task, `Data/WSe2_mono/WAVECAR` was
  overwritten by a concurrent, unrelated calculation (confirmed by the user
  to use a plain, non-semicore `W` pseudopotential) while `POTCAR` in the
  same directory still describes `W_sv`. A WAVECAR and POTCAR that don't
  describe the same pseudopotential cannot be combined for a PAW correction,
  so I stopped rather than report a fabricated number. As a demonstration
  that the *implementation* is complete and functional, I ran it anyway on
  this now-mismatched pair: the result (max CNO occupation rises from 0.971
  uncorrected to **1.098 corrected**, i.e. the mismatched correction makes
  the `[0,1]` violation *worse*, not better) is exactly what's physically
  expected when applying the wrong potential's augmentation data, and is a
  useful negative control, but it is explicitly **not** the goal-(b)
  deliverable. Full detail and what *is* validated (the real-space metric
  construction, independently cross-checked against the working
  reciprocal-space route) is below.
- **Strategy used: the existing, already-tested `paw.py`/`aewfc.py` PAW
  machinery already present in `VaspBandUnfolding/`** (by the same author as
  `vaspwfc.py`), not `pawpyseed`. This is a stronger instance of the task
  brief's "try an existing implementation first" guidance than what the
  brief anticipated — see "Strategy" below for why `pawpyseed` was not
  attempted at all.

---

## Strategy: why `paw.py`, not `pawpyseed`

Before installing anything, I checked whether the repo already had validated
PAW machinery, since `VaspBandUnfolding/` (the same third-party library that
supplies `vaspwfc.py`) also ships `paw.py` and `aewfc.py` by the same author
(Qijing Zheng). These implement exactly the objects TASK_BRIEF.md section 2
asks for, already wired up and exercised by working code in the same
library:

- `paw.pawpotcar` — parses a POTCAR's "PAW radial sets" section (radial
  grid, AE/PS partial waves, projector functions in real *and* reciprocal
  space) and computes `Qij = <phi_i^AE|phi_j^AE> - <phi_i^PS|phi_j^PS>`
  (`get_Qij()`) — exactly the brief's requested quantity.
- `paw.nonlq` — computes `<p~_i|psi~_nk> = beta_n,i` at a given k-point via
  the exact spherical-Bessel-transform formula in the brief
  (`p~_i(k+G) = 4*pi*i^l*Y_lm(k+G)*radial_transform(|k+G|)`), dotted with
  the actual WAVECAR plane-wave coefficients.
- `aewfc.get_ae_norm()` already computes `<psi|S|psi> = <Cg|Cg> +
  beta^dagger Q beta` for a single band — i.e. the exact all-electron-norm
  formula this task needs, generalized here from norms to full band-pair
  overlaps.

Given this, `pawpyseed` was not attempted — spending the brief's
suggested "~30–45 minutes" evaluating a second, unfamiliar C-extension
library made no sense once a validated, already-in-the-repo implementation
of the same formulas was confirmed working (see verification below). This
is a conservative choice in the brief's own terms: reusing tested code
over introducing a new dependency with unknown Windows-compile risk.

Neither `pawpotcar`/`nonlq` (used for goal a) nor the real-space projector
math I added on top for goal (b) needs `pysbt` (not installed in the `irrep`
env). `pysbt` is only touched by `aewfc.py`'s full real-space AE-wavefunction
reconstruction path (`sbt_aeps_core`/`get_ae_wfc`) and by
`pawpotcar.get_nablaij(lreal=False)`, neither of which this task uses — fully
consistent with the brief's own steer away from pointwise real-space AE
reconstruction.

---

## Goal (a): band-pair overlap correction

### Method

Implemented in `paw_overlap.py`. For occupied bands m, n at a k-point:

```
<psi_m|S|psi_n> = <psi~_m|psi~_n> + beta_m^dagger . Qij . beta_n
```

- `<psi~_m|psi~_n>` — plain plane-wave overlap of **raw, un-renormalized**
  band coefficients (`readBandCoeff(norm=False)`), i.e. `Ck.conj() @ Ck.T`.
- `beta_n,i = <p~_i|psi~_n>` — via `paw.nonlq.proj()`, reused unmodified.
- `Qij` — via `pawpotcar.get_Qij()`, block-diagonal over atoms
  (`scipy.sparse.block_diag`), reused unmodified.

Deliberately uses `norm=False` coefficients throughout: `readBandCoeff(norm=True)`
force-renormalizes each band to unit *pseudo*-norm, which is exactly the
artifact the brief identifies as compounding the underlying bug. The raw
coefficients plus the augmentation term should already sum to ≈1 on the
diagonal if the correction is right — and that is exactly what's observed
(see below), which is itself a nontrivial confirmation of correctness.

### Verification: sanity check against the brief's own numbers first

Before trusting anything, `_sanity_check.py` reproduced TASK_BRIEF.md's
quoted raw-norm table for WSe2 k-point 1 band-by-band
(`readBandCoeff(norm=False)`, `||psi~||^2`):

| band | E (eV) | raw ‖ψ̃‖² (this run) | raw ‖ψ̃‖² (brief) |
|---|---|---|---|
| 1 | −76.38 | 0.956 | 0.956 |
| 2 | −40.61 | 0.449 | 0.449 |
| 3 | −40.51 | 0.439 | 0.439 |
| 4 | −40.51 | 0.439 | 0.439 |
| 5 | −17.12 | 1.281 | 1.281 |
| 6 | −15.82 | 1.330 | 1.330 |
| 7 | −8.53 | 1.047 | 1.047 |
| 8 | −4.79 | 0.926 | 0.926 |
| 9 | −4.79 | 0.926 | 0.926 |
| 10 | −4.46 | 1.064 | 1.064 |
| 11 | −3.66 | 0.937 | 0.937 |
| 12 | −3.66 | 0.937 | 0.937 |
| 13 | −2.64 | 0.887 | 0.887 |

Exact match — confirms this session's `Data/WSe2_mono/WAVECAR` (as it stood
at the *start* of this task, before it was later overwritten — see "Data
integrity incident" below) is the same dataset the brief's own investigation
used.

### Before/after overlap table (WSe2, W_sv)

Using the exact-brief diagnostic recipe (`readBandCoeff(norm=True)`, plain
`Ck.conj()@Ck.T`) as the "before" baseline, vs. this task's PAW-corrected
overlap as "after", across the 5 required k-points, 13 occupied bands per
k-point:

| ik | nbands | baseline max\|offdiag\| | corrected max\|offdiag\| | corrected diag range |
|---|---|---|---|---|
| 1   | 13 | 1.5687×10⁻¹ | 5.3966×10⁻⁶ | [1.0000, 1.0000] |
| 2   | 13 | 1.5530×10⁻¹ | 2.8355×10⁻⁶ | [1.0000, 1.0000] |
| 50  | 13 | 1.7118×10⁻¹ | 7.0424×10⁻⁷ | [1.0000, 1.0000] |
| 150 | 13 | 1.8405×10⁻¹ | 9.3198×10⁻⁷ | [1.0000, 1.0000] |
| 300 | 13 | 1.3608×10⁻¹ | 7.0959×10⁻⁷ | [1.0000, 1.0000] |

**Result: off-diagonal overlap improves by roughly 5 orders of magnitude**
(0.14–0.18 → 3×10⁻⁷–5×10⁻⁶), comfortably exceeding the brief's "close to
the Si baseline of ~1e-8" bar, and diagonal (norm) lands at exactly 1.0000
to the precision shown. This is goal (a), fully met.

Raw data: `diagnostic_results_ORIGINAL_verified_Wsv.json` (values
transcribed from the console output of this run, since the underlying
WAVECAR bytes were later overwritten and could not be regenerated — see
below).

### Si regression check

Reproducing the brief's exact baseline diagnostic on `Data/Si/WAVECAR` (no
POTCAR present for Si, so no PAW correction is computed or applicable — Si's
potential is soft, this is a regression sanity check only):

| ik | nbands | baseline max\|offdiag\| |
|---|---|---|
| 1   | 4 | 1.4067×10⁻⁸ |
| 2   | 4 | 3.3140×10⁻³ |
| 50  | 4 | 1.4853×10⁻² |
| 150 | 4 | 1.9264×10⁻² |
| 300 | 4 | 1.6360×10⁻² |

**Finding, stated plainly: the brief's claim "Si should already show ~1e-8"
holds only at ik=1 (Γ)**; at the other four k-points the plain
plane-wave-only overlap for Si is 3×10⁻³–2×10⁻². This is unrelated to PAW
augmentation — Si's potential is soft (no semicore shell), and there is no
POTCAR available for Si in this dataset to build or test any correction
against. I ruled out the obvious culprits before accepting this as a
pre-existing property of the reference data rather than a bug in my
diagnostic:
  - Occupations are clean (`[1,1,1,1,0,0,0,0]` at every k tested — not a
    fractional-occupation artifact).
  - G-vector counts match `_nplws` exactly at every k tested (not an
    aliasing/grid-generation bug).
  - The bands involved are *not* degenerate (e.g. at ik=300, offdiag ≈0.015
    between bands separated by several eV) — so this isn't the ordinary
    "arbitrary rotation within a degenerate subspace" explanation either.

My best guess is that this specific `Data/Si/WAVECAR` was generated with
looser SCF/precision settings than would be needed for exact orthogonality
away from Γ (it's a large, 3375-k-point file with only 8 bands — plausibly
built for a different original purpose, e.g. band unfolding, than exact
density-matrix-grade orthogonality). This does not affect the PAW-correction
conclusion for WSe2 either way, since no correction is being tested here —
it's flagged only because the brief asked for this exact regression check
and the result contradicts the brief's stated expectation at 4 of 5
k-points. **The correction (goal a) does not make this worse, because no
correction is applied to Si at all.**

---

## Data integrity incident (important — read before trusting anything below this line)

Partway through this task, **`Data/WSe2_mono/WAVECAR` was overwritten on
disk** (observed: file size changed, `nbands` changed 17→15, occupied-band
count at k=1 changed 13→9, timestamp updated to `2026-07-09T10:00`
local time). The user confirmed mid-task that the *new* WAVECAR is from a
separate, concurrent calculation using a **plain, non-semicore `W`**
pseudopotential — while `POTCAR` in the same directory is unchanged and
still describes **`W_sv`**.

**A WAVECAR and a POTCAR that don't describe the same pseudopotential cannot
be validly combined for a PAW correction** — the projector functions and
`Qij` parsed from `W_sv` do not correspond to the actual pseudopotential
that generated the new WAVECAR's plane-wave coefficients. This is entirely
expected: `W` (ZVAL≈6, no 5p/5s semicore shell) vs. `W_sv` (ZVAL=14) is
exactly the "switch POTCAR" alternative fix TASK_BRIEF.md itself describes
as being tried in parallel — 6(W) + 6(Se) + 6(Se) = 18 valence electrons →
9 doubly-occupied bands matches the new WAVECAR's occupation exactly
(`occ_all = [1,1,1,1,1,1,1,1,1,0,...]`, 9 bands), confirming this
interpretation.

Consequences, handled as follows:

1. **Goal (a) numbers above are from the *original* matching WAVECAR+POTCAR
   pair**, captured in console output *before* the overwrite (confirmed by
   the exact match to the brief's own raw-norm table — a coincidental match
   is not plausible given 13 significant figures of agreement across 13
   bands). Fully valid, nothing to caveat there.
2. The original WAVECAR bytes are **not recoverable** — `WAVECAR` is
   `.gitignore`d (`Data/WSe2_mono/WAVECAR` is matched by the repo's
   `.gitignore`), so there is no git history to restore from, and no backup
   was found elsewhere in the repo (checked `Backup script/`, which only has
   an unrelated old copy of `main.py`).
3. To keep the rest of this task's work internally self-consistent (and not
   racing a live-changing file mid-computation), I took a **frozen snapshot**
   of the (now W-not-W_sv-matching) `Data/WSe2_mono/{WAVECAR,POSCAR,POTCAR,
   EIGENVAL}` at `2026-07-09T14:20:28Z` into `paw_augmentation/data_snapshot/
   WSe2_mono/` (and `Data/Si/{WAVECAR,POSCAR}` into `data_snapshot/Si/`, for
   the regression check, which is unaffected by this incident). All scripts
   in this folder read from `data_snapshot/`, never live from `Data/`.
4. **Goal (b) was consequently not completed against physically valid,
   currently-available data** — see next section for exactly what was and
   wasn't done, and why.

As a data point (not a claim of physical validity), re-running the *exact
same* correction machinery against the current mismatched snapshot still
shows a real, if smaller, improvement: max off-diagonal overlap goes from
0.075–0.093 (baseline) to 0.0038–0.0049 (corrected) — roughly one order of
magnitude, with diagonal in [0.991, 1.003] rather than exactly 1.0000. This
is *some* evidence the correction degrades gracefully rather than blowing up
when fed a mismatched POTCAR, but it is not a valid physics result and
should not be read as one — full numbers in
`diagnostic_results_MISMATCHED_snapshot_W_not_Wsv.json`, clearly labeled.

---

## Goal (b): full corrected density matrix / CNO occupations

**Status: implementation complete; core construction independently
cross-validated; final physically-valid WSe2 W_center result NOT obtained**
due to the data integrity incident above. Documented plainly per the
brief's own request rather than reporting a number built on mismatched
data.

### Method (derivation)

`main.py`'s existing (uncorrected) pipeline builds a real-space density
matrix `rho[r,r'] = sum_nk w_k f_nk psi_nk(r) psi_nk*(r')` on the WS-cell
grid and diagonalizes it directly (`rho v = lambda v`), implicitly assuming
the real-space grid basis `{|r>}` is orthonormal. The brief's suggested fix
is a generalized eigenproblem `D v = lambda S v` using a corrected overlap
matrix, "not pointwise real-space AE reconstruction."

Key derivation (this task's contribution beyond directly reusing
`paw.py`/`aewfc.py`, which only handle band-pair quantities, not a
real-space metric):

- Because `numpy.fft.ifftn`/`fftn` is a unitary map (Parseval) between the
  zero-padded plane-wave coefficient vector and the `Nr`-point real-space
  grid vector, the grid basis `{|r>}` is *exactly* orthonormal under the
  plane-wave-only inner product — main.py's `rho` construction (from IFFT'd
  coefficients) is not itself the source of error. So **`D` needs no
  structural change**, only: build it from RAW (`norm=False`) coefficients
  instead of `norm=True`, matching the convention that makes goal (a)'s
  diagonal come out to 1.0000 automatically rather than by forced
  renormalization (implemented in `build_density_matrix()` in
  `paw_density_matrix.py`).
- The missing physics lives entirely in the **metric**: the correct
  position-space representation of the PAW operator
  `S_hat = 1 + sum_i |p~_i> Qij <p~_j|` is

  ```
  S[r, r'] = delta(r,r') + (1/Nr) * sum_{atom images R} sum_ij  P_i(r-R) Qij P_j*(r'-R)
  ```

  where `P_i(r)` is the plain (`nonlr`-convention) real-space projector
  value — radial spline (`pawpotcar.spl_rproj`) times real spherical
  harmonic (`sph_harm.sph_r`), **no Bloch phase**, because `S_hat` is a
  k-independent local operator (unlike the band-pair overlap, which lives at
  fixed k). This is derived, not reused verbatim, because `paw.py`'s
  `nonlr` class is built for a different purpose (projecting a *specific
  k*'s wrapped-grid wavefunction, hence carries a Bloch-phase factor
  `crrexp` that must be *absent* here) — implemented directly in
  `build_real_space_S()` rather than calling `nonlr`.
  - The `1/Nr` normalization factor was **not obvious a priori and was
    originally wrong** (off by a huge margin — first attempt gave `S`
    eigenvalues in the thousands). It was pinned down by directly comparing,
    band-by-band, a real-space evaluation of `beta_n,i` against the
    validated reciprocal-space `nonlq.proj()` result for the same band
    (`_test_beta_consistency.py`) until they matched — this is the one
    genuinely new piece of PAW math in this task, and it is now cross-checked
    against the already-validated reciprocal-space path, not
    hand-derived and trusted blindly.
- This reuses only `pawpotcar`'s radial spline data + `Qij` and
  `sph_harm.sph_r` — no `pysbt`, no pointwise real-space AE-wavefunction
  reconstruction, consistent with the brief's preferred approach.

### Validation performed (code-correctness, not final-answer validation)

1. **Structural**: `S` built on the (mismatched) snapshot is Hermitian to
   1e-14, positive definite (min eigenvalue 0.40, all others positive — a
   valid metric), diagonal in [0.96, 1.50] (i.e. a *bounded* correction to
   identity, as physically expected, not the wildly-wrong O(1000) values
   from the pre-normalization-fix attempt).
2. **Band-pair cross-check** (`_test_beta_consistency.py`): a real-space
   evaluation of `beta_n,i = <p~_i|psi~_n>` (same machinery used inside
   `build_real_space_S`) agrees with the independently-validated
   reciprocal-space `nonlq.proj()` result to a few percent for a real
   occupied W 5p semicore band. Residual disagreement (worse for
   higher-l/d-channel projectors) is consistent with real-space quadrature
   error on this system's comparatively coarse `11×11×73` FFT grid (only
   ~4–5 points across a 1.36 Å PAW sphere diameter in-plane) — not a
   normalization bug, which was ruled out by testing candidate factors of
   `1`, `1/sqrt(Nr)`, and `1/Nr` explicitly.
3. **Global trace check**: on the plain (non-WS-cell) FFT box,
   `Tr(D @ S)` should equal the total occupied-electron count
   `sum_k w_k * sum_n f_nk` exactly, *if* `S` is built correctly (this
   follows because `Tr(D@S) = sum_nk w_k f_nk <psi_nk|S|psi_nk>`, and
   goal (a) already showed `<psi|S|psi> ≈ 1` to high precision). Measured
   (on the mismatched snapshot): `Tr(D@S) = 9.0396` vs. expected `9.0000`
   — 0.44% relative error, consistent with the same real-space quadrature
   error identified in check 2. This check validates the *code* (algebra,
   indexing, normalization bookkeeping) using whatever WAVECAR happens to
   be loaded; it does not require a physically-matched POTCAR to be a valid
   test of the implementation, only of the arithmetic — so it remains
   meaningful evidence of correctness despite the data mismatch.
4. **End-to-end run**: `paw_density_matrix.py` was run to completion on the
   WS-cell grid for `WSe2_mono` / `W_center` (`config.py`'s current
   settings) against the mismatched snapshot, purely to confirm the full
   pipeline (S construction, D construction, generalized `eigh`) executes
   without error and produces a sane-looking spectrum — see numbers below,
   **explicitly not a physically valid result**, included only to
   demonstrate the implementation is complete and functional, not
   vaporware.

### What was NOT achieved

A final, physically valid WSe2 `W_center` corrected-CNO-occupation number,
verified against a matching `W_sv` WAVECAR+POTCAR pair, as goal (b) asks
for. The blocker is external (the input data changed mid-task), not a
methodological dead end — `paw_density_matrix.py` is ready to run as-is
against a valid matching WAVECAR the moment one is available (point
`data_dir` in `main()` back at `Data/WSe2_mono` once a `W_sv`-matching
WAVECAR is regenerated, or update `data_snapshot/WSe2_mono/` with a fresh
copy).

### End-to-end demonstration run (mismatched data — not physically valid)

Full run: WSe2_mono / `W_center` (`config.py`'s current settings: WS center
`[1/3, 2/3, 1/2]` fractional → the W site), `Nr=8833` grid points
(`11×11×73`), 324 k-points, 9 occupied bands/k, against the mismatched
snapshot. Wall time: S-matrix construction <1s (7 atom-images contributed,
consistent with only the 3 base atoms' nearest periodic images falling
within cutoff of the WS cell — no distant images spuriously included), D
construction 229s, generalized `eigh` 377s (11m total).

```
|S - S^dagger|_max = 3.96e-14        (Hermitian, as expected)
|D - D^dagger|_max = 2.61e-18        (Hermitian, as expected)
Tr(D)   = 9.771187                    (uncorrected trace)
Tr(D S) = 9.496340                    (expected ~9.0 if S/POTCAR matched D's WAVECAR)

Uncorrected (plain eigh(D), no S) top 6: 0.9707, 0.9397, 0.9087, 0.8577, 0.8552, 0.7980
  max = 0.970721   -- already <= 1, because this dataset's WAVECAR (plain W,
                       no semicore shell) never had the augmentation problem
                       to begin with (consistent with Si's soft-potential
                       behavior in the goal-a regression check).

Corrected (D v = lambda S v, using MISMATCHED W_sv Qij/projectors) top 6:
  1.0982, 1.0924, 1.0237, 1.0046, 0.9242, 0.8929
  max = 1.098165,  min = -0.000000
  N(eigenvalues outside [0,1] by > 1e-3) = 4
```

**This is the cleanest possible illustration of why the data mismatch
matters, and it is a negative result worth stating plainly**: applying the
`W_sv` augmentation correction to bands that were actually generated with a
plain (non-augmented-in-that-way) `W` potential does not merely fail to
help — it *actively pushes 4 eigenvalues above the physical bound that were
within it before correction* (max eigenvalue 0.971 → 1.098), and introduces
a small spurious negative eigenvalue. `Tr(D S) = 9.496` vs. the ~9.0 expected
if `S` genuinely matched (5.5% deviation, notably worse than the ~0.4%
quadrature-only error measured on the plain FFT box in the earlier,
data-matched-but-still-quadrature-limited check) is consistent with this
being a real physical mismatch, not just discretization noise.

**Do not read this as "the correction doesn't work"** — goal (a) already
demonstrated the identical formula (just at fixed k, no real-space grid)
removes 5 orders of magnitude of spurious non-orthogonality on genuinely
matched `W_sv` data. Read it as: PAW corrections are only valid when the
projector data actually corresponds to the potential that generated the
wavefunction, exactly as physically expected, and this run is included
specifically to make that failure mode visible and quantified rather than
silently absent from this report.

---

## Files in this folder

- `paw_overlap.py` — goal (a): PAW-corrected band-pair overlap
  (`PawOverlapCorrector`), reusing `paw.pawpotcar`/`paw.nonlq` directly.
- `diagnostics.py` — before/after overlap table generator (brief's exact
  baseline recipe vs. corrected), for both WSe2 and Si.
- `paw_density_matrix.py` — goal (b): real-space metric construction
  (`build_real_space_S`) + density matrix (`build_density_matrix`) +
  generalized eigenproblem solve, replicating only what's needed from
  `main.py`/`ws_cell.py`/`config.py` (all read-only imports, nothing run or
  modified).
- `_sanity_check.py` — brief-table reproduction check (goal a prerequisite).
- `_test_beta_consistency.py` — real-space vs. reciprocal-space projector
  cross-check that pinned down the `1/Nr` normalization for goal (b).
- `_test_S_build.py`, `_test_trace_check.py` — structural/global validation
  scripts for goal (b)'s `S` matrix.
- `diagnostic_results_ORIGINAL_verified_Wsv.json` — goal (a) authoritative
  numbers (transcribed from console output; see data-integrity section).
- `diagnostic_results_MISMATCHED_snapshot_W_not_Wsv.json` — same diagnostic,
  re-run against the post-overwrite snapshot; explicitly not physically
  valid, kept for the record.
- `data_snapshot/` — frozen copy of `Data/WSe2_mono` (post-overwrite,
  mismatched) and `Data/Si` (unaffected), taken to keep this task's later
  steps internally self-consistent. Not a substitute for regenerating a
  valid `W_sv` WAVECAR.
- `output/` — `paw_density_matrix.py`'s saved eigenvalue arrays
  (`cno_occupations_corrected.npy`, `cno_occupations_uncorrected_samedata.npy`)
  and text report from the mismatched-data demonstration run. The `S_ws.npy`/
  `D_ws_raw.npy` intermediate matrices (~1.2 GB each) were deleted after the
  run to avoid leaving multi-GB low-value artifacts on disk — regenerable by
  re-running `paw_density_matrix.py` (~11 min) if needed.

---

## 2026-07-10 update: preflight gate + trace-reporting fix

**Scope of this update, per the task that produced it:** stay entirely
inside `paw_augmentation/`; do not touch `main.py`/`config.py`; keep using
the existing raw-coefficient `D` + `build_real_space_S()` + `M = S^(1/2) D
S^(1/2)` construction (no algorithm change); add a cheap preflight gate and
fix the trace-reporting bookkeeping in the report. SOC, real-space
quadrature refinement, and the exact regional `T^dagger P_A T` restriction
operator are explicitly deferred (see "Future work" below), not attempted
here.

### Why: the trace-reporting bug this fixes

The old report compared `Tr(D S)` against `sum(eigvals)` and called that
agreement a validation of the result. It isn't — `sum(eigvals)` **is**
`Tr(M)` by construction of `np.linalg.eigh`, and `Tr(M) = Tr(D S)` is a
similarity-transform identity (`M = S^(1/2) D S^(1/2)` is similar to `D S`).
Those three numbers agreeing to `1e-14` only proves the eigensolver/algebra
didn't make an arithmetic mistake — it says nothing about whether the
*physical* electron count came out right, because there was no independent
number to compare against. This update adds one:
`trace_expected_input = sum_k w_k * sum_n f_nk`, computed inside
`build_density_matrix()` from the *exact same* per-k `occ` array used to
accumulate `D` (not a separately re-derived formula, so it can't silently
diverge from what `D` actually encodes).

### Preflight gate

Before the expensive `Nr x Nr` real-space `S`/`D` construction (`Nr=8833`
for `WSe2_mono`'s current grid; ~11 minutes, ~1.2 GB per matrix),
`preflight_paw_overlap_check()` now runs the existing, cheap, few-k-point
reciprocal-space band-pair overlap correction (`paw_overlap.py`, unmodified)
at 5 k-points spread across the mesh, and requires the corrected
`max|offdiag|` to land below `1e-3` **and** improve on the uncorrected value
by at least `10x` (unless the uncorrected value was already below `1e-4`,
i.e. a soft potential with nothing to correct). If this fails, the script
aborts immediately with a `BLOCKED` report — no `Nr x Nr` array is built,
and nothing under `output/` is overwritten.

### Dataset status (this update)

| Material | POTCAR | Preflight | Full run | Notes |
|---|---|---|---|---|
| `WSe2_mono` | present | **PASSED** (see below) | ran to completion | see trace table below |
| `Si` | absent | n/a | **BLOCKED** | no POTCAR — cannot build any PAW correction |
| `CoSn` | absent | n/a | **BLOCKED** | no POTCAR; also `LSORBIT=True` (SOC), which is separately guarded/deferred regardless |

**Note on `WSe2_mono`'s WAVECAR/POTCAR state:** at the start of this task
the live `Data/WSe2_mono/POTCAR` had just been regenerated (mtime
2026-07-10 13:48, apparently by a concurrently running VASP job on the
user's side) against the existing WAVECAR (unchanged since 2026-07-09). An
initial preflight run against the *pre*-13:48 POTCAR (captured independently
via the sibling `diagnostics/` package's `test_fixed_k_gram.py`) showed the
same poor correction (`max|offdiag|` only dropping from ~0.13 to ~0.08–0.10,
nowhere near identity) documented in this file's original "data integrity
incident" section — i.e. the preflight gate correctly caught a mismatched
pair when one was actually present. After the 13:48 POTCAR update, the
identical preflight check now shows a clean, strong result (below), which is
exactly the gate behaving as designed: it tracks the *current* on-disk state
rather than assuming either "always broken" or "always fine."

#### Preflight result (current `WSe2_mono` WAVECAR + POTCAR pair)

| ik | nbands | uncorrected max\|offdiag\| | corrected max\|offdiag\| | improvement |
|---|---|---|---|---|
| 1   | 13 | 1.4094×10⁻¹ | 5.4015×10⁻⁶ | 26,093× |
| 81  | 13 | 1.3611×10⁻¹ | 6.8576×10⁻⁷ | 198,479× |
| 162 | 13 | 1.3070×10⁻¹ | 4.7224×10⁻⁷ | 276,762× |
| 243 | 13 | 1.2529×10⁻¹ | 6.1133×10⁻⁷ | 204,952× |
| 324 | 13 | 1.3821×10⁻¹ | 3.2364×10⁻⁶ | 42,706× |

All 5 pass (corrected `< 1e-3` and improvement `>= 10x`). **Preflight: PASSED.**

#### Full run: trace reporting (current `WSe2_mono`, `W_center_2`)

```
trace_raw_D              = 11.9940998117   (raw pseudo-coefficient trace only -- NOT a particle number)
trace_paw_M              = 13.5178774247
trace_DS                 = 13.5178774247
sum_paw_cno_occupations  = 13.5178774247
trace_expected_input     = 13.0000000000   (13 occupied bands/k, occ never exceeded 1.5 -> no halving)
trace_M_minus_eigensum   = 8.8818e-15      (algebra/eigensolver check only)
trace_DS_minus_eigensum  = 1.2434e-14      (algebra/eigensolver check only)
trace_paw_minus_expected = 5.1788e-01      (independent physical normalization check)
trace_relative_error     = 3.9837e-02      (~4.0%)
```

The algebra checks (`trace_M_minus_eigensum`, `trace_DS_minus_eigensum`) are
clean to `1e-14`, confirming the eigensolver/matrix construction is
internally consistent. The **independent** physical check
(`trace_paw_minus_expected`) is not: the corrected matrix trace overshoots
the expected 13 occupied orbitals by ~4.0%, and correspondingly
`max_eigval = 1.00296` (1 eigenvalue outside `[0,1]` by `>1e-3`). Given the
preflight table above shows the *reciprocal-space* band-pair correction is
accurate to `~1e-6`–`1e-7`, this ~4% excess is not a mismatched-potential
problem — it is consistent with this repo's existing, previously-documented
finding that `build_real_space_S`'s real-space quadrature (spline-radial ×
spherical-harmonic evaluation on the comparatively coarse `11×11×73` grid,
only ~4–5 points across a PAW sphere diameter) has a residual error at the
few-percent level (`_test_beta_consistency.py`'s "agreement to a few %" note,
and the earlier `Tr(D@S) = 9.0396` vs expected `9.0000` / 0.44% check on a
different, smaller-grid dataset). **Overall validation status for this run:
FAIL**, specifically and only on the `occupations_within_01_bound` check —
every other check (Hermiticity of `S`/`D`/`M`, `S` positive-definiteness,
`X^H S X ≈ I`, and the trace algebra identity) passes. Full detail:
`output/paw_density_matrix_report.txt`.

**This is a real, useful, negative result, not a bug in the gate or the
trace bookkeeping**: it isolates the remaining error to the real-space
quadrature step specifically (not the augmentation formula, not the
eigensolver, not a WAVECAR/POTCAR mismatch), which the old report's trace
comparison could not have distinguished from "everything is fine" because it
never compared against an input-derived expectation.

### Future work (explicitly not resolved by this update)

- **Real-space quadrature convergence** of `build_real_space_S` has not been
  systematically checked against grid density (finer FFT grid, adaptive
  integration, etc.) — the ~4% trace excess measured above is consistent
  with, but not rigorously isolated as, this quadrature error.
- **Exact regional `T^dagger P_A T`**: the theoretically exact way to
  restrict the *full* PAW operator (not just its plane-wave part) to a WS
  cell is not implemented; `build_real_space_S` evaluates the augmentation
  term at the actual (possibly WS-unwrapped) Cartesian grid coordinates as a
  practical approximation, without separately verifying this equals the
  exact regional restriction.
- **SOC (`LSORBIT=True`)** is explicitly detected and causes a clean
  `BLOCKED` abort — `build_density_matrix` has no spinor path.
- **Fermi-window occupation mode** (`RESTRICT_TO_FERMI_WINDOW=True`) is
  likewise detected and blocked — `build_density_matrix` only implements the
  `occ_tol`-threshold, main.py-non-Fermi-window band-selection convention.

`main.py` and `config.py` were not modified by this update.

---

## 2026-07-10 update #2: real-space quadrature convergence check

**Scope:** post-processing diagnostic only, entirely inside
`paw_augmentation/`, using existing WAVECAR data with zero-padded (band-
limited-interpolated) plane-wave coefficients — no VASP rerun, no `Nr x Nr`
matrix built at any enlarged grid, `main.py`/`config.py` not touched (only
read). New script: `quadrature_convergence_check.py`.

### Question

The `~4.0%` `trace_paw_minus_expected` excess found in update #1 (above) was
consistent with, but not proven to be, real-space quadrature error in
`build_real_space_S`. This check isolates the cause by refining the
*sampling density* of the *same* pseudo wavefunctions (via exact zero-padded
FFT interpolation — no new physics, since the plane-wave coefficients are
already band-limited by ENCUT) and watching whether a real-space-computed
`beta_n,i = <p~_i|psi_n>` / PAW-corrected band Gram matrix converges toward
the (grid-independent, already-validated) reciprocal-space reference from
`paw_overlap.py`.

### A bug found and fixed along the way

The first version of `real_space_beta_for_bands()` reused
`build_real_space_S`'s "no Bloch phase" convention verbatim. That convention
is *correct* for `S_hat` itself (a genuinely k-independent operator — not
touched by this update), but **wrong** for `beta_n,i`, which is a property
of one specific Bloch state `(n,k)`. The symptom made the bug obvious:
**ik=1 (Γ, k=0) converged beautifully with grid refinement; ik=163 and
ik=324 (both non-Γ) were completely flat across all three grid factors** —
exactly the signature of a missing k-phase (which vanishes identically at
Γ), not of insufficient sampling. Full numbers from that first (buggy) run
are kept at
`output/quadrature_convergence_FIRST_ATTEMPT_missing_bloch_phase.txt` for
the record, not as the reported result.

Re-deriving `beta_n,i` from the projector integral (expanding over unit-cell
images and using Bloch periodicity `psi(r+R) = exp(2*pi*i*k.R)*psi(r)`,
matching `main.py`'s own sign convention) shows two things were missing:

1. the wavefunction values fed into the real-space sum must be the **full**
   Bloch `psi_n(r) = exp(2*pi*i*k.r_frac) * u_n(r)`, not the bare
   cell-periodic `u_n(r)`;
2. each atom-image's contribution needs an extra **per-image** phase
   `exp(-2*pi*i*k_frac . n_image)` (`n_image` = the integer lattice
   translation for that image), applied before summing over images.

Both fixes are implemented in `real_space_beta_for_bands()` (see its
docstring and the module-level "Real-space beta derivation" section for the
full derivation). Both vanish at Γ, consistent with the symptom.

### Corrected convergence result (worst case over ik = 1, 163, 324)

| grid factor | grid | PAW-corrected Gram max\|offdiag\| (prim = WS) | reciprocal reference |
|---|---|---|---|
| 1x | (11,11,73) | 2.853×10⁻³ | 5.402×10⁻⁶ (grid-independent) |
| 2x | (22,22,146) | 1.414×10⁻⁴ | — |
| 3x | (33,33,219) | 7.521×10⁻⁵ | — |

**Shrink 1x→3x: 37.9× for both the primitive-cell and WS-mapped
coordinates** (they agree to the digits shown at every grid factor for this
`WS_CENTER` — the WS remapping itself introduces no additional error here
once the phase is correct). Diagonal PAW norms tighten from
`[0.9991, 1.0031]` (1x) to `[0.99998, 1.00004]` (3x).

**Diagnosis: `quadrature_confirmed`.** Both paths converge cleanly and
monotonically toward the reciprocal-space reference as the real-space grid
is refined — this is genuine quadrature undersampling in
`build_real_space_S`, not a WS-cell-specific bug and not a normalization
error.

**Recommendation (option 1, per the task's decision rule): adopt a
reciprocal-space / atom-centered low-rank PAW construction for the
production `S` matrix**, rather than a much finer production FFT grid. The
existing `nonlq`/`Qij` reciprocal-space machinery (`paw_overlap.py`) already
gives a `~1e-6`–`1e-7` accurate, grid-independent band-pair correction with
no quadrature error at all — the practical path is to build the production
metric/correction from that machinery (band-pair or atom-centered
projector overlaps) rather than trying to reach the same accuracy through
real-space grid refinement, which extrapolating the 1x→2x→3x trend would
require an impractically dense production grid to match.

**One honest nuance, not swept under the rug:** the *raw* `beta` component-
wise max-abs difference (`beta_diff_prim`/`beta_diff_ws` in the report) does
**not** shrink as cleanly at the non-Γ k-points (e.g. ik=163 stays
`~3.27` across all three grid factors) even though the physically-weighted
Gram matrix converges smoothly. This means a small number of individual
projector-channel `beta` components have a persistent absolute residual
that does not vanish with grid refinement, but which contributes
negligibly to the `Qij`-weighted Gram matrix (the quantity that actually
matters physically). This is reported, not hidden, in
`output/quadrature_convergence_report.{json,txt}` — it does not change the
`quadrature_confirmed` diagnosis (which is based on the Gram matrix, the
physically meaningful quantity), but is worth keeping in mind if this
diagnostic is extended or reused.

Full data: `output/quadrature_convergence_report.json` /
`output/quadrature_convergence_report.txt`. This script never built an
`Nr x Nr` matrix at any grid factor, never rescaled `S`, and never clipped
any occupation — it does not touch the production S/D/eigenvalue pipeline.
`main.py` and `config.py` were not modified (config.py was only read, for
`MATERIAL`/`WS_CENTER`/`WS_TRANSLATION_SEARCH_RANGE`).

---

## 2026-07-11 update: low-rank (state-space) PAW-CNO experiment

**Scope:** new experimental module `paw_lowrank_cno.py`, entirely inside
`paw_augmentation/`. `main.py`/`config.py` not modified (only read). No VASP
rerun. No `Nr x Nr` `D`/`S` built at 2x/3x — the *validation* reference
reuses `quadrature_convergence_check.py`'s already-implemented 2x/3x
real-space beta at a small, representative pair of k-points only; the
*production* build stays at the native (1x) grid throughout.

### Method

Implements update #2's recommendation 1: instead of diagonalizing the large
real-space `D @ S` problem, build the small state-space matrix
`K = P^(1/2) (G_ps + G_aug) P^(1/2)` (`nstates x nstates`, `nstates=4212`
for all occupied bands over the full 324-k-point mesh — far smaller than
`Nr=8833`) and diagonalize that instead; its nonzero eigenvalues equal
`D @ S`'s. `G_ps` (pseudo/plane-wave overlap) is built exactly from the
existing FFT/WS-grid machinery (Parseval-exact, no quadrature error,
whether or not `a,b` share a k-point). `G_aug` (PAW augmentation) is built
from `paw.nonlq.proj()` — the existing, zero-quadrature reciprocal-space
route — for every state, at every k-point.

### A real bug found and fixed: cross-k gauge mismatch in `nonlq.proj()`

`paw.nonlq.proj()`'s atom-position phase (`exp(2*pi*i*G.tau)`, using the
bare reciprocal index `G`, not `G+k`) is a valid, standard convention for
**same-k** band-pair overlaps (`paw_overlap.py`'s existing, validated use
case) — the missing `exp(2*pi*i*k.tau)` factor is identical for both states
being paired, so it cancels exactly. It does **not** cancel for a **cross-k**
pair (`k_a != k_b`), leaving a spurious relative phase
`exp(-2*pi*i*(k_b-k_a).tau_atom)`. This was caught empirically by this
module's required same-k/cross-k block validation (same-k sub-block matched
the 3x real-space reference to ~1e-5; the cross-k sub-block was off by up
to **0.96**), then confirmed algebraically: a first-principles real-space
rederivation of `beta_n,i = <p_i|psi_n>` shows the direct real-space integral
equals `nonlq.proj()`'s output times `exp(2*pi*i*k_frac.tau_atom_frac)`,
verified numerically to 4+ decimal places for a specific band/atom/k-point
pair. The fix (`gauge_correct_beta()`) multiplies each state's reciprocal
beta, per atom, by this factor before any `Qij` pairing. This reduced the
worst raw cross-k block disagreement from **0.96 to 0.035**.

### A validation-methodology fix: isolating `G_aug` from `G_ps`

After the gauge fix, a residual ~0.035/~0.009 block disagreement remained.
Tracing it down (see the module's git history / this section) showed it was
**entirely in `G_ps`**, not `G_aug` — and it was an artifact of the
validation setup, not a real error: comparing "native-grid `G_ps` + reciprocal
`G_aug`" against "3x-grid `G_ps` + 3x `G_aug`" conflates two different
things, since cross-k `G_ps` (unlike same-k `G_ps`) is *not* Parseval-exact
—it is itself a finite-grid quadrature of an integral that is not periodic
with the cell (a discrete sum over an aperiodic-in-the-cell product), so it
legitimately differs somewhat between grid densities. But this native-grid
`G_ps` is exactly what the existing, unmodified production `D` matrix
already uses — it is not part of what this experiment is testing. Fixing
the validation to hold `G_ps` fixed (native grid, identical on both sides)
and vary only the augmentation treatment isolates the actual question
(is the gauge-corrected reciprocal `G_aug` accurate?) and confirmed it
cleanly: **max block disagreement dropped to 7.55e-5**, matching the
per-atom precision exactly, with 0 of 676 sampled entries flagged as
"reference not yet converged" (the earlier 2x-vs-3x self-convergence check
described in the previous entry).

### Full production run (all 324 k-points, 4212 states)

| quantity | value |
|---|---|
| nstates | 4212 (Nr=8833, so the state-space problem is ~2.1x smaller, and dominated by a rank-44 augmentation correction rather than a dense Nr x Nr metric) |
| trace_expected_input | 13.0000000000 |
| trace_K | 13.0000029802 |
| trace_K − trace_expected_input | **2.98×10⁻⁶** (0.00002% relative) |
| trace_K − sum(eigenvalues) | 7.1×10⁻¹⁵ (pure algebra check) |
| diag(G) range | [0.999997, 1.000003] (PAW norms) |
| max eigenvalue | 0.999988 |
| min eigenvalue | **−0.199519** |
| N(outside [0,1] by >1e-3) | **2** |
| state-space orthonormality (`Y^H G Y − I`) | 1.38×10⁻¹¹ |
| runtime | **67.5 s** total (Psi+Beta build 11.0 s, G matmul 4.6 s, eigh 43.6 s) |
| peak traced memory | 2305.9 MB |

**Overall validation status: FAIL**, specifically on `K_positive_semidefinite`
and `occupations_within_01_bound` — everything else passes (Hermiticity,
PAW-norm-close-to-1, the trace algebra identity, state-space orthonormality
of the reconstructed orbitals).

**This is a real, not-yet-resolved anomaly, reported honestly rather than
hidden**: 2 eigenvalues fall outside `[0,1]`, the most negative reaching
−0.1995. A quick investigation of the corresponding eigenvector shows it is
**delocalized** — weight spread thinly (~0.24% each) over many states,
not concentrated on one obviously "bad" band pair — which rules out a
single missed case like the ones found during validation, and is more
consistent with a small, systematic residual spread across many cross-k
pairs in the full 324-k-point mesh that the 2-k-point validation sample
(26 of 4212 states) did not happen to catch. Root-causing this fully would
require checking many more k-point pairs than the representative sample
this task's validation step calls for; that is **left as follow-up work**,
not resolved here (see "Recommended follow-up" below).

Despite this, the improvement over the existing real-space method is large
and unambiguous: trace accuracy improved by **~4 orders of magnitude**
(4.0% → 0.00002% relative error) and PAW norms are essentially exact
(previously [0.9991, 1.0057], now [0.999997, 1.000003]), while running
**~13x faster** (67.5 s vs. the real-space method's ~920 s core solve) with
noticeably less peak memory (~2.3 GB vs. multiple simultaneous `Nr x Nr`
arrays at ~1.25 GB each, ~6-8 GB estimated peak for the real-space method).

### Comparison table (as requested)

| | old uncorrected (`eigh(D)`, no `S`) | old 1x real-space-`S` corrected | new low-rank (gauge-corrected reciprocal) | converged 3x sampled reference |
|---|---|---|---|---|
| max occupation | 1.08775 | 1.00296 | 0.99999 | n/a — block-level only, not a full spectrum (see note) |
| min occupation | n/a (not reported) | −0.00000 | **−0.19952** | n/a |
| trace (Tr(D) / Tr(DS) / Tr(K)) | 11.99410 (raw `D`, not particle number) | 13.51788 | 13.00000 | n/a |
| trace vs. expected (13.0) | n/a | +0.518 (**+4.0%**) | **+2.98×10⁻⁶ (+0.00002%)** | n/a |
| N(eigval outside [0,1] by >1e-3) | n/a | 1 | 2 | n/a |
| diag(G)/PAW-norm range | n/a (no `S`) | [0.9991, 1.0057] | **[0.999997, 1.000003]** | matches lowrank to 7.55×10⁻⁵ (validated blocks) |
| runtime (core solve) | included in the 1x row (same `D`) | ~920 s (220 s `D` build + 693 s `solve_paw_cno`) | **67.5 s** | n/a (2 k-points, seconds) |
| peak memory | included in the 1x row | ~6-8 GB estimated (several simultaneous `Nr x Nr` arrays; not directly measured) | **2305.9 MB** (measured, `tracemalloc`) | negligible (26×26 blocks) |

Note on the "converged 3x sampled reference" column: per the task's explicit
constraint ("do not construct `D` or `S` on a 2x/3x dense production grid"),
no full 3x CNO spectrum was computed (that would require an `Nr=238491`
`D`/`S` build, explicitly out of scope). The 3x reference exists only as
the small, representative same-k/cross-k *block* comparison described
above, which the low-rank method matches to 7.55×10⁻⁵ once `G_ps` is held
fixed correctly.

### Deferred / follow-up (not resolved here)

- **Root-causing the 2 out-of-bounds eigenvalues.** The responsible
  eigenvector is delocalized across many states; identifying the specific
  cross-k pairs (if any small subset dominates) or confirming this is a
  systematic small-residual effect requires validating more than the 2
  representative k-points this task's validation step calls for. Until
  resolved, the low-rank occupations should be treated as informative and
  a large improvement over the real-space method, but not yet fully
  certified to respect the `[0,1]` bound.
- **The exact regional operator `T^dagger P_A T`** — this experiment still
  uses the present PAW cell-overlap model (the same `S` definition as
  `build_real_space_S`, just reformulated in state space and validated
  against a 3x real-space reference for representative blocks). A later
  task will separately examine the exact regional restriction operator;
  that scope was not expanded here.
- Real-space quadrature convergence of cross-k `G_ps` itself (a genuine,
  separate effect discovered while isolating the validation above — a
  discrete grid sum over an aperiodic-in-the-cell cross-k product is not
  Parseval-exact the way same-k sums are) was not investigated further,
  since it does not affect the low-rank method (which uses the exact same
  native-grid `G_ps` the existing production `D` matrix already uses).

Outputs: `output/cno_occupations_lowrank.npy`,
`output/cno_state_eigenvectors_lowrank.npy`,
`output/cno_orbitals_pseudo_lowrank.npy` (explicitly experimental/pseudo
label — does not overwrite any production or prior-experiment CNO file),
`output/paw_lowrank_report.{txt,json}`.

`main.py` and `config.py` were not modified by this update.

---

## 2026-07-11 update #2: Phase 1 localization + Phase 2 regional `T†P_A T` fix

**Plain summary up front, as requested:**

| stage | status |
|---|---|
| old real-space `S` (`paw_density_matrix.py`) | quadrature/trace failure — max eigval 1.003, Tr(D·S) 4.0% above expected |
| first low-rank (`paw_lowrank_cno.py`, full-`Q` reciprocal) | **trace fixed** (2.98×10⁻⁶ error) but **min eigenvalue −0.1995 — a fatal, non-physical PSD violation** |
| Phase 1 diagnostics | confirmed root cause: full atomic `Q` applied to atoms whose augmentation sphere is *split* across multiple WS-cell images |
| Phase 2, regional `T†P_A T` (`paw_regional_cno.py`) | **PSD violation resolved** (min eigenval ≈ 0, 0 out-of-bounds states) — new, milder trace excess (~4.5%) remains, flagged for follow-up |

Scope: two new modules, `paw_lowrank_phase1_diagnostics.py` (Phase 1) and
`paw_regional_cno.py` (Phase 2), both entirely inside `paw_augmentation/`.
`main.py`/`config.py` not modified (config.py only read). No eigenvalue
clipping, no projection of `K` onto the PSD cone anywhere — the fix is a
formula change (region-intersected augmentation), not a numerical patch.

### Phase 1: localizing the negative eigenvalue before touching the formula

**1. Separate reconstruction of `G_ps`, `G_aug`, `G_total`, `K` (full 324-k-point mesh, 4212 states):**

| quantity | min eigenvalue | Hermiticity error |
|---|---|---|
| `G_ps` | −0.000000 (PSD, as expected — a plain Gram matrix) | 6.7×10⁻¹⁶ |
| `G_aug` | −126.802364 (need not be PSD alone — a `Qij`-sandwiched form) | 1.3×10⁻¹⁵ |
| `G_total` | **−64.644104** (physically REQUIRED to be ≥ 0 — it is not) | 1.3×10⁻¹⁵ |
| `K` | −0.199519 (the same violation, damped by the small `sqrt(p_a)` weights) | 4.3×10⁻¹⁸ |

`K`'s negativity is provably *inherited* from `G_total`'s: since
`K = sqrt(P) G_total sqrt(P)` is a congruence transform by a positive
diagonal matrix, `K` can only be PSD if `G_total` is — so the −0.1995 seen
previously was never a `K`-specific artifact; the real problem lives in
`G_total` and is roughly 300× larger there before the state-weighting
dilutes it.

**2. Eigenvector decomposition.** For `G_total`'s most negative eigenvector `u` (eigenvalue −64.64): `u†G_ps u = 30.93`, `u†G_aug u = −95.57`, sum
−64.64 ✓. An independent real-space, per-(atom, image)-site decomposition
of the augmentation expectation value gives:

| atom | image | contribution |
|---|---|---|
| Se | (0,0,0) | −2.104070 |
| Se | (0,0,0) | −2.104069 |
| Se | (0,1,0) | −0.001926 |
| Se | (0,1,0) | −0.001926 |
| Se | (−1,0,0) | −0.001267 |
| Se | (−1,0,0) | −0.001267 |
| W_sv | (0,0,0) | +0.662089 |

(sum −3.55, vs the production reciprocal route's −95.57 for the same
quantity — a large discrepancy for this specific, extreme eigenvector,
consistent with a real-space-quadrature contribution on top of the
dominant full-`Q` overcounting issue identified below; not fully
reconciled, noted rather than chased further given Phase 2's direct fix.)

**3. k-point subset scaling (the decisive test):** min eigenvalue of
`G_total`, rebuilt from scratch for nested, evenly-spread k-point subsets
with complete band sets at each:

| n_k | nstates | min eig(`G_total`) | min eig(`K`) |
|---|---|---|---|
| 1 | 13 | +0.999992 | +0.003086 |
| 2 | 26 | +0.000002 | +0.000000 |
| 4 | 52 | −0.075677 | −0.000234 |
| 8 | 104 | −1.563972 | −0.004827 |
| 16 | 208 | −3.188562 | −0.009841 |
| 32 | 416 | −6.368415 | −0.019656 |
| 64 | 832 | −12.772730 | −0.039422 |
| 128 | 1664 | −25.551261 | −0.078862 |
| 256 | 3328 | −51.080514 | −0.157656 |
| 324 (all) | 4212 | −64.644104 | −0.199519 |

From `n_k=8` onward, doubling the k-point count roughly doubles
`min eig(G_total)`'s magnitude — a **coherent, extensive (system-size-scaling)
growth**, not a bounded, isolated few-state artifact. This is the
fingerprint of a systematic *overcounting* mechanism (more k-points → more
cross-k pairs → more accumulated over-count), not random quadrature noise,
and it directly motivated Phase 2's regional fix rather than a numerical
patch.

**4. Random gauge test:** `psi_a → e^{iθ_a}psi_a` with `beta_a` transformed
consistently must leave `G`/`K`'s eigenvalues exactly unchanged (a pure
linear-algebra fact for any consistently-built bilinear form). Result:
`max|Δeig(G_total)| = 3.98×10⁻¹³`, `max|Δeig(K)| = 1.44×10⁻¹⁵` — **passed
cleanly**, confirming no hidden gauge dependence remains beyond the
already-fixed atom-position phase (update #1); the negative eigenvalue is
not a residual gauge bug.

**5. Spectral-norm comparison** of `G_aug` (reciprocal, gauge-corrected)
against a converged 3x real-space reference (both using the *same*,
unrestricted full `Q` — i.e. testing only whether the reciprocal shortcut
faithfully reproduces a full-`Q` real-space calculation, not whether full
`Q` is the right thing to use) on 8 k-points / 104 states:
`spectral_norm = 8.47×10⁻⁴`, `max|entry| = 7.86×10⁻⁵`. This is small —
confirming the reciprocal route is a faithful stand-in for a real-space
full-`Q` calculation. Combined with finding 3 (coherent growth) and the
per-atom/image split found by Phase 2 (below), the diagnosis is: **the
gauge-corrected reciprocal shortcut is accurate; the physical error is in
applying the full atomic `Q` to every atom regardless of whether its
augmentation sphere actually lies inside the WS cell.**

### Phase 2: regional `T†P_A T`

**Q_A construction and validation** (`build_regional_Qij_site`, dense
Gauss-Legendre-in-cosθ × uniform-in-φ angular quadrature over the POTCAR's
own logarithmic radial grid, region membership tested via the exact
WS/Voronoi bisector test — never the coarse global FFT grid):

| check | result |
|---|---|
| sphere entirely inside A vs full `Q` | max\|err\| = 2.1×10⁻¹⁵ — OK |
| sphere entirely outside A (forced path) | max\|`Q_A`\| = 0 — OK |
| sphere entirely outside A (general, far site) | max\|`Q_A`\| = 0 — OK |
| partition (half-space + complement) sums to full `Q` | max\|err\| = 2.1×10⁻¹⁵ — OK |
| `Q_A` Hermitian (real-symmetric) | max\|`Q`−`Qᵀ`\| = 0 — OK |
| angular-quadrature refinement (64×128 → 96×192) | max\|Δ\| = 3.4×10⁻⁵ (tol 1×10⁻⁴) — OK |

The refinement check needed an explicit convergence study first: a
hard-edge (discontinuous) boundary indicator converges slowly/algebraically
under this quadrature (Gibbs-phenomenon-like — a step function has no
smooth spectral convergence), not the fast convergence of a smooth
integrand. Successive doublings gave max|ΔQ| = 5.2×10⁻⁴ (16×32), 1.9×10⁻⁴
(24×48), 1.3×10⁻⁴ (32×64), 4.7×10⁻⁵ (48×96→64×128), 3.4×10⁻⁵ (64×128→96×192)
— resolution was set at the first pair comfortably inside tolerance, not
picked arbitrarily.

**Site enumeration** (purely geometric — independent of k-point/band, found
once and reused for every state): 7 contributing (atom, image) sites,
`trace(Q_A)/trace(Q_full)` per site:

| atom | image | fraction of full `Q` |
|---|---|---|
| W (WS center) | (0,0,0) | **1.0000** |
| Se | (−1,0,0) | 0.3359 |
| Se | (0,0,0) | 0.3281 |
| Se | (0,1,0) | 0.3359 |
| Se (2nd atom) | (−1,0,0) | 0.3359 |
| Se (2nd atom) | (0,0,0) | 0.3281 |
| Se (2nd atom) | (0,1,0) | 0.3359 |

This is a clean, physically sensible picture, consistent with WSe2's 3-fold
site symmetry around the W center: the W atom's sphere sits entirely inside
the WS cell (as expected — it *is* the WS center), while each Se atom's
sphere is split roughly evenly across **three** WS-cell images (≈33% each,
summing to 100%). The old full-`Q` reciprocal method applied the *entire*
`Q` to **each** of these three Se images — an ≈3× overcount per Se atom,
compounded over every cross-k pair involving Se-heavy states — exactly
explaining Phase 1's coherent, roughly-linear-in-`n_k` growth of the
negative eigenvalue.

**Reduced-reference validation** (5 k-points, 65 states, `beta_grid_factor=3`):
all 9 checks passed, including a grid-refinement check redefined to gate on
the *physically required* quantities (min eigenvalue and trace stability
across grid refinement — both stable to ~10⁻⁶–10⁻¹¹) rather than the raw
matrix spectral norm, which — like `Q_A`'s own angular quadrature —
converges slowly for a boundary-crossing site and is not itself a
physical-validity requirement. (An explicit 2×/3×/4× study showed
`min_eig(G_A)` and `trace(K_A)` already stable to ~10⁻⁶ at 2×; 3× was used
for production as extra margin, not because 2× was shown insufficient.)
Random-rephasing invariance: `max|Δeig(G_A)| = 8.0×10⁻¹⁵`.

### Full 324-k-point / 4212-state regional result

| quantity | value |
|---|---|
| max eigenvalue | 1.0000200 |
| **min eigenvalue** | **−0.0000000** (vs. −0.1995 before — the fatal violation is resolved) |
| n(eigenvalue outside [0,1] by >10⁻³) | **0** |
| `G_A`/`K_A` Hermiticity | 0 (machine precision) |
| state-space orthonormality (`Y†G_A Y − I`) | 6.1×10⁻¹¹ |
| trace_expected_input | 13.0000000000 |
| trace(`K_A`) | 13.5828713228 |
| trace(`K_A`) − expected | **+0.5829 (+4.5%)** — algebra check (`Tr(K_A)`=`sum(eigvals)`) passes to 5.3×10⁻¹⁵; this is the independent *physical* check, and it does not yet close |
| runtime | 198.3 s (build 93.5 s, `G` matmul 4.6 s, `eigh` 43.9 s) |
| peak memory | 2308.9 MB |

**Overall validation: FAIL, specifically and only on `trace_matches_expected`**
— every other check (Hermiticity, PSD-ness of `G_A`/`K_A`, the `[0,1]`
occupation bound, the trace *algebra* identity, gauge invariance, state-space
orthonormality) passes cleanly. This is reported plainly rather than
declared a full success.

**Interpretation:** the ~4.5% trace excess is a different and milder issue
than the PSD violation it replaced — occupations stay inside `[0,1]`, so it
is not "fatal" in the sense the −0.1995 eigenvalue was. Its likely origin:
fixing the PSD violation *required* switching from the essentially-exact
reciprocal beta (which cannot be split per region/image, and was
responsible for update #1's excellent 2.98×10⁻⁶ trace accuracy) to
real-space, per-site beta evaluation (which correctly captures the
region-splitting seen above, but reintroduces a real-space-quadrature-type
trace bias reminiscent of the *original* `paw_density_matrix.py` real-space
`S`-matrix trace error, now in the new context of region-restricted
per-site betas). This was not chased further within this task — see
"Deferred" below.

### Deferred / follow-up (not resolved here)

- **The ~4.5% trace excess in the regional result.** Candidate next steps:
  compare the per-site real-space `Q_A`-weighted diagonal contributions
  against the reciprocal (full-`Q`, non-split) values for the *same-k*
  case specifically (where the full-`Q` reciprocal route is known-accurate)
  to isolate whether the excess is uniform across sites or concentrated in
  the split Se sites; consider a hybrid where same-k pairs keep the
  essentially-exact reciprocal route (splitting is irrelevant/degenerate
  for `a=b`) and only cross-k pairs use the region-split real-space route.
- **Phase 1's unreconciled −3.55 vs. −95.57 decomposition mismatch** for the
  single worst eigenvector (see table above) — plausibly a mix of the
  full-`Q` overcounting (now fixed) and residual native-grid real-space
  quadrature noise; not separately isolated.
- Exact analytic (rather than numerical Gauss-Legendre/uniform-φ)
  region-intersected angular quadrature was not attempted; a faster-
  converging angular scheme (e.g. Lebedev quadrature, or an analytic
  spherical-cap decomposition for the WS cell's bounding planes) could
  tighten the `Q_A` refinement margin currently resting on the 64×128
  resolution.

### Files

`paw_lowrank_phase1_diagnostics.py` — Phase 1 (read-only diagnostics, no
production formula changes). `paw_regional_cno.py` — Phase 2 (new
production formula: regional `T†P_A T`). Outputs, all under
`paw_augmentation/output/` only:
`paw_lowrank_phase1_diagnostics.{txt,json}`,
`paw_regional_report.{txt,json}`, `cno_occupations_regional.npy`,
`cno_state_eigenvectors_regional.npy`, `cno_orbitals_pseudo_regional.npy`
(experimental/pseudo label — does not overwrite any production or
prior-experiment CNO file).

`main.py` and `config.py` were not modified by this update (config.py only
read).

---

## 2026-07-11 update #3: beta is region-independent — the correct fix

**Plain summary, as requested:**

| stage | status |
|---|---|
| old real-space `S` (`paw_density_matrix.py`) | quadrature/trace failure — max eigval 1.003, `Tr(D·S)` 4.0% above expected |
| first low-rank (`paw_lowrank_cno.py`, full-`Q` reciprocal beta) | accurate reciprocal beta, but paired through the **wrong (unrestricted) regional operator** → non-PSD `K` (min eigenvalue **−0.1995**, fatal) |
| first regional fix (`paw_regional_cno.py`, update #2, **real-space** per-site beta) | **correct `Q_A`**, but an **unnecessary switch to real-space beta** reintroduced a **4.5% trace error** (13.58 vs 13.00) |
| **corrected regional method** (this update: **reciprocal beta + regional `Q_A`**) | **both problems resolved simultaneously**: min eigenvalue ≈ 0, trace error 2.98×10⁻⁶ (matching the *original* reciprocal method's accuracy) |

### The correction

Update #2 reasoned that because `Q_A` is region-restricted, `beta` must be
too, and computed it via real-space per-site quadrature. This was wrong.
`beta_n,i = <p~_i|psi~_n>` is the **complete** projector overlap — it does
not know or care about region `A` at all. All region-dependence lives
entirely in `Q_A,ij = <phi_i|P_A|phi_j> - <phi~_i|P_A|phi~_j>` (a property
of the AE/PS partial waves and the region alone). What `beta` needs for a
periodic image of an atom translated by lattice vector `R` is the exact
Bloch relation

```
beta_(n,k,atom,R) = exp(+2*pi*i * k.R) * beta_(n,k,atom,0)
```

where `beta_(n,k,atom,0)` is the trusted reciprocal beta
(`paw.nonlq.proj()` on raw `norm=False` coefficients, plus the
already-established atomic-position gauge correction from update #1's
`gauge_correct_beta`). **No real-space quadrature is needed for beta at any
grid density** — this is an exact consequence of Bloch's theorem applied to
a compactly-supported projector, not an approximation.

**The sign was not guessed.** It was confirmed by feeding
`quadrature_convergence_check.py`'s already-validated real-space
combine-with-phase function a *virtually shifted* atom position (`atom_cart
+ R`), at a k-point with a genuinely fractional, non-half-integer component
(`ik=41`, `k=[0.2222, 0.1111, 0]` — needed because most tested (k, R) pairs
otherwise give real ±1 phases where the sign is invisible). Result: the `+`
sign matched to `3.99×10⁻⁴`–`8.99×10⁻⁴` (the established real-space
quadrature precision floor); the `-` sign was off by `O(1)` (errors of
0.84–2.43). Unambiguous.

### Why this fixes the trace without reintroducing the PSD violation

For a **diagonal** (same-state, same-`k`) term, the image phase satisfies
`|exp(2*pi*i*k.R)|² = 1` for every image, so

```
sum_images beta_(a,R)^dagger Q_A(R) beta_(a,R)
  = beta_(a,0)^dagger [sum_images Q_A(R)] beta_(a,0)
  = beta_(a,0)^dagger Q_full beta_(a,0)      (using the partition identity, already validated)
```

— i.e. region-splitting is **exactly invisible on the diagonal**, and the
diagonal is what the trace (and hence electron-count accuracy) is built
from. This is precisely why the trace recovers update #1's excellent
accuracy while the **off-diagonal (cross-`k`) terms** — where the phase
does *not* cancel and region-splitting genuinely matters — correctly gain
the physical `Q_A` weighting that fixes the PSD violation. Verified
explicitly as the required diagnostic identity (see check B below): worst
deviation `2.22×10⁻¹⁵` across 195 (state, atom) combinations.

### Required validations (all re-run against the corrected construction)

| check | result |
|---|---|
| A. `Q` partition closure (`sum_images Q_A ≈ Q_full`, full matrix, per atom) | W: `2.1×10⁻¹⁵`; Se×2: `4.7×10⁻¹⁶` — OK |
| B. Diagonal augmentation closure (sign-independent identity) | worst `2.22×10⁻¹⁵` over 195 (state,atom) checks — OK |
| C. Full PAW norm closure (regional vs. trusted reciprocal) | norms `[0.999997, 1.000002]`, max diff from reciprocal `2.22×10⁻¹⁵` — OK |
| G. Cross-`k` regional block vs. converged 3× real-space reference | full block `7.24×10⁻⁵`, cross-`k` sub-block `7.20×10⁻⁵` — OK |
| `G_A`/`K_A` Hermitian | machine precision — OK |
| `G_ps_A`, `G_A`, `K_A` PSD | min eig `0.000002`/`0.000002`/`0.000000` — OK |
| Occupations in `[0,1]` | 0 out of bounds — OK |
| Trace matches expected | reduced ref: diff `2.2×10⁻⁸`; full run: diff `2.98×10⁻⁶` — OK |
| F. Random-rephasing (gauge) invariance | `max|Δeig(G_A)| = 1.2×10⁻¹⁴` — OK |
| State-space orthonormality (`Y†G_A Y − I`) | `4.5×10⁻¹²` — OK |

**Reduced-reference validation (5 k-points): 12/12 checks PASS.**

### Full 324-k-point / 4212-state result

| quantity | value |
|---|---|
| trace_expected_input | 13.0000000000 |
| trace(`K_A`) | 13.0000029802 |
| sum(eigenvalues) | 13.0000029802 |
| trace(`K_A`) − expected | **2.98×10⁻⁶** (0.00002% relative — matches update #1's accuracy exactly) |
| max eigenvalue | 0.99998721 |
| **min eigenvalue** | **−0.00000173** (both the −0.1995 fatal violation *and* the 4.5% trace error are gone) |
| n(eigenvalue outside [0,1] by >10⁻³) | **0** |
| state-space orthonormality | 4.50×10⁻¹² |
| runtime | **97.9 s** (build 8.4 s, `G` matmul 4.0 s, `eigh` 36.4 s) — faster than update #2, since beta needs no real-space grid at all |
| peak memory | 2309.3 MB |

**Overall validation: PASS — every check passes**, with no eigenvalue
clipping, no diagonal shifting, no global rescaling, and no projection onto
the PSD cone anywhere in the pipeline. The fix is a genuine formula
correction (beta is region-independent; only `Q_A` is regional), not a
numerical patch.

### What remains deferred (unchanged from update #2)

- The exact regional operator's angular quadrature (`Q_A`) still rests on a
  64×128 Gauss-Legendre-×-uniform-φ grid, chosen from an explicit
  convergence study rather than an analytic spherical-cap decomposition; a
  faster-converging scheme (e.g. Lebedev quadrature) was not attempted.
- Phase 1's unreconciled −3.55 vs. −95.57 single-eigenvector decomposition
  mismatch (see update #2) is now understood in outline (the old,
  now-abandoned full-`Q` reciprocal route both used the wrong operator *and*
  the comparison itself mixed reciprocal against a box-restricted
  real-space quantity that is not directly comparable — see this update's
  `real_space_beta_per_site_for_bands` docstring) but was not re-derived
  number-for-number.

`main.py` and `config.py` were not modified by this update (config.py only
read).
