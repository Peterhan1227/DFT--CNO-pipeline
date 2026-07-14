# Known issues flagged by this diagnostics package (not fixed here)

Per the task that produced this package: flag, do not fix.

## main.py's SOC branch omits `occ` from the rho accumulation

**Location:** `main.py:260-267`.

```python
if LSORBIT:
    cg_dn = np.zeros_like(cg)
    cg[:, gx, gy, gz]    = Ck[:, :nG]
    cg_dn[:, gx, gy, gz] = Ck[:, nG:]
    psi_up = _to_psi(np.fft.ifftn(cg,    axes=(1, 2, 3)) * np.sqrt(Nr), k_frac)
    psi_dn = _to_psi(np.fft.ifftn(cg_dn, axes=(1, 2, 3)) * np.sqrt(Nr), k_frac)
    rho += wk * (psi_up.T @ (psi_up).conj()
               + psi_dn.T @ (psi_dn).conj())          # <-- no `occ` weight
```

Compare the non-SOC branch three lines later:

```python
cg[:, gx, gy, gz] = Ck
psi = _to_psi(np.fft.ifftn(cg, axes=(1, 2, 3)) * np.sqrt(Nr), k_frac)
rho += wk * (psi.T @ (occ[:, None] * psi).conj())     # <-- occ IS applied
```

`bands` (and therefore which columns of `Ck` end up in `psi_up`/`psi_dn`) is
still selected via `occ_all > occ_tol` (main.py:246-247), and `occ` itself is
still computed (including the `>1.5` halving check, main.py:248-249) — but in
the `LSORBIT` branch that `occ` array is never referenced again. Every band
that clears the `occ_tol` inclusion threshold is accumulated into `rho` with
an implicit weight of exactly 1, regardless of its actual occupation number.

This is a no-op (i.e. harmless) whenever every included band's true
occupation is already ~1 — the ordinary insulator/semiconductor case with
`RESTRICT_TO_FERMI_WINDOW=False`, or whenever `RESTRICT_TO_FERMI_WINDOW=True`
(which already forces `occ = np.ones(...)` for every LSORBIT setting,
main.py:241). It is **not** a no-op for a metal/semimetal under partial-
occupation (Fermi-Dirac/Gaussian smearing) with `RESTRICT_TO_FERMI_WINDOW=
False` — exactly the regime `LSORBIT=True` calculations are most likely to
be used for (band/spin-orbit physics near a Fermi surface).

### Concrete evidence this is live, not theoretical

`Data/CoSn/WAVECAR` (the only SOC dataset in this repo) has genuinely
fractional occupations under `occ_tol=1e-6`:

- 29 of its 60 k-points have at least one band with `occ_tol < occ < 1 -
  1e-3`.
- 50 (k, band) instances total are fractionally occupied.
- The largest deviation from full occupation among these is
  `|occ - 1| ≈ 1.0` (i.e. some included bands have occupation close to 0,
  not close to 1) — see
  `diagnostics/output/test_fixed_k_gram__CoSn.json`'s
  `n_fractional` / `gram_raw_fractional_subset` fields, and
  `diagnostics/_common.py:occupied_bands_split` / `pick_representative_kpoints`
  for how this was found (`test_fixed_k_gram.py` deliberately seeks out the
  k-point with the most fractionally-occupied bands as one of its
  representative sample points).

If `main.py` were run today with `LSORBIT=True` against `Data/CoSn`, every
one of those 50 (k, band) instances would be accumulated into `rho` with
weight 1 instead of its true (possibly near-zero) occupation, over-counting
the electron count and distorting the resulting CNO occupation spectrum in a
way that depends on the smearing/Fermi-surface details — not obviously
bounded, and not caught by the `[0, 1]` eigenvalue sanity check alone (an
over-weighted rho can still land eigenvalues inside `[0,1]` while being
quantitatively wrong).

### Suggested direction for whoever fixes this (not done here)

Mirror the non-SOC branch: weight both spinor channels by `occ`, i.e.
`rho += wk * (psi_up.T @ (occ[:, None] * psi_up).conj() + psi_dn.T @ (occ[:, None] * psi_dn).conj())`.
This diagnostics package deliberately does not make this change — see the
task constraints this package was built under (`main.py`'s physics is
out of scope for this change).
