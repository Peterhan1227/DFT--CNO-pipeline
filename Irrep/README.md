# Irrep — Diamond Si (SG 227)

## Workflow

### 1. Identify high-symmetry k-points

```
irrep -config high_symmetry.yml > output/high_symmetry_points.out
```

### 2. Build VASP KPOINTS file

```
python scripts/build_irrep_kpoints.py output/high_symmetry_points.out
```

### 3. Run the irrep calculation

```
irrep -config irrep_run.yml
```

Output JSON: `output/test_si_irrep.json`  
Note: `trace.txt`, `irreps.dat`, `irreptable-template` are written to the working directory by IrRep and cannot be redirected.

---

## Stage 2: Extract symmetry matrices

`scripts/extract_symmetry_matrices.py` loads the DFT wavefunctions, identifies
irreps, and exports the full symmetry-operation matrices for every degenerate
energy block at each configured high-symmetry k-point.

**Run from the `Irrep/` directory:**

```
python scripts/extract_symmetry_matrices.py symm_config.yml
```

**Outputs** (written to `output/`):

| File | Contents |
|------|----------|
| `symmetry_matrices.npz` | All complex matrices as numpy arrays |
| `symmetry_matrices.json` | Full metadata + matrices as `[real, imag]` pairs |
| `symmetry_matrices_report.txt` | Human-readable per-block summary |

**Basis modes** (set in `symm_config.yml → basis.mode`):

- `raw` — retain the VASP eigenstate basis
- `diagonalize` — diagonalise one chosen symmetry operation per block;
  apply the **same** U to every other operation in that block
  (`primary_symop: auto` picks the non-identity op with the most distinct
  eigenvalues; supply a 1-based BCS index to override)

**CLI overrides:**

```
python scripts/extract_symmetry_matrices.py --dump-default-config
python scripts/extract_symmetry_matrices.py symm_config.yml --mode raw
python scripts/extract_symmetry_matrices.py symm_config.yml --primary-symop 18
```

---

## Optional: post-processing character consistency check

`scripts/check_irrep_characters.py` is a standalone diagnostic that verifies the
character-table assignment after the main calculation has finished.  It reads only
the completed JSON; it does **not** re-read WAVECAR and does **not** call
`get_irreps_from_table()`.

It independently reconstructs the character vector for each degenerate block from
raw `IrrepTable` data using the patched transformation formula:

```
chi_alpha_dftcell[isym] = raw_char[isym].conj() * sign * exp(-2πi dt · k_table)
```

and compares against the DFT-derived characters stored in the JSON.

**Run it:**

```
python scripts/check_irrep_characters.py output/test_si_irrep.json
```

Options:

```
--kpoint-index N    select k-point by 0-based index (default: auto-detect W)
--tolerance T       absolute tolerance for |chi_table - chi_DFT| (default 1e-6)
```

Exits 0 if all blocks pass, 1 if any block fails.

---

## Patch warning

The W-point complex-character fix lives in the conda environment.
If `irrep` is upgraded, re-apply:

```
cd C:\Users\hanziruopeter\miniconda3\envs\irrep\Lib\site-packages
patch -p1 < path\to\Irrep\scripts\fix_complex_chars.patch
```

Run `diagnostics/test_w_complex_chars.py` to confirm the fix is in place.
