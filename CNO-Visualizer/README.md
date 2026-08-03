# CNO-Visualizer

Interactive viewer for **complex Cell Natural Orbitals (CNOs)**.  The CNO is
rendered as a three-dimensional density isosurface whose surface color encodes
the complex phase via a cyclic colormap.  The viewer also shows the Wigner–Seitz
cell, primitive-cell boundaries, atoms and bonds, and periodic copies of the
crystal and orbital, with interactive controls for the CNO index, isovalue,
phase offset, opacity, and object visibility.

License: **not yet set** — the project ships without a `LICENSE` file today.

> CNO-Visualizer does **not** perform any DFT calculation, density-matrix
> diagonalization, or CNO construction.  It is purely a viewer that consumes a
> single self-contained `.npz` file produced upstream.  The producer is out of
> scope for this repository.

## Input contract

A single file, `cno_visualizer_data.npz`, with the fields below.  Strings are
stored as zero-dimensional NumPy arrays (no pickle is needed to read them).

| field                   | dtype              | shape           | notes                                                                |
|-------------------------|--------------------|-----------------|----------------------------------------------------------------------|
| `format_version`        | `<U…`              | `()`            | Must be `"cno-visualizer-v1"`.                                       |
| `cno_values`            | `complex128`       | `(n_cno, Nsample)` | Rows are CNOs; ordinary grids have `Nsample = Nx*Ny*Nz`, while finite-volume WS data may retain additional unwrapped boundary images. |
| `cno_occupations`       | `float64`          | `(n_cno,)`      | Eigenvalue / occupation per CNO.                                     |
| `cno_indices`           | `int64`            | `(n_cno,)`      | Original CNO numbers as shown to the user.                           |
| `grid_shape`            | int                | `(3,)`          | `(Nx, Ny, Nz)`.                                                      |
| `lattice`               | `float64`          | `(3, 3)`        | Rows are lattice vectors; Cartesian in Å.                            |
| `atom_symbols`          | string             | `(natoms,)`     | Element symbol per atom.                                             |
| `atom_numbers`          | `int64`            | `(natoms,)`     | Atomic number per atom.                                              |
| `atoms_frac`            | `float64`          | `(natoms, 3)`   | Fractional coordinates.                                              |
| `atoms_cart`            | `float64`          | `(natoms, 3)`   | Cartesian coordinates in Å.                                          |
| `ws_enabled`            | `bool`             | `()`            | If `True` the file must also contain the fields below.               |
| `points_cart`           | `float64`          | `(Nr, 3)`       | WS-mode evaluation points in Cartesian.                              |
| `points_frac_cont`      | `float64`          | `(Nr, 3)`       | Continuous fractional coords (not wrapped).                          |
| `base_indices`          | `int64`            | `(Nr, 3)`       | Original FFT indices.                                                |
| `translations`          | `int64`            | `(Nr, 3)`       | Integer lattice translations so `actual = base + n * grid_shape`.    |
| `ws_center_cart`        | `float64`          | `(3,)`          | WS cell center in Cartesian coordinates.                             |
| `ws_center_frac`        | `float64`          | `(3,)`          | (alias `ws_center_frac_wrapped` is accepted).                        |

Validation is strict — any contract violation raises a clear `CNODataError`.

## Installation

This project does not import any upstream package and does not depend on the
exporter being installed.  It only needs the `.npz`.

```bash
# A. From the project directory (editable install — recommended during development).
pip install -e .

# B. With tests:
pip install -e ".[dev]"

# C. From requirements.txt (pinned ranges only, identical to pyproject.toml):
pip install -r requirements.txt
```

Python ≥ 3.10.  The package pins `pyvista>=0.43,<1.0` and `vtk>=9.2` because the
slider / key-event APIs used here have stabilized in that range.

## Launching

```bash
cno-visualizer --data "/path/to/cno_visualizer_data.npz"
# or
python -m cno_visualizer --data "/path/to/cno_visualizer_data.npz"
```

### Clickable launcher (no terminal)

Double-click **`Launch CNO-Visualizer.bat`** — it opens a file picker for the
`.npz` and starts the viewer with no console window.  Right-click it →
*Send to ▸ Desktop (create shortcut)* to pin it; the shortcut's *Properties ▸
Change Icon* lets you give it a custom icon.  Equivalent CLI: `cno-visualizer-gui`
(also accepts all the flags below).  The `.bat` uses the `physics` conda env at
`%USERPROFILE%\miniconda3\envs\physics`; edit the path inside if yours differs.

The data path may contain spaces and works on Windows.  All other options are
optional:

| flag                              | meaning                                       |
|-----------------------------------|-----------------------------------------------|
| `--cno INTEGER`                   | Local CNO row index to show at startup.       |
| `--iso FLOAT`                     | Initial isovalue fraction (default `0.05`).   |
| `--display phase|density|real|imaginary` | Initial surface coloring (default `phase`).|
| `--view crystal|ws`               | `crystal` (default): periodic isosurface over `--replicate` cells, matching the cube. `ws`: one orbital clipped to the Wigner–Seitz cell. |
| `--replicate NX NY NZ`            | Number of cells along each lattice vector.    |
| `--no-atoms / --no-bonds / --no-cells / --no-ws` | Start with that object hidden. |
| `--background dark|light`         | Background color (default `dark`).            |
| `--screenshot-dir PATH`           | Where `S` writes PNGs (default `./screenshots`). |
| `--camera PATH`                   | JSON camera-state to load at startup.         |
| `--off-screen`                    | Build the scene headlessly and save one PNG.  |

## Mouse and keyboard

| input | action                                  |
|-------|-----------------------------------------|
| LMB drag | Rotate                               |
| MMB / Shift+LMB | Pan                            |
| Scroll | Zoom                                   |
| `P` / `D` / `R` / `I` | Display mode: phase / density / real / imaginary |
| `V` | Toggle crystal ↔ WS view                  |
| `A` / `B` / `C` / `W` / `M` / `X` | Toggle atoms / bonds / cells / WS cell / center marker / axes |
| `S` | Save screenshot                          |
| `0` | Reset camera                             |
| `H` | Show / hide help overlay                 |

Sliders along the left edge control: CNO index (integer, snaps to whole values),
isovalue fraction, global phase offset (`0` to `2π`), and surface opacity.  The
status overlay (top-left) shows the **absolute isovalue** `fraction × max(|ψ|²)`;
because the stored CNOs satisfy `Σ|ψ|² = 1` this number equals the level you would
type into VESTA for the corresponding `.cube`.

### Coloring (auto-selected)

By default the viewer **auto-detects** each CNO:

* **real CNOs** (Γ-point / real-Hamiltonian → real eigenvectors, phase only `0`/`π`)
  are colored by **signed amplitude** `Re ψ` with a diverging `coolwarm` map
  centered at zero — clean solid lobes, sign visible at low isovalues.
* **complex CNOs** are colored by **wrapped phase** using the perceptually-uniform
  cyclic colormap `colorcet` **CET_C7** (far more legible than `twilight_shifted`).

Pressing `P`/`D`/`R`/`I` pins an explicit mode and turns auto-selection off.

## Density isosurfaces and phase coloring

The geometry of the displayed surface is the isosurface

```
rho(r) = |psi(r)|^2 = rho_iso = fraction * max(|psi|^2).
```

The color on that surface is the **wrapped complex phase**

```
phi(r) = mod( arctan2(Im psi, Re psi) + offset, 2*pi ).
```

Because the phase is a cyclic quantity, the colormap is `twilight_shifted`, with
fixed range `[0, 2*pi]`.  The color at `0` and `2π` is identical so users can
visually distinguish smooth phase variation from genuine wrap discontinuities.
A noncyclic blue-to-red colormap would visually fabricate a discontinuity along
the line `phi == 0` and is deliberately avoided.

### Why a global phase offset

The eigenvector ψ is only defined up to a global gauge:

```
psi(r)  ↦  e^{i * alpha} * psi(r)
```

leaves the physics unchanged.  The phase-offset slider implements exactly this
transformation.  It updates surface colors only — it never re-runs the contour
or mutates `cno_values`.

`set_phase_reference(psi_value)` (used by the future point-picking UI) returns
the offset that maps a chosen reference point's displayed phase to zero.  Points
where `|psi|` is numerically zero are rejected because the phase is undefined.

## How the orbital is rendered

Ordinary data is contoured on a **regular structured grid** that follows the
(possibly non-orthogonal) lattice, matching what a cube viewer (VESTA) shows.
Finite-volume WS data with distinct unwrapped boundary images cannot be folded to
that grid without overwriting samples.  It is instead contoured on an unstructured
mesh of the complete saved FFT hexahedra; no exterior values are filled or averaged.

* **Primitive data** (`ws_enabled = False`) — `cno_values` is the C-order
  flattening of the `(Nx, Ny, Nz)` FFT grid; the viewer builds a
  `pyvista.StructuredGrid` over the lattice and contours `|psi|²`.
* **One-to-one WS data** (`ws_enabled = True`, one saved row per FFT node) – the
  viewer inverts `base_indices` into a lookup table and builds the same regular grid.
* **Expanded finite-volume WS data** – every saved row stays at its true unwrapped
  FFT position.  The viewer creates only hexahedra with all eight saved vertices;
  it does not use Delaunay tetrahedra or a point-cloud surface.

The contour interpolates `psi_real` and `psi_imag` to the surface vertices, and
the wrapped phase is computed there — phase angles are never interpolated directly.

The **WS cell** is generated analytically from the lattice via
`scipy.spatial.HalfspaceIntersection` + `ConvexHull`, independent of the data.

## Crystal view vs. WS view (`--view`, `V`)

* **`crystal`** (default) — the periodic isosurface contoured over `--replicate`
  primitive cells, with a one-layer wrap so lobes cross cell boundaries
  seamlessly.  This is the natural crystal picture and matches the cube.
* **`ws`** – the field is **clipped to the analytic WS polyhedron** (via its active
  half-space planes), showing one localized orbital inside its own cell.  Expanded
  finite-volume CNOs use this physical regional view only and show nearby periodic
  atoms as context without replicating the CNO itself.

## Periodic replicas and bonds

`--replicate NX NY NZ` controls how many primitive cells are drawn.  The orbital
is contoured directly over the replicated grid (one mesh, opacity from the
`Opacity` slider).  Atoms and bonds are built with a one-cell **halo**: bonds are
searched over the displayed cells *plus* a surrounding layer, so atoms at a
supercell face keep their neighbours (in the diamond structure a bond partner
always lives in an adjacent cell) and the whole network is connected — no
dangling sticks and no missing bonds.

## Screenshots and camera

Press `S` (or call `viewer.screenshot()`) to write a PNG with the name

```
<material>_cno_<idx>_<view>_<mode>_iso_<fraction>.png
```

into `./screenshots/` (or `--screenshot-dir`).  The `material` label is taken
from the `.npz` file (if it carries a `material` field) or otherwise from the
filename stem.

Camera state can be saved and reloaded as JSON via `CameraState.from_plotter`
and `--camera path/to/cam.json`.

## Tests

```bash
pytest -q
```

All fixtures are synthetic; no real DFT output is required.  Tests cover loader
validation, primitive-grid axis ordering (the NumPy ↔ VTK Fortran-order gotcha
is the most common silent bug here), phase wrapping and gauge invariance, the
WS integer relation, half-space and volume properties, and contouring/empty
contour edge cases.

## Symmetry animations (`cno-anim`)

A **separate, isolated** subpackage (`cno_visualizer.animation`) renders smooth,
Manim-style eased site-symmetry rotations of the crystal cluster and Wigner–Seitz
cell about a basis center.  It only reuses the pure-geometry helpers
(`crystal`, `ws_geometry`) and never touches the interactive viewer, so it cannot
affect it.  The isosurface is intentionally not included yet.

```bash
# 3-fold rotation about [111] at the WS/bond center, shown three times (-> identity),
# with a gentle cinematic camera drift, written to MP4:
cno-anim --data "/path/to/cno_visualizer_data.npz" \
    --axis 111 --nfold 3 --steps 3 --orbit 25 --output si_c3.mp4

# A 4-fold about z as a looping GIF, slower, with the smootherstep easing:
cno-anim --data "...npz" --axis z --nfold 4 --duration 5 --easing smootherstep --output c4.gif
```

The crystal cluster (a sphere of radius `--radius` about the center) rotates while
a **full 3-D xyz coordinate frame** (Manim/3b1b style: colored x/y/z arrows at the
same scale, with ticks and labels), the WS cell, rotation axis (yellow rod) and
center stay fixed as the reference frame.  The coordinate frame is on by default —
use `--no-axes` to hide it or `--axes-numbers` to label the ticks with numbers.

A **translucent grey “ghost” copy of the original lattice** is also drawn at the
starting position (on by default; `--no-ghost` to hide).  The moving copy keeps
element colors, so you can see at a glance whether the operation maps the lattice
onto itself: at a true symmetry angle the colored atoms land back exactly on the
ghost; at a non-symmetric angle the ghost shows through.

Motion uses a rate function — `smootherstep` (default; zero velocity at both ends,
the “ease-in/ease-out” feel), or `linear`, `smoothstep`, `ease_in_out_sine`,
`rush_into`, `rush_from`, `there_and_back`.  Output is `.mp4` or `.gif`; `--show`
opens an interactive window instead.  Axes accept `x/y/z` or crystallographic
directions (`111`, `001`, `1-10`), resolved through the lattice.

## Report snapshots (`cno_visualizer.snapshot`)

For automating figures (e.g. dropping a 3-D view of a CNO next to its fatband
plot), `cno_visualizer.snapshot` renders a **VESTA-like** density isosurface in the
same look as the symmetry animation: a single solid **blue** isosurface (**no
phase**) with atoms + bonds, on the **dark gradient** background, inside a full
**xyz coordinate frame**.  No operation is applied — the structure simply turns
slowly (a turntable) → GIF/MP4 (or a still PNG):

```python
from cno_visualizer.snapshot import render_density_gif
# cno_grid is a regular (Nx, Ny, Nz) complex/real density grid
render_density_gif(cno_grid, lattice, atoms_cart, atom_symbols,
                   "cno_000_structure.gif", iso_fraction=0.5, replication=(2, 2, 2))
# or, from a loaded CNOData / .npz:
from cno_visualizer.snapshot import render_cno_gif
render_cno_gif(data, cno_index=0, output="cno.gif", iso_fraction=0.5)
```

`iso_fraction` is the level as a fraction of `max(|ψ|²)` (≈0.1–0.2 gives fuller
lobes; 0.5 gives tight cores).  `seconds`/`fps` set the spin pace (default ≈30°/s,
matching the symmetry animation; larger `seconds` = slower).  `surface_color` and
`background` are overridable.  `CNOData.from_arrays(...)` builds a viewer dataset
straight from in-memory arrays, so no `.npz` is needed.  This is a one-way hook:
external scripts import the viewer; the viewer never imports them.

## Known limitations

* The cyclic phase colormap is required for correctness, but you will see a
  thin “seam” along the line `phi == 0`/`2π`.  That is the colormap wrap, not
  a numerical artefact.
* Vertices where `|psi| < 1e-10` are coerced to the median surface phase so
  nodal lines do not appear speckled.  In strict-rendering use cases this can
  be loosened in `field.color_surface_by_phase`.
* In `phase` mode a real-valued CNO renders as essentially one color (its phase
  is `0` or `π`).  That is correct, not a bug — use `R`/`I` for signed amplitude.
* The WS view requires `base_indices` to cover every FFT grid point so the field
  can be back-mapped to a regular grid; the loader raises a clear error otherwise.
* `compute_symmetry_overlap` is a documented placeholder; computing the true
  ⟨ψ | T ψ⟩ overlap requires the full complex field on a common grid.

## Roadmap

* Symmetry animations now ship as `cno-anim` (see above).  Next: improper
  operations (mirror/inversion/roto-inversion) animated as continuous morphs, and
  optionally overlaying the orbital isosurface on the rotating cluster.
* A Trame browser frontend for remote use — not enabled yet to keep the local
  PyVista path simple.
