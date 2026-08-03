# ── material selection ────────────────────────────────────────────────────────
# Change this to switch between datasets.  Each material must have its own
# subfolder under Data/ with the same internal structure (WAVECAR, POSCAR, …
# and an output/ subdirectory that the scripts create automatically).
MATERIAL = "WSe2_mono"
LSORBIT  = False   # True for non-collinear (SOC) calculations, False otherwise

# ── output directory ──────────────────────────────────────────────────────────
# All scripts write to  Data/<MATERIAL>/output/<OUTPUT_SUBDIR>/
# Change this to keep different parameter runs side-by-side, e.g. "original",
# "isym0", "ws_bond".  Never leave it empty — every run needs a name.
OUTPUT_SUBDIR = "W_center_finite_volume_native"

# Select the regional metric.  ``False`` is the direct pseudo-WAVECAR route;
# ``True`` adds the PAW regional augmentation correction.  Before enabling
# PAW, run ``check_paw_augmentation_needed.py`` for this material/WS centre.
# WSe2_mono at the W-centred WS region has augmentation spheres crossing the
# regional boundary (verified by check_paw_augmentation_needed.py), so its
# current production configuration needs the PAW route.  Set False for a
# material/centre whose pre-check says regional augmentation is unnecessary.
USE_PAW_AUGMENTATION = True

# The physical regional quadrature used by BOTH metric choices.  ``finite_volume``
# clips the sampling-lattice voxels against the continuous WS polyhedron;
# factor > 1 evaluates the same plane-wave basis on a denser grid.
WS_QUADRATURE = "finite_volume"
WS_QUADRATURE_FACTOR = 1

# ── density matrix settings ───────────────────────────────────────────────────
ISPIN = 1                      # 1 = spin-up / non-spin-polarised,  2 = spin-down

RESTRICT_TO_FERMI_WINDOW = True   # True = include only bands within ±FERMI_WINDOW_EV of EFERMI
EFERMI          = -2.229           # Fermi energy (eV); used only when RESTRICT_TO_FERMI_WINDOW = True
FERMI_WINDOW_EV = 3.0              # half-width of the energy window (eV)

# ── Wigner-Seitz cell settings ────────────────────────────────────────────────
USE_WS_CELL = True

# Manual WS center. Do not implement automatic symmetry/Wyckoff/bond selection yet.
# Later another module will compute this coordinate and pass it in.
WS_CENTER_COORD_TYPE = "fractional"   # "fractional" or "cartesian"

# WSe2_mono (1H, P-6m2/D3h): two maximal-symmetry Wyckoff centers, each with
# full D3h site symmetry (order 12). Run once per center (own OUTPUT_SUBDIR):
WS_CENTER = [1/3, 2/3, 0.5]     # Center 1: W site               -- current run
# WS_CENTER = [2/3, 1/3, 0.5]   # Center 2: Se-Se bond axis (hollow site)

WS_TRANSLATION_SEARCH_RANGE = 3
