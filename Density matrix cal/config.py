# ── material selection ────────────────────────────────────────────────────────
# Change this to switch between datasets.  Each material must have its own
# subfolder under Data/ with the same internal structure (WAVECAR, POSCAR, …
# and an output/ subdirectory that the scripts create automatically).
MATERIAL = "Si"
LSORBIT  = False   # True for non-collinear (SOC) calculations, False otherwise

# ── output directory ──────────────────────────────────────────────────────────
# All scripts write to  Data/<MATERIAL>/output/<OUTPUT_SUBDIR>/
# Change this to keep different parameter runs side-by-side, e.g. "original",
# "isym0", "ws_bond".  Never leave it empty — every run needs a name.
OUTPUT_SUBDIR = "testing"

# ── density matrix settings ───────────────────────────────────────────────────
ISPIN = 1                      # 1 = spin-up / non-spin-polarised,  2 = spin-down

RESTRICT_TO_FERMI_WINDOW = False   # True = include only bands within ±FERMI_WINDOW_EV of EFERMI
EFERMI          = 5.9837           # Fermi energy (eV); used only when RESTRICT_TO_FERMI_WINDOW = True
FERMI_WINDOW_EV = 5.0              # half-width of the energy window (eV)

# ── Wigner-Seitz cell settings ────────────────────────────────────────────────
USE_WS_CELL = True

# Manual WS center. Do not implement automatic symmetry/Wyckoff/bond selection yet.
# Later another module will compute this coordinate and pass it in.
WS_CENTER_COORD_TYPE = "fractional"   # "fractional" or "cartesian"
WS_CENTER = [0.875, 0.875, 0.875]     # user-editable placeholder

WS_TRANSLATION_SEARCH_RANGE = 3
