"""
Sanity check: reproduce the raw (un-renormalized) plane-wave norms quoted in
TASK_BRIEF.md for WSe2_mono at k-point 1, to confirm we're reading the same
WAVECAR/bands the brief's investigation used before trusting any correction
built on top of vaspwfc/paw.py.

NOTE: this script's ORIGINAL run (against live Data/WSe2_mono/WAVECAR, early
in this task's session) produced an exact match to TASK_BRIEF.md's table --
see RESULTS.md and diagnostic_results_ORIGINAL_verified_Wsv.json. Data/WSe2_
mono/WAVECAR was overwritten mid-task by a concurrent, unrelated calculation
(see RESULTS.md "data integrity incident"), so this script now reads the
frozen (post-overwrite, mismatched) snapshot instead for reproducibility of
"what this script does now" -- re-running it will NOT reproduce the table
above, by design.
"""
import sys
from pathlib import Path
import numpy as np
from vaspwfc import vaspwfc

data_dir = Path(__file__).resolve().parent / "data_snapshot" / "WSe2_mono"
wfc = vaspwfc(str(data_dir / "WAVECAR"), lsorbit=False)
ik = 1
occ_all = wfc._occs[0, ik - 1, :]
bands = np.where(occ_all > 1e-6)[0] + 1
print(f"Occupied bands at ik={ik}: {bands}")
for ib in bands:
    Cg = wfc.readBandCoeff(ispin=1, ikpt=ik, iband=int(ib), norm=False)
    E = wfc._bands[0, ik - 1, ib - 1]
    raw_norm = np.sum(Cg.conj() * Cg).real
    print(f"band {ib:2d}  E={E:8.2f} eV   raw ||psi~||^2 = {raw_norm:.3f}")
