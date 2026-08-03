"""
plot_cno_eigenvalues.py

Plot the eigenvalues of the single-particle density matrix (CNO occupations)
sorted from largest to smallest.  CNO occupations are not Bloch-state
occupations and need not follow a step function — they show how the total
electron density is distributed across natural orbitals.
"""
from pathlib import Path
import os
from config import MATERIAL, OUTPUT_SUBDIR

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ── paths ──────────────────────────────────────────────────────────────────────
base_dir   = Path(__file__).resolve().parent
output_subdir = os.environ.get("CNO_OUTPUT_SUBDIR", OUTPUT_SUBDIR)
output_dir = base_dir / "Data" / MATERIAL / "output" / output_subdir
occ_file   = output_dir / "cno_occupations.npy"

# ── load ───────────────────────────────────────────────────────────────────────
occ = np.load(occ_file)   # already sorted largest → smallest by Wavecar_to_Coeff.py
n   = len(occ)

print(f"Loaded {n} CNO eigenvalues from {occ_file}")
print(f"  Max occupation       : {occ[0]:.6f}")
print(f"  Min occupation       : {occ[-1]:.6e}")
print(f"  Sum                  : {occ.sum():.4f}")
n_sig = int(np.sum(occ > 1e-4))
print(f"  Significant (> 1e-4) : {n_sig}")

# ── figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# — left panel: full spectrum —
ax = axes[0]
ax.plot(np.arange(n), occ, '.', color='royalblue', markersize=2)
ax.axhline(0.0, color='gray', lw=0.5, ls=':')
ax.set_xlabel("CNO index (sorted by occupation)")
ax.set_ylabel("Occupation")
ax.set_title("Full CNO eigenvalue spectrum")
ax.set_xlim(0, n - 1)
ax.set_ylim(-0.05, float(occ[0]) * 1.1)

# — right panel: zoom in on significant eigenvalues —
n_zoom = min(max(n_sig + max(n_sig // 5, 5), 10), n)
ax2 = axes[1]
ax2.plot(np.arange(n_zoom), occ[:n_zoom], '.-', color='royalblue', markersize=4)
ax2.axhline(0.0, color='gray', lw=0.5, ls=':')
ax2.set_xlabel("CNO index")
ax2.set_ylabel("Occupation")
ax2.set_title(f"Zoom: top {n_zoom} eigenvalues")
ax2.set_xlim(-0.5, n_zoom - 0.5)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.suptitle("Density matrix eigenvalues (CNO occupations)", fontsize=12)
plt.tight_layout()

out_path = output_dir / "cno_eigenvalue_spectrum.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
