"""
Sanity check (read-only): plot psi_{n,k}(x, 0, 0) along the first lattice
vector direction across a cell boundary, two ways, and confirm they agree
(no kink/discontinuity at the boundary):

  (1) "direct": plug x running continuously from 0 to 2*a1 straight into
      the native (k, G, C) plane-wave sum -- trivially smooth by
      construction (a finite sum of smooth exponentials), the ground truth.

  (2) "wrap + Bloch phase": for the SECOND cell (x in [a1, 2a1]), wrap the
      coordinate back into the first cell (x' = x - a1) and multiply by
      the Bloch translation phase e^{i k.a1} -- this is the SAME wrap+phase
      logic this codebase's production code uses (e.g. the WS-grid-map
      Bloch-phase reindexing in paw_regional_cno.py/cno_fatband.py), so
      this is a real test of that pattern, not a tautology.

Uses a genuinely fractional k-point (ik=41, k=[0.2222,0.1111,0], the same
one already established in this project as sign-sensitive) -- at Gamma the
Bloch phase is exactly 1 and would make a real bug invisible.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vaspwfc import vaspwfc

WAVECAR = (r"C:\Users\hanziruopeter\OneDrive\Personal files\Academic\Physics"
           r"\Research\Columbia Raquel\Density matrix cal\Data\WSe2_mono\WAVECAR")
IK = 41
NPTS = 200

wfc = vaspwfc(WAVECAR, lsorbit=False)
k1 = wfc._kvecs[IK - 1]
print(f"k1 (frac) = {k1}")

occ = wfc._occs[0, IK - 1, :]
iband = int(np.where(occ > 0.5)[0][-1]) + 1
print(f"band {iband}  occ={occ[iband-1]:.4f}  E={wfc._bands[0, IK-1, iband-1]:.4f} eV")

G1 = wfc.gvectors(IK)
C1 = wfc.readBandCoeff(ispin=1, ikpt=IK, iband=iband, norm=True)

a1 = wfc._Acell[0]
Bcell = wfc._Bcell
TPI = 2 * np.pi
kg1_cart = (k1[None, :] + G1) @ (TPI * Bcell)   # (nG, 3)

# Bloch phase for one a1 translation: e^{i k.a1}. In fractional units,
# k.a1 (Cartesian) = 2*pi*k_frac . (1,0,0) exactly (a1 = 1*a1 + 0*a2 + 0*a3).
bloch_phase_a1 = np.exp(1j * TPI * k1[0])
print(f"Bloch phase e^(i k.a1) = {bloch_phase_a1}  |phase|={abs(bloch_phase_a1):.6f}")


def psi_native(x_cart):
    """psi_(n,k1)(x_cart, 0, 0), x_cart may be any real number (not
    restricted to one cell) -- direct evaluation of the native sum."""
    r_cart = np.array([x_cart, 0.0, 0.0])
    return np.sum(C1 * np.exp(1j * (kg1_cart @ r_cart)))


s = np.linspace(0, 1, NPTS, endpoint=False)   # fractional coordinate within one cell

# segment 1: first cell, direct
x1_cart = s[:, None] * a1[None, :]
psi_seg1 = np.array([psi_native(x1_cart[i, 0]) for i in range(NPTS)])

# segment 2, method (1) direct: plug in the TRUE unwrapped position (s+1)*a1
x2_cart_true = (s[:, None] + 1.0) * a1[None, :]
psi_seg2_direct = np.array([psi_native(x2_cart_true[i, 0]) for i in range(NPTS)])

# segment 2, method (2) wrap + Bloch phase: wrap back to [0,1) and multiply by e^{ik.a1}
psi_seg2_wrapped = bloch_phase_a1 * psi_seg1   # psi_seg1 was evaluated at the SAME wrapped s

diff = np.max(np.abs(psi_seg2_direct - psi_seg2_wrapped))
print(f"max|psi_direct - psi_(wrap+phase)| over segment 2 = {diff:.3e}")

# absolute x-axis (Angstrom, distance along a1) spanning both cells
L = np.linalg.norm(a1)
x_abs_1 = s * L
x_abs_2 = (s + 1.0) * L

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
axes[0].plot(x_abs_1, psi_seg1.real, 'b-', lw=1.5, label="cell 1 (direct)")
axes[0].plot(x_abs_2, psi_seg2_direct.real, 'g-', lw=1.5, label="cell 2 (direct, unwrapped x)")
axes[0].plot(x_abs_2, psi_seg2_wrapped.real, 'r--', lw=1.5, label="cell 2 (wrap + Bloch phase)")
axes[0].axvline(L, color='0.6', lw=0.8, ls=':')
axes[0].set_ylabel(r"Re $\psi_{n,k}(x,0,0)$")
axes[0].legend(fontsize=8)
axes[0].set_title(f"WSe2_mono, ik={IK}, band {iband}, y=z=0, across the a1 cell boundary")

axes[1].plot(x_abs_1, psi_seg1.imag, 'b-', lw=1.5)
axes[1].plot(x_abs_2, psi_seg2_direct.imag, 'g-', lw=1.5)
axes[1].plot(x_abs_2, psi_seg2_wrapped.imag, 'r--', lw=1.5)
axes[1].axvline(L, color='0.6', lw=0.8, ls=':')
axes[1].set_ylabel(r"Im $\psi_{n,k}(x,0,0)$")
axes[1].set_xlabel(r"x (angstrom, along $a_1$)")

fig.tight_layout()
outpath = "bloch_boundary_smoothness.png"
fig.savefig(outpath, dpi=150)
print(f"Saved -> {outpath}")
