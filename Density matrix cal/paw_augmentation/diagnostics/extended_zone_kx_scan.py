"""
Extended-zone-scheme smoothness check (read-only): x-axis is the actual
crystal momentum k_x, not G. FIX a real-space point r and a band index n.
Walk through the k-mesh's REAL, independently-DFT-computed k-points along
k_y=k_z=0 (18 of them, k_x in [-0.444, 0.5], the first BZ) and evaluate
psi_n(r) at each -- genuine physics, no relabeling trick needed for this
part. Then, once the boundary (k_x=0.5) is reached, continue the x-axis
into the NEXT zone by reusing the SAME 18 k-points relabeled as k+G0 with
G0=(1,0,0) (Bloch's theorem: psi_{k+G}(r) == psi_k(r) exactly, already
verified in bloch_periodicity_check.py) -- i.e. tile the same fundamental-
domain data into the extended zone scheme and check the tiles connect with
no discontinuity at the seam (k_x=0.5 -> 0.5+1/18=0.556).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vaspwfc import vaspwfc

WAVECAR = (r"C:\Users\hanziruopeter\OneDrive\Personal files\Academic\Physics"
           r"\Research\Columbia Raquel\Density matrix cal\Data\WSe2_mono\WAVECAR")
IBAND = 13   # fixed band index (near VBM, occupied at every k here)

wfc = vaspwfc(WAVECAR, lsorbit=False)
kv = wfc._kvecs
mask = (np.abs(kv[:, 1]) < 1e-9) & (np.abs(kv[:, 2]) < 1e-9)
ik_list = np.where(mask)[0] + 1   # 1-based

r_frac = np.array([0.30, 0.20, 0.10])   # fixed r, chosen once
r_cart = r_frac @ wfc._Acell
print(f"fixed r (frac) = {r_frac}")
print(f"band {IBAND}, {len(ik_list)} k-points along ky=kz=0")

Bcell = wfc._Bcell
TPI = 2 * np.pi


def psi_at(ik, r_cart, G_shift=(0, 0, 0)):
    """psi_{n,k+G_shift}(r_cart) using k-point ik's OWN native (G, C) data,
    with an explicit reciprocal-lattice shift applied to k (not to G) --
    physically psi_{k+G0}(r) == psi_k(r) exactly, so this just re-tags the
    x-axis position, not the value; included for clarity/verification."""
    kvec = wfc._kvecs[ik - 1]
    G = wfc.gvectors(ik)
    C = wfc.readBandCoeff(ispin=1, ikpt=ik, iband=IBAND, norm=True)
    kG_cart = (kvec[None, :] + G) @ (TPI * Bcell)
    return np.sum(C * np.exp(1j * (kG_cart @ r_cart)))


kx_list = []
psi_list = []
for ik in ik_list:
    kx_list.append(kv[ik - 1, 0])
    psi_list.append(psi_at(ik, r_cart))
kx_list = np.array(kx_list)
psi_list = np.array(psi_list)

order = np.argsort(kx_list)
kx_sorted = kx_list[order]
psi_sorted = psi_list[order]
ik_sorted = ik_list[order]

print("\ntile 1 (G=(0,0,0)): kx from", kx_sorted[0], "to", kx_sorted[-1])
for kx, ik, p in zip(kx_sorted, ik_sorted, psi_sorted):
    print(f"  ik={ik:4d}  kx={kx: .6f}  psi(r)={p: .6e}")

# tile 2: SAME k-points, relabeled kx -> kx+1 (G0=(1,0,0)), psi value UNCHANGED
kx_tile2 = kx_sorted + 1.0
psi_tile2 = psi_sorted.copy()   # psi_{k+G0}(r) == psi_k(r) exactly (Bloch)

# seam check: last point of tile 1 (kx=0.5) vs first point of tile 2 (kx=0.556)
seam_gap = kx_tile2[0] - kx_sorted[-1]
seam_jump = abs(psi_tile2[0] - psi_sorted[-1])
typical_jump = np.mean(np.abs(np.diff(psi_sorted)))
print(f"\nseam: kx {kx_sorted[-1]:.6f} -> {kx_tile2[0]:.6f}  (gap={seam_gap:.6f}, "
      f"matches mesh spacing {1/18:.6f}? {np.isclose(seam_gap, 1/18)})")
print(f"|psi| jump AT seam           = {seam_jump:.4e}")
print(f"typical |psi| jump elsewhere = {typical_jump:.4e}")

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(kx_sorted, psi_sorted.real, 'o-', ms=5, color='C0', label="tile 1, G=(0,0,0)")
axes[0].plot(kx_tile2, psi_tile2.real, 's-', ms=5, color='C1', label="tile 2, G=(1,0,0)")
axes[0].axvline(kx_sorted[-1], color='0.6', lw=0.8, ls=':')
axes[0].set_ylabel(r"Re $\psi_{n,k}(r)$")
axes[0].legend(fontsize=8)
axes[0].set_title(f"WSe2_mono, band {IBAND}, fixed r={r_frac} (frac), $k_y=k_z=0$, extended zone scheme")

axes[1].plot(kx_sorted, psi_sorted.imag, 'o-', ms=5, color='C0')
axes[1].plot(kx_tile2, psi_tile2.imag, 's-', ms=5, color='C1')
axes[1].axvline(kx_sorted[-1], color='0.6', lw=0.8, ls=':')
axes[1].set_ylabel(r"Im $\psi_{n,k}(r)$")
axes[1].set_xlabel(r"$k_x$ (extended zone scheme)")

fig.tight_layout()
outpath = "extended_zone_kx_scan.png"
fig.savefig(outpath, dpi=150)
print(f"\nSaved -> {outpath}")
