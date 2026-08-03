"""
Trial script (read-only, no data files modified): for one k-point and one
band of WSe2_mono, pick a fixed in-plane G=(Gx,Gy) and scan |C_G|^2 over
every available Gz on that "rod", to check whether out-of-plane plane-wave
content is negligible for a monolayer-in-25-Ang-vacuum setup.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vaspwfc import vaspwfc

WAVECAR = (r"C:\Users\hanziruopeter\OneDrive\Personal files\Academic\Physics"
           r"\Research\Columbia Raquel\Density matrix cal\Data\WSe2_mono\WAVECAR")
IK = 1          # k-point index (1-based)
GXY = (0, 0)    # fixed in-plane (Gx, Gy) -- the "specular rod"

wfc = vaspwfc(WAVECAR, lsorbit=False)
kvec = wfc._kvecs[IK - 1]
print(f"k-point {IK}: k_frac = {kvec}")

occ = wfc._occs[0, IK - 1, :]
occ_bands = np.where(occ > 0.5)[0]
bands_to_plot = dict(
    deep_semicore=int(occ_bands[0]),     # first occupied band (very bound)
    near_VBM=int(occ_bands[-1]),         # last occupied band (VBM region)
)

gvec = wfc.gvectors(IK)                                   # (npw, 3) int (Gx,Gy,Gz)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for label, iband in bands_to_plot.items():
    Ck = wfc.readBandCoeff(ispin=1, ikpt=IK, iband=iband + 1, norm=True)
    mask = (gvec[:, 0] == GXY[0]) & (gvec[:, 1] == GXY[1])
    gz = gvec[mask, 2]
    c = Ck[mask]
    order = np.argsort(gz)
    gz, c = gz[order], c[order]

    frac = np.sum(np.abs(c) ** 2) / np.sum(np.abs(Ck) ** 2)
    print(f"\nband {iband+1} ({label}): occ={occ[iband]:.4f}  "
          f"energy={wfc._bands[0, IK-1, iband]:.4f} eV")
    print(f"  G=({GXY[0]},{GXY[1]},Gz): {mask.sum()} plane waves on this rod, "
          f"Gz in [{gz.min()}, {gz.max()}]")
    print(f"  sum|C_G|^2 on this rod = {np.sum(np.abs(c)**2):.6f} "
          f"({frac*100:.2f}% of the total band norm)")
    print(f"  |C_G|^2 at Gz=0:  {np.abs(c[gz == 0][0])**2:.6e}")
    print(f"  |C_G|^2 at largest |Gz| ({gz[np.argmax(np.abs(gz))]}): "
          f"{np.abs(c[np.argmax(np.abs(gz))])**2:.6e}")

    ax.plot(gz, np.abs(c) ** 2, 'o-', ms=4,
            label=f"band {iband+1} ({label}, E={wfc._bands[0,IK-1,iband]:.2f} eV)")

ax.set_xlabel(r"$G_z$ (reciprocal lattice index)")
ax.set_ylabel(r"$|C_{\mathbf{G}}|^2$")
ax.set_yscale("log")
ax.set_title(f"WSe2_mono, k-point {IK}, $G_{{xy}}$=({GXY[0]},{GXY[1]})")
ax.legend(fontsize=8)
fig.tight_layout()
outpath = "gz_coefficient_scan.png"
fig.savefig(outpath, dpi=150)
print(f"\nSaved -> {outpath}")
