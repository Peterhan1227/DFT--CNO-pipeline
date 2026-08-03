"""
Sanity check (read-only, no data files modified): Bloch periodicity in k,
psi_{n,k+G0}(r) == psi_{n,k}(r) for any reciprocal lattice vector G0.

WSe2_mono's actual k-mesh (Gamma-centered, reduced to the first BZ) never
contains two ENTRIES that are literally k and k+G0 for some real reciprocal
lattice vector G0 (checked directly -- zero such pairs in the 324-k list).
So this can't be tested by comparing two independently-VASP-computed
k-points directly. Instead it exercises vaspwfc.gvectors()'s own
cutoff-sphere selection at an EXPLICIT k2 = k1 + G0 (gvectors() accepts an
optional kvec override and recomputes everything from scratch for it,
independent of k1) -- a genuine test of the code's internal consistency
under Bloch periodicity, not a tautology.
"""
import numpy as np
from vaspwfc import vaspwfc

WAVECAR = (r"C:\Users\hanziruopeter\OneDrive\Personal files\Academic\Physics"
           r"\Research\Columbia Raquel\Density matrix cal\Data\WSe2_mono\WAVECAR")
IK = 1
G0 = np.array([1, 0, 0])   # one full reciprocal lattice vector shift

wfc = vaspwfc(WAVECAR, lsorbit=False)
k1 = wfc._kvecs[IK - 1]
k2 = k1 + G0
print(f"k1 (frac) = {k1}")
print(f"k2 = k1 + G0 (frac) = {k2}   (G0={G0})")

occ = wfc._occs[0, IK - 1, :]
iband = int(np.where(occ > 0.5)[0][-1]) + 1   # last occupied band (near VBM)
print(f"band {iband}  occ={occ[iband-1]:.4f}  E={wfc._bands[0, IK-1, iband-1]:.4f} eV")

G1 = wfc.gvectors(IK)                       # native cutoff-sphere G-set at k1
C1 = wfc.readBandCoeff(ispin=1, ikpt=IK, iband=iband, norm=True)
G2 = wfc.gvectors(IK, kvec=k2)              # INDEPENDENTLY recomputed cutoff-sphere G-set at k2
print(f"|G1| = {len(G1)}   |G2| = {len(G2)}")

# Step 1: does gvectors() find the SAME physical points under relabeling?
# physically (k1+G1) == (k2 + (G1-G0)), so G2 should equal {G1-G0} as sets.
expected_G2 = G1 - G0[None, :]
set_G2 = {tuple(g) for g in G2}
set_expected = {tuple(g) for g in expected_G2}
missing = set_expected - set_G2
extra = set_G2 - set_expected
print(f"\nSet check: expected {len(set_expected)} points, found {len(set_G2)} in G2")
print(f"  missing (should be in G2 but isn't): {len(missing)}")
print(f"  extra   (in G2 but not expected):    {len(extra)}")

# Step 2: reindex C1 onto G2 order and evaluate psi(r) both ways.
# map each G1 row -> its position in G2 (via G1-G0)
g2_index = {tuple(g): i for i, g in enumerate(G2)}
C2 = np.zeros(len(G2), dtype=np.complex128)
matched = 0
for i, g1 in enumerate(G1):
    key = tuple(g1 - G0)
    if key in g2_index:
        C2[g2_index[key]] = C1[i]
        matched += 1
print(f"  matched {matched}/{len(G1)} coefficients from G1 into G2's ordering")

rng = np.random.default_rng(0)
r_frac = rng.uniform(0, 1, size=3)
r_cart = r_frac @ wfc._Acell
print(f"\nr (frac) = {r_frac}")

Bcell = wfc._Bcell  # reciprocal lattice rows, b_i . a_j = delta_ij (no 2pi)
TPI = 2 * np.pi

kg1_cart = (k1[None, :] + G1) @ (TPI * Bcell)
psi1 = np.sum(C1 * np.exp(1j * (kg1_cart @ r_cart)))

kg2_cart = (k2[None, :] + G2) @ (TPI * Bcell)
psi2 = np.sum(C2 * np.exp(1j * (kg2_cart @ r_cart)))

print(f"\npsi_(n,k1)(r)     = {psi1}")
print(f"psi_(n,k1+G0)(r)  = {psi2}   (via independently-recomputed G2 + reindexed C)")
print(f"|difference|      = {abs(psi1 - psi2):.3e}")
