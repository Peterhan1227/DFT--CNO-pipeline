import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helper functions"))
from ws_cell import read_poscar_structure

from vaspwfc import vaspwfc
from paw_overlap import load_pawpp
from paw_density_matrix import build_real_space_S, build_density_matrix

# Frozen snapshot, not live Data/ -- see RESULTS.md "data integrity incident"
data_dir = Path(__file__).resolve().parent / "data_snapshot" / "WSe2_mono"
latvec, species, counts, atom_symbols, atom_numbers, frac_coords, cart_coords = \
    read_poscar_structure(data_dir / "POSCAR")
pawpp = load_pawpp(data_dir / "POTCAR")
pawpp_elements = [pp.element.split('_')[0] for pp in pawpp]
elements_idx = [pawpp_elements.index(s) for s in atom_symbols]

wfc = vaspwfc(str(data_dir / "WAVECAR"), lsorbit=False)
Nx, Ny, Nz = wfc._ngrid
Nr = Nx * Ny * Nz

def _read_eigenval(path, nkpts_expected, nbands_expected):
    with open(path) as fh:
        lines = fh.readlines()
    nkpts = int(lines[5].split()[1]); nbands = int(lines[5].split()[2])
    kfrac = np.zeros((nkpts, 3)); kweights = np.zeros(nkpts)
    idx = 6
    for ik in range(nkpts):
        while not lines[idx].split():
            idx += 1
        kline = lines[idx].split()
        kfrac[ik] = [float(x) for x in kline[:3]]
        kweights[ik] = float(kline[3])
        idx += 1 + nbands_expected
    kweights /= kweights.sum()
    return kfrac, kweights

kfrac_all, kweights = _read_eigenval(data_dir / "EIGENVAL", wfc._nkpts, wfc._nbands)

# expected total occupied count
total_expected = 0.0
for ik in range(1, wfc._nkpts + 1):
    occ_all = wfc._occs[0, ik - 1, :]
    occ = occ_all[occ_all > 1e-6]
    if occ.max() > 1.5:
        occ = occ / 2.0
    total_expected += kweights[ik-1] * occ.sum()
print("expected total (sum w_k * sum f_nk):", total_expected)

ix, iy, iz = [a.ravel() for a in np.mgrid[0:Nx, 0:Ny, 0:Nz]]
r_box_frac = np.column_stack([ix / Nx, iy / Ny, iz / Nz])
r_box_cart = r_box_frac @ latvec

print("Building S_box ...")
S_box, n_img = build_real_space_S(pawpp, elements_idx, cart_coords, latvec, r_box_cart, nmax=3)
print("n_img:", n_img)

print("Building D_box (raw coeffs) ...")
t0 = time.time()
D_box = build_density_matrix(wfc, kfrac_all, kweights, 1, Nr, (Nx, Ny, Nz),
                              r_box_frac, None, verbose=True)
print(f"done in {time.time()-t0:.1f}s")

tr_D = np.trace(D_box).real
tr_DS = np.trace(D_box @ S_box).real
print(f"Tr(D) = {tr_D:.6f}   (uncorrected reference, no augmentation)")
print(f"Tr(D S) = {tr_DS:.6f}   (should match expected total {total_expected:.6f})")
