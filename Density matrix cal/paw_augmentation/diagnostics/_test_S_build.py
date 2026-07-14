import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "helper functions"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for paw_overlap (paw_augmentation/)
from ws_cell import read_poscar_structure

from paw_overlap import load_pawpp
from paw_density_matrix import build_real_space_S

# Frozen snapshot, not live Data/ -- see RESULTS.md "data integrity incident"
data_dir = Path(__file__).resolve().parent / "data_snapshot" / "WSe2_mono"
latvec, species, counts, atom_symbols, atom_numbers, frac_coords, cart_coords = \
    read_poscar_structure(data_dir / "POSCAR")
pawpp = load_pawpp(data_dir / "POTCAR")
pawpp_elements = [pp.element.split('_')[0] for pp in pawpp]
elements_idx = [pawpp_elements.index(s) for s in atom_symbols]
print("atoms:", atom_symbols, "cart:\n", cart_coords)
print("elements_idx:", elements_idx)

Nx, Ny, Nz = 11, 11, 73
ix, iy, iz = [a.ravel() for a in np.mgrid[0:Nx, 0:Ny, 0:Nz]]
r_box_frac = np.column_stack([ix / Nx, iy / Ny, iz / Nz])
r_box_cart = r_box_frac @ latvec
Nr = len(ix)
print("Nr =", Nr)

t0 = time.time()
S_box, n_img = build_real_space_S(pawpp, elements_idx, cart_coords, latvec, r_box_cart, nmax=4)
print(f"built S_box in {time.time()-t0:.1f}s, n_img={n_img}")
herm_err = np.max(np.abs(S_box - S_box.conj().T))
print("herm err:", herm_err)
print("diag range:", S_box.diagonal().real.min(), S_box.diagonal().real.max())
offdiag_count = np.sum(np.abs(S_box - np.eye(Nr)) > 1e-6)
print("nnz-ish off-identity entries:", offdiag_count, "/", Nr*Nr)

# eigenvalues of S alone (should be positive; a valid metric)
evals = np.linalg.eigvalsh(S_box)
print("S eigenvalue range:", evals.min(), evals.max())
print("num negative eigenvalues:", np.sum(evals < 0))
