import os
import sys
from pathlib import Path

import numpy as np

from config import MATERIAL, OUTPUT_SUBDIR

sys.path.insert(0, str(Path(__file__).resolve().parent / "helper functions"))
from ws_cell import read_poscar_structure
from cno_quadrature import SavedCNOQuadrature


N_EXPORT = 10

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "Data" / MATERIAL

output_subdir = os.environ.get("CNO_OUTPUT_SUBDIR", OUTPUT_SUBDIR)
output_dir = data_dir / "output" / output_subdir
requested_field = os.environ.get("CNO_FIELD_FILE")
if requested_field is not None:
    orbital_path = output_dir / requested_field
else:
    # Keep the historical symmetrized export when it exists, but make a saved
    # finite-volume CNO result exportable without any manual source edit.
    orbital_path = output_dir / "cnos_sym_adapted.npy"
    if not orbital_path.exists():
        orbital_path = output_dir / "cno_orbitals.npy"

orbitals = np.load(orbital_path)
occupations = np.load(output_dir / "cno_occupations.npy")
quadrature = SavedCNOQuadrature.load(output_dir)
quadrature.validate_cno_rows(orbitals)
grid_shape = np.asarray(quadrature.sample_grid_shape, dtype=int)

(
    lattice,
    species,
    counts,
    atom_symbols,
    atom_numbers,
    atoms_frac,
    atoms_cart,
) = read_poscar_structure(data_dir / "POSCAR")

ws_enabled = quadrature.method != "regular_fft_grid"

n_export = min(N_EXPORT, orbitals.shape[1])

data = {
    "format_version": np.array("cno-visualizer-v1"),
    "cno_values": orbitals[:, :n_export].T,
    "cno_occupations": occupations[:n_export],
    "cno_indices": np.arange(n_export),
    "grid_shape": grid_shape,
    "lattice": lattice,
    "atom_symbols": np.asarray(atom_symbols),
    "atom_numbers": np.asarray(atom_numbers),
    "atoms_frac": atoms_frac,
    "atoms_cart": atoms_cart,
    "ws_enabled": np.array(ws_enabled),
    "material": np.array(MATERIAL),
}

if ws_enabled:
    data.update(
        points_cart=np.asarray(quadrature.points_cart, dtype=float),
        points_frac_cont=np.asarray(quadrature.points_frac_cont, dtype=float),
        base_indices=np.asarray(quadrature.base_indices, dtype=int),
        translations=np.asarray(quadrature.translations, dtype=int),
        ws_center_cart=np.load(
            output_dir / "ws_center_cart.npy"
        ),
        ws_center_frac=np.load(
            output_dir / "ws_center_frac_wrapped.npy"
        ),
    )

visualizer_dir = output_dir / "visualizer"
visualizer_dir.mkdir(parents=True, exist_ok=True)

save_path = visualizer_dir / "cno_visualizer_data.npz"
np.savez_compressed(save_path, **data)

print(f"Saved: {save_path}")
