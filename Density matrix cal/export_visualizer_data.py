import sys
from pathlib import Path

import numpy as np

from config import MATERIAL, OUTPUT_SUBDIR

sys.path.insert(0, str(Path(__file__).resolve().parent / "helper functions"))
from ws_cell import read_poscar_structure


N_EXPORT = 10

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "Data" / MATERIAL
output_dir = data_dir / "output" / OUTPUT_SUBDIR

orbitals = np.load(output_dir / "cnos_sym_adapted.npy")
occupations = np.load(output_dir / "cno_occupations.npy")
grid_shape = np.load(output_dir / "fft_grid_shape.npy").astype(int)

(
    lattice,
    species,
    counts,
    atom_symbols,
    atom_numbers,
    atoms_frac,
    atoms_cart,
) = read_poscar_structure(data_dir / "POSCAR")

ws_enabled_file = output_dir / "ws_enabled.npy"
ws_enabled = (
    bool(np.load(ws_enabled_file))
    if ws_enabled_file.exists()
    else False
)

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
}

if ws_enabled:
    data.update(
        points_cart=np.load(output_dir / "ws_points_cart.npy"),
        points_frac_cont=np.load(
            output_dir / "ws_points_frac_cont.npy"
        ),
        base_indices=np.load(
            output_dir / "ws_base_indices.npy"
        ),
        translations=np.load(
            output_dir / "ws_translation_int.npy"
        ),
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