"""CNOData.from_arrays + VESTA-like snapshot rendering."""

from __future__ import annotations

import numpy as np
import pytest

from cno_visualizer.data import CNOData, CNODataError


def _synthetic_grid(n=8):
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    frac = np.stack([ii / n, jj / n, kk / n], -1)
    cart = frac @ np.diag([4.0, 4.0, 4.0])
    r2 = np.sum((cart - 2.0) ** 2, axis=-1)
    return np.exp(-r2).astype(np.complex128)  # (n,n,n) localized blob


def test_from_arrays_builds_primitive_cnodata():
    g = _synthetic_grid(8)
    cv = g.reshape(-1, order="C")[None, :]
    data = CNOData.from_arrays(
        cv, (8, 8, 8), np.diag([4.0, 4.0, 4.0]),
        atom_symbols=["Si"], atoms_cart=np.zeros((1, 3)),
    )
    assert data.ws_enabled is False
    assert data.n_cnos == 1 and data.n_points == 512
    assert data.atom_numbers[0] == 14            # derived from "Si"
    np.testing.assert_allclose(data.atoms_frac[0], [0, 0, 0])


def test_from_arrays_rejects_wrong_width():
    with pytest.raises(CNODataError):
        CNOData.from_arrays(np.zeros((1, 10), complex), (8, 8, 8), np.eye(3))


def test_render_density_gif_writes_png(tmp_path):
    pytest.importorskip("pyvista")
    import pyvista as pv
    pv.OFF_SCREEN = True
    from cno_visualizer.snapshot import render_density_gif

    g = _synthetic_grid(8)
    out = tmp_path / "snap.png"
    path = render_density_gif(
        g, np.diag([4.0, 4.0, 4.0]), np.zeros((1, 3)), ["Si"], str(out),
        iso_fraction=0.3, replication=(1, 1, 1), show_atoms=True, show_bonds=False,
    )
    assert out.exists() and out.stat().st_size > 0
    assert str(path) == str(out)


def test_render_density_gif_spins_to_gif(tmp_path):
    # Exercises the spin loop (the .png path skips it). Guards the n/n_frames
    # name-collision regression and the spin_axis camera math.
    pytest.importorskip("pyvista")
    import pyvista as pv
    pv.OFF_SCREEN = True
    from cno_visualizer.snapshot import render_density_gif

    g = _synthetic_grid(8)
    for axis in ("x", "z"):
        out = tmp_path / f"spin_{axis}.gif"
        render_density_gif(
            g, np.diag([4.0, 4.0, 4.0]), np.zeros((1, 3)), ["Si"], str(out),
            iso_fraction=0.3, replication=(1, 1, 1),
            seconds=0.4, fps=10, deg_per_sec=12.0, spin_axis=axis,
        )
        assert out.exists() and out.stat().st_size > 0
