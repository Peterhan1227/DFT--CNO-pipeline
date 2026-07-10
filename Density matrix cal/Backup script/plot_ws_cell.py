"""
plot_ws_cell.py — Interactive 3D visualisation of the Wigner-Seitz cell map.

Opens an interactive HTML plot in the browser showing:
  - WS grid points coloured by distance from the center (drag/rotate/zoom)
  - chosen WS center (red diamond)
  - primitive cell wireframe (grey dashed lines)
  - home-cell atoms (large, opaque) and ±1 periodic images (small, faint)

Hover any point to see coordinates.  Toggle layers via the legend.

Output: Data/<MATERIAL>/output/ws_cell_debug.html  (auto-opened in browser)

Requires plotly:  pip install plotly
"""

import sys
import numpy as np
from pathlib import Path

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("plotly is required.  Install with:  pip install plotly")

PIPELINE_DIR = Path(__file__).resolve().parent.parent   # Density matrix cal/
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(PIPELINE_DIR / "helper functions"))

from config import (
    MATERIAL, OUTPUT_SUBDIR,
    WS_CENTER, WS_CENTER_COORD_TYPE, WS_TRANSLATION_SEARCH_RANGE,
    USE_WS_CELL,
)
from ws_cell import read_poscar_structure, parse_ws_center, build_ws_grid_map


# ── paths ──────────────────────────────────────────────────────────────────────
base_dir    = PIPELINE_DIR
data_dir    = base_dir / "Data" / MATERIAL
output_dir  = data_dir / "output" / OUTPUT_SUBDIR
poscar_path = data_dir / "POSCAR"
output_dir.mkdir(parents=True, exist_ok=True)


# ── structure ──────────────────────────────────────────────────────────────────
latvec, species, counts, atom_symbols, atom_numbers, frac_coords, cart_coords = \
    read_poscar_structure(poscar_path)
volume = abs(np.dot(latvec[0], np.cross(latvec[1], latvec[2])))
print(f"POSCAR   : volume={volume:.4f} Å³  species={dict(zip(species, counts))}")


# ── WS center ──────────────────────────────────────────────────────────────────
center_cart, center_frac_cont, center_frac_wrapped = parse_ws_center(
    WS_CENTER, WS_CENTER_COORD_TYPE, latvec
)
print(f"WS center (frac): {np.round(center_frac_cont, 6)}")
print(f"WS center (cart): {np.round(center_cart, 6)} Å")


# ── FFT grid ───────────────────────────────────────────────────────────────────
grid_file = output_dir / "fft_grid_shape.npy"
if grid_file.exists():
    Nx, Ny, Nz = np.load(grid_file).astype(int)
    print(f"FFT grid from file: ({Nx}, {Ny}, {Nz})")
else:
    Nx = Ny = Nz = 20
    print(f"WARNING: fft_grid_shape.npy not found; using coarse {Nx}×{Ny}×{Nz} preview grid.")


# ── build or load WS map ───────────────────────────────────────────────────────
ws_cart_file = output_dir / "ws_points_cart.npy"
ws_frac_file = output_dir / "ws_points_frac_cont.npy"

if ws_cart_file.exists():
    ws_points_cart = np.load(ws_cart_file)
    ws_points_frac = np.load(ws_frac_file) if ws_frac_file.exists() else None
    print(f"Loaded WS points: {ws_points_cart.shape}")
else:
    print(f"Building WS map  grid=({Nx},{Ny},{Nz})  nmax={WS_TRANSLATION_SEARCH_RANGE} ...")
    ws_points_cart, ws_points_frac, _, _ = build_ws_grid_map(
        latvec, (Nx, Ny, Nz), center_cart, nmax=WS_TRANSLATION_SEARCH_RANGE
    )
    print(f"Built WS map  Nr={len(ws_points_cart)}")


# ── subsample WS points ────────────────────────────────────────────────────────
Nr   = len(ws_points_cart)
step = max(1, Nr // 5000)
pts  = ws_points_cart[::step]
dist = np.linalg.norm(pts - center_cart, axis=1)
if ws_points_frac is not None:
    frac_sub = ws_points_frac[::step]
    hover_frac = [f"({f[0]:.3f}, {f[1]:.3f}, {f[2]:.3f})" for f in frac_sub]
else:
    hover_frac = ["n/a"] * len(pts)
print(f"Plotting {len(pts)}/{Nr} WS grid points (every {step}th)")


# ── separate home cell from periodic images ────────────────────────────────────
home_cart = cart_coords
home_syms = atom_symbols

image_carts, image_syms = [], []
for n1 in range(-1, 3):    # show a 4-cell-wide neighbourhood so the crystal
    for n2 in range(-1, 3): # context is similar to the 3×3×3 cube output
        for n3 in range(-1, 3):
            if n1 == n2 == n3 == 0:
                continue
            shift = np.array([n1, n2, n3]) @ latvec
            image_carts.append(cart_coords + shift)
            image_syms.extend(atom_symbols)
image_cart = np.vstack(image_carts)


# ── primitive cell wireframe ───────────────────────────────────────────────────
corners = np.array([
    [0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]
], dtype=float) @ latvec
edges = [(0,1),(0,2),(0,3),(1,4),(1,5),(2,4),(2,6),(3,5),(3,6),(4,7),(5,7),(6,7)]
wire_x, wire_y, wire_z = [], [], []
for a, b in edges:
    wire_x += [corners[a, 0], corners[b, 0], None]
    wire_y += [corners[a, 1], corners[b, 1], None]
    wire_z += [corners[a, 2], corners[b, 2], None]


# ── plotly figure ──────────────────────────────────────────────────────────────
_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f']
_species_set = list(dict.fromkeys(atom_symbols))

fig = go.Figure()

# Primitive cell wireframe
fig.add_trace(go.Scatter3d(
    x=wire_x, y=wire_y, z=wire_z,
    mode='lines',
    line=dict(color='grey', width=2, dash='dash'),
    name='Primitive cell',
    hoverinfo='skip',
))

# WS grid points coloured by distance
fig.add_trace(go.Scatter3d(
    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
    mode='markers',
    marker=dict(
        size=2,
        color=dist,
        colorscale='Viridis_r',
        opacity=0.35,
        colorbar=dict(title='Distance from<br>WS center (Å)', thickness=14, x=1.02),
        showscale=True,
    ),
    name='WS grid',
    hovertemplate=(
        'cart: (%{x:.3f}, %{y:.3f}, %{z:.3f}) Å<br>'
        'frac: %{customdata}<extra>WS grid</extra>'
    ),
    customdata=hover_frac,
))

# Home-cell atoms — large, labelled
for si, sym in enumerate(_species_set):
    mask = np.array([s == sym for s in home_syms])
    if not mask.any():
        continue
    ax = home_cart[mask]
    af = frac_coords[mask]
    labels = [f"{sym}{j}" for j in np.where(mask)[0]]
    frac_hover = [f"({f[0]:.4f}, {f[1]:.4f}, {f[2]:.4f})" for f in af]
    fig.add_trace(go.Scatter3d(
        x=ax[:, 0], y=ax[:, 1], z=ax[:, 2],
        mode='markers+text',
        marker=dict(size=10, color=_colors[si % len(_colors)],
                    opacity=1.0, symbol='circle',
                    line=dict(color='black', width=1)),
        text=labels,
        textposition='top center',
        name=f'{sym} (home cell)',
        hovertemplate=(
            f'<b>{sym}</b> (home)<br>'
            'cart: (%{x:.3f}, %{y:.3f}, %{z:.3f}) Å<br>'
            'frac: %{customdata}<extra></extra>'
        ),
        customdata=frac_hover,
    ))

# Periodic images — small, faint
for si, sym in enumerate(_species_set):
    mask = np.array([s == sym for s in image_syms])
    if not mask.any():
        continue
    ax = image_cart[mask]
    fig.add_trace(go.Scatter3d(
        x=ax[:, 0], y=ax[:, 1], z=ax[:, 2],
        mode='markers',
        marker=dict(size=4, color=_colors[si % len(_colors)],
                    opacity=0.2, symbol='circle',
                    line=dict(color='grey', width=0.5)),
        name=f'{sym} (images)',
        hovertemplate=(
            f'<b>{sym}</b> (periodic image)<br>'
            'cart: (%{x:.3f}, %{y:.3f}, %{z:.3f}) Å<extra></extra>'
        ),
    ))

# WS center — red diamond
fig.add_trace(go.Scatter3d(
    x=[center_cart[0]], y=[center_cart[1]], z=[center_cart[2]],
    mode='markers',
    marker=dict(size=14, color='red', symbol='diamond',
                opacity=1.0, line=dict(color='darkred', width=2)),
    name='WS center',
    hovertemplate=(
        '<b>WS center</b><br>'
        f'frac = {np.round(center_frac_cont, 5).tolist()}<br>'
        'cart: (%{x:.3f}, %{y:.3f}, %{z:.3f}) Å<extra></extra>'
    ),
))


# ── layout ─────────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text=(
            f'<b>{MATERIAL}</b> — Wigner-Seitz cell (interactive, drag to rotate)<br>'
            f'<sup>WS center frac = {np.round(center_frac_cont, 4).tolist()}  '
            f'grid = ({Nx}, {Ny}, {Nz})  |  '
            f'toggle layers via legend</sup>'
        ),
        x=0.5,
        font=dict(size=13),
    ),
    scene=dict(
        xaxis_title='x (Å)',
        yaxis_title='y (Å)',
        zaxis_title='z (Å)',
        aspectmode='data',      # equal axis scaling — preserves true geometry
        camera=dict(eye=dict(x=1.6, y=1.4, z=1.0)),
    ),
    legend=dict(x=0.0, y=1.0, bgcolor='rgba(255,255,255,0.8)',
                bordercolor='lightgrey', borderwidth=1),
    margin=dict(l=0, r=0, b=0, t=90),
    width=1050,
    height=780,
)

out_path = output_dir / "ws_cell_debug.html"
fig.write_html(str(out_path), auto_open=True)
print(f"Saved → {out_path}  (opened in browser)")
