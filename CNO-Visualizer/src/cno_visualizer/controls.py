"""Interactive controls — sliders, keyboard shortcuts, and the help panel."""

from __future__ import annotations

import math

from cno_visualizer.camera import CameraState


HELP_TEXT = (
    "CNO-Visualizer  -  keyboard:\n"
    "  P phase   D density   R real   I imag\n"
    "  V crystal/WS view\n"
    "  A atoms   B bonds     C cell   W WS cell\n"
    "  M center marker       X axes\n"
    "  S screenshot   0 reset cam   H help\n"
)


def attach_controls(viewer) -> None:
    """Register all sliders, key events, and the help overlay on ``viewer``."""
    plotter = viewer.plotter
    data = viewer.data
    state = viewer.state

    n_cnos = data.n_cnos

    # Forward reference — the status text actor is created after the sliders.
    _status_actor_ref = {"actor": None}

    def _status_text() -> str:
        try:
            abs_iso = viewer.current_abs_isovalue()
        except Exception:
            abs_iso = float("nan")
        return (
            f"[{state.view_mode} | {state.display_mode}]  "
            f"{data.resolve_cno_label(state.selected_cno)}\n"
            f"iso = {state.isovalue_fraction:.3f} x max = {abs_iso:.2e}   "
            f"(|psi|^2, sum=1; matches VESTA cube level)"
        )

    def _refresh_status():
        actor = _status_actor_ref["actor"]
        if actor is None:
            return
        try:
            actor.SetInput(_status_text())
            plotter.render()
        except Exception:
            pass

    def _on_cno(value):
        viewer.set_selected_cno(int(round(float(value))))
        _refresh_status()

    def _on_iso(value):
        viewer.set_isovalue(float(value))
        _refresh_status()

    def _on_phase(value):
        viewer.set_phase_offset(float(value))

    def _on_opacity(value):
        viewer.set_central_opacity(float(value))

    # (title, callback, min, max, value, y, fmt)
    slider_specs = [
        ("CNO index", _on_cno, 0, max(0, n_cnos - 1), state.selected_cno, 0.92, "%.0f"),
        ("Iso (frac. of max)", _on_iso, 0.001, 0.5, state.isovalue_fraction, 0.74, "%.3f"),
        ("Phase offset", _on_phase, 0.0, 2.0 * math.pi, state.phase_offset, 0.56, "%.2f"),
        ("Opacity", _on_opacity, 0.0, 1.0, state.central_opacity, 0.38, "%.2f"),
    ]

    for title, callback, smin, smax, sval, y, fmt in slider_specs:
        kwargs = dict(
            callback=callback,
            rng=(float(smin), float(smax)),
            value=float(sval),
            title=title,
            pointa=(0.025, y),
            pointb=(0.255, y),
            fmt=fmt,
        )
        try:
            widget = plotter.add_slider_widget(
                style="modern", title_height=0.018, slider_width=0.025,
                tube_width=0.005, **kwargs,
            )
        except Exception:
            widget = plotter.add_slider_widget(**kwargs)
        # Discrete integer steps on the CNO slider.
        if title == "CNO index" and n_cnos > 1:
            try:
                widget.GetRepresentation().SetNumberOfSliderSteps(n_cnos)
            except Exception:
                pass

    # Keyboard shortcuts.
    plotter.add_key_event("p", lambda: viewer.set_display_mode("phase"))
    plotter.add_key_event("d", lambda: viewer.set_display_mode("density"))
    plotter.add_key_event("r", lambda: viewer.set_display_mode("real"))
    plotter.add_key_event("i", lambda: viewer.set_display_mode("imaginary"))
    plotter.add_key_event("v", lambda: (viewer.toggle_view_mode(), _refresh_status()))
    plotter.add_key_event("a", lambda: viewer.toggle("atoms"))
    plotter.add_key_event("b", lambda: viewer.toggle("bonds"))
    plotter.add_key_event("c", lambda: viewer.toggle("cells"))
    plotter.add_key_event("w", lambda: viewer.toggle("ws"))
    plotter.add_key_event("m", lambda: viewer.toggle("center"))
    plotter.add_key_event("x", lambda: viewer.toggle("axes"))
    plotter.add_key_event("s", lambda: _save_screenshot())
    plotter.add_key_event("h", lambda: _toggle_help())
    plotter.add_key_event("0", lambda: (plotter.reset_camera(), plotter.render()))

    help_actor = plotter.add_text(
        HELP_TEXT, position="upper_right", font_size=8, color="white"
    )

    status_actor = plotter.add_text(
        _status_text(), position="upper_left", font_size=10, color="white",
    )
    _status_actor_ref["actor"] = status_actor

    def _toggle_help():
        try:
            help_actor.SetVisibility(not bool(help_actor.GetVisibility()))
            plotter.render()
        except Exception:
            pass

    def _save_screenshot():
        try:
            path = viewer.screenshot()
            print(f"[cno-visualizer] saved screenshot -> {path}")
        except Exception as exc:
            print(f"[cno-visualizer] screenshot failed: {exc}")

    if state.camera_path is not None and state.camera_path.is_file():
        try:
            CameraState.from_json(state.camera_path).apply(plotter)
        except Exception as exc:
            print(f"[cno-visualizer] could not apply camera state: {exc}")


__all__ = ["attach_controls", "HELP_TEXT"]
