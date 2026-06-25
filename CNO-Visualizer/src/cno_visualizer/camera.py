"""Camera state save/load — simple JSON, no PyVista dependency at import time."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple


@dataclass
class CameraState:
    position: Tuple[float, float, float]
    focal_point: Tuple[float, float, float]
    view_up: Tuple[float, float, float]
    parallel_scale: float = 0.0
    parallel_projection: bool = False

    @classmethod
    def from_plotter(cls, plotter) -> "CameraState":
        cam = plotter.camera
        return cls(
            position=tuple(float(x) for x in cam.position),
            focal_point=tuple(float(x) for x in cam.focal_point),
            view_up=tuple(float(x) for x in cam.up),
            parallel_scale=float(cam.parallel_scale),
            parallel_projection=bool(cam.parallel_projection),
        )

    def apply(self, plotter) -> None:
        plotter.camera_position = [
            tuple(self.position),
            tuple(self.focal_point),
            tuple(self.view_up),
        ]
        plotter.camera.parallel_scale = float(self.parallel_scale)
        plotter.camera.parallel_projection = bool(self.parallel_projection)

    def to_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
        return p

    @classmethod
    def from_json(cls, path: str | Path) -> "CameraState":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Camera state file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(
            position=tuple(data["position"]),
            focal_point=tuple(data["focal_point"]),
            view_up=tuple(data["view_up"]),
            parallel_scale=float(data.get("parallel_scale", 0.0)),
            parallel_projection=bool(data.get("parallel_projection", False)),
        )


__all__ = ["CameraState"]
