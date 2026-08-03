"""Load and validate the ``cno_visualizer_data.npz`` produced by an external exporter.

The visualizer never imports anything from the exporter.  Its only dependency on the
upstream pipeline is the contract documented in the README — every contract violation
must raise a clear :class:`CNODataError` rather than producing an opaque NumPy or VTK
error downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

SUPPORTED_VERSIONS = frozenset({"cno-visualizer-v1"})

_REQUIRED = (
    "format_version",
    "cno_values",
    "cno_occupations",
    "cno_indices",
    "grid_shape",
    "lattice",
    "atom_symbols",
    "atom_numbers",
    "atoms_frac",
    "atoms_cart",
    "ws_enabled",
)

_WS_REQUIRED = (
    "points_cart",
    "points_frac_cont",
    "base_indices",
    "translations",
    "ws_center_cart",
)


class CNODataError(ValueError):
    """Raised when the input ``.npz`` violates the expected data contract."""


def _scalar(arr: np.ndarray):
    """Return a Python scalar from a 0-d numpy array (no pickle, safe for strings)."""
    return arr[()] if arr.shape == () else arr


def _as_str(value) -> str:
    if isinstance(value, np.ndarray):
        value = _scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


@dataclass(frozen=True)
class CNOData:
    """In-memory representation of the visualizer input file.

    All array fields are NumPy arrays with the dtypes documented in the README.
    ``cno_values`` is always 2-D complex with shape ``(n_cno, Nr)``; arrays are
    never mutated by the viewer.
    """

    format_version: str
    material: Optional[str]
    cno_values: np.ndarray
    cno_occupations: np.ndarray
    cno_indices: np.ndarray
    grid_shape: np.ndarray
    lattice: np.ndarray
    atom_symbols: np.ndarray
    atom_numbers: np.ndarray
    atoms_frac: np.ndarray
    atoms_cart: np.ndarray
    ws_enabled: bool
    points_cart: Optional[np.ndarray]
    points_frac_cont: Optional[np.ndarray]
    base_indices: Optional[np.ndarray]
    translations: Optional[np.ndarray]
    ws_center_cart: Optional[np.ndarray]
    ws_center_frac: Optional[np.ndarray]
    source_path: Optional[Path] = None

    # ------------------------------------------------------------------ basic info
    @property
    def n_cnos(self) -> int:
        return int(self.cno_values.shape[0])

    @property
    def n_points(self) -> int:
        return int(self.cno_values.shape[1])

    @property
    def expanded_ws(self) -> bool:
        """Whether this WS field has more samples than one periodic FFT cube.

        A finite-volume WS quadrature may retain distinct unwrapped images of a
        boundary FFT node.  Those rows cannot be folded back to a single
        ``grid_shape`` cube without losing data, so the renderer must use the
        saved unwrapped connectivity instead.
        """
        return bool(self.ws_enabled and self.n_points != int(np.prod(self.grid_shape)))

    @property
    def cell_volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    # ------------------------------------------------------------------ accessors
    def _check_index(self, local_index: int) -> int:
        i = int(local_index)
        if not (0 <= i < self.n_cnos):
            raise CNODataError(
                f"CNO local index {local_index} out of range [0, {self.n_cnos})."
            )
        return i

    def cno(self, local_index: int) -> np.ndarray:
        """Return a read-only view of the complex CNO values for ``local_index``."""
        i = self._check_index(local_index)
        view = self.cno_values[i]
        view.setflags(write=False)
        return view

    def density(self, local_index: int) -> np.ndarray:
        return np.abs(self.cno(local_index)) ** 2

    def phase(self, local_index: int, offset: float = 0.0) -> np.ndarray:
        psi = self.cno(local_index)
        return np.mod(np.angle(psi) + float(offset), 2.0 * np.pi)

    def resolve_cno_label(self, local_index: int) -> str:
        i = self._check_index(local_index)
        original = int(self.cno_indices[i])
        occ = float(self.cno_occupations[i])
        return f"local {i} | orig {original} | occ {occ:.4f}"

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_arrays(
        cls,
        cno_values: np.ndarray,
        grid_shape,
        lattice: np.ndarray,
        *,
        atom_symbols=None,
        atom_numbers=None,
        atoms_frac: Optional[np.ndarray] = None,
        atoms_cart: Optional[np.ndarray] = None,
        cno_occupations: Optional[np.ndarray] = None,
        cno_indices: Optional[np.ndarray] = None,
        material: Optional[str] = None,
    ) -> "CNOData":
        """Build a primitive-mode ``CNOData`` from in-memory arrays (no file).

        ``cno_values`` is the C-order flattening of the ``(Nx, Ny, Nz)`` grid,
        shape ``(Nr,)`` or ``(n_cno, Nr)``.  WS fields are left empty
        (``ws_enabled=False``); callers that already hold the field on a regular
        grid (e.g. an external report script) use this to render without an
        ``.npz``.  Missing atom metadata is derived from the lattice / symbols.
        """
        cv = np.asarray(cno_values)
        if cv.ndim == 1:
            cv = cv[None, :]
        cv = np.ascontiguousarray(cv, dtype=np.complex128)
        grid_shape = np.asarray(grid_shape, dtype=np.int64).reshape(3)
        lattice = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
        expected_Nr = int(np.prod(grid_shape))
        if cv.shape[1] != expected_Nr:
            raise CNODataError(
                f"cno_values width {cv.shape[1]} != prod(grid_shape) {expected_Nr}"
            )
        n_cno = cv.shape[0]

        syms = [] if atom_symbols is None else [str(s) for s in atom_symbols]
        natoms = len(syms)
        if atoms_cart is not None:
            atoms_cart = np.asarray(atoms_cart, dtype=np.float64).reshape(-1, 3)
            natoms = atoms_cart.shape[0]
        if atoms_frac is not None:
            atoms_frac = np.asarray(atoms_frac, dtype=np.float64).reshape(-1, 3)
            natoms = atoms_frac.shape[0]
        if atoms_cart is None and atoms_frac is not None:
            atoms_cart = atoms_frac @ lattice
        if atoms_frac is None and atoms_cart is not None:
            atoms_frac = atoms_cart @ np.linalg.inv(lattice)
        if atoms_cart is None:
            atoms_cart = np.zeros((natoms, 3), dtype=np.float64)
            atoms_frac = np.zeros((natoms, 3), dtype=np.float64)

        if atom_numbers is None:
            from ase.data import chemical_symbols

            atom_numbers = np.array(
                [chemical_symbols.index(s) if s in chemical_symbols else 0 for s in syms],
                dtype=np.int64,
            )
        else:
            atom_numbers = np.asarray(atom_numbers, dtype=np.int64)

        if cno_occupations is None:
            cno_occupations = np.zeros(n_cno, dtype=np.float64)
        if cno_indices is None:
            cno_indices = np.arange(n_cno, dtype=np.int64)

        return cls(
            format_version="cno-visualizer-v1",
            material=material,
            cno_values=cv,
            cno_occupations=np.asarray(cno_occupations, dtype=np.float64),
            cno_indices=np.asarray(cno_indices, dtype=np.int64),
            grid_shape=grid_shape,
            lattice=lattice,
            atom_symbols=np.asarray(syms, dtype="<U4"),
            atom_numbers=atom_numbers,
            atoms_frac=atoms_frac,
            atoms_cart=atoms_cart,
            ws_enabled=False,
            points_cart=None,
            points_frac_cont=None,
            base_indices=None,
            translations=None,
            ws_center_cart=None,
            ws_center_frac=None,
            source_path=None,
        )

    @classmethod
    def from_ws_arrays(
        cls,
        cno_values: np.ndarray,
        grid_shape,
        lattice: np.ndarray,
        *,
        base_indices: np.ndarray,
        translations: np.ndarray,
        ws_center_cart: np.ndarray,
        points_frac_cont: Optional[np.ndarray] = None,
        points_cart: Optional[np.ndarray] = None,
        atom_symbols=None,
        atom_numbers=None,
        atoms_frac: Optional[np.ndarray] = None,
        atoms_cart: Optional[np.ndarray] = None,
        cno_occupations: Optional[np.ndarray] = None,
        cno_indices: Optional[np.ndarray] = None,
        material: Optional[str] = None,
    ) -> "CNOData":
        """Build a WS-mode object from saved CNO rows and their explicit map.

        Unlike :meth:`from_arrays`, the sample count need not equal
        ``prod(grid_shape)``.  This is the required representation for the
        expanded, finite-volume WS quadrature: every CNO row keeps its own
        unwrapped FFT image rather than being folded into a periodic cube.
        """
        cv = np.asarray(cno_values)
        if cv.ndim == 1:
            cv = cv[None, :]
        cv = np.ascontiguousarray(cv, dtype=np.complex128)
        if cv.ndim != 2 or not np.all(np.isfinite(cv.view(np.float64))):
            raise CNODataError("cno_values must be a finite one- or two-dimensional complex array")
        grid_shape = np.asarray(grid_shape, dtype=np.int64).reshape(3)
        lattice = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
        if np.any(grid_shape <= 0) or abs(float(np.linalg.det(lattice))) < 1e-10:
            raise CNODataError("grid_shape must be positive and lattice must be nonsingular")

        n_cno, n_samples = cv.shape
        base_indices = np.asarray(base_indices, dtype=np.int64)
        translations = np.asarray(translations, dtype=np.int64)
        if base_indices.shape != (n_samples, 3) or translations.shape != (n_samples, 3):
            raise CNODataError("base_indices/translations must have shape (n_samples, 3)")
        if np.any(base_indices < 0) or np.any(base_indices >= grid_shape[None, :]):
            raise CNODataError("base_indices lie outside grid_shape")
        frac = base_indices / grid_shape[None, :] + translations
        if points_frac_cont is None:
            points_frac_cont = frac
        else:
            points_frac_cont = np.asarray(points_frac_cont, dtype=np.float64)
            if points_frac_cont.shape != (n_samples, 3) or not np.allclose(
                points_frac_cont, frac, rtol=0.0, atol=2.0e-7
            ):
                raise CNODataError("points_frac_cont disagrees with base_indices/translations")
        if points_cart is None:
            points_cart = points_frac_cont @ lattice
        else:
            points_cart = np.asarray(points_cart, dtype=np.float64)
            if points_cart.shape != (n_samples, 3) or not np.allclose(
                points_cart, points_frac_cont @ lattice, rtol=0.0, atol=2.0e-7
            ):
                raise CNODataError("points_cart disagrees with points_frac_cont/lattice")
        actual = base_indices + translations * grid_shape[None, :]
        if len(np.unique(actual, axis=0)) != n_samples:
            raise CNODataError("WS map contains duplicate unwrapped FFT samples")

        syms = [] if atom_symbols is None else [str(s) for s in atom_symbols]
        natoms = len(syms)
        if atoms_cart is not None:
            atoms_cart = np.asarray(atoms_cart, dtype=np.float64).reshape(-1, 3)
            natoms = len(atoms_cart)
        if atoms_frac is not None:
            atoms_frac = np.asarray(atoms_frac, dtype=np.float64).reshape(-1, 3)
            natoms = len(atoms_frac)
        if atoms_cart is None and atoms_frac is not None:
            atoms_cart = atoms_frac @ lattice
        if atoms_frac is None and atoms_cart is not None:
            atoms_frac = atoms_cart @ np.linalg.inv(lattice)
        if atoms_cart is None:
            atoms_cart = np.zeros((natoms, 3), dtype=np.float64)
            atoms_frac = np.zeros((natoms, 3), dtype=np.float64)
        if atom_numbers is None:
            from ase.data import chemical_symbols

            atom_numbers = np.asarray(
                [chemical_symbols.index(s) if s in chemical_symbols else 0 for s in syms],
                dtype=np.int64,
            )
        else:
            atom_numbers = np.asarray(atom_numbers, dtype=np.int64)
        if cno_occupations is None:
            cno_occupations = np.zeros(n_cno, dtype=np.float64)
        if cno_indices is None:
            cno_indices = np.arange(n_cno, dtype=np.int64)

        ws_center_cart = np.asarray(ws_center_cart, dtype=np.float64).reshape(3)
        return cls(
            format_version="cno-visualizer-v1",
            material=material,
            cno_values=cv,
            cno_occupations=np.asarray(cno_occupations, dtype=np.float64),
            cno_indices=np.asarray(cno_indices, dtype=np.int64),
            grid_shape=grid_shape,
            lattice=lattice,
            atom_symbols=np.asarray(syms, dtype="<U4"),
            atom_numbers=atom_numbers,
            atoms_frac=np.asarray(atoms_frac, dtype=np.float64),
            atoms_cart=np.asarray(atoms_cart, dtype=np.float64),
            ws_enabled=True,
            points_cart=np.asarray(points_cart, dtype=np.float64),
            points_frac_cont=np.asarray(points_frac_cont, dtype=np.float64),
            base_indices=base_indices,
            translations=translations,
            ws_center_cart=ws_center_cart,
            ws_center_frac=ws_center_cart @ np.linalg.inv(lattice),
            source_path=None,
        )

    # ------------------------------------------------------------------ loader
    @classmethod
    def from_npz(cls, path: str | Path) -> "CNOData":
        p = Path(path)
        if not p.is_file():
            raise CNODataError(f"Input .npz file not found: {p}")
        try:
            arr = np.load(p, allow_pickle=False)
        except Exception as exc:  # pragma: no cover - delegated to numpy
            raise CNODataError(f"Failed to read {p}: {exc}") from exc

        try:
            keys = set(arr.files)
            missing = [k for k in _REQUIRED if k not in keys]
            if missing:
                raise CNODataError(
                    f"Missing required fields in {p.name}: {missing}"
                )

            format_version = _as_str(arr["format_version"])
            if format_version not in SUPPORTED_VERSIONS:
                raise CNODataError(
                    f"Unsupported format_version {format_version!r}; "
                    f"supported: {sorted(SUPPORTED_VERSIONS)}"
                )

            cno_values = np.asarray(arr["cno_values"])
            if cno_values.ndim != 2:
                raise CNODataError(
                    f"cno_values must be 2-D, got shape {cno_values.shape}"
                )
            if not np.issubdtype(cno_values.dtype, np.complexfloating):
                raise CNODataError(
                    f"cno_values must be complex, got dtype {cno_values.dtype}"
                )
            if not np.all(np.isfinite(cno_values.view(np.float64))):
                raise CNODataError("cno_values contains NaN or infinite entries.")

            grid_shape = np.asarray(arr["grid_shape"]).astype(np.int64)
            if grid_shape.shape != (3,):
                raise CNODataError(
                    f"grid_shape must have shape (3,), got {grid_shape.shape}"
                )
            if np.any(grid_shape <= 0):
                raise CNODataError(f"grid_shape must be positive, got {grid_shape}")
            expected_Nr = int(np.prod(grid_shape))
            ws_enabled = bool(_scalar(np.asarray(arr["ws_enabled"])))
            if not ws_enabled and cno_values.shape[1] != expected_Nr:
                raise CNODataError(
                    f"cno_values.shape[1] = {cno_values.shape[1]} does not match "
                    f"np.prod(grid_shape) = {expected_Nr} for primitive data"
                )

            cno_occupations = np.asarray(arr["cno_occupations"]).astype(np.float64)
            cno_indices = np.asarray(arr["cno_indices"]).astype(np.int64)
            n_cno = cno_values.shape[0]
            if cno_occupations.shape != (n_cno,):
                raise CNODataError(
                    f"cno_occupations shape {cno_occupations.shape} != ({n_cno},)"
                )
            if cno_indices.shape != (n_cno,):
                raise CNODataError(
                    f"cno_indices shape {cno_indices.shape} != ({n_cno},)"
                )
            if not np.all(np.isfinite(cno_occupations)):
                raise CNODataError("cno_occupations contains NaN/infinite values.")

            lattice = np.asarray(arr["lattice"]).astype(np.float64)
            if lattice.shape != (3, 3):
                raise CNODataError(
                    f"lattice must have shape (3, 3), got {lattice.shape}"
                )
            if not np.all(np.isfinite(lattice)):
                raise CNODataError("lattice contains NaN/infinite values.")
            det = float(np.linalg.det(lattice))
            if abs(det) < 1e-10:
                raise CNODataError(
                    f"lattice is singular (det = {det:.3e}); cannot define a cell."
                )

            atom_symbols = np.asarray(arr["atom_symbols"])
            atom_numbers = np.asarray(arr["atom_numbers"]).astype(np.int64)
            atoms_frac = np.asarray(arr["atoms_frac"]).astype(np.float64)
            atoms_cart = np.asarray(arr["atoms_cart"]).astype(np.float64)
            natoms = atom_symbols.shape[0]
            if atom_numbers.shape != (natoms,):
                raise CNODataError(
                    f"atom_numbers shape {atom_numbers.shape} != ({natoms},)"
                )
            if atoms_frac.shape != (natoms, 3):
                raise CNODataError(
                    f"atoms_frac shape {atoms_frac.shape} != ({natoms}, 3)"
                )
            if atoms_cart.shape != (natoms, 3):
                raise CNODataError(
                    f"atoms_cart shape {atoms_cart.shape} != ({natoms}, 3)"
                )
            if not np.all(np.isfinite(atoms_frac)):
                raise CNODataError("atoms_frac contains NaN/infinite values.")
            if not np.all(np.isfinite(atoms_cart)):
                raise CNODataError("atoms_cart contains NaN/infinite values.")

            (
                points_cart,
                points_frac_cont,
                base_indices,
                translations,
                ws_center_cart,
                ws_center_frac,
            ) = (None, None, None, None, None, None)

            if ws_enabled:
                ws_missing = [k for k in _WS_REQUIRED if k not in keys]
                if ws_missing:
                    raise CNODataError(
                        f"ws_enabled=True but missing WS fields: {ws_missing}"
                    )
                points_cart = np.asarray(arr["points_cart"]).astype(np.float64)
                points_frac_cont = np.asarray(arr["points_frac_cont"]).astype(np.float64)
                base_indices = np.asarray(arr["base_indices"]).astype(np.int64)
                translations = np.asarray(arr["translations"]).astype(np.int64)
                ws_center_cart = np.asarray(arr["ws_center_cart"]).astype(np.float64)

                # Normalize: prefer `ws_center_frac`; fall back to `ws_center_frac_wrapped`.
                if "ws_center_frac" in keys:
                    ws_center_frac = np.asarray(arr["ws_center_frac"]).astype(np.float64)
                elif "ws_center_frac_wrapped" in keys:
                    ws_center_frac = np.asarray(arr["ws_center_frac_wrapped"]).astype(
                        np.float64
                    )

                Nr = int(cno_values.shape[1])
                for name, a, shape in (
                    ("points_cart", points_cart, (Nr, 3)),
                    ("points_frac_cont", points_frac_cont, (Nr, 3)),
                    ("base_indices", base_indices, (Nr, 3)),
                    ("translations", translations, (Nr, 3)),
                ):
                    if a.shape != shape:
                        raise CNODataError(
                            f"{name} shape {a.shape} != {shape}"
                        )
                if ws_center_cart.shape != (3,):
                    raise CNODataError(
                        f"ws_center_cart shape {ws_center_cart.shape} != (3,)"
                    )
                if ws_center_frac is not None and ws_center_frac.shape != (3,):
                    raise CNODataError(
                        f"ws_center_frac shape {ws_center_frac.shape} != (3,)"
                    )
                if not np.all(np.isfinite(points_cart)):
                    raise CNODataError("points_cart contains NaN/infinite values.")
                if not np.all(np.isfinite(points_frac_cont)):
                    raise CNODataError("points_frac_cont contains NaN/infinite values.")
                if np.any(base_indices < 0) or np.any(base_indices >= grid_shape[None, :]):
                    raise CNODataError("base_indices lie outside grid_shape.")
                if not (
                    np.issubdtype(base_indices.dtype, np.integer)
                    and np.issubdtype(translations.dtype, np.integer)
                ):
                    raise CNODataError(
                        "base_indices and translations must be integer arrays."
                    )
                expected_frac = base_indices / grid_shape[None, :] + translations
                if not np.allclose(points_frac_cont, expected_frac, rtol=0.0, atol=2.0e-7):
                    raise CNODataError(
                        "points_frac_cont disagrees with base_indices/translations."
                    )
                if not np.allclose(points_cart, points_frac_cont @ lattice, rtol=0.0, atol=2.0e-7):
                    raise CNODataError("points_cart disagrees with points_frac_cont/lattice.")
                actual = base_indices + translations * grid_shape[None, :]
                if len(np.unique(actual, axis=0)) != Nr:
                    raise CNODataError("WS map contains duplicate unwrapped FFT samples.")

            material: Optional[str] = None
            if "material" in keys:
                material = _as_str(arr["material"]) or None

            return cls(
                format_version=format_version,
                material=material,
                cno_values=cno_values,
                cno_occupations=cno_occupations,
                cno_indices=cno_indices,
                grid_shape=grid_shape,
                lattice=lattice,
                atom_symbols=atom_symbols,
                atom_numbers=atom_numbers,
                atoms_frac=atoms_frac,
                atoms_cart=atoms_cart,
                ws_enabled=ws_enabled,
                points_cart=points_cart,
                points_frac_cont=points_frac_cont,
                base_indices=base_indices,
                translations=translations,
                ws_center_cart=ws_center_cart,
                ws_center_frac=ws_center_frac,
                source_path=p,
            )
        finally:
            arr.close()


__all__ = ["CNOData", "CNODataError", "SUPPORTED_VERSIONS"]
