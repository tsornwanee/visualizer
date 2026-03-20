from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


def _coerce_unit_interval_array(values: npt.ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must contain values inside [0, 1].")

    return array


def _clamp_progress(progress: float) -> float:
    return float(np.clip(progress, 0.0, 1.0))


@dataclass(frozen=True)
class Curve:
    """A single curve in normalized [0, 1] x [0, 1] coordinates."""

    curve_id: str
    x: npt.ArrayLike
    y: npt.ArrayLike
    line_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x_array = _coerce_unit_interval_array(self.x, "x")
        y_array = _coerce_unit_interval_array(self.y, "y")

        if x_array.shape != y_array.shape:
            raise ValueError("x and y must have the same shape.")

        object.__setattr__(self, "x", x_array)
        object.__setattr__(self, "y", y_array)
        object.__setattr__(self, "line_kwargs", dict(self.line_kwargs))

    @property
    def is_empty(self) -> bool:
        return self.x.size == 0

    def copy_with(
        self,
        *,
        x: npt.ArrayLike | None = None,
        y: npt.ArrayLike | None = None,
        line_kwargs: Mapping[str, Any] | None = None,
    ) -> Curve:
        return Curve(
            curve_id=self.curve_id,
            x=self.x if x is None else x,
            y=self.y if y is None else y,
            line_kwargs=self.line_kwargs if line_kwargs is None else line_kwargs,
        )

    def reveal_until(self, progress: float) -> Curve:
        """Reveal points with x <= progress."""

        threshold = _clamp_progress(progress)
        mask = self.x <= threshold
        return self.copy_with(x=self.x[mask], y=self.y[mask])

    def hide_until(self, progress: float) -> Curve:
        """Hide points with x <= progress."""

        threshold = _clamp_progress(progress)
        mask = self.x > threshold
        return self.copy_with(x=self.x[mask], y=self.y[mask])


@dataclass(frozen=True)
class Scene:
    """A snapshot of all curves visible after a transition."""

    curves: Mapping[str, Curve] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[str, Curve] = {}

        for curve_id, curve in dict(self.curves).items():
            if curve_id != curve.curve_id:
                raise ValueError("Scene keys must match each curve's curve_id.")
            normalized[curve_id] = curve

        object.__setattr__(self, "curves", normalized)

    def __len__(self) -> int:
        return len(self.curves)

    def contains(self, curve_id: str) -> bool:
        return curve_id in self.curves

    def get_curve(self, curve_id: str) -> Curve:
        try:
            return self.curves[curve_id]
        except KeyError as exc:
            raise KeyError(f"Curve {curve_id!r} does not exist in the scene.") from exc

    def add_curve(self, curve: Curve) -> Scene:
        if self.contains(curve.curve_id):
            raise ValueError(f"Curve {curve.curve_id!r} already exists in the scene.")

        updated = dict(self.curves)
        updated[curve.curve_id] = curve
        return Scene(updated)

    def update_curve(self, curve: Curve) -> Scene:
        if not self.contains(curve.curve_id):
            raise ValueError(f"Curve {curve.curve_id!r} does not exist in the scene.")

        updated = dict(self.curves)
        updated[curve.curve_id] = curve
        return Scene(updated)

    def remove_curve(self, curve_id: str) -> Scene:
        if not self.contains(curve_id):
            raise ValueError(f"Curve {curve_id!r} does not exist in the scene.")

        updated = dict(self.curves)
        updated.pop(curve_id)
        return Scene(updated)
