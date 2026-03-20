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


def _coerce_matching_unit_interval_array(
    values: npt.ArrayLike | float,
    name: str,
    shape: tuple[int, ...],
) -> FloatArray:
    array = np.asarray(values, dtype=float)

    if array.ndim == 0:
        array = np.full(shape, float(array), dtype=float)
    elif array.ndim != 1:
        raise ValueError(f"{name} must be a scalar or a one-dimensional array.")

    if array.shape != shape:
        raise ValueError(f"{name} must match the reference shape {shape}.")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must contain values inside [0, 1].")

    return array


def _clamp_progress(progress: float) -> float:
    return float(np.clip(progress, 0.0, 1.0))


def _validate_alpha(alpha: float | None) -> float | None:
    if alpha is None:
        return None
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be inside [0, 1].")
    return alpha


def _merge_style_kwargs(
    base_kwargs: Mapping[str, Any],
    *,
    color: str | None,
    alpha: float | None,
    linestyle: str | None,
    linewidth: float | None,
) -> dict[str, Any]:
    kwargs = dict(base_kwargs)

    if color is not None:
        kwargs["color"] = color
    if alpha is not None:
        kwargs["alpha"] = alpha
    if linestyle is not None:
        kwargs["linestyle"] = linestyle
    if linewidth is not None:
        kwargs["linewidth"] = linewidth

    return kwargs


@dataclass(frozen=True)
class Curve:
    """A single curve in normalized [0, 1] x [0, 1] coordinates."""

    curve_id: str
    x: npt.ArrayLike
    y: npt.ArrayLike
    color: str | None = None
    alpha: float | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    line_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x_array = _coerce_unit_interval_array(self.x, "x")
        y_array = _coerce_unit_interval_array(self.y, "y")

        if x_array.shape != y_array.shape:
            raise ValueError("x and y must have the same shape.")

        object.__setattr__(self, "x", x_array)
        object.__setattr__(self, "y", y_array)
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))
        object.__setattr__(self, "line_kwargs", dict(self.line_kwargs))

    @property
    def is_empty(self) -> bool:
        return self.x.size == 0

    def mpl_line_kwargs(self) -> dict[str, Any]:
        return _merge_style_kwargs(
            self.line_kwargs,
            color=self.color,
            alpha=self.alpha,
            linestyle=self.linestyle,
            linewidth=self.linewidth,
        )

    def copy_with(
        self,
        *,
        x: npt.ArrayLike | None = None,
        y: npt.ArrayLike | None = None,
        color: str | None = None,
        alpha: float | None = None,
        linestyle: str | None = None,
        linewidth: float | None = None,
        line_kwargs: Mapping[str, Any] | None = None,
    ) -> Curve:
        return Curve(
            curve_id=self.curve_id,
            x=self.x if x is None else x,
            y=self.y if y is None else y,
            color=self.color if color is None else color,
            alpha=self.alpha if alpha is None else alpha,
            linestyle=self.linestyle if linestyle is None else linestyle,
            linewidth=self.linewidth if linewidth is None else linewidth,
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
class FillBetweenArea:
    """A normalized filled region between y1(x) and y2(x)."""

    fill_id: str
    x: npt.ArrayLike
    y1: npt.ArrayLike
    y2: npt.ArrayLike | float
    color: str | None = None
    alpha: float | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    fill_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x_array = _coerce_unit_interval_array(self.x, "x")
        y1_array = _coerce_unit_interval_array(self.y1, "y1")

        if x_array.shape != y1_array.shape:
            raise ValueError("x and y1 must have the same shape.")

        y2_array = _coerce_matching_unit_interval_array(self.y2, "y2", x_array.shape)

        object.__setattr__(self, "x", x_array)
        object.__setattr__(self, "y1", y1_array)
        object.__setattr__(self, "y2", y2_array)
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))
        object.__setattr__(self, "fill_kwargs", dict(self.fill_kwargs))

    @property
    def is_empty(self) -> bool:
        return self.x.size == 0

    def mpl_fill_kwargs(self) -> dict[str, Any]:
        return _merge_style_kwargs(
            self.fill_kwargs,
            color=self.color,
            alpha=self.alpha,
            linestyle=self.linestyle,
            linewidth=self.linewidth,
        )

    def copy_with(
        self,
        *,
        x: npt.ArrayLike | None = None,
        y1: npt.ArrayLike | None = None,
        y2: npt.ArrayLike | float | None = None,
        color: str | None = None,
        alpha: float | None = None,
        linestyle: str | None = None,
        linewidth: float | None = None,
        fill_kwargs: Mapping[str, Any] | None = None,
    ) -> FillBetweenArea:
        return FillBetweenArea(
            fill_id=self.fill_id,
            x=self.x if x is None else x,
            y1=self.y1 if y1 is None else y1,
            y2=self.y2 if y2 is None else y2,
            color=self.color if color is None else color,
            alpha=self.alpha if alpha is None else alpha,
            linestyle=self.linestyle if linestyle is None else linestyle,
            linewidth=self.linewidth if linewidth is None else linewidth,
            fill_kwargs=self.fill_kwargs if fill_kwargs is None else fill_kwargs,
        )

    def reveal_until(self, progress: float) -> FillBetweenArea:
        threshold = _clamp_progress(progress)
        mask = self.x <= threshold
        return self.copy_with(x=self.x[mask], y1=self.y1[mask], y2=self.y2[mask])

    def hide_until(self, progress: float) -> FillBetweenArea:
        threshold = _clamp_progress(progress)
        mask = self.x > threshold
        return self.copy_with(x=self.x[mask], y1=self.y1[mask], y2=self.y2[mask])


@dataclass(frozen=True)
class Scene:
    """A snapshot of all persistent visual elements after a transition."""

    curves: Mapping[str, Curve] = field(default_factory=dict)
    fills: Mapping[str, FillBetweenArea] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_curves: dict[str, Curve] = {}
        normalized_fills: dict[str, FillBetweenArea] = {}

        for curve_id, curve in dict(self.curves).items():
            if curve_id != curve.curve_id:
                raise ValueError("Scene curve keys must match each curve's curve_id.")
            normalized_curves[curve_id] = curve

        for fill_id, fill in dict(self.fills).items():
            if fill_id != fill.fill_id:
                raise ValueError("Scene fill keys must match each fill's fill_id.")
            normalized_fills[fill_id] = fill

        object.__setattr__(self, "curves", normalized_curves)
        object.__setattr__(self, "fills", normalized_fills)

    def __len__(self) -> int:
        return len(self.curves) + len(self.fills)

    def contains(self, curve_id: str) -> bool:
        return self.contains_curve(curve_id)

    def contains_curve(self, curve_id: str) -> bool:
        return curve_id in self.curves

    def contains_fill(self, fill_id: str) -> bool:
        return fill_id in self.fills

    def get_curve(self, curve_id: str) -> Curve:
        try:
            return self.curves[curve_id]
        except KeyError as exc:
            raise KeyError(f"Curve {curve_id!r} does not exist in the scene.") from exc

    def get_fill(self, fill_id: str) -> FillBetweenArea:
        try:
            return self.fills[fill_id]
        except KeyError as exc:
            raise KeyError(f"Fill {fill_id!r} does not exist in the scene.") from exc

    def add_curve(self, curve: Curve) -> Scene:
        if self.contains_curve(curve.curve_id):
            raise ValueError(f"Curve {curve.curve_id!r} already exists in the scene.")

        updated = dict(self.curves)
        updated[curve.curve_id] = curve
        return Scene(curves=updated, fills=self.fills)

    def update_curve(self, curve: Curve) -> Scene:
        if not self.contains_curve(curve.curve_id):
            raise ValueError(f"Curve {curve.curve_id!r} does not exist in the scene.")

        updated = dict(self.curves)
        updated[curve.curve_id] = curve
        return Scene(curves=updated, fills=self.fills)

    def remove_curve(self, curve_id: str) -> Scene:
        if not self.contains_curve(curve_id):
            raise ValueError(f"Curve {curve_id!r} does not exist in the scene.")

        updated = dict(self.curves)
        updated.pop(curve_id)
        return Scene(curves=updated, fills=self.fills)

    def add_fill(self, fill: FillBetweenArea) -> Scene:
        if self.contains_fill(fill.fill_id):
            raise ValueError(f"Fill {fill.fill_id!r} already exists in the scene.")

        updated = dict(self.fills)
        updated[fill.fill_id] = fill
        return Scene(curves=self.curves, fills=updated)

    def update_fill(self, fill: FillBetweenArea) -> Scene:
        if not self.contains_fill(fill.fill_id):
            raise ValueError(f"Fill {fill.fill_id!r} does not exist in the scene.")

        updated = dict(self.fills)
        updated[fill.fill_id] = fill
        return Scene(curves=self.curves, fills=updated)

    def remove_fill(self, fill_id: str) -> Scene:
        if not self.contains_fill(fill_id):
            raise ValueError(f"Fill {fill_id!r} does not exist in the scene.")

        updated = dict(self.fills)
        updated.pop(fill_id)
        return Scene(curves=self.curves, fills=updated)
