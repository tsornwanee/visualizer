from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
Bounds = tuple[float, float]
_UNSET = object()


def _coerce_coordinate_array(values: npt.ArrayLike, name: str) -> FloatArray:
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def _coerce_matching_coordinate_array(
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
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def _clamp_progress(progress: float) -> float:
    return float(np.clip(progress, 0.0, 1.0))


def _resolve_x_domain(
    x_values: FloatArray,
    timeline_domain: tuple[float, float] | None,
) -> tuple[float, float]:
    if timeline_domain is None:
        start = float(np.min(x_values))
        end = float(np.max(x_values))
    else:
        start, end = map(float, timeline_domain)
        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError("timeline_domain must contain only finite values.")
        if start > end:
            start, end = end, start

    return start, end


def _normalize_bounds(bounds: Bounds | None, name: str) -> Bounds | None:
    if bounds is None:
        return None

    start, end = map(float, bounds)
    if not np.isfinite(start) or not np.isfinite(end):
        raise ValueError(f"{name} must contain only finite values.")
    if start > end:
        start, end = end, start

    return (start, end)


def _intersect_bounds(primary: Bounds | None, secondary: Bounds | None) -> Bounds | None:
    if primary is None:
        return secondary
    if secondary is None:
        return primary

    start = max(primary[0], secondary[0])
    end = min(primary[1], secondary[1])
    if start > end:
        return None
    return (start, end)


def _bounds_overlap(primary: Bounds | None, secondary: Bounds | None) -> bool:
    if primary is None or secondary is None:
        return True
    return not (primary[1] < secondary[0] or secondary[1] < primary[0])


def _point_is_visible(
    x_value: float,
    y_value: float,
    *,
    domain: Bounds | None,
    value_range: Bounds | None,
) -> bool:
    if domain is not None and not (domain[0] <= x_value <= domain[1]):
        return False
    if value_range is not None and not (value_range[0] <= y_value <= value_range[1]):
        return False
    return True


def _points_close(x0: float, y0: float, x1: float, y1: float) -> bool:
    return bool(np.isclose(x0, x1) and np.isclose(y0, y1))


def _clip_segment_to_window(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    domain: Bounds | None,
    value_range: Bounds | None,
) -> tuple[float, float, float, float] | None:
    dx = x1 - x0
    dy = y1 - y0
    t_min = 0.0
    t_max = 1.0

    constraints: list[tuple[float, float]] = []
    if domain is not None:
        x_min, x_max = domain
        constraints.extend(((-dx, x0 - x_min), (dx, x_max - x0)))
    if value_range is not None:
        y_min, y_max = value_range
        constraints.extend(((-dy, y0 - y_min), (dy, y_max - y0)))

    for p_value, q_value in constraints:
        if np.isclose(p_value, 0.0):
            if q_value < 0.0:
                return None
            continue

        ratio = q_value / p_value
        if p_value < 0.0:
            if ratio > t_max:
                return None
            t_min = max(t_min, ratio)
        else:
            if ratio < t_min:
                return None
            t_max = min(t_max, ratio)

    if t_min > t_max:
        return None

    return (
        x0 + dx * t_min,
        y0 + dy * t_min,
        x0 + dx * t_max,
        y0 + dy * t_max,
    )


def _clip_polyline_to_window(
    x_values: FloatArray,
    y_values: FloatArray,
    *,
    domain: Bounds | None,
    value_range: Bounds | None,
) -> tuple[FloatArray, FloatArray]:
    if x_values.size == 0:
        return x_values.copy(), y_values.copy()

    if domain is None and value_range is None:
        return x_values.copy(), y_values.copy()

    if x_values.size == 1:
        if _point_is_visible(
            float(x_values[0]),
            float(y_values[0]),
            domain=domain,
            value_range=value_range,
        ):
            return x_values.copy(), y_values.copy()
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )

    clipped_x: list[float] = []
    clipped_y: list[float] = []
    segment_open = False

    for index in range(x_values.size - 1):
        clipped_segment = _clip_segment_to_window(
            float(x_values[index]),
            float(y_values[index]),
            float(x_values[index + 1]),
            float(y_values[index + 1]),
            domain=domain,
            value_range=value_range,
        )
        if clipped_segment is None:
            segment_open = False
            continue

        x_start, y_start, x_end, y_end = clipped_segment
        if (
            not segment_open
            or not clipped_x
            or np.isnan(clipped_x[-1])
            or not _points_close(clipped_x[-1], clipped_y[-1], x_start, y_start)
        ):
            if clipped_x and not np.isnan(clipped_x[-1]):
                clipped_x.append(np.nan)
                clipped_y.append(np.nan)
            clipped_x.append(x_start)
            clipped_y.append(y_start)
        if not _points_close(clipped_x[-1], clipped_y[-1], x_end, y_end):
            clipped_x.append(x_end)
            clipped_y.append(y_end)
        segment_open = True

    if not clipped_x:
        return (
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )

    return (
        np.asarray(clipped_x, dtype=float),
        np.asarray(clipped_y, dtype=float),
    )


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


def _transform_array(values: FloatArray, source: Bounds, target: Bounds) -> FloatArray:
    transformed = values.copy()
    visible_mask = ~np.isnan(values)
    if not np.any(visible_mask):
        return transformed

    source_width = source[1] - source[0]
    if np.isclose(source_width, 0.0):
        raise ValueError("Cannot transform through a zero-width source interval.")

    transformed[visible_mask] = target[0] + (
        (values[visible_mask] - source[0]) * (target[1] - target[0]) / source_width
    )
    return transformed


@dataclass(frozen=True)
class Theater:
    """A rectangular affine-mapped subspace for curves and fills."""

    theater_id: str
    xlim: Bounds
    ylim: Bounds
    local_xlim: Bounds = (0.0, 1.0)
    local_ylim: Bounds = (0.0, 1.0)
    facecolor: str | None = None
    edgecolor: str | None = None
    alpha: float | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    patch_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "xlim", _normalize_bounds(self.xlim, "xlim"))
        object.__setattr__(self, "ylim", _normalize_bounds(self.ylim, "ylim"))
        object.__setattr__(self, "local_xlim", _normalize_bounds(self.local_xlim, "local_xlim"))
        object.__setattr__(self, "local_ylim", _normalize_bounds(self.local_ylim, "local_ylim"))
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))
        object.__setattr__(self, "patch_kwargs", dict(self.patch_kwargs))

        if self.linewidth is not None and self.linewidth < 0:
            raise ValueError("linewidth must be non-negative.")
        if np.isclose(self.xlim[0], self.xlim[1]) or np.isclose(self.ylim[0], self.ylim[1]):
            raise ValueError("Theater actual bounds must have non-zero width and height.")
        if np.isclose(self.local_xlim[0], self.local_xlim[1]) or np.isclose(
            self.local_ylim[0], self.local_ylim[1]
        ):
            raise ValueError("Theater local bounds must have non-zero width and height.")

    def mpl_patch_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self.patch_kwargs)
        if self.facecolor is not None:
            kwargs["facecolor"] = self.facecolor
        if self.edgecolor is not None:
            kwargs["edgecolor"] = self.edgecolor
        if self.alpha is not None:
            kwargs["alpha"] = self.alpha
        if self.linestyle is not None:
            kwargs["linestyle"] = self.linestyle
        if self.linewidth is not None:
            kwargs["linewidth"] = self.linewidth
        return kwargs

    def copy_with(
        self,
        *,
        xlim: Bounds | None = None,
        ylim: Bounds | None = None,
        local_xlim: Bounds | None = None,
        local_ylim: Bounds | None = None,
        facecolor: str | None = None,
        edgecolor: str | None = None,
        alpha: float | None = None,
        linestyle: str | None = None,
        linewidth: float | None = None,
        patch_kwargs: Mapping[str, Any] | None = None,
    ) -> Theater:
        return Theater(
            theater_id=self.theater_id,
            xlim=self.xlim if xlim is None else xlim,
            ylim=self.ylim if ylim is None else ylim,
            local_xlim=self.local_xlim if local_xlim is None else local_xlim,
            local_ylim=self.local_ylim if local_ylim is None else local_ylim,
            facecolor=self.facecolor if facecolor is None else facecolor,
            edgecolor=self.edgecolor if edgecolor is None else edgecolor,
            alpha=self.alpha if alpha is None else alpha,
            linestyle=self.linestyle if linestyle is None else linestyle,
            linewidth=self.linewidth if linewidth is None else linewidth,
            patch_kwargs=self.patch_kwargs if patch_kwargs is None else patch_kwargs,
        )

    @property
    def width(self) -> float:
        return float(self.xlim[1] - self.xlim[0])

    @property
    def height(self) -> float:
        return float(self.ylim[1] - self.ylim[0])

    def transform_x(self, x_values: FloatArray) -> FloatArray:
        return _transform_array(np.asarray(x_values, dtype=float), self.local_xlim, self.xlim)

    def transform_y(self, y_values: FloatArray) -> FloatArray:
        return _transform_array(np.asarray(y_values, dtype=float), self.local_ylim, self.ylim)

    def transform_points(
        self,
        x_values: FloatArray,
        y_values: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        return self.transform_x(x_values), self.transform_y(y_values)

    def visible_extents(self) -> tuple[float, float, float, float]:
        return (self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1])


@dataclass(frozen=True)
class Curve:
    """A single curve in plot coordinates."""

    curve_id: str
    x: npt.ArrayLike
    y: npt.ArrayLike
    color: str | None = None
    alpha: float | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    domain: Bounds | None = None
    value_range: Bounds | None = None
    theater_id: str | None = None
    line_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x_array = _coerce_coordinate_array(self.x, "x")
        y_array = _coerce_coordinate_array(self.y, "y")

        if x_array.shape != y_array.shape:
            raise ValueError("x and y must have the same shape.")

        object.__setattr__(self, "x", x_array)
        object.__setattr__(self, "y", y_array)
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))
        object.__setattr__(self, "domain", _normalize_bounds(self.domain, "domain"))
        object.__setattr__(self, "value_range", _normalize_bounds(self.value_range, "value_range"))
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
        domain: Bounds | None = None,
        value_range: Bounds | None = None,
        theater_id: str | None | object = _UNSET,
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
            domain=self.domain if domain is None else domain,
            value_range=self.value_range if value_range is None else value_range,
            theater_id=self.theater_id if theater_id is _UNSET else theater_id,
            line_kwargs=self.line_kwargs if line_kwargs is None else line_kwargs,
        )

    def clipped_line_data(
        self,
        theater: Theater | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        if not self._has_visible_window(theater):
            return (
                np.asarray([], dtype=float),
                np.asarray([], dtype=float),
            )
        clipped_x, clipped_y = _clip_polyline_to_window(
            self.x,
            self.y,
            domain=self._effective_domain(theater),
            value_range=self._effective_value_range(theater),
        )
        if theater is None or clipped_x.size == 0:
            return clipped_x, clipped_y
        return theater.transform_points(clipped_x, clipped_y)

    def point_is_visible(
        self,
        x_value: float,
        y_value: float,
        theater: Theater | None = None,
    ) -> bool:
        if not self._has_visible_window(theater):
            return False
        return _point_is_visible(
            x_value,
            y_value,
            domain=self._effective_domain(theater),
            value_range=self._effective_value_range(theater),
        )

    def visible_extents(self, theater: Theater | None = None) -> tuple[float, float, float, float] | None:
        clipped_x, clipped_y = self.clipped_line_data(theater)
        if clipped_x.size == 0:
            return None

        visible_mask = ~(np.isnan(clipped_x) | np.isnan(clipped_y))
        if not np.any(visible_mask):
            return None

        visible_x = clipped_x[visible_mask]
        visible_y = clipped_y[visible_mask]
        return (
            float(np.min(visible_x)),
            float(np.max(visible_x)),
            float(np.min(visible_y)),
            float(np.max(visible_y)),
        )

    def actual_point(
        self,
        x_value: float,
        y_value: float,
        theater: Theater | None = None,
    ) -> tuple[float, float]:
        if theater is None:
            return (float(x_value), float(y_value))

        x_array, y_array = theater.transform_points(
            np.asarray([x_value], dtype=float),
            np.asarray([y_value], dtype=float),
        )
        return (float(x_array[0]), float(y_array[0]))

    def reveal_until(self, progress: float) -> Curve:
        """Reveal points from left to right based on normalized progress."""

        return self.reveal_by_progress(progress)

    def reveal_by_progress(
        self,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None = None,
        direction: str = "forward",
    ) -> Curve:
        threshold = _clamp_progress(progress)
        if self.is_empty:
            return self

        x_min, x_max = _resolve_x_domain(self.x, timeline_domain)
        if np.isclose(x_min, x_max):
            mask = np.full(self.x.shape, threshold > 0.0, dtype=bool)
        else:
            if direction == "forward":
                x_threshold = x_min + (x_max - x_min) * threshold
                mask = self.x <= x_threshold
            elif direction == "backward":
                x_threshold = x_max - (x_max - x_min) * threshold
                mask = self.x >= x_threshold
            else:
                raise ValueError("direction must be 'forward' or 'backward'.")
        return self.copy_with(x=self.x[mask], y=self.y[mask])

    def hide_until(self, progress: float) -> Curve:
        """Hide points from left to right based on normalized progress."""

        return self.hide_by_progress(progress)

    def hide_by_progress(
        self,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None = None,
        direction: str = "forward",
    ) -> Curve:
        threshold = _clamp_progress(progress)
        if self.is_empty:
            return self

        x_min, x_max = _resolve_x_domain(self.x, timeline_domain)
        if np.isclose(x_min, x_max):
            mask = np.full(self.x.shape, threshold < 1.0, dtype=bool)
        else:
            if direction == "forward":
                x_threshold = x_min + (x_max - x_min) * threshold
                mask = self.x > x_threshold
            elif direction == "backward":
                x_threshold = x_max - (x_max - x_min) * threshold
                mask = self.x < x_threshold
            else:
                raise ValueError("direction must be 'forward' or 'backward'.")
        return self.copy_with(x=self.x[mask], y=self.y[mask])

    def _effective_domain(self, theater: Theater | None) -> Bounds | None:
        if theater is None:
            return self.domain
        return _intersect_bounds(self.domain, theater.local_xlim)

    def _effective_value_range(self, theater: Theater | None) -> Bounds | None:
        if theater is None:
            return self.value_range
        return _intersect_bounds(self.value_range, theater.local_ylim)

    def _has_visible_window(self, theater: Theater | None) -> bool:
        if theater is None:
            return True
        return _bounds_overlap(self.domain, theater.local_xlim) and _bounds_overlap(
            self.value_range,
            theater.local_ylim,
        )


@dataclass(frozen=True)
class FillBetweenArea:
    """A filled region between y1(x) and y2(x)."""

    fill_id: str
    x: npt.ArrayLike
    y1: npt.ArrayLike
    y2: npt.ArrayLike | float
    color: str | None = None
    alpha: float | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    domain: Bounds | None = None
    value_range: Bounds | None = None
    theater_id: str | None = None
    fill_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x_array = _coerce_coordinate_array(self.x, "x")
        y1_array = _coerce_coordinate_array(self.y1, "y1")

        if x_array.shape != y1_array.shape:
            raise ValueError("x and y1 must have the same shape.")

        y2_array = _coerce_matching_coordinate_array(self.y2, "y2", x_array.shape)

        object.__setattr__(self, "x", x_array)
        object.__setattr__(self, "y1", y1_array)
        object.__setattr__(self, "y2", y2_array)
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))
        object.__setattr__(self, "domain", _normalize_bounds(self.domain, "domain"))
        object.__setattr__(self, "value_range", _normalize_bounds(self.value_range, "value_range"))
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
        domain: Bounds | None = None,
        value_range: Bounds | None = None,
        theater_id: str | None | object = _UNSET,
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
            domain=self.domain if domain is None else domain,
            value_range=self.value_range if value_range is None else value_range,
            theater_id=self.theater_id if theater_id is _UNSET else theater_id,
            fill_kwargs=self.fill_kwargs if fill_kwargs is None else fill_kwargs,
        )

    def clipped_fill_data(
        self,
        theater: Theater | None = None,
    ) -> tuple[FloatArray, FloatArray, FloatArray, npt.NDArray[np.bool_]]:
        if self.is_empty:
            empty = np.asarray([], dtype=float)
            return empty, empty, empty, np.asarray([], dtype=bool)
        if not self._has_visible_window(theater):
            empty = np.asarray([], dtype=float)
            return empty, empty, empty, np.asarray([], dtype=bool)

        where = np.ones(self.x.shape, dtype=bool)
        effective_domain = self._effective_domain(theater)
        effective_value_range = self._effective_value_range(theater)
        if effective_domain is not None:
            where &= (self.x >= effective_domain[0]) & (self.x <= effective_domain[1])

        clipped_y1 = self.y1.copy()
        clipped_y2 = self.y2.copy()
        if effective_value_range is not None:
            clipped_y1 = np.clip(clipped_y1, effective_value_range[0], effective_value_range[1])
            clipped_y2 = np.clip(clipped_y2, effective_value_range[0], effective_value_range[1])

        clipped_x = self.x.copy()
        if theater is not None:
            clipped_x = theater.transform_x(clipped_x)
            clipped_y1 = theater.transform_y(clipped_y1)
            clipped_y2 = theater.transform_y(clipped_y2)

        return clipped_x, clipped_y1, clipped_y2, where

    def visible_extents(self, theater: Theater | None = None) -> tuple[float, float, float, float] | None:
        if self.is_empty:
            return None

        x_values, y1_values, y2_values, where = self.clipped_fill_data(theater)
        effective_domain = self._effective_domain(theater)
        if effective_domain is not None and theater is None:
            x_min = max(float(np.min(self.x)), effective_domain[0])
            x_max = min(float(np.max(self.x)), effective_domain[1])
            if x_min > x_max:
                return None
        else:
            if not np.any(where):
                return None
            visible_x = x_values[where]
            x_min = float(np.min(visible_x))
            x_max = float(np.max(visible_x))

        if not np.any(where):
            return None

        visible_y = np.concatenate((y1_values[where], y2_values[where]))
        return (
            x_min,
            x_max,
            float(np.min(visible_y)),
            float(np.max(visible_y)),
        )

    def _effective_domain(self, theater: Theater | None) -> Bounds | None:
        if theater is None:
            return self.domain
        return _intersect_bounds(self.domain, theater.local_xlim)

    def _effective_value_range(self, theater: Theater | None) -> Bounds | None:
        if theater is None:
            return self.value_range
        return _intersect_bounds(self.value_range, theater.local_ylim)

    def _has_visible_window(self, theater: Theater | None) -> bool:
        if theater is None:
            return True
        return _bounds_overlap(self.domain, theater.local_xlim) and _bounds_overlap(
            self.value_range,
            theater.local_ylim,
        )

    def reveal_until(self, progress: float) -> FillBetweenArea:
        return self.reveal_by_progress(progress)

    def reveal_by_progress(
        self,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None = None,
        direction: str = "forward",
    ) -> FillBetweenArea:
        threshold = _clamp_progress(progress)
        if self.is_empty:
            return self

        x_min, x_max = _resolve_x_domain(self.x, timeline_domain)
        if np.isclose(x_min, x_max):
            mask = np.full(self.x.shape, threshold > 0.0, dtype=bool)
        else:
            if direction == "forward":
                x_threshold = x_min + (x_max - x_min) * threshold
                mask = self.x <= x_threshold
            elif direction == "backward":
                x_threshold = x_max - (x_max - x_min) * threshold
                mask = self.x >= x_threshold
            else:
                raise ValueError("direction must be 'forward' or 'backward'.")
        return self.copy_with(x=self.x[mask], y1=self.y1[mask], y2=self.y2[mask])

    def hide_until(self, progress: float) -> FillBetweenArea:
        return self.hide_by_progress(progress)

    def hide_by_progress(
        self,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None = None,
        direction: str = "forward",
    ) -> FillBetweenArea:
        threshold = _clamp_progress(progress)
        if self.is_empty:
            return self

        x_min, x_max = _resolve_x_domain(self.x, timeline_domain)
        if np.isclose(x_min, x_max):
            mask = np.full(self.x.shape, threshold < 1.0, dtype=bool)
        else:
            if direction == "forward":
                x_threshold = x_min + (x_max - x_min) * threshold
                mask = self.x > x_threshold
            elif direction == "backward":
                x_threshold = x_max - (x_max - x_min) * threshold
                mask = self.x < x_threshold
            else:
                raise ValueError("direction must be 'forward' or 'backward'.")
        return self.copy_with(x=self.x[mask], y1=self.y1[mask], y2=self.y2[mask])


@dataclass(frozen=True)
class Scene:
    """A snapshot of all persistent visual elements after a transition."""

    curves: Mapping[str, Curve] = field(default_factory=dict)
    fills: Mapping[str, FillBetweenArea] = field(default_factory=dict)
    theaters: Mapping[str, Theater] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_curves: dict[str, Curve] = {}
        normalized_fills: dict[str, FillBetweenArea] = {}
        normalized_theaters: dict[str, Theater] = {}

        for theater_id, theater in dict(self.theaters).items():
            if theater_id != theater.theater_id:
                raise ValueError("Scene theater keys must match each theater's theater_id.")
            normalized_theaters[theater_id] = theater

        for curve_id, curve in dict(self.curves).items():
            if curve_id != curve.curve_id:
                raise ValueError("Scene curve keys must match each curve's curve_id.")
            if curve.theater_id is not None and curve.theater_id not in normalized_theaters:
                raise ValueError(f"Curve {curve.curve_id!r} references missing theater {curve.theater_id!r}.")
            normalized_curves[curve_id] = curve

        for fill_id, fill in dict(self.fills).items():
            if fill_id != fill.fill_id:
                raise ValueError("Scene fill keys must match each fill's fill_id.")
            if fill.theater_id is not None and fill.theater_id not in normalized_theaters:
                raise ValueError(f"Fill {fill.fill_id!r} references missing theater {fill.theater_id!r}.")
            normalized_fills[fill_id] = fill

        object.__setattr__(self, "curves", normalized_curves)
        object.__setattr__(self, "fills", normalized_fills)
        object.__setattr__(self, "theaters", normalized_theaters)

    def __len__(self) -> int:
        return len(self.curves) + len(self.fills) + len(self.theaters)

    def contains(self, curve_id: str) -> bool:
        return self.contains_curve(curve_id)

    def contains_curve(self, curve_id: str) -> bool:
        return curve_id in self.curves

    def contains_fill(self, fill_id: str) -> bool:
        return fill_id in self.fills

    def contains_theater(self, theater_id: str) -> bool:
        return theater_id in self.theaters

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

    def get_theater(self, theater_id: str) -> Theater:
        try:
            return self.theaters[theater_id]
        except KeyError as exc:
            raise KeyError(f"Theater {theater_id!r} does not exist in the scene.") from exc

    def add_curve(self, curve: Curve) -> Scene:
        if self.contains_curve(curve.curve_id):
            raise ValueError(f"Curve {curve.curve_id!r} already exists in the scene.")

        updated = dict(self.curves)
        updated[curve.curve_id] = curve
        return Scene(curves=updated, fills=self.fills, theaters=self.theaters)

    def update_curve(self, curve: Curve) -> Scene:
        if not self.contains_curve(curve.curve_id):
            raise ValueError(f"Curve {curve.curve_id!r} does not exist in the scene.")

        updated = dict(self.curves)
        updated[curve.curve_id] = curve
        return Scene(curves=updated, fills=self.fills, theaters=self.theaters)

    def remove_curve(self, curve_id: str) -> Scene:
        if not self.contains_curve(curve_id):
            raise ValueError(f"Curve {curve_id!r} does not exist in the scene.")

        updated = dict(self.curves)
        updated.pop(curve_id)
        return Scene(curves=updated, fills=self.fills, theaters=self.theaters)

    def add_fill(self, fill: FillBetweenArea) -> Scene:
        if self.contains_fill(fill.fill_id):
            raise ValueError(f"Fill {fill.fill_id!r} already exists in the scene.")

        updated = dict(self.fills)
        updated[fill.fill_id] = fill
        return Scene(curves=self.curves, fills=updated, theaters=self.theaters)

    def update_fill(self, fill: FillBetweenArea) -> Scene:
        if not self.contains_fill(fill.fill_id):
            raise ValueError(f"Fill {fill.fill_id!r} does not exist in the scene.")

        updated = dict(self.fills)
        updated[fill.fill_id] = fill
        return Scene(curves=self.curves, fills=updated, theaters=self.theaters)

    def remove_fill(self, fill_id: str) -> Scene:
        if not self.contains_fill(fill_id):
            raise ValueError(f"Fill {fill_id!r} does not exist in the scene.")

        updated = dict(self.fills)
        updated.pop(fill_id)
        return Scene(curves=self.curves, fills=updated, theaters=self.theaters)

    def add_theater(self, theater: Theater) -> Scene:
        if self.contains_theater(theater.theater_id):
            raise ValueError(f"Theater {theater.theater_id!r} already exists in the scene.")

        updated = dict(self.theaters)
        updated[theater.theater_id] = theater
        return Scene(curves=self.curves, fills=self.fills, theaters=updated)

    def update_theater(self, theater: Theater) -> Scene:
        if not self.contains_theater(theater.theater_id):
            raise ValueError(f"Theater {theater.theater_id!r} does not exist in the scene.")

        updated = dict(self.theaters)
        updated[theater.theater_id] = theater
        return Scene(curves=self.curves, fills=self.fills, theaters=updated)

    def remove_theater(self, theater_id: str) -> Scene:
        if not self.contains_theater(theater_id):
            raise ValueError(f"Theater {theater_id!r} does not exist in the scene.")
        if any(curve.theater_id == theater_id for curve in self.curves.values()):
            raise ValueError(f"Cannot remove theater {theater_id!r} while a curve still references it.")
        if any(fill.theater_id == theater_id for fill in self.fills.values()):
            raise ValueError(f"Cannot remove theater {theater_id!r} while a fill still references it.")

        updated = dict(self.theaters)
        updated.pop(theater_id)
        return Scene(curves=self.curves, fills=self.fills, theaters=updated)
