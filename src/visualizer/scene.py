from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]
Bounds = tuple[float, float]
_SCALAR_TOLERANCE = 1e-12


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
    return abs(x0 - x1) <= _SCALAR_TOLERANCE and abs(y0 - y1) <= _SCALAR_TOLERANCE


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
        if abs(p_value) <= _SCALAR_TOLERANCE:
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
        return x_values, y_values

    if domain is None and value_range is None:
        return x_values, y_values

    if x_values.size == 1:
        if _point_is_visible(
            float(x_values[0]),
            float(y_values[0]),
            domain=domain,
            value_range=value_range,
        ):
            return x_values, y_values
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


def _merge_text_kwargs(
    base_kwargs: Mapping[str, Any],
    *,
    color: str | None,
    alpha: float | None,
    fontsize: float | None,
    ha: str | None,
    va: str | None,
    rotation: float | None,
) -> dict[str, Any]:
    kwargs = dict(base_kwargs)

    if color is not None:
        kwargs["color"] = color
    if alpha is not None:
        kwargs["alpha"] = alpha
    if fontsize is not None:
        kwargs["fontsize"] = fontsize
    if ha is not None:
        kwargs["ha"] = ha
    if va is not None:
        kwargs["va"] = va
    if rotation is not None:
        kwargs["rotation"] = rotation

    return kwargs


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
    line_kwargs: Mapping[str, Any] = field(default_factory=dict)
    _line_style_cache: dict[str, Any] | None = field(default=None, init=False, repr=False, compare=False)
    _clipped_line_cache: tuple[FloatArray, FloatArray] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

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
        if self._line_style_cache is None:
            object.__setattr__(
                self,
                "_line_style_cache",
                _merge_style_kwargs(
                    self.line_kwargs,
                    color=self.color,
                    alpha=self.alpha,
                    linestyle=self.linestyle,
                    linewidth=self.linewidth,
                ),
            )
        return self._line_style_cache

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
            line_kwargs=self.line_kwargs if line_kwargs is None else line_kwargs,
        )

    def clipped_line_data(self) -> tuple[FloatArray, FloatArray]:
        if self._clipped_line_cache is None:
            object.__setattr__(
                self,
                "_clipped_line_cache",
                _clip_polyline_to_window(
                    self.x,
                    self.y,
                    domain=self.domain,
                    value_range=self.value_range,
                ),
            )
        return self._clipped_line_cache

    def point_is_visible(self, x_value: float, y_value: float) -> bool:
        return _point_is_visible(
            x_value,
            y_value,
            domain=self.domain,
            value_range=self.value_range,
        )

    def visible_extents(self) -> tuple[float, float, float, float] | None:
        clipped_x, clipped_y = self.clipped_line_data()
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


@dataclass(frozen=True)
class Scatter:
    """A scatter layer in plot coordinates."""

    scatter_id: str
    x: npt.ArrayLike
    y: npt.ArrayLike
    color: Any | None = None
    alpha: float | None = None
    marker: Any = "o"
    size: npt.ArrayLike | float = 36.0
    linewidth: float | None = None
    edgecolor: Any | None = None
    domain: Bounds | None = None
    value_range: Bounds | None = None
    scatter_kwargs: Mapping[str, Any] = field(default_factory=dict)
    _clipped_scatter_cache: tuple[FloatArray, FloatArray, FloatArray] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        x_array = _coerce_coordinate_array(self.x, "x")
        y_array = _coerce_coordinate_array(self.y, "y")

        if x_array.shape != y_array.shape:
            raise ValueError("x and y must have the same shape.")
        if self.linewidth is not None and self.linewidth < 0:
            raise ValueError("linewidth must be non-negative.")

        size_array = _coerce_matching_coordinate_array(self.size, "size", x_array.shape)
        if np.any(size_array < 0.0):
            raise ValueError("size must be non-negative.")

        object.__setattr__(self, "x", x_array)
        object.__setattr__(self, "y", y_array)
        object.__setattr__(self, "size", size_array)
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))
        object.__setattr__(self, "domain", _normalize_bounds(self.domain, "domain"))
        object.__setattr__(self, "value_range", _normalize_bounds(self.value_range, "value_range"))
        object.__setattr__(self, "scatter_kwargs", dict(self.scatter_kwargs))

    @property
    def is_empty(self) -> bool:
        return self.x.size == 0

    def mpl_scatter_kwargs(
        self,
        *,
        size: npt.ArrayLike | float | None = None,
    ) -> dict[str, Any]:
        kwargs = dict(self.scatter_kwargs)
        if self.color is not None:
            kwargs["color"] = self.color
        if self.alpha is not None:
            kwargs["alpha"] = self.alpha
        kwargs.setdefault("marker", self.marker)
        kwargs.setdefault("s", self.size if size is None else size)
        if self.linewidth is not None:
            kwargs.setdefault("linewidths", self.linewidth)
        if self.edgecolor is not None:
            kwargs.setdefault("edgecolors", self.edgecolor)
        return kwargs

    def copy_with(
        self,
        *,
        x: npt.ArrayLike | None = None,
        y: npt.ArrayLike | None = None,
        color: Any | None = None,
        alpha: float | None = None,
        marker: Any | None = None,
        size: npt.ArrayLike | float | None = None,
        linewidth: float | None = None,
        edgecolor: Any | None = None,
        domain: Bounds | None = None,
        value_range: Bounds | None = None,
        scatter_kwargs: Mapping[str, Any] | None = None,
    ) -> Scatter:
        return Scatter(
            scatter_id=self.scatter_id,
            x=self.x if x is None else x,
            y=self.y if y is None else y,
            color=self.color if color is None else color,
            alpha=self.alpha if alpha is None else alpha,
            marker=self.marker if marker is None else marker,
            size=self.size if size is None else size,
            linewidth=self.linewidth if linewidth is None else linewidth,
            edgecolor=self.edgecolor if edgecolor is None else edgecolor,
            domain=self.domain if domain is None else domain,
            value_range=self.value_range if value_range is None else value_range,
            scatter_kwargs=self.scatter_kwargs if scatter_kwargs is None else scatter_kwargs,
        )

    def clipped_scatter_data(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        if self._clipped_scatter_cache is not None:
            return self._clipped_scatter_cache

        if self.is_empty:
            empty = np.asarray([], dtype=float)
            result = (empty, empty, empty)
            object.__setattr__(self, "_clipped_scatter_cache", result)
            return result

        if self.domain is None and self.value_range is None:
            result = (self.x, self.y, self.size)
            object.__setattr__(self, "_clipped_scatter_cache", result)
            return result

        mask = np.ones(self.x.shape, dtype=bool)
        if self.domain is not None:
            mask &= (self.x >= self.domain[0]) & (self.x <= self.domain[1])
        if self.value_range is not None:
            mask &= (self.y >= self.value_range[0]) & (self.y <= self.value_range[1])

        result = (self.x[mask], self.y[mask], self.size[mask])
        object.__setattr__(self, "_clipped_scatter_cache", result)
        return result

    def point_is_visible(self, x_value: float, y_value: float) -> bool:
        return _point_is_visible(
            x_value,
            y_value,
            domain=self.domain,
            value_range=self.value_range,
        )

    def visible_extents(self) -> tuple[float, float, float, float] | None:
        clipped_x, clipped_y, _ = self.clipped_scatter_data()
        if clipped_x.size == 0:
            return None

        return (
            float(np.min(clipped_x)),
            float(np.max(clipped_x)),
            float(np.min(clipped_y)),
            float(np.max(clipped_y)),
        )

    def reveal_by_progress(
        self,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None = None,
        direction: str = "forward",
    ) -> Scatter:
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

        return self.copy_with(x=self.x[mask], y=self.y[mask], size=self.size[mask])

    def hide_by_progress(
        self,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None = None,
        direction: str = "forward",
    ) -> Scatter:
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

        return self.copy_with(x=self.x[mask], y=self.y[mask], size=self.size[mask])


@dataclass(frozen=True)
class Text:
    """A single text label in plot coordinates."""

    text_id: str
    x: float
    y: float
    content: str
    color: str | None = None
    alpha: float | None = None
    fontsize: float | None = None
    ha: str | None = None
    va: str | None = None
    rotation: float | None = None
    domain: Bounds | None = None
    value_range: Bounds | None = None
    text_kwargs: Mapping[str, Any] = field(default_factory=dict)
    _text_style_cache: dict[str, Any] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        x_value = float(self.x)
        y_value = float(self.y)

        if not np.isfinite(x_value):
            raise ValueError("x must be finite.")
        if not np.isfinite(y_value):
            raise ValueError("y must be finite.")
        if self.fontsize is not None and self.fontsize < 0:
            raise ValueError("fontsize must be non-negative.")
        if self.rotation is not None and not np.isfinite(self.rotation):
            raise ValueError("rotation must be finite.")

        object.__setattr__(self, "x", x_value)
        object.__setattr__(self, "y", y_value)
        object.__setattr__(self, "content", str(self.content))
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))
        object.__setattr__(self, "domain", _normalize_bounds(self.domain, "domain"))
        object.__setattr__(self, "value_range", _normalize_bounds(self.value_range, "value_range"))
        object.__setattr__(self, "text_kwargs", dict(self.text_kwargs))

    def mpl_text_kwargs(self) -> dict[str, Any]:
        if self._text_style_cache is None:
            object.__setattr__(
                self,
                "_text_style_cache",
                _merge_text_kwargs(
                    self.text_kwargs,
                    color=self.color,
                    alpha=self.alpha,
                    fontsize=self.fontsize,
                    ha=self.ha,
                    va=self.va,
                    rotation=self.rotation,
                ),
            )
        return self._text_style_cache

    def copy_with(
        self,
        *,
        x: float | None = None,
        y: float | None = None,
        content: str | None = None,
        color: str | None = None,
        alpha: float | None = None,
        fontsize: float | None = None,
        ha: str | None = None,
        va: str | None = None,
        rotation: float | None = None,
        domain: Bounds | None = None,
        value_range: Bounds | None = None,
        text_kwargs: Mapping[str, Any] | None = None,
    ) -> Text:
        return Text(
            text_id=self.text_id,
            x=self.x if x is None else x,
            y=self.y if y is None else y,
            content=self.content if content is None else content,
            color=self.color if color is None else color,
            alpha=self.alpha if alpha is None else alpha,
            fontsize=self.fontsize if fontsize is None else fontsize,
            ha=self.ha if ha is None else ha,
            va=self.va if va is None else va,
            rotation=self.rotation if rotation is None else rotation,
            domain=self.domain if domain is None else domain,
            value_range=self.value_range if value_range is None else value_range,
            text_kwargs=self.text_kwargs if text_kwargs is None else text_kwargs,
        )

    def is_visible(self) -> bool:
        return _point_is_visible(
            self.x,
            self.y,
            domain=self.domain,
            value_range=self.value_range,
        )

    def visible_extents(self) -> tuple[float, float, float, float] | None:
        if not self.is_visible():
            return None
        return (self.x, self.x, self.y, self.y)


@dataclass(frozen=True)
class FillBetweenArea:
    """A filled region between y1(x) and y2(x)."""

    fill_id: str
    x: npt.ArrayLike
    y1: npt.ArrayLike
    y2: npt.ArrayLike | float
    color: str | None = None
    positive_color: str | None = None
    negative_color: str | None = None
    alpha: float | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    domain: Bounds | None = None
    value_range: Bounds | None = None
    fill_kwargs: Mapping[str, Any] = field(default_factory=dict)
    _fill_style_cache: dict[str, Any] | None = field(default=None, init=False, repr=False, compare=False)
    _clipped_fill_cache: tuple[FloatArray, FloatArray, FloatArray, npt.NDArray[np.bool_]] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

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
        if self._fill_style_cache is None:
            object.__setattr__(
                self,
                "_fill_style_cache",
                _merge_style_kwargs(
                    self.fill_kwargs,
                    color=self.color,
                    alpha=self.alpha,
                    linestyle=self.linestyle,
                    linewidth=self.linewidth,
                ),
            )
        return self._fill_style_cache

    def copy_with(
        self,
        *,
        x: npt.ArrayLike | None = None,
        y1: npt.ArrayLike | None = None,
        y2: npt.ArrayLike | float | None = None,
        color: str | None = None,
        positive_color: str | None = None,
        negative_color: str | None = None,
        alpha: float | None = None,
        linestyle: str | None = None,
        linewidth: float | None = None,
        domain: Bounds | None = None,
        value_range: Bounds | None = None,
        fill_kwargs: Mapping[str, Any] | None = None,
    ) -> FillBetweenArea:
        return FillBetweenArea(
            fill_id=self.fill_id,
            x=self.x if x is None else x,
            y1=self.y1 if y1 is None else y1,
            y2=self.y2 if y2 is None else y2,
            color=self.color if color is None else color,
            positive_color=self.positive_color if positive_color is None else positive_color,
            negative_color=self.negative_color if negative_color is None else negative_color,
            alpha=self.alpha if alpha is None else alpha,
            linestyle=self.linestyle if linestyle is None else linestyle,
            linewidth=self.linewidth if linewidth is None else linewidth,
            domain=self.domain if domain is None else domain,
            value_range=self.value_range if value_range is None else value_range,
            fill_kwargs=self.fill_kwargs if fill_kwargs is None else fill_kwargs,
        )

    def fill_segments(
        self,
    ) -> list[tuple[FloatArray, FloatArray, FloatArray, npt.NDArray[np.bool_], dict[str, Any]]]:
        x_values, y1_values, y2_values, where = self.clipped_fill_data()
        if not np.any(where):
            return []

        base_kwargs = self.mpl_fill_kwargs()
        if self.positive_color is None and self.negative_color is None:
            return [(x_values, y1_values, y2_values, where, base_kwargs)]

        segments: list[tuple[FloatArray, FloatArray, FloatArray, npt.NDArray[np.bool_], dict[str, Any]]] = []
        positive_where = where & (self.y1 >= self.y2)
        negative_where = where & (self.y1 < self.y2)

        if np.any(positive_where):
            positive_kwargs = dict(base_kwargs)
            if self.positive_color is not None:
                positive_kwargs["color"] = self.positive_color
            segments.append((x_values, y1_values, y2_values, positive_where, positive_kwargs))

        if np.any(negative_where):
            negative_kwargs = dict(base_kwargs)
            if self.negative_color is not None:
                negative_kwargs["color"] = self.negative_color
            segments.append((x_values, y1_values, y2_values, negative_where, negative_kwargs))

        return segments

    def clipped_fill_data(self) -> tuple[FloatArray, FloatArray, FloatArray, npt.NDArray[np.bool_]]:
        if self._clipped_fill_cache is not None:
            return self._clipped_fill_cache

        if self.is_empty:
            empty = np.asarray([], dtype=float)
            result = (empty, empty, empty, np.asarray([], dtype=bool))
            object.__setattr__(self, "_clipped_fill_cache", result)
            return result

        if self.domain is None:
            where = np.ones(self.x.shape, dtype=bool)
        else:
            where = (self.x >= self.domain[0]) & (self.x <= self.domain[1])

        if self.value_range is None:
            clipped_y1 = self.y1
            clipped_y2 = self.y2
        else:
            clipped_y1 = np.clip(self.y1, self.value_range[0], self.value_range[1])
            clipped_y2 = np.clip(self.y2, self.value_range[0], self.value_range[1])

        result = (self.x, clipped_y1, clipped_y2, where)
        object.__setattr__(self, "_clipped_fill_cache", result)
        return result

    def visible_extents(self) -> tuple[float, float, float, float] | None:
        if self.is_empty:
            return None

        x_values, y1_values, y2_values, where = self.clipped_fill_data()
        if self.domain is not None:
            x_min = max(float(np.min(self.x)), self.domain[0])
            x_max = min(float(np.max(self.x)), self.domain[1])
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
    scatters: Mapping[str, Scatter] = field(default_factory=dict)
    fills: Mapping[str, FillBetweenArea] = field(default_factory=dict)
    texts: Mapping[str, Text] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_curves: dict[str, Curve] = {}
        normalized_scatters: dict[str, Scatter] = {}
        normalized_fills: dict[str, FillBetweenArea] = {}
        normalized_texts: dict[str, Text] = {}

        for curve_id, curve in dict(self.curves).items():
            if curve_id != curve.curve_id:
                raise ValueError("Scene curve keys must match each curve's curve_id.")
            normalized_curves[curve_id] = curve

        for scatter_id, scatter in dict(self.scatters).items():
            if scatter_id != scatter.scatter_id:
                raise ValueError("Scene scatter keys must match each scatter's scatter_id.")
            normalized_scatters[scatter_id] = scatter

        for fill_id, fill in dict(self.fills).items():
            if fill_id != fill.fill_id:
                raise ValueError("Scene fill keys must match each fill's fill_id.")
            normalized_fills[fill_id] = fill

        for text_id, text in dict(self.texts).items():
            if text_id != text.text_id:
                raise ValueError("Scene text keys must match each text's text_id.")
            normalized_texts[text_id] = text

        object.__setattr__(self, "curves", normalized_curves)
        object.__setattr__(self, "scatters", normalized_scatters)
        object.__setattr__(self, "fills", normalized_fills)
        object.__setattr__(self, "texts", normalized_texts)

    def __len__(self) -> int:
        return len(self.curves) + len(self.scatters) + len(self.fills) + len(self.texts)

    def contains(self, item_id: str) -> bool:
        return (
            self.contains_curve(item_id)
            or self.contains_scatter(item_id)
            or self.contains_fill(item_id)
            or self.contains_text(item_id)
        )

    def contains_curve(self, curve_id: str) -> bool:
        return curve_id in self.curves

    def contains_fill(self, fill_id: str) -> bool:
        return fill_id in self.fills

    def contains_scatter(self, scatter_id: str) -> bool:
        return scatter_id in self.scatters

    def contains_text(self, text_id: str) -> bool:
        return text_id in self.texts

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

    def get_scatter(self, scatter_id: str) -> Scatter:
        try:
            return self.scatters[scatter_id]
        except KeyError as exc:
            raise KeyError(f"Scatter {scatter_id!r} does not exist in the scene.") from exc

    def get_text(self, text_id: str) -> Text:
        try:
            return self.texts[text_id]
        except KeyError as exc:
            raise KeyError(f"Text {text_id!r} does not exist in the scene.") from exc

    def add_curve(self, curve: Curve) -> Scene:
        if self.contains_curve(curve.curve_id):
            raise ValueError(f"Curve {curve.curve_id!r} already exists in the scene.")

        updated = dict(self.curves)
        updated[curve.curve_id] = curve
        return Scene(curves=updated, scatters=self.scatters, fills=self.fills, texts=self.texts)

    def update_curve(self, curve: Curve) -> Scene:
        if not self.contains_curve(curve.curve_id):
            raise ValueError(f"Curve {curve.curve_id!r} does not exist in the scene.")

        updated = dict(self.curves)
        updated[curve.curve_id] = curve
        return Scene(curves=updated, scatters=self.scatters, fills=self.fills, texts=self.texts)

    def remove_curve(self, curve_id: str) -> Scene:
        if not self.contains_curve(curve_id):
            raise ValueError(f"Curve {curve_id!r} does not exist in the scene.")

        updated = dict(self.curves)
        updated.pop(curve_id)
        return Scene(curves=updated, scatters=self.scatters, fills=self.fills, texts=self.texts)

    def add_scatter(self, scatter: Scatter) -> Scene:
        if self.contains_scatter(scatter.scatter_id):
            raise ValueError(f"Scatter {scatter.scatter_id!r} already exists in the scene.")

        updated = dict(self.scatters)
        updated[scatter.scatter_id] = scatter
        return Scene(curves=self.curves, scatters=updated, fills=self.fills, texts=self.texts)

    def update_scatter(self, scatter: Scatter) -> Scene:
        if not self.contains_scatter(scatter.scatter_id):
            raise ValueError(f"Scatter {scatter.scatter_id!r} does not exist in the scene.")

        updated = dict(self.scatters)
        updated[scatter.scatter_id] = scatter
        return Scene(curves=self.curves, scatters=updated, fills=self.fills, texts=self.texts)

    def remove_scatter(self, scatter_id: str) -> Scene:
        if not self.contains_scatter(scatter_id):
            raise ValueError(f"Scatter {scatter_id!r} does not exist in the scene.")

        updated = dict(self.scatters)
        updated.pop(scatter_id)
        return Scene(curves=self.curves, scatters=updated, fills=self.fills, texts=self.texts)

    def add_fill(self, fill: FillBetweenArea) -> Scene:
        if self.contains_fill(fill.fill_id):
            raise ValueError(f"Fill {fill.fill_id!r} already exists in the scene.")

        updated = dict(self.fills)
        updated[fill.fill_id] = fill
        return Scene(curves=self.curves, scatters=self.scatters, fills=updated, texts=self.texts)

    def update_fill(self, fill: FillBetweenArea) -> Scene:
        if not self.contains_fill(fill.fill_id):
            raise ValueError(f"Fill {fill.fill_id!r} does not exist in the scene.")

        updated = dict(self.fills)
        updated[fill.fill_id] = fill
        return Scene(curves=self.curves, scatters=self.scatters, fills=updated, texts=self.texts)

    def remove_fill(self, fill_id: str) -> Scene:
        if not self.contains_fill(fill_id):
            raise ValueError(f"Fill {fill_id!r} does not exist in the scene.")

        updated = dict(self.fills)
        updated.pop(fill_id)
        return Scene(curves=self.curves, scatters=self.scatters, fills=updated, texts=self.texts)

    def add_text(self, text: Text) -> Scene:
        if self.contains_text(text.text_id):
            raise ValueError(f"Text {text.text_id!r} already exists in the scene.")

        updated = dict(self.texts)
        updated[text.text_id] = text
        return Scene(curves=self.curves, scatters=self.scatters, fills=self.fills, texts=updated)

    def update_text(self, text: Text) -> Scene:
        if not self.contains_text(text.text_id):
            raise ValueError(f"Text {text.text_id!r} does not exist in the scene.")

        updated = dict(self.texts)
        updated[text.text_id] = text
        return Scene(curves=self.curves, scatters=self.scatters, fills=self.fills, texts=updated)

    def remove_text(self, text_id: str) -> Scene:
        if not self.contains_text(text_id):
            raise ValueError(f"Text {text_id!r} does not exist in the scene.")

        updated = dict(self.texts)
        updated.pop(text_id)
        return Scene(curves=self.curves, scatters=self.scatters, fills=self.fills, texts=updated)
