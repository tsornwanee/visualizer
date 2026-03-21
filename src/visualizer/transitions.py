from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
from matplotlib.colors import to_hex, to_rgb

from .scene import (
    Curve,
    FillBetweenArea,
    FloatArray,
    Scene,
    _clamp_progress,
    _coerce_coordinate_array,
    _coerce_matching_coordinate_array,
    _validate_alpha,
)

Bounds = tuple[float, float]


def _interpolate_float(start: float, end: float, progress: float) -> float:
    t = _clamp_progress(progress)
    return float(start + (end - start) * t)


def _interpolate_color(start_color: Any, end_color: Any, progress: float) -> str:
    t = _clamp_progress(progress)
    start_rgb = np.asarray(to_rgb(start_color), dtype=float)
    end_rgb = np.asarray(to_rgb(end_color), dtype=float)
    return to_hex(start_rgb + (end_rgb - start_rgb) * t)


def _current_curve_color(curve: Curve) -> Any:
    return curve.mpl_line_kwargs().get("color", "#1f77b4")


def _current_fill_color(fill: FillBetweenArea) -> Any:
    return fill.mpl_fill_kwargs().get("color", "#1f77b4")


def _current_curve_alpha(curve: Curve) -> float:
    style = curve.mpl_line_kwargs()
    alpha = style.get("alpha")
    if alpha is None:
        return 1.0
    return float(alpha)


def _current_fill_alpha(fill: FillBetweenArea) -> float:
    style = fill.mpl_fill_kwargs()
    alpha = style.get("alpha")
    if alpha is None:
        return 1.0
    return float(alpha)


def _current_curve_linewidth(curve: Curve) -> float:
    style = curve.mpl_line_kwargs()
    linewidth = style.get("linewidth", 1.5)
    return float(linewidth)


def _current_fill_linewidth(fill: FillBetweenArea) -> float:
    style = fill.mpl_fill_kwargs()
    linewidth = style.get("linewidth", 1.0)
    return float(linewidth)


def _validate_direction(direction: str) -> str:
    if direction not in {"forward", "backward"}:
        raise ValueError("direction must be 'forward' or 'backward'.")
    return direction


def _normalize_timeline_domain(
    timeline_domain: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if timeline_domain is None:
        return None

    start, end = map(float, timeline_domain)
    if not np.isfinite(start) or not np.isfinite(end):
        raise ValueError("timeline_domain must contain only finite values.")
    if start > end:
        start, end = end, start

    return (start, end)


def _interpolate_bounds(
    start_bounds: Bounds,
    end_bounds: Bounds,
    progress: float,
) -> Bounds:
    return (
        _interpolate_float(start_bounds[0], end_bounds[0], progress),
        _interpolate_float(start_bounds[1], end_bounds[1], progress),
    )


@dataclass(frozen=True)
class PointerOverlay:
    x: float
    y: float
    artist_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artist_kwargs", dict(self.artist_kwargs))


@dataclass(frozen=True)
class GlowOverlay:
    x: FloatArray
    y: FloatArray
    artist_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x_array = np.asarray(self.x, dtype=float)
        y_array = np.asarray(self.y, dtype=float)

        if x_array.ndim != 1:
            raise ValueError("glow x must be a one-dimensional array.")
        if y_array.ndim != 1:
            raise ValueError("glow y must be a one-dimensional array.")
        if np.any(np.isinf(x_array)):
            raise ValueError("glow x must not contain infinite values.")
        if np.any(np.isinf(y_array)):
            raise ValueError("glow y must not contain infinite values.")

        if x_array.shape != y_array.shape:
            raise ValueError("Glow overlay x and y must have the same shape.")

        object.__setattr__(self, "x", x_array)
        object.__setattr__(self, "y", y_array)
        object.__setattr__(self, "artist_kwargs", dict(self.artist_kwargs))


@dataclass(frozen=True)
class FrameState:
    scene: Scene
    pointers: tuple[PointerOverlay, ...] = ()
    glows: tuple[GlowOverlay, ...] = ()


@dataclass(frozen=True)
class Transition(ABC):
    """Base class for all scene-to-scene transitions."""

    @abstractmethod
    def interpolate(self, scene: Scene, progress: float) -> Scene:
        """Return the in-between scene for a normalized progress value."""

    @abstractmethod
    def apply(self, scene: Scene) -> Scene:
        """Return the scene after the transition completes."""

    def frame_state(
        self,
        scene: Scene,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None = None,
    ) -> FrameState:
        return FrameState(scene=self.interpolate(scene, progress))


@dataclass(frozen=True)
class PauseTransition(Transition):
    """A timed no-op transition that keeps the current scene unchanged."""

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        return scene

    def apply(self, scene: Scene) -> Scene:
        return scene


@dataclass(frozen=True)
class ParallelTransition(Transition):
    transitions: tuple[Transition, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "transitions", tuple(self.transitions))
        if not self.transitions:
            raise ValueError("ParallelTransition requires at least one child transition.")

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        return self.frame_state(scene, progress).scene

    def apply(self, scene: Scene) -> Scene:
        current_scene = scene
        for transition in self.transitions:
            current_scene = transition.apply(current_scene)
        return current_scene

    def frame_state(
        self,
        scene: Scene,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None = None,
    ) -> FrameState:
        current_scene = scene
        pointers: list[PointerOverlay] = []
        glows: list[GlowOverlay] = []
        shared_domain = timeline_domain or self._shared_timeline_domain(scene)

        for transition in self.transitions:
            state = transition.frame_state(
                current_scene,
                progress,
                timeline_domain=shared_domain,
            )
            current_scene = state.scene
            pointers.extend(state.pointers)
            glows.extend(state.glows)

        return FrameState(scene=current_scene, pointers=tuple(pointers), glows=tuple(glows))

    def _shared_timeline_domain(self, scene: Scene) -> tuple[float, float] | None:
        domains: list[tuple[float, float]] = []
        current_scene = scene

        for transition in self.transitions:
            timeline_getter = getattr(transition, "_timeline_domain", None)
            if callable(timeline_getter):
                domain = timeline_getter(current_scene)
                if domain is not None:
                    domains.append(domain)
            current_scene = transition.apply(current_scene)

        if not domains:
            return None

        return (
            min(domain[0] for domain in domains),
            max(domain[1] for domain in domains),
        )


@dataclass(frozen=True)
class DrawTransition(Transition):
    curve: Curve
    show_pointer: bool = True
    pointer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    direction: str = "forward"
    timeline_domain: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pointer_kwargs", dict(self.pointer_kwargs))
        object.__setattr__(self, "direction", _validate_direction(self.direction))
        object.__setattr__(self, "timeline_domain", _normalize_timeline_domain(self.timeline_domain))

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        if scene.contains_curve(self.curve.curve_id):
            raise ValueError(f"Curve {self.curve.curve_id!r} already exists in the scene.")

        partial_curve = self.curve.reveal_by_progress(
            _clamp_progress(progress),
            timeline_domain=self._effective_timeline_domain(None),
            direction=self.direction,
        )
        updated = dict(scene.curves)
        updated[self.curve.curve_id] = partial_curve
        return Scene(curves=updated, fills=scene.fills)

    def apply(self, scene: Scene) -> Scene:
        return scene.add_curve(self.curve)

    def frame_state(
        self,
        scene: Scene,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None = None,
    ) -> FrameState:
        in_between_scene = self._interpolated_scene(scene, progress, timeline_domain=timeline_domain)
        pointers: tuple[PointerOverlay, ...] = ()

        if self.show_pointer:
            partial_curve = in_between_scene.get_curve(self.curve.curve_id)
            if not partial_curve.is_empty:
                pointer_x = float(
                    partial_curve.x[-1]
                    if self.direction == "forward"
                    else partial_curve.x[0]
                )
                pointer_y = float(
                    partial_curve.y[-1]
                    if self.direction == "forward"
                    else partial_curve.y[0]
                )
                curve_style = self.curve.mpl_line_kwargs()
                pointer_style = {
                    "marker": "o",
                    "linestyle": "None",
                    "markersize": max(float(curve_style.get("linewidth", 2.0)) * 2.5, 6.0),
                    "color": curve_style.get("color", "#111827"),
                    "markeredgecolor": "white",
                    "markeredgewidth": 1.0,
                    "alpha": curve_style.get("alpha", 1.0),
                    "zorder": curve_style.get("zorder", 2.0) + 0.5,
                }
                pointer_style.update(self.pointer_kwargs)
                if partial_curve.point_is_visible(pointer_x, pointer_y):
                    pointers = (
                        PointerOverlay(
                            x=pointer_x,
                            y=pointer_y,
                            artist_kwargs=pointer_style,
                        ),
                    )

        return FrameState(scene=in_between_scene, pointers=pointers)

    def _timeline_domain(self, scene: Scene) -> tuple[float, float]:
        return self._effective_timeline_domain(None)

    def _effective_timeline_domain(
        self,
        external_timeline_domain: tuple[float, float] | None,
    ) -> tuple[float, float]:
        if self.timeline_domain is not None:
            return self.timeline_domain
        if external_timeline_domain is not None:
            return external_timeline_domain
        return (float(np.min(self.curve.x)), float(np.max(self.curve.x)))

    def _interpolated_scene(
        self,
        scene: Scene,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None,
    ) -> Scene:
        if scene.contains_curve(self.curve.curve_id):
            raise ValueError(f"Curve {self.curve.curve_id!r} already exists in the scene.")

        partial_curve = self.curve.reveal_by_progress(
            _clamp_progress(progress),
            timeline_domain=self._effective_timeline_domain(timeline_domain),
            direction=self.direction,
        )
        updated = dict(scene.curves)
        updated[self.curve.curve_id] = partial_curve
        return Scene(curves=updated, fills=scene.fills)


@dataclass(frozen=True)
class FillBetweenTransition(Transition):
    fill: FillBetweenArea
    direction: str = "forward"
    timeline_domain: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", _validate_direction(self.direction))
        object.__setattr__(self, "timeline_domain", _normalize_timeline_domain(self.timeline_domain))

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        if scene.contains_fill(self.fill.fill_id):
            raise ValueError(f"Fill {self.fill.fill_id!r} already exists in the scene.")

        partial_fill = self.fill.reveal_by_progress(
            _clamp_progress(progress),
            timeline_domain=self._effective_timeline_domain(None),
            direction=self.direction,
        )
        updated = dict(scene.fills)
        updated[self.fill.fill_id] = partial_fill
        return Scene(curves=scene.curves, fills=updated)

    def apply(self, scene: Scene) -> Scene:
        return scene.add_fill(self.fill)

    def frame_state(
        self,
        scene: Scene,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None = None,
    ) -> FrameState:
        return FrameState(
            scene=self._interpolated_scene(scene, progress, timeline_domain=timeline_domain)
        )

    def _timeline_domain(self, scene: Scene) -> tuple[float, float]:
        return self._effective_timeline_domain(None)

    def _effective_timeline_domain(
        self,
        external_timeline_domain: tuple[float, float] | None,
    ) -> tuple[float, float]:
        if self.timeline_domain is not None:
            return self.timeline_domain
        if external_timeline_domain is not None:
            return external_timeline_domain
        return (float(np.min(self.fill.x)), float(np.max(self.fill.x)))

    def _interpolated_scene(
        self,
        scene: Scene,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None,
    ) -> Scene:
        if scene.contains_fill(self.fill.fill_id):
            raise ValueError(f"Fill {self.fill.fill_id!r} already exists in the scene.")

        partial_fill = self.fill.reveal_by_progress(
            _clamp_progress(progress),
            timeline_domain=self._effective_timeline_domain(timeline_domain),
            direction=self.direction,
        )
        updated = dict(scene.fills)
        updated[self.fill.fill_id] = partial_fill
        return Scene(curves=scene.curves, fills=updated)


@dataclass(frozen=True)
class EraseTransition(Transition):
    curve_id: str
    direction: str = "forward"
    timeline_domain: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", _validate_direction(self.direction))
        object.__setattr__(self, "timeline_domain", _normalize_timeline_domain(self.timeline_domain))

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        return self._interpolated_scene(scene, progress, timeline_domain=None)

    def apply(self, scene: Scene) -> Scene:
        return scene.remove_curve(self.curve_id)

    def frame_state(
        self,
        scene: Scene,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None = None,
    ) -> FrameState:
        return FrameState(
            scene=self._interpolated_scene(scene, progress, timeline_domain=timeline_domain)
        )

    def _timeline_domain(self, scene: Scene) -> tuple[float, float]:
        curve = scene.get_curve(self.curve_id)
        if self.timeline_domain is not None:
            return self.timeline_domain
        return (float(np.min(curve.x)), float(np.max(curve.x)))

    def _interpolated_scene(
        self,
        scene: Scene,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None,
    ) -> Scene:
        curve = scene.get_curve(self.curve_id)
        effective_domain = self.timeline_domain or timeline_domain or (
            float(np.min(curve.x)),
            float(np.max(curve.x)),
        )
        updated = dict(scene.curves)
        updated[self.curve_id] = curve.hide_by_progress(
            _clamp_progress(progress),
            timeline_domain=effective_domain,
            direction=self.direction,
        )
        return Scene(curves=updated, fills=scene.fills)


@dataclass(frozen=True)
class EraseFillBetweenTransition(Transition):
    fill_id: str
    direction: str = "forward"
    timeline_domain: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", _validate_direction(self.direction))
        object.__setattr__(self, "timeline_domain", _normalize_timeline_domain(self.timeline_domain))

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        return self._interpolated_scene(scene, progress, timeline_domain=None)

    def apply(self, scene: Scene) -> Scene:
        return scene.remove_fill(self.fill_id)

    def frame_state(
        self,
        scene: Scene,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None = None,
    ) -> FrameState:
        return FrameState(
            scene=self._interpolated_scene(scene, progress, timeline_domain=timeline_domain)
        )

    def _timeline_domain(self, scene: Scene) -> tuple[float, float]:
        fill = scene.get_fill(self.fill_id)
        if self.timeline_domain is not None:
            return self.timeline_domain
        return (float(np.min(fill.x)), float(np.max(fill.x)))

    def _interpolated_scene(
        self,
        scene: Scene,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None,
    ) -> Scene:
        fill = scene.get_fill(self.fill_id)
        effective_domain = self.timeline_domain or timeline_domain or (
            float(np.min(fill.x)),
            float(np.max(fill.x)),
        )
        updated = dict(scene.fills)
        updated[self.fill_id] = fill.hide_by_progress(
            _clamp_progress(progress),
            timeline_domain=effective_domain,
            direction=self.direction,
        )
        return Scene(curves=scene.curves, fills=updated)


@dataclass(frozen=True)
class MoveTransition(Transition):
    curve_id: str
    x_prime: npt.ArrayLike | None
    y_prime: npt.ArrayLike
    color: str | None = None
    alpha: float | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    domain: Bounds | None = None
    value_range: Bounds | None = None
    line_kwargs: Mapping[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        y_prime = _coerce_coordinate_array(self.y_prime, "y_prime")
        x_prime = None
        if self.x_prime is not None:
            x_prime = _coerce_coordinate_array(self.x_prime, "x_prime")

        if x_prime is not None and x_prime.shape != y_prime.shape:
            raise ValueError("x_prime and y_prime must have the same shape.")

        object.__setattr__(self, "x_prime", x_prime)
        object.__setattr__(self, "y_prime", y_prime)
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))
        object.__setattr__(self, "domain", _normalize_timeline_domain(self.domain))
        object.__setattr__(self, "value_range", _normalize_timeline_domain(self.value_range))
        if self.line_kwargs is not None:
            object.__setattr__(self, "line_kwargs", dict(self.line_kwargs))

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        source_curve = scene.get_curve(self.curve_id)
        x_prime = self._effective_x_prime(source_curve)
        self._validate_source_shape(source_curve, x_prime)

        t = _clamp_progress(progress)
        x_values = source_curve.x + (x_prime - source_curve.x) * t
        y_values = source_curve.y + (self.y_prime - source_curve.y) * t

        return scene.update_curve(
            self._updated_curve(
                source_curve,
                x_values,
                y_values,
                domain=self._interpolated_domain(source_curve, t),
                value_range=self._interpolated_value_range(source_curve, t),
            )
        )

    def apply(self, scene: Scene) -> Scene:
        source_curve = scene.get_curve(self.curve_id)
        x_prime = self._effective_x_prime(source_curve)
        self._validate_source_shape(source_curve, x_prime)
        return scene.update_curve(
            self._updated_curve(
                source_curve,
                x_prime,
                self.y_prime,
                domain=self.domain,
                value_range=self.value_range,
            )
        )

    def _updated_curve(
        self,
        source_curve: Curve,
        x_values: npt.ArrayLike,
        y_values: npt.ArrayLike,
        *,
        domain: Bounds | None = None,
        value_range: Bounds | None = None,
    ) -> Curve:
        return source_curve.copy_with(
            x=x_values,
            y=y_values,
            color=self.color,
            alpha=self.alpha,
            linestyle=self.linestyle,
            linewidth=self.linewidth,
            domain=domain,
            value_range=value_range,
            line_kwargs=source_curve.line_kwargs if self.line_kwargs is None else self.line_kwargs,
        )

    def _effective_x_prime(self, source_curve: Curve) -> FloatArray:
        if self.x_prime is None:
            return source_curve.x
        return self.x_prime

    def _interpolated_domain(self, source_curve: Curve, progress: float) -> Bounds | None:
        if self.domain is None:
            return source_curve.domain

        start_bounds = source_curve.domain or self._curve_x_extent(source_curve)
        return _interpolate_bounds(start_bounds, self.domain, progress)

    def _interpolated_value_range(self, source_curve: Curve, progress: float) -> Bounds | None:
        if self.value_range is None:
            return source_curve.value_range

        start_bounds = source_curve.value_range or self._curve_y_extent(source_curve)
        return _interpolate_bounds(start_bounds, self.value_range, progress)

    def _curve_x_extent(self, curve: Curve) -> Bounds:
        return (float(np.min(curve.x)), float(np.max(curve.x)))

    def _curve_y_extent(self, curve: Curve) -> Bounds:
        return (float(np.min(curve.y)), float(np.max(curve.y)))

    def _validate_source_shape(
        self,
        source_curve: Curve,
        x_prime: FloatArray,
    ) -> None:
        if source_curve.x.shape != x_prime.shape or source_curve.x.shape != self.y_prime.shape:
            raise ValueError(
                "MoveTransition requires x_prime and y_prime to match the source curve shape."
            )


@dataclass(frozen=True)
class MoveFillBetweenTransition(Transition):
    fill_id: str
    x_prime: npt.ArrayLike | None
    y1_prime: npt.ArrayLike
    y2_prime: npt.ArrayLike | float
    color: str | None = None
    alpha: float | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    domain: Bounds | None = None
    value_range: Bounds | None = None
    fill_kwargs: Mapping[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        y1_prime = _coerce_coordinate_array(self.y1_prime, "y1_prime")
        x_prime = None
        if self.x_prime is not None:
            x_prime = _coerce_coordinate_array(self.x_prime, "x_prime")

        if x_prime is not None and x_prime.shape != y1_prime.shape:
            raise ValueError("x_prime and y1_prime must have the same shape.")

        y2_prime = _coerce_matching_coordinate_array(self.y2_prime, "y2_prime", y1_prime.shape)

        object.__setattr__(self, "x_prime", x_prime)
        object.__setattr__(self, "y1_prime", y1_prime)
        object.__setattr__(self, "y2_prime", y2_prime)
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))
        object.__setattr__(self, "domain", _normalize_timeline_domain(self.domain))
        object.__setattr__(self, "value_range", _normalize_timeline_domain(self.value_range))
        if self.fill_kwargs is not None:
            object.__setattr__(self, "fill_kwargs", dict(self.fill_kwargs))

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        source_fill = scene.get_fill(self.fill_id)
        x_prime = self._effective_x_prime(source_fill)
        self._validate_source_shape(source_fill, x_prime)

        t = _clamp_progress(progress)
        x_values = source_fill.x + (x_prime - source_fill.x) * t
        y1_values = source_fill.y1 + (self.y1_prime - source_fill.y1) * t
        y2_values = source_fill.y2 + (self.y2_prime - source_fill.y2) * t

        return scene.update_fill(
            self._updated_fill(
                source_fill,
                x_values,
                y1_values,
                y2_values,
                domain=self._interpolated_domain(source_fill, t),
                value_range=self._interpolated_value_range(source_fill, t),
            )
        )

    def apply(self, scene: Scene) -> Scene:
        source_fill = scene.get_fill(self.fill_id)
        x_prime = self._effective_x_prime(source_fill)
        self._validate_source_shape(source_fill, x_prime)
        return scene.update_fill(
            self._updated_fill(
                source_fill,
                x_prime,
                self.y1_prime,
                self.y2_prime,
                domain=self.domain,
                value_range=self.value_range,
            )
        )

    def _updated_fill(
        self,
        source_fill: FillBetweenArea,
        x_values: npt.ArrayLike,
        y1_values: npt.ArrayLike,
        y2_values: npt.ArrayLike,
        *,
        domain: Bounds | None = None,
        value_range: Bounds | None = None,
    ) -> FillBetweenArea:
        return source_fill.copy_with(
            x=x_values,
            y1=y1_values,
            y2=y2_values,
            color=self.color,
            alpha=self.alpha,
            linestyle=self.linestyle,
            linewidth=self.linewidth,
            domain=domain,
            value_range=value_range,
            fill_kwargs=source_fill.fill_kwargs if self.fill_kwargs is None else self.fill_kwargs,
        )

    def _effective_x_prime(self, source_fill: FillBetweenArea) -> FloatArray:
        if self.x_prime is None:
            return source_fill.x
        return self.x_prime

    def _interpolated_domain(
        self,
        source_fill: FillBetweenArea,
        progress: float,
    ) -> Bounds | None:
        if self.domain is None:
            return source_fill.domain

        start_bounds = source_fill.domain or self._fill_x_extent(source_fill)
        return _interpolate_bounds(start_bounds, self.domain, progress)

    def _interpolated_value_range(
        self,
        source_fill: FillBetweenArea,
        progress: float,
    ) -> Bounds | None:
        if self.value_range is None:
            return source_fill.value_range

        start_bounds = source_fill.value_range or self._fill_y_extent(source_fill)
        return _interpolate_bounds(start_bounds, self.value_range, progress)

    def _fill_x_extent(self, fill: FillBetweenArea) -> Bounds:
        return (float(np.min(fill.x)), float(np.max(fill.x)))

    def _fill_y_extent(self, fill: FillBetweenArea) -> Bounds:
        return (
            float(min(np.min(fill.y1), np.min(fill.y2))),
            float(max(np.max(fill.y1), np.max(fill.y2))),
        )

    def _validate_source_shape(
        self,
        source_fill: FillBetweenArea,
        x_prime: FloatArray,
    ) -> None:
        if (
            source_fill.x.shape != x_prime.shape
            or source_fill.x.shape != self.y1_prime.shape
            or source_fill.x.shape != self.y2_prime.shape
        ):
            raise ValueError(
                "MoveFillBetweenTransition requires target arrays to match the source fill shape."
            )


@dataclass(frozen=True)
class CurveStyleTransition(Transition):
    curve_id: str
    color: str | None = None
    alpha: float | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    line_kwargs: Mapping[str, Any] | None = field(default=None)
    interpolate_color: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))
        if self.linewidth is not None and self.linewidth < 0:
            raise ValueError("linewidth must be non-negative.")
        if self.line_kwargs is not None:
            object.__setattr__(self, "line_kwargs", dict(self.line_kwargs))

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        curve = scene.get_curve(self.curve_id)
        t = _clamp_progress(progress)

        color = curve.color
        if self.color is not None:
            color = (
                _interpolate_color(_current_curve_color(curve), self.color, t)
                if self.interpolate_color
                else curve.color
            )

        alpha = curve.alpha
        if self.alpha is not None:
            alpha = _interpolate_float(_current_curve_alpha(curve), self.alpha, t)

        linewidth = curve.linewidth
        if self.linewidth is not None:
            linewidth = _interpolate_float(_current_curve_linewidth(curve), self.linewidth, t)

        updated_curve = curve.copy_with(
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )
        return scene.update_curve(updated_curve)

    def apply(self, scene: Scene) -> Scene:
        curve = scene.get_curve(self.curve_id)
        updated_curve = curve.copy_with(
            color=self.color,
            alpha=self.alpha,
            linestyle=self.linestyle,
            linewidth=self.linewidth,
            line_kwargs=curve.line_kwargs if self.line_kwargs is None else self.line_kwargs,
        )
        return scene.update_curve(updated_curve)


@dataclass(frozen=True)
class FillStyleTransition(Transition):
    fill_id: str
    color: str | None = None
    alpha: float | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    fill_kwargs: Mapping[str, Any] | None = field(default=None)
    interpolate_color: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))
        if self.linewidth is not None and self.linewidth < 0:
            raise ValueError("linewidth must be non-negative.")
        if self.fill_kwargs is not None:
            object.__setattr__(self, "fill_kwargs", dict(self.fill_kwargs))

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        fill = scene.get_fill(self.fill_id)
        t = _clamp_progress(progress)

        color = fill.color
        if self.color is not None:
            color = (
                _interpolate_color(_current_fill_color(fill), self.color, t)
                if self.interpolate_color
                else fill.color
            )

        alpha = fill.alpha
        if self.alpha is not None:
            alpha = _interpolate_float(_current_fill_alpha(fill), self.alpha, t)

        linewidth = fill.linewidth
        if self.linewidth is not None:
            linewidth = _interpolate_float(_current_fill_linewidth(fill), self.linewidth, t)

        updated_fill = fill.copy_with(
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )
        return scene.update_fill(updated_fill)

    def apply(self, scene: Scene) -> Scene:
        fill = scene.get_fill(self.fill_id)
        updated_fill = fill.copy_with(
            color=self.color,
            alpha=self.alpha,
            linestyle=self.linestyle,
            linewidth=self.linewidth,
            fill_kwargs=fill.fill_kwargs if self.fill_kwargs is None else self.fill_kwargs,
        )
        return scene.update_fill(updated_fill)


@dataclass(frozen=True)
class StressTransition(Transition):
    curve_id: str
    glow_color: str | None = None
    glow_width: float | None = None
    # Backward-compatible aliases for earlier drafts of the API.
    color: str | None = None
    max_alpha: float = 0.35
    glow_linewidth: float | None = None
    linestyle: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_alpha", _validate_alpha(self.max_alpha))
        if self.glow_width is not None and self.glow_width < 0:
            raise ValueError("glow_width must be non-negative.")
        if self.glow_linewidth is not None and self.glow_linewidth < 0:
            raise ValueError("glow_linewidth must be non-negative.")
        if (
            self.glow_color is not None
            and self.color is not None
            and self.glow_color != self.color
        ):
            raise ValueError("glow_color and color cannot disagree.")
        if (
            self.glow_width is not None
            and self.glow_linewidth is not None
            and self.glow_width != self.glow_linewidth
        ):
            raise ValueError("glow_width and glow_linewidth cannot disagree.")

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        return scene

    def apply(self, scene: Scene) -> Scene:
        return scene

    def frame_state(
        self,
        scene: Scene,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None = None,
    ) -> FrameState:
        curve = scene.get_curve(self.curve_id)
        if curve.is_empty:
            return FrameState(scene=scene)

        clipped_x, clipped_y = curve.clipped_line_data()
        if clipped_x.size == 0:
            return FrameState(scene=scene)

        strength = float(np.sin(np.pi * _clamp_progress(progress)))
        if strength <= 0.0:
            return FrameState(scene=scene)

        curve_style = curve.mpl_line_kwargs()
        base_linewidth = float(curve_style.get("linewidth", 2.0))
        glow_color = self.glow_color if self.glow_color is not None else self.color
        glow_width = self.glow_width if self.glow_width is not None else self.glow_linewidth
        glow_style = {
            "color": glow_color or curve_style.get("color", "#f59e0b"),
            "alpha": self.max_alpha * strength,
            "linewidth": (
                glow_width
                if glow_width is not None
                else max(base_linewidth * 3.0, base_linewidth + 2.0)
            ),
            "linestyle": self.linestyle or curve_style.get("linestyle", "-"),
            "solid_capstyle": "round",
            "zorder": curve_style.get("zorder", 2.0) - 0.1,
        }

        return FrameState(
            scene=scene,
            glows=(GlowOverlay(x=clipped_x, y=clipped_y, artist_kwargs=glow_style),),
        )


@dataclass(frozen=True)
class JitterTransition(Transition):
    curve_id: str
    x_amplitude: float = 0.0
    y_amplitude: float = 0.02
    cycles: float = 10.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.x_amplitude < 0.0 or self.y_amplitude < 0.0:
            raise ValueError("Jitter amplitudes must be non-negative.")
        if self.cycles <= 0.0:
            raise ValueError("cycles must be positive.")

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        return self._perturbed_scene(scene, progress)

    def apply(self, scene: Scene) -> Scene:
        return scene

    def frame_state(
        self,
        scene: Scene,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None = None,
    ) -> FrameState:
        return FrameState(scene=self._perturbed_scene(scene, progress))

    def _perturbed_scene(self, scene: Scene, progress: float) -> Scene:
        curve = scene.get_curve(self.curve_id)
        if curve.is_empty:
            return scene

        envelope = float(np.sin(np.pi * _clamp_progress(progress)))
        if envelope <= 0.0 or (self.x_amplitude == 0.0 and self.y_amplitude == 0.0):
            return scene

        oscillation = 2.0 * np.pi * self.cycles * _clamp_progress(progress)
        spatial = np.linspace(0.0, 2.0 * np.pi, curve.x.size)
        rng = np.random.default_rng(self.seed)
        phase_x = float(rng.uniform(0.0, 2.0 * np.pi))
        phase_y = float(rng.uniform(0.0, 2.0 * np.pi))

        x_offset = envelope * self.x_amplitude * np.sin(oscillation + 2.0 * spatial + phase_x)
        y_offset = envelope * self.y_amplitude * np.sin(
            1.3 * oscillation + 3.0 * spatial + phase_y
        )

        perturbed_curve = curve.copy_with(
            x=curve.x + x_offset,
            y=curve.y + y_offset,
        )
        return scene.update_curve(perturbed_curve)


@dataclass(frozen=True)
class JitterFillBetweenTransition(Transition):
    fill_id: str
    x_amplitude: float = 0.0
    y1_amplitude: float = 0.02
    y2_amplitude: float = 0.0
    cycles: float = 10.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.x_amplitude < 0.0 or self.y1_amplitude < 0.0 or self.y2_amplitude < 0.0:
            raise ValueError("Jitter amplitudes must be non-negative.")
        if self.cycles <= 0.0:
            raise ValueError("cycles must be positive.")

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        return self._perturbed_scene(scene, progress)

    def apply(self, scene: Scene) -> Scene:
        return scene

    def frame_state(
        self,
        scene: Scene,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None = None,
    ) -> FrameState:
        return FrameState(scene=self._perturbed_scene(scene, progress))

    def _perturbed_scene(self, scene: Scene, progress: float) -> Scene:
        fill = scene.get_fill(self.fill_id)
        if fill.is_empty:
            return scene

        envelope = float(np.sin(np.pi * _clamp_progress(progress)))
        if envelope <= 0.0 or (
            self.x_amplitude == 0.0 and self.y1_amplitude == 0.0 and self.y2_amplitude == 0.0
        ):
            return scene

        oscillation = 2.0 * np.pi * self.cycles * _clamp_progress(progress)
        spatial = np.linspace(0.0, 2.0 * np.pi, fill.x.size)
        rng = np.random.default_rng(self.seed)
        phase_x = float(rng.uniform(0.0, 2.0 * np.pi))
        phase_y1 = float(rng.uniform(0.0, 2.0 * np.pi))
        phase_y2 = float(rng.uniform(0.0, 2.0 * np.pi))

        x_offset = envelope * self.x_amplitude * np.sin(oscillation + 2.0 * spatial + phase_x)
        y1_offset = envelope * self.y1_amplitude * np.sin(
            1.3 * oscillation + 3.0 * spatial + phase_y1
        )
        y2_offset = envelope * self.y2_amplitude * np.sin(
            0.9 * oscillation + 2.5 * spatial + phase_y2
        )

        perturbed_fill = fill.copy_with(
            x=fill.x + x_offset,
            y1=fill.y1 + y1_offset,
            y2=fill.y2 + y2_offset,
        )
        return scene.update_fill(perturbed_fill)
