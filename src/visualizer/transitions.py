from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from .scene import (
    Curve,
    FillBetweenArea,
    FloatArray,
    Scene,
    _clamp_progress,
    _coerce_matching_unit_interval_array,
    _coerce_unit_interval_array,
    _validate_alpha,
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
        x_array = _coerce_unit_interval_array(self.x, "glow x")
        y_array = _coerce_unit_interval_array(self.y, "glow y")

        if x_array.shape != y_array.shape:
            raise ValueError("Glow overlay x and y must have the same shape.")

        object.__setattr__(self, "x", x_array)
        object.__setattr__(self, "y", y_array)
        object.__setattr__(self, "artist_kwargs", dict(self.artist_kwargs))


@dataclass(frozen=True)
class FrameState:
    scene: Scene
    pointer: PointerOverlay | None = None
    glow: GlowOverlay | None = None


@dataclass(frozen=True)
class Transition(ABC):
    """Base class for all scene-to-scene transitions."""

    @abstractmethod
    def interpolate(self, scene: Scene, progress: float) -> Scene:
        """Return the in-between scene for a normalized progress value."""

    @abstractmethod
    def apply(self, scene: Scene) -> Scene:
        """Return the scene after the transition completes."""

    def frame_state(self, scene: Scene, progress: float) -> FrameState:
        return FrameState(scene=self.interpolate(scene, progress))


@dataclass(frozen=True)
class DrawTransition(Transition):
    curve: Curve
    show_pointer: bool = True
    pointer_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pointer_kwargs", dict(self.pointer_kwargs))

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        if scene.contains_curve(self.curve.curve_id):
            raise ValueError(f"Curve {self.curve.curve_id!r} already exists in the scene.")

        partial_curve = self.curve.reveal_until(_clamp_progress(progress))
        updated = dict(scene.curves)
        updated[self.curve.curve_id] = partial_curve
        return Scene(curves=updated, fills=scene.fills)

    def apply(self, scene: Scene) -> Scene:
        return scene.add_curve(self.curve)

    def frame_state(self, scene: Scene, progress: float) -> FrameState:
        in_between_scene = self.interpolate(scene, progress)
        pointer = None

        if self.show_pointer:
            partial_curve = in_between_scene.get_curve(self.curve.curve_id)
            if not partial_curve.is_empty:
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
                pointer = PointerOverlay(
                    x=float(partial_curve.x[-1]),
                    y=float(partial_curve.y[-1]),
                    artist_kwargs=pointer_style,
                )

        return FrameState(scene=in_between_scene, pointer=pointer)


@dataclass(frozen=True)
class FillBetweenTransition(Transition):
    fill: FillBetweenArea

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        if scene.contains_fill(self.fill.fill_id):
            raise ValueError(f"Fill {self.fill.fill_id!r} already exists in the scene.")

        partial_fill = self.fill.reveal_until(_clamp_progress(progress))
        updated = dict(scene.fills)
        updated[self.fill.fill_id] = partial_fill
        return Scene(curves=scene.curves, fills=updated)

    def apply(self, scene: Scene) -> Scene:
        return scene.add_fill(self.fill)


@dataclass(frozen=True)
class EraseTransition(Transition):
    curve_id: str

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        curve = scene.get_curve(self.curve_id)
        updated = dict(scene.curves)
        updated[self.curve_id] = curve.hide_until(_clamp_progress(progress))
        return Scene(curves=updated, fills=scene.fills)

    def apply(self, scene: Scene) -> Scene:
        return scene.remove_curve(self.curve_id)


@dataclass(frozen=True)
class MoveTransition(Transition):
    curve_id: str
    x_prime: npt.ArrayLike
    y_prime: npt.ArrayLike
    color: str | None = None
    alpha: float | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    line_kwargs: Mapping[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        x_prime = _coerce_unit_interval_array(self.x_prime, "x_prime")
        y_prime = _coerce_unit_interval_array(self.y_prime, "y_prime")

        if x_prime.shape != y_prime.shape:
            raise ValueError("x_prime and y_prime must have the same shape.")

        object.__setattr__(self, "x_prime", x_prime)
        object.__setattr__(self, "y_prime", y_prime)
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))
        if self.line_kwargs is not None:
            object.__setattr__(self, "line_kwargs", dict(self.line_kwargs))

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        source_curve = scene.get_curve(self.curve_id)
        self._validate_source_shape(source_curve)

        t = _clamp_progress(progress)
        x_values = source_curve.x + (self.x_prime - source_curve.x) * t
        y_values = source_curve.y + (self.y_prime - source_curve.y) * t

        return scene.update_curve(self._updated_curve(source_curve, x_values, y_values))

    def apply(self, scene: Scene) -> Scene:
        source_curve = scene.get_curve(self.curve_id)
        self._validate_source_shape(source_curve)
        return scene.update_curve(self._updated_curve(source_curve, self.x_prime, self.y_prime))

    def _updated_curve(
        self,
        source_curve: Curve,
        x_values: npt.ArrayLike,
        y_values: npt.ArrayLike,
    ) -> Curve:
        return source_curve.copy_with(
            x=x_values,
            y=y_values,
            color=self.color,
            alpha=self.alpha,
            linestyle=self.linestyle,
            linewidth=self.linewidth,
            line_kwargs=source_curve.line_kwargs if self.line_kwargs is None else self.line_kwargs,
        )

    def _validate_source_shape(self, source_curve: Curve) -> None:
        if source_curve.x.shape != self.x_prime.shape:
            raise ValueError(
                "MoveTransition requires x_prime and y_prime to match the source curve shape."
            )


@dataclass(frozen=True)
class MoveFillBetweenTransition(Transition):
    fill_id: str
    x_prime: npt.ArrayLike
    y1_prime: npt.ArrayLike
    y2_prime: npt.ArrayLike | float
    color: str | None = None
    alpha: float | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    fill_kwargs: Mapping[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        x_prime = _coerce_unit_interval_array(self.x_prime, "x_prime")
        y1_prime = _coerce_unit_interval_array(self.y1_prime, "y1_prime")

        if x_prime.shape != y1_prime.shape:
            raise ValueError("x_prime and y1_prime must have the same shape.")

        y2_prime = _coerce_matching_unit_interval_array(self.y2_prime, "y2_prime", x_prime.shape)

        object.__setattr__(self, "x_prime", x_prime)
        object.__setattr__(self, "y1_prime", y1_prime)
        object.__setattr__(self, "y2_prime", y2_prime)
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))
        if self.fill_kwargs is not None:
            object.__setattr__(self, "fill_kwargs", dict(self.fill_kwargs))

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        source_fill = scene.get_fill(self.fill_id)
        self._validate_source_shape(source_fill)

        t = _clamp_progress(progress)
        x_values = source_fill.x + (self.x_prime - source_fill.x) * t
        y1_values = source_fill.y1 + (self.y1_prime - source_fill.y1) * t
        y2_values = source_fill.y2 + (self.y2_prime - source_fill.y2) * t

        return scene.update_fill(self._updated_fill(source_fill, x_values, y1_values, y2_values))

    def apply(self, scene: Scene) -> Scene:
        source_fill = scene.get_fill(self.fill_id)
        self._validate_source_shape(source_fill)
        return scene.update_fill(
            self._updated_fill(source_fill, self.x_prime, self.y1_prime, self.y2_prime)
        )

    def _updated_fill(
        self,
        source_fill: FillBetweenArea,
        x_values: npt.ArrayLike,
        y1_values: npt.ArrayLike,
        y2_values: npt.ArrayLike,
    ) -> FillBetweenArea:
        return source_fill.copy_with(
            x=x_values,
            y1=y1_values,
            y2=y2_values,
            color=self.color,
            alpha=self.alpha,
            linestyle=self.linestyle,
            linewidth=self.linewidth,
            fill_kwargs=source_fill.fill_kwargs if self.fill_kwargs is None else self.fill_kwargs,
        )

    def _validate_source_shape(self, source_fill: FillBetweenArea) -> None:
        if source_fill.x.shape != self.x_prime.shape:
            raise ValueError(
                "MoveFillBetweenTransition requires target arrays to match the source fill shape."
            )


@dataclass(frozen=True)
class StressTransition(Transition):
    curve_id: str
    color: str | None = None
    max_alpha: float = 0.35
    glow_linewidth: float | None = None
    linestyle: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_alpha", _validate_alpha(self.max_alpha))

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        return scene

    def apply(self, scene: Scene) -> Scene:
        return scene

    def frame_state(self, scene: Scene, progress: float) -> FrameState:
        curve = scene.get_curve(self.curve_id)
        if curve.is_empty:
            return FrameState(scene=scene)

        strength = float(np.sin(np.pi * _clamp_progress(progress)))
        if strength <= 0.0:
            return FrameState(scene=scene)

        curve_style = curve.mpl_line_kwargs()
        base_linewidth = float(curve_style.get("linewidth", 2.0))
        glow_style = {
            "color": self.color or curve_style.get("color", "#f59e0b"),
            "alpha": self.max_alpha * strength,
            "linewidth": (
                self.glow_linewidth
                if self.glow_linewidth is not None
                else max(base_linewidth * 3.0, base_linewidth + 2.0)
            ),
            "linestyle": self.linestyle or curve_style.get("linestyle", "-"),
            "solid_capstyle": "round",
            "zorder": curve_style.get("zorder", 2.0) - 0.1,
        }

        return FrameState(
            scene=scene,
            glow=GlowOverlay(x=curve.x, y=curve.y, artist_kwargs=glow_style),
        )
