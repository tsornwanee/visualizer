from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy.typing as npt

from .scene import Curve, Scene, _clamp_progress, _coerce_unit_interval_array


@dataclass(frozen=True)
class Transition(ABC):
    """Base class for all scene-to-scene transitions."""

    @abstractmethod
    def interpolate(self, scene: Scene, progress: float) -> Scene:
        """Return the in-between scene for a normalized progress value."""

    @abstractmethod
    def apply(self, scene: Scene) -> Scene:
        """Return the scene after the transition completes."""


@dataclass(frozen=True)
class DrawTransition(Transition):
    curve: Curve

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        if scene.contains(self.curve.curve_id):
            raise ValueError(f"Curve {self.curve.curve_id!r} already exists in the scene.")

        partial_curve = self.curve.reveal_until(_clamp_progress(progress))
        updated = dict(scene.curves)
        updated[self.curve.curve_id] = partial_curve
        return Scene(updated)

    def apply(self, scene: Scene) -> Scene:
        return scene.add_curve(self.curve)


@dataclass(frozen=True)
class EraseTransition(Transition):
    curve_id: str

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        curve = scene.get_curve(self.curve_id)
        updated = dict(scene.curves)
        updated[self.curve_id] = curve.hide_until(_clamp_progress(progress))
        return Scene(updated)

    def apply(self, scene: Scene) -> Scene:
        return scene.remove_curve(self.curve_id)


@dataclass(frozen=True)
class MoveTransition(Transition):
    curve_id: str
    x_prime: npt.ArrayLike
    y_prime: npt.ArrayLike
    line_kwargs: Mapping[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        x_prime = _coerce_unit_interval_array(self.x_prime, "x_prime")
        y_prime = _coerce_unit_interval_array(self.y_prime, "y_prime")

        if x_prime.shape != y_prime.shape:
            raise ValueError("x_prime and y_prime must have the same shape.")

        object.__setattr__(self, "x_prime", x_prime)
        object.__setattr__(self, "y_prime", y_prime)
        if self.line_kwargs is not None:
            object.__setattr__(self, "line_kwargs", dict(self.line_kwargs))

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        source_curve = scene.get_curve(self.curve_id)
        self._validate_source_shape(source_curve)

        t = _clamp_progress(progress)
        x_values = source_curve.x + (self.x_prime - source_curve.x) * t
        y_values = source_curve.y + (self.y_prime - source_curve.y) * t

        updated_curve = source_curve.copy_with(
            x=x_values,
            y=y_values,
            line_kwargs=source_curve.line_kwargs if self.line_kwargs is None else self.line_kwargs,
        )
        return scene.update_curve(updated_curve)

    def apply(self, scene: Scene) -> Scene:
        source_curve = scene.get_curve(self.curve_id)
        self._validate_source_shape(source_curve)

        updated_curve = source_curve.copy_with(
            x=self.x_prime,
            y=self.y_prime,
            line_kwargs=source_curve.line_kwargs if self.line_kwargs is None else self.line_kwargs,
        )
        return scene.update_curve(updated_curve)

    def _validate_source_shape(self, source_curve: Curve) -> None:
        if source_curve.x.shape != self.x_prime.shape:
            raise ValueError(
                "MoveTransition requires x_prime and y_prime to match the source curve shape."
            )
