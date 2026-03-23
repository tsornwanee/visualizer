from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt
from matplotlib.colors import to_hex, to_rgb

from .scene import (
    Curve,
    FillBetweenArea,
    FloatArray,
    Scatter,
    Scene,
    Text,
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


def _current_fill_positive_color(fill: FillBetweenArea) -> Any:
    if fill.positive_color is not None:
        return fill.positive_color
    return _current_fill_color(fill)


def _current_fill_negative_color(fill: FillBetweenArea) -> Any:
    if fill.negative_color is not None:
        return fill.negative_color
    return _current_fill_color(fill)


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


def _current_text_color(text: Text) -> Any:
    return text.mpl_text_kwargs().get("color", "#111827")


def _current_text_alpha(text: Text) -> float:
    style = text.mpl_text_kwargs()
    alpha = style.get("alpha")
    if alpha is None:
        return 1.0
    return float(alpha)


def _current_text_fontsize(text: Text) -> float:
    style = text.mpl_text_kwargs()
    fontsize = style.get("fontsize", 12.0)
    return float(fontsize)


def _current_text_rotation(text: Text) -> float:
    style = text.mpl_text_kwargs()
    rotation = style.get("rotation", 0.0)
    return float(rotation)


def _validate_direction(direction: str) -> str:
    if direction not in {"forward", "backward"}:
        raise ValueError("direction must be 'forward' or 'backward'.")
    return direction


def _coerce_pause_sequence(
    values: float | Sequence[float] | npt.NDArray[np.float64] | None,
    *,
    name: str,
) -> list[float]:
    if values is None:
        return []
    return _coerce_component_values(values, name=name)


def _normalize_reveal_pauses(
    pause_at: float | Sequence[float] | npt.NDArray[np.float64] | None,
    pause_for: float | Sequence[float] | npt.NDArray[np.float64],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    points = _coerce_pause_sequence(pause_at, name="pause_at")
    durations = _coerce_pause_sequence(pause_for, name="pause_for") if points else []

    if not points:
        if any(duration != 0.0 for duration in _coerce_pause_sequence(pause_for, name="pause_for")):
            raise ValueError("pause_for requires pause_at.")
        return (), ()

    if len(durations) == 1 and len(points) > 1:
        durations = durations * len(points)
    if len(durations) != len(points):
        raise ValueError("pause_for must have length 1 or match pause_at.")
    if any(point < 0.0 or point > 1.0 for point in points):
        raise ValueError("pause_at values must lie in [0, 1].")
    if any(duration < 0.0 for duration in durations):
        raise ValueError("pause_for values must be non-negative.")

    merged_points: list[float] = []
    merged_durations: list[float] = []
    for point, duration in sorted(zip(points, durations), key=lambda pair: pair[0]):
        if merged_points and np.isclose(merged_points[-1], point):
            merged_durations[-1] += float(duration)
            continue
        merged_points.append(float(point))
        merged_durations.append(float(duration))

    total_pause = float(sum(merged_durations))
    if total_pause >= 1.0:
        raise ValueError("pause_for values must sum to less than 1.0.")

    return tuple(merged_points), tuple(merged_durations)


def _apply_reveal_pauses(
    progress: float,
    pause_at: Sequence[float],
    pause_for: Sequence[float],
) -> float:
    t = _clamp_progress(progress)
    if not pause_at:
        return t

    total_pause = float(sum(pause_for))
    if np.isclose(total_pause, 0.0):
        return t

    motion_share = 1.0 - total_pause
    elapsed_time = 0.0
    previous_point = 0.0

    for point, hold in zip(pause_at, pause_for):
        segment_progress = float(point) - previous_point
        segment_time = motion_share * segment_progress

        if segment_time > 0.0 and t <= elapsed_time + segment_time:
            local_progress = (t - elapsed_time) / segment_time
            return previous_point + local_progress * segment_progress

        elapsed_time += max(segment_time, 0.0)
        if hold > 0.0 and t <= elapsed_time + hold:
            return float(point)

        elapsed_time += hold
        previous_point = float(point)

    final_segment_progress = 1.0 - previous_point
    final_segment_time = motion_share * final_segment_progress
    if final_segment_time <= 0.0:
        return 1.0
    local_progress = (t - elapsed_time) / final_segment_time
    return _clamp_progress(previous_point + local_progress * final_segment_progress)


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


def _jitter_phase_pair(seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    return (
        float(rng.uniform(0.0, 2.0 * np.pi)),
        float(rng.uniform(0.0, 2.0 * np.pi)),
    )


def _coerce_component_values(
    values: float | Sequence[float] | npt.NDArray[np.float64],
    *,
    name: str,
) -> list[float]:
    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            result = [float(values)]
        elif values.ndim == 1:
            result = [float(value) for value in values.tolist()]
        else:
            raise ValueError(f"{name} must be a scalar or one-dimensional sequence.")
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        result = [float(value) for value in values]
    else:
        result = [float(values)]

    if not result:
        raise ValueError(f"{name} must not be empty.")
    if not all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")

    return result


def _coerce_seed_values(
    seeds: int | Sequence[int] | npt.NDArray[np.int64],
    *,
    components: int,
    name: str,
) -> list[int]:
    if isinstance(seeds, np.ndarray):
        if seeds.ndim == 0:
            base_seed = int(seeds)
            return [base_seed + index for index in range(components)]
        if seeds.ndim != 1:
            raise ValueError(f"{name} must be a scalar or one-dimensional sequence.")
        values = [int(value) for value in seeds.tolist()]
    elif isinstance(seeds, Sequence) and not isinstance(seeds, (str, bytes)):
        values = [int(value) for value in seeds]
    else:
        base_seed = int(seeds)
        return [base_seed + index for index in range(components)]

    if not values:
        raise ValueError(f"{name} must not be empty.")
    if len(values) == 1 and components > 1:
        base_seed = values[0]
        return [base_seed + index for index in range(components)]
    if len(values) != components:
        raise ValueError(f"{name} must have length 1 or {components}.")
    return values


def _normalize_jitter_components(
    amplitudes: float | Sequence[float] | npt.NDArray[np.float64],
    cycles: float | Sequence[float] | npt.NDArray[np.float64],
    seeds: int | Sequence[int] | npt.NDArray[np.int64],
    *,
    amplitude_name: str,
    cycles_name: str,
    seed_name: str,
) -> tuple[list[float], list[float], list[int]]:
    amplitude_values = _coerce_component_values(amplitudes, name=amplitude_name)
    cycle_values = _coerce_component_values(cycles, name=cycles_name)

    component_count = max(len(amplitude_values), len(cycle_values))
    if len(amplitude_values) == 1 and component_count > 1:
        amplitude_values = amplitude_values * component_count
    if len(cycle_values) == 1 and component_count > 1:
        cycle_values = cycle_values * component_count

    if len(amplitude_values) != component_count:
        raise ValueError(f"{amplitude_name} must have length 1 or {component_count}.")
    if len(cycle_values) != component_count:
        raise ValueError(f"{cycles_name} must have length 1 or {component_count}.")

    if any(amplitude < 0.0 for amplitude in amplitude_values):
        raise ValueError(f"{amplitude_name} must be non-negative.")
    if any(cycle <= 0.0 for cycle in cycle_values):
        raise ValueError(f"{cycles_name} must be positive.")

    seed_values = _coerce_seed_values(seeds, components=component_count, name=seed_name)
    return amplitude_values, cycle_values, seed_values


def _combine_jitter_components(
    size: int,
    *,
    progress: float,
    envelope: float,
    amplitudes: float | Sequence[float] | npt.NDArray[np.float64],
    cycles: float | Sequence[float] | npt.NDArray[np.float64],
    seeds: int | Sequence[int] | npt.NDArray[np.int64],
    oscillation_scale: float,
    spatial_scale: float,
    axis: str,
    amplitude_name: str,
    cycles_name: str,
    seed_name: str,
) -> FloatArray:
    amplitude_values, cycle_values, seed_values = _normalize_jitter_components(
        amplitudes,
        cycles,
        seeds,
        amplitude_name=amplitude_name,
        cycles_name=cycles_name,
        seed_name=seed_name,
    )

    offset = np.zeros(size, dtype=float)
    spatial = np.linspace(0.0, 2.0 * np.pi, size)

    for amplitude, cycle, seed in zip(amplitude_values, cycle_values, seed_values):
        if amplitude == 0.0:
            continue

        oscillation = 2.0 * np.pi * cycle * progress
        phase_x, phase_y = _jitter_phase_pair(seed)
        phase = phase_x if axis == "x" else phase_y
        offset += envelope * amplitude * np.sin(
            oscillation_scale * oscillation + spatial_scale * spatial + phase
        )

    return offset


def _jitter_x_offset(
    size: int,
    *,
    progress: float,
    envelope: float,
    amplitude: float | Sequence[float] | npt.NDArray[np.float64],
    cycles: float | Sequence[float] | npt.NDArray[np.float64],
    seed: int | Sequence[int] | npt.NDArray[np.int64],
) -> FloatArray:
    return _combine_jitter_components(
        size,
        progress=progress,
        envelope=envelope,
        amplitudes=amplitude,
        cycles=cycles,
        seeds=seed,
        oscillation_scale=1.0,
        spatial_scale=2.0,
        axis="x",
        amplitude_name="x_amplitude",
        cycles_name="cycles",
        seed_name="seed",
    )


def _jitter_y_offset(
    size: int,
    *,
    progress: float,
    envelope: float,
    amplitude: float | Sequence[float] | npt.NDArray[np.float64],
    cycles: float | Sequence[float] | npt.NDArray[np.float64],
    seed: int | Sequence[int] | npt.NDArray[np.int64],
    oscillation_scale: float,
    spatial_scale: float,
) -> FloatArray:
    return _combine_jitter_components(
        size,
        progress=progress,
        envelope=envelope,
        amplitudes=amplitude,
        cycles=cycles,
        seeds=seed,
        oscillation_scale=oscillation_scale,
        spatial_scale=spatial_scale,
        axis="y",
        amplitude_name="y_amplitude",
        cycles_name="cycles",
        seed_name="seed",
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
    pause_at: float | Sequence[float] | npt.NDArray[np.float64] | None = None
    pause_for: float | Sequence[float] | npt.NDArray[np.float64] = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "pointer_kwargs", dict(self.pointer_kwargs))
        object.__setattr__(self, "direction", _validate_direction(self.direction))
        object.__setattr__(self, "timeline_domain", _normalize_timeline_domain(self.timeline_domain))
        normalized_pause_at, normalized_pause_for = _normalize_reveal_pauses(
            self.pause_at,
            self.pause_for,
        )
        object.__setattr__(self, "pause_at", normalized_pause_at)
        object.__setattr__(self, "pause_for", normalized_pause_for)

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        if scene.contains_curve(self.curve.curve_id):
            raise ValueError(f"Curve {self.curve.curve_id!r} already exists in the scene.")

        partial_curve = self.curve.reveal_by_progress(
            self._effective_progress(progress),
            timeline_domain=self._effective_timeline_domain(None),
            direction=self.direction,
        )
        updated = dict(scene.curves)
        updated[self.curve.curve_id] = partial_curve
        return Scene(curves=updated, scatters=scene.scatters, fills=scene.fills, texts=scene.texts)

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
            self._effective_progress(progress),
            timeline_domain=self._effective_timeline_domain(timeline_domain),
            direction=self.direction,
        )
        updated = dict(scene.curves)
        updated[self.curve.curve_id] = partial_curve
        return Scene(curves=updated, scatters=scene.scatters, fills=scene.fills, texts=scene.texts)

    def _effective_progress(self, progress: float) -> float:
        return _apply_reveal_pauses(progress, self.pause_at, self.pause_for)


@dataclass(frozen=True)
class FillBetweenTransition(Transition):
    fill: FillBetweenArea
    direction: str = "forward"
    timeline_domain: tuple[float, float] | None = None
    pause_at: float | Sequence[float] | npt.NDArray[np.float64] | None = None
    pause_for: float | Sequence[float] | npt.NDArray[np.float64] = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", _validate_direction(self.direction))
        object.__setattr__(self, "timeline_domain", _normalize_timeline_domain(self.timeline_domain))
        normalized_pause_at, normalized_pause_for = _normalize_reveal_pauses(
            self.pause_at,
            self.pause_for,
        )
        object.__setattr__(self, "pause_at", normalized_pause_at)
        object.__setattr__(self, "pause_for", normalized_pause_for)

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        if scene.contains_fill(self.fill.fill_id):
            raise ValueError(f"Fill {self.fill.fill_id!r} already exists in the scene.")

        partial_fill = self.fill.reveal_by_progress(
            self._effective_progress(progress),
            timeline_domain=self._effective_timeline_domain(None),
            direction=self.direction,
        )
        updated = dict(scene.fills)
        updated[self.fill.fill_id] = partial_fill
        return Scene(curves=scene.curves, scatters=scene.scatters, fills=updated, texts=scene.texts)

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
            self._effective_progress(progress),
            timeline_domain=self._effective_timeline_domain(timeline_domain),
            direction=self.direction,
        )
        updated = dict(scene.fills)
        updated[self.fill.fill_id] = partial_fill
        return Scene(curves=scene.curves, scatters=scene.scatters, fills=updated, texts=scene.texts)

    def _effective_progress(self, progress: float) -> float:
        return _apply_reveal_pauses(progress, self.pause_at, self.pause_for)


@dataclass(frozen=True)
class DrawScatterTransition(Transition):
    scatter: Scatter
    direction: str = "forward"
    timeline_domain: tuple[float, float] | None = None
    pause_at: float | Sequence[float] | npt.NDArray[np.float64] | None = None
    pause_for: float | Sequence[float] | npt.NDArray[np.float64] = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", _validate_direction(self.direction))
        object.__setattr__(self, "timeline_domain", _normalize_timeline_domain(self.timeline_domain))
        normalized_pause_at, normalized_pause_for = _normalize_reveal_pauses(
            self.pause_at,
            self.pause_for,
        )
        object.__setattr__(self, "pause_at", normalized_pause_at)
        object.__setattr__(self, "pause_for", normalized_pause_for)

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        if scene.contains_scatter(self.scatter.scatter_id):
            raise ValueError(f"Scatter {self.scatter.scatter_id!r} already exists in the scene.")

        partial_scatter = self.scatter.reveal_by_progress(
            self._effective_progress(progress),
            timeline_domain=self._effective_timeline_domain(None),
            direction=self.direction,
        )
        updated = dict(scene.scatters)
        updated[self.scatter.scatter_id] = partial_scatter
        return Scene(curves=scene.curves, scatters=updated, fills=scene.fills, texts=scene.texts)

    def apply(self, scene: Scene) -> Scene:
        return scene.add_scatter(self.scatter)

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
        return (float(np.min(self.scatter.x)), float(np.max(self.scatter.x)))

    def _interpolated_scene(
        self,
        scene: Scene,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None,
    ) -> Scene:
        if scene.contains_scatter(self.scatter.scatter_id):
            raise ValueError(f"Scatter {self.scatter.scatter_id!r} already exists in the scene.")

        partial_scatter = self.scatter.reveal_by_progress(
            self._effective_progress(progress),
            timeline_domain=self._effective_timeline_domain(timeline_domain),
            direction=self.direction,
        )
        updated = dict(scene.scatters)
        updated[self.scatter.scatter_id] = partial_scatter
        return Scene(curves=scene.curves, scatters=updated, fills=scene.fills, texts=scene.texts)

    def _effective_progress(self, progress: float) -> float:
        return _apply_reveal_pauses(progress, self.pause_at, self.pause_for)


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
        return Scene(curves=updated, scatters=scene.scatters, fills=scene.fills, texts=scene.texts)


@dataclass(frozen=True)
class EraseScatterTransition(Transition):
    scatter_id: str
    direction: str = "forward"
    timeline_domain: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", _validate_direction(self.direction))
        object.__setattr__(self, "timeline_domain", _normalize_timeline_domain(self.timeline_domain))

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        return self._interpolated_scene(scene, progress, timeline_domain=None)

    def apply(self, scene: Scene) -> Scene:
        return scene.remove_scatter(self.scatter_id)

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
        scatter = scene.get_scatter(self.scatter_id)
        if self.timeline_domain is not None:
            return self.timeline_domain
        return (float(np.min(scatter.x)), float(np.max(scatter.x)))

    def _interpolated_scene(
        self,
        scene: Scene,
        progress: float,
        *,
        timeline_domain: tuple[float, float] | None,
    ) -> Scene:
        scatter = scene.get_scatter(self.scatter_id)
        effective_domain = self.timeline_domain or timeline_domain or (
            float(np.min(scatter.x)),
            float(np.max(scatter.x)),
        )
        updated = dict(scene.scatters)
        updated[self.scatter_id] = scatter.hide_by_progress(
            _clamp_progress(progress),
            timeline_domain=effective_domain,
            direction=self.direction,
        )
        return Scene(curves=scene.curves, scatters=updated, fills=scene.fills, texts=scene.texts)


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
        return Scene(curves=scene.curves, scatters=scene.scatters, fills=updated, texts=scene.texts)


@dataclass(frozen=True, init=False)
class MoveTransition(Transition):
    curve_id: str
    newx: npt.ArrayLike | None = None
    newy: npt.ArrayLike | None = None
    color: str | None = None
    alpha: float | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    domain: Bounds | None = None
    value_range: Bounds | None = None
    line_kwargs: Mapping[str, Any] | None = field(default=None)

    def __init__(
        self,
        curve_id: str,
        newx: npt.ArrayLike | None = None,
        newy: npt.ArrayLike | None = None,
        *,
        x_prime: npt.ArrayLike | None = None,
        y_prime: npt.ArrayLike | None = None,
        color: str | None = None,
        alpha: float | None = None,
        linestyle: str | None = None,
        linewidth: float | None = None,
        domain: Bounds | None = None,
        value_range: Bounds | None = None,
        line_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if newx is not None and x_prime is not None:
            raise ValueError("newx and x_prime cannot both be provided.")
        if newy is not None and y_prime is not None:
            raise ValueError("newy and y_prime cannot both be provided.")

        object.__setattr__(self, "curve_id", curve_id)
        object.__setattr__(self, "newx", x_prime if newx is None else newx)
        object.__setattr__(self, "newy", y_prime if newy is None else newy)
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "linestyle", linestyle)
        object.__setattr__(self, "linewidth", linewidth)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "value_range", value_range)
        object.__setattr__(self, "line_kwargs", line_kwargs)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.newy is None:
            raise TypeError("MoveTransition requires newy.")

        newy = _coerce_coordinate_array(self.newy, "newy")
        newx = None
        if self.newx is not None:
            newx = _coerce_coordinate_array(self.newx, "newx")

        if newx is not None and newx.shape != newy.shape:
            raise ValueError("newx and newy must have the same shape.")

        object.__setattr__(self, "newx", newx)
        object.__setattr__(self, "newy", newy)
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))
        object.__setattr__(self, "domain", _normalize_timeline_domain(self.domain))
        object.__setattr__(self, "value_range", _normalize_timeline_domain(self.value_range))
        if self.line_kwargs is not None:
            object.__setattr__(self, "line_kwargs", dict(self.line_kwargs))

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        source_curve = scene.get_curve(self.curve_id)
        newx = self._effective_newx(source_curve)
        self._validate_source_shape(source_curve, newx)

        t = _clamp_progress(progress)
        x_values = source_curve.x + (newx - source_curve.x) * t
        y_values = source_curve.y + (self.newy - source_curve.y) * t

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
        newx = self._effective_newx(source_curve)
        self._validate_source_shape(source_curve, newx)
        return scene.update_curve(
            self._updated_curve(
                source_curve,
                newx,
                self.newy,
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

    def _effective_newx(self, source_curve: Curve) -> FloatArray:
        if self.newx is None:
            return source_curve.x
        return self.newx

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
        newx: FloatArray,
    ) -> None:
        if source_curve.x.shape != newx.shape or source_curve.x.shape != self.newy.shape:
            raise ValueError(
                "MoveTransition requires newx and newy to match the source curve shape."
            )


@dataclass(frozen=True, init=False)
class MoveScatterTransition(Transition):
    scatter_id: str
    newx: npt.ArrayLike | None = None
    newy: npt.ArrayLike | None = None
    color: Any | None = None
    alpha: float | None = None
    marker: Any | None = None
    size: npt.ArrayLike | float | None = None
    linewidth: float | None = None
    edgecolor: Any | None = None
    domain: Bounds | None = None
    value_range: Bounds | None = None
    scatter_kwargs: Mapping[str, Any] | None = field(default=None)

    def __init__(
        self,
        scatter_id: str,
        newx: npt.ArrayLike | None = None,
        newy: npt.ArrayLike | None = None,
        *,
        x_prime: npt.ArrayLike | None = None,
        y_prime: npt.ArrayLike | None = None,
        color: Any | None = None,
        alpha: float | None = None,
        marker: Any | None = None,
        size: npt.ArrayLike | float | None = None,
        linewidth: float | None = None,
        edgecolor: Any | None = None,
        domain: Bounds | None = None,
        value_range: Bounds | None = None,
        scatter_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if newx is not None and x_prime is not None:
            raise ValueError("newx and x_prime cannot both be provided.")
        if newy is not None and y_prime is not None:
            raise ValueError("newy and y_prime cannot both be provided.")

        object.__setattr__(self, "scatter_id", scatter_id)
        object.__setattr__(self, "newx", x_prime if newx is None else newx)
        object.__setattr__(self, "newy", y_prime if newy is None else newy)
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "marker", marker)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "linewidth", linewidth)
        object.__setattr__(self, "edgecolor", edgecolor)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "value_range", value_range)
        object.__setattr__(self, "scatter_kwargs", scatter_kwargs)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.newy is None:
            raise TypeError("MoveScatterTransition requires newy.")

        newy = _coerce_coordinate_array(self.newy, "newy")
        newx = None
        if self.newx is not None:
            newx = _coerce_coordinate_array(self.newx, "newx")
        if newx is not None and newx.shape != newy.shape:
            raise ValueError("newx and newy must have the same shape.")

        size = None
        if self.size is not None:
            size = _coerce_matching_coordinate_array(self.size, "size", newy.shape)
            if np.any(size < 0.0):
                raise ValueError("size must be non-negative.")

        if self.linewidth is not None and self.linewidth < 0:
            raise ValueError("linewidth must be non-negative.")

        object.__setattr__(self, "newx", newx)
        object.__setattr__(self, "newy", newy)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))
        object.__setattr__(self, "domain", _normalize_timeline_domain(self.domain))
        object.__setattr__(self, "value_range", _normalize_timeline_domain(self.value_range))
        if self.scatter_kwargs is not None:
            object.__setattr__(self, "scatter_kwargs", dict(self.scatter_kwargs))

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        source_scatter = scene.get_scatter(self.scatter_id)
        newx = self._effective_newx(source_scatter)
        newsize = self._effective_size(source_scatter)
        self._validate_source_shape(source_scatter, newx, newsize)

        t = _clamp_progress(progress)
        x_values = source_scatter.x + (newx - source_scatter.x) * t
        y_values = source_scatter.y + (self.newy - source_scatter.y) * t
        size_values = source_scatter.size + (newsize - source_scatter.size) * t

        return scene.update_scatter(
            self._updated_scatter(
                source_scatter,
                x_values,
                y_values,
                size_values,
                domain=self._interpolated_domain(source_scatter, t),
                value_range=self._interpolated_value_range(source_scatter, t),
            )
        )

    def apply(self, scene: Scene) -> Scene:
        source_scatter = scene.get_scatter(self.scatter_id)
        newx = self._effective_newx(source_scatter)
        newsize = self._effective_size(source_scatter)
        self._validate_source_shape(source_scatter, newx, newsize)

        return scene.update_scatter(
            self._updated_scatter(
                source_scatter,
                newx,
                self.newy,
                newsize,
                domain=self.domain,
                value_range=self.value_range,
            )
        )

    def _updated_scatter(
        self,
        source_scatter: Scatter,
        x_values: npt.ArrayLike,
        y_values: npt.ArrayLike,
        size_values: npt.ArrayLike,
        *,
        domain: Bounds | None = None,
        value_range: Bounds | None = None,
    ) -> Scatter:
        return source_scatter.copy_with(
            x=x_values,
            y=y_values,
            color=self.color,
            alpha=self.alpha,
            marker=self.marker,
            size=size_values,
            linewidth=self.linewidth,
            edgecolor=self.edgecolor,
            domain=domain,
            value_range=value_range,
            scatter_kwargs=source_scatter.scatter_kwargs if self.scatter_kwargs is None else self.scatter_kwargs,
        )

    def _effective_newx(self, source_scatter: Scatter) -> FloatArray:
        if self.newx is None:
            return source_scatter.x
        return self.newx

    def _effective_size(self, source_scatter: Scatter) -> FloatArray:
        if self.size is None:
            return source_scatter.size
        return self.size

    def _interpolated_domain(self, source_scatter: Scatter, progress: float) -> Bounds | None:
        if self.domain is None:
            return source_scatter.domain

        start_bounds = source_scatter.domain or self._scatter_x_extent(source_scatter)
        return _interpolate_bounds(start_bounds, self.domain, progress)

    def _interpolated_value_range(
        self,
        source_scatter: Scatter,
        progress: float,
    ) -> Bounds | None:
        if self.value_range is None:
            return source_scatter.value_range

        start_bounds = source_scatter.value_range or self._scatter_y_extent(source_scatter)
        return _interpolate_bounds(start_bounds, self.value_range, progress)

    def _scatter_x_extent(self, scatter: Scatter) -> Bounds:
        return (float(np.min(scatter.x)), float(np.max(scatter.x)))

    def _scatter_y_extent(self, scatter: Scatter) -> Bounds:
        return (float(np.min(scatter.y)), float(np.max(scatter.y)))

    def _validate_source_shape(
        self,
        source_scatter: Scatter,
        newx: FloatArray,
        newsize: FloatArray,
    ) -> None:
        if (
            source_scatter.x.shape != newx.shape
            or source_scatter.x.shape != self.newy.shape
            or source_scatter.x.shape != newsize.shape
        ):
            raise ValueError(
                "MoveScatterTransition requires newx, newy, and size to match the source scatter shape."
            )


@dataclass(frozen=True, init=False)
class MoveFillBetweenTransition(Transition):
    fill_id: str
    newx: npt.ArrayLike | None = None
    newy1: npt.ArrayLike | None = None
    newy2: npt.ArrayLike | float | None = None
    color: str | None = None
    positive_color: str | None = None
    negative_color: str | None = None
    alpha: float | None = None
    linestyle: str | None = None
    linewidth: float | None = None
    domain: Bounds | None = None
    value_range: Bounds | None = None
    fill_kwargs: Mapping[str, Any] | None = field(default=None)

    def __init__(
        self,
        fill_id: str,
        newx: npt.ArrayLike | None = None,
        newy1: npt.ArrayLike | None = None,
        newy2: npt.ArrayLike | float | None = None,
        *,
        x_prime: npt.ArrayLike | None = None,
        y1_prime: npt.ArrayLike | None = None,
        y2_prime: npt.ArrayLike | float | None = None,
        color: str | None = None,
        positive_color: str | None = None,
        negative_color: str | None = None,
        alpha: float | None = None,
        linestyle: str | None = None,
        linewidth: float | None = None,
        domain: Bounds | None = None,
        value_range: Bounds | None = None,
        fill_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if newx is not None and x_prime is not None:
            raise ValueError("newx and x_prime cannot both be provided.")
        if newy1 is not None and y1_prime is not None:
            raise ValueError("newy1 and y1_prime cannot both be provided.")
        if newy2 is not None and y2_prime is not None:
            raise ValueError("newy2 and y2_prime cannot both be provided.")

        object.__setattr__(self, "fill_id", fill_id)
        object.__setattr__(self, "newx", x_prime if newx is None else newx)
        object.__setattr__(self, "newy1", y1_prime if newy1 is None else newy1)
        object.__setattr__(self, "newy2", y2_prime if newy2 is None else newy2)
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "positive_color", positive_color)
        object.__setattr__(self, "negative_color", negative_color)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "linestyle", linestyle)
        object.__setattr__(self, "linewidth", linewidth)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "value_range", value_range)
        object.__setattr__(self, "fill_kwargs", fill_kwargs)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.newy1 is None or self.newy2 is None:
            raise TypeError("MoveFillBetweenTransition requires newy1 and newy2.")

        newy1 = _coerce_coordinate_array(self.newy1, "newy1")
        newx = None
        if self.newx is not None:
            newx = _coerce_coordinate_array(self.newx, "newx")

        if newx is not None and newx.shape != newy1.shape:
            raise ValueError("newx and newy1 must have the same shape.")

        newy2 = _coerce_matching_coordinate_array(self.newy2, "newy2", newy1.shape)

        object.__setattr__(self, "newx", newx)
        object.__setattr__(self, "newy1", newy1)
        object.__setattr__(self, "newy2", newy2)
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))
        object.__setattr__(self, "domain", _normalize_timeline_domain(self.domain))
        object.__setattr__(self, "value_range", _normalize_timeline_domain(self.value_range))
        if self.fill_kwargs is not None:
            object.__setattr__(self, "fill_kwargs", dict(self.fill_kwargs))

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        source_fill = scene.get_fill(self.fill_id)
        newx = self._effective_newx(source_fill)
        self._validate_source_shape(source_fill, newx)

        t = _clamp_progress(progress)
        x_values = source_fill.x + (newx - source_fill.x) * t
        y1_values = source_fill.y1 + (self.newy1 - source_fill.y1) * t
        y2_values = source_fill.y2 + (self.newy2 - source_fill.y2) * t

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
        newx = self._effective_newx(source_fill)
        self._validate_source_shape(source_fill, newx)
        return scene.update_fill(
            self._updated_fill(
                source_fill,
                newx,
                self.newy1,
                self.newy2,
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
            positive_color=self.positive_color,
            negative_color=self.negative_color,
            alpha=self.alpha,
            linestyle=self.linestyle,
            linewidth=self.linewidth,
            domain=domain,
            value_range=value_range,
            fill_kwargs=source_fill.fill_kwargs if self.fill_kwargs is None else self.fill_kwargs,
        )

    def _effective_newx(self, source_fill: FillBetweenArea) -> FloatArray:
        if self.newx is None:
            return source_fill.x
        return self.newx

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
        newx: FloatArray,
    ) -> None:
        if (
            source_fill.x.shape != newx.shape
            or source_fill.x.shape != self.newy1.shape
            or source_fill.x.shape != self.newy2.shape
        ):
            raise ValueError(
                "MoveFillBetweenTransition requires newx, newy1, and newy2 to match the source fill shape."
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
    positive_color: str | None = None
    negative_color: str | None = None
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

        positive_color = fill.positive_color
        if self.positive_color is not None:
            positive_color = (
                _interpolate_color(_current_fill_positive_color(fill), self.positive_color, t)
                if self.interpolate_color
                else fill.positive_color
            )

        negative_color = fill.negative_color
        if self.negative_color is not None:
            negative_color = (
                _interpolate_color(_current_fill_negative_color(fill), self.negative_color, t)
                if self.interpolate_color
                else fill.negative_color
            )

        alpha = fill.alpha
        if self.alpha is not None:
            alpha = _interpolate_float(_current_fill_alpha(fill), self.alpha, t)

        linewidth = fill.linewidth
        if self.linewidth is not None:
            linewidth = _interpolate_float(_current_fill_linewidth(fill), self.linewidth, t)

        updated_fill = fill.copy_with(
            color=color,
            positive_color=positive_color,
            negative_color=negative_color,
            alpha=alpha,
            linewidth=linewidth,
        )
        return scene.update_fill(updated_fill)

    def apply(self, scene: Scene) -> Scene:
        fill = scene.get_fill(self.fill_id)
        updated_fill = fill.copy_with(
            color=self.color,
            positive_color=self.positive_color,
            negative_color=self.negative_color,
            alpha=self.alpha,
            linestyle=self.linestyle,
            linewidth=self.linewidth,
            fill_kwargs=fill.fill_kwargs if self.fill_kwargs is None else self.fill_kwargs,
        )
        return scene.update_fill(updated_fill)


@dataclass(frozen=True)
class DrawTextTransition(Transition):
    text: Text

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        if scene.contains_text(self.text.text_id):
            raise ValueError(f"Text {self.text.text_id!r} already exists in the scene.")

        t = _clamp_progress(progress)
        target_alpha = _current_text_alpha(self.text)
        partial_text = self.text.copy_with(alpha=_interpolate_float(0.0, target_alpha, t))
        return scene.add_text(partial_text)

    def apply(self, scene: Scene) -> Scene:
        return scene.add_text(self.text)


@dataclass(frozen=True)
class EraseTextTransition(Transition):
    text_id: str

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        text = scene.get_text(self.text_id)
        t = _clamp_progress(progress)
        updated_text = text.copy_with(alpha=_interpolate_float(_current_text_alpha(text), 0.0, t))
        return scene.update_text(updated_text)

    def apply(self, scene: Scene) -> Scene:
        return scene.remove_text(self.text_id)


@dataclass(frozen=True, init=False)
class MoveTextTransition(Transition):
    text_id: str
    newx: float | None = None
    newy: float | None = None
    content: str | None = None
    color: str | None = None
    alpha: float | None = None
    fontsize: float | None = None
    ha: str | None = None
    va: str | None = None
    rotation: float | None = None
    domain: Bounds | None = None
    value_range: Bounds | None = None
    text_kwargs: Mapping[str, Any] | None = field(default=None)
    interpolate_color: bool = True

    def __init__(
        self,
        text_id: str,
        newx: float | None = None,
        newy: float | None = None,
        *,
        x_prime: float | None = None,
        y_prime: float | None = None,
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
        interpolate_color: bool = True,
    ) -> None:
        if newx is not None and x_prime is not None:
            raise ValueError("newx and x_prime cannot both be provided.")
        if newy is not None and y_prime is not None:
            raise ValueError("newy and y_prime cannot both be provided.")

        object.__setattr__(self, "text_id", text_id)
        object.__setattr__(self, "newx", x_prime if newx is None else newx)
        object.__setattr__(self, "newy", y_prime if newy is None else newy)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "color", color)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "fontsize", fontsize)
        object.__setattr__(self, "ha", ha)
        object.__setattr__(self, "va", va)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "value_range", value_range)
        object.__setattr__(self, "text_kwargs", text_kwargs)
        object.__setattr__(self, "interpolate_color", interpolate_color)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.newx is not None and not np.isfinite(self.newx):
            raise ValueError("newx must be finite.")
        if self.newy is not None and not np.isfinite(self.newy):
            raise ValueError("newy must be finite.")
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))
        if self.fontsize is not None and self.fontsize < 0:
            raise ValueError("fontsize must be non-negative.")
        if self.rotation is not None and not np.isfinite(self.rotation):
            raise ValueError("rotation must be finite.")
        object.__setattr__(self, "domain", _normalize_timeline_domain(self.domain))
        object.__setattr__(self, "value_range", _normalize_timeline_domain(self.value_range))
        if self.text_kwargs is not None:
            object.__setattr__(self, "text_kwargs", dict(self.text_kwargs))

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        text = scene.get_text(self.text_id)
        t = _clamp_progress(progress)

        color = text.color
        if self.color is not None:
            color = (
                _interpolate_color(_current_text_color(text), self.color, t)
                if self.interpolate_color
                else text.color
            )

        alpha = text.alpha
        if self.alpha is not None:
            alpha = _interpolate_float(_current_text_alpha(text), self.alpha, t)

        fontsize = text.fontsize
        if self.fontsize is not None:
            fontsize = _interpolate_float(_current_text_fontsize(text), self.fontsize, t)

        rotation = text.rotation
        if self.rotation is not None:
            rotation = _interpolate_float(_current_text_rotation(text), self.rotation, t)

        updated_text = text.copy_with(
            x=_interpolate_float(text.x, self._effective_newx(text), t),
            y=_interpolate_float(text.y, self._effective_newy(text), t),
            color=color,
            alpha=alpha,
            fontsize=fontsize,
            rotation=rotation,
            domain=self._interpolated_domain(text, t),
            value_range=self._interpolated_value_range(text, t),
        )
        return scene.update_text(updated_text)

    def apply(self, scene: Scene) -> Scene:
        text = scene.get_text(self.text_id)
        updated_text = text.copy_with(
            x=self._effective_newx(text),
            y=self._effective_newy(text),
            content=self.content,
            color=self.color,
            alpha=self.alpha,
            fontsize=self.fontsize,
            ha=self.ha,
            va=self.va,
            rotation=self.rotation,
            domain=self.domain,
            value_range=self.value_range,
            text_kwargs=text.text_kwargs if self.text_kwargs is None else self.text_kwargs,
        )
        return scene.update_text(updated_text)

    def _effective_newx(self, text: Text) -> float:
        if self.newx is None:
            return text.x
        return float(self.newx)

    def _effective_newy(self, text: Text) -> float:
        if self.newy is None:
            return text.y
        return float(self.newy)

    def _interpolated_domain(self, text: Text, progress: float) -> Bounds | None:
        if self.domain is None:
            return text.domain

        start_bounds = text.domain or (text.x, text.x)
        return _interpolate_bounds(start_bounds, self.domain, progress)

    def _interpolated_value_range(self, text: Text, progress: float) -> Bounds | None:
        if self.value_range is None:
            return text.value_range

        start_bounds = text.value_range or (text.y, text.y)
        return _interpolate_bounds(start_bounds, self.value_range, progress)


@dataclass(frozen=True)
class TextStyleTransition(Transition):
    text_id: str
    content: str | None = None
    color: str | None = None
    alpha: float | None = None
    fontsize: float | None = None
    ha: str | None = None
    va: str | None = None
    rotation: float | None = None
    text_kwargs: Mapping[str, Any] | None = field(default=None)
    interpolate_color: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))
        if self.fontsize is not None and self.fontsize < 0:
            raise ValueError("fontsize must be non-negative.")
        if self.rotation is not None and not np.isfinite(self.rotation):
            raise ValueError("rotation must be finite.")
        if self.text_kwargs is not None:
            object.__setattr__(self, "text_kwargs", dict(self.text_kwargs))

    def interpolate(self, scene: Scene, progress: float) -> Scene:
        text = scene.get_text(self.text_id)
        t = _clamp_progress(progress)

        color = text.color
        if self.color is not None:
            color = (
                _interpolate_color(_current_text_color(text), self.color, t)
                if self.interpolate_color
                else text.color
            )

        alpha = text.alpha
        if self.alpha is not None:
            alpha = _interpolate_float(_current_text_alpha(text), self.alpha, t)

        fontsize = text.fontsize
        if self.fontsize is not None:
            fontsize = _interpolate_float(_current_text_fontsize(text), self.fontsize, t)

        rotation = text.rotation
        if self.rotation is not None:
            rotation = _interpolate_float(_current_text_rotation(text), self.rotation, t)

        updated_text = text.copy_with(
            color=color,
            alpha=alpha,
            fontsize=fontsize,
            rotation=rotation,
        )
        return scene.update_text(updated_text)

    def apply(self, scene: Scene) -> Scene:
        text = scene.get_text(self.text_id)
        updated_text = text.copy_with(
            content=self.content,
            color=self.color,
            alpha=self.alpha,
            fontsize=self.fontsize,
            ha=self.ha,
            va=self.va,
            rotation=self.rotation,
            text_kwargs=text.text_kwargs if self.text_kwargs is None else self.text_kwargs,
        )
        return scene.update_text(updated_text)


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
    x_amplitude: float | Sequence[float] | npt.NDArray[np.float64] = 0.0
    y_amplitude: float | Sequence[float] | npt.NDArray[np.float64] = 0.02
    cycles: float | Sequence[float] | npt.NDArray[np.float64] = 10.0
    seed: int | Sequence[int] | npt.NDArray[np.int64] = 0

    def __post_init__(self) -> None:
        _normalize_jitter_components(
            self.x_amplitude,
            self.cycles,
            self.seed,
            amplitude_name="x_amplitude",
            cycles_name="cycles",
            seed_name="seed",
        )
        _normalize_jitter_components(
            self.y_amplitude,
            self.cycles,
            self.seed,
            amplitude_name="y_amplitude",
            cycles_name="cycles",
            seed_name="seed",
        )

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

        progress = _clamp_progress(progress)
        envelope = float(np.sin(np.pi * progress))
        x_components, _, _ = _normalize_jitter_components(
            self.x_amplitude,
            self.cycles,
            self.seed,
            amplitude_name="x_amplitude",
            cycles_name="cycles",
            seed_name="seed",
        )
        y_components, _, _ = _normalize_jitter_components(
            self.y_amplitude,
            self.cycles,
            self.seed,
            amplitude_name="y_amplitude",
            cycles_name="cycles",
            seed_name="seed",
        )
        if envelope <= 0.0 or (
            all(amplitude == 0.0 for amplitude in x_components)
            and all(amplitude == 0.0 for amplitude in y_components)
        ):
            return scene

        x_offset = _jitter_x_offset(
            curve.x.size,
            progress=progress,
            envelope=envelope,
            amplitude=self.x_amplitude,
            cycles=self.cycles,
            seed=self.seed,
        )
        y_offset = _jitter_y_offset(
            curve.x.size,
            progress=progress,
            envelope=envelope,
            amplitude=self.y_amplitude,
            cycles=self.cycles,
            seed=self.seed,
            oscillation_scale=1.3,
            spatial_scale=3.0,
        )

        perturbed_curve = curve.copy_with(
            x=curve.x + x_offset,
            y=curve.y + y_offset,
        )
        return scene.update_curve(perturbed_curve)


@dataclass(frozen=True)
class JitterFillBetweenTransition(Transition):
    fill_id: str
    x_amplitude: float | Sequence[float] | npt.NDArray[np.float64] = 0.0
    upper_y_amplitude: float | Sequence[float] | npt.NDArray[np.float64] | None = None
    lower_y_amplitude: float | Sequence[float] | npt.NDArray[np.float64] | None = None
    y1_amplitude: float | Sequence[float] | npt.NDArray[np.float64] = 0.02
    y2_amplitude: float | Sequence[float] | npt.NDArray[np.float64] = 0.0
    cycles: float | Sequence[float] | npt.NDArray[np.float64] | None = None
    seed: int | Sequence[int] | npt.NDArray[np.int64] | None = None
    upper_cycles: float | Sequence[float] | npt.NDArray[np.float64] | None = None
    lower_cycles: float | Sequence[float] | npt.NDArray[np.float64] | None = None
    upper_seed: int | Sequence[int] | npt.NDArray[np.int64] | None = None
    lower_seed: int | Sequence[int] | npt.NDArray[np.int64] | None = None

    def __post_init__(self) -> None:
        _normalize_jitter_components(
            self.x_amplitude,
            self._effective_x_cycles(),
            self._effective_x_seed(),
            amplitude_name="x_amplitude",
            cycles_name="cycles",
            seed_name="seed",
        )
        _normalize_jitter_components(
            self._effective_upper_y_amplitude(),
            self._effective_upper_cycles(),
            self._effective_upper_seed(),
            amplitude_name="upper_y_amplitude",
            cycles_name="upper_cycles",
            seed_name="upper_seed",
        )
        _normalize_jitter_components(
            self._effective_lower_y_amplitude(),
            self._effective_lower_cycles(),
            self._effective_lower_seed(),
            amplitude_name="lower_y_amplitude",
            cycles_name="lower_cycles",
            seed_name="lower_seed",
        )

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

        progress = _clamp_progress(progress)
        envelope = float(np.sin(np.pi * progress))
        x_components, _, _ = _normalize_jitter_components(
            self.x_amplitude,
            self._effective_x_cycles(),
            self._effective_x_seed(),
            amplitude_name="x_amplitude",
            cycles_name="cycles",
            seed_name="seed",
        )
        upper_components, _, _ = _normalize_jitter_components(
            self._effective_upper_y_amplitude(),
            self._effective_upper_cycles(),
            self._effective_upper_seed(),
            amplitude_name="upper_y_amplitude",
            cycles_name="upper_cycles",
            seed_name="upper_seed",
        )
        lower_components, _, _ = _normalize_jitter_components(
            self._effective_lower_y_amplitude(),
            self._effective_lower_cycles(),
            self._effective_lower_seed(),
            amplitude_name="lower_y_amplitude",
            cycles_name="lower_cycles",
            seed_name="lower_seed",
        )
        if envelope <= 0.0 or (
            all(amplitude == 0.0 for amplitude in x_components)
            and all(amplitude == 0.0 for amplitude in upper_components)
            and all(amplitude == 0.0 for amplitude in lower_components)
        ):
            return scene

        x_offset = _jitter_x_offset(
            fill.x.size,
            progress=progress,
            envelope=envelope,
            amplitude=self.x_amplitude,
            cycles=self._effective_x_cycles(),
            seed=self._effective_x_seed(),
        )
        y1_offset = _jitter_y_offset(
            fill.x.size,
            progress=progress,
            envelope=envelope,
            amplitude=self._effective_upper_y_amplitude(),
            cycles=self._effective_upper_cycles(),
            seed=self._effective_upper_seed(),
            oscillation_scale=1.3,
            spatial_scale=3.0,
        )
        y2_offset = _jitter_y_offset(
            fill.x.size,
            progress=progress,
            envelope=envelope,
            amplitude=self._effective_lower_y_amplitude(),
            cycles=self._effective_lower_cycles(),
            seed=self._effective_lower_seed(),
            oscillation_scale=0.9,
            spatial_scale=2.5,
        )

        perturbed_fill = fill.copy_with(
            x=fill.x + x_offset,
            y1=fill.y1 + y1_offset,
            y2=fill.y2 + y2_offset,
        )
        return scene.update_fill(perturbed_fill)

    def _effective_upper_y_amplitude(
        self,
    ) -> float | Sequence[float] | npt.NDArray[np.float64]:
        if self.upper_y_amplitude is not None:
            return self.upper_y_amplitude
        return self.y1_amplitude

    def _effective_lower_y_amplitude(
        self,
    ) -> float | Sequence[float] | npt.NDArray[np.float64]:
        if self.lower_y_amplitude is not None:
            return self.lower_y_amplitude
        return self.y2_amplitude

    def _effective_upper_cycles(
        self,
    ) -> float | Sequence[float] | npt.NDArray[np.float64]:
        if self.upper_cycles is not None:
            return self.upper_cycles
        if self.cycles is not None:
            return self.cycles
        return 10.0

    def _effective_lower_cycles(
        self,
    ) -> float | Sequence[float] | npt.NDArray[np.float64]:
        if self.lower_cycles is not None:
            return self.lower_cycles
        if self.cycles is not None:
            return self.cycles
        return 10.0

    def _effective_upper_seed(
        self,
    ) -> int | Sequence[int] | npt.NDArray[np.int64]:
        if self.upper_seed is not None:
            return self.upper_seed
        if self.seed is not None:
            return self.seed
        return 0

    def _effective_lower_seed(
        self,
    ) -> int | Sequence[int] | npt.NDArray[np.int64]:
        if self.lower_seed is not None:
            return self.lower_seed
        if self.seed is not None:
            if isinstance(self.seed, np.ndarray):
                return np.asarray(self.seed, dtype=int) + 1
            if isinstance(self.seed, Sequence) and not isinstance(self.seed, (str, bytes)):
                return [int(value) + 1 for value in self.seed]
            return int(self.seed) + 1
        return 1

    def _effective_x_cycles(self) -> float | Sequence[float] | npt.NDArray[np.float64]:
        if self.cycles is not None:
            return self.cycles
        if self.upper_cycles is not None:
            return self.upper_cycles
        if self.lower_cycles is not None:
            return self.lower_cycles
        return 10.0

    def _effective_x_seed(self) -> int | Sequence[int] | npt.NDArray[np.int64]:
        if self.seed is not None:
            return self.seed
        if self.upper_seed is not None:
            return self.upper_seed
        if self.lower_seed is not None:
            return self.lower_seed
        return 0


# Short public aliases.
Pause = PauseTransition
Parallel = ParallelTransition
Draw = DrawTransition
DrawText = DrawTextTransition
FillBetween = FillBetweenTransition
DrawScatter = DrawScatterTransition
Erase = EraseTransition
EraseScatter = EraseScatterTransition
EraseFillBetween = EraseFillBetweenTransition
EraseText = EraseTextTransition
Move = MoveTransition
MoveScatter = MoveScatterTransition
MoveFillBetween = MoveFillBetweenTransition
MoveText = MoveTextTransition
CurveStyle = CurveStyleTransition
FillStyle = FillStyleTransition
TextStyle = TextStyleTransition
Stress = StressTransition
Jitter = JitterTransition
JitterFillBetween = JitterFillBetweenTransition
