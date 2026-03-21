from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .scene import Curve, FillBetweenArea, Scene, Theater
from .transitions import FrameState, Pause, Transition


@dataclass(frozen=True)
class ScheduledTransition:
    transition: Transition
    duration: float

    def __post_init__(self) -> None:
        if self.duration < 0:
            raise ValueError("Transition duration must be non-negative.")


@dataclass(frozen=True)
class _PreparedTransition:
    scheduled: ScheduledTransition
    start_scene: Scene
    end_scene: Scene
    start_time: float
    end_time: float


@dataclass
class Schedule:
    initial_scene: Scene = field(default_factory=Scene)
    entries: list[ScheduledTransition] = field(default_factory=list)

    def add(self, transition: Transition, duration: float) -> Schedule:
        self.entries.append(ScheduledTransition(transition=transition, duration=duration))
        return self

    def add_break(self, duration: float) -> Schedule:
        self.entries.append(ScheduledTransition(transition=Pause(), duration=duration))
        return self

    def pause(self, duration: float) -> Schedule:
        return self.add_break(duration)

    @property
    def total_duration(self) -> float:
        return sum(entry.duration for entry in self.entries)

    @property
    def final_scene(self) -> Scene:
        current_scene = self.initial_scene

        for entry in self.entries:
            current_scene = entry.transition.apply(current_scene)

        return current_scene

    def next_act(self) -> Schedule:
        return Schedule(initial_scene=self.final_scene)

    def extend_schedule(
        self,
        other: Schedule,
        *,
        validate_initial_scene: bool = False,
    ) -> Schedule:
        if not isinstance(other, Schedule):
            raise TypeError("other must be a Schedule.")

        if validate_initial_scene and not _scenes_equal(self.final_scene, other.initial_scene):
            raise ValueError(
                "The appended schedule's initial_scene does not match the current final_scene."
            )

        self.entries.extend(other.entries)
        return self

    def appended(
        self,
        other: Schedule,
        *,
        validate_initial_scene: bool = False,
    ) -> Schedule:
        combined = Schedule(
            initial_scene=self.initial_scene,
            entries=list(self.entries),
        )
        return combined.extend_schedule(
            other,
            validate_initial_scene=validate_initial_scene,
        )

    @classmethod
    def combine(
        cls,
        schedules: Iterable[Schedule],
        *,
        validate_initial_scene: bool = False,
    ) -> Schedule:
        schedule_list = list(schedules)
        if not schedule_list:
            return cls()

        combined = cls(
            initial_scene=schedule_list[0].initial_scene,
            entries=list(schedule_list[0].entries),
        )

        for schedule in schedule_list[1:]:
            combined.extend_schedule(
                schedule,
                validate_initial_scene=validate_initial_scene,
            )

        return combined

    def scenes(self) -> list[Scene]:
        scenes = [self.initial_scene]
        current_scene = self.initial_scene

        for entry in self.entries:
            current_scene = entry.transition.apply(current_scene)
            scenes.append(current_scene)

        return scenes

    def scene_at(self, elapsed_seconds: float) -> Scene:
        prepared = self._prepare()
        return self._frame_state_from_prepared(prepared, elapsed_seconds).scene

    def build_animation(
        self,
        *,
        fig: Figure | None = None,
        ax: Axes | None = None,
        fps: int = 30,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        title: str | None = None,
        blit: bool = False,
        repeat: bool = False,
    ) -> FuncAnimation:
        if fps <= 0:
            raise ValueError("fps must be positive.")

        prepared = self._prepare()
        final_scene = prepared[-1].end_scene if prepared else self.initial_scene
        curve_templates = self._collect_curve_templates(prepared, final_scene)
        fill_templates = self._collect_fill_templates(prepared, final_scene)
        theater_templates = self._collect_theater_templates(prepared, final_scene)

        if ax is None and fig is None:
            fig, ax = plt.subplots()
        elif ax is None and fig is not None:
            ax = fig.gca()
        elif ax is not None and fig is None:
            fig = ax.figure

        assert fig is not None
        assert ax is not None

        resolved_xlim = xlim or self._infer_axis_limits(prepared, final_scene, axis="x")
        resolved_ylim = ylim or self._infer_axis_limits(prepared, final_scene, axis="y")

        ax.set_xlim(*resolved_xlim)
        ax.set_ylim(*resolved_ylim)
        if title is not None:
            ax.set_title(title)

        theater_patches: dict[str, Rectangle] = {}
        for theater_id, theater in theater_templates.items():
            patch = Rectangle(
                (theater.xlim[0], theater.ylim[0]),
                theater.width,
                theater.height,
                visible=False,
                **theater.mpl_patch_kwargs(),
            )
            patch.set_label(theater_id)
            ax.add_patch(patch)
            theater_patches[theater_id] = patch

        lines: dict[str, Line2D] = {}
        for curve_id, curve in curve_templates.items():
            (line,) = ax.plot([], [], **curve.mpl_line_kwargs())
            line.set_label(curve_id)
            line.set_visible(False)
            lines[curve_id] = line

        fills: dict[str, PolyCollection | None] = {fill_id: None for fill_id in fill_templates}
        pointer_artists: list[Line2D] = []
        glow_artists: list[Line2D] = []

        def render_frame(frame_state: FrameState) -> list[Artist]:
            artists: list[Artist] = []

            for theater_id, patch in theater_patches.items():
                if frame_state.scene.contains_theater(theater_id):
                    theater = frame_state.scene.get_theater(theater_id)
                    patch.set_xy((theater.xlim[0], theater.ylim[0]))
                    patch.set_width(theater.width)
                    patch.set_height(theater.height)
                    patch.set(**theater.mpl_patch_kwargs())
                    patch.set_visible(True)
                else:
                    patch.set_visible(False)
                artists.append(patch)

            for curve_id, line in lines.items():
                if frame_state.scene.contains_curve(curve_id):
                    curve = frame_state.scene.get_curve(curve_id)
                    theater = (
                        frame_state.scene.get_theater(curve.theater_id)
                        if curve.theater_id is not None
                        else None
                    )
                    if curve.is_empty:
                        line.set_data([], [])
                        line.set_visible(False)
                        line.set_clip_path(None)
                    else:
                        clipped_x, clipped_y = curve.clipped_line_data(theater)
                        if clipped_x.size == 0:
                            line.set_data([], [])
                            line.set_visible(False)
                            line.set_clip_path(None)
                        else:
                            line.set_data(clipped_x, clipped_y)
                            line.set(**curve.mpl_line_kwargs())
                            line.set_clip_path(
                                theater_patches[curve.theater_id]
                                if curve.theater_id is not None
                                else None
                            )
                            line.set_visible(True)
                else:
                    line.set_data([], [])
                    line.set_visible(False)
                    line.set_clip_path(None)
                artists.append(line)

            for fill_id, collection in fills.items():
                if collection is not None:
                    collection.remove()
                    fills[fill_id] = None

                if frame_state.scene.contains_fill(fill_id):
                    fill = frame_state.scene.get_fill(fill_id)
                    theater = (
                        frame_state.scene.get_theater(fill.theater_id)
                        if fill.theater_id is not None
                        else None
                    )
                    if not fill.is_empty:
                        x_values, y1_values, y2_values, where = fill.clipped_fill_data(theater)
                        if not np.any(where):
                            continue
                        fills[fill_id] = ax.fill_between(
                            x_values,
                            y1_values,
                            y2_values,
                            where=where,
                            interpolate=True,
                            **fill.mpl_fill_kwargs(),
                        )
                        if fill.theater_id is not None:
                            fills[fill_id].set_clip_path(theater_patches[fill.theater_id])
                        artists.append(fills[fill_id])

            while pointer_artists:
                pointer_artists.pop().remove()
            for pointer in frame_state.pointers:
                pointer_kwargs = {
                    "marker": "o",
                    "linestyle": "None",
                }
                pointer_kwargs.update(pointer.artist_kwargs)
                (artist,) = ax.plot(
                    [pointer.x],
                    [pointer.y],
                    **pointer_kwargs,
                )
                pointer_artists.append(artist)
                artists.append(artist)

            while glow_artists:
                glow_artists.pop().remove()
            for glow in frame_state.glows:
                (artist,) = ax.plot(glow.x, glow.y, **glow.artist_kwargs)
                glow_artists.append(artist)
                artists.append(artist)

            return artists

        def init() -> list[Artist]:
            return render_frame(self._frame_state_from_prepared(prepared, 0.0))

        if self.total_duration == 0:
            frame_times = np.array([0.0])
        else:
            frame_count = max(int(np.ceil(self.total_duration * fps)), 1) + 1
            frame_times = np.linspace(0.0, self.total_duration, frame_count)

        def update(elapsed_seconds: float) -> list[Artist]:
            frame_state = self._frame_state_from_prepared(prepared, float(elapsed_seconds))
            return render_frame(frame_state)

        return FuncAnimation(
            fig=fig,
            func=update,
            frames=frame_times,
            init_func=init,
            interval=1000 / fps,
            blit=blit,
            repeat=repeat,
        )

    def _prepare(self) -> list[_PreparedTransition]:
        prepared: list[_PreparedTransition] = []
        current_scene = self.initial_scene
        current_time = 0.0

        for entry in self.entries:
            end_scene = entry.transition.apply(current_scene)
            prepared.append(
                _PreparedTransition(
                    scheduled=entry,
                    start_scene=current_scene,
                    end_scene=end_scene,
                    start_time=current_time,
                    end_time=current_time + entry.duration,
                )
            )
            current_scene = end_scene
            current_time += entry.duration

        return prepared

    def _frame_state_from_prepared(
        self,
        prepared: list[_PreparedTransition],
        elapsed_seconds: float,
    ) -> FrameState:
        if not prepared:
            return FrameState(scene=self.initial_scene)

        elapsed = max(float(elapsed_seconds), 0.0)
        current_scene = self.initial_scene

        for item in prepared:
            duration = item.scheduled.duration

            if duration == 0:
                if elapsed < item.start_time:
                    return FrameState(scene=current_scene)
                current_scene = item.end_scene
                continue

            if elapsed <= item.start_time:
                return FrameState(scene=current_scene)

            if elapsed < item.end_time:
                progress = (elapsed - item.start_time) / duration
                return item.scheduled.transition.frame_state(item.start_scene, progress)

            current_scene = item.end_scene

        return FrameState(scene=current_scene)

    def _collect_curve_templates(
        self,
        prepared: list[_PreparedTransition],
        final_scene: Scene,
    ) -> dict[str, Curve]:
        templates: dict[str, Curve] = dict(self.initial_scene.curves)

        for item in prepared:
            templates.update(item.start_scene.curves)
            templates.update(item.end_scene.curves)

        templates.update(final_scene.curves)
        return templates

    def _collect_fill_templates(
        self,
        prepared: list[_PreparedTransition],
        final_scene: Scene,
    ) -> dict[str, FillBetweenArea]:
        templates: dict[str, FillBetweenArea] = dict(self.initial_scene.fills)

        for item in prepared:
            templates.update(item.start_scene.fills)
            templates.update(item.end_scene.fills)

        templates.update(final_scene.fills)
        return templates

    def _collect_theater_templates(
        self,
        prepared: list[_PreparedTransition],
        final_scene: Scene,
    ) -> dict[str, Theater]:
        templates: dict[str, Theater] = dict(self.initial_scene.theaters)

        for item in prepared:
            templates.update(item.start_scene.theaters)
            templates.update(item.end_scene.theaters)

        templates.update(final_scene.theaters)
        return templates

    def _infer_axis_limits(
        self,
        prepared: list[_PreparedTransition],
        final_scene: Scene,
        *,
        axis: str,
    ) -> tuple[float, float]:
        values: list[np.ndarray] = []

        for scene in self._scenes_for_limits(prepared, final_scene):
            for theater in scene.theaters.values():
                extents = theater.visible_extents()
                if axis == "x":
                    values.append(np.asarray([extents[0], extents[1]], dtype=float))
                elif axis == "y":
                    values.append(np.asarray([extents[2], extents[3]], dtype=float))
                else:
                    raise ValueError("axis must be 'x' or 'y'.")

            for curve in scene.curves.values():
                theater = scene.theaters.get(curve.theater_id) if curve.theater_id is not None else None
                extents = curve.visible_extents(theater)
                if extents is None:
                    continue
                if axis == "x":
                    values.append(np.asarray([extents[0], extents[1]], dtype=float))
                elif axis == "y":
                    values.append(np.asarray([extents[2], extents[3]], dtype=float))
                else:
                    raise ValueError("axis must be 'x' or 'y'.")

            for fill in scene.fills.values():
                theater = scene.theaters.get(fill.theater_id) if fill.theater_id is not None else None
                extents = fill.visible_extents(theater)
                if extents is None:
                    continue
                if axis == "x":
                    values.append(np.asarray([extents[0], extents[1]], dtype=float))
                elif axis == "y":
                    values.append(np.asarray([extents[2], extents[3]], dtype=float))
                else:
                    raise ValueError("axis must be 'x' or 'y'.")

        if not values:
            return (0.0, 1.0)

        minimum = float(min(np.min(value) for value in values))
        maximum = float(max(np.max(value) for value in values))

        if np.isclose(minimum, maximum):
            padding = max(abs(minimum) * 0.05, 0.5)
            return (minimum - padding, maximum + padding)

        padding = (maximum - minimum) * 0.05
        return (minimum - padding, maximum + padding)

    def _scenes_for_limits(
        self,
        prepared: list[_PreparedTransition],
        final_scene: Scene,
    ) -> list[Scene]:
        scenes = [self.initial_scene]
        for item in prepared:
            scenes.append(item.start_scene)
            scenes.append(item.end_scene)
        scenes.append(final_scene)
        return scenes


def _scenes_equal(left: Scene, right: Scene) -> bool:
    return (
        _curve_mapping_equal(left.curves, right.curves)
        and _fill_mapping_equal(left.fills, right.fills)
        and _theater_mapping_equal(left.theaters, right.theaters)
    )


def _curve_mapping_equal(left: Mapping[str, Curve], right: Mapping[str, Curve]) -> bool:
    if left.keys() != right.keys():
        return False

    return all(_curves_equal(left[curve_id], right[curve_id]) for curve_id in left)


def _fill_mapping_equal(left: Mapping[str, FillBetweenArea], right: Mapping[str, FillBetweenArea]) -> bool:
    if left.keys() != right.keys():
        return False

    return all(_fills_equal(left[fill_id], right[fill_id]) for fill_id in left)


def _theater_mapping_equal(left: Mapping[str, Theater], right: Mapping[str, Theater]) -> bool:
    if left.keys() != right.keys():
        return False

    return all(_theaters_equal(left[theater_id], right[theater_id]) for theater_id in left)


def _curves_equal(left: Curve, right: Curve) -> bool:
    return (
        left.curve_id == right.curve_id
        and np.array_equal(left.x, right.x)
        and np.array_equal(left.y, right.y)
        and left.color == right.color
        and left.alpha == right.alpha
        and left.linestyle == right.linestyle
        and left.linewidth == right.linewidth
        and _values_equal(left.domain, right.domain)
        and _values_equal(left.value_range, right.value_range)
        and left.theater_id == right.theater_id
        and _values_equal(left.line_kwargs, right.line_kwargs)
    )


def _fills_equal(left: FillBetweenArea, right: FillBetweenArea) -> bool:
    return (
        left.fill_id == right.fill_id
        and np.array_equal(left.x, right.x)
        and np.array_equal(left.y1, right.y1)
        and np.array_equal(left.y2, right.y2)
        and left.color == right.color
        and left.alpha == right.alpha
        and left.linestyle == right.linestyle
        and left.linewidth == right.linewidth
        and _values_equal(left.domain, right.domain)
        and _values_equal(left.value_range, right.value_range)
        and left.theater_id == right.theater_id
        and _values_equal(left.fill_kwargs, right.fill_kwargs)
    )


def _theaters_equal(left: Theater, right: Theater) -> bool:
    return (
        left.theater_id == right.theater_id
        and _values_equal(left.xlim, right.xlim)
        and _values_equal(left.ylim, right.ylim)
        and _values_equal(left.local_xlim, right.local_xlim)
        and _values_equal(left.local_ylim, right.local_ylim)
        and left.facecolor == right.facecolor
        and left.edgecolor == right.edgecolor
        and left.alpha == right.alpha
        and left.linestyle == right.linestyle
        and left.linewidth == right.linewidth
        and _values_equal(left.patch_kwargs, right.patch_kwargs)
    )


def _values_equal(left: object, right: object) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.array_equal(np.asarray(left), np.asarray(right))
        except Exception:
            return False

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if left.keys() != right.keys():
            return False
        return all(_values_equal(left[key], right[key]) for key in left)

    if isinstance(left, Sequence) and isinstance(right, Sequence):
        if isinstance(left, (str, bytes)) or isinstance(right, (str, bytes)):
            return left == right
        if len(left) != len(right):
            return False
        return all(_values_equal(left_item, right_item) for left_item, right_item in zip(left, right))

    return left == right
