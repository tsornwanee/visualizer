from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text as MplText

from .scene import Curve, FillBetweenArea, Scatter, Scene, Text
from .transitions import FrameState, PauseTransition, Transition


def plot_scene(
    scene: Scene,
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    if ax is None and fig is None:
        fig, ax = plt.subplots()
    elif ax is None and fig is not None:
        ax = fig.gca()
    elif ax is not None and fig is None:
        fig = ax.figure

    assert fig is not None
    assert ax is not None

    resolved_xlim = xlim or _infer_axis_limits_for_templates(
        scene.curves,
        scene.scatters,
        scene.fills,
        scene.texts,
        axis="x",
    )
    resolved_ylim = ylim or _infer_axis_limits_for_templates(
        scene.curves,
        scene.scatters,
        scene.fills,
        scene.texts,
        axis="y",
    )

    ax.set_xlim(*resolved_xlim)
    ax.set_ylim(*resolved_ylim)
    if title is not None:
        ax.set_title(title)

    for fill in scene.fills.values():
        if fill.is_empty:
            continue
        for x_values, y1_values, y2_values, where, fill_kwargs in fill.fill_segments():
            ax.fill_between(
                x_values,
                y1_values,
                y2_values,
                where=where,
                interpolate=True,
                **fill_kwargs,
            )

    for scatter in scene.scatters.values():
        if scatter.is_empty:
            continue
        x_values, y_values, sizes = scatter.clipped_scatter_data()
        if x_values.size == 0:
            continue
        ax.scatter(
            x_values,
            y_values,
            **scatter.mpl_scatter_kwargs(size=sizes),
        )

    for curve in scene.curves.values():
        if curve.is_empty:
            continue
        clipped_x, clipped_y = curve.clipped_line_data()
        if clipped_x.size == 0:
            continue
        ax.plot(clipped_x, clipped_y, **curve.mpl_line_kwargs())

    for text in scene.texts.values():
        if not text.is_visible():
            continue
        ax.text(
            text.x,
            text.y,
            text.content,
            **text.mpl_text_kwargs(),
        )

    return fig, ax


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
        self.entries.append(ScheduledTransition(transition=PauseTransition(), duration=duration))
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

    def plot_scene(
        self,
        elapsed_seconds: float | None = None,
        *,
        scene: Scene | None = None,
        fig: Figure | None = None,
        ax: Axes | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        title: str | None = None,
    ) -> tuple[Figure, Axes]:
        if scene is not None and elapsed_seconds is not None:
            raise ValueError("Pass either scene or elapsed_seconds, not both.")

        resolved_scene = scene
        if resolved_scene is None:
            resolved_scene = (
                self.final_scene if elapsed_seconds is None else self.scene_at(elapsed_seconds)
            )

        return plot_scene(
            resolved_scene,
            fig=fig,
            ax=ax,
            xlim=xlim,
            ylim=ylim,
            title=title,
        )

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
        scatter_templates = self._collect_scatter_templates(prepared, final_scene)
        fill_templates = self._collect_fill_templates(prepared, final_scene)
        text_templates = self._collect_text_templates(prepared, final_scene)

        if ax is None and fig is None:
            fig, ax = plt.subplots()
        elif ax is None and fig is not None:
            ax = fig.gca()
        elif ax is not None and fig is None:
            fig = ax.figure

        assert fig is not None
        assert ax is not None

        resolved_xlim = xlim or self._infer_axis_limits(
            curve_templates,
            scatter_templates,
            fill_templates,
            text_templates,
            axis="x",
        )
        resolved_ylim = ylim or self._infer_axis_limits(
            curve_templates,
            scatter_templates,
            fill_templates,
            text_templates,
            axis="y",
        )

        ax.set_xlim(*resolved_xlim)
        ax.set_ylim(*resolved_ylim)
        if title is not None:
            ax.set_title(title)

        lines: dict[str, Line2D] = {}
        for curve_id, curve in curve_templates.items():
            (line,) = ax.plot([], [], **curve.mpl_line_kwargs())
            line.set_label(curve_id)
            line.set_visible(False)
            lines[curve_id] = line

        scatters: dict[str, PathCollection | None] = {
            scatter_id: None for scatter_id in scatter_templates
        }
        fills: dict[str, list[PolyCollection]] = {fill_id: [] for fill_id in fill_templates}
        texts: dict[str, MplText] = {}
        for text_id in text_templates:
            artist = ax.text(0.0, 0.0, "")
            artist.set_visible(False)
            texts[text_id] = artist
        pointer_artists: list[Line2D] = []
        glow_artists: list[Line2D] = []

        def render_frame(frame_state: FrameState) -> list[Artist]:
            artists: list[Artist] = []

            for curve_id, line in lines.items():
                if frame_state.scene.contains_curve(curve_id):
                    curve = frame_state.scene.get_curve(curve_id)
                    if curve.is_empty:
                        line.set_data([], [])
                        line.set_visible(False)
                    else:
                        clipped_x, clipped_y = curve.clipped_line_data()
                        if clipped_x.size == 0:
                            line.set_data([], [])
                            line.set_visible(False)
                        else:
                            line.set_data(clipped_x, clipped_y)
                            line.set(**curve.mpl_line_kwargs())
                            line.set_visible(True)
                else:
                    line.set_data([], [])
                    line.set_visible(False)
                artists.append(line)

            for fill_id, collection in fills.items():
                for artist in collection:
                    artist.remove()
                fills[fill_id] = []

                if frame_state.scene.contains_fill(fill_id):
                    fill = frame_state.scene.get_fill(fill_id)
                    if not fill.is_empty:
                        for x_values, y1_values, y2_values, where, fill_kwargs in fill.fill_segments():
                            collection = ax.fill_between(
                                x_values,
                                y1_values,
                                y2_values,
                                where=where,
                                interpolate=True,
                                **fill_kwargs,
                            )
                            fills[fill_id].append(collection)
                            artists.append(collection)

            for scatter_id, collection in scatters.items():
                if collection is not None:
                    collection.remove()
                    scatters[scatter_id] = None

                if frame_state.scene.contains_scatter(scatter_id):
                    scatter = frame_state.scene.get_scatter(scatter_id)
                    if scatter.is_empty:
                        continue
                    x_values, y_values, sizes = scatter.clipped_scatter_data()
                    if x_values.size == 0:
                        continue
                    scatters[scatter_id] = ax.scatter(
                        x_values,
                        y_values,
                        **scatter.mpl_scatter_kwargs(size=sizes),
                    )
                    artists.append(scatters[scatter_id])

            for text_id, artist in texts.items():
                if frame_state.scene.contains_text(text_id):
                    text = frame_state.scene.get_text(text_id)
                    if text.is_visible():
                        artist.set_position((text.x, text.y))
                        artist.set_text(text.content)
                        artist.set(**text.mpl_text_kwargs())
                        artist.set_visible(True)
                    else:
                        artist.set_visible(False)
                else:
                    artist.set_visible(False)
                artists.append(artist)

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

    def _collect_scatter_templates(
        self,
        prepared: list[_PreparedTransition],
        final_scene: Scene,
    ) -> dict[str, Scatter]:
        templates: dict[str, Scatter] = dict(self.initial_scene.scatters)

        for item in prepared:
            templates.update(item.start_scene.scatters)
            templates.update(item.end_scene.scatters)

        templates.update(final_scene.scatters)
        return templates

    def _collect_text_templates(
        self,
        prepared: list[_PreparedTransition],
        final_scene: Scene,
    ) -> dict[str, Text]:
        templates: dict[str, Text] = dict(self.initial_scene.texts)

        for item in prepared:
            templates.update(item.start_scene.texts)
            templates.update(item.end_scene.texts)

        templates.update(final_scene.texts)
        return templates

    def _infer_axis_limits(
        self,
        curve_templates: dict[str, Curve],
        scatter_templates: dict[str, Scatter],
        fill_templates: dict[str, FillBetweenArea],
        text_templates: dict[str, Text],
        *,
        axis: str,
    ) -> tuple[float, float]:
        return _infer_axis_limits_for_templates(
            curve_templates,
            scatter_templates,
            fill_templates,
            text_templates,
            axis=axis,
        )


def _infer_axis_limits_for_templates(
    curve_templates: Mapping[str, Curve],
    scatter_templates: Mapping[str, Scatter],
    fill_templates: Mapping[str, FillBetweenArea],
    text_templates: Mapping[str, Text],
    *,
    axis: str,
) -> tuple[float, float]:
    values: list[np.ndarray] = []

    if axis == "x":
        for curve in curve_templates.values():
            extents = curve.visible_extents()
            if extents is not None:
                values.append(np.asarray([extents[0], extents[1]], dtype=float))
        for scatter in scatter_templates.values():
            extents = scatter.visible_extents()
            if extents is not None:
                values.append(np.asarray([extents[0], extents[1]], dtype=float))
        for fill in fill_templates.values():
            extents = fill.visible_extents()
            if extents is not None:
                values.append(np.asarray([extents[0], extents[1]], dtype=float))
        for text in text_templates.values():
            extents = text.visible_extents()
            if extents is not None:
                values.append(np.asarray([extents[0], extents[1]], dtype=float))
    elif axis == "y":
        for curve in curve_templates.values():
            extents = curve.visible_extents()
            if extents is not None:
                values.append(np.asarray([extents[2], extents[3]], dtype=float))
        for scatter in scatter_templates.values():
            extents = scatter.visible_extents()
            if extents is not None:
                values.append(np.asarray([extents[2], extents[3]], dtype=float))
        for fill in fill_templates.values():
            extents = fill.visible_extents()
            if extents is not None:
                values.append(np.asarray([extents[2], extents[3]], dtype=float))
        for text in text_templates.values():
            extents = text.visible_extents()
            if extents is not None:
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


def _scenes_equal(left: Scene, right: Scene) -> bool:
    return (
        _curve_mapping_equal(left.curves, right.curves)
        and _scatter_mapping_equal(left.scatters, right.scatters)
        and _fill_mapping_equal(left.fills, right.fills)
        and _text_mapping_equal(left.texts, right.texts)
    )


def _curve_mapping_equal(left: Mapping[str, Curve], right: Mapping[str, Curve]) -> bool:
    if left.keys() != right.keys():
        return False

    return all(_curves_equal(left[curve_id], right[curve_id]) for curve_id in left)


def _fill_mapping_equal(left: Mapping[str, FillBetweenArea], right: Mapping[str, FillBetweenArea]) -> bool:
    if left.keys() != right.keys():
        return False

    return all(_fills_equal(left[fill_id], right[fill_id]) for fill_id in left)


def _scatter_mapping_equal(left: Mapping[str, Scatter], right: Mapping[str, Scatter]) -> bool:
    if left.keys() != right.keys():
        return False

    return all(_scatters_equal(left[scatter_id], right[scatter_id]) for scatter_id in left)


def _text_mapping_equal(left: Mapping[str, Text], right: Mapping[str, Text]) -> bool:
    if left.keys() != right.keys():
        return False

    return all(_texts_equal(left[text_id], right[text_id]) for text_id in left)


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
        and _values_equal(left.line_kwargs, right.line_kwargs)
    )


def _fills_equal(left: FillBetweenArea, right: FillBetweenArea) -> bool:
    return (
        left.fill_id == right.fill_id
        and np.array_equal(left.x, right.x)
        and np.array_equal(left.y1, right.y1)
        and np.array_equal(left.y2, right.y2)
        and left.color == right.color
        and left.positive_color == right.positive_color
        and left.negative_color == right.negative_color
        and left.alpha == right.alpha
        and left.linestyle == right.linestyle
        and left.linewidth == right.linewidth
        and _values_equal(left.domain, right.domain)
        and _values_equal(left.value_range, right.value_range)
        and _values_equal(left.fill_kwargs, right.fill_kwargs)
    )


def _scatters_equal(left: Scatter, right: Scatter) -> bool:
    return (
        left.scatter_id == right.scatter_id
        and np.array_equal(left.x, right.x)
        and np.array_equal(left.y, right.y)
        and np.array_equal(left.size, right.size)
        and _values_equal(left.color, right.color)
        and left.alpha == right.alpha
        and _values_equal(left.marker, right.marker)
        and left.linewidth == right.linewidth
        and _values_equal(left.edgecolor, right.edgecolor)
        and _values_equal(left.domain, right.domain)
        and _values_equal(left.value_range, right.value_range)
        and _values_equal(left.scatter_kwargs, right.scatter_kwargs)
    )


def _texts_equal(left: Text, right: Text) -> bool:
    return (
        left.text_id == right.text_id
        and left.x == right.x
        and left.y == right.y
        and left.content == right.content
        and left.color == right.color
        and left.alpha == right.alpha
        and left.fontsize == right.fontsize
        and left.ha == right.ha
        and left.va == right.va
        and left.rotation == right.rotation
        and _values_equal(left.domain, right.domain)
        and _values_equal(left.value_range, right.value_range)
        and _values_equal(left.text_kwargs, right.text_kwargs)
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
