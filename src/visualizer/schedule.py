from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import FillBetweenPolyCollection, PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text as MplText

from .scene import Curve, FillBetweenArea, Scatter, Scene, Text
from .transitions import FrameState, PauseTransition, Transition


_EMPTY_FLOAT_ARRAY = np.asarray([], dtype=float)
_EMPTY_BOOL_ARRAY = np.asarray([], dtype=bool)
_EMPTY_OFFSETS = np.empty((0, 2), dtype=float)


@dataclass
class _ScatterArtistState:
    collection: PathCollection
    marker: object


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
        ax.set_autoscale_on(False)

        lines: dict[str, Line2D] = {}
        for curve_id, curve in curve_templates.items():
            (line,) = ax.plot([], [], **curve.mpl_line_kwargs())
            line.set_label(curve_id)
            line.set_animated(blit)
            line.set_visible(False)
            lines[curve_id] = line

        scatters: dict[str, _ScatterArtistState] = {
            scatter_id: _create_scatter_artist_state(ax, scatter, animated=blit)
            for scatter_id, scatter in scatter_templates.items()
        }
        fills: dict[str, dict[str, FillBetweenPolyCollection]] = {
            fill_id: {
                variant: _create_fill_artist(ax, animated=blit)
                for variant in ("base", "positive", "negative")
            }
            for fill_id in fill_templates
        }
        texts: dict[str, MplText] = {}
        for text_id in text_templates:
            artist = ax.text(0.0, 0.0, "")
            artist.set_animated(blit)
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

            for fill_id, variant_artists in fills.items():
                if frame_state.scene.contains_fill(fill_id):
                    fill = frame_state.scene.get_fill(fill_id)
                    payloads = _fill_artist_payloads(fill)
                else:
                    payloads = {}

                for variant, artist in variant_artists.items():
                    _update_fill_artist(artist, payloads.get(variant))
                    artists.append(artist)

            for scatter_id, state in scatters.items():
                if frame_state.scene.contains_scatter(scatter_id):
                    scatter = frame_state.scene.get_scatter(scatter_id)
                    scatters[scatter_id] = _update_scatter_artist(
                        ax,
                        state,
                        scatter,
                        animated=blit,
                    )
                else:
                    _hide_scatter_artist(state.collection)
                artists.append(scatters[scatter_id].collection)

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

            _ensure_line_artist_count(
                ax,
                pointer_artists,
                len(frame_state.pointers),
                animated=blit,
            )
            for index, pointer in enumerate(frame_state.pointers):
                pointer_kwargs = {
                    "marker": "o",
                    "linestyle": "None",
                }
                pointer_kwargs.update(pointer.artist_kwargs)
                artist = pointer_artists[index]
                artist.set_data([pointer.x], [pointer.y])
                artist.set(**pointer_kwargs)
                artist.set_visible(True)
                artists.append(artist)
            for artist in pointer_artists[len(frame_state.pointers) :]:
                artist.set_data([], [])
                artist.set_visible(False)
                artists.append(artist)

            _ensure_line_artist_count(
                ax,
                glow_artists,
                len(frame_state.glows),
                animated=blit,
            )
            for index, glow in enumerate(frame_state.glows):
                artist = glow_artists[index]
                artist.set_data(glow.x, glow.y)
                artist.set(**glow.artist_kwargs)
                artist.set_visible(True)
                artists.append(artist)
            for artist in glow_artists[len(frame_state.glows) :]:
                artist.set_data([], [])
                artist.set_visible(False)
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
            cache_frame_data=False,
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


def _create_fill_artist(
    ax: Axes,
    *,
    animated: bool,
) -> FillBetweenPolyCollection:
    artist = FillBetweenPolyCollection(
        "x",
        _EMPTY_FLOAT_ARRAY,
        _EMPTY_FLOAT_ARRAY,
        _EMPTY_FLOAT_ARRAY,
        where=_EMPTY_BOOL_ARRAY,
        interpolate=True,
    )
    artist.set_animated(animated)
    artist.set_visible(False)
    ax.add_collection(artist)
    return artist


def _fill_artist_payloads(
    fill: FillBetweenArea,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]]:
    x_values, y1_values, y2_values, where = fill.clipped_fill_data()
    if not np.any(where):
        return {}

    base_kwargs = fill.mpl_fill_kwargs()
    if fill.positive_color is None and fill.negative_color is None:
        return {
            "base": (
                x_values,
                y1_values,
                y2_values,
                where,
                base_kwargs,
            )
        }

    payloads: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]] = {}
    positive_where = where & (fill.y1 >= fill.y2)
    negative_where = where & (fill.y1 < fill.y2)

    if np.any(positive_where):
        positive_kwargs = dict(base_kwargs)
        if fill.positive_color is not None:
            positive_kwargs["color"] = fill.positive_color
        payloads["positive"] = (
            x_values,
            y1_values,
            y2_values,
            positive_where,
            positive_kwargs,
        )

    if np.any(negative_where):
        negative_kwargs = dict(base_kwargs)
        if fill.negative_color is not None:
            negative_kwargs["color"] = fill.negative_color
        payloads["negative"] = (
            x_values,
            y1_values,
            y2_values,
            negative_where,
            negative_kwargs,
        )

    return payloads


def _apply_fill_style(
    artist: FillBetweenPolyCollection,
    fill_kwargs: Mapping[str, object],
) -> None:
    kwargs = dict(fill_kwargs)

    if "alpha" in kwargs:
        artist.set_alpha(kwargs.pop("alpha"))

    if "color" in kwargs:
        artist.set_color(kwargs.pop("color"))

    if "facecolor" in kwargs:
        artist.set_facecolor(kwargs.pop("facecolor"))
    if "edgecolor" in kwargs:
        artist.set_edgecolor(kwargs.pop("edgecolor"))
    if "linewidth" in kwargs:
        artist.set_linewidth(kwargs.pop("linewidth"))
    if "linewidths" in kwargs:
        artist.set_linewidths(kwargs.pop("linewidths"))
    if "linestyle" in kwargs:
        artist.set_linestyle(kwargs.pop("linestyle"))

    kwargs.pop("interpolate", None)

    if kwargs:
        artist.set(**kwargs)


def _update_fill_artist(
    artist: FillBetweenPolyCollection,
    payload: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]] | None,
) -> None:
    if payload is None:
        artist.set_data(
            _EMPTY_FLOAT_ARRAY,
            _EMPTY_FLOAT_ARRAY,
            _EMPTY_FLOAT_ARRAY,
            where=_EMPTY_BOOL_ARRAY,
        )
        artist.set_visible(False)
        return

    x_values, y1_values, y2_values, where, fill_kwargs = payload
    artist.set_data(x_values, y1_values, y2_values, where=where)
    _apply_fill_style(artist, fill_kwargs)
    artist.set_visible(np.any(where))


def _create_scatter_artist_state(
    ax: Axes,
    scatter: Scatter,
    *,
    animated: bool,
) -> _ScatterArtistState:
    marker = scatter.marker
    collection = ax.scatter(_EMPTY_FLOAT_ARRAY, _EMPTY_FLOAT_ARRAY, marker=marker)
    collection.set_animated(animated)
    collection.set_visible(False)
    return _ScatterArtistState(collection=collection, marker=marker)


def _apply_scatter_style(
    collection: PathCollection,
    scatter_kwargs: Mapping[str, object],
) -> None:
    kwargs = dict(scatter_kwargs)

    if "alpha" in kwargs:
        collection.set_alpha(kwargs.pop("alpha"))

    if "color" in kwargs:
        collection.set_color(kwargs.pop("color"))
    elif "c" in kwargs:
        collection.set_color(kwargs.pop("c"))

    if "facecolor" in kwargs:
        collection.set_facecolor(kwargs.pop("facecolor"))
    if "facecolors" in kwargs:
        collection.set_facecolors(kwargs.pop("facecolors"))
    if "edgecolor" in kwargs:
        collection.set_edgecolor(kwargs.pop("edgecolor"))
    if "edgecolors" in kwargs:
        collection.set_edgecolors(kwargs.pop("edgecolors"))
    if "linewidth" in kwargs:
        collection.set_linewidth(kwargs.pop("linewidth"))
    if "linewidths" in kwargs:
        collection.set_linewidths(kwargs.pop("linewidths"))

    if kwargs:
        collection.set(**kwargs)


def _update_scatter_artist(
    ax: Axes,
    state: _ScatterArtistState,
    scatter: Scatter,
    *,
    animated: bool,
) -> _ScatterArtistState:
    x_values, y_values, sizes = scatter.clipped_scatter_data()
    if x_values.size == 0:
        _hide_scatter_artist(state.collection)
        return state

    scatter_kwargs = scatter.mpl_scatter_kwargs(size=sizes)
    marker = scatter_kwargs.pop("marker", scatter.marker)
    size_values = np.asarray(scatter_kwargs.pop("s", sizes), dtype=float)

    if not _values_equal(state.marker, marker):
        state.collection.remove()
        replacement_kwargs = dict(scatter_kwargs)
        replacement_kwargs["s"] = size_values
        replacement = ax.scatter(
            x_values,
            y_values,
            marker=marker,
            **replacement_kwargs,
        )
        replacement.set_animated(animated)
        replacement.set_visible(True)
        return _ScatterArtistState(collection=replacement, marker=marker)

    state.collection.set_offsets(np.column_stack((x_values, y_values)))
    state.collection.set_sizes(size_values)
    _apply_scatter_style(state.collection, scatter_kwargs)
    state.collection.set_visible(True)
    return state


def _hide_scatter_artist(collection: PathCollection) -> None:
    collection.set_offsets(_EMPTY_OFFSETS)
    collection.set_sizes(_EMPTY_FLOAT_ARRAY)
    collection.set_visible(False)


def _ensure_line_artist_count(
    ax: Axes,
    artists: list[Line2D],
    count: int,
    *,
    animated: bool,
) -> None:
    while len(artists) < count:
        (artist,) = ax.plot([], [])
        artist.set_animated(animated)
        artist.set_visible(False)
        artists.append(artist)


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
