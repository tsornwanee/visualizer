from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .scene import Curve, FillBetweenArea, Scene
from .transitions import FrameState, Transition


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

    @property
    def total_duration(self) -> float:
        return sum(entry.duration for entry in self.entries)

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
        xlim: tuple[float, float] = (0.0, 1.0),
        ylim: tuple[float, float] = (0.0, 1.0),
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

        if ax is None and fig is None:
            fig, ax = plt.subplots()
        elif ax is None and fig is not None:
            ax = fig.gca()
        elif ax is not None and fig is None:
            fig = ax.figure

        assert fig is not None
        assert ax is not None

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        if title is not None:
            ax.set_title(title)

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

            for curve_id, line in lines.items():
                if frame_state.scene.contains_curve(curve_id):
                    curve = frame_state.scene.get_curve(curve_id)
                    if curve.is_empty:
                        line.set_data([], [])
                        line.set_visible(False)
                    else:
                        line.set_data(curve.x, curve.y)
                        line.set(**curve.mpl_line_kwargs())
                        line.set_visible(True)
                else:
                    line.set_data([], [])
                    line.set_visible(False)
                artists.append(line)

            for fill_id, collection in fills.items():
                if collection is not None:
                    collection.remove()
                    fills[fill_id] = None

                if frame_state.scene.contains_fill(fill_id):
                    fill = frame_state.scene.get_fill(fill_id)
                    if not fill.is_empty:
                        fills[fill_id] = ax.fill_between(
                            fill.x,
                            fill.y1,
                            fill.y2,
                            **fill.mpl_fill_kwargs(),
                        )
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
