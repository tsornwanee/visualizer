from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from visualizer import (
    Curve,
    CurveStyle,
    Draw,
    DrawScatter,
    DrawText,
    Erase,
    EraseFillBetween,
    EraseScatter,
    EraseText,
    FillBetween,
    FillBetweenArea,
    Jitter,
    Move,
    MoveFillBetween,
    MoveScatter,
    MoveText,
    Parallel,
    Scatter,
    Schedule,
    Scene,
    Stress,
    Text,
    TextStyle,
)
from visualizer.schedule import ScheduledTransition
from visualizer.transitions import ParallelTransition, PauseTransition, StressTransition


FIGSIZE = (10, 6)
XLIM = (-0.2, 1.2)
YLIM = (-0.25, 1.2)
FPS = 6
VIDEO_DPI = 96
PLOT_RANGE = np.array([-0.08, 1.0], dtype=float)
SAMPLE_BUDGETS = np.linspace(0.04, 0.96, 11)
DEFAULT_SAMPLE_STEP = 12
BANK_LANES = np.array([1.14, 1.10, 1.06], dtype=float)
BANK_GUIDE_Y = 1.10
TIMING_SCALE = 1.20
STRESS_BREATH = 0.55
NARRATION_LAYOUT = {
    "narration1": (-0.16, -0.14),
    "narration2": (-0.06, -0.19),
}


@dataclass
class TieredRemunerationBundle:
    x: np.ndarray
    data: dict[str, np.ndarray | float | int]
    stages: Mapping[str, Mapping[str, float]]
    axes: Mapping[str, Mapping[str, np.ndarray]]
    acts: OrderedDict[str, Schedule]

    @property
    def combined_schedule(self) -> Schedule:
        return Schedule.combine(list(self.acts.values()), validate_initial_scene=True)


@dataclass(frozen=True)
class LevelCrossing:
    x: float
    left_index: int
    right_index: int


def build_tiered_remuneration_bundle(sample_step: int = DEFAULT_SAMPLE_STEP) -> TieredRemunerationBundle:
    x, data = build_data(sample_step=sample_step)
    stages = build_stages()
    axes = build_axes()

    initial_scene = build_initial_scene(stages, axes)
    act_1 = build_act_1(initial_scene, x, data, stages, axes)
    act_2 = build_act_2(act_1.final_scene, x, data, stages, axes)
    act_3 = build_act_3(act_2.final_scene, x, data)
    act_4 = build_act_4(act_3.final_scene, x, data)
    act_5 = build_act_5(act_4.final_scene, x, data)

    acts: OrderedDict[str, Schedule] = OrderedDict(
        [
            ("Act 1", retime_schedule(act_1)),
            ("Act 2", retime_schedule(act_2)),
            ("Act 3", retime_schedule(act_3)),
            ("Act 4", retime_schedule(act_4)),
            ("Act 5", retime_schedule(act_5)),
        ]
    )

    return TieredRemunerationBundle(
        x=x,
        data=data,
        stages=stages,
        axes=axes,
        acts=acts,
    )


def build_data(sample_step: int = 10) -> tuple[np.ndarray, dict[str, np.ndarray | float | int]]:
    if sample_step <= 0:
        raise ValueError("sample_step must be positive.")

    data: dict[str, np.ndarray | float | int] = {}

    np.random.seed(626)
    n_points = 10_000
    epsilon = 1e-2
    uprime_cut = 30.0

    g = np.random.randn(n_points) / np.sqrt(n_points)
    cumulative_g = np.cumsum(g)
    cumulative_penalty = np.cumsum(
        epsilon + np.power(np.abs(cumulative_g), 1.7)
    ) / np.sqrt(n_points)

    x_full = np.linspace(0.0, 1.0, n_points)
    uprime_full = 10.0 - cumulative_penalty + 1.0 / (x_full + 0.02)
    u_full = np.cumsum(uprime_full) / n_points - 2.0
    u_full /= np.max(u_full)
    uprime_full /= uprime_cut

    rho0_base = (2.0 - np.cumsum(cumulative_g) / np.sqrt(n_points))[::-1] / 15.0

    np.random.seed(60)
    g = np.random.randn(n_points) / np.sqrt(n_points)
    cumulative_g = np.cumsum(g)
    monotone_penalty = np.cumsum(
        epsilon + np.power(np.abs(cumulative_g), 1.7)
    ) / np.sqrt(n_points) / np.sqrt(n_points)
    rho1_base = (0.09 - monotone_penalty) * 2.7

    sampled = slice(None, None, sample_step)
    x = x_full[sampled]
    data["u"] = u_full[sampled]
    data["uprime"] = uprime_full[sampled]
    data["rho0"] = rho0_base[sampled]
    data["uprime_plus_rho0"] = uprime_full[sampled] + rho0_base[sampled]
    data["rho1"] = rho1_base[sampled]
    data["uprime_plus_rho1"] = uprime_full[sampled] + rho1_base[sampled]

    data["i"] = 0.20
    data["cb"] = 0.26
    data["cl"] = 0.04
    data["i_plus_cb"] = float(data["i"]) + float(data["cb"])
    data["i_minus_cl"] = float(data["i"]) - float(data["cl"])

    return x, data


def build_stages() -> dict[str, dict[str, float]]:
    return {
        "main": {"startx": 0.0, "lengthx": 1.0, "starty": 0.0, "lengthy": 1.0},
        "up": {"startx": 0.0, "lengthx": 1.0, "starty": 0.52, "lengthy": 0.59},
        "down": {"startx": 0.0, "lengthx": 1.0, "starty": -0.10, "lengthy": 0.49},
    }


def build_axes() -> dict[str, dict[str, np.ndarray]]:
    arrow_move = 0.03
    back_move = 0.01
    xy_scaling = 0.55

    return {
        "axis": {
            "x": np.array([0.0, 0.0, 1.0]),
            "appendx": np.array([0.0, 0.0, 0.05 + xy_scaling * (arrow_move - back_move)]),
            "y": np.array([1.0, 0.0, 0.0]),
            "appendy": np.array([0.05 + arrow_move - back_move, 0.0, 0.0]),
        },
        "yhead": {
            "x": np.array([0.0, 0.0, 0.0]),
            "appendx": 0.001 + np.array([xy_scaling * arrow_move, 0.0, -xy_scaling * arrow_move]),
            "y": np.array([1.0, 1.0, 1.0]),
            "appendy": np.array([0.05, 0.05 + arrow_move, 0.05]),
        },
        "xhead": {
            "x": np.array([1.0, 1.0, 1.0]),
            "appendx": np.array([0.05, 0.05 + xy_scaling * arrow_move, 0.05]),
            "y": np.array([0.0, 0.0, 0.0]),
            "appendy": np.array([arrow_move, 0.0, -arrow_move]),
        },
    }


def scale_axis(axis: Mapping[str, np.ndarray], stage: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    x = axis["x"] * stage["lengthx"] + stage["startx"] + axis["appendx"]
    y = axis["y"] * stage["lengthy"] + stage["starty"] + axis["appendy"]
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def stage_value_range(stage: Mapping[str, float]) -> np.ndarray:
    return PLOT_RANGE * stage["lengthy"] + stage["starty"]


def build_initial_scene(
    stages: Mapping[str, Mapping[str, float]],
    axes: Mapping[str, Mapping[str, np.ndarray]],
) -> Scene:
    scene = Scene()
    for stage_name, stage in stages.items():
        for axis_name, axis in axes.items():
            scene = scene.add_curve(
                Curve(
                    f"{stage_name} {axis_name}",
                    *scale_axis(axis, stage),
                    color="black",
                    linewidth=2.0,
                    alpha=0.0,
                )
            )
    return scene


def build_act_1(
    initial_scene: Scene,
    x: np.ndarray,
    data: Mapping[str, np.ndarray | float | int],
    stages: Mapping[str, Mapping[str, float]],
    axes: Mapping[str, Mapping[str, np.ndarray]],
) -> Schedule:
    schedule = Schedule(initial_scene=initial_scene)

    _add_transitions(schedule, axis_styles(["main"], axes, alpha=1.0), duration=0.25)
    schedule.add(
        Draw(
            Curve(
                "u",
                x,
                _array(data["u"]),
                color="red",
                linewidth=2.0,
                alpha=1.0,
                value_range=PLOT_RANGE,
            )
        ),
        duration=1.0,
    )
    schedule.add(
        DrawText(
            Text(
                "u_label",
                0.16,
                0.42,
                r"$u(b)$",
                color="red",
                alpha=1.0,
                fontsize=14.0,
                ha="left",
            )
        ),
        duration=0.35,
    )
    schedule.add(Stress("u", glow_color="#f87171", glow_width=7.0), duration=0.55)

    replace_narration(
        schedule,
        r"The liquidity value $u(b)$ is taken as given.",
        r"We now move to $u'(b)$ to analyze the first-order condition.",
        draw_duration=1.0,
        hold_duration=1.1,
    )

    up_stage = stages["up"]
    _add_transitions(
        schedule,
        [
            Move(
                "u",
                newy=_array(data["u"]) * up_stage["lengthy"] + up_stage["starty"],
                value_range=stage_value_range(up_stage),
            ),
            MoveText("u_label", newy=0.80),
            *axis_styles(["main"], axes, alpha=0.0),
        ],
        duration=1.0,
    )
    schedule.add_break(0.6)

    _add_transitions(schedule, axis_styles(["up", "down"], axes, alpha=1.0), duration=0.25)

    down_stage = stages["down"]
    schedule.add(
        Draw(
            Curve(
                "uprime",
                x,
                _array(data["uprime"]) * down_stage["lengthy"] + down_stage["starty"],
                color="blue",
                linewidth=2.0,
                alpha=1.0,
                linestyle="solid",
                value_range=stage_value_range(down_stage),
            )
        ),
        duration=1.0,
    )
    schedule.add(
        DrawText(
            Text(
                "uprime_label",
                0.90,
                -0.02,
                r"$u'(b)$",
                color="blue",
                alpha=1.0,
                fontsize=14.0,
                ha="left",
            )
        ),
        duration=0.25,
    )
    schedule.add(Stress("uprime", glow_color="#60a5fa", glow_width=8.0), duration=0.6)

    replace_narration(
        schedule,
        r"The upper panel keeps $u(b)$ as a reference curve.",
        r"The lower panel shows $u'(b)$, where the policy problem is solved.",
        draw_duration=1.0,
        hold_duration=1.0,
    )

    _add_transitions(
        schedule,
        [
            *axis_styles(["up", "down"], axes, alpha=0.0),
            Erase("u"),
            EraseText("u_label"),
        ],
        duration=0.45,
    )
    _add_transitions(
        schedule,
        [
            *axis_styles(["main"], axes, alpha=1.0),
            Move("uprime", newy=_array(data["uprime"]), value_range=PLOT_RANGE),
            MoveText("uprime_label", newy=0.13),
        ],
        duration=1.0,
    )

    clear_narration(schedule, duration=0.2)
    schedule.add_break(1.2)
    return schedule


def build_act_2(
    initial_scene: Scene,
    x: np.ndarray,
    data: dict[str, np.ndarray | float | int],
    stages: Mapping[str, Mapping[str, float]],
    axes: Mapping[str, Mapping[str, np.ndarray]],
) -> Schedule:
    del stages, axes
    schedule = Schedule(initial_scene=initial_scene)

    schedule.add(
        Parallel(
            (
                Draw(Curve("i", x, x * 0.0 + _scalar(data["i"]), color="black", linewidth=2.0, alpha=1.0)),
                DrawText(
                    Text(
                        "i_label",
                        1.02,
                        _scalar(data["i"]) - 0.02,
                        r"$i$",
                        color="black",
                        alpha=1.0,
                        fontsize=14.0,
                        ha="left",
                    )
                ),
            )
        ),
        duration=0.4,
    )
    schedule.add(Stress("i", glow_color="#111827", glow_width=8.0), duration=0.45)

    intersection_x = nearest_crossing(
        x,
        _array(data["uprime"]),
        _scalar(data["i"]),
        target_x=0.50,
    ).x
    intersection_y = _scalar(data["i"])

    replace_narration(
        schedule,
        r"$i$ is the interbank interest rate.",
        r"Without remuneration or frictions, every bank chooses the same budget level.",
        draw_duration=1.0,
        hold_duration=0.9,
    )
    schedule.add(Draw(point_marker("intersection_i", intersection_x, intersection_y, color="orange", marker="d", markersize=10.0)), duration=0.0)
    schedule.add(Stress("intersection_i", glow_color="#fb923c", glow_width=14.0), duration=0.5)
    draw_sample_banks(schedule, SAMPLE_BUDGETS, prefix="act2_bank", duration=0.0, color="#dc2626")
    schedule.add_break(0.5)
    move_sample_banks_to_point(schedule, SAMPLE_BUDGETS, intersection_x, prefix="act2_bank", duration=1.1)
    schedule.add_break(1.0)

    replace_narration(
        schedule,
        r"We now introduce directional frictions: $c_b$ for borrowing and $c_l$ for lending.",
        r"These frictions create a band where moving the budget is not worthwhile.",
        draw_duration=1.0,
        hold_duration=0.8,
    )
    reset_sample_banks(schedule, SAMPLE_BUDGETS, prefix="act2_bank", duration=0.0)

    schedule.add(
        Parallel(
            (
                Erase("intersection_i"),
                Erase("i"),
                EraseText("i_label"),
                Draw(Curve("i_plus_cb", x, x * 0.0 + _scalar(data["i_plus_cb"]), color="red", linewidth=2.0, alpha=1.0)),
                DrawText(Text("i_plus_cb_label", 1.02, _scalar(data["i_plus_cb"]) - 0.02, r"$i+c_b$", color="red", alpha=1.0, fontsize=14.0, ha="left")),
                Draw(Curve("i_minus_cl", x, x * 0.0 + _scalar(data["i_minus_cl"]), color="red", linewidth=2.0, alpha=1.0)),
                DrawText(Text("i_minus_cl_label", 1.02, _scalar(data["i_minus_cl"]) - 0.02, r"$i-c_l$", color="red", alpha=1.0, fontsize=14.0, ha="left")),
                FillBetween(FillBetweenArea("friction_band", x, x * 0.0 + _scalar(data["i_plus_cb"]), x * 0.0 + _scalar(data["i_minus_cl"]), color="orange", alpha=0.25, linewidth=0.0)),
                MoveText("uprime_label", newx=0.82, newy=0.03),
            )
        ),
        duration=1.0,
    )
    schedule.add(Stress("i_plus_cb", glow_color="#f87171", glow_width=9.0), duration=0.4)
    schedule.add(Stress("i_minus_cl", glow_color="#f87171", glow_width=9.0), duration=0.4)

    left_x = nearest_crossing(
        x,
        _array(data["uprime"]),
        _scalar(data["i_plus_cb"]),
        target_x=0.40,
    ).x
    right_x = nearest_crossing(
        x,
        _array(data["uprime"]),
        _scalar(data["i_minus_cl"]),
        target_x=0.60,
    ).x

    schedule.add(
        Parallel(
            (
                Draw(point_marker("intersection_i_plus_cb", left_x, _scalar(data["i_plus_cb"]), color="orange", marker="d", markersize=10.0)),
                Draw(point_marker("intersection_i_minus_cl", right_x, _scalar(data["i_minus_cl"]), color="orange", marker="d", markersize=10.0)),
            )
        ),
        duration=0.0,
    )
    schedule.add(
        Parallel(
            (
                Stress("intersection_i_plus_cb", glow_color="#fb923c", glow_width=14.0),
                Stress("intersection_i_minus_cl", glow_color="#fb923c", glow_width=14.0),
            )
        ),
        duration=0.5,
    )
    schedule.add_break(0.4)
    move_sample_banks_to_band(schedule, SAMPLE_BUDGETS, left_x, right_x, prefix="act2_bank", duration=1.0)
    schedule.add_break(1.0)

    clear_narration(schedule, duration=0.2)
    erase_sample_banks(schedule, SAMPLE_BUDGETS, prefix="act2_bank", duration=0.2)
    _add_transitions(
        schedule,
        [
            Erase("intersection_i_plus_cb"),
            Erase("intersection_i_minus_cl"),
        ],
        duration=0.0,
    )

    replace_narration(
        schedule,
        r"The remuneration rate $\rho(b)$ can be used to reshape marginal incentives.",
        r"This moves the effective curve from $u'(b)$ to $u'(b)+\rho(b)$.",
        draw_duration=1.0,
        hold_duration=0.7,
    )
    schedule.add(
        Parallel(
            (
                Draw(Curve("rho", x, _array(data["rho0"]), color="green", linewidth=2.0, alpha=1.0, value_range=PLOT_RANGE)),
                FillBetween(FillBetweenArea("rho_fill", x, _array(data["rho0"]), 0.0, color="green", alpha=0.3, linewidth=0.0, value_range=PLOT_RANGE)),
                DrawText(Text("rho_label", 0.64, 0.285, r"$\rho(b)$", color="green", alpha=1.0, fontsize=14.0, ha="left")),
            )
        ),
        duration=1.0,
    )
    schedule.add(Stress("rho", glow_color="#22c55e", glow_width=8.0), duration=0.5)

    schedule.add(
        MoveFillBetween(
            "rho_fill",
            newy1=_array(data["uprime_plus_rho0"]),
            newy2=_array(data["uprime"]),
            color="green",
            alpha=0.3,
            value_range=PLOT_RANGE,
        ),
        duration=1.0,
    )
    schedule.add(
        Parallel(
            (
                CurveStyle("uprime", color="grey", linestyle="--", alpha=0.72),
                CurveStyle("rho", color="teal", linestyle="--", alpha=0.72),
                EraseFillBetween("rho_fill"),
                Draw(Curve("uprime_plus_rho", x, _array(data["uprime_plus_rho0"]), color="black", alpha=1.0, linewidth=2.0, value_range=PLOT_RANGE)),
                TextStyle("uprime_label", color="grey"),
                TextStyle("rho_label", color="teal"),
                DrawText(Text("uprime_plus_rho_label", 0.90, 0.32, r"$u'(b)+\rho(b)$", color="black", alpha=1.0, fontsize=14.0, ha="left")),
            )
        ),
        duration=1.0,
    )
    schedule.add(Stress("uprime_plus_rho", glow_color="#111827", glow_width=9.0), duration=0.6)

    clear_narration(schedule, duration=0.2)
    schedule.add_break(1.2)
    return schedule


def build_act_3(
    initial_scene: Scene,
    x: np.ndarray,
    data: dict[str, np.ndarray | float | int],
) -> Schedule:
    schedule = Schedule(initial_scene=initial_scene)

    replace_narration(
        schedule,
        r"By changing $\rho$, we change the marginal incentive $u'(b)+\rho(b)$.",
        r"A non-monotone remuneration schedule can create non-local profitable deviations.",
        draw_duration=1.0,
        hold_duration=0.7,
    )
    schedule.add(
        Parallel(
            (
                Jitter("rho", y_amplitude=[0.05, 0.02], cycles=[0.8, 2.2], seed=7),
                Jitter("uprime_plus_rho", y_amplitude=[0.05, 0.02], cycles=[0.8, 2.2], seed=7),
            )
        ),
        duration=1.7,
    )
    clear_narration(schedule, duration=0.2)

    _add_transitions(
        schedule,
        [
            CurveStyle("uprime", alpha=0.0),
            CurveStyle("rho", alpha=0.0),
            TextStyle("uprime_label", alpha=0.0),
            TextStyle("rho_label", alpha=0.0),
        ],
        duration=0.8,
    )
    schedule.add(Stress("uprime_plus_rho", glow_color="#111827", glow_width=10.0), duration=0.6)

    start_index = _closest_index(x, 0.565)
    borrow_crossings = level_crossings(
        x,
        _array(data["uprime_plus_rho0"]),
        _scalar(data["i_plus_cb"]),
    )
    borrow_entry_index = borrow_crossings[1].right_index
    borrow_exit_index = borrow_crossings[2].left_index
    borrow_loss_pause = midpoint_index(start_index, borrow_entry_index)
    borrow_gain_pause = midpoint_index(borrow_entry_index, borrow_exit_index)
    schedule.add(Draw(point_marker("initial_budget", float(x[start_index]), float(_array(data["uprime_plus_rho0"])[start_index]), color="green", marker=6, markersize=10.0)), duration=0.0)
    replace_narration(
        schedule,
        r"Consider a bank starting from the green triangle.",
        r"Small borrowing first creates losses, shown by the red area.",
        draw_duration=1.0,
        hold_duration=0.6,
    )
    schedule.add(
        region_fill(
            "borrow_red_1",
            x,
            _array(data["uprime_plus_rho0"]),
            _scalar(data["i_plus_cb"]),
            start=start_index,
            stop=borrow_loss_pause,
            color="red",
        ),
        duration=1.0,
    )
    schedule.add_break(0.8)

    replace_narration(
        schedule,
        r"Borrowing further can become locally profitable because of remuneration.",
        r"The green region grows and can outweigh the earlier loss.",
        draw_duration=1.0,
        hold_duration=0.5,
    )
    _add_transitions(
        schedule,
        [
            region_fill(
                "borrow_red_2",
                x,
                _array(data["uprime_plus_rho0"]),
                _scalar(data["i_plus_cb"]),
                start=max(start_index, borrow_loss_pause - 1),
                stop=borrow_entry_index,
                color="red",
            ),
            region_fill(
                "borrow_green_1",
                x,
                _array(data["uprime_plus_rho0"]),
                _scalar(data["i_plus_cb"]),
                start=max(start_index, borrow_entry_index - 1),
                stop=borrow_gain_pause,
                color="green",
            ),
        ],
        duration=1.0,
    )
    schedule.add_break(0.5)
    schedule.add(
        region_fill(
            "borrow_green_2",
            x,
            _array(data["uprime_plus_rho0"]),
            _scalar(data["i_plus_cb"]),
            start=max(borrow_entry_index, borrow_gain_pause - 1),
            stop=borrow_exit_index,
            color="green",
        ),
        duration=1.0,
    )
    schedule.add_break(1.0)

    clear_narration(schedule, duration=0.2)
    _add_transitions(
        schedule,
        [
            EraseFillBetween("borrow_green_2", direction="backward"),
            EraseFillBetween("borrow_green_1", direction="backward"),
            EraseFillBetween("borrow_red_2", direction="backward"),
            EraseFillBetween("borrow_red_1", direction="backward"),
        ],
        duration=0.8,
    )

    replace_narration(
        schedule,
        r"The same non-monotonicity shows up on the lending side.",
        r"Again, locally unprofitable moves can be dominated by larger gains later on.",
        draw_duration=1.0,
        hold_duration=0.5,
    )
    lend_crossings = level_crossings(
        x,
        _array(data["uprime_plus_rho0"]),
        _scalar(data["i_minus_cl"]),
    )
    lend_entry_index = lend_crossings[0].right_index
    lend_exit_index = lend_crossings[1].left_index
    lend_lower_bound_index = _closest_index(x, 0.01)
    _add_transitions(
        schedule,
        [
            region_fill(
                "lend_signed",
                x,
                _array(data["uprime_plus_rho0"]),
                _scalar(data["i_minus_cl"]),
                start=lend_lower_bound_index,
                stop=start_index,
                color="red",
                positive_color="green",
                negative_color="red",
                direction="backward",
            ),
        ],
        duration=1.1,
    )
    schedule.add_break(0.8)

    clear_narration(schedule, duration=0.2)
    _add_transitions(
        schedule,
        [
            EraseFillBetween("lend_signed"),
            Erase("initial_budget"),
        ],
        duration=0.9,
    )
    schedule.add_break(0.8)
    return schedule


def build_act_4(
    initial_scene: Scene,
    x: np.ndarray,
    data: dict[str, np.ndarray | float | int],
) -> Schedule:
    schedule = Schedule(initial_scene=initial_scene)

    replace_narration(
        schedule,
        r"Now switch to a monotone remuneration schedule.",
        r"This will recover a clean cutoff rule.",
        draw_duration=0.9,
        hold_duration=0.4,
    )
    _add_transitions(
        schedule,
        [
            CurveStyle("uprime", alpha=1.0),
            CurveStyle("rho", alpha=1.0),
            TextStyle("uprime_label", alpha=1.0),
            TextStyle("rho_label", alpha=1.0),
            Move("rho", newy=_array(data["rho1"])),
            Move("uprime_plus_rho", newy=_array(data["uprime_plus_rho1"])),
            MoveText("rho_label", newx=0.29, newy=0.075),
            MoveText("uprime_plus_rho_label", newx=0.96, newy=0.05),
            MoveText("uprime_label", newx=0.29, newy=0.272),
        ],
        duration=1.1,
    )
    schedule.add(Stress("uprime_plus_rho", glow_color="#111827", glow_width=10.0), duration=0.6)

    left_crossing = nearest_crossing(
        x,
        _array(data["uprime_plus_rho1"]),
        _scalar(data["i_plus_cb"]),
        target_x=0.30,
    )
    right_crossing = nearest_crossing(
        x,
        _array(data["uprime_plus_rho1"]),
        _scalar(data["i_minus_cl"]),
        target_x=0.70,
    )
    left_index = left_crossing.right_index
    right_index = right_crossing.left_index
    left_x = left_crossing.x
    right_x = right_crossing.x

    schedule.add(
        Parallel(
            (
                Draw(point_marker("intersection_i_plus_cb", left_x, _scalar(data["i_plus_cb"]), color="orange", marker="d", markersize=10.0)),
                Draw(point_marker("intersection_i_minus_cl", right_x, _scalar(data["i_minus_cl"]), color="orange", marker="d", markersize=10.0)),
            )
        ),
        duration=0.0,
    )
    schedule.add(
        Parallel(
            (
                DrawText(Text("bL_label", left_x - 0.01, _scalar(data["i_plus_cb"]) + 0.03, r"$b_L$", color="red", alpha=1.0, fontsize=14.0, ha="left")),
                DrawText(Text("bH_label", right_x - 0.01, _scalar(data["i_minus_cl"]) + 0.03, r"$b_H$", color="red", alpha=1.0, fontsize=14.0, ha="left")),
            )
        ),
        duration=0.2,
    )
    schedule.add(
        Parallel(
            (
                Stress("intersection_i_plus_cb", glow_color="#fb923c", glow_width=14.0),
                Stress("intersection_i_minus_cl", glow_color="#fb923c", glow_width=14.0),
            )
        ),
        duration=0.6,
    )

    replace_narration(
        schedule,
        r"Monotonicity makes $u'(b)+\rho(b)$ cross each friction line only once.",
        r"This creates a borrower cutoff $b_L$ and a lender cutoff $b_H$.",
        draw_duration=1.0,
        hold_duration=0.7,
    )

    draw_sample_banks(schedule, SAMPLE_BUDGETS, prefix="act4_bank", duration=0.0, color="#b91c1c")
    replace_narration(
        schedule,
        r"Banks with low initial budgets borrow up to $b_L$.",
        r"High-budget banks lend down to $b_H$, while the middle region stays inactive.",
        draw_duration=1.0,
        hold_duration=0.5,
    )
    move_sample_banks_to_band(
        schedule,
        SAMPLE_BUDGETS,
        left_x,
        right_x,
        prefix="act4_bank",
        duration=0.9,
        recolor=True,
    )
    schedule.add_break(0.8)
    erase_sample_banks(schedule, SAMPLE_BUDGETS, prefix="act4_bank", duration=0.3)

    replace_narration(
        schedule,
        r"Only what happens inside $[b_L, b_H]$ matters for the final distribution.",
        r"So we can simplify $\rho$ outside that interval without changing behavior.",
        draw_duration=1.0,
        hold_duration=1.1,
    )

    rho2 = np.clip(_array(data["rho1"]), _array(data["rho1"])[right_index], _array(data["rho1"])[0])
    data["rho2"] = rho2
    data["uprime_plus_rho2"] = _array(data["uprime"]) + rho2
    update_rho_with_narration(
        schedule,
        data,
        rho_key="rho2",
        combined_key="uprime_plus_rho2",
        line1=r"We first flatten the upper tail of $\rho$.",
        line2=r"The cutoff points stay put, so the induced allocation is unchanged.",
        hold_duration=1.0,
    )
    schedule.add(
        Parallel(
            (
                Stress("intersection_i_plus_cb", glow_color="#fb923c", glow_width=14.0),
                Stress("intersection_i_minus_cl", glow_color="#fb923c", glow_width=14.0),
            )
        ),
        duration=0.45,
    )

    rho3 = np.clip(_array(data["rho2"]), _array(data["rho2"])[-1], _array(data["rho2"])[left_index])
    data["rho3"] = rho3
    data["uprime_plus_rho3"] = _array(data["uprime"]) + rho3
    update_rho_with_narration(
        schedule,
        data,
        rho_key="rho3",
        combined_key="uprime_plus_rho3",
        line1=r"We can also flatten the low-budget side of $\rho$.",
        line2=r"Again, nothing changes as long as $b_L$ and $b_H$ remain fixed.",
        hold_duration=1.0,
    )
    schedule.add(
        Parallel(
            (
                Stress("intersection_i_plus_cb", glow_color="#fb923c", glow_width=14.0),
                Stress("intersection_i_minus_cl", glow_color="#fb923c", glow_width=14.0),
            )
        ),
        duration=0.45,
    )

    rho4 = (x <= left_x) * (_array(data["rho2"])[left_index] - _array(data["rho2"])[right_index]) + _array(data["rho2"])[right_index]
    data["rho4"] = rho4
    data["uprime_plus_rho4"] = _array(data["uprime"]) + rho4
    update_rho_with_narration(
        schedule,
        data,
        rho_key="rho4",
        combined_key="uprime_plus_rho4",
        line1=r"The middle region can be reduced as well.",
        line2=r"The result is the simplest monotone schedule with the same cutoffs.",
        hold_duration=1.2,
    )
    schedule.add_break(1.0)
    return schedule


def build_act_5(
    initial_scene: Scene,
    x: np.ndarray,
    data: dict[str, np.ndarray | float | int],
) -> Schedule:
    schedule = Schedule(initial_scene=initial_scene)

    replace_narration(
        schedule,
        r"Any monotone remuneration schedule can be improved to a two-tier monotone schedule.",
        r"The key object is the feasible pair $(b_L, b_H)$.",
        draw_duration=1.0,
        hold_duration=1.0,
    )

    left_index, right_index = choose_feasible_cutoff_pair(
        x,
        _array(data["uprime"]),
        _scalar(data["cb"]) + _scalar(data["cl"]),
        target_pair=(0.405, 0.500),
    )
    b_l = float(x[left_index])
    b_h = float(x[right_index])
    low_value = _scalar(data["i_minus_cl"]) - _array(data["uprime"])[right_index]
    high_value = _scalar(data["i_plus_cb"]) - _array(data["uprime"])[left_index]
    rho5 = (x <= b_l) * (high_value - low_value) + low_value
    data["rho5"] = rho5
    data["uprime_plus_rho5"] = _array(data["uprime"]) + rho5

    _add_transitions(
        schedule,
        [
            Move("intersection_i_plus_cb", newx=[b_l], newy=[_scalar(data["i_plus_cb"])]),
            Move("intersection_i_minus_cl", newx=[b_h], newy=[_scalar(data["i_minus_cl"])]),
            MoveText("bL_label", newx=b_l - 0.01, newy=_scalar(data["i_plus_cb"]) + 0.03),
            MoveText("bH_label", newx=b_h - 0.01, newy=_scalar(data["i_minus_cl"]) + 0.03),
        ],
        duration=0.8,
    )
    schedule.add(
        Parallel(
            (
                Stress("intersection_i_plus_cb", glow_color="#fb923c", glow_width=14.0),
                Stress("intersection_i_minus_cl", glow_color="#fb923c", glow_width=14.0),
            )
        ),
        duration=0.5,
    )

    replace_narration(
        schedule,
        r"For any feasible $(b_L, b_H)$ with $u'(b_L)-u'(b_H)\geq c_l+c_b$,",
        r"we can choose a two-tier $\rho$ that supports exactly those cutoffs.",
        draw_duration=1.0,
        hold_duration=0.9,
    )
    _add_transitions(
        schedule,
        [
            Move("rho", newy=_array(data["rho5"]), color="green", alpha=1.0, linestyle="solid"),
            Move("uprime_plus_rho", newy=_array(data["uprime_plus_rho5"]), color="black", alpha=1.0, linestyle="solid"),
            MoveText("rho_label", newx=0.14, newy=0.25, color="green", alpha=1.0),
            MoveText("uprime_plus_rho_label", newx=0.10, newy=0.85, color="black", alpha=1.0),
        ],
        duration=1.1,
    )
    schedule.add(
        Parallel(
            (
                Stress("rho", glow_color="#22c55e", glow_width=9.0),
                Stress("uprime_plus_rho", glow_color="#111827", glow_width=9.0),
            )
        ),
        duration=0.7,
    )
    clear_narration(schedule, duration=0.2)
    schedule.add_break(1.2)
    return schedule


def axis_styles(
    stage_names: Iterable[str],
    axes: Mapping[str, Mapping[str, np.ndarray]],
    *,
    alpha: float,
) -> list[CurveStyle]:
    return [
        CurveStyle(f"{stage_name} {axis_name}", alpha=alpha)
        for stage_name in stage_names
        for axis_name in axes
    ]


def clear_narration(schedule: Schedule, duration: float = 0.2) -> None:
    transitions = [
        EraseText(text_id)
        for text_id in NARRATION_LAYOUT
        if schedule.final_scene.contains_text(text_id)
    ]
    _add_transitions(schedule, transitions, duration=duration)


def retime_schedule(
    schedule: Schedule,
    *,
    duration_scale: float = TIMING_SCALE,
    stress_pause: float = STRESS_BREATH,
) -> Schedule:
    if duration_scale <= 0:
        raise ValueError("duration_scale must be positive.")
    if stress_pause < 0:
        raise ValueError("stress_pause must be non-negative.")

    retimed = Schedule(initial_scene=schedule.initial_scene)
    for entry in schedule.entries:
        scaled_duration = entry.duration * duration_scale
        retimed.entries.append(
            ScheduledTransition(
                transition=entry.transition,
                duration=scaled_duration,
            )
        )
        if stress_pause > 0 and transition_contains(entry.transition, StressTransition):
            retimed.entries.append(
                ScheduledTransition(
                    transition=PauseTransition(),
                    duration=stress_pause,
                )
            )

    return retimed


def transition_contains(transition: object, transition_type: type[object]) -> bool:
    if isinstance(transition, transition_type):
        return True
    if isinstance(transition, ParallelTransition):
        return any(transition_contains(child, transition_type) for child in transition.transitions)
    return False


def replace_narration(
    schedule: Schedule,
    line1: str | None,
    line2: str | None = None,
    *,
    draw_duration: float = 1.0,
    hold_duration: float = 0.8,
    clear_duration: float = 0.2,
) -> None:
    clear_narration(schedule, duration=clear_duration)
    transitions = []
    if line1 is not None:
        transitions.append(DrawText(narration_text("narration1", line1)))
    if line2 is not None:
        transitions.append(DrawText(narration_text("narration2", line2)))
    _add_transitions(schedule, transitions, duration=draw_duration)
    if hold_duration > 0:
        schedule.add_break(hold_duration)


def narration_text(text_id: str, content: str) -> Text:
    x_position, y_position = NARRATION_LAYOUT[text_id]
    return Text(
        text_id,
        x_position,
        y_position,
        content,
        color="black",
        alpha=1.0,
        fontsize=11.0,
        ha="left",
    )


def point_marker(
    curve_id: str,
    x_value: float,
    y_value: float,
    *,
    color: str,
    marker: str | int,
    markersize: float,
) -> Curve:
    return Curve(
        curve_id,
        [x_value],
        [y_value],
        color=color,
        alpha=1.0,
        linewidth=0.0,
        line_kwargs={"marker": marker, "markersize": markersize},
    )


def bank_lane_positions(count: int) -> np.ndarray:
    if count < 0:
        raise ValueError("count must be non-negative.")
    if count == 0:
        return np.asarray([], dtype=float)
    return BANK_LANES[np.arange(count) % BANK_LANES.size]


def bank_guide_curve(curve_id: str) -> Curve:
    return Curve(
        curve_id,
        [0.0, 1.0],
        [BANK_GUIDE_Y, BANK_GUIDE_Y],
        color="#cbd5e1",
        alpha=0.9,
        linewidth=1.4,
        linestyle="--",
    )


def bank_guide_label(text_id: str) -> Text:
    return Text(
        text_id,
        -0.09,
        BANK_GUIDE_Y,
        "Banks",
        color="#64748b",
        alpha=1.0,
        fontsize=10.5,
        ha="left",
        va="center",
    )


def bank_scatter(
    scatter_id: str,
    budgets: Sequence[float],
    *,
    x_values: np.ndarray | None = None,
    colors: str | Sequence[str] = "#dc2626",
) -> Scatter:
    budgets_array = np.asarray(budgets, dtype=float)
    return Scatter(
        scatter_id,
        budgets_array if x_values is None else np.asarray(x_values, dtype=float),
        bank_lane_positions(budgets_array.size),
        color=colors,
        alpha=1.0,
        marker="s",
        size=np.full(budgets_array.shape, 115.0, dtype=float),
        linewidth=1.1,
        edgecolor="white",
    )


def draw_sample_banks(
    schedule: Schedule,
    budgets: Sequence[float],
    *,
    prefix: str,
    duration: float,
    color: str,
) -> None:
    _add_transitions(
        schedule,
        [
            Draw(bank_guide_curve(f"{prefix}_guide")),
            DrawText(bank_guide_label(f"{prefix}_label")),
            DrawScatter(bank_scatter(prefix, budgets, colors=color)),
        ],
        duration=duration,
    )


def move_sample_banks_to_point(
    schedule: Schedule,
    budgets: Sequence[float],
    target_x: float,
    *,
    prefix: str,
    duration: float,
) -> None:
    budgets_array = np.asarray(budgets, dtype=float)
    schedule.add(
        MoveScatter(
            prefix,
            newx=np.full(budgets_array.shape, float(target_x), dtype=float),
            newy=bank_lane_positions(budgets_array.size),
        ),
        duration=duration,
    )


def move_sample_banks_to_band(
    schedule: Schedule,
    budgets: Sequence[float],
    left_x: float,
    right_x: float,
    *,
    prefix: str,
    duration: float,
    recolor: bool = False,
) -> None:
    budgets_array = np.asarray(budgets, dtype=float)
    colors: str | list[str]
    if recolor:
        colors = [
            bank_color_for_cutoffs(float(budget), left_x, right_x)
            for budget in budgets_array
        ]
    else:
        colors = "#dc2626"

    schedule.add(
        MoveScatter(
            prefix,
            newx=np.clip(budgets_array, left_x, right_x),
            newy=bank_lane_positions(budgets_array.size),
            color=colors,
        ),
        duration=duration,
    )


def reset_sample_banks(
    schedule: Schedule,
    budgets: Sequence[float],
    *,
    prefix: str,
    duration: float,
) -> None:
    budgets_array = np.asarray(budgets, dtype=float)
    schedule.add(
        MoveScatter(
            prefix,
            newx=budgets_array,
            newy=bank_lane_positions(budgets_array.size),
            color="#dc2626",
        ),
        duration=duration,
    )


def erase_sample_banks(
    schedule: Schedule,
    budgets: Sequence[float],
    *,
    prefix: str,
    duration: float,
) -> None:
    transitions: list[object] = []
    if schedule.final_scene.contains_scatter(prefix):
        transitions.append(EraseScatter(prefix))
    if schedule.final_scene.contains_curve(f"{prefix}_guide"):
        transitions.append(Erase(f"{prefix}_guide"))
    if schedule.final_scene.contains_text(f"{prefix}_label"):
        transitions.append(EraseText(f"{prefix}_label"))
    _add_transitions(schedule, transitions, duration=duration)


def bank_color_for_cutoffs(budget: float, left_x: float, right_x: float) -> str:
    if budget < left_x:
        return "#dc2626"
    if budget > right_x:
        return "#059669"
    return "#64748b"


def level_crossings(
    x_values: np.ndarray,
    y_values: np.ndarray,
    level: float,
) -> list[LevelCrossing]:
    crossings: list[LevelCrossing] = []
    diff = np.asarray(y_values, dtype=float) - float(level)

    for index in range(diff.size - 1):
        left_value = float(diff[index])
        right_value = float(diff[index + 1])

        if np.isclose(left_value, 0.0) and np.isclose(right_value, 0.0):
            continue
        if np.signbit(left_value) == np.signbit(right_value) and not np.isclose(left_value, 0.0) and not np.isclose(right_value, 0.0):
            continue

        x_left = float(x_values[index])
        x_right = float(x_values[index + 1])
        y_left = float(y_values[index])
        y_right = float(y_values[index + 1])

        if np.isclose(y_left, y_right):
            crossing_x = x_left
        else:
            weight = (float(level) - y_left) / (y_right - y_left)
            crossing_x = x_left + weight * (x_right - x_left)

        if crossings and np.isclose(crossings[-1].x, crossing_x):
            continue
        crossings.append(LevelCrossing(x=float(crossing_x), left_index=index, right_index=index + 1))

    return crossings


def nearest_crossing(
    x_values: np.ndarray,
    y_values: np.ndarray,
    level: float,
    *,
    target_x: float | None = None,
) -> LevelCrossing:
    crossings = level_crossings(x_values, y_values, level)
    if not crossings:
        raise ValueError("No crossing found.")

    if target_x is None:
        return crossings[0]
    return min(crossings, key=lambda crossing: abs(crossing.x - target_x))


def midpoint_index(left_index: int, right_index: int) -> int:
    if right_index < left_index:
        left_index, right_index = right_index, left_index
    return int((left_index + right_index) // 2)


def choose_feasible_cutoff_pair(
    x_values: np.ndarray,
    uprime_values: np.ndarray,
    total_spread: float,
    *,
    target_pair: tuple[float, float] = (0.40, 0.50),
) -> tuple[int, int]:
    target_left, target_right = target_pair
    best_score = float("inf")
    best_pair: tuple[int, int] | None = None

    for left_index in range(len(x_values) - 1):
        for right_index in range(left_index + 1, len(x_values)):
            if float(uprime_values[left_index] - uprime_values[right_index]) < total_spread:
                continue

            score = abs(float(x_values[left_index]) - target_left) + abs(float(x_values[right_index]) - target_right)
            if score < best_score:
                best_score = score
                best_pair = (left_index, right_index)

    if best_pair is None:
        raise ValueError("Could not find a feasible cutoff pair.")
    return best_pair


def region_fill(
    fill_id: str,
    x_values: np.ndarray,
    upper_values: np.ndarray,
    lower_values: np.ndarray | float,
    *,
    start: int,
    stop: int,
    color: str,
    positive_color: str | None = None,
    negative_color: str | None = None,
    alpha: float = 0.3,
    direction: str = "forward",
) -> FillBetween:
    x_slice = x_values[start:stop]
    upper_slice = upper_values[start:stop]
    if np.isscalar(lower_values):
        lower_slice = lower_values
    else:
        lower_slice = lower_values[start:stop]

    return FillBetween(
        FillBetweenArea(
            fill_id,
            x_slice,
            upper_slice,
            lower_slice,
            color=color,
            positive_color=positive_color,
            negative_color=negative_color,
            alpha=alpha,
            value_range=PLOT_RANGE,
            linewidth=0.0,
        ),
        direction=direction,
    )


def update_rho_with_narration(
    schedule: Schedule,
    data: Mapping[str, np.ndarray | float | int],
    *,
    rho_key: str,
    combined_key: str,
    line1: str,
    line2: str,
    hold_duration: float,
) -> None:
    replace_narration(
        schedule,
        line1,
        line2,
        draw_duration=1.0,
        hold_duration=0.5,
    )
    _add_transitions(
        schedule,
        [
            Move("rho", newy=_array(data[rho_key])),
            Move("uprime_plus_rho", newy=_array(data[combined_key])),
        ],
        duration=1.1,
    )
    schedule.add_break(hold_duration)
    clear_narration(schedule, duration=0.2)


def render_schedule_html(
    schedule: Schedule,
    *,
    title: str | None = None,
    fps: int = FPS,
) -> str:
    fig, ax = make_figure()
    anim = schedule.build_animation(fig=fig, ax=ax, fps=fps, xlim=XLIM, ylim=YLIM, title=title)
    plt.close(fig)
    return anim.to_jshtml()


def save_schedule_html(
    schedule: Schedule,
    output_path: str | Path,
    *,
    title: str | None = None,
    fps: int = FPS,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_schedule_html(schedule, title=title, fps=fps), encoding="utf-8")
    return output


def save_schedule_video(
    schedule: Schedule,
    output_path: str | Path,
    *,
    title: str | None = None,
    fps: int = FPS,
    dpi: int = VIDEO_DPI,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = make_figure()
    anim = schedule.build_animation(fig=fig, ax=ax, fps=fps, xlim=XLIM, ylim=YLIM, title=title)
    writer = FFMpegWriter(fps=fps, bitrate=2200)
    anim.save(str(output), writer=writer, dpi=dpi)
    plt.close(fig)
    return output


def export_video_bundle(
    bundle: TieredRemunerationBundle,
    output_dir: str | Path,
    *,
    fps: int = FPS,
    dpi: int = VIDEO_DPI,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    exported: dict[str, Path] = {}
    act_videos: list[Path] = []

    for index, (act_name, schedule) in enumerate(bundle.acts.items(), start=1):
        video_path = output_path / f"tieredremuneration_act_{index}.mp4"
        exported[act_name] = save_schedule_video(
            schedule,
            video_path,
            title=act_name,
            fps=fps,
            dpi=dpi,
        )
        act_videos.append(video_path)

    full_video = output_path / "tieredremuneration_full.mp4"
    concatenate_videos(act_videos, full_video)
    exported["full_video"] = full_video
    return exported


def export_bundle_artifacts(
    bundle: TieredRemunerationBundle,
    output_dir: str | Path,
    *,
    fps: int = FPS,
    dpi: int = VIDEO_DPI,
) -> dict[str, object]:
    output_path = Path(output_dir)
    html_dir = output_path / "html"
    video_dir = output_path / "video"
    html_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    html_exports: OrderedDict[str, Path] = OrderedDict()
    video_exports: OrderedDict[str, Path] = OrderedDict()
    act_videos: list[Path] = []

    for index, (act_name, schedule) in enumerate(bundle.acts.items(), start=1):
        slug = f"tieredremuneration_act_{index}"
        html_exports[act_name] = save_schedule_html(
            schedule,
            html_dir / f"{slug}.html",
            title=act_name,
            fps=fps,
        )
        video_path = save_schedule_video(
            schedule,
            video_dir / f"{slug}.mp4",
            title=act_name,
            fps=fps,
            dpi=dpi,
        )
        video_exports[act_name] = video_path
        act_videos.append(video_path)

    combined_video = concatenate_videos(
        act_videos,
        video_dir / "tieredremuneration_full.mp4",
    )

    return {
        "html": html_exports,
        "video": video_exports,
        "combined_video": combined_video,
    }


def concatenate_videos(video_paths: Sequence[Path], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.parent / "tieredremuneration_concat.txt"
    manifest_path.write_text(
        "\n".join(f"file '{video_path.resolve()}'" for video_path in video_paths),
        encoding="utf-8",
    )

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


def make_figure() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axis("off")
    return fig, ax


def _add_transitions(
    schedule: Schedule,
    transitions: Iterable[object],
    *,
    duration: float,
) -> None:
    prepared = tuple(transition for transition in transitions if transition is not None)
    if not prepared:
        return
    if len(prepared) == 1:
        schedule.add(prepared[0], duration=duration)
        return
    schedule.add(Parallel(prepared), duration=duration)


def _closest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def _array(value: np.ndarray | float | int) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _scalar(value: np.ndarray | float | int) -> float:
    return float(value)
