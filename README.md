# visualizer

`visualizer` is a small Python package for building graph animation sequences on top of `matplotlib`.

The library models animation as:

- `Scene`: the persistent state of curves, filled regions, and text
- `Transition`: a time-dependent change from one scene to the next
- `Schedule`: a sequence of timed transitions compiled into a `FuncAnimation`

Curves, fills, and text can use arbitrary finite plot coordinates.

## Features

- draw, move, erase, and pause transitions
- erase transitions for `fill_between` regions
- `fill_between` creation and movement
- text labels with draw, move, style, and erase transitions
- concurrent transitions with `Parallel`
- style changes for color, alpha, linewidth, and linestyle
- transient emphasis effects like stress glow and jitter
- support for pre-populated initial scenes
- act-based composition via `final_scene`, `next_act()`, and `Schedule.combine(...)`
- automatic axis fitting for arbitrary coordinate ranges
- per-curve and per-fill clipping windows via `domain` and `value_range`

## Installation

Install from the repository for local development:

```bash
pip install -e .
```

Install the publish tooling when preparing a release:

```bash
pip install -e .[publish]
```

Install the notebook tooling when running the demos locally:

```bash
pip install -e .[notebooks]
```

## Quick Start

```python
import numpy as np
from visualizer import (
    Curve,
    Draw,
    FillBetweenArea,
    FillBetween,
    Parallel,
    Schedule,
)

x = np.linspace(0.0, 1.0, 250)
y = np.abs(np.sin(x))

schedule = Schedule()
schedule.add(
    Parallel(
        (
            Draw(Curve("wave", x, y, color="#0f766e", linewidth=3.0)),
            FillBetween(
                FillBetweenArea(
                    "wave_fill",
                    x,
                    y,
                    0.0,
                    color="#99f6e4",
                    alpha=0.35,
                    linewidth=0.0,
                )
            ),
        )
    ),
    duration=1.5,
)

anim = schedule.build_animation()
```

## Common Patterns

Start from a pre-drawn scene:

```python
from visualizer import Curve, Scene, Schedule

initial_scene = Scene().add_curve(
    Curve("reference", x, 0.2 + 0.3 * x, color="#94a3b8", linestyle="--")
)
schedule = Schedule(initial_scene=initial_scene)
```

Insert dead time:

```python
schedule.add_break(0.75)
# or
schedule.pause(0.75)
```

Build animation in modular acts:

```python
from visualizer import Curve, Draw, Erase, Move, Schedule

act_1 = Schedule()
act_1.add(Draw(Curve("u", x, y, color="#dc2626", linewidth=3.0)), duration=1.5)

act_2 = act_1.next_act()
act_2.add_break(0.5)
act_2.add(Move("u", newy=y_prime), duration=1.25)

act_3 = act_2.next_act()
act_3.add(Erase("u"), duration=1.0)

full_schedule = Schedule.combine(
    [act_1, act_2, act_3],
    validate_initial_scene=True,
)
```

`next_act()` starts a new schedule from the previous act's final scene, which makes it easy to debug or render each act independently. `Schedule.combine(...)` stitches those acts back into one continuous schedule later. If you want to append onto an existing schedule in place, use `extend_schedule(...)`; if you want a new combined schedule without mutating the original, use `appended(...)`.

Add animated text labels:

```python
from visualizer import DrawText, MoveText, Text

schedule.add(
    DrawText(Text("label", 0.2, 0.85, "Peak", color="#111827", fontsize=14)),
    duration=0.4,
)
schedule.add(
    MoveText("label", newx=0.65, newy=0.55, color="#dc2626", rotation=-12),
    duration=0.8,
)
```

Jitter a curve and its fill together, including multiple frequencies:

```python
from visualizer import Jitter, JitterFillBetween, Parallel

schedule.add(
    Parallel(
        (
            Jitter(
                "wave",
                y_amplitude=[0.02, 0.01],
                cycles=[6.0, 14.0],
                seed=7,
            ),
            JitterFillBetween(
                "wave_fill",
                upper_y_amplitude=[0.02, 0.01],
                lower_y_amplitude=[0.006, 0.004],
                upper_cycles=[6.0, 14.0],
                lower_cycles=[4.0, 9.0],
                upper_seed=7,
                lower_seed=21,
            ),
        )
    ),
    duration=0.8,
)
```

You can still use the original scalar form as well:

```python
Jitter("wave", y_amplitude=0.03, cycles=10.0, seed=7)
JitterFillBetween("wave_fill", y1_amplitude=0.03, y2_amplitude=0.0, cycles=10.0, seed=7)
```

Move a curve while changing its clip window:

```python
schedule.add(
    Move(
        "wave",
        newy=y_square,
        domain=(0.1, 0.9),
        value_range=(0.05, 0.8),
    ),
    duration=1.2,
)
```

Use arbitrary axis ranges when rendering:

```python
fig, ax = plt.subplots(figsize=(10, 6))
anim = schedule.build_animation(fig=fig, ax=ax, xlim=(-0.2, 1.2), ylim=(-0.2, 1.2))
```

If you omit `xlim` and `ylim`, the animation now auto-fits to the data range of the curves, fills, and text anchors in the schedule.

Clip a curve or fill to a specific plotting window:

```python
curve = Curve(
    "main",
    x,
    y,
    color="#0f766e",
    linewidth=3.0,
    domain=(0.0, 1.0),
    value_range=(0.0, 1.0),
)

fill = FillBetweenArea(
    "main_fill",
    x,
    y,
    0.0,
    color="#99f6e4",
    alpha=0.35,
    domain=(0.0, 1.0),
    value_range=(0.0, 1.0),
)
```

For lines, geometry outside the window is hidden and the visible parts are split into separate segments. For fills, `x` is masked by `domain` and `y` values are clipped into `value_range`.

## Notebook Demos

- [`notebooks/basic_demo.ipynb`](notebooks/basic_demo.ipynb): basic drawing, styling, clipping, modular scheduling, and combined-transition examples

## Publishing

The repository includes:

- package metadata in `pyproject.toml`
- a release workflow in `.github/workflows/python-publish.yml`
- a packaging CI workflow in `.github/workflows/package-check.yml`
- release notes in [`CHANGELOG.md`](CHANGELOG.md)
- contributor setup in [`CONTRIBUTING.md`](CONTRIBUTING.md)
- a release checklist in [`RELEASING.md`](RELEASING.md)

If you plan to publish to PyPI, check the release guide first. In particular, confirm that your chosen distribution name is available before the first release.

## Versioning

The package version is declared in `pyproject.toml` and exposed as `visualizer.__version__`.
