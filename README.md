# visualizer

`visualizer` is a small Python package for building graph animation sequences on top of `matplotlib`.

The library models animation as:

- `Scene`: the persistent state of curves and filled regions
- `Transition`: a time-dependent change from one scene to the next
- `Schedule`: a sequence of timed transitions compiled into a `FuncAnimation`

Curves and fills can use arbitrary finite plot coordinates.

## Features

- draw, move, erase, and pause transitions
- erase transitions for `fill_between` regions
- `fill_between` creation and movement
- concurrent transitions with `Parallel`
- style changes for color, alpha, linewidth, and linestyle
- transient emphasis effects like stress glow and jitter
- support for pre-populated initial scenes
- act-based composition via `final_scene`, `next_act()`, and `Schedule.combine(...)`
- affine-mapped theater subspaces with clipping and optional background patches
- automatic axis fitting for arbitrary coordinate ranges
- per-curve and per-fill clipping windows via `domain` and `value_range`

## Installation

```bash
pip install -e .
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
act_2.add(Move("u", x_prime=None, y_prime=y_prime), duration=1.25)

act_3 = act_2.next_act()
act_3.add(Erase("u"), duration=1.0)

full_schedule = Schedule.combine(
    [act_1, act_2, act_3],
    validate_initial_scene=True,
)
```

`next_act()` starts a new schedule from the previous act's final scene, which makes it easy to debug or render each act independently. `Schedule.combine(...)` stitches those acts back into one continuous schedule later. If you want to append onto an existing schedule in place, use `extend_schedule(...)`; if you want a new combined schedule without mutating the original, use `appended(...)`.

Draw into a theater and resize it:

```python
from visualizer import (
    Curve,
    Draw,
    DrawTheater,
    FillBetween,
    FillBetweenArea,
    MoveTheater,
    Schedule,
    Theater,
)

panel = Theater(
    "panel",
    xlim=(0.15, 0.55),
    ylim=(0.2, 0.65),
    local_xlim=(0.0, 1.0),
    local_ylim=(0.0, 1.0),
    facecolor="#dbeafe",
    edgecolor="#2563eb",
    alpha=0.25,
)

schedule = Schedule()
schedule.add(DrawTheater(panel), duration=0.4)
schedule.add(
    Draw(Curve("wave", x, y, theater_id="panel", color="#1d4ed8", linewidth=3.0)),
    duration=1.0,
)
schedule.add(
    FillBetween(
        FillBetweenArea(
            "wave_fill",
            x,
            y,
            0.0,
            theater_id="panel",
            color="#93c5fd",
            alpha=0.35,
        )
    ),
    duration=1.0,
)
schedule.add(MoveTheater("panel", xlim=(0.05, 0.85), ylim=(0.15, 0.85)), duration=1.0)
```

The theater maps local coordinates into an actual rectangle with a linear transform. Curves and fills assigned to `theater_id="panel"` are clipped to the theater border automatically.

Jitter a curve and its fill together:

```python
from visualizer import Jitter, JitterFillBetween, Parallel

schedule.add(
    Parallel(
        (
            Jitter("wave", y_amplitude=0.03, cycles=10.0, seed=7),
            JitterFillBetween(
                "wave_fill",
                upper_y_amplitude=0.03,
                lower_y_amplitude=0.01,
                upper_cycles=10.0,
                lower_cycles=7.0,
                upper_seed=7,
                lower_seed=21,
            ),
        )
    ),
    duration=0.8,
)
```

For `JitterFillBetween`, the upper and lower boundaries can now use different amplitudes, cycles, and seeds. Horizontal jitter is still shared because `fill_between` uses one common `x` grid.

Use arbitrary axis ranges when rendering:

```python
fig, ax = plt.subplots(figsize=(10, 6))
anim = schedule.build_animation(fig=fig, ax=ax, xlim=(-0.2, 1.2), ylim=(-0.2, 1.2))
```

If you omit `xlim` and `ylim`, the animation now auto-fits to the data range of the curves and fills in the schedule.

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

- [`notebooks/basic_demo.ipynb`](notebooks/basic_demo.ipynb): basic drawing, styling, clipping, and combined-transition examples
- [`notebooks/modular_scheduling.ipynb`](notebooks/modular_scheduling.ipynb): act-based scheduling with `next_act()` and `Schedule.combine(...)`
- [`notebooks/theater_demo.ipynb`](notebooks/theater_demo.ipynb): theater layout, clipping, and linear resizing examples

## Versioning

The package version is declared in `pyproject.toml` and exposed as `visualizer.__version__`.
