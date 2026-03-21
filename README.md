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
- concurrent transitions with `ParallelTransition`
- style changes for color, alpha, linewidth, and linestyle
- transient emphasis effects like stress glow and jitter
- support for pre-populated initial scenes
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
    DrawTransition,
    FillBetweenArea,
    FillBetweenTransition,
    ParallelTransition,
    Schedule,
)

x = np.linspace(0.0, 1.0, 250)
y = np.abs(np.sin(x))

schedule = Schedule()
schedule.add(
    ParallelTransition(
        (
            DrawTransition(Curve("wave", x, y, color="#0f766e", linewidth=3.0)),
            FillBetweenTransition(
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

Jitter a curve and its fill together:

```python
from visualizer import JitterFillBetweenTransition, JitterTransition, ParallelTransition

schedule.add(
    ParallelTransition(
        (
            JitterTransition("wave", y_amplitude=0.03, cycles=10.0, seed=7),
            JitterFillBetweenTransition("wave_fill", y1_amplitude=0.03, cycles=10.0, seed=7),
        )
    ),
    duration=0.8,
)
```

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

## Notebook Demo

See [`notebooks/basic_demo.ipynb`](notebooks/basic_demo.ipynb) for direct, cell-by-cell examples of the API.

## Versioning

The package version is declared in `pyproject.toml` and exposed as `visualizer.__version__`.
