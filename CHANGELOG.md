# Changelog

All notable changes to this project will be documented in this file.

The format follows Keep a Changelog, and the project aims to follow Semantic Versioning.

## [Unreleased]

## [0.1.0] - 2026-03-21

### Added

- Core `Scene` / `Transition` / `Schedule` model for graph animation.
- Persistent curve, fill-between, and text elements.
- Draw, move, erase, style, pause, stress, and jitter transitions.
- Parallel transitions with shared draw timing across curve domains.
- Per-element clipping via `domain` and `value_range`.
- Modular act composition with `final_scene`, `next_act()`, and `Schedule.combine(...)`.
- Demo notebooks covering the current API surface.
- Packaging metadata, release documentation, and CI packaging checks.
