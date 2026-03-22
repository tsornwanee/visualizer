You are the planner agent for a from-scratch visualization build.
Use only `packet.md` in this workspace.
Do not inspect the original tiered-remuneration notebook, support file, or previous implementation.
Your job is to turn the packet into a detailed natural-language brief that the builder can follow from scratch.
The brief must explain the mathematical story, the animation beats, the purpose of the pauses, and how friction should be visualized when moving from one budget level to another.
Do not introduce structural labels such as acts, scenes, or phases unless you explicitly decide that they help the builder and you state them in the brief.
Write your output into `planner_brief.md`.

## Tiered Remuneration

Create a tiered-remuneration visualization that starts from an increasing concave utility, moves to its derivative, and uses remuneration schedules to explain local versus non-local deviations.

Audience: technical audience seeing the mechanism for the first time

Curves:
- `u` (utility)
  - property: increasing
  - property: concave
  - numeric goal: normalize to [0, 1]
  - numeric goal: keep visually smooth
  - presentation note: show u on its own before introducing u'
  - presentation note: use u mainly to communicate concavity
- `uprime` (derivative)
  - property: positive
  - property: decreasing
  - numeric goal: derive from u after rescaling
  - presentation note: do not over-claim convexity or concavity of u'
- `rho0` (remuneration)
  - property: non-monotone
  - numeric goal: make uprime + rho0 cross i + cb about 3 times
  - numeric goal: make uprime + rho0 cross i - cl about 3 times
  - presentation note: use this to create non-local deviation incentives
- `rho1` (remuneration)
  - property: monotone
  - numeric goal: place the monotone cutoffs with clear horizontal spacing, roughly near x = 0.30 and x = 0.70
  - presentation note: use this to recover clean cutoffs
  - presentation note: single crossings come from monotonicity; the real design goal is visual separation

Levels:
- `i` = i: reference interbank rate
- `i_plus_cb` = i + cb: upper borrowing threshold
- `i_minus_cl` = i - cl: lower lending threshold

Intersection goals:
- `uprime` vs `i_plus_cb`: 1 intersection(s), create one side of the friction band under monotone incentives (roughly left of the central budget region)
- `uprime` vs `i_minus_cl`: 1 intersection(s), create the other side of the friction band under monotone incentives (roughly right of the central budget region)
- `uprime_plus_rho0` vs `i_plus_cb`: 3 intersection(s), show that non-monotone remuneration can create multiple borrowing incentives
- `uprime_plus_rho0` vs `i_minus_cl`: 3 intersection(s), show that non-monotone remuneration can create multiple lending incentives
- `uprime_plus_rho1` vs `i_plus_cb`: 1 intersection(s), place the borrowing cutoff where the later band visualization stays readable (roughly near x = 0.30)
- `uprime_plus_rho1` vs `i_minus_cl`: 1 intersection(s), place the lending cutoff far enough away to leave a clear inactive region (roughly near x = 0.70)

Narrative beats:
- Beat 1: Introduce u on its own.
- Beat 2: Move to u' and add the friction band.
- Beat 3: Use a non-monotone rho to create multiple deviation incentives.
- Beat 4: Contrast with a monotone rho.

Animation directives:
- Start with a quiet single-curve view of u(b), then pause long enough for the audience to read that u is increasing and concave.
- When moving from u to u'(b), make the change spatially obvious: keep u as a temporary reference in the upper panel, then expand u' back to the main panel.
- Show friction by comparing movement from one budget level to another, not just by drawing horizontal lines.
- Use a visible start marker for the bank's current budget and a visible target direction along the x-axis when discussing deviations.
- For a small move away from the initial budget, first reveal the red loss area where the friction-adjusted threshold dominates the effective marginal value.
- Then extend the target farther and reveal the green gain area, so the audience sees how a larger move can dominate the earlier local loss.
- Pause after the red-only stage and again after the red-plus-green stage so the audience can compare local and non-local incentives.
- Repeat the same logic for lending-side downward movement, so the symmetry of the argument is visible.
- For the monotone rho case, emphasize that the important feature is a wide readable gap between the borrower cutoff and lender cutoff, not merely that each line is crossed once.
- Use bank markers or scatter layers to show how banks move toward a point or into a band after the cutoffs are identified.

Constraints:
- Only use u to communicate concavity; do not impose extra curvature claims on u'.
- Prefer deterministic recipe-based functions over opaque hard-coded arrays.
- Keep the numerical functions easy to tune for future stories.
- Do not assume any decomposition into acts, scenes, or phases unless it is introduced explicitly in the planning brief.