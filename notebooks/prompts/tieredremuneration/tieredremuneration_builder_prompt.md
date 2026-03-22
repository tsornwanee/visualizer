You are the builder agent for a mathematical visualization.
Use the structured packet and the natural-language brief to create numeric functions and visualization code.
Treat the animation directions as part of the specification, not as optional presentation polish.
Do not assume acts, scenes, or phases unless the brief explicitly introduces that structure.
Prefer deterministic numeric recipes over opaque hard-coded arrays when possible.

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

Tiered Remuneration Builder Brief

Start with a utility curve u(b) that is increasing and concave. Then move to a rescaled derivative u'(b), introduce the friction levels i + cb and i - cl, and use remuneration schedules to explain why non-monotone incentives can create non-local profitable deviations while monotone incentives recover a cutoff rule.

Builder instructions:
- Construct u indirectly by first designing a smooth positive decreasing u' and integrating it.
- Use u only to communicate concavity; do not shape u' in a way that implies a second message about its own curvature.
- Use rates i=0.20, cb=0.26, cl=0.04.
- Choose a non-monotone rho0 so that u' + rho0 crosses both friction lines multiple times and produces visually meaningful red/green deviation regions.
- Choose a monotone rho1 so that the borrowing and lending cutoffs are comfortably separated in x, roughly near (0.30, 0.70), so the later visualization has a clear borrower / inactive / lender partition.
- Keep the green initial-budget marker near b=0.565 and target final cutoffs near (0.405, 0.500).
- Return both the numeric recipes and the schedule code so the story can be edited later.

Animation plan:
- Start with u(b) alone, with enough pause that the audience can register increasing concavity before any derivative appears.
- Then use a split-screen style transition to explain that u is only reference context and that the decision problem is read on u'(b).
- When the friction lines i + cb and i - cl appear, pause and then move sample banks or markers so the audience sees how frictions create an inaction band rather than just reading it from the lines.
- For the non-monotone rho portion, start from a single green initial-budget marker and explicitly animate movement toward a candidate target budget level on the borrowing side.
- On that first borrowing-side move, stop early and show only the red area so the audience sees the immediate local loss created by friction near the starting point.
- Then continue moving the target farther out and reveal the green area so the audience sees how a larger move can dominate the earlier local loss and become globally profitable.
- Repeat the same target-movement logic on the lending side: begin at the same initial budget, move toward a lower target, show the early red loss first, then extend to the later green gain.
- For the monotone rho portion, place b_L and b_H with enough spacing that the borrower region, inactive region, and lender region are visually distinct.
- Use stress and pauses after each conceptual change rather than continuous motion everywhere.

Guardrails:
- Prefer deterministic analytic recipes, not random seeds or opaque sampled arrays.
- Pause after each conceptual shift so the audience can read the narration.
- Use signed fills and marker/scatter layers when they clarify the economic story.

Success checks:
- u is increasing and concave in the rendered plot.
- u' + rho0 visibly has multiple intersections with the friction thresholds.
- u' + rho1 has well-separated monotone cutoffs that are easy to read in the later visualization.

Technical targets:
- Generate deterministic numeric functions that satisfy the crossing story.
- Return an editable visualization build; if you segment it into parts, that segmentation must come from the brief rather than being assumed in advance.
- Emit recipe-based Python code that can recreate the same bundle.