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