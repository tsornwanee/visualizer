# Tiered Remuneration From-Scratch Workspace

This workspace is for a staged agent workflow:

1. The instruction maker may inspect the existing tiered-remuneration materials and produce `packet.md`.
2. The planner agent should read only `packet.md` and write `planner_brief.md`.
3. The builder agent should read only `packet.md` and `planner_brief.md`, then create the implementation in `builder_output/`.

Intended discipline:

- Do not use the old notebook or support module during the planner or builder stages.
- Work from scratch using only the files in this folder.
- Keep the implementation deterministic and editable.
- The planner defines the animation structure. The builder should not assume structural labels that are absent from `planner_brief.md`.

Important limitation:

- This repository setup cannot technically sandbox file reads for local agents.
- Operationally, enforce isolation by spawning planner and builder agents without forked context and providing only these files.
