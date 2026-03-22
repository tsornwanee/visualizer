from __future__ import annotations

from pathlib import Path
import sys


def find_project_root() -> Path:
    candidates = [Path.cwd(), Path.cwd().parent]
    for candidate in candidates:
        if (candidate / "src" / "visualizer").exists() and (candidate / "notebooks").exists():
            return candidate.resolve()
    raise RuntimeError("Could not find the project root.")


PROJECT_ROOT = find_project_root()
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

for extra_path in (PROJECT_ROOT / "src", NOTEBOOK_DIR):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

from tieredremuneration_support import (  # noqa: E402
    TieredRemunerationSpec,
    build_tiered_remuneration_artifact,
    build_tiered_remuneration_packet,
    write_tiered_remuneration_brief,
)


def main() -> None:
    spec = TieredRemunerationSpec()
    packet = build_tiered_remuneration_packet(spec)
    brief = write_tiered_remuneration_brief(packet)
    artifact = build_tiered_remuneration_artifact(spec, sample_step=12)

    print("=== Structured packet ===")
    print(packet.to_markdown())
    print()
    print("=== Natural-language brief ===")
    print(brief.as_text())
    print()
    print("=== Builder output ===")
    print(f"Acts: {list(artifact.bundle.acts.keys())}")
    print(f"Combined duration: {artifact.bundle.combined_schedule.total_duration:.2f}s")
    print()
    print("=== Reproduction code ===")
    print(artifact.generated_code)


if __name__ == "__main__":
    main()
