from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class CurveInstruction:
    name: str
    role: str
    qualitative_properties: tuple[str, ...] = ()
    numeric_goals: tuple[str, ...] = ()
    presentation_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "qualitative_properties": list(self.qualitative_properties),
            "numeric_goals": list(self.numeric_goals),
            "presentation_notes": list(self.presentation_notes),
        }


@dataclass(frozen=True)
class LevelInstruction:
    name: str
    expression: str
    purpose: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expression": self.expression,
            "purpose": self.purpose,
        }


@dataclass(frozen=True)
class IntersectionInstruction:
    left_curve: str
    right_curve: str
    desired_count: int | str
    importance: str
    location_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "left_curve": self.left_curve,
            "right_curve": self.right_curve,
            "desired_count": self.desired_count,
            "importance": self.importance,
        }
        if self.location_hint is not None:
            payload["location_hint"] = self.location_hint
        return payload


@dataclass(frozen=True)
class NarrativeBeatInstruction:
    title: str
    goal: str
    emphasis: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "goal": self.goal,
            "emphasis": list(self.emphasis),
        }


@dataclass(frozen=True)
class VisualizationInstructionPacket:
    title: str
    synopsis: str
    target_audience: str
    curves: tuple[CurveInstruction, ...] = ()
    levels: tuple[LevelInstruction, ...] = ()
    intersections: tuple[IntersectionInstruction, ...] = ()
    narrative_beats: tuple[NarrativeBeatInstruction, ...] = ()
    animation_directives: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "synopsis": self.synopsis,
            "target_audience": self.target_audience,
            "curves": [curve.to_dict() for curve in self.curves],
            "levels": [level.to_dict() for level in self.levels],
            "intersections": [intersection.to_dict() for intersection in self.intersections],
            "narrative_beats": [beat.to_dict() for beat in self.narrative_beats],
            "animation_directives": list(self.animation_directives),
            "constraints": list(self.constraints),
            "metadata": dict(self.metadata),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        lines = [f"## {self.title}", "", self.synopsis, "", f"Audience: {self.target_audience}"]

        if self.curves:
            lines.extend(["", "Curves:"])
            for curve in self.curves:
                lines.append(f"- `{curve.name}` ({curve.role})")
                for item in curve.qualitative_properties:
                    lines.append(f"  - property: {item}")
                for item in curve.numeric_goals:
                    lines.append(f"  - numeric goal: {item}")
                for item in curve.presentation_notes:
                    lines.append(f"  - presentation note: {item}")

        if self.levels:
            lines.extend(["", "Levels:"])
            for level in self.levels:
                lines.append(f"- `{level.name}` = {level.expression}: {level.purpose}")

        if self.intersections:
            lines.extend(["", "Intersection goals:"])
            for intersection in self.intersections:
                line = (
                    f"- `{intersection.left_curve}` vs `{intersection.right_curve}`: "
                    f"{intersection.desired_count} intersection(s), {intersection.importance}"
                )
                if intersection.location_hint is not None:
                    line += f" ({intersection.location_hint})"
                lines.append(line)

        if self.narrative_beats:
            lines.extend(["", "Narrative beats:"])
            for beat in self.narrative_beats:
                lines.append(f"- {beat.title}: {beat.goal}")

        if self.animation_directives:
            lines.extend(["", "Animation directives:"])
            lines.extend(f"- {directive}" for directive in self.animation_directives)

        if self.constraints:
            lines.extend(["", "Constraints:"])
            lines.extend(f"- {constraint}" for constraint in self.constraints)

        return "\n".join(lines)

    def planner_prompt(self) -> str:
        return "\n".join(
            [
                "You are the planning agent for a mathematical visualization.",
                "Read the structured packet below and write a concise natural-language brief for the builder agent.",
                "The brief should explain the story, what numerical shapes matter, what should not be over-claimed, how the animation should unfold, and what the audience should notice at each stage.",
                "Do not invent structural labels such as acts, scenes, or phases unless the packet itself justifies them.",
                "",
                self.to_markdown(),
            ]
        )


@dataclass(frozen=True)
class NarrativeBrief:
    title: str
    summary: str
    builder_instructions: tuple[str, ...]
    animation_plan: tuple[str, ...] = ()
    guardrails: tuple[str, ...] = ()
    success_checks: tuple[str, ...] = ()

    def as_text(self) -> str:
        lines = [self.title, "", self.summary]

        if self.builder_instructions:
            lines.extend(["", "Builder instructions:"])
            lines.extend(f"- {instruction}" for instruction in self.builder_instructions)

        if self.animation_plan:
            lines.extend(["", "Animation plan:"])
            lines.extend(f"- {step}" for step in self.animation_plan)

        if self.guardrails:
            lines.extend(["", "Guardrails:"])
            lines.extend(f"- {guardrail}" for guardrail in self.guardrails)

        if self.success_checks:
            lines.extend(["", "Success checks:"])
            lines.extend(f"- {check}" for check in self.success_checks)

        return "\n".join(lines)

    def as_markdown(self) -> str:
        return self.as_text()

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "builder_instructions": list(self.builder_instructions),
            "animation_plan": list(self.animation_plan),
            "guardrails": list(self.guardrails),
            "success_checks": list(self.success_checks),
        }


@dataclass(frozen=True)
class BuilderHandoff:
    packet: VisualizationInstructionPacket
    brief: NarrativeBrief
    technical_targets: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet": self.packet.to_dict(),
            "brief": self.brief.to_dict(),
            "technical_targets": list(self.technical_targets),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def builder_prompt(self) -> str:
        return "\n".join(
            [
                "You are the builder agent for a mathematical visualization.",
                "Use the structured packet and the natural-language brief to create numeric functions and visualization code.",
                "Treat the animation directions as part of the specification, not as optional presentation polish.",
                "Do not assume acts, scenes, or phases unless the brief explicitly introduces that structure.",
                "Prefer deterministic numeric recipes over opaque hard-coded arrays when possible.",
                "",
                self.packet.to_markdown(),
                "",
                self.brief.as_markdown(),
                "",
                "Technical targets:",
                *[f"- {target}" for target in self.technical_targets],
            ]
        )


@dataclass(frozen=True)
class VisualizationBuildArtifact:
    packet: VisualizationInstructionPacket
    brief: NarrativeBrief
    handoff: BuilderHandoff
    generated_code: str
    notes: tuple[str, ...] = ()


class VisualizationBuilder(ABC):
    @abstractmethod
    def build(
        self,
        packet: VisualizationInstructionPacket,
        brief: NarrativeBrief,
    ) -> VisualizationBuildArtifact:
        raise NotImplementedError


@dataclass
class VisualizationPipeline:
    brief_writer: Callable[[VisualizationInstructionPacket], NarrativeBrief]
    builder: VisualizationBuilder

    def run(self, packet: VisualizationInstructionPacket) -> VisualizationBuildArtifact:
        brief = self.brief_writer(packet)
        return self.builder.build(packet, brief)


def export_instruction_prompts(
    packet: VisualizationInstructionPacket,
    *,
    brief: NarrativeBrief | None = None,
    handoff: BuilderHandoff | None = None,
    output_dir: str | Path,
    prefix: str = "visualization",
) -> dict[str, Path]:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    packet_path = resolved_output_dir / f"{prefix}_packet.md"
    planner_prompt_path = resolved_output_dir / f"{prefix}_planner_prompt.md"
    exports: dict[str, Path] = {
        "packet": packet_path,
        "planner_prompt": planner_prompt_path,
    }

    packet_path.write_text(packet.to_markdown(), encoding="utf-8")
    planner_prompt_path.write_text(packet.planner_prompt(), encoding="utf-8")

    if brief is not None:
        brief_path = resolved_output_dir / f"{prefix}_brief.md"
        brief_path.write_text(brief.as_markdown(), encoding="utf-8")
        exports["brief"] = brief_path

    if handoff is not None:
        builder_prompt_path = resolved_output_dir / f"{prefix}_builder_prompt.md"
        builder_prompt_path.write_text(handoff.builder_prompt(), encoding="utf-8")
        exports["builder_prompt"] = builder_prompt_path

    return exports
