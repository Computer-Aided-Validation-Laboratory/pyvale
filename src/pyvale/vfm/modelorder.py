"""Post-processing model-order selection for complete guarded BF trajectories."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def select_noise_resolved_basis_count(
    transitions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the existing cumulative noise-significance selector unchanged.

    ``transitions`` are the parent/child replay rows produced by the guarded
    EGI noise audit.  Every BF state must be solved before this retrospective
    selector is applied.
    """

    if not transitions:
        raise ValueError("Model-order selection requires at least one transition.")
    first_fail = next(
        (index for index, row in enumerate(transitions) if not row["pass"]),
        None,
    )
    first = (
        transitions[-1]["child_stage"]
        if first_fail is None
        else (
            "Phase0"
            if first_fail == 0
            else transitions[first_fail - 1]["child_stage"]
        )
    )
    consecutive = None
    for index in range(len(transitions) - 1):
        if not transitions[index]["pass"] and not transitions[index + 1]["pass"]:
            consecutive = (
                "Phase0"
                if index == 0
                else transitions[index - 1]["child_stage"]
            )
            break
    if consecutive is None:
        consecutive = transitions[-1]["child_stage"]
    later = (
        []
        if first_fail is None
        else [
            row["child_stage"]
            for row in transitions[first_fail + 1 :]
            if row["pass"]
        ]
    )
    cumulative: list[dict[str, Any]] = []
    cumulative_selected = first
    if first_fail is not None:
        reference = transitions[first_fail]
        parent_stage = str(reference["transition"]).split("→")[0]
        for row in transitions[first_fail:]:
            improvement = reference["parent_j"] - row["child_j"]
            passed = bool(improvement > reference["q95_absolute_noise_change"])
            cumulative.append({
                "transition": f"{parent_stage}→{row['child_stage']}",
                "observed_improvement": improvement,
                "q95": reference["q95_absolute_noise_change"],
                "pass": passed,
            })
            if passed:
                cumulative_selected = row["child_stage"]
    return {
        "first_fail_selected": first,
        "two_consecutive_fail_selected": consecutive,
        "later_individual_passes": later,
        "cumulative_from_last_significant": cumulative,
        "cumulative_selected": cumulative_selected,
    }


def basis_count_from_stage(stage: str) -> int:
    """Parse a selected ``BFn`` stage without supplying a fixed fallback."""

    if not stage.startswith("BF") or not stage[2:].isdigit():
        raise ValueError(
            "The noise-resolved selector did not select a heterogeneous BF state: "
            f"{stage!r}."
        )
    value = int(stage[2:])
    if value < 1:
        raise ValueError(f"Invalid selected basis stage: {stage!r}.")
    return value
