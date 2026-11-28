# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Physics sampling and ordered reductions of virtual experiment losses."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from numbers import Integral

import numpy as np
import numpy.typing as npt

from pyvale.dataio.simdata import SimData


class ISimSampler(ABC):
    """Configure and draw one batch of physics scenarios per optimisation."""

    num_sim_scenarios: int

    @abstractmethod
    def sample(self, *, rng: np.random.Generator) -> tuple[SimData, ...]:
        """Return num_sim_scenarios cases in draw order."""


@dataclass(slots=True, kw_only=True)
class SimSamplerLibrary(ISimSampler):
    """Sample a named library uniformly or using relative case weights.

    With replacement, a case may appear several times in a batch. Each draw
    counts equally in subsequent evaluation reductions by default.
    """

    library: Mapping[str, SimData]
    num_sim_scenarios: int
    weights: Mapping[str, float] | None = None
    replace: bool = True

    def sample(self, *, rng: np.random.Generator) -> tuple[SimData, ...]:
        count = self.num_sim_scenarios
        if not isinstance(count, Integral) or isinstance(count, bool) or count < 1:
            raise ValueError("num_sim_scenarios must be a positive integer.")
        if not self.library:
            raise ValueError("The simulation library must contain at least one case.")

        keys = tuple(self.library)
        weights = _get_scenario_weights(self.weights, keys)
        eligible = len(keys) if weights is None else np.count_nonzero(weights)
        if not self.replace and count > eligible:
            raise ValueError(
                "Too few eligible library cases to sample without replacement."
            )
        probabilities = None if weights is None else weights / weights.sum()
        indices = rng.choice(
            len(keys), size=int(count), replace=self.replace, p=probabilities
        )
        return tuple(self.library[keys[index]] for index in indices)


class ESampling(Enum):
    """How virtual experiments are allocated to physics scenarios."""

    NESTED = auto()
    PAIRED = auto()


class EEvalAxis(Enum):
    """Named axes of an objective's loss array."""

    SIM_SCENARIO = 0
    VIRTUAL_EXP = 1


class EEvalReduce(Enum):
    """Statistics for reducing losses along an evaluation axis."""

    MEAN = auto()
    MAX = auto()
    MEDIAN = auto()


@dataclass(slots=True, kw_only=True)
class EvalReduction:
    """One reduction step. Weights apply to a mean over named scenarios."""

    axis: EEvalAxis
    method: EEvalReduce
    weights: Mapping[str, float] | None = None


@dataclass(slots=True, kw_only=True)
class EvalStrat:
    """Ordered reductions over a loss array shaped (sim_scenario, virtual_exp)."""

    reductions: tuple[EvalReduction, ...]

    def reduce(
        self,
        losses: npt.NDArray[np.float64],
        *,
        scenario_keys: tuple[str, ...],
    ) -> npt.NDArray[np.float64]:
        if losses.ndim == 0:
            return losses
        if losses.ndim != 2 or losses.shape[0] != len(scenario_keys) or not losses.size:
            raise ValueError("Losses must have shape (sim_scenario, virtual_exp).")

        reduced_axes: set[EEvalAxis] = set()

        for step in self.reductions:
            if step.axis in reduced_axes:
                raise ValueError("Each evaluation axis can be reduced only once.")
            reduced_axes.add(step.axis)

            axis = step.axis.value
            if step.weights is not None and (
                step.axis is not EEvalAxis.SIM_SCENARIO
                or step.method is not EEvalReduce.MEAN
            ):
                raise ValueError("Weights require a mean over SIM_SCENARIO.")

            match step.method:
                case EEvalReduce.MEAN:
                    weights = _get_scenario_weights(step.weights, scenario_keys)
                    losses = np.average(
                        losses, axis=axis, weights=weights, keepdims=True
                    )
                case EEvalReduce.MAX:
                    losses = np.max(losses, axis=axis, keepdims=True)
                case EEvalReduce.MEDIAN:
                    losses = np.median(losses, axis=axis, keepdims=True)
                case _:
                    raise ValueError(
                        f"Unsupported evaluation reduction {step.method!r}."
                    )

        return losses


def _get_scenario_weights(
    weights: Mapping[str, float] | None, scenario_keys: tuple[str, ...]
) -> npt.NDArray[np.float64] | None:
    if weights is None:
        return None

    if set(weights) != set(scenario_keys):
        raise ValueError("Scenario weights must match the simulation scenario keys.")

    values = np.array([weights[key] for key in scenario_keys], dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or not np.any(values > 0):
        raise ValueError(
            "Scenario weights must be finite, nonnegative and not all zero."
        )
    return values / np.max(values)
