"""Workflow case and parameter-design data."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import product

import numpy as np


class EParameterKind(Enum):
    """Semantic kind of a workflow parameter."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


@dataclass(frozen=True, slots=True)
class ParameterValues:
    """Named discrete values accepted by a workflow parameter."""

    name: str
    kind: EParameterKind
    values: tuple[object, ...]

    def __post_init__(self) -> None:
        """Reject empty parameter definitions."""
        if not self.name or not self.values:
            raise ValueError("Workflow parameters require a name and values.")


@dataclass(frozen=True, slots=True)
class WorkflowCase:
    """One deterministic workflow parameter assignment."""

    index: int
    values: Mapping[str, object]
    seed: int


class ICaseDesign(ABC):
    """Interface for deterministic workflow case designs.

    Implementations describe their number of cases without executing a
    workflow, then generate repeatable :class:`WorkflowCase` instances from a
    supplied root seed.
    """

    @abstractmethod
    def count(self) -> int:
        """Return the number of cases represented by this design."""

    @abstractmethod
    def generate(self, seed: int) -> Iterator[WorkflowCase]:
        """Yield deterministic cases derived from ``seed``."""


class FullFactorial(ICaseDesign):
    """Lazily generate a deterministic Cartesian product of parameter values."""

    def __init__(self, parameters: Sequence[ParameterValues]) -> None:
        """Store parameter definitions."""
        self.parameters = tuple(parameters)

    def count(self) -> int:
        """Return the number of cases without expanding them."""
        count = 1
        for parameter in self.parameters:
            count *= len(parameter.values)
        return count

    def generate(self, seed: int) -> Iterator[WorkflowCase]:
        """Yield deterministic cases with independent child seeds."""
        seed_sequence = np.random.SeedSequence(seed)
        child_seeds = seed_sequence.spawn(self.count())
        values_product = product(*(item.values for item in self.parameters))
        for index, values in enumerate(values_product):
            case_values = {
                parameter.name: value
                for parameter, value in zip(self.parameters, values)
            }
            yield WorkflowCase(
                index=index,
                values=case_values,
                seed=int(child_seeds[index].generate_state(1)[0]),
            )


class ExplicitCases(ICaseDesign):
    """Generate cases from user-supplied parameter combinations."""

    def __init__(self, values: Sequence[Mapping[str, object]]) -> None:
        """Store immutable copies of supplied combinations."""
        self.values = tuple(dict(item) for item in values)

    def count(self) -> int:
        """Return the number of supplied cases."""
        return len(self.values)

    def generate(self, seed: int) -> Iterator[WorkflowCase]:
        """Yield supplied combinations with deterministic child seeds."""
        child_seeds = np.random.SeedSequence(seed).spawn(self.count())
        for index, values in enumerate(self.values):
            yield WorkflowCase(
                index=index,
                values=values,
                seed=int(child_seeds[index].generate_state(1)[0]),
            )


class RandomSampling(ICaseDesign):
    """Generate numeric or categorical values from user-supplied samplers."""

    def __init__(
        self,
        samplers: Mapping[str, Callable[[np.random.Generator], object]],
        count: int,
    ) -> None:
        """Store a sampler per parameter and the requested case count."""
        if count <= 0:
            raise ValueError("RandomSampling count must be positive.")
        self.samplers = dict(samplers)
        self.sample_count = count

    def count(self) -> int:
        """Return the requested number of cases."""
        return self.sample_count

    def generate(self, seed: int) -> Iterator[WorkflowCase]:
        """Yield deterministic cases from samplers accepting a NumPy generator."""
        seed_sequence = np.random.SeedSequence(seed)
        child_seeds = seed_sequence.spawn(self.sample_count)
        random_generator = np.random.default_rng(seed)
        for index, child_seed in enumerate(child_seeds):
            values = {
                name: sampler(random_generator)
                for name, sampler in self.samplers.items()
            }
            yield WorkflowCase(
                index=index,
                values=values,
                seed=int(child_seed.generate_state(1)[0]),
            )
