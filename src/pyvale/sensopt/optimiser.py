"""Deterministic search algorithms for discrete sensor sites."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum, auto
from itertools import combinations, islice
from math import comb


class EOptStopCond(Enum):
    """Reason a placement search terminated."""

    SENSOR_COUNT_REACHED = auto()
    OBJECTIVE_TARGET_REACHED = auto()
    INVENTORY_EXHAUSTED = auto()
    NO_FEASIBLE_MOVE = auto()
    NO_IMPROVEMENT = auto()
    SEARCH_EXHAUSTED = auto()
    ITERATION_LIMIT = auto()


@dataclass(slots=True, kw_only=True)
class OptimiserGreedyForward:
    """Add sites from an empty layout until a count or objective target is met."""

    max_iterations: int
    num_sensors: int | None = None
    target_objective: float | None = None


@dataclass(slots=True, kw_only=True)
class OptimiserGreedyBackward:
    """Remove sites from the installed layout until the target count is met."""

    max_iterations: int
    num_sensors: int


@dataclass(slots=True, kw_only=True)
class OptimiserExchange:
    """Improve the installed layout by swapping one site at a time."""

    max_iterations: int


@dataclass(slots=True, kw_only=True)
class OptimiserExhaustive:
    """Test fixed-count combinations in candidate order."""

    max_iterations: int
    num_sensors: int


DiscreteOptimiser = (
    OptimiserGreedyForward
    | OptimiserGreedyBackward
    | OptimiserExchange
    | OptimiserExhaustive
)
Layout = tuple[int, ...]
ScoreLayout = Callable[[Layout], float | None]


@dataclass(slots=True)
class _ScoredLayout:
    sites: Layout
    objective: float


@dataclass(slots=True)
class _SearchResult:
    layout: _ScoredLayout | None
    stop_condition: EOptStopCond
    num_iterations: int
    success: bool


def _select_best_layout(
    layouts: Iterable[Layout], score_layout: ScoreLayout
) -> _ScoredLayout | None:
    best: _ScoredLayout | None = None
    for layout in layouts:
        objective = score_layout(layout)
        if objective is not None and (best is None or objective < best.objective):
            best = _ScoredLayout(layout, objective)
    return best


def _search_forward(
    optimiser: OptimiserGreedyForward,
    *,
    num_sites: int,
    inventory: int,
    score_layout: ScoreLayout,
) -> _SearchResult:
    current_sites: Layout = ()
    best: _ScoredLayout | None = None

    for iteration in range(1, optimiser.max_iterations + 1):
        moves = (
            tuple(sorted((*current_sites, site)))
            for site in range(num_sites)
            if site not in current_sites
        )
        selected = _select_best_layout(moves, score_layout)
        if selected is None:
            return _SearchResult(best, EOptStopCond.NO_FEASIBLE_MOVE, iteration, False)

        current_sites = selected.sites
        if optimiser.num_sensors is not None:
            best = selected  # Prefer progress toward the requested count.
            if len(current_sites) == optimiser.num_sensors:
                return _SearchResult(
                    best, EOptStopCond.SENSOR_COUNT_REACHED, iteration, True
                )
        else:
            if best is None or selected.objective < best.objective:
                best = selected
            target = optimiser.target_objective
            assert target is not None
            if selected.objective <= target:
                return _SearchResult(
                    best, EOptStopCond.OBJECTIVE_TARGET_REACHED, iteration, True
                )
        if len(current_sites) == inventory:
            return _SearchResult(
                best, EOptStopCond.INVENTORY_EXHAUSTED, iteration, False
            )

    return _SearchResult(
        best, EOptStopCond.ITERATION_LIMIT, optimiser.max_iterations, False
    )


def _search_backward(
    optimiser: OptimiserGreedyBackward,
    *,
    initial_sites: Layout,
    score_layout: ScoreLayout,
) -> _SearchResult:
    current_sites = initial_sites
    best = _select_best_layout((current_sites,), score_layout)
    if len(current_sites) == optimiser.num_sensors:
        if best is None:
            return _SearchResult(None, EOptStopCond.NO_FEASIBLE_MOVE, 0, False)
        return _SearchResult(best, EOptStopCond.SENSOR_COUNT_REACHED, 0, True)

    for iteration in range(1, optimiser.max_iterations + 1):
        moves = (
            current_sites[:index] + current_sites[index + 1 :]
            for index in range(len(current_sites))
        )
        selected = _select_best_layout(moves, score_layout)
        if selected is None:
            return _SearchResult(best, EOptStopCond.NO_FEASIBLE_MOVE, iteration, False)
        best = selected
        current_sites = selected.sites
        if len(current_sites) == optimiser.num_sensors:
            return _SearchResult(
                best, EOptStopCond.SENSOR_COUNT_REACHED, iteration, True
            )

    return _SearchResult(
        best, EOptStopCond.ITERATION_LIMIT, optimiser.max_iterations, False
    )


def _search_exchange(
    optimiser: OptimiserExchange,
    *,
    initial_sites: Layout,
    num_sites: int,
    score_layout: ScoreLayout,
) -> _SearchResult:
    current_sites = initial_sites
    best = _select_best_layout((current_sites,), score_layout)

    for iteration in range(1, optimiser.max_iterations + 1):
        moves = (
            tuple(sorted((*current_sites[:index], *current_sites[index + 1 :], site)))
            for site in range(num_sites)
            if site not in current_sites
            for index in range(len(current_sites))
        )
        selected = _select_best_layout(moves, score_layout)
        if selected is None and best is None:
            return _SearchResult(None, EOptStopCond.NO_FEASIBLE_MOVE, iteration, False)
        if selected is None or (
            best is not None and selected.objective >= best.objective
        ):
            return _SearchResult(best, EOptStopCond.NO_IMPROVEMENT, iteration, True)

        best = selected
        current_sites = selected.sites

    return _SearchResult(
        best, EOptStopCond.ITERATION_LIMIT, optimiser.max_iterations, True
    )


def _search_exhaustive(
    optimiser: OptimiserExhaustive,
    *,
    num_sites: int,
    score_layout: ScoreLayout,
) -> _SearchResult:
    total_combinations = comb(num_sites, optimiser.num_sensors)
    num_iterations = min(total_combinations, optimiser.max_iterations)
    layouts = combinations(range(num_sites), optimiser.num_sensors)
    best = _select_best_layout(islice(layouts, num_iterations), score_layout)
    stop_condition = (
        EOptStopCond.SEARCH_EXHAUSTED
        if num_iterations == total_combinations
        else EOptStopCond.ITERATION_LIMIT
    )
    return _SearchResult(best, stop_condition, num_iterations, best is not None)
