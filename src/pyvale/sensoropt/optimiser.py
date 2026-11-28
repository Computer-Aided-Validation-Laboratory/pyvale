# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Deterministic search algorithms for discrete sensor sites."""

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum, auto
from itertools import combinations
from math import comb, prod
from random import Random


class EOptStopCond(Enum):
    """Reason a placement search terminated."""

    SENSOR_COUNT_REACHED = auto()
    OBJECTIVE_TARGET_REACHED = auto()
    INVENTORY_EXHAUSTED = auto()
    SENSOR_LIMIT_REACHED = auto()
    NO_FEASIBLE_MOVE = auto()
    NO_FEASIBLE_START = auto()
    NO_IMPROVEMENT = auto()
    SEARCH_EXHAUSTED = auto()
    ITERATION_LIMIT = auto()


@dataclass(slots=True, kw_only=True)
class OptStartLayout:
    """Use these exact candidate indices, local to each named array."""

    sites: Mapping[str, tuple[int, ...]]


@dataclass(slots=True, kw_only=True)
class OptStartExhaustive:
    """Find the best feasible starting layout at the specified array counts."""

    counts: Mapping[str, int]


@dataclass(slots=True, kw_only=True)
class OptStartRandom:
    """Test up to num_layouts distinct random starts at the given array counts."""

    counts: Mapping[str, int]
    num_layouts: int


@dataclass(slots=True, kw_only=True)
class OptGreedyForward:
    """Add sites to an optional starting layout until a count or target is met."""

    num_sensors: int | None = None
    target_objective: float | None = None
    max_sensors: int | None = None
    start: OptStartLayout | OptStartExhaustive | OptStartRandom | None = None


@dataclass(slots=True, kw_only=True)
class OptGreedyBackward:
    """Remove sites from the installed layout until the target count is met."""

    num_sensors: int


@dataclass(slots=True, kw_only=True)
class OptExchange:
    """Improve the installed layout by swapping one site at a time."""

    max_iterations: int


@dataclass(slots=True, kw_only=True)
class OptExhaustive:
    """Test all permitted combinations unless an iteration limit is supplied.

    Supply an exact count or an ascending range with step one. With a target
    and no explicit count, search from one through the available inventory.
    A target prefers the smallest qualifying layout, then the lowest score.
    """

    max_iterations: int | None = None
    num_sensors: int | range | None = None
    target_objective: float | None = None


@dataclass(slots=True, kw_only=True)
class OptFixedLayout:
    """Keep every array at its configured positions and search sample times."""


OptDiscrete = (
    OptGreedyForward | OptGreedyBackward | OptExchange | OptExhaustive | OptFixedLayout
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
    optimiser: OptGreedyForward,
    *,
    num_sites: int,
    inventory: int,
    score_layout: ScoreLayout,
    start: _ScoredLayout | None = None,
) -> _SearchResult:
    current_sites = () if start is None else start.sites
    best = start
    target = optimiser.target_objective
    if start is not None:
        if len(current_sites) == optimiser.num_sensors:
            return _SearchResult(start, EOptStopCond.SENSOR_COUNT_REACHED, 0, True)
        if target is not None and start.objective <= target:
            return _SearchResult(start, EOptStopCond.OBJECTIVE_TARGET_REACHED, 0, True)

    count_limit = optimiser.num_sensors or optimiser.max_sensors or inventory
    num_additions = count_limit - len(current_sites)
    for iteration in range(1, num_additions + 1):
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
            assert target is not None
            if selected.objective <= target:
                return _SearchResult(
                    best, EOptStopCond.OBJECTIVE_TARGET_REACHED, iteration, True
                )
    stop_condition = (
        EOptStopCond.INVENTORY_EXHAUSTED
        if optimiser.max_sensors is None
        else EOptStopCond.SENSOR_LIMIT_REACHED
    )
    return _SearchResult(best, stop_condition, num_additions, False)


def _enumerate_starts(
    domains: tuple[tuple[range, int], ...], index: int = 0
) -> Iterator[Layout]:
    if index == len(domains):
        yield ()
        return
    sites, count = domains[index]
    for selection in combinations(sites, count):
        for remaining in _enumerate_starts(domains, index + 1):
            yield selection + remaining


def _sample_starts(
    domains: tuple[tuple[range, int], ...], num_layouts: int, seed: int | None
) -> Iterator[Layout]:
    pool_sizes = tuple(comb(len(sites), count) for sites, count in domains)
    total = prod(pool_sizes)
    rng = Random(seed)
    # Sample ranks without replacement using a sparse partial shuffle.
    swaps: dict[int, int] = {}
    for remaining in range(total, total - min(num_layouts, total), -1):
        draw = rng.randrange(remaining)
        rank = swaps.get(draw, draw)
        last = remaining - 1
        swaps[draw] = swaps.pop(last, last)
        if draw == last:
            del swaps[draw]

        selections: list[Layout] = []
        for (sites, count), size in zip(
            reversed(domains), reversed(pool_sizes), strict=True
        ):
            rank, local_rank = divmod(rank, size)
            selections.append(_unrank_combination(sites, count, local_rank))
        yield tuple(site for selection in reversed(selections) for site in selection)


def _unrank_combination(sites: range, count: int, rank: int) -> Layout:
    """Decode a lexicographic combination rank without storing the combinations."""
    selected: list[int] = []
    for index, site in enumerate(sites):
        if count == 0:
            break
        block = comb(len(sites) - index - 1, count - 1)
        if rank < block:
            selected.append(site)
            count -= 1
        else:
            rank -= block
    return tuple(selected)


def _search_backward(
    optimiser: OptGreedyBackward,
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

    num_removals = len(initial_sites) - optimiser.num_sensors
    for iteration in range(1, num_removals + 1):
        moves = (
            current_sites[:index] + current_sites[index + 1 :]
            for index in range(len(current_sites))
        )
        selected = _select_best_layout(moves, score_layout)
        if selected is None:
            return _SearchResult(best, EOptStopCond.NO_FEASIBLE_MOVE, iteration, False)
        best = selected
        current_sites = selected.sites
    return _SearchResult(best, EOptStopCond.SENSOR_COUNT_REACHED, num_removals, True)


def _search_exchange(
    optimiser: OptExchange,
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
    optimiser: OptExhaustive,
    *,
    num_sites: int,
    inventory: int,
    score_layout: ScoreLayout,
) -> _SearchResult:
    count = optimiser.num_sensors
    if count is None:
        counts = range(1, min(inventory, num_sites) + 1)
    elif isinstance(count, range):
        counts = count
    else:
        counts = (count,)
    layouts = (
        layout for count in counts for layout in combinations(range(num_sites), count)
    )
    best: _ScoredLayout | None = None
    qualifying: _ScoredLayout | None = None
    target = optimiser.target_objective
    num_iterations = 0
    stop_condition = EOptStopCond.SEARCH_EXHAUSTED
    for layout in layouts:
        if optimiser.max_iterations == num_iterations:
            stop_condition = EOptStopCond.ITERATION_LIMIT
            break
        num_iterations += 1
        score = score_layout(layout)
        if score is None:
            continue
        if best is None or score < best.objective:
            best = _ScoredLayout(layout, score)
        if target is not None and score <= target and (
            qualifying is None
            or (len(layout) == len(qualifying.sites) and score < qualifying.objective)
        ):
            qualifying = _ScoredLayout(layout, score)

    success = best is not None if target is None else qualifying is not None
    return _SearchResult(
        qualifying or best, stop_condition, num_iterations, success
    )
