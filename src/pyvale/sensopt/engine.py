# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Evaluate discrete placements on engine-owned sensor arrays."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np
import numpy.typing as npt

from pyvale.dataio.simdata import SimData
from pyvale.sensopt.constraintout import ConstrainObjVal
from pyvale.sensopt.designspace import DesignSpace, EDesignAxis
from pyvale.sensopt.evalstrat import ESampling, ISimSampler
from pyvale.sensopt.objective import (
    IObjFunc,
    ObjFieldRecon,
    ObjSignal,
    ObjSignalPeakAbs,
    ObjSignalRMS,
    ObjSpec,
    ObjTerm,
    ObjWeighted,
)
from pyvale.sensopt.optimiser import (
    EOptStopCond,
    Layout,
    OptDiscrete,
    OptExchange,
    OptExhaustive,
    OptGreedyBackward,
    OptGreedyForward,
    OptStartExhaustive,
    OptStartLayout,
    _enumerate_starts,
    _sample_starts,
    _ScoredLayout,
    _search_backward,
    _search_exchange,
    _search_exhaustive,
    _search_forward,
    _SearchResult,
)
from pyvale.sensopt.simscenario import _prepare_objective
from pyvale.sensorsim.experimentsimulator import ExperimentSimulator
from pyvale.sensorsim.sensorarray import ISensorArray
from pyvale.sensorsim.sensordata import SensorData

_NOMINAL_SCENARIO = "nominal"


@dataclass(slots=True)
class OptResult:
    """Best layout found and the reason the optimiser stopped.

    Objective values contain the single score used for search. An empty layout
    and score mean no feasible candidate was evaluated. Success means the layout
    met the requested count or objective target. An iteration-limited exchange or
    exhaustive search can return a feasible layout with success and an incomplete
    search reported by the stop condition.

    num_start_layouts counts layouts considered before greedy additions,
    including rejected layouts. num_iterations counts the subsequent rounds.
    """

    sensor_data: dict[str, SensorData]
    objective_values: npt.NDArray[np.float64]
    stop_condition: EOptStopCond
    num_iterations: int
    success: bool
    num_start_layouts: int = 0


@dataclass(slots=True, kw_only=True)
class SensorOptEngine:
    """Optimise named arrays over discrete sites in already-bound fields.

    Each array's configured count is its inventory limit. Backward and exchange
    start at the configured positions. Each candidate gets one scalar score and
    virtual experiments for each simulation scenario. Unselected arrays are omitted.
    """

    sensor_arrays: Mapping[str, ISensorArray]
    design_space: DesignSpace
    objectives: tuple[IObjFunc | ObjSpec, ...] | ObjWeighted
    optimiser: OptDiscrete
    seed: int | None = None
    sim_scenarios: Mapping[str, SimData] | ISimSampler | None = None
    num_sim_scenarios: int | None = None
    num_virtual_exps: int = 1
    sampling: ESampling = ESampling.NESTED

    def optimise(self) -> OptResult:
        """Search the configured design space without mutating any supplied array."""
        if not self.sensor_arrays:
            raise ValueError("Supply at least one sensor array.")
        if self.design_space.positions.keys() != self.sensor_arrays.keys():
            raise ValueError("Position domains must match the supplied sensor arrays.")
        if self.design_space.position_axes != dict.fromkeys(
            self.sensor_arrays, EDesignAxis.DISC
        ):
            raise ValueError("This engine requires discrete position domains.")
        optimiser = self.optimiser
        nominal_data: dict[str, SensorData] = {}
        inventories: dict[str, int] = {}
        array_sites: list[npt.NDArray[np.float64]] = []
        site_domains: dict[str, range] = {}
        site_keys: list[str] = []
        installed_sites: list[int] = []
        available_count = 0
        for key, array in self.sensor_arrays.items():
            sites = np.asarray(self.design_space.positions[key], dtype=np.float64)
            if sites.ndim != 2 or sites.shape[1] != 3 or not np.all(np.isfinite(sites)):
                raise ValueError(f"Candidate sites for {key!r} need finite XYZ rows.")
            if len(sites) == 0 or len(np.unique(sites, axis=0)) != len(sites):
                raise ValueError(f"The position domain for {key!r} needs unique sites.")
            data = array.get_sensor_data()
            positions = data.positions
            if (
                positions is None
                or positions.ndim != 2
                or positions.shape[1] != 3
                or len(positions) == 0
                or not np.all(np.isfinite(positions))
            ):
                raise ValueError(f"Array {key!r} needs finite nominal XYZ positions.")
            if data.angles is not None and len(data.angles) != 1:
                raise ValueError("Discrete placement requires a shared rotation per array.")
            if isinstance(optimiser, (OptGreedyBackward, OptExchange)):
                site_indices = {
                    tuple(position): index for index, position in enumerate(sites)
                }
                array_installed: set[int] = set()
                for position in positions:
                    site_index = site_indices.get(tuple(position))
                    if site_index is None or site_index in array_installed:
                        raise ValueError(
                            f"Installed positions for {key!r} must match distinct sites."
                        )
                    array_installed.add(site_index)
                installed_sites.extend(
                    len(site_keys) + index for index in array_installed
                )
            nominal_data[key] = data
            inventories[key] = len(positions)
            available_count += min(len(sites), len(positions))
            site_domains[key] = range(len(site_keys), len(site_keys) + len(sites))
            site_keys.extend([key] * len(sites))
            array_sites.append(sites)

        all_sites = np.concatenate(array_sites)
        initial_sites: Layout = tuple(sorted(installed_sites))
        if isinstance(optimiser, (OptExchange, OptExhaustive)):
            limit = optimiser.max_iterations
            if (limit is not None or isinstance(optimiser, OptExchange)) and (
                not _is_integer(limit, minimum=1)
            ):
                raise ValueError("max_iterations must be a positive integer.")
        if self.seed is not None and not _is_integer(self.seed, minimum=0):
            raise ValueError("seed must be a nonnegative integer.")

        if isinstance(optimiser, OptGreedyForward):
            if (optimiser.num_sensors is None) == (optimiser.target_objective is None):
                raise ValueError(
                    "Specify exactly one of num_sensors and target_objective."
                )
            max_sensors = optimiser.max_sensors
            if max_sensors is not None:
                if optimiser.target_objective is None:
                    raise ValueError("max_sensors requires target_objective.")
                if (
                    not _is_integer(max_sensors, minimum=1)
                    or max_sensors > available_count
                ):
                    raise ValueError("max_sensors must be within the available count.")
        if (
            isinstance(optimiser, OptExhaustive)
            and optimiser.num_sensors is None
            and optimiser.target_objective is None
        ):
            raise ValueError("Specify num_sensors or target_objective.")
        if isinstance(optimiser, (OptGreedyForward, OptExhaustive)):
            target = optimiser.target_objective
            if target is not None and not np.isfinite(target):
                raise ValueError("target_objective must be finite.")
        if not isinstance(optimiser, OptExchange):
            count = optimiser.num_sensors
            if isinstance(optimiser, OptGreedyBackward) and count is None:
                raise ValueError("This optimiser requires num_sensors.")
            if isinstance(count, range):
                if count.step != 1 or count.start >= count.stop:
                    raise ValueError("num_sensors must be a nonempty range with step 1.")
                if count.start < 1 or count.stop - 1 > available_count:
                    raise ValueError(
                        "num_sensors must be within the inventory and site count."
                    )
            elif count is not None:
                if not _is_integer(count, minimum=1):
                    raise ValueError("num_sensors must be a positive integer.")
                if count > available_count:
                    raise ValueError(
                        "num_sensors must be within the inventory and site count."
                    )

        if isinstance(self.objectives, ObjWeighted):
            terms = self.objectives.terms
            if any(not np.isfinite(term.weight) or term.weight < 0 for term in terms):
                raise ValueError("Objective weights must be finite and nonnegative.")
            terms = tuple(term for term in terms if term.weight > 0)
            if not terms:
                raise ValueError("A weighted objective requires a positive weight.")
        else:
            if len(self.objectives) != 1:
                raise ValueError("Supply one objective or an ObjWeighted combination.")
            terms = (ObjTerm(objective=self.objectives[0]),)

        if not _is_integer(self.num_virtual_exps, minimum=1):
            raise ValueError("num_virtual_exps must be a positive integer.")
        if not isinstance(self.sampling, ESampling):
            raise ValueError("sampling must be an ESampling member.")
        if self.sampling is ESampling.PAIRED and self.num_virtual_exps != 1:
            raise ValueError("PAIRED sampling requires num_virtual_exps=1.")
        sim_scenarios: dict[str, SimData] = {}
        physics_seed = np.random.SeedSequence(self.seed)
        source = self.sim_scenarios
        if isinstance(source, ISimSampler):
            count = self.num_sim_scenarios
            if count is None or not _is_integer(count, minimum=1):
                raise ValueError("A sampler requires a positive num_sim_scenarios.")
            rng = np.random.default_rng(physics_seed.spawn(1)[0])
            sim_scenarios = {f"sample_{index}": source.sample(rng=rng) for index in range(count)}
            if not all(isinstance(case, SimData) for case in sim_scenarios.values()):
                raise ValueError("The simulation sampler must return SimData.")
        else:
            if self.num_sim_scenarios is not None:
                raise ValueError("num_sim_scenarios requires an ISimSampler.")
            if source is not None:
                if not source:
                    raise ValueError("Supply at least one named simulation scenario.")
                sim_scenarios = dict(source)

        scenario_keys = tuple(sim_scenarios) or (_NOMINAL_SCENARIO,)
        objective_specs = tuple(
            term.objective if isinstance(term.objective, ObjSpec) else None
            for term in terms
        )
        resolved_objectives = tuple(
            _prepare_objective(
                term.objective,
                sim_scenarios=sim_scenarios,
                sensor_arrays=self.sensor_arrays,
                nominal_data=nominal_data,
                nominal_key=_NOMINAL_SCENARIO,
            )
            for term in terms
        )
        output_constraints = tuple(
            replace(
                constraint,
                objective=_prepare_objective(
                    constraint.objective,
                    sim_scenarios=sim_scenarios,
                    sensor_arrays=self.sensor_arrays,
                    nominal_data=nominal_data,
                    nominal_key=_NOMINAL_SCENARIO,
                ),
            )
            if isinstance(constraint, ConstrainObjVal)
            else constraint
            for constraint in self.design_space.output_constraints
        )

        empty_score_keys: list[tuple[str, ...]] = []
        for term, objective in zip(terms, resolved_objectives, strict=True):
            if term.empty_score is None:
                empty_score_keys.append(())
                continue
            if not np.isfinite(term.empty_score):
                raise ValueError("empty_score must be finite.")
            keys = term.sensor_keys
            if keys is None and isinstance(
                objective,
                (ObjFieldRecon, ObjSignal, ObjSignalPeakAbs, ObjSignalRMS),
            ):
                keys = objective.sensor_keys
            if not keys or not self.sensor_arrays.keys() >= set(keys):
                raise ValueError("empty_score requires valid input sensor keys.")
            empty_score_keys.append(keys)

        required_outputs = frozenset().union(
            *(objective.get_required_outputs() for objective in resolved_objectives),
            *(constraint.get_required_outputs() for constraint in output_constraints),
        )
        scenario_arrays: dict[str, dict[str, ISensorArray]] = {}
        num_streams = len(scenario_keys) * len(self.sensor_arrays)
        array_seeds: list[int | None] = [self.seed] * num_streams
        if self.seed is not None and (num_streams > 1 or isinstance(source, ISimSampler)):
            array_seeds = [
                int(sequence.generate_state(1)[0])
                for sequence in physics_seed.spawn(num_streams)
            ]
        if required_outputs:
            seeds = iter(array_seeds)
            for scenario_key in scenario_keys:
                arrays: dict[str, ISensorArray] = {}
                for key, array in self.sensor_arrays.items():
                    working_array = array.copy_for_simulation(seed=next(seeds))
                    if scenario_key in sim_scenarios:
                        working_array.get_field().set_sim_data(sim_scenarios[scenario_key])
                    arrays[key] = working_array
                scenario_arrays[scenario_key] = arrays

        def decode_layout(layout: Layout) -> dict[str, SensorData]:
            sensor_indices: dict[str, list[int]] = {}
            for index in layout:
                sensor_indices.setdefault(site_keys[index], []).append(index)
            return {
                key: replace(nominal_data[key], positions=all_sites[indices])
                for key, indices in sensor_indices.items()
            }

        def score_layout(layout: Layout) -> float | None:
            sensor_data = decode_layout(layout)
            for key, data in sensor_data.items():
                assert data.positions is not None
                if len(data.positions) > inventories[key]:
                    return None
            for constraint in self.design_space.input_constraints:
                residuals = constraint.eval(sensor_data)
                if not np.all(np.isfinite(residuals)):
                    raise ValueError("Input constraint residuals must be finite.")
                if np.any(residuals > 0):
                    return None

            experiment_data = ExperimentSimulator.collect_virtual_exps(
                scenario_arrays=scenario_arrays,
                sensor_data=sensor_data,
                num_virtual_exps=self.num_virtual_exps,
                required_outputs=required_outputs,
            )
            for constraint in output_constraints:
                residuals = constraint.eval(
                    sensor_data=sensor_data,
                    experiment_data=experiment_data,
                    scenario_keys=scenario_keys,
                )
                if not np.all(np.isfinite(residuals)):
                    raise ValueError("Output constraint residuals must be finite.")
                if np.any(residuals > 0):
                    return None

            score = 0.0
            for term, keys, objective, spec in zip(
                terms, empty_score_keys, resolved_objectives, objective_specs, strict=True
            ):
                if term.empty_score is not None and not any(
                    key in sensor_data for key in keys
                ):
                    value = term.empty_score
                else:
                    values = objective.eval(
                        sensor_data=sensor_data,
                        experiment_data=experiment_data,
                        scenario_keys=scenario_keys,
                    )
                    if not isinstance(values, np.ndarray) or not np.all(np.isfinite(values)):
                        raise ValueError("An objective must return finite values in an array.")
                    if values.shape not in ((), (len(scenario_keys), self.num_virtual_exps)):
                        raise ValueError("An objective must return a scalar or a loss per experiment.")
                    if spec is not None:
                        values = spec.eval_strat.reduce(values, scenario_keys=scenario_keys)
                    if values.shape not in ((), (1, 1)):
                        raise ValueError("Use EvalStrat to reduce the objective to one score.")
                    value = float(values.item())
                if not np.isfinite(value):
                    raise ValueError("An objective must return a finite value.")
                if term.normalisation is not None:
                    value = term.normalisation.eval(value)
                score += term.weight * value
            if not np.isfinite(score):
                raise ValueError("The combined objective score must be finite.")
            return score

        num_start_layouts = 0
        if isinstance(optimiser, OptGreedyForward):
            start = optimiser.start
            best_start: _ScoredLayout | None = None
            if start is not None:
                starts: Iterable[Layout]
                if isinstance(start, OptStartLayout):
                    counts = {key: len(indices) for key, indices in start.sites.items()}
                else:
                    counts = start.counts
                count_limit = optimiser.num_sensors or optimiser.max_sensors
                domains = _get_start_domains(
                    counts, site_domains, inventories, count_limit
                )

                if isinstance(start, OptStartLayout):
                    indices: list[int] = []
                    for key, local_sites in start.sites.items():
                        domain = site_domains[key]
                        if any(
                            not _is_integer(site, minimum=0) or site >= len(domain)
                            for site in local_sites
                        ) or len(set(local_sites)) != len(local_sites):
                            raise ValueError(
                                f"Invalid or duplicate starting sites for {key!r}."
                            )
                        indices.extend(domain[int(site)] for site in local_sites)
                    starts = (tuple(sorted(indices)),)
                elif isinstance(start, OptStartExhaustive):
                    starts = _enumerate_starts(domains)
                else:
                    if not _is_integer(start.num_layouts, minimum=1):
                        raise ValueError("num_layouts must be a positive integer.")
                    random_seed = None if self.seed is None else int(self.seed)
                    starts = _sample_starts(
                        domains, int(start.num_layouts), random_seed
                    )

                for layout in starts:
                    num_start_layouts += 1
                    score = score_layout(layout)
                    if score is not None and (
                        best_start is None or score < best_start.objective
                    ):
                        best_start = _ScoredLayout(layout, score)

            if start is not None and best_start is None:
                if isinstance(start, OptStartLayout):
                    raise ValueError("The starting layout violates constraints.")
                search = _SearchResult(None, EOptStopCond.NO_FEASIBLE_START, 0, False)
            else:
                search = _search_forward(
                    optimiser,
                    num_sites=len(all_sites),
                    inventory=available_count,
                    score_layout=score_layout,
                    start=best_start,
                )
        elif isinstance(optimiser, OptGreedyBackward):
            search = _search_backward(
                optimiser, initial_sites=initial_sites, score_layout=score_layout
            )
        elif isinstance(optimiser, OptExchange):
            search = _search_exchange(
                optimiser,
                initial_sites=initial_sites,
                num_sites=len(all_sites),
                score_layout=score_layout,
            )
        else:
            search = _search_exhaustive(
                optimiser,
                num_sites=len(all_sites),
                inventory=available_count,
                score_layout=score_layout,
            )

        selected_data: dict[str, SensorData] = {}
        objective_values = np.empty(0, dtype=np.float64)
        if search.layout is not None:
            selected_data = decode_layout(search.layout.sites)
            objective_values = np.array([search.layout.objective], dtype=np.float64)
        return OptResult(
            sensor_data=selected_data,
            objective_values=objective_values,
            stop_condition=search.stop_condition,
            num_iterations=search.num_iterations,
            success=search.success,
            num_start_layouts=num_start_layouts,
        )


def _get_start_domains(
    counts: Mapping[str, int],
    site_domains: Mapping[str, range],
    inventories: Mapping[str, int],
    count_limit: int | None,
) -> tuple[tuple[range, int], ...]:
    for key, count in counts.items():
        if key not in site_domains:
            raise ValueError(f"Unknown starting sensor array {key!r}.")
        if not _is_integer(count, minimum=0) or count > min(
            len(site_domains[key]), inventories[key]
        ):
            raise ValueError(f"Invalid starting count for {key!r}.")
    total = sum(int(count) for count in counts.values())
    if total == 0 or (count_limit is not None and total > count_limit):
        raise ValueError("A start must select at least one sensor within the count limit.")
    return tuple(
        (domain, int(counts[key]))
        for key, domain in site_domains.items()
        if counts.get(key)
    )


def _is_integer(value: object, *, minimum: int) -> bool:
    return (
        isinstance(value, Integral)
        and not isinstance(value, bool)
        and int(value) >= minimum
    )
