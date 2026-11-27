# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Evaluate discrete placements on engine-owned sensor arrays."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np
import numpy.typing as npt

from pyvale.sensopt.designspace import DesignSpace, EDesignAxis
from pyvale.sensopt.objective import IObjFunc, ObjTerm, ObjWeighted
from pyvale.sensopt.optimiser import (
    EOptStopCond,
    Layout,
    OptDiscrete,
    OptExchange,
    OptExhaustive,
    OptGreedyBackward,
    OptGreedyForward,
    _search_backward,
    _search_exchange,
    _search_exhaustive,
    _search_forward,
)
from pyvale.sensorsim.enums import EExpSimKeys
from pyvale.sensorsim.sensorarray import ISensorArray
from pyvale.sensorsim.sensordata import SensorData

_NOMINAL_SCENARIO = "nominal"
_MEASURED_OUTPUTS = frozenset(
    {
        EExpSimKeys.MEASUREMENT,
        EExpSimKeys.RANDOM_ERROR,
        EExpSimKeys.SYSTEMATIC_ERROR,
    }
)
_SUPPORTED_OUTPUTS = _MEASURED_OUTPUTS | frozenset(
    {
        EExpSimKeys.TRUTH,
        EExpSimKeys.SENSOR_POSITIONS,
        EExpSimKeys.SENSOR_TIMES,
    }
)


@dataclass(slots=True)
class OptResult:
    """Best layout found and the reason the optimiser stopped.

    Objective values have shape (objective,). An empty layout and score mean no
    feasible candidate was evaluated. Success means the layout met the requested
    count or objective target, where applicable. An iteration-limited exchange or
    exhaustive search can return a feasible layout with success and an incomplete
    search reported by the stop condition.
    """

    sensor_data: dict[str, SensorData]
    objective_values: npt.NDArray[np.float64]
    stop_condition: EOptStopCond
    num_iterations: int
    success: bool


@dataclass(slots=True, kw_only=True)
class SensorOptEngine:
    """Optimise named arrays over discrete sites in already-bound fields.

    Each array's configured count is its inventory limit. Backward and exchange
    start at the configured positions. Each candidate gets one scalar score and
    one realisation per active array. Unselected arrays are omitted from layouts.
    """

    sensor_arrays: Mapping[str, ISensorArray]
    design_space: DesignSpace
    objectives: tuple[IObjFunc, ...] | ObjWeighted
    optimiser: OptDiscrete
    seed: int | None = None

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
                raise ValueError(f"Array {key!r} must define finite nominal XYZ positions.")
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
                installed_sites.extend(len(site_keys) + index for index in array_installed)
            nominal_data[key] = data
            inventories[key] = len(positions)
            available_count += min(len(sites), len(positions))
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
                if not _is_integer(max_sensors, minimum=1) or max_sensors > available_count:
                    raise ValueError("max_sensors must be within the available sensor count.")
        if isinstance(optimiser, OptExhaustive):
            if optimiser.num_sensors is None and optimiser.target_objective is None:
                raise ValueError("Specify num_sensors or target_objective.")
        if isinstance(optimiser, (OptGreedyForward, OptExhaustive)):
            target = optimiser.target_objective
            if target is not None and not np.isfinite(target):
                raise ValueError("target_objective must be finite.")
        if not isinstance(optimiser, OptExchange):
            count = optimiser.num_sensors
            if isinstance(optimiser, OptGreedyBackward) and count is None:
                raise ValueError("This optimiser requires num_sensors.")
            if isinstance(optimiser, OptExhaustive) and isinstance(count, range):
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

        required_outputs = frozenset().union(
            *(term.objective.get_required_outputs() for term in terms)
        )
        if not required_outputs <= _SUPPORTED_OUTPUTS:
            raise ValueError("The objective requests an unsupported experiment output.")
        array_seeds: list[int | None] = [self.seed] * len(self.sensor_arrays)
        if self.seed is not None and len(self.sensor_arrays) > 1:
            array_seeds = [
                int(sequence.generate_state(1)[0])
                for sequence in np.random.SeedSequence(self.seed).spawn(
                    len(self.sensor_arrays)
                )
            ]
        working_arrays = {
            key: array.copy_for_simulation(seed=array_seed)
            for (key, array), array_seed in zip(
                self.sensor_arrays.items(), array_seeds, strict=True
            )
        }

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

            experiment_data: dict[
                tuple[str, str, EExpSimKeys], npt.NDArray[np.float64]
            ] = {}
            for sensor_key, candidate_data in sensor_data.items():
                if not required_outputs:
                    break
                working_array = working_arrays[sensor_key]
                working_array.set_sensor_data(candidate_data)
                measurements = None
                if required_outputs & _MEASURED_OUTPUTS:
                    measurements = working_array.sim_measurements()

                for output_key in required_outputs:
                    match output_key:
                        case EExpSimKeys.MEASUREMENT:
                            assert measurements is not None
                            output = measurements[np.newaxis, ...]
                        case EExpSimKeys.TRUTH:
                            output = working_array.get_truth()[np.newaxis, ...]
                        case EExpSimKeys.SENSOR_TIMES:
                            output = candidate_data.sample_times
                            if output is None:
                                output = working_array.get_field().get_time_steps()
                        case EExpSimKeys.SENSOR_POSITIONS:
                            assert candidate_data.positions is not None
                            output = candidate_data.positions
                        case EExpSimKeys.SYSTEMATIC_ERROR | EExpSimKeys.RANDOM_ERROR:
                            if output_key is EExpSimKeys.SYSTEMATIC_ERROR:
                                errors = working_array.get_errors_systematic()
                            else:
                                errors = working_array.get_errors_random()
                            if errors is None:
                                assert measurements is not None
                                errors = np.zeros_like(measurements)
                            output = errors[np.newaxis, ...]
                        case _:
                            raise ValueError("Unsupported experiment output.")

                    output_view = output.view()
                    output_view.setflags(write=False)
                    experiment_data[_NOMINAL_SCENARIO, sensor_key, output_key] = (
                        output_view
                    )

            score = 0.0
            for term in terms:
                values = term.objective.eval(
                    sensor_data=sensor_data,
                    experiment_data=experiment_data,
                    scenario_keys=(_NOMINAL_SCENARIO,),
                )
                if not isinstance(values, np.ndarray) or values.shape not in ((), (1, 1)):
                    raise ValueError("An objective must return a scalar or a (1, 1) array.")
                value = float(values.item())
                if not np.isfinite(value):
                    raise ValueError("An objective must return a finite value.")
                if term.normalisation is not None:
                    value = term.normalisation.eval(value)
                score += term.weight * value
            if not np.isfinite(score):
                raise ValueError("The combined objective score must be finite.")
            return score

        if isinstance(optimiser, OptGreedyForward):
            search = _search_forward(
                optimiser,
                num_sites=len(all_sites),
                inventory=available_count,
                score_layout=score_layout,
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
        )


def _is_integer(value: object, *, minimum: int) -> bool:
    return (
        isinstance(value, Integral)
        and not isinstance(value, bool)
        and int(value) >= minimum
    )
