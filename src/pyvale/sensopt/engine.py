"""Evaluate discrete placements on an engine-owned sensor array."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np
import numpy.typing as npt

from pyvale.sensopt.designspace import DesignSpace, EDesignAxis
from pyvale.sensopt.objective import IObjFunc
from pyvale.sensopt.optimiser import (
    DiscreteOptimiser,
    EOptStopCond,
    Layout,
    OptimiserExchange,
    OptimiserGreedyBackward,
    OptimiserGreedyForward,
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
class OptimisationResult:
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
    """Optimise one named array over discrete sites in an already-bound field.

    The array's configured count is its inventory limit. Backward and exchange
    searches use its configured positions as a starting layout. This implementation
    evaluates one objective and one realisation per candidate. A seed controls
    the working array's error models when the objective requests measurements.
    """

    sensor_arrays: Mapping[str, ISensorArray]
    design_space: DesignSpace
    objectives: tuple[IObjFunc, ...]
    optimiser: DiscreteOptimiser
    seed: int | None = None

    def optimise(self) -> OptimisationResult:
        """Search the configured design space without mutating any supplied array."""
        if len(self.sensor_arrays) != 1 or len(self.objectives) != 1:
            raise ValueError("This engine requires one sensor array and one objective.")

        sensor_key, source_array = next(iter(self.sensor_arrays.items()))
        if set(self.design_space.positions) != {sensor_key}:
            raise ValueError("Position domains must match the supplied sensor array.")
        if self.design_space.position_axes != {sensor_key: EDesignAxis.DISC}:
            raise ValueError("This engine requires a discrete position domain.")
        sites = np.asarray(self.design_space.positions[sensor_key], dtype=np.float64)
        if sites.ndim != 2 or sites.shape[1] != 3 or not np.all(np.isfinite(sites)):
            raise ValueError("Candidate sites must be finite and have shape (site, 3).")
        if sites.shape[0] == 0 or np.unique(sites, axis=0).shape[0] != sites.shape[0]:
            raise ValueError("The position domain must contain unique candidate sites.")

        nominal_data = source_array.get_sensor_data()
        positions = nominal_data.positions
        if positions is None or positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                "The supplied array must define nominal (sensor, 3) positions."
            )
        if not np.all(np.isfinite(positions)):
            raise ValueError("The supplied array's nominal positions must be finite.")
        inventory = positions.shape[0]
        if inventory == 0:
            raise ValueError("The supplied array must have at least one sensor.")
        if nominal_data.angles is not None and len(nominal_data.angles) != 1:
            raise ValueError("Discrete placement requires a shared sensor rotation.")

        optimiser = self.optimiser
        if not _is_integer(optimiser.max_iterations, minimum=1):
            raise ValueError("max_iterations must be a positive integer.")
        if self.seed is not None and not _is_integer(self.seed, minimum=0):
            raise ValueError("seed must be a nonnegative integer.")

        if isinstance(optimiser, OptimiserGreedyForward):
            if (optimiser.num_sensors is None) == (optimiser.target_objective is None):
                raise ValueError(
                    "Specify exactly one of num_sensors and target_objective."
                )
            target = optimiser.target_objective
            if target is not None and not np.isfinite(target):
                raise ValueError("target_objective must be finite.")
        if not isinstance(optimiser, OptimiserExchange):
            count = optimiser.num_sensors
            if count is None and not isinstance(optimiser, OptimiserGreedyForward):
                raise ValueError("This optimiser requires num_sensors.")
            if count is not None:
                if not _is_integer(count, minimum=1):
                    raise ValueError("num_sensors must be a positive integer.")
                if count > min(len(sites), inventory):
                    raise ValueError(
                        "num_sensors must be within the inventory and site count."
                    )

        initial_sites: Layout = ()
        if isinstance(optimiser, (OptimiserGreedyBackward, OptimiserExchange)):
            site_indices = {
                tuple(position): index for index, position in enumerate(sites)
            }
            installed_sites: list[int] = []
            for position in positions:
                site_index = site_indices.get(tuple(position))
                if site_index is None or site_index in installed_sites:
                    raise ValueError(
                        "Installed positions must match distinct candidate sites."
                    )
                installed_sites.append(site_index)
            initial_sites = tuple(sorted(installed_sites))

        objective = self.objectives[0]
        required_outputs = objective.get_required_outputs()
        if not required_outputs <= _SUPPORTED_OUTPUTS:
            raise ValueError("The objective requests an unsupported experiment output.")
        working_array = source_array.copy_for_simulation(seed=self.seed)

        def score_layout(layout: Layout) -> float | None:
            candidate_positions = sites[list(layout)]
            candidate_data = replace(nominal_data, positions=candidate_positions)
            sensor_data = {sensor_key: candidate_data}
            for constraint in self.design_space.input_constraints:
                residuals = constraint.eval(sensor_data)
                if not np.all(np.isfinite(residuals)):
                    raise ValueError("Input constraint residuals must be finite.")
                if np.any(residuals > 0):
                    return None

            experiment_data = {}
            if required_outputs:
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
                            output = candidate_positions
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

            values = objective.eval(
                sensor_data=sensor_data,
                experiment_data=experiment_data,
                scenario_keys=(_NOMINAL_SCENARIO,),
            )
            if not isinstance(values, np.ndarray) or values.shape not in ((), (1, 1)):
                raise ValueError("An objective must return a scalar or a (1, 1) array.")
            score = float(values.item())
            if not np.isfinite(score):
                raise ValueError("The objective must return a finite value.")
            return score

        if isinstance(optimiser, OptimiserGreedyForward):
            search = _search_forward(
                optimiser,
                num_sites=len(sites),
                inventory=inventory,
                score_layout=score_layout,
            )
        elif isinstance(optimiser, OptimiserGreedyBackward):
            search = _search_backward(
                optimiser, initial_sites=initial_sites, score_layout=score_layout
            )
        elif isinstance(optimiser, OptimiserExchange):
            search = _search_exchange(
                optimiser,
                initial_sites=initial_sites,
                num_sites=len(sites),
                score_layout=score_layout,
            )
        else:
            search = _search_exhaustive(
                optimiser, num_sites=len(sites), score_layout=score_layout
            )

        selected_data: dict[str, SensorData] = {}
        objective_values = np.empty(0, dtype=np.float64)
        if search.layout is not None:
            selected_data[sensor_key] = replace(
                nominal_data, positions=sites[list(search.layout.sites)]
            )
            objective_values = np.array([search.layout.objective], dtype=np.float64)
        return OptimisationResult(
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
