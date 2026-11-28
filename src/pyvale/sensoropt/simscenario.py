# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Resolve reconstruction references once per realised set of scenarios."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from pyvale.dataio.simdata import SimData
from pyvale.sensoropt.evalstrat import _get_scenario_weights
from pyvale.sensoropt.objective import EReconRef, IObjFunc, ObjFieldRecon, ObjSpec
from pyvale.sensorsim.enums import EDim
from pyvale.sensorsim.fieldscalar import FieldScalar
from pyvale.sensorsim.sensorarray import ISensorArray
from pyvale.sensorsim.sensordata import SensorData


def _prepare_objective(
    objective: IObjFunc | ObjSpec,
    *,
    sim_scenarios: Mapping[str, SimData],
    sensor_arrays: Mapping[str, ISensorArray],
    nominal_data: Mapping[str, SensorData],
    nominal_key: str,
) -> IObjFunc:
    if isinstance(objective, ObjSpec):
        objective = objective.objective
    if not isinstance(objective, ObjFieldRecon):
        return objective

    scenario_keys = tuple(sim_scenarios) or (nominal_key,)
    mode = objective.reference_mode
    weights = objective.reference_weights
    if weights is not None and mode is not EReconRef.MEAN_SCENARIO:
        raise ValueError("reference_weights require MEAN_SCENARIO.")
    if not isinstance(mode, EReconRef):
        raise TypeError("reference_mode must be an EReconRef member.")
    field_key = objective.reference_field
    references = objective.target_values
    if (field_key is None) == (references is None):
        raise ValueError("Specify exactly one of target_values and reference_field.")
    if field_key is not None:
        if (
            not objective.sensor_keys
            or not set(objective.sensor_keys) <= sensor_arrays.keys()
        ):
            raise ValueError("Reconstruction reference requires valid sensor keys.")
        first_key = objective.sensor_keys[0]
        cases = sim_scenarios or {
            nominal_key: sensor_arrays[first_key].get_field().get_sim_data()
        }
        references = {}
        common_times: npt.NDArray[np.float64] | None = None
        for scenario_key, simulation in cases.items():
            if simulation.node_vars is None or field_key not in simulation.node_vars:
                raise ValueError(
                    f"Scenario {scenario_key!r} has no field {field_key!r}."
                )
            times = nominal_data[first_key].sample_times
            if times is None:
                times = simulation.time
            if times is None:
                raise ValueError(
                    f"Scenario {scenario_key!r} has no reference sample times."
                )
            for key in objective.sensor_keys[1:]:
                other_times = nominal_data[key].sample_times
                if other_times is None:
                    other_times = (
                        simulation.time
                        if sim_scenarios
                        else sensor_arrays[key].get_field().get_time_steps()
                    )
                if other_times is None or not np.array_equal(times, other_times):
                    raise ValueError(
                        "Reference field requires matching sensor sample times."
                    )
            if mode is not EReconRef.SIM_SCENARIO:
                if common_times is not None and not np.array_equal(times, common_times):
                    raise ValueError(
                        "Ensemble references require matching scenario sample times."
                    )
                common_times = times
            if (
                simulation.coords is not None
                and simulation.time is not None
                and np.array_equal(objective.target_positions, simulation.coords)
                and np.array_equal(times, simulation.time)
            ):
                references[scenario_key] = simulation.node_vars[field_key]
            else:
                field = FieldScalar(
                    simulation, field_key, EDim(simulation.num_spat_dims)
                )
                references[scenario_key] = field.sample_field(
                    objective.target_positions, times
                )[:, 0, :]

    assert references is not None
    if not set(scenario_keys) <= references.keys():
        raise ValueError(
            "Supply reconstruction references for all simulation scenarios."
        )
    if mode is EReconRef.SIM_SCENARIO:
        return replace(objective, target_values=references, reference_field=None)

    stacked = np.stack([references[key] for key in scenario_keys], axis=0)
    if mode is EReconRef.MEAN_SCENARIO:
        reference = np.average(
            stacked, axis=0, weights=_get_scenario_weights(weights, scenario_keys)
        )
    else:
        reference = np.median(stacked, axis=0)
    return replace(
        objective,
        target_values=dict.fromkeys(scenario_keys, reference),
        reference_field=None,
        reference_mode=EReconRef.SIM_SCENARIO,
        reference_weights=None,
    )
