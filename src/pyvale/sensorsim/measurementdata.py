# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Container for simulated sensor measurements with spatial and temporal
metadata.
"""

from dataclasses import dataclass
import numpy as np

from pyvale.sensorsim.sensordescriptor import SensorDescriptor
from pyvale.sensorsim.sensorarray import ISensorArray


@dataclass(slots=True)
class MeasurementData:
    """Carries measurement values alongside temporal, spatial, and component
    metadata for downstream post-processing pipelines.
    """

    values: np.ndarray
    """Measurement values array. Shape can be 3D
    (num_sensors, num_components, num_times) for a single experiment or 4D
    (num_trials, num_sensors, num_components, num_times) for Monte Carlo.
    """

    sample_times: np.ndarray
    """Sampling time coordinates with shape (num_times,) or
    (num_sensors, num_times).
    """

    positions: np.ndarray | None = None
    """Spatial coordinates of sensors with shape (num_sensors, 3).
    """

    components: tuple[str, ...] = ("value",)
    """Names of measurement components along the components axis.
    """

    units: str = ""
    """Engineering units string (e.g. 'mm', 'm/s', 'N', 'με').
    """

    descriptor: SensorDescriptor | None = None
    """Optional sensor descriptor containing name, tag, and symbol metadata.
    """

    @classmethod
    def from_sensor_array(
        cls,
        sensor: ISensorArray,
        use_truth: bool = False,
    ) -> "MeasurementData":
        """Constructs a MeasurementData instance from an active ISensorArray.

        Parameters
        ----------
        sensor : ISensorArray
            Active sensor array instance.
        use_truth : bool, optional
            If True, extracts ground-truth simulation values without errors;
            if False, extracts simulated measurements with error models.
            Default is False.

        Returns
        -------
        MeasurementData
            Constructed measurement data object.
        """
        if use_truth:
            vals = sensor.get_truth()
        else:
            vals = sensor.sim_measurements()

        sens_data = sensor.get_sensor_data()
        desc = sensor.get_descriptor()
        comps = sensor.get_field().get_all_components()
        t_steps = sens_data.sample_times
        if t_steps is None:
            t_steps = sensor.get_field().get_sim_data().time

        units_str = desc.units if desc is not None else ""

        return cls(
            values=vals,
            sample_times=t_steps,
            positions=sens_data.positions,
            components=comps,
            units=units_str,
            descriptor=desc,
        )
