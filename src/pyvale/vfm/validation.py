import numpy as np

from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.identificationconfig import (
    IdentificationConfig,
    IdentificationPhase,
)
from pyvale.vfm.metricsliceforce import SliceWiseForceReconstructionMetric
from pyvale.vfm.optimiserslicewiseindependent import (
    SliceWiseIndependentLeastSquares,
)
from pyvale.vfm.spatialparam import get_num_degrees_of_freedom
from pyvale.vfm.spatialparamknown import SpatialParameterisationKnown
from pyvale.vfm.spatialparamslicewise import SliceWiseSpatialParameterisation


def validate_experiment_data(
    experiment_data: ExperimentData
) -> None:
    geometry = experiment_data.specimen_geometry
    boundary_conditions = experiment_data.boundary_conditions

    # Shape and dtype checks
    strain = experiment_data.strain
    if strain.ndim != 4:
        raise ValueError(
            f"strain must be 4D (timesteps, components, y, x), "
            f"got ndim={strain.ndim}"
        )
    if strain.shape[1] != 3:
        raise ValueError(
            f"strain must have exactly 3 components [xx, yy, xy], "
            f"got {strain.shape[1]}"
        )
    if strain.dtype != np.float64:
        raise ValueError(
            f"strain must be float64, got {strain.dtype}"
        )

    timesteps = experiment_data.timesteps
    if timesteps.ndim != 1:
        raise ValueError(
            f"timesteps must be 1D, got ndim={timesteps.ndim}"
        )
    if timesteps.dtype != np.float64:
        raise ValueError(
            f"timesteps must be float64, got {timesteps.dtype}"
        )

    force = boundary_conditions.force
    if force.ndim != 2:
        raise ValueError(
            f"force must be 2D (timesteps, 2) with columns [Fx, Fy], "
            f"got ndim={force.ndim}"
        )
    if force.dtype != np.float64:
        raise ValueError(
            f"force must be float64, got {force.dtype}"
        )

    for field_name, array in [
        ("x", geometry.x),
        ("y", geometry.y),
        ("pixel_area", geometry.pixel_area),
    ]:
        if array.ndim != 2:
            raise ValueError(
                f"{field_name} must be 2D (y, x), got ndim={array.ndim}"
            )
        if array.dtype != np.float64:
            raise ValueError(
                f"{field_name} must be float64, got {array.dtype}"
            )

    specimen_mask = geometry.region_of_interest.sample_specimen_mask(
        geometry.x,
        geometry.y
    )

    if specimen_mask.ndim != 2:
        raise ValueError(
            f"specimen_mask must be 2D (y, x), "
            f"got ndim={specimen_mask.ndim}"
        )
    if specimen_mask.dtype != np.bool_:
        raise ValueError(
            f"specimen_mask must be bool dtype, "
            f"got {specimen_mask.dtype}"
        )

    # Cross-field dimension agreement
    n_timesteps, _, n_y, n_x = strain.shape

    if timesteps.shape[0] != n_timesteps:
        raise ValueError(
            f"timesteps length ({timesteps.shape[0]}) does not match "
            f"strain timesteps ({n_timesteps})"
        )
    if force.shape[0] != n_timesteps:
        raise ValueError(
            f"force timesteps ({force.shape[0]}) does not match "
            f"strain timesteps ({n_timesteps})"
        )

    if geometry.x.shape != (n_y, n_x):
        raise ValueError(
            f"x shape {geometry.x.shape} does not match "
            f"strain spatial dims ({n_y}, {n_x})"
        )
    if geometry.y.shape != (n_y, n_x):
        raise ValueError(
            f"y shape {geometry.y.shape} does not match "
            f"strain spatial dims ({n_y}, {n_x})"
        )
    if specimen_mask.shape != (n_y, n_x):
        raise ValueError(
            f"specimen_mask shape {specimen_mask.shape} "
            f"does not match strain spatial dims ({n_y}, {n_x})"
        )
    if geometry.pixel_area.shape != (n_y, n_x):
        raise ValueError(
            f"pixel_area shape {geometry.pixel_area.shape} does not match "
            f"strain spatial dims ({n_y}, {n_x})"
        )

    # Value constraints
    if geometry.thickness <= 0:
        raise ValueError(
            f"thickness must be positive, got {geometry.thickness}"
        )
    if np.any(geometry.pixel_area <= 0):
        raise ValueError("pixel_area must be positive everywhere")
    if not np.all(np.isfinite(timesteps)):
        raise ValueError("timesteps contains NaN or Inf values")
    if np.any(np.diff(timesteps) <= 0):
        raise ValueError("timesteps must be strictly increasing")
    if not np.all(np.isfinite(force)):
        raise ValueError("force contains NaN or Inf values")

    # NaN is only allowed where the specimen mask is False (outside the mask)
    flat_mask = specimen_mask.ravel()
    flat_strain = strain.reshape(strain.shape[0], strain.shape[1], -1)
    if not np.all(np.isfinite(flat_strain[:, :, flat_mask])):
        raise ValueError(
            "strain contains NaN or Inf within the specimen mask"
        )

    # Coordinate grid conventions
    if not np.all(np.isfinite(geometry.x)):
        raise ValueError("x contains NaN or Inf values")
    if not np.all(np.isfinite(geometry.y)):
        raise ValueError("y contains NaN or Inf values")

    if n_x >= 2 and np.any(np.diff(geometry.x, axis=1) <= 0):
        raise ValueError("x must increase left to right along each row")
    if n_y >= 2 and np.any(np.diff(geometry.y, axis=0) <= 0):
        raise ValueError("y must increase top to bottom down each column")

    # x and y must form an axis-aligned grid: x varies only across columns and
    # y only across rows. The area of every element is then computed from the
    # grid spacing and must match the supplied pixel_area.
    if n_x >= 2 and n_y >= 2:
        if not np.allclose(np.diff(geometry.x, axis=0), 0.0):
            raise ValueError("x must be constant down each column")
        if not np.allclose(np.diff(geometry.y, axis=1), 0.0):
            raise ValueError("y must be constant along each row")

        if not np.all(
            np.isclose(
                geometry.pixel_area,
                geometry.pixel_area[0],
                rtol=1e-9,
                atol=0.0
            )
        ):
            raise ValueError(
                "Pixel area should be effectively constant across all elements"
                "as x and y must form an axis-aligned grid with uniform spacing"
            )

def validate_identification_config(
    config: IdentificationConfig
) -> None:
    # Structure checks
    if not config.phases:
        raise ValueError("identification must have at least one phase")

    if not config.parameters:
        raise ValueError("identification must have at least one parameter")

    for i, phase in enumerate(config.phases):
        if not phase.metrics:
            raise ValueError(
                f"phase {i} must have at least one metric"
            )

        validate_slicewise_independent_phase(
            phase,
            i,
        )

    # Constitutive-law parameter requirements
    required = set(config.constitutive_law.get_required_parameters())
    given = set(config.parameters.keys())

    extra = given - required
    missing = required - given

    if extra:
        raise ValueError(f"unexpected parameter(s): {extra}")
    if missing:
        raise ValueError(f"missing required parameter(s): {missing}")

    # Cross-field consistency: parameter name agreement
    param_names = set(config.parameters.keys())

    for i, phase in enumerate(config.phases):
        phase_param_names = set(phase.spatial_parameterisations.keys())

        if phase_param_names != param_names:
            missing = param_names - phase_param_names
            extra = phase_param_names - param_names
            parts = []
            if missing:
                parts.append(f"missing: {missing}")
            if extra:
                parts.append(f"unknown: {extra}")
            raise ValueError(
                f"phase {i} spatial parameterisations do not match "
                f"config parameters; {'; '.join(parts)}"
            )

        required_type = phase.optimiser.get_required_objective_function_type()
        if not isinstance(phase.objective_function, required_type):
            raise ValueError(
                f"phase {i}: optimiser requires {required_type.__name__}, "
                f"got {type(phase.objective_function).__name__}"
            )

        # A SpatialParameterisationKnown fully specifies a parameter, so if
        # one appears in a list it must be the only parameterisation in it.
        for param_name, sps in phase.spatial_parameterisations.items():
            has_known = any(
                isinstance(sp, SpatialParameterisationKnown) for sp in sps
            )
            if has_known and len(sps) > 1:
                raise ValueError(
                    f"phase {i} parameter '{param_name}': a "
                    f"SpatialParameterisationKnown must be the only spatial "
                    f"parameterisation in its list"
                )

        # At least one parameter must be identifiable (i.e. not fully
        # specified by a SpatialParameterisationKnown).
        if all(
            any(isinstance(sp, SpatialParameterisationKnown) for sp in sps)
            for sps in phase.spatial_parameterisations.values()
        ):
            raise ValueError(
                f"phase {i}: all parameters are specified by a "
                f"SpatialParameterisationKnown; at least one parameter "
                f"must be identifiable"
            )

    # Value constraints
    for name, param in config.parameters.items():
        if not np.isfinite(param.lower_bound):
            raise ValueError(
                f"parameter '{name}': lower_bound ({param.lower_bound}) "
                f"must be finite"
            )
        if not np.isfinite(param.upper_bound):
            raise ValueError(
                f"parameter '{name}': upper_bound ({param.upper_bound}) "
                f"must be finite"
            )


def validate_slicewise_independent_phase(
    phase: IdentificationPhase,
    phase_index: int,
) -> None:
    if not isinstance(phase.optimiser, SliceWiseIndependentLeastSquares):
        return

    if len(phase.metrics) != 1 or not isinstance(phase.metrics[0], SliceWiseForceReconstructionMetric):
        raise ValueError(
            f"phase {phase_index}: SliceWiseIndependentLeastSquares requires exactly one "
            "SliceWiseForceReconstructionMetric."
        )

    slice_metric = phase.metrics[0]
    if slice_metric.support is None:
        raise ValueError(
            f"phase {phase_index}: SliceWiseForceReconstructionMetric must define a "
            "slice support."
        )

    for param_name, sps in phase.spatial_parameterisations.items():
        if len(sps) != 1:
            raise ValueError(
                f"phase {phase_index} parameter '{param_name}': independent slice-wise "
                "identification currently requires exactly one spatial parameterisation."
            )

        sp = sps[0]
        if not isinstance(sp, SliceWiseSpatialParameterisation):
            if get_num_degrees_of_freedom(sps) != 0:
                raise ValueError(
                    f"phase {phase_index} parameter '{param_name}': all unknown parameters must "
                    "use SliceWiseSpatialParameterisation for independent slice-wise identification."
                )
            continue

        if sp.support is None:
            raise ValueError(
                f"phase {phase_index} parameter '{param_name}': SliceWiseSpatialParameterisation "
                "must define a slice support when used for independent slice-wise identification."
            )

        if sp.support is not slice_metric.support:
            raise ValueError(
                f"phase {phase_index} parameter '{param_name}': independent slice-wise identification "
                "requires this parameterisation to reference the same SupportSlice object as the "
                "slice-force metric."
            )
