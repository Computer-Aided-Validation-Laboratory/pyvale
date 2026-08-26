import numpy as np

from pyvale.vfm.experimentdata import ExperimentData
from pyvale.vfm.identificationconfig import (
    IdentificationConfig,
    IdentificationPhase,
)
from pyvale.vfm.metricsliceforce import SliceWiseForceReconstructionMetric
from pyvale.vfm.metricequilibriumgap import EquilibriumGapMetric
from pyvale.vfm.objectivefunccombinedfreegi import (
    CombinedForceAndEquilibriumGapObjective,
    CombinedObjectiveBaselineMode,
)
from pyvale.vfm.optimiserslicewiseindependent import (
    SliceWiseIndependentLeastSquares,
)
from pyvale.vfm.refinement import EquilibriumGapBasisGrowthRefinement
from pyvale.vfm.spatialparam import get_num_degrees_of_freedom
from pyvale.vfm.spatialparambasisfuncs import SpatialParameterisationBasisFunction
from pyvale.vfm.spatialparamknown import SpatialParameterisationKnown
from pyvale.vfm.spatialparamslicewise import SliceWiseSpatialParameterisation


def run_validation(
    experiment_data: ExperimentData,
    identification_config: IdentificationConfig | None = None,
) -> None:
    """Validate the VFM inputs, reporting every problem found in one go.

    Runs the experiment data checks and, when an identification config is
    given, the identification config checks. The errors from both are collected
    and raised together as a single ``ValueError`` so that the user sees every
    problem at once rather than fixing them one at a time.

    Parameters
    ----------
    experiment_data : ExperimentData
        The measured full-field strain, geometry, boundary conditions and
        timesteps
    identification_config : IdentificationConfig | None, optional
        The identification config to validate. Omit when only the experiment
        data is available (e.g. when processing input data), by default None

    Raises
    ------
    ValueError
        If any check fails, listing every collected error.
    """
    sections: list[tuple[str, list[str]]] = [
        ("experiment data", _collect_experiment_data_errors(experiment_data)),
    ]

    if identification_config is not None:
        sections.append((
            "identification config",
            _collect_identification_config_errors(identification_config),
        ))

    _raise_collected_section_errors("VFM validation", sections)


def _collect_experiment_data_errors(
    experiment_data: ExperimentData
) -> list[str]:
    geometry = experiment_data.specimen_geometry
    boundary_conditions = experiment_data.boundary_conditions

    errors: list[str] = []

    # Shape and dtype checks
    strain = experiment_data.strain
    strain_is_4d = strain.ndim == 4
    if not strain_is_4d:
        errors.append(
            f"strain must be 4D (timesteps, components, y, x), "
            f"got ndim={strain.ndim}"
        )
    elif strain.shape[1] != 3:
        errors.append(
            f"strain must have exactly 3 components [xx, yy, xy], "
            f"got {strain.shape[1]}"
        )
    if strain.dtype != np.float64:
        errors.append(
            f"strain must be float64, got {strain.dtype}"
        )

    timesteps = experiment_data.timesteps
    timesteps_is_1d = timesteps.ndim == 1
    if not timesteps_is_1d:
        errors.append(
            f"timesteps must be 1D, got ndim={timesteps.ndim}"
        )
    if timesteps.dtype != np.float64:
        errors.append(
            f"timesteps must be float64, got {timesteps.dtype}"
        )

    force = boundary_conditions.force
    if force.ndim != 2:
        errors.append(
            f"force must be 2D (timesteps, 2) with columns [Fx, Fy], "
            f"got ndim={force.ndim}"
        )
    if force.dtype != np.float64:
        errors.append(
            f"force must be float64, got {force.dtype}"
        )

    grid_arrays_are_2d = True
    for field_name, array in [
        ("x", geometry.x),
        ("y", geometry.y),
        ("pixel_area", geometry.pixel_area),
    ]:
        if array.ndim != 2:
            grid_arrays_are_2d = False
            errors.append(
                f"{field_name} must be 2D (y, x), got ndim={array.ndim}"
            )
        if array.dtype != np.float64:
            errors.append(
                f"{field_name} must be float64, got {array.dtype}"
            )

    specimen_mask = geometry.region_of_interest.sample_specimen_mask(
        geometry.x,
        geometry.y
    )

    specimen_mask_is_2d = specimen_mask.ndim == 2
    if not specimen_mask_is_2d:
        errors.append(
            f"specimen_mask must be 2D (y, x), "
            f"got ndim={specimen_mask.ndim}"
        )
    if specimen_mask.dtype != np.bool_:
        errors.append(
            f"specimen_mask must be bool dtype, "
            f"got {specimen_mask.dtype}"
        )

    # Cross-field dimension agreement. The spatial dimensions are taken from
    # the strain array, so these checks are only meaningful once it is 4D.
    if strain_is_4d:
        n_timesteps, _, n_y, n_x = strain.shape

        if timesteps.shape[0] != n_timesteps:
            errors.append(
                f"timesteps length ({timesteps.shape[0]}) does not match "
                f"strain timesteps ({n_timesteps})"
            )
        if force.shape[0] != n_timesteps:
            errors.append(
                f"force timesteps ({force.shape[0]}) does not match "
                f"strain timesteps ({n_timesteps})"
            )

        if geometry.x.shape != (n_y, n_x):
            errors.append(
                f"x shape {geometry.x.shape} does not match "
                f"strain spatial dims ({n_y}, {n_x})"
            )
        if geometry.y.shape != (n_y, n_x):
            errors.append(
                f"y shape {geometry.y.shape} does not match "
                f"strain spatial dims ({n_y}, {n_x})"
            )
        if specimen_mask.shape != (n_y, n_x):
            errors.append(
                f"specimen_mask shape {specimen_mask.shape} "
                f"does not match strain spatial dims ({n_y}, {n_x})"
            )
        if geometry.pixel_area.shape != (n_y, n_x):
            errors.append(
                f"pixel_area shape {geometry.pixel_area.shape} does not match "
                f"strain spatial dims ({n_y}, {n_x})"
            )

    # Value constraints
    if geometry.thickness <= 0:
        errors.append(
            f"thickness must be positive, got {geometry.thickness}"
        )
    if np.any(geometry.pixel_area <= 0):
        errors.append("pixel_area must be positive everywhere")
    if not np.all(np.isfinite(timesteps)):
        errors.append("timesteps contains NaN or Inf values")
    if timesteps_is_1d and np.any(np.diff(timesteps) <= 0):
        errors.append("timesteps must be strictly increasing")
    if not np.all(np.isfinite(force)):
        errors.append("force contains NaN or Inf values")

    # NaN is only allowed where the specimen mask is False (outside the mask)
    if strain_is_4d and specimen_mask.shape == strain.shape[2:]:
        flat_mask = specimen_mask.ravel()
        flat_strain = strain.reshape(strain.shape[0], strain.shape[1], -1)
        if not np.all(np.isfinite(flat_strain[:, :, flat_mask])):
            errors.append(
                "strain contains NaN or Inf within the specimen mask"
            )

    # Coordinate grid conventions
    if not np.all(np.isfinite(geometry.x)):
        errors.append("x contains NaN or Inf values")
    if not np.all(np.isfinite(geometry.y)):
        errors.append("y contains NaN or Inf values")

    if grid_arrays_are_2d:
        n_y, n_x = geometry.x.shape

        if n_x >= 2 and np.any(np.diff(geometry.x, axis=1) <= 0):
            errors.append("x must increase left to right along each row")
        if n_y >= 2 and np.any(np.diff(geometry.y, axis=0) <= 0):
            errors.append("y must increase top to bottom down each column")

        # x and y must form an axis-aligned grid: x varies only across columns
        # and y only across rows. The area of every element is then computed
        # from the grid spacing and must match the supplied pixel_area.
        if n_x >= 2 and n_y >= 2:
            if not np.allclose(np.diff(geometry.x, axis=0), 0.0):
                errors.append("x must be constant down each column")
            if not np.allclose(np.diff(geometry.y, axis=1), 0.0):
                errors.append("y must be constant along each row")

            if not np.all(
                np.isclose(
                    geometry.pixel_area,
                    geometry.pixel_area[0],
                    rtol=1e-9,
                    atol=0.0
                )
            ):
                errors.append(
                    "Pixel area should be effectively constant across all "
                    "elements as x and y must form an axis-aligned grid with "
                    "uniform spacing"
                )

    return errors


def _collect_identification_config_errors(
    config: IdentificationConfig
) -> list[str]:
    errors: list[str] = []

    # Structure checks
    if not config.phases:
        errors.append("identification must have at least one phase")

    if not config.parameters:
        errors.append("identification must have at least one parameter")

    for i, phase in enumerate(config.phases):
        if not phase.metrics:
            errors.append(
                f"phase {i} must have at least one metric"
            )

        # Check that the radial return Newton-Raphson tolerance is finite and positive.
        if (
            not np.isfinite(phase.optimisation_newton_tolerance)
            or phase.optimisation_newton_tolerance <= 0.0
        ):
            errors.append(
                f"phase {i}: optimisation_newton_tolerance must be finite "
                "and greater than zero"
            )

        errors.extend(
            _collect_slicewise_independent_phase_errors(
                phase,
                i,
            )
        )
        errors.extend(_collect_egi_basis_growth_errors(phase, i))

    # Constitutive-law parameter requirements
    required = set(config.constitutive_law.get_required_parameters())
    given = set(config.parameters.keys())

    extra = given - required
    missing = required - given

    if extra:
        errors.append(f"unexpected parameter(s): {extra}")
    if missing:
        errors.append(f"missing required parameter(s): {missing}")

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
            errors.append(
                f"phase {i} spatial parameterisations do not match "
                f"config parameters; {'; '.join(parts)}"
            )

        required_type = phase.optimiser.get_required_objective_function_type()
        if not isinstance(phase.objective_function, required_type):
            errors.append(
                f"phase {i}: optimiser requires {required_type.__name__}, "
                f"got {type(phase.objective_function).__name__}"
            )

        if isinstance(
            phase.objective_function,
            CombinedForceAndEquilibriumGapObjective,
        ):
            baseline = phase.objective_function.baseline
            if baseline.mode is CombinedObjectiveBaselineMode.PRIOR_PHASE:
                if baseline.phase_index is None or baseline.phase_index < 0:
                    errors.append(
                        f"phase {i}: prior-phase baseline requires a "
                        "non-negative phase_index"
                    )
                elif baseline.phase_index >= i:
                    errors.append(
                        f"phase {i}: prior-phase baseline must reference "
                        "an earlier phase, got phase_index "
                        f"{baseline.phase_index}"
                    )
            elif baseline.mode is CombinedObjectiveBaselineMode.MANUAL:
                expected_egi_count = sum(
                    isinstance(metric, EquilibriumGapMetric)
                    for metric in phase.metrics
                )
                if (
                    expected_egi_count
                    and baseline.egi_values is not None
                    and len(baseline.egi_values) != expected_egi_count
                ):
                    errors.append(
                        f"phase {i}: manual baseline has "
                        f"{len(baseline.egi_values)} EGI values but phase "
                        f"defines {expected_egi_count} EGI metrics"
                    )

        for parameter_name, parameterisations in (
            phase.spatial_parameterisations.items()
        ):
            basis_count = sum(
                isinstance(parameterisation, SpatialParameterisationBasisFunction)
                for parameterisation in parameterisations
            )
            if basis_count > 1:
                errors.append(
                    f"phase {i} parameter '{parameter_name}': at most one "
                    "SpatialParameterisationBasisFunction is allowed"
                )

        # Known maps may be combined with active additive parameterisations;
        # this is useful for a frozen baseline plus basis-function increments.
        # A phase is fixed only when every list contains Known entries alone.
        if phase.spatial_parameterisations and all(
            bool(sps) and all(isinstance(sp, SpatialParameterisationKnown) for sp in sps)
            for sps in phase.spatial_parameterisations.values()
        ):
            errors.append(
                f"phase {i}: all parameters are specified by a "
                f"SpatialParameterisationKnown; at least one parameter "
                f"must be identifiable"
            )

    # Value constraints
    for name, param in config.parameters.items():
        if not np.isfinite(param.lower_bound):
            errors.append(
                f"parameter '{name}': lower_bound ({param.lower_bound}) "
                f"must be finite"
            )
        if not np.isfinite(param.upper_bound):
            errors.append(
                f"parameter '{name}': upper_bound ({param.upper_bound}) "
                f"must be finite"
            )

    return errors


def _collect_egi_basis_growth_errors(
    phase: IdentificationPhase,
    phase_index: int,
) -> list[str]:
    policy = phase.refinement_policy
    if not isinstance(policy, EquilibriumGapBasisGrowthRefinement):
        return []

    errors: list[str] = []
    egi_count = sum(
        isinstance(metric, EquilibriumGapMetric) for metric in phase.metrics
    )
    if not egi_count:
        errors.append(
            f"phase {phase_index}: EGI basis growth requires at least one "
            "EquilibriumGapMetric"
        )
    if not isinstance(
        phase.objective_function,
        CombinedForceAndEquilibriumGapObjective,
    ):
        if policy.baseline_phase_index is None:
            errors.append(
                f"phase {phase_index}: EGI basis growth with a non-combined "
                "objective requires baseline_phase_index"
            )
        elif policy.baseline_phase_index >= phase_index:
            errors.append(
                f"phase {phase_index}: EGI refinement baseline must reference "
                f"an earlier phase, got {policy.baseline_phase_index}"
            )
        if policy.egi_window_weights is None:
            errors.append(
                f"phase {phase_index}: EGI basis growth with a non-combined "
                "objective requires egi_window_weights"
            )
        elif len(policy.egi_window_weights) != egi_count:
            errors.append(
                f"phase {phase_index}: EGI refinement has "
                f"{len(policy.egi_window_weights)} window weights but phase "
                f"defines {egi_count} EGI metrics"
            )
    if (
        isinstance(policy.target, tuple)
        and len(policy.target) == 2
        and isinstance(policy.target[0], str)
    ):
        parameter_name, parameterisation_index = policy.target
        parameterisations = phase.spatial_parameterisations.get(parameter_name, [])
        target_is_basis = (
            isinstance(parameterisation_index, int)
            and 0 <= parameterisation_index < len(parameterisations)
            and isinstance(
                parameterisations[parameterisation_index],
                SpatialParameterisationBasisFunction,
            )
        )
    else:
        target_is_basis = any(
            parameterisation is policy.target
            and isinstance(parameterisation, SpatialParameterisationBasisFunction)
            for parameterisations in phase.spatial_parameterisations.values()
            for parameterisation in parameterisations
        )
    if not target_is_basis:
        errors.append(
            f"phase {phase_index}: EGI basis growth target must select a "
            "SpatialParameterisationBasisFunction"
        )
    return errors


def _raise_collected_errors(
    context: str,
    errors: list[str],
) -> None:
    """Raise a single ValueError listing every error collected by a validator.

    Does nothing when no errors were collected.
    """
    _raise_collected_section_errors(context, [("", errors)])


def _raise_collected_section_errors(
    context: str,
    sections: list[tuple[str, list[str]]],
) -> None:
    """Raise a single ValueError listing every error from every section.

    Each section is a name and the errors collected by the validator of that
    name. Errors are numbered continuously across sections and unnamed or empty
    sections are not given a heading. Does nothing when no errors were
    collected.
    """
    total = sum(len(errors) for _, errors in sections)
    if not total:
        return

    lines: list[str] = []
    number = 1
    for name, errors in sections:
        if not errors:
            continue
        indent = "  "
        if name:
            lines.append(f"  {name}:")
            indent = "    "
        for error in errors:
            lines.append(f"{indent}({number}) {error}")
            number += 1

    error_list = "\n".join(lines)
    raise ValueError(f"{context} found {total} error(s):\n{error_list}")


def validate_slicewise_independent_phase(
    phase: IdentificationPhase,
    phase_index: int,
) -> None:
    _raise_collected_errors(
        f"phase {phase_index} slice-wise independent validation",
        _collect_slicewise_independent_phase_errors(phase, phase_index),
    )


def _collect_slicewise_independent_phase_errors(
    phase: IdentificationPhase,
    phase_index: int,
) -> list[str]:
    if not isinstance(phase.optimiser, SliceWiseIndependentLeastSquares):
        return []

    errors: list[str] = []

    if len(phase.metrics) != 1 or not isinstance(phase.metrics[0], SliceWiseForceReconstructionMetric):
        errors.append(
            f"phase {phase_index}: SliceWiseIndependentLeastSquares requires exactly one "
            "SliceWiseForceReconstructionMetric."
        )
        # Without the slice-force metric there is no support to compare the
        # spatial parameterisations against, so the remaining checks are
        # skipped.
        return errors

    slice_metric = phase.metrics[0]
    if slice_metric.support is None:
        errors.append(
            f"phase {phase_index}: SliceWiseForceReconstructionMetric must define a "
            "slice support."
        )

    for param_name, sps in phase.spatial_parameterisations.items():
        if len(sps) != 1:
            errors.append(
                f"phase {phase_index} parameter '{param_name}': independent slice-wise "
                "identification currently requires exactly one spatial parameterisation."
            )
            continue

        sp = sps[0]
        if not isinstance(sp, SliceWiseSpatialParameterisation):
            if get_num_degrees_of_freedom(sps) != 0:
                errors.append(
                    f"phase {phase_index} parameter '{param_name}': all unknown parameters must "
                    "use SliceWiseSpatialParameterisation for independent slice-wise identification."
                )
            continue

        if sp.support is None:
            errors.append(
                f"phase {phase_index} parameter '{param_name}': SliceWiseSpatialParameterisation "
                "must define a slice support when used for independent slice-wise identification."
            )
            continue

        if sp.support is not slice_metric.support:
            errors.append(
                f"phase {phase_index} parameter '{param_name}': independent slice-wise identification "
                "requires this parameterisation to reference the same SupportSlice object as the "
                "slice-force metric."
            )

    return errors
