from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.experimentdata import (
    EEdgeCondition,
    ExperimentData,
)
from pyvale.vfm.metric import IMetric, MetricResult
from pyvale.vfm.normalisation import (
    denormalise_degree_of_freedom,
    normalise_degree_of_freedom,
)
from pyvale.vfm.spatialparam import (
    ISpatialParameterisation,
    evaluate_parameterisations_to_map,
    get_num_degrees_of_freedom,
    PhaseSpatialState,
)
from pyvale.vfm.vfmesh import (
    GlobalVirtualFields,
    VirtualFieldsMesh,
    generate_virtual_fields_from_mesh,
    generate_virtual_fields_mesh,
)


@dataclass(slots=True)
class StressSensitivity:
    """Total and incremental stress sensitivity for one perturbation target.

    Both arrays use the constitutive-update layout
    ``(timesteps, components, y, x)``.
    """

    total: npt.NDArray[np.float64]
    incremental: npt.NDArray[np.float64]


@dataclass(slots=True)
class MetricSBVF(IMetric):
    """
    Sensitivity-based virtual fields (SBVF) metric.

    Constructs virtual fields automatically from the sensitivity of the
    stress field to each constitutive parameter, on a virtual mesh of the
    given size, and returns the residual between the internal and external
    virtual work as the objective to minimise
    """

    mesh_size: npt.NDArray[np.uint32] = field(
        default_factory=lambda: np.asarray((15, 15), dtype=np.uint32)
    )
    """Number of virtual-mesh elements along each axis, defaulting to ``[15, 15]``."""

    vf_scaling_fraction: float | None = 0.3
    """Fraction of the largest-magnitude timesteps used to compute the
    virtual-field scaling factor, defaulting to the top 30%. Set to ``None``
    to disable scaling."""

    stress_area_scale: float = 1.0
    """Convert ``stress * area`` to force for the experiment-data units.

    PyVale's VFM data contract uses mm, MPa and N, for which this is ``1``.
    Legacy SI-coordinate datasets with stress still expressed in MPa should
    supply ``1e6`` explicitly.
    """

    perturbation_type: Literal["constitutive_parameter", "dof"] = (
        "constitutive_parameter"
    )
    """Quantity perturbed to construct the stress sensitivities."""

    perturbation_factor: float = 0.15
    """Perturbation magnitude used by the selected perturbation type.

    Constitutive-parameter perturbations are relative to the current physical
    parameter map. DOF perturbations are additive in normalised DOF space.
    """

    _virtual_fields_mesh: VirtualFieldsMesh | None = field(
        default=None,
        init=False
    )

    # Internal and external virtual work vectors computed by the most recent
    # call to evaluate, stacked per virtual field with shape
    # (num_virtual_fields, timesteps). Both are None until evaluate has run
    _internal_virtual_work: npt.NDArray[np.float64] | None = field(
        default=None,
        init=False
    )
    _external_virtual_work: npt.NDArray[np.float64] | None = field(
        default=None,
        init=False
    )

    # Cached sensitivity-based virtual fields. They are (re)computed when the
    # cache is None or when _recompute_virtual_fields is True, and otherwise
    # reused across evaluations to avoid recomputing the stress sensitivities
    _sensitivity_based_virtual_fields: list[GlobalVirtualFields] | None = field(
        default=None,
        init=False
    )

    # Toggle whether to recompute sensitivity based virtual fields
    # on each metric evaluation
    _recompute_virtual_fields: bool = field(
        default=False,
        init=False
    )

    def __post_init__(self) -> None:
        if self.perturbation_type not in {"constitutive_parameter", "dof"}:
            raise ValueError(
                "perturbation_type must be 'constitutive_parameter' or 'dof'."
            )
        if not 0.0 < self.perturbation_factor < 1.0:
            raise ValueError("perturbation_factor must lie in (0, 1).")


    def initialise(
        self,
        experiment_data: ExperimentData
    ) -> None:
        x = experiment_data.specimen_geometry.x
        y = experiment_data.specimen_geometry.y
        roi = experiment_data.specimen_geometry.region_of_interest
        specimen_mask = roi.sample_specimen_mask(x, y)

        self._virtual_fields_mesh = generate_virtual_fields_mesh(
            x,
            y,
            specimen_mask,
            experiment_data.boundary_conditions.edge_conditions,
            self.mesh_size
        )
        # A prepare call follows every structural refinement. Rebuild the
        # SBVFs once on the next evaluation, then reuse them during the solve.
        self._sensitivity_based_virtual_fields = None

    def evaluate(
        self,
        stress: npt.NDArray[np.float64],
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
        experiment_data: ExperimentData,
    ) -> MetricResult:
        if self._virtual_fields_mesh is None:
            raise RuntimeError(
                "Virtual fields mesh has not been generated. "
                "initialise() must be called before evaluate()."
            )


        if (
            self._sensitivity_based_virtual_fields is None
            or self._recompute_virtual_fields
        ):
            # Compute stress sensitivites for each DOF or constitutive
            # parameter (depending on perturbation type)
            stress_sensitivities = self.calculate_stress_sensitivities(
                experiment_data.strain,
                stress,
                constitutive_law,
                parameter_map_size,
                spatial_parameterisations,
                experiment_data.delta_timesteps,
                perturbation_type=self.perturbation_type,
                perturbation_factor_param=self.perturbation_factor,
                perturbation_factor_dof=self.perturbation_factor,
            )

            # Generate sensitivity-based virtual fields (SBVF) from
            # stress sensitivities
            sensitivity_based_virtual_fields = []
            for stress_sensitivity in stress_sensitivities:
                sensitivity_based_virtual_fields.append(
                    # TODO: option to use incremental stress sensitivities
                    generate_virtual_fields_from_mesh(
                        stress_sensitivity.total,  
                        self._virtual_fields_mesh
                    )
                )

            self._sensitivity_based_virtual_fields = sensitivity_based_virtual_fields
        else:
            sensitivity_based_virtual_fields = self._sensitivity_based_virtual_fields

        # Reshape pixel area to be broadcastable with stress and
        # virtual strain arrays
        pixel_area = experiment_data.specimen_geometry.pixel_area[np.newaxis, np.newaxis, :, :]


        # Determine which edge index has traction boundary condition # TODO: tidy up
        for edge_name in ("min_x_edge", "max_x_edge", "min_y_edge", "max_y_edge"):
            edge = getattr(experiment_data.boundary_conditions.edge_conditions, edge_name)
            if edge.x is EEdgeCondition.Traction or edge.y is EEdgeCondition.Traction:
                traction_edge_name = edge_name
                break
        else:
            raise ValueError("No traction edge found")
        edge_to_index = {
            "min_y_edge": 0,
            "min_x_edge": 1,
            "max_y_edge": 2,
            "max_x_edge": 3,
        }
        traction_edge_index = edge_to_index[traction_edge_name]


        # Store per-virtual-field IVW/EVW vectors so they can be accessed later
        internal_virtual_work_vectors = []
        external_virtual_work_vectors = []

        residual_vector = []

        # Compute PVW residuals for each SBVF and concatenate into single residual vector
        for sbvf in sensitivity_based_virtual_fields:
            # Compute 4d IVW term for current SBVF
            # TODO: we have a 1e6 term here as stress in in MPa,
            #   and pixel area is in m^2, would be nice to avoid
            #   having this magic number, maybe rescale stress into Pa
            #   at the start of the func?
            internal_virtual_work_4d = (
                stress
                * sbvf.virtual_strain
                * pixel_area * self.stress_area_scale
                * experiment_data.specimen_geometry.thickness
            )

            # Set any NaN values in IVW to zero
            internal_virtual_work_4d = np.nan_to_num(
                internal_virtual_work_4d,
                nan=0
            )

            # Sum IVW across spatial dimensions and components to get single IVW scalar for each timestep
            internal_virtual_work_vector = np.sum(
                internal_virtual_work_4d,
                axis=(1, 2, 3)
            )

            # Compute 4d EVW term for current SBVF 
            # force_x * virtual_displacement_x + force_y * virtual_displacement_y on traction edge 
            # summed to get single EVW scalar for each timestep
            external_virtual_work_vector = (
                experiment_data.boundary_conditions.force[:, 0]
                * sbvf.virtual_displacement_edge[:, 0, traction_edge_index]
                + experiment_data.boundary_conditions.force[:, 1]
                * sbvf.virtual_displacement_edge[:, 1, traction_edge_index]
            )

            # Store the (unscaled) IVW/EVW vectors for the current virtual field
            internal_virtual_work_vectors.append(internal_virtual_work_vector)
            external_virtual_work_vectors.append(external_virtual_work_vector)

            if self.vf_scaling_fraction is not None:
                # Compute number of timesteps to use for scaling based on the chosen fraction (1 step min).
                num_timesteps_used_for_scaling = max(
                    1, 
                    int(np.floor(len(external_virtual_work_vector) * self.vf_scaling_fraction)),
                )

                # Select the timesteps with the largest absolute IVW values.
                # NumPy sorts ascending, so take the last n indices.
                largest_ivw_indices = np.argsort(np.abs(internal_virtual_work_vector))[-num_timesteps_used_for_scaling:]

                # Compute scaling factor as the reciprocal of the mean absolute IVW
                # over the selected timesteps.
                mean_abs_ivw_for_scaling = np.mean(
                    np.abs(internal_virtual_work_vector[largest_ivw_indices])
                )
                if mean_abs_ivw_for_scaling != 0.0:
                    vw_scaling_factor = 1.0 / mean_abs_ivw_for_scaling
                else:
                    vw_scaling_factor = 1.0

                # Scale the full PVW residual for the current virtual field.
                residual_vector.append(
                    (internal_virtual_work_vector - external_virtual_work_vector)
                    * vw_scaling_factor
                )
            else:
                residual_vector.append(
                    internal_virtual_work_vector - external_virtual_work_vector
                )


        # Stack per-virtual-field vectors into (num_virtual_fields, timesteps)
        # arrays and store them for later access.
        self._internal_virtual_work = np.array(internal_virtual_work_vectors)
        self._external_virtual_work = np.array(external_virtual_work_vectors)

        residual = np.concatenate(residual_vector)

        return MetricResult(residual)


    def calculate_stress_sensitivities(
        self,
        strain: npt.NDArray[np.float64],
        stress_reference: npt.NDArray[np.float64],
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
        delta_timesteps: npt.NDArray[np.float64],
        perturbation_type: str = "constitutive_parameter",   #TODO better as enum? 
        perturbation_factor_param: float = 0.15,   #TODO: single perturbation factor or separate for param and dof? 
        perturbation_factor_dof: float = 0.05,
    ) -> list[StressSensitivity]:
        """
        Calculate stress sensitivity objects for the provided spatial
        parameterisations.

        DOF perturbation (``perturbation_factor_dof``):

        * fixed additive step in normalised DOF space (e.g. 0.05 perturbs by
          5% of the full allowed range of the DOF)
        * a fixed step in normalised DOF space is a good simple choice when
          comparing sensitivities across DOFs with different scales

        Parameter perturbation (``perturbation_factor_param``):

        * multiplicative step in physical parameter space (e.g. 0.15 perturbs
          by 15% of the current parameter value)
        * a multiplicative step in physical parameter space is a simple choice
          when comparing sensitivities of physical constitutive parameters
          (as used in Marek et al. 2023)

        The perturbation type determines how many downstream virtual fields
        (VF) are created by the SBVF metric:

        * ``"constitutive_parameter"``: one sensitivity history per active
          constitutive parameter, so downstream nVF = nParameters
        * ``"dof"``: one sensitivity history per active optimisation DOF, so
          downstream nVF = nDof
        """

        if perturbation_type == "constitutive_parameter":
            stress_sensitivities = _calculate_stress_sensitivities_parameter(
                strain,
                stress_reference,
                constitutive_law,
                parameter_map_size,
                spatial_parameterisations,
                delta_timesteps,
                perturbation_factor_param
            )
        elif perturbation_type == "dof":
            stress_sensitivities = _calculate_stress_sensitivities_dof(
                strain,
                stress_reference,
                constitutive_law,
                parameter_map_size,
                spatial_parameterisations,
                delta_timesteps,
                perturbation_factor_dof
            )
        else:
            raise ValueError(
                f"Invalid perturbation type: {perturbation_type}. "
                "Supported types are 'constitutive_parameter' and 'dof'."
            )

        return stress_sensitivities


def _calculate_stress_sensitivities_dof(
        strain: npt.NDArray[np.float64],
        stress_reference: npt.NDArray[np.float64],
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
        delta_timesteps: npt.NDArray[np.float64],
        perturbation_factor: float,
    ) -> list[StressSensitivity]:
    """Calculate stress sensitivity maps for the provided spatial parameterisations by perturbing each DOF.

    Perturb DOF in normalised space to ensure consistent perturbation factor across different DOF types and ranges

    normalised_perturbed = normalised_dof - perturbation_factor

    Note in this case, the pertubation factor is a factor of the full allowed range of the DOF rather 
    than the current DOF value.
    Hence, 0.05 means perturbing by 5% of the full allowed range of the DOF, not 5% of current value.

    """

    phase_spatial_state = PhaseSpatialState(spatial_parameterisations)
    phase_degrees_of_freedom = phase_spatial_state.collect_degrees_of_freedom()
    normalised_phase_degrees_of_freedom = (
        phase_spatial_state.collect_normalised_degrees_of_freedom()
    )

    stress_sensitivities = []
    for dof_index, dof in enumerate(phase_degrees_of_freedom):
        normalised_dof_value = normalise_degree_of_freedom(dof)

        # Default perturbation lowers the DOF value, but if the value is
        # already close to the lower bound perturb upwards instead so the
        # perturbation magnitude remains effective.
        if normalised_dof_value >= perturbation_factor:
            perturbed_dof_value_normalised = (
                normalised_dof_value - perturbation_factor
            )
        else:
            perturbed_dof_value_normalised = (
                normalised_dof_value + perturbation_factor
            )

        perturbed_phase_degrees_of_freedom = (
            normalised_phase_degrees_of_freedom.copy()
        )
        perturbed_phase_degrees_of_freedom[dof_index] = (
            perturbed_dof_value_normalised
        )

        perturbed_phase_spatial_state = phase_spatial_state.copy()
        perturbed_phase_spatial_state.update_from_normalised_degrees_of_freedom(
            perturbed_phase_degrees_of_freedom
        )

        perturbed_spatial_parameter_maps = (
            perturbed_phase_spatial_state.evaluate_parameter_maps(
                parameter_map_size
            )
        )

        perturbed_stress = constitutive_law.calculate_stress(
            strain,
            perturbed_spatial_parameter_maps,
        )

        # Compute stress sensitivity as difference between reference stress and
        # perturbed stress.
        total_stress_sensitivity = stress_reference - perturbed_stress

        # Compute incremental stress sensitivity as difference in stress
        # sensitivity between consecutive timesteps. The first timestep has
        # zero incremental sensitivity as there is no previous step to compare
        # to.
        incremental_stress_sensitivity = np.zeros_like(total_stress_sensitivity)
        incremental_stress_sensitivity[1:, :, :, :] = np.diff(
            total_stress_sensitivity,
            axis=0,
        )

        stress_sensitivities.append(
            StressSensitivity(
                total=total_stress_sensitivity,
                incremental=incremental_stress_sensitivity,
            )
        )

        plot_debug = False
        if plot_debug:
            # Debug: plot perturbed stress
                step = 14
                component = 0
                img = perturbed_stress[step, component, :, :]   # 10th step, 1st component, all y, all x
                plt.figure()
                im1 = plt.imshow(img, aspect='auto', origin='lower', cmap='viridis')
                plt.colorbar(label='Stress')
                plt.xlabel('x')
                plt.ylabel('y')
                #include param name and dof index in title
                plt.title(f'Perturbed stress: {param_name}, DOF {i}, step {step}, component {component}')
                vmin = np.nanpercentile(img, 5)
                vmax = np.nanpercentile(img, 95)
                im1.set_clim(vmin, vmax)
                im1=plt.show()

                # Debug: plot total SS stress
                img = total_stress_sensitivity[step, component, :, :]   # 10th step, 1st component, all y, all x
                plt.figure()
                im2 = plt.imshow(img, aspect='auto', origin='lower', cmap='viridis')
                plt.colorbar(label='Stress')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(f'Total stress sensitivity: {param_name}, DOF {i}, step {step}, component {component}')
                vmin = np.nanpercentile(img, 5)
                vmax = np.nanpercentile(img, 95)
                im2.set_clim(vmin, vmax)
                im2=plt.show()

                # Debug: plot incremental SS stress
                img = incremental_stress_sensitivity[step, component, :, :]   # 10th step, 1st component, all y, all x
                plt.figure()
                im3 = plt.imshow(img, aspect='auto', origin='lower', cmap='viridis')
                plt.colorbar(label='Stress')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(f'Incremental stress sensitivity: {param_name}, DOF {i}, step {step}, component {component}')
                vmin = np.nanpercentile(img, 5)
                vmax = np.nanpercentile(img, 95)
                im3.set_clim(vmin, vmax)
                im3=plt.show()

    return stress_sensitivities


def _calculate_stress_sensitivities_parameter(
        strain: npt.NDArray[np.float64],
        stress_reference: npt.NDArray[np.float64],
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, list[ISpatialParameterisation]],
        delta_timesteps: npt.NDArray[np.float64],
        perturbation_factor: float,
) -> list[StressSensitivity]:
    """Calculate stress sensitivity maps for the provided spatial parameterisations by perturbing each parameter map datapoint.

    Perturb DOF in physical space

    value_perturbed = value * (1 - perturbation_factor)

    """

    parameter_maps = {
        parameter_name: evaluate_parameterisations_to_map(sps, parameter_map_size)
        for parameter_name, sps in spatial_parameterisations.items()
    }
    active_parameter_names = tuple(
        parameter_name
        for parameter_name, sps in spatial_parameterisations.items()
        if get_num_degrees_of_freedom(sps) > 0
    )
    sensitivities = calculate_parameter_stress_sensitivities(
        strain,
        stress_reference,
        constitutive_law,
        parameter_maps,
        active_parameter_names,
        perturbation_factor,
    )
    return list(sensitivities.values())


def calculate_parameter_stress_sensitivities(
    strain: npt.NDArray[np.float64],
    stress_reference: npt.NDArray[np.float64],
    constitutive_law: IConstitutiveLaw,
    parameter_maps: dict[str, npt.NDArray[np.float64]],
    active_parameter_names: tuple[str, ...] | list[str],
    perturbation_factor: float = 0.15,
) -> dict[str, StressSensitivity]:
    """Return total and incremental stress sensitivities for active parameters.

    Each active parameter map is reduced by the same relative fraction while
    all other maps remain fixed. Inputs are copied before perturbation, so the
    accepted phase-start parameter maps are never mutated.
    """

    if not 0.0 < perturbation_factor < 1.0:
        raise ValueError("perturbation_factor must lie in (0, 1).")
    if not active_parameter_names:
        raise ValueError("At least one active parameter name is required.")

    reference = np.asarray(stress_reference, dtype=np.float64)
    resolved_maps = {
        name: np.asarray(parameter_map, dtype=np.float64)
        for name, parameter_map in parameter_maps.items()
    }
    sensitivities: dict[str, StressSensitivity] = {}
    for parameter_name in active_parameter_names:
        if parameter_name in sensitivities:
            raise ValueError(f"Duplicate active parameter name '{parameter_name}'.")
        if parameter_name not in resolved_maps:
            raise KeyError(
                f"Active parameter '{parameter_name}' does not have a parameter map."
            )

        perturbed_maps = {
            name: parameter_map.copy()
            for name, parameter_map in resolved_maps.items()
        }
        perturbed_maps[parameter_name] *= 1.0 - perturbation_factor
        perturbed_stress = np.asarray(
            constitutive_law.calculate_stress(strain, perturbed_maps),
            dtype=np.float64,
        )
        if perturbed_stress.shape != reference.shape:
            raise ValueError(
                "Perturbed and reference stress shapes differ: "
                f"{perturbed_stress.shape} vs {reference.shape}."
            )

        total = reference - perturbed_stress
        incremental = np.zeros_like(total)
        incremental[1:] = np.diff(total, axis=0)
        sensitivities[parameter_name] = StressSensitivity(
            total=total,
            incremental=incremental,
        )

    return sensitivities


def calculate_local_parameter_stress_sensitivity(
    strain: npt.NDArray[np.float64],
    stress_reference: npt.NDArray[np.float64],
    constitutive_law: IConstitutiveLaw,
    parameter_maps: dict[str, npt.NDArray[np.float64]],
    parameter_name: str,
    perturbation_factor: float = 0.01,
) -> npt.NDArray[np.float64]:
    """Estimate pointwise ``d(stress)/d(parameter)`` with one map solve.

    This is valid for PyVale's pointwise constitutive updates: perturbing the
    full map yields independent local stress responses, which are divided by
    the local physical perturbation before metric adjoints are applied.
    """
    sensitivity = calculate_parameter_stress_sensitivities(
        strain,
        stress_reference,
        constitutive_law,
        parameter_maps,
        [parameter_name],
        perturbation_factor,
    )[parameter_name].total
    delta = (
        perturbation_factor
        * np.asarray(parameter_maps[parameter_name], dtype=np.float64)
    )
    return np.divide(
        sensitivity,
        delta[np.newaxis, np.newaxis],
        out=np.zeros_like(sensitivity),
        where=np.abs(delta[np.newaxis, np.newaxis]) > np.finfo(np.float64).eps,
    )
