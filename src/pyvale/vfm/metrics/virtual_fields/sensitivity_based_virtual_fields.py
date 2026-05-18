from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constitutive_laws.constitutive_law import ConstitutiveLaw
from pyvale.vfm.experiment_data import EdgeConditions, ExperimentData
from pyvale.vfm.metrics.metric import Metric
from pyvale.vfm.metrics.virtual_fields.virtual_fields_mesh import (
    VirtualFieldsMesh,
    generate_virtual_fields_mesh,
    generate_virtual_fields_from_mesh,
)
from pyvale.vfm.spatial_parameterisations.spatial_parameterisation import (
    DegreeOfFreedom,
    SpatialParameterisation,
)


@dataclass(slots=True)
class SensitivityBasedVirtualFieldsMetric(Metric):
    virtual_fields_mesh: VirtualFieldsMesh

    def __init__(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        region_of_interest: npt.NDArray[np.uint32],
        edge_conditions: EdgeConditions,
        mesh_size: npt.NDArray[np.uint32],
    ) -> None:
        # TODO: input shape and size checking where appropriate

        self.virtual_fields_mesh = generate_virtual_fields_mesh(
            x,
            y,
            region_of_interest,
            edge_conditions,
            mesh_size
        )

    def evaluate(
        self,
        stress: npt.NDArray[np.float64],
        constitutive_law: ConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, SpatialParameterisation],
        experiment_data: ExperimentData
    ) -> npt.NDArray[np.float64]:
        stress_sensitivities = self.calculate_stress_sensitivities(
            experiment_data.strain,
            stress,
            constitutive_law,
            parameter_map_size,
            spatial_parameterisations,
            experiment_data.delta_timesteps
        )

        # TODO: option to use incremental stress sensitivities
        sensitivity_based_virtual_fields = [
            generate_virtual_fields_from_mesh(
                total_stress_sensitivity,
                self.virtual_fields_mesh
            )
            for (total_stress_sensitivity, _) in stress_sensitivities
        ]

        pixel_area = experiment_data.specimen_geometry.pixel_area[np.newaxis, np.newaxis, :, :]

        evaluations = []
        for sbvf in sensitivity_based_virtual_fields:
            internal_virtual_work = (
                stress
                * sbvf.virtual_strain
                * pixel_area
                * experiment_data.specimen_geometry.thickness
            )

            internal_virtual_work = np.nan_to_num(
                internal_virtual_work,
                nan=0
            )

            internal_virtual_work = np.sum(
                internal_virtual_work,
                axis=(1, 2, 3)
            )

            # TODO: add option for configurable traction edge, or can we
            #   calc this somehow?
            external_virtual_work = (
                experiment_data.boundary_conditions.force[:, 0]
                * sbvf.virtual_displacement_edge[:, 0, 3]
                + experiment_data.boundary_conditions.force[:, 1]
                * sbvf.virtual_displacement_edge[:, 1, 3]
            )

            # TODO: add option for scaling
            evaluations.append(internal_virtual_work - external_virtual_work)

        return np.concatenate(evaluations)


    # TODO: need to normalise
    #   2 types:
    #     normalise dofs
    #     normalise params themselves
    # TODO: perturbation on params rather than dofs
    def calculate_stress_sensitivities(
        self,
        strain: npt.NDArray[np.float64],
        stress_reference: npt.NDArray[np.float64],
        constitutive_law: ConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, SpatialParameterisation],
        delta_timesteps: npt.NDArray[np.float64]
    ) -> list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        # TODO: make this a config option
        perturbation_factor = 0.15

        stress_sensitivities = []

        for param_name, sp in spatial_parameterisations.items():
            sp_dofs =  sp.collect_degrees_of_freedom()

            for i, dof in enumerate(sp_dofs):
                perturbed_dof_value = (
                    dof.value * (1 - perturbation_factor)
                )

                if perturbed_dof_value < dof.lower_bound:
                    perturbed_dof_value = dof.lower_bound

                elif perturbed_dof_value > dof.upper_bound:
                    perturbed_dof_value = dof.upper_bound

                perturbed_sp_dofs = deepcopy(sp_dofs)

                perturbed_sp_dofs[i] = DegreeOfFreedom(
                    perturbed_dof_value,
                    dof.lower_bound,
                    dof.upper_bound
                )

                perturbed_spatial_parameterisations = deepcopy(
                    spatial_parameterisations
                )

                perturbed_spatial_parameterisations[
                    param_name
                ].update_from_degrees_of_freedom(perturbed_sp_dofs)

                perturbed_spatial_parameter_maps = {
                    parameter_name: sp.to_map(parameter_map_size)
                    for parameter_name, sp
                    in perturbed_spatial_parameterisations.items()
                }

                perturbed_stress = constitutive_law.calculate_stress(
                    strain, perturbed_spatial_parameter_maps
                )

                total_stress_sensitivity = stress_reference - perturbed_stress

                incremental_stress_sensitivity = np.zeros_like(
                    total_stress_sensitivity
                )

                incremental_stress_sensitivity[1:, :, :, :] = np.diff(
                    total_stress_sensitivity,
                    axis=0,
                )

                incremental_stress_sensitivity = (
                    incremental_stress_sensitivity
                    / delta_timesteps[:, np.newaxis, np.newaxis, np.newaxis]
                )

                stress_sensitivities.append(
                    (total_stress_sensitivity, incremental_stress_sensitivity)
                )

        return stress_sensitivities
