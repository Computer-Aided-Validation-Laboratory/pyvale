from copy import copy, deepcopy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from pyvale.vfm.constlaw import IConstitutiveLaw
from pyvale.vfm.experimentdata import (
    EdgeConditions,
    EEdgeCondition,
    ExperimentData,
)
from pyvale.vfm.metric import IMetric
from pyvale.vfm.normalisation import (
    denormalise_degree_of_freedom,
    normalise_degree_of_freedom,
)
from pyvale.vfm.spatialparam import ISpatialParameterisation
from pyvale.vfm.vfmesh import (
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
class SensitivityBasedVirtualFieldsMetric(IMetric):
    virtual_fields_mesh: VirtualFieldsMesh
    vf_scaling_fraction: float | None

    def __init__(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        region_of_interest: npt.NDArray[np.uint32],
        edge_conditions: EdgeConditions,
        mesh_size: npt.NDArray[np.uint32],
        # TODO: option to adjust fraction of largest timesteps used for calculating VF scaling factor
        vf_scaling_fraction: float | None = None
    ) -> None:

        self.virtual_fields_mesh = generate_virtual_fields_mesh(
            x,
            y,
            region_of_interest,
            edge_conditions,
            mesh_size
        )

        self.vf_scaling_fraction = vf_scaling_fraction

    def evaluate(
        self,
        stress: npt.NDArray[np.float64],
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, ISpatialParameterisation],
        experiment_data: ExperimentData,
    ) -> npt.NDArray[np.float64]:

        # Compute stress sensitivites for each DOF or constitutive parameter (depending on perturbation type)
        stress_sensitivities = self.calculate_stress_sensitivities(
            experiment_data.strain,
            stress,
            constitutive_law,
            parameter_map_size,
            spatial_parameterisations,
            experiment_data.delta_timesteps,
            perturbation_type = "constitutive_parameter",
        )
        
        # Generate sensitivity-based virtual fields (SBVF) from stress sensitivities
        sensitivity_based_virtual_fields = []
        for stress_sensitivity in stress_sensitivities:
            sensitivity_based_virtual_fields.append(
                generate_virtual_fields_from_mesh(
                    stress_sensitivity.total,  # TODO: option to use incremental stress sensitivities
                    self.virtual_fields_mesh
                )
            )

        # Reshape pixel area to be broadcastable with stress and virtual strain arrays
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


        residual_vector = []
        # Compute PVW residuals for each SBVF and concatenate into single residual vector
        for sbvf in sensitivity_based_virtual_fields:
            
            # Compute 4d IVW term for current SBVF
            internal_virtual_work_4d = (
                stress
                * sbvf.virtual_strain
                * pixel_area
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


        return np.concatenate(residual_vector)
    

    def calculate_stress_sensitivities(
        self,
        strain: npt.NDArray[np.float64],
        stress_reference: npt.NDArray[np.float64],
        constitutive_law: IConstitutiveLaw,
        parameter_map_size: npt.NDArray[np.uint32],
        spatial_parameterisations: dict[str, ISpatialParameterisation],
        delta_timesteps: npt.NDArray[np.float64],
        perturbation_type: str = "constitutive_parameter",   #TODO better as enum? 
        perturbation_factor_param: float = 0.15,   #TODO: single perturbation factor or separate for param and dof? 
        perturbation_factor_dof: float = 0.05,
        
    ) -> list[StressSensitivity]:
        """Calculate stress sensitivity objects for the provided spatial parameterisations.
        

        stress_sensitivities_dof: 
            - fixed additive step in normalised DOF space (e.g. 0.05 means perturbing by 5% of the full allowed range of the DOF)
            - comparing sensitivities across DOFs with different scales, so a fixed step in 
            normalised DOF space is a good simple choice


        stress_sensitivities_parameter: 
            - multiplicative step in physical parameter space (e.g. 0.15 means perturbing by 15% of the current parameter value)
            - comparing sensitivities of physical constitutive parameters, so a multiplicative step in 
            physical parameter space is a simple choice (that is used in Marek et al. 2023)


        The pertubation type determines how many downstream virtual fields (VF) will 
        be created by the SBVF metric:
            -"constitutive_parameter": one sensitivity history per active constitutive parameter, 
            so downstream nVF = nParameters.
            - "dof": one sensitivity history per active optimisation DOF, so downstream nVF = nDof.
        
        
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
        spatial_parameterisations: dict[str, ISpatialParameterisation],
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
    
    stress_sensitivities = []
    # Loop through each constitutive parameter
    for param_name, sp in spatial_parameterisations.items():
        dofs =  sp.collect_degrees_of_freedom()

        # Loop through each DOF of the current parameter
        for i, dof in enumerate(dofs):

            # Normalise DOF value to [0, 1] range based on its lower and upper bounds
            normalised_dof_value = normalise_degree_of_freedom(dof)

            # Default pertubation lowers DOF value, but if DOF is close to lower bound, perturb upwards instead to ensure effective perturbation
            if normalised_dof_value >= perturbation_factor:
                perturbed_dof_value_normalised = (normalised_dof_value - perturbation_factor)
            else:
                perturbed_dof_value_normalised = ( normalised_dof_value + perturbation_factor)

            # Denormalise perturbed DOF value back to physical space
            perturbed_dof_value = denormalise_degree_of_freedom(
                perturbed_dof_value_normalised,
                dof.lower_bound,
                dof.upper_bound
            )

            # Create copy of spatial parameterisation with perturbed DOF
            perturbed_spatial_parameterisations = deepcopy(spatial_parameterisations)
            perturbed_dof = copy(dof)                 # copy DegreeOfFreedom dataclass
            perturbed_dof.value = perturbed_dof_value # update DOF value to perturbed value
            perturbed_dofs = deepcopy(dofs)           # copy list of DOFs
            perturbed_dofs[i] = perturbed_dof         # update current perturbed DOF in list of DOFs            

            
            # Update spatial parameterisation using perturbed DOF
            perturbed_spatial_parameterisations[
                param_name
            ].update_from_degrees_of_freedom(perturbed_dofs)

            # Update spatial parameter maps using perturbed spatial parameterisation
            # TODO: can we just update the relevant parameter map for current dof?
            perturbed_spatial_parameter_maps = {
                parameter_name: sp.to_map(parameter_map_size)
                for parameter_name, sp
                in perturbed_spatial_parameterisations.items()
            }

            # Compute perturbed stress using perturbed spatial parameter maps
            perturbed_stress = constitutive_law.calculate_stress(
                strain, perturbed_spatial_parameter_maps
            )

            # Compute stress sensitivity as difference between reference stress and perturbed stress
            total_stress_sensitivity = stress_reference - perturbed_stress

            # Compute incremental stress sensitivity as difference in stress sensitivity between consecutive timesteps
            # First timestep has zero incremental sensitivity as there is no previous step to compare to
            incremental_stress_sensitivity = np.zeros_like(total_stress_sensitivity)
            incremental_stress_sensitivity[1:, :, :, :] = np.diff(
                total_stress_sensitivity,
                axis=0,
            )

            # Store sensitivities for current perturbed DOF
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
        spatial_parameterisations: dict[str, ISpatialParameterisation],
        delta_timesteps: npt.NDArray[np.float64],
        perturbation_factor: float,
    ) -> list[StressSensitivity]:
    """Calculate stress sensitivity maps for the provided spatial parameterisations by perturbing each parameter map datapoint.
    
    Perturb DOF in physical space

    value_perturbed = value * (1 - perturbation_factor)

    """

    stress_sensitivities = []
    # Loop through each constitutive parameter
    for param_name, sp in spatial_parameterisations.items():
        
        # If constitutive parameter is not being identified: skip
        if sp.get_num_degrees_of_freedom() == 0:
            continue

        # get parameter map for current spatial parameterisation
        map = sp.to_map(parameter_map_size)  

        # Perturb parameter map by multiplying by (1 - perturbation_factor)
        perturbed_map = map * (1 - perturbation_factor)

        # Create copy of original spatial parameterisation maps
        perturbed_spatial_parameter_maps = {
            parameter_name: sp.to_map(parameter_map_size)
            for parameter_name, sp
            in spatial_parameterisations.items()
        }

        # Update parameter map for current parameter with perturbed map
        perturbed_spatial_parameter_maps[param_name] = perturbed_map

        # Compute perturbed stress using perturbed spatial parameter maps
        perturbed_stress = constitutive_law.calculate_stress(
            strain, perturbed_spatial_parameter_maps
        )

        # Compute stress sensitivity as difference between reference stress and perturbed stress
        total_stress_sensitivity = stress_reference - perturbed_stress

        # Compute incremental stress sensitivity as difference in stress sensitivity between consecutive timesteps
        # First timestep has zero incremental sensitivity as there is no previous step to compare to
        incremental_stress_sensitivity = np.zeros_like(total_stress_sensitivity)
        incremental_stress_sensitivity[1:, :, :, :] = np.diff(
            total_stress_sensitivity,
            axis=0,
        )

        # Store sensitivities for current perturbed DOF
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
            plt.title(f'Perturbed stress: {param_name}, step {step}, component {component}')
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
            plt.title(f'Total stress sensitivity: {param_name}, step {step}, component {component}')
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
            plt.title(f'Incremental stress sensitivity: {param_name}, step {step}, component {component}')
            vmin = np.nanpercentile(img, 5)
            vmax = np.nanpercentile(img, 95)
            im3.set_clim(vmin, vmax)
            im3=plt.show()

    return stress_sensitivities
