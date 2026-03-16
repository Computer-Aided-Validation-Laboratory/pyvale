import numpy as np
import numpy.typing as npt

from pyvale.vfm.mechanical_properties import (
    MechanicalProperties,
    ParameterName,
    parameter_to_map,
    parameter_to_scalar,
)


# Receiving strain as a 4d array with the convention (timestep, component, y, x)
# Returning stress as a 4d array with the convention (timestep, component, y, x)
# TODO: finish docstring
# TODO: support non linear geometries
# TODO: update incremental strain to 4d
# TODO: remove order F and assume everything to be row major
# TODO: create the framework to support different kinds of hardening
#       (hardeningfun in the matlab)
# TODO: understand the different phases of this algorithm and pull them
#       out into helper functions
# Phases of the algorithm:
#   - Compute trial elastic stress
#   - Check yield criterion
#   - Compute plastic multiplier
#   - Update internal variables
#   - Tangent modulus update (is this part of our algo?)
def radial_return(
    strain: npt.NDArray[np.float64],
    mechanical_properties: MechanicalProperties,
    error_tolerance=1e-8,
    iteration_limit=100
) -> npt.NDArray[np.float64]:
    """Brief description

    Parameters
    ----------
    strain : np.ndarray
        Shape is (timesteps x datapoints x components)

    Returns
    -------
    np.ndarray
        Description


    Overview
    -------
    # ROB- High-level radial return logic:
    # ROB- 1) form an elastic trial stress from the strain increment,
    # ROB- 2) check whether that trial stress violates the yield condition,
    # ROB- 3) if not, accept the elastic trial state,
    # ROB- 4) if yes, solve for the plastic multiplier delta_lambda so the
    # ROB-    updated stress lies back on the yield surface,
    # ROB- 5) update equivalent plastic strain and yield stress from the
    # ROB-    hardening law,
    # ROB- 6) move to the next timestep using the updated internal state.


    """

    # UNPACK KEY INPUT DATA
    num_timesteps = strain.shape[0]
    num_components = strain.shape[1]
    size_y = strain.shape[2]
    size_x = strain.shape[3]
    num_datapoints = size_y * size_x
    parameters = mechanical_properties.parameters
    
    # ROB - Move the below into seperate function for hardening (some hardening laws may not use yield strength or hardening modulus, although linear hardening does)
    yield_strength_param = parameters[ParameterName.YieldStrength]
    hardening_modulus_param = parameters[ParameterName.HardeningModulus]
    yield_strength = parameter_to_map(yield_strength_param, size_x, size_y)
    hardening_modulus = parameter_to_map(hardening_modulus_param, size_x, size_y)


    # CONSTRUCT ELASTIC STIFFNESS MATRIX
    # TODO: in non homogeneous params, elastic mod and poissons ratio
    #       wont be scalars
    elastic_modulus_param = parameters[ParameterName.ElasticModulus]
    poissons_ratio_param = parameters[ParameterName.PoissonsRatio]
    elastic_modulus = parameter_to_scalar(elastic_modulus_param)
    poissons_ratio = parameter_to_scalar(poissons_ratio_param)
    # Compute shear modulus
    shear_modulus = elastic_modulus / (
        2 * (1 + poissons_ratio)
    )

    elastic_stiffness = elastic_modulus / (1 - poissons_ratio ** 2) * (
        np.array([
            [1.0, poissons_ratio, 0.0],
            [poissons_ratio, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - poissons_ratio)],
        ])
    )


    # INITIALISE VARIABLES FOR RADIAL RETURN
    # TODO: some of these could have more descriptive names?
    # TODO: could some of these be grouped up like stress and prev
    #       stress above? might give some clarity on which things
    #       are related?
    stress = np.zeros_like(strain)
    prev_stress = np.zeros((num_components, size_y, size_x))
    trial_stress = np.zeros((num_components, size_y, size_x))
    delta_lambda              = np.zeros(num_datapoints)
    delta_ksi_delta_lambda    = np.zeros(num_datapoints)
    ksi                       = np.zeros(num_datapoints)
    plastic_criterion         = np.zeros(num_datapoints)
    plastic_criterion_prime   = np.zeros(num_datapoints)
    h_bar                     = np.zeros(num_datapoints)
    prev_plasticity_mask      = np.zeros(num_datapoints)
    equivalent_stress         = np.zeros(num_datapoints)
    error                     = np.zeros(num_datapoints)

    # ROB- stress at all timesteps for all comps/pts; stores final corrected stress after each increment
    stress = np.zeros_like(strain)
    # ROB- stress at end of previous timestep; starting state for current increment
    prev_stress = np.zeros((num_components, size_y, size_x))
    # ROB- elastic predictor stress for current increment before yield check / plastic correction
    trial_stress = np.zeros((num_components, size_y, size_x))
    # ROB- plastic multiplier increment; main unknown solved in the return-mapping step
    delta_lambda = np.zeros(num_datapoints)
    # ROB- derivative of ksi wrt delta_lambda; used in Newton-Raphson update
    delta_ksi_delta_lambda = np.zeros(num_datapoints)
    # ROB- internal scalar in plane-stress J2 update linked to effective stress / PEEQ update
    ksi = np.zeros(num_datapoints)
    # ROB- yield/consistency residual; driven to zero for plastically active points
    plastic_criterion = np.zeros(num_datapoints)
    # ROB- derivative of plastic_criterion wrt delta_lambda; used in Newton update
    plastic_criterion_prime = np.zeros(num_datapoints)
    # ROB- hardening term in consistency derivative; accounts for evolving yield stress
    h_bar = np.zeros(num_datapoints)
    # ROB- mask of points that were plastic at previous timestep; used for load history/unloading
    prev_plasticity_mask = np.zeros(num_datapoints)
    # ROB- trial von Mises equivalent stress measure used in the yield check
    equivalent_stress = np.zeros(num_datapoints)
    # ROB- normalised residual used as the Newton-Raphson convergence measure
    error = np.zeros(num_datapoints)

    # TODO: should we switch these to be (timesteps, datapoints) to fit
    #       with our conventions?
    equivalent_plastic_strain = np.zeros((num_datapoints, num_timesteps))
    prev_equivalent_plastic_strain = np.zeros((num_datapoints))
    sigma_xx                  = np.zeros((num_datapoints, num_timesteps))
    sigma_xy                  = np.zeros((num_datapoints, num_timesteps))
    sigma_yy                  = np.zeros((num_datapoints, num_timesteps))
    von_mises_stress          = np.zeros((num_datapoints, num_timesteps))


    # COMPUTE INCREMENTAL STRAINS
    # ROB- The constitutive update is increment-based, so even though the input
    # ROB- strain is total strain over time, the return-mapping algorithm works
    # ROB- with strain increments. First timestep uses the total strain as the
    # ROB- first increment; later timesteps use np.diff(..., axis=0).
    incremental_strain = np.zeros_like(strain)
    incremental_strain[0, :, :, :] = strain[0, :, :, :]
    incremental_strain[1:, :, :, :] = np.diff(strain, axis=0)

    # ROB- The shear strain increment is doubled because the constitutive
    # ROB- matrix D is written using engineering shear strain gamma_xy,
    # ROB- while the incoming field is assumed to contain tensorial shear
    # ROB- epsilon_xy. Need to confirm that the Python strain input is provided
    # ROB- as tensorial strain.
    # Convert tensorial shear strain component to engineering shear strain
    incremental_strain[:, 2, :, :] *= 2     



    # LOOP THROUGH TIMESTEPS
    for t in range(num_timesteps):
        equivalent_plastic_strain[:, t] = prev_equivalent_plastic_strain

        # COMPUTE TRIAL STRESS
        # ROB- assume the entire current strain increment is elastic and compute the
        # ROB- corresponding trial stress from Hooke's law.
        # TODO: need to reshape incremental strain, previous form was
        #       (35708, 3), so the x and y were flattened together
        # ROB - Agreed, unsure below is correct. Also have a look at tensordot multiplication operation?
        trial_stress = prev_stress + (
            incremental_strain[t, :, :, :].dot(elastic_stiffness)
        )

        # COMPUTE UPDATED YIELD STRENGTH
        # ROB- update to use subfunction to allow various hardening laws
        yield_stress = yield_strength.ravel() + (
            hardening_modulus.ravel() * prev_equivalent_plastic_strain
        )

        # ROB- unsure the if block is required?
        if t == 0:
            stress[0, :, :] = incremental_strain[0, :, :].dot(elastic_stiffness)

            # Flatten with column major ordering
            yield_stress = yield_strength.flatten(order="F") + (
                hardening_modulus.flatten(order="F")
                * equivalent_plastic_strain[:, 0]
            )
        else:
            prev_stress = stress[t - 1, :, :]

            stress[t, :, :] = prev_stress + (
                incremental_strain[t, :, :].dot(elastic_stiffness)
            )

            yield_stress = yield_strength.flatten(order="F") + (
                hardening_modulus.flatten(order="F")
                * equivalent_plastic_strain[:, t - 1]
            )

            equivalent_plastic_strain[:, t] = (
                equivalent_plastic_strain[:, t - 1]
            )


        # COMPUTE EQUIVALENT STRESS
        equivalent_stress[:] = 1/3 * (
            stress[t, :, 0] ** 2
            + stress[t, :, 1] ** 2
            - stress[t, :, 0] * stress[t, :, 1]
            + (3 * stress[t, :, 2] ** 2)
        )

        # CHECK YIELD CRITERION
        yield_criterion_check = equivalent_stress > (1/3 * (yield_stress ** 2))
        # Define mask of which data points have yielded (plastic: point as yielded, elastic: point has not yielded)
        plasticity_mask = yield_criterion_check
        elasticity_mask = np.logical_not(plasticity_mask)


        # ROB- ksi is an internal scalar used in the plane-stress projected J2
        # ROB- formulation from the MATLAB code / de Souza Neto reference.
        # ROB- It is related to the effective stress measure used inside the
        # ROB- nonlinear consistency equation for delta_lambda.

        # ROB- plastic_criterion is the consistency / yield residual.
        # ROB- If it is nonzero, the current stress state is not exactly on the
        # ROB- yield surface, so we iterate on delta_lambda until the residual is
        # ROB- sufficiently small.


        ksi[plasticity_mask] = (
            1/6 * np.sum(stress[t, plasticity_mask, 0:2], axis=1) ** 2
            + 0.5
            * (stress[t, plasticity_mask, 1]
                - stress[t, plasticity_mask, 0]) ** 2
            + 2 * (stress[t, plasticity_mask, 2]) ** 2
        )

        plastic_criterion[plasticity_mask] = (
            1/2 * ksi[plasticity_mask] - 1/3
            * yield_stress[plasticity_mask] ** 2
        )

        error[plasticity_mask] = plastic_criterion[plasticity_mask]
        error[plasticity_mask] = error[plasticity_mask] / ksi[plasticity_mask]

        stress_sum = np.sum(stress[t, :, 0:2], axis=1) ** 2
        stress_diff = (stress[t, :, 1] - stress[t, :, 0]) ** 2


        # ROB- Newton-Raphson loop for the plastic corrector:
        # ROB- solve for delta_lambda such that the updated stress satisfies the
        # ROB- yield condition together with the hardening law.
        # ROB- This is the core nonlinear part of the return-mapping algorithm.
        i = 0
        while np.any(error > error_tolerance) and (i < iteration_limit):
            delta_ksi_delta_lambda_all = (
                -elastic_modulus / (1 - poissons_ratio) * stress_sum
                / (
                    9 * (1 + elastic_modulus * delta_lambda
                    / (3 * (1 - poissons_ratio))) ** 3
                )
                - 2 * shear_modulus * (stress_diff + 4 * stress[t, :, 2] ** 2)
                / (1 + 2 * shear_modulus * delta_lambda) ** 3
            )

            delta_ksi_delta_lambda[plasticity_mask] = (
                delta_ksi_delta_lambda_all[plasticity_mask]
            )

            delta_equivalent_plastic_strain = delta_lambda * np.sqrt(2/3 * ksi)

            if t == 0:
                equivalent_plastic_strain[plasticity_mask, t] = (
                    delta_equivalent_plastic_strain[plasticity_mask]
                )
            else:
                equivalent_plastic_strain[plasticity_mask, t] = (
                    equivalent_plastic_strain[plasticity_mask, t - 1]
                    + delta_equivalent_plastic_strain[plasticity_mask]
                )

            yield_stress = yield_strength.flatten(order="F") + (
                hardening_modulus.flatten(order="F")
                * equivalent_plastic_strain[:, t]
            )

            delta_yield_stress_delta_equivalent_plastic_strain = (
                hardening_modulus.flatten(order="F")
            )

            h_bar_all = 2 * (
                yield_stress
                * delta_yield_stress_delta_equivalent_plastic_strain
                * np.sqrt(2/3)
                * (
                    np.sqrt(ksi)
                    + delta_lambda * delta_ksi_delta_lambda / (2 * np.sqrt(ksi))
                )
            )

            h_bar[plasticity_mask] = h_bar_all[plasticity_mask]

            plastic_criterion_prime[plasticity_mask] = (
                1/2 * delta_ksi_delta_lambda[plasticity_mask]
            ) - (1/3 * h_bar[plasticity_mask])

            delta_lambda[plasticity_mask] = delta_lambda[plasticity_mask] - (
                plastic_criterion[plasticity_mask]
                / plastic_criterion_prime[plasticity_mask]
            )

            ksi_all = (
                stress_sum / (6 * (
                    1 + elastic_modulus * delta_lambda
                    / (3 * (1 - poissons_ratio))
                ) ** 2)
                + (0.5 * stress_diff + 2 * stress[t, :, 2] ** 2)
                / (1 + 2 * shear_modulus * delta_lambda) ** 2
            )

            # Eliminate any negatives
            ksi_all = np.maximum(ksi_all, 0)
            ksi[plasticity_mask] = ksi_all[plasticity_mask]

            delta_equivalent_plastic_strain = delta_lambda * np.sqrt(2/3 * ksi)
            if t == 0:
                equivalent_plastic_strain[plasticity_mask, t] = (
                    delta_equivalent_plastic_strain[plasticity_mask]
                )
            else:
                equivalent_plastic_strain[plasticity_mask, t] = (
                    equivalent_plastic_strain[plasticity_mask, t - 1]
                    + delta_equivalent_plastic_strain[plasticity_mask]
                )

            yield_stress = yield_strength.flatten(order="F") + (
                hardening_modulus.flatten(order="F")
                * equivalent_plastic_strain[:, t]
            )

            delta_yield_stress_delta_equivalent_plastic_strain = (
                hardening_modulus.flatten(order="F")
            )

            plastic_criterion[plasticity_mask] = 0.5 * ksi[plasticity_mask] - (
                (1/3) * (yield_stress[plasticity_mask] ** 2)
            )

            error.fill(0)
            error[plasticity_mask] = np.abs(plastic_criterion[plasticity_mask])

            # Normalise by effective stress
            error[plasticity_mask] = (
                error[plasticity_mask]
                / ksi[plasticity_mask]
            )

            i += 1

            if i == (iteration_limit - 1):
                print((
                    f"The convergence has not been achieved within"
                    f"{iteration_limit} iterations in step {t}"
                 ))
                
            # end while loop


        # COMPUTE THE CORRECTED STRESS STATE USING THE DETERMINED PLASTIC MULTIPLIER (delta_lambda)
        # ROB- These A* factors apply the actual plastic stress correction after
        # ROB- the scalar plastic multiplier has converged.
        # ROB- Conceptually: start from the elastic trial stress and reduce it to
        # ROB- the admissible stress on the yield surface for the current hardening
        # ROB- state.
        a11_star = (
            3 * (1 - poissons_ratio)
            / (3 * (1 - poissons_ratio) + elastic_modulus * delta_lambda)
        )
        a22_star = 1 / (1 + 2 * shear_modulus * delta_lambda)
        a_sum = 0.5 * (a11_star + a22_star)
        a_diff = 0.5 * (a11_star - a22_star)

        stress[t, :, :] = np.column_stack(
            (
                a_sum * stress[t, :, 0] + a_diff * stress[t, :, 1],
                a_diff * stress[t, :, 0] + a_sum * stress[t, :, 1],
                a22_star * stress[t, :, 2],
            )
        )

        # Update stress variables with elastic values
        sigma_xx[elasticity_mask, t] = stress[t, elasticity_mask, 0]
        sigma_yy[elasticity_mask, t] = stress[t, elasticity_mask, 1]
        sigma_xy[elasticity_mask, t] = stress[t, elasticity_mask, 2]
        von_mises_stress[elasticity_mask, t] = np.sqrt(
            3 * equivalent_stress[elasticity_mask]
        )

        # Update stress variables with plastic values
        sigma_xx[plasticity_mask, t] = stress[t, plasticity_mask, 0]
        sigma_yy[plasticity_mask, t] = stress[t, plasticity_mask, 1]
        sigma_xy[plasticity_mask, t] = stress[t, plasticity_mask, 2]
        von_mises_stress[plasticity_mask, t] = yield_stress[plasticity_mask]

        if t > 0:
            mask = prev_plasticity_mask & (~plasticity_mask)
            sigma_xx[mask, t] = sigma_xx[mask, t - 1]
            sigma_yy[mask, t] = sigma_yy[mask, t - 1]
            sigma_xy[mask, t] = sigma_xy[mask, t - 1]

        prev_stress = stress[t, :, :, :]
        # TODO: swap t around when we change peeq shape
        prev_equivalent_plastic_strain = equivalent_plastic_strain[:, t]
        prev_plasticity_mask = plasticity_mask

        delta_lambda.fill(0)
        delta_ksi_delta_lambda.fill(0)
        ksi.fill(0)
        plastic_criterion.fill(0)
        plastic_criterion_prime.fill(0)
        equivalent_stress.fill(0)

    return (sigma_xx, sigma_xy, sigma_yy, von_mises_stress)





# TODO: delete below if we dont need it
# @dataclass(slots=True)
# class Stress:
#     xx: npt.NDArray[np.float64]
#     xy: npt.NDArray[np.float64]
#     yy: npt.NDArray[np.float64]
#     von_mises: npt.NDArray[np.float64]

# # TODO: do we need eqvStress like original impl?
# # Convert stress to 4D tensor of the shape (x_points, y_points, components, timestep]
# def convert_stress_to_4d(
#     stress: Stress,
#     dic_config: DICConfig
# ) -> npt.NDArray[np.float64]:
#     # TODO: might be different for non linear geometry
#     num_stress_components = 3

#     stress_4d = np.zeros((
#         dic_config.x_dimension,
#         dic_config.y_dimension,
#         num_stress_components,
#         dic_config.timesteps.size
#     ))

#     component_dimensions = (
#         dic_config.x_dimension,
#         dic_config.y_dimension,
#         # 1,
#         dic_config.timesteps.size
#     )

#     stress_4d[:, :, 0, :] = np.reshape(
#         stress.xx,
#         component_dimensions,
#         order="F"
#     )

#     stress_4d[:, :, 1, :] = np.reshape(
#         stress.yy,
#         component_dimensions,
#         order="F"
#     )

#     stress_4d[:, :, 2, :] = np.reshape(
#         stress.xy,
#         component_dimensions,
#         order="F"
#     )

#     return stress_4d
