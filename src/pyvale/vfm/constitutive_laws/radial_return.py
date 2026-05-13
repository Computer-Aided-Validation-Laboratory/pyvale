import enum

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constitutive_laws.hardening import EHardening, hardening_func

class EUnloading(enum.Enum):
    NoCompensation = enum.auto()
    ConstantStrain = enum.auto()
    LinearExtrapolation = enum.auto()


# TODO: update docstring
def radial_return(
    strain: npt.NDArray[np.float64],
    constitutive_parameter_maps: dict[str, npt.NDArray[np.float64]],
    hardening: EHardening,
    error_tolerance: float = 1e-8,
    iteration_limit: int = 100,
    unloading: EUnloading = EUnloading.ConstantStrain,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.bool_],
    npt.NDArray[np.float64]
]:
    """Radial return mapping for J2 plasticity in plane stress.

    Parameters
    ----------
    strain : np.ndarray
        Strain tensor history with shape (timesteps, components, y, x).
        Component order is [eps_xx, eps_yy, eps_xy] where eps_xy is
        tensorial shear strain (converted internally to engineering shear).
    mechanical_properties : MechanicalProperties
        Material properties and selected constitutive hardening law.
    error_tolerance : float, optional
        Normalized Newton-Raphson convergence tolerance.
    iteration_limit : int, optional
        Maximum number of Newton-Raphson iterations per timestep.
    unloading : EUnloading, optional
        Output-only unloading compensation mode:
        - NoCompensation: no output correction on unloading
        - ConstantStrain: hold previous output stress (default)
        - LinearExtrapolation: extrapolate from two previous outputs

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        stress_output : shape (timesteps, 3, y, x)
            Reported stress history [sig_xx, sig_yy, sig_xy].
        equivalent_stress : shape (timesteps, y, x)
            Von Mises equivalent stress.
        yield_map : shape (timesteps, y, x), dtype=bool
            True where a point is plastic at a timestep.
        equivalent_plastic_strain : shape (timesteps, y, x)
            Accumulated equivalent plastic strain.


    Overview
    -------    
    This function applies radial return mappingto integrate J2 plasticity 
    with isotropic hardening over a strain history.

    Algorithm Overview:
        1. Form elastic trial stress from strain increment via Hooke's law
        2. Check if trial stress violates yield criterion: f(sig_trial) = sqrt(3*J2) - sigma_Y
        3. If elastic (f < 0): accept trial stress as the final stress for this increment
        4. If plastic (f >= 0): solve nonlinear consistency condition via Newton-Raphson iteration
           to find plastic multiplier plastic_multiplier such that updated stress satisfies:
           f(sig_corrected) = 0 (stress returns to yield surface)
        5. Update equivalent plastic strain using: p_eq_new = p_eq_old + plastic_multiplier * sqrt(2/3 * J2)
        6. Update yield stress from hardening law: sigma_Y = sigma_Y(p_eq_new)
        7. Repeat until consistency condition is satisfied within error_tolerance
        8. Move to next timestep with updated stress and plastic strain as initial conditions
        9. optionally smooth output stress to correct for unloading

    Future enchancements:
    # TODO: support non linear geometries
    # TODO: support tangent modulus output
    # TODO: require inputs to be flattened 2d arrays, to prevent wasted 
    #       computation on reshaping (or perform reshaping once at start of function
    #       Quick test of removing reshaping suggests it accounts for ~ 5% of time. 

    Key Equations:
        Elastic trial stress (plane stress): sig_trial = sig_prev + D : delta_eps
            where D is plane stress stiffness matrix with terms like E/(1-nu^2)
        Yield criterion (J2 plasticity): f = sqrt(3*J2(sig)) - sigma_Y(p_eq) <= 0
            where J2 = (1/2)*dev(sig):dev(sig) and dev(sig) is deviatoric stress
        ksi variable (effective stress squared measure): 
            ksi = (1/6)*(sig_xx + sig_yy)^2 + (1/2)*(sig_yy - sig_xx)^2 + 2*(sig_xy)^2
            This represents 3*J2 in the plane stress projected formulation
        Consistency/plastic criterion residual:
            plastic_criterion = (1/2)*ksi - (1/3)*sigma_Y^2
            Driven to zero to ensure stress lies on yield surface
        Newton-Raphson update for plastic_multiplier:
            plastic_multiplier_new = plastic_multiplier_old - plastic_criterion / (d(plastic_criterion)/d(plastic_multiplier))
        Corrected stress from plastic multiplier:
            sig_corrected = A* : sig_trial
            where A* factors account for stress reduction proportional to plastic_multiplier
            A11* = 3(1-nu) / (3(1-nu) + E*plastic_multiplier)
            A22* = 1 / (1 + 2*G*plastic_multiplier)
            G is shear modulus, E is elastic modulus
        Equivalent plastic strain evolution:
            p_eq = p_eq_old + plastic_multiplier * sqrt(2/3) * sqrt(ksi)
            Accumulates total inelastic deformation magnitude
    
    Spatial Variation: Material parameters may vary spatially through parameter maps. Each datapoint
    is evaluated independently in the Newton loop.
    
    Load History: The unloading compensation is output-only. The internal constitutive
    state (`stress_state`) is never patched.

    """

    # Check inputs
    if strain.ndim != 4 or strain.shape[1] != 3:
        raise ValueError("strain must have shape (timesteps, 3, y, x)")

    # == UNPACK COMMON VARIABLES == 
    num_timesteps = strain.shape[0]
    size_y = strain.shape[2]
    size_x = strain.shape[3]
    num_datapoints = size_y * size_x

    # Elastic modulus and poissons ratio are always required to compute trial elastic stress
    elastic_modulus = constitutive_parameter_maps["elastic_modulus"]
    poissons_ratio = constitutive_parameter_maps["poissons_ratio"]

    # == COMPUTE PLANE STRESS FACTOR (AND / OR ELASTIC STIFFNESS MATRIX) ==
    shear_modulus = elastic_modulus / (2 * (1 + poissons_ratio))     

    # Plane stress factor used rather than elastic stiffness matrix to more easily
    # support spatially varying material properties without needing to construct
    # a full elastic stiffness matrix for each point.
    plane_stress_factor = elastic_modulus / (1 - poissons_ratio ** 2)

    elastic_modulus_flat = elastic_modulus.ravel()
    poissons_ratio_flat = poissons_ratio.ravel()
    shear_modulus_flat = shear_modulus.ravel()
 
    # Old: explicit elastic stiffness matrix for plane stress with engineering shear strain (only required if 
    #       we want to do the elastic predictor with a single matrix multiply or output the stiffness matrix for some reason)
    # elastic_stiffness = elastic_modulus / (1 - poissons_ratio ** 2) * (
    #     np.array([
    #         [1.0, poissons_ratio, 0.0],
    #         [poissons_ratio, 1.0, 0.0],
    #         [0.0, 0.0, 0.5 * (1.0 - poissons_ratio)],
    #     ])
    # )


    # == COMPUTE INCREMENTAL STRAINS ==
    # The return-mapping algorithm works with strain increments. 
    incremental_strain = np.zeros_like(strain) 
    incremental_strain[0, :, :, :] = strain[0, :, :, :] # First timestep uses the total strain as the first increment
    incremental_strain[1:, :, :, :] = np.diff(strain, axis=0)

    # Convert tensorial shear strain component to engineering shear strain
    # The constitutive matrix D is written using engineering shear strain gamma_xy,
    # while the incoming field is assumed to contain tensorial shear epsilon_xy. 
    # Need to confirm that the Python strain input is provided as tensorial strain.
    incremental_strain[:, 2, :, :] *= 2    


    # == INITIALISE VARIABLES FOR RADIAL RETURN ==
    stress_state = np.zeros_like(strain)  # Internal constitutive state used by each step's trial-stress predictor
    stress_output = np.zeros_like(strain)  # Reported stress history returned to the caller
    equivalent_plastic_strain = np.zeros((num_timesteps, num_datapoints))
    yield_map = np.zeros((num_timesteps, size_y, size_x), dtype=bool)

    plastic_multiplier = np.zeros(num_datapoints)  # Plastic multiplier increment; main unknown solved in the return-mapping step
    delta_ksi_plastic_multiplier = np.zeros(num_datapoints)  # Derivative of ksi wrt plastic_multiplier; used in Newton-Raphson update
    ksi = np.zeros(num_datapoints)  # Internal scalar in plane-stress J2 update linked to effective stress / PEEQ update
    plastic_criterion = np.zeros(num_datapoints)  # Yield/consistency residual; driven to zero for plastic dataoints
    plastic_criterion_prime = np.zeros(num_datapoints)  # Derivative of plastic_criterion wrt plastic_multiplier; used in Newton update
    h_bar = np.zeros(num_datapoints)  # Hardening term in consistency derivative; accounts for evolving yield stress
    prev_plasticity_mask = np.zeros(num_datapoints, dtype=bool)  # Mask of points that were plastic at previous timestep; used for load history/unloading
    yield_stress_measure = np.zeros(num_datapoints)  # Stores sigma_vm^2 / 3, matching the MATLAB yield-check convention
    error = np.zeros(num_datapoints)  # Normalised residual used as the Newton-Raphson convergence measure

    # loop through timesteps
    for t in range(num_timesteps):
        # Define index of previous timestep (t=0 uses t_prev=0, falling back to zero-initialized state)
        t_prev = max(0, t - 1)
        # Freeze previous-step PEEQ for this entire timestep. This must stay
        # constant during Newton iterations; otherwise at t=0 we'd get
        # self-referential updates of the form PEEQ[0] = PEEQ[0] + dPEEQ.
        prev_equivalent_plastic_strain = equivalent_plastic_strain[t_prev, :].copy()

        # Initialize current equivalent plastic strain from previous timestep
        equivalent_plastic_strain[t, :] = prev_equivalent_plastic_strain


        # == COMPUTE TRIAL STRESS ==
        # Assume the entire current strain increment is elastic and compute the
        # corresponding trial stress from Hooke's law.
        # sig_xx = sig_xx_prev + E / (1 - nu**2) * (delta_eps_xx + nu * delta_eps_yy)

        # xx component of trial stress at each point (shape: y, x)
        # sig_xx = sig_xx_prev + plane_stress_factor * (delta_eps_xx + nu * delta_eps_yy)
        trial_stress_xx = stress_state[t_prev, 0, :, :] + plane_stress_factor * (
            incremental_strain[t, 0, :, :] + poissons_ratio * incremental_strain[t, 1, :, :]
        )

        # yy component of trial stress at each point (shape: y, x)
        # sig_yy = sig_yy_prev + plane_stress_factor * (delta_eps_yy + nu * delta_eps_xx)
        trial_stress_yy = stress_state[t_prev, 1, :, :] + plane_stress_factor * (
            poissons_ratio * incremental_strain[t, 0, :, :] + incremental_strain[t, 1, :, :]
        )

        # xy component of trial stress at each point (shape: y, x)
        # sig_xy = sig_xy_prev + G * delta_gamma_xy  (note: shear strain is engineering shear strain, so no factor of 2 needed here)
        trial_stress_xy = stress_state[t_prev, 2, :, :] + shear_modulus * incremental_strain[t, 2, :, :]
       
        # assemble trial stress into a single array  (shape: component, y, x)
        trial_stress = np.stack((trial_stress_xx, trial_stress_yy, trial_stress_xy), axis=0)
        # reshape trial stress to shape (datapoints, component) 
        trial_stress_flat = np.moveaxis(trial_stress, 0, -1).reshape(num_datapoints, 3)


        # == CHECK YIELD CRITERION ==
        # Note: for heterogeneous material properties, evaluate yield pointwise.

        # Compute the yield stress measure (3*J2 in the plane stress projected formulation) 
        # for the trial stress state at each point. This is used to evaluate the yield criterion 
        # and determine which points have yielded.
        yield_stress_measure[:] = 1/3 * (
            trial_stress_xx.ravel() ** 2
            + trial_stress_yy.ravel() ** 2
            - trial_stress_xx.ravel() * trial_stress_yy.ravel()
            + (3 * trial_stress_xy.ravel() ** 2)
        )

        # Compute yield stress for current plastic strain using the active
        # hardening law. The tangent term may be None for laws that do not
        # expose a simple constant hardening modulus.
        yield_stress, _ = hardening_func(
            hardening,
            constitutive_parameter_maps,
            prev_equivalent_plastic_strain,
        )

        # Check yield criterion for each point
        yield_criterion_check = yield_stress_measure > (1/3 * (yield_stress ** 2))
        # Define mask of which data points have yielded (plastic: point as yielded, elastic: point has not yielded)
        plasticity_mask = yield_criterion_check


        # == COMPUTE PLASTIC MULTIPLIER TO CORRECT THE STRESS STATE FOR PLASTIC POINTS ==
        # For points that have yielded, we need to solve for the plastic_multiplier such 
        # that the updated stress state lies on the yield surface. This is done using a 
        # Newton-Raphson iteration to solve the nonlinear consistency condition. 
        # The main unknown in this iteration is plastic_multiplier, which controls how 
        # much we reduce the trial stress back to the yield surface. The iteration 
        # continues until the yield residual (plastic_criterion) is sufficiently small, 
        # indicating that we've found a stress state that satisfies the yield condition 
        # for the current hardening state.
        
        # Compute initial ksi for all points
        # ksi is an internal scalar used in the plane-stress projected J2 (see de Souza Neto et al. 2008, section 3.6.2) 
        # ksi = (1/6)*(sig_xx + sig_yy)^2 + (1/2)*(sig_yy - sig_xx)^2 + 2*(sig_xy)^2
        # This is the projected invariant used by this plane-stress return-mapping implementation.
        ksi[plasticity_mask] = (
            1 / 6 * np.sum(trial_stress_flat[plasticity_mask, 0:2], axis=1) ** 2
            + 0.5 * (trial_stress_flat[plasticity_mask, 1] - trial_stress_flat[plasticity_mask, 0]) ** 2
            + 2 * trial_stress_flat[plasticity_mask, 2] ** 2
        )

        # Compute plastic criterion 
        # Plastic criterion is the residual of the consistency condition (difference between
        # the current effective stress measure ksi and the yield stress squared).
        # If plastic criterion is nonzero, the current stress state is not exactly on the yield surface, 
        # and we need to iterate on plastic_multiplier to reduce the stress back to the yield surface.
        plastic_criterion[plasticity_mask] = (
            0.5 * ksi[plasticity_mask]
            - 1 / 3 * yield_stress[plasticity_mask] ** 2
        )

        # Compute initial error for Newton-Raphson iteration; this is the normalised residual 
        # that drives convergence. We normalise by ksi to avoid issues with points that have 
        # very small effective stress measures, which could otherwise lead to artificially 
        # small residuals and premature convergence.
        error.fill(0)
        error[plasticity_mask] = np.abs(plastic_criterion[plasticity_mask])
        error[plasticity_mask] = error[plasticity_mask] / ksi[plasticity_mask]

        # Newton-Raphson iteration to solve for plastic_multiplier such that plastic_criterion -> 0
        # stress sum and difference terms are trial-state invariants used in derivatives below
        # and remain fixed during Newton updates. Hence, compute once before iteration.
        trial_stress_sum_sq = np.sum(trial_stress_flat[:, 0:2], axis=1) ** 2
        trial_stress_diff_sq = (trial_stress_flat[:, 1] - trial_stress_flat[:, 0]) ** 2
        # Set current stress state to the trial elastic stress; this will be updated for plastic points after the return-mapping iteration
        stress_flat = trial_stress_flat.copy()
        i = 0
        while np.any(error > error_tolerance) and (i < iteration_limit):
            # Compute derivative of plastic criterion with respect to plastic_multiplier
            delta_ksi_plastic_multiplier_all = (
                -elastic_modulus_flat / (1 - poissons_ratio_flat) * trial_stress_sum_sq
                / (
                    9
                    * (
                        1
                        + elastic_modulus_flat * plastic_multiplier
                        / (3 * (1 - poissons_ratio_flat))
                    ) ** 3
                )
                - 2
                * shear_modulus_flat
                * (trial_stress_diff_sq + 4 * stress_flat[:, 2] ** 2)
                / (1 + 2 * shear_modulus_flat * plastic_multiplier) ** 3
            )

            # Only update plastic points as elastic points have plastic_multiplier = 0 and 
            # do not need to be updated in the iteration
            delta_ksi_plastic_multiplier[plasticity_mask] = (
                delta_ksi_plastic_multiplier_all[plasticity_mask]
            )

            # Compute the increment in equivalent plastic strain from the current plastic multiplier.
            # The relationship delta_p_eq = plastic_multiplier * sqrt(2/3 * J2) comes from the flow rule
            # in J2 plasticity: d(eps_p) = plastic_multiplier * d(f)/d(sig), where the norm of the
            # deviatoric stress gives sqrt(2/3 * J2). The ksi variable represents 3*J2 in the
            # plane stress formulation, so sqrt(2/3 * ksi) = sqrt(2*J2) which scales plastic_multiplier
            # appropriately to accumulate the equivalent plastic strain magnitude.
            # physical interpration: when plastic deformation occurs, the plastic strain increment
            # points in the direction normal to the yield surface in stress space.  (i.e. the direction of plastic
            # strain is determined by the gradient of the yield function wrt stress - which is equal to
            # sqrt(2/3 * J2) in the case of J2 plasticity). The plastic multiplier scales this increment 
            # so that the stress state is returned to the yield surface. 
            #
            # In this plane-stress projected J2 formulation, ksi is the projected invariant with:
            #     ksi = 2*J2
            # Hence:
            #     delta_equivalent_plastic_strain = plastic_multiplier * sqrt(2/3 * ksi)
            # The factor sqrt(2/3 * ksi) is a scalar stress-magnitude term related to the current
            # deviatoric stress level. It is not the plastic flow direction. The flow direction is
            # given by the gradient of the yield function (normal to the yield surface), while this
            # expression gives the scalar magnitude of equivalent plastic strain accumulated.
            delta_equivalent_plastic_strain = plastic_multiplier * np.sqrt(2 / 3 * ksi)

            # Update equivalent plastic strain for plastic points using the current plastic_multiplier 
            # and ksi values. This is needed to evaluate the hardening law and its derivative in the consistency condition.
            equivalent_plastic_strain[t, plasticity_mask] = (
                prev_equivalent_plastic_strain[plasticity_mask]
                + delta_equivalent_plastic_strain[plasticity_mask]
            )

            # Compute current yield stress and hardening variable using current equivalent plastic strain. 
            (
                yield_stress,
                delta_yield_stress_delta_equivalent_plastic_strain
            ) = hardening_func(
                hardening,
                constitutive_parameter_maps,
                equivalent_plastic_strain[t, :],
            )

            # Compute h_bar (hardening term in the consistency condition derivative) for
            # plastic points only, avoiding 0/0 at ksi == 0 for non-plastic points.
            m = plasticity_mask
            sqrt_ksi_m = np.sqrt(ksi[m])
            safe_denom = np.where(sqrt_ksi_m > 0, 2.0 * sqrt_ksi_m, 1.0)
            h_bar[m] = 2.0 * (
                yield_stress[m]
                * delta_yield_stress_delta_equivalent_plastic_strain[m]
                * np.sqrt(2.0 / 3.0)
                * (sqrt_ksi_m + plastic_multiplier[m] * delta_ksi_plastic_multiplier[m] / safe_denom)
            )

            # Compute derivative of plastic criterion wrt plastic multiplier
            plastic_criterion_prime[plasticity_mask] = (
                0.5 * delta_ksi_plastic_multiplier[plasticity_mask]
                - 1 / 3 * h_bar[plasticity_mask]
            )

            # Compute plastic multiplier update using Newton-Raphson update
            # plastic_multiplier_new = plastic_multiplier_old - plastic_criterion / plastic_criterion_prime
            plastic_multiplier[plasticity_mask] = (
                plastic_multiplier[plasticity_mask]
                - plastic_criterion[plasticity_mask]
                / plastic_criterion_prime[plasticity_mask]
            )

            # == COMPUTE EQUIVALENT PLASTIC STRAIN USING IDENTIFIED PLASTIC MULTIPLIER ==
            # Compute updated ksi (effective stress measure) using the current plastic_multiplier. 
            ksi_all = (
                trial_stress_sum_sq
                / (
                    6
                    * (
                        1
                        + elastic_modulus_flat * plastic_multiplier
                        / (3 * (1 - poissons_ratio_flat))
                    ) ** 2
                )
                + (0.5 * trial_stress_diff_sq + 2 * stress_flat[:, 2] ** 2)
                / (1 + 2 * shear_modulus_flat * plastic_multiplier) ** 2
            )
            ksi_all = np.maximum(ksi_all, 0)
            ksi[plasticity_mask] = ksi_all[plasticity_mask]

            # Compute updated plastic strain increment from the current plastic multiplier and ksi values. 
            delta_equivalent_plastic_strain = plastic_multiplier * np.sqrt(2 / 3 * ksi)

            # Compute updated equivalent plastic strain
            equivalent_plastic_strain[t, plasticity_mask] = (
                prev_equivalent_plastic_strain[plasticity_mask]
                + delta_equivalent_plastic_strain[plasticity_mask]
            )


            # == COMPUTE UPDATED YIELD STRESS USING UPDATED EQUIVALENT PLASTIC STRAIN ==
            yield_stress, _ = hardening_func(
                hardening,
                constitutive_parameter_maps,
                equivalent_plastic_strain[t, :],
            )


            # == EVALUATE PLASTIC CRITERION (CONSISTENCY RESIDUAL) WITH UPDATED VARIABLES ==
            # Computer plastic criterion using the updated effective stress measure ksi 
            # and yield stress to evaluate the current plastic criterion (consistency residual). 
            # This is the value we are driving to zero in the Newton-Raphson iteration.
            plastic_criterion[plasticity_mask] = (
                0.5 * ksi[plasticity_mask]
                - 1 / 3 * yield_stress[plasticity_mask] ** 2
            )

            # Compute updated error for Newton-Raphson iteration; this is the normalised residual 
            # that drives convergence. We normalise by ksi to avoid issues with points that have 
            # very small effective stress measures, which could otherwise lead to artificially 
            # small residuals and premature convergence.
            error.fill(0)
            error[plasticity_mask] = np.abs(plastic_criterion[plasticity_mask])
            error[plasticity_mask] = error[plasticity_mask] / ksi[plasticity_mask]

            i += 1
            # Check iteration limit
            if i == iteration_limit:
                print(
                    "The convergence has not been achieved within "
                    f"{iteration_limit} iterations in step {t}"
                )


        # == COMPUTE THE CORRECTED STRESS STATE USING THE DETERMINED PLASTIC MULTIPLIER == 
        # Compute stress correction factors that scale the trial stress back to the yield surface
        # correction_factor_1 applies to normal stress components (xx, yy)
        # correction_factor_2 applies to shear stress component (xy)
        correction_factor_1 = (
            3 * (1 - poissons_ratio_flat)
            / (3 * (1 - poissons_ratio_flat) + elastic_modulus_flat * plastic_multiplier)
        )
        correction_factor_2 = 1 / (1 + 2 * shear_modulus_flat * plastic_multiplier)
        
        # Compute average and difference of correction factors for stress transformation
        correction_factor_avg = 0.5 * (correction_factor_1 + correction_factor_2)
        correction_factor_diff = 0.5 * (correction_factor_1 - correction_factor_2)

        # Apply correction factors to trial stress to obtain corrected stress on yield surface.
        stress_flat_corrected = np.column_stack(
            (
            correction_factor_avg * stress_flat[:, 0] + correction_factor_diff * stress_flat[:, 1],
            correction_factor_diff * stress_flat[:, 0] + correction_factor_avg * stress_flat[:, 1],
            correction_factor_2 * stress_flat[:, 2],
            )
        )

        # Keep elastic trial points and overwrite only yielded points with corrected stresses.
        stress_flat[plasticity_mask] = stress_flat_corrected[plasticity_mask]

        # Internal state: the return-mapped stress for all points, never patched by unloading.
        # This is what the next step's trial-stress predictor must use.
        stress_state[t, :, :, :] = np.moveaxis(
            stress_flat.reshape(size_y, size_x, 3),
            -1,
            0,
        )


        # == HANDLE UNLOADING: OUTPUT-ONLY CORRECTION TO SMOOTH DISCONTINUITIES ==
        # Noise in strain data may result in points that were plastic in the previous step 
        # being classified as elastic in the current step, which can lead to artificial 
        # stress discontinuities. 
        # We have three options for how to handle this unloading compensation:
        # 1) no compensation: use the current step's trial elastic stress for points that have unloaded
        # 2) constant strain: for points that were plastic in the previous step but are
        #    now elastic (unloading), report the previous step's stress state rather than the current trial elastic stress. 
        # 3) linear extrapolation: for points that were plastic in the previous step but
        #    are now elastic (unloading), report a linearly extrapolated stress state based on the previous two steps' stress states.
        # Note, only the output stress is patched on unloading, not the internal state used 
        # by the predictor (so the stress state progresses according to the return-mapping 
        # algorithm regardless of unloading, but the reported output stress can be 
        # smoothed on unloading to avoid discontinuities).
        stress_output[t] = stress_state[t].copy()
        match unloading:
            case EUnloading.NoCompensation:
                pass  # keep return-mapped current-step output without unloading correction
            case EUnloading.ConstantStrain:
                if t > 0: 
                    unload_mask = prev_plasticity_mask & (~plasticity_mask)  # Points that were plastic in the previous step but are now elastic
                    if np.any(unload_mask):
                        out_flat = np.moveaxis(stress_output[t], 0, -1).reshape(num_datapoints, 3)
                        prev_out_flat = np.moveaxis(stress_output[t - 1], 0, -1).reshape(num_datapoints, 3)
                        out_flat[unload_mask] = prev_out_flat[unload_mask]
                        stress_output[t] = np.moveaxis(out_flat.reshape(size_y, size_x, 3), -1, 0)

            case EUnloading.LinearExtrapolation:
                if t > 1: 
                    unload_mask = prev_plasticity_mask & (~plasticity_mask)  # Points that were plastic in the previous step but are now elastic
                    if np.any(unload_mask):
                        out_flat = np.moveaxis(stress_output[t], 0, -1).reshape(num_datapoints, 3)
                        prev_out_flat = np.moveaxis(stress_output[t - 1], 0, -1).reshape(num_datapoints, 3)
                        prev2_out_flat = np.moveaxis(stress_output[t - 2], 0, -1).reshape(num_datapoints, 3)
                        out_flat[unload_mask] = (prev_out_flat[unload_mask] + (prev_out_flat[unload_mask] - prev2_out_flat[unload_mask]) )
                        stress_output[t] = np.moveaxis(out_flat.reshape(size_y, size_x, 3), -1, 0)
            case _:
                # Defensive fallback; validated once before the timestep loop.
                raise ValueError(
                    f"Invalid unloading option '{unloading}'. Supported options: "
                    "'no_compensation', 'constant_strain', 'linear_extrapolation'"
                )
        
        yield_map[t] = plasticity_mask.reshape(size_y, size_x)
        prev_plasticity_mask = plasticity_mask

        # Scratch workspace reset (temporary buffers reused each timestep)
        plastic_multiplier.fill(0)
        delta_ksi_plastic_multiplier.fill(0)
        ksi.fill(0)
        plastic_criterion.fill(0)
        plastic_criterion_prime.fill(0)
        h_bar.fill(0)
        error.fill(0)

    sig_xx = stress_output[:, 0]
    sig_yy = stress_output[:, 1]
    sig_xy = stress_output[:, 2]
    equivalent_stress = np.sqrt(sig_xx**2 + sig_yy**2 - sig_xx * sig_yy + 3 * sig_xy**2)

    return (
        stress_output,
        equivalent_stress,
        yield_map,
        np.reshape(equivalent_plastic_strain, (num_timesteps, size_y, size_x)),
    )
