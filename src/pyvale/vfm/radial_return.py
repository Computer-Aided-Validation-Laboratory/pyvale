from dataclasses import astuple, dataclass

import numpy as np
import numpy.typing as npt


@dataclass(slots=True)
class MaterialProperties:
    youngs_modulus: float
    poissons_ratio: float
    yield_strength: npt.NDArray[np.float64]
    hardening_modulus: npt.NDArray[np.float64]


def radial_return(
    strain: npt.NDArray[np.float64], material_properties: MaterialProperties
):
    """Brief description

    Parameters
    ----------
    strain : np.ndarray
        Shape is (timesteps x datapoints x components)

    Returns
    -------
    np.ndarray
        Description
    """

    num_timesteps = strain.shape[0]
    num_datapoints = strain.shape[1]

    (
        youngs_modulus,
        poissons_ratio,
        yield_strength,
        hardening_modulus
    )= astuple(material_properties)

    # TODO: should these be passed in?
    error_tolerance = 1e-8
    iteration_limit = 100

    delta_lambda              = np.zeros(num_datapoints)
    delta_ksi_delta_lambda    = np.zeros(num_datapoints)
    ksi                       = np.zeros(num_datapoints)
    plastic_criterion         = np.zeros(num_datapoints) # flyt
    plastic_criterion_prime   = np.zeros(num_datapoints) # flyt_prime
    h_bar                     = np.zeros(num_datapoints)
    prev_plasticity_mask      = np.zeros(num_datapoints)
    equivalent_stress         = np.zeros(num_datapoints)
    error                     = np.zeros(num_datapoints)
    equivalent_plastic_strain = np.zeros((num_datapoints, num_timesteps)) # peeq
    sigma_xx                  = np.zeros((num_datapoints, num_timesteps))
    sigma_xy                  = np.zeros((num_datapoints, num_timesteps))
    sigma_yy                  = np.zeros((num_datapoints, num_timesteps))
    von_mises_stress          = np.zeros((num_datapoints, num_timesteps))
    incremental_strain        = np.zeros_like(strain)
    stress                    = np.zeros_like(strain)

    shear_modulus = youngs_modulus / (
        2 * (1 + poissons_ratio)
    )

    # Valid for Engineering Shear Strain only!
    # elasticity_matrix
    elasticity = youngs_modulus / ( 1 - poissons_ratio ** 2 ) * np.array([
        [1.0, poissons_ratio, 0.0],
        [poissons_ratio, 1.0, 0.0],
        [0.0, 0.0, (0.5 * (1.0 - poissons_ratio))],
    ])

    incremental_strain[0, :, :] = strain[0, :, :]
    incremental_strain[1:, :, :] = np.diff(strain, axis=0)

    # Convert shear strain component from tensorial shear strain to engineering
    # shear strain. By convention, tensorial shear strain is half engineering
    # shear strain
    incremental_strain[:, :, 2] *= 2

    for t in range(num_timesteps):
        if t == 0:
            stress[0, :, :] = incremental_strain[0, :, :].dot(elasticity)

            # Flatten with column major ordering
            yield_stress = yield_strength.flatten(order="F") + (
                hardening_modulus.flatten(order="F")
                * equivalent_plastic_strain[:, 0]
            )
        else:
            prev_stress = stress[t - 1, :, :]

            stress[t, :, :] = prev_stress + (
                incremental_strain[t, :, :].dot(elasticity)
            )

            yield_stress = yield_strength.flatten(order="F") + (
                hardening_modulus.flatten(order="F")
                * equivalent_plastic_strain[:, t - 1]
            )

            equivalent_plastic_strain[:, t] = (
                equivalent_plastic_strain[:, t - 1]
            )

        equivalent_stress[:] = 1/3 * (
            stress[t, :, 0] ** 2
            + stress[t, :, 1] ** 2
            - stress[t, :, 0] * stress[t, :, 1]
            + (3 * stress[t, :, 2] ** 2)
        )

        yield_criterion_check = equivalent_stress > (1/3 * (yield_stress ** 2))

        plasticity_mask = yield_criterion_check
        elasticity_mask = np.logical_not(plasticity_mask)

        sigma_xx[elasticity_mask, t] = stress[t, elasticity_mask, 0]
        sigma_yy[elasticity_mask, t] = stress[t, elasticity_mask, 1]
        sigma_xy[elasticity_mask, t] = stress[t, elasticity_mask, 2]
        von_mises_stress[elasticity_mask, t] = np.sqrt(
            3 * equivalent_stress[elasticity_mask]
        )

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

        i = 0
        while np.any(error > error_tolerance) and (i < iteration_limit):
            delta_ksi_delta_lambda_all = (
                -youngs_modulus / (1 - poissons_ratio) * stress_sum
                / (
                    9 * (1 + youngs_modulus * delta_lambda
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

            print(2 * np.sqrt(ksi))
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
                    1 + youngs_modulus * delta_lambda
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

        a11_star = (
            3 * (1 - poissons_ratio)
            / (3 * (1 - poissons_ratio) + youngs_modulus * delta_lambda)
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

        sigma_xx[plasticity_mask, t] = stress[t, plasticity_mask, 0]
        sigma_yy[plasticity_mask, t] = stress[t, plasticity_mask, 1]
        sigma_xy[plasticity_mask, t] = stress[t, plasticity_mask, 2]
        von_mises_stress[plasticity_mask, t] = yield_stress[plasticity_mask]

        if t > 0:
            mask = prev_plasticity_mask & (~plasticity_mask)
            sigma_xx[mask, t] = sigma_xx[mask, t - 1]
            sigma_yy[mask, t] = sigma_yy[mask, t - 1]
            sigma_xy[mask, t] = sigma_xy[mask, t - 1]

        delta_lambda.fill(0)
        delta_ksi_delta_lambda.fill(0)
        ksi.fill(0)
        plastic_criterion.fill(0)
        plastic_criterion_prime.fill(0)
        equivalent_stress.fill(0)

        prev_plasticity_mask = plasticity_mask

    return (sigma_xx, sigma_xy, sigma_yy, von_mises_stress)

