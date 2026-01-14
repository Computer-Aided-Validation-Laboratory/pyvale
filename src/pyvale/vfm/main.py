from scipy.io import loadmat
import numpy as np
# Overall approach to computing the trail stress/elastic predictor:
# - Take strain components
# - Generate a new array based on the differences between each element at each timestep
# - Calculate elasticity matrix
# - Generate stress tensor

# 113y by 316x points on the surface
# in matlab 0,0 is bottom left of data structure
#
# Need to think about how these values are laid out in memory.
# At the moment matlab lays them out in a certain way when it converts these from
# some kind of 2d system into a flat 1d array
#
# possible array axes: c11, c12, c22, time
# or possibly: x, y, (c11, c12, c22), timestep <- but do we need x and y? Would be nice to have for
# intuition but would this reduce performance?
#
# Data should be 23 slices x 35708 rows x 3 cols
# Col index is last index, and row index is second to last

# R - Can pull this info from the size of strain data passed into function (e.g. NUM_TIMESTEPS=)
NUM_TIMESTEPS = 23
NUM_POINTS = 35708
NUM_COMPONENTS = 3

# R- this will eventually be an input (material properties)
YOUNGS_MODULUS = 190000.0
POISSONS_RATIO = 0.28
SHEAR_MODULUS = YOUNGS_MODULUS / (2 * (1 + POISSONS_RATIO))

# R- I wonder if we can sort this so we share a very lightweight python datafile in git (after initial translation from matlab) so we can both use the same input data and not require maintaining data on both individual PCs.
# e.g. for this function I don't image just the raw strain data is very large? But would need to look into how best to manage this.
strain_data = loadmat("/Users/chris/work/vfmap-numerical-paper/scripts/strain.mat")
spatial_param_data = loadmat(
    "/Users/chris/work/vfmap-numerical-paper/scripts/spatialParamData.mat"
)

# Contains componenets c11, c12, c22 arrays with 35708 values per column and 23 columns (timesteps)
strain = strain_data["strain"]
# 23 timesteps x 35708 values
c11 = strain["c11"][0][0]
c12 = strain["c12"][0][0]
c22 = strain["c22"][0][0]

# 23 timesteps x 35708 values x 3 components          # R - need to decide on convention and ensure consistent (it may be currently, unsure.). Prob best to go npts x nsteps x ncomp for 3d, where npts is consistently wrapped / unwrapped to and from x by y grid
strain = np.stack((c11, c22, c12), axis=2).transpose((1, 0, 2))

# R - unsure what convention we want to use for blockers / headers etc to help keep code readable. 
# 
# Perhaps none (just standard comment like below line):
# Initialisation
#
# Banner:
# ---------------
# Initialisation
# ---------------
#
#  === Initialisation ===  

# Output stresses
sigma_xx = np.zeros((NUM_POINTS, NUM_TIMESTEPS))
sigma_xy = np.zeros((NUM_POINTS, NUM_TIMESTEPS))
sigma_yy = np.zeros((NUM_POINTS, NUM_TIMESTEPS))
von_mises_stress = np.zeros((NUM_POINTS, NUM_TIMESTEPS))

ep_c11 = np.zeros((NUM_POINTS, NUM_TIMESTEPS))
ep_c22 = np.zeros((NUM_POINTS, NUM_TIMESTEPS))
ep_c12 = np.zeros((NUM_POINTS, NUM_TIMESTEPS))
ee_c11 = np.zeros((NUM_POINTS, NUM_TIMESTEPS))
ee_c22 = np.zeros((NUM_POINTS, NUM_TIMESTEPS))
ee_c12 = np.zeros((NUM_POINTS, NUM_TIMESTEPS))
eps_33 = np.zeros((NUM_POINTS, NUM_TIMESTEPS))


# von Mises effective stress matrix
p = np.array([[2 / 3, -1 / 3, 0], [-1 / 3, 2 / 3, 0], [0, 0, 2]])

delta_lambda = np.zeros(NUM_POINTS)
delta_ksi_delta_lambda = np.zeros(NUM_POINTS)
# equivalent plastic strain
peeq = np.empty((NUM_POINTS, NUM_TIMESTEPS))
hbar = np.zeros(NUM_POINTS)
ksi = np.zeros(NUM_POINTS)
# Plastic criterion
flyt = np.zeros(NUM_POINTS)
flyt_prime = np.zeros(NUM_POINTS)
prev_plasticity_mask = np.zeros(NUM_POINTS)

incremental_strain = np.empty_like(strain)

incremental_strain[0, :, :] = strain[0, :, :]
incremental_strain[1:, :, :] = np.diff(strain, axis=0)

stress = np.empty_like(strain)


# Valid for Engineering Shear Strain only!
elasticity_matrix = (YOUNGS_MODULUS / (1 - POISSONS_RATIO**2)) * np.array(
    [
        [1.0, POISSONS_RATIO, 0.0],
        [POISSONS_RATIO, 1.0, 0.0],
        [0.0, 0.0, (0.5 * (1.0 - POISSONS_RATIO))],
    ]
)

# R - currently merged hardening function with main loop. Once happy, be sure to disentangle again 
yield_strength = spatial_param_data["spatialParamData"]["param3"][0][0]["parameterMap"][
    0
][0]
hardening_modulus = spatial_param_data["spatialParamData"]["param4"][0][0][
    "parameterMap"
][0][0]

for t in range(NUM_TIMESTEPS):
    # Convert shear strain component from tensorial shear strain to engineering shear strain
    # Since by convention tensorial shear strain is half of engineering shear strain
    incremental_strain[t, :, 2] *= 2

    if t == 0:
        stress[0, :, :] = incremental_strain[0, :, :].dot(elasticity_matrix)
        # Flatten with order F to use column major ordering same as matlab
        yield_stress = yield_strength.flatten(order="F") + (
            hardening_modulus.flatten(order="F") * peeq[:, 0]
        )
    else:
        prev_stress = stress[t - 1, :, :]
        stress[t, :, :] = prev_stress + (
            incremental_strain[t, :, :].dot(elasticity_matrix)
        )
        yield_stress = yield_strength.flatten(order="F") + (
            hardening_modulus.flatten(order="F") * peeq[:, t - 1]
        )
        peeq[:, t] = peeq[:, t - 1]

    # Check yield criterion
    equivalent_stress = np.zeros(NUM_POINTS)
    equivalent_stress[:] = (1 / 3) * (
        stress[t, :, 0] ** 2
        + stress[t, :, 1] ** 2
        - stress[t, :, 0] * stress[t, :, 1]
        + (3 * stress[t, :, 2] ** 2)
    )

    yield_criterion_check = equivalent_stress > (1 / 3 * (yield_stress**2))

    plasticity_mask = yield_criterion_check
    elasticity_mask = np.logical_not(yield_criterion_check)

    # Update elastic stresses
    sigma_xx[plasticity_mask, t] = stress[t, plasticity_mask, 0]
    sigma_yy[plasticity_mask, t] = stress[t, plasticity_mask, 1]
    sigma_xy[plasticity_mask, t] = stress[t, plasticity_mask, 2]
    von_mises_stress[plasticity_mask, t] = np.sqrt(
        3 * equivalent_stress[plasticity_mask]
    )

    ksi[plasticity_mask] = (
        (1 / 6) * np.sum(stress[t, plasticity_mask, 0:2], axis=1) ** 2
        + 0.5 * (stress[t, plasticity_mask, 1] - stress[t, plasticity_mask, 0]) ** 2
        + 2 * (stress[t, plasticity_mask, 2]) ** 2
    )

    flyt[plasticity_mask] = (1 / 2) * ksi[plasticity_mask] - (1 / 3) * (
        yield_stress[plasticity_mask] ** 2
    )

    # Calculate error and normalise it by effective stress
    err = np.zeros(NUM_POINTS)
    err[plasticity_mask] = flyt[plasticity_mask]
    err[plasticity_mask] = err[plasticity_mask] / ksi[plasticity_mask]

    # Square of sum of normal components (for efficiency)
    stress_sum = np.sum(stress[t, :, 0:2], axis=1) ** 2
    # Square of difference of  normal components (for efficiency)
    stress_diff = (stress[t, :, 1] - stress[t, :, 0]) ** 2

    err_tolerance = 1e-8
    num_iter = 0
    iter_limit = 100

    while np.any(err > err_tolerance) and (num_iter < iter_limit):
        # Derivative of ksi wrt plastic multiplier
        delta_ksi_delta_lambda_all = -YOUNGS_MODULUS / (
            1 - POISSONS_RATIO
        ) * stress_sum / (
            9 * (1 + YOUNGS_MODULUS * delta_lambda / (3 * (1 - POISSONS_RATIO))) ** 3
        ) - 2 * SHEAR_MODULUS * (stress_diff + 4 * stress[t, :, 2] ** 2) / (
            (1 + 2 * SHEAR_MODULUS * delta_lambda) ** 3
        )

        delta_ksi_delta_lambda[plasticity_mask] = delta_ksi_delta_lambda_all[
            plasticity_mask
        ]

        delta_peeq = delta_lambda * np.sqrt((2 / 3) * ksi)
        if t == 0:
            peeq[plasticity_mask, t] = delta_peeq[plasticity_mask]
        else:
            peeq[plasticity_mask, t] = (
                peeq[plasticity_mask, t - 1] + delta_peeq[plasticity_mask]
            )

        yield_stress = yield_strength.flatten(order="F") + (
            hardening_modulus.flatten(order="F") * peeq[:, 0]
        )
        delta_yield_stress_delta_peeq = hardening_modulus.flatten(order="F")

        # TODO: this can get filled with nans, is that correct?
        hbar_all = 2 * (
            yield_stress
            * delta_yield_stress_delta_peeq
            * np.sqrt(2 / 3)
            * (
                np.sqrt(ksi)
                + delta_lambda * delta_ksi_delta_lambda / (2 * np.sqrt(ksi))
            )
        )

        hbar[plasticity_mask] = hbar_all[plasticity_mask]

        # Derivative of plastic criterion wrt plastic multiplier
        flyt_prime[plasticity_mask] = (
            1 / 2 * delta_ksi_delta_lambda[plasticity_mask]
        ) - (1 / 3 * hbar[plasticity_mask])

        # Update plastic multiplier using Newton-Raphson scheme
        delta_lambda[plasticity_mask] = delta_lambda[plasticity_mask] - (
            flyt[plasticity_mask] / flyt_prime[plasticity_mask]
        )

        # Update ksi
        ksi_all = stress_sum / (
            6 * (1 + YOUNGS_MODULUS * delta_lambda / (3 * (1 - POISSONS_RATIO))) ** 2
        ) + (0.5 * stress_diff + 2 * stress[t, :, 2] ** 2) / (
            (1 + 2 * SHEAR_MODULUS * delta_lambda) ** 2
        )
        # Eliminate any negatives
        ksi_all = np.maximum(ksi_all, 0)
        ksi[plasticity_mask] = ksi_all[plasticity_mask]

        delta_peeq = delta_lambda * np.sqrt((2 / 3) * ksi)
        if t == 0:
            peeq[plasticity_mask, t] = delta_peeq[plasticity_mask]
        else:
            peeq[plasticity_mask, t] = (
                peeq[plasticity_mask, t - 1] + delta_peeq[plasticity_mask]
            )

        yield_stress = yield_strength.flatten(order="F") + (
            hardening_modulus.flatten(order="F") * peeq[:, 0]
        )
        delta_yield_stress_delta_peeq = hardening_modulus.flatten(order="F")

        flyt[plasticity_mask] = 0.5 * ksi[plasticity_mask] - (
            1 / 3 * (yield_stress[plasticity_mask] ** 2)
        )

        err = np.zeros(NUM_POINTS)
        err[plasticity_mask] = np.abs(flyt[plasticity_mask])

        # Normalise by effective stress
        err[plasticity_mask] = err[plasticity_mask] / ksi[plasticity_mask]

        num_iter += 1

        if num_iter == (iter_limit - 1):
            print(
                f"The convergence has not been achieved within {iter_limit} iterations in step {t}"
            )

    # Calculate stresses with obtained p;astic multiplier
    a11_star = (
        3
        * (1 - POISSONS_RATIO)
        / (3 * (1 - POISSONS_RATIO) + YOUNGS_MODULUS * delta_lambda)
    )
    a22_star = 1 / (1 + 2 * SHEAR_MODULUS * delta_lambda)
    a_sum = 0.5 * (a11_star + a22_star)
    a_diff = 0.5 * (a11_star - a22_star)

    stress[t, :, :] = np.column_stack(
        (
            a_sum * stress[t, :, 0] + a_diff * stress[t, :, 1],
            a_diff * stress[t, :, 0] + a_sum * stress[t, :, 1],
            a22_star * stress[t, :, 2],
        )
    )

    # Update outputs with stresses obtained in plastic steps
    sigma_xx[plasticity_mask, t] = stress[t, plasticity_mask, 0]
    sigma_yy[plasticity_mask, t] = stress[t, plasticity_mask, 1]
    sigma_xy[plasticity_mask, t] = stress[t, plasticity_mask, 2]
    von_mises_stress[plasticity_mask, t] = yield_stress[plasticity_mask]

    # Unloading
    if t > 0:
        mask = prev_plasticity_mask & (~plasticity_mask)
        sigma_xx[mask, t] = sigma_xx[mask, t - 1]
        sigma_yy[mask, t] = sigma_yy[mask, t - 1]
        sigma_xy[mask, t] = sigma_xy[mask, t - 1]

    dep_c11 = delta_lambda * (
        p[0, 0] * stress[t, :, 0]
        + p[0, 1] * stress[t, :, 1]
        + p[0, 2] * stress[t, :, 2]
    )

    dep_c22 = delta_lambda * (
        p[1, 0] * stress[t, :, 0]
        + p[1, 1] * stress[t, :, 1]
        + p[1, 2] * stress[t, :, 2]
    )

    dep_c12 = delta_lambda * (
        p[2, 0] * stress[t, :, 0]
        + p[2, 1] * stress[t, :, 1]
        + p[2, 2] * stress[t, :, 2]
    )

    dee_c11 = incremental_strain[t, :, 0] - dep_c11
    dee_c22 = incremental_strain[t, :, 1] - dep_c22
    dee_c12 = incremental_strain[t, :, 2] - dep_c12

    deps_33 = (-POISSONS_RATIO / (1 - POISSONS_RATIO)) * (dee_c11 + dee_c22) - (
        dep_c11 + dep_c22
    )

    if t == 0:
        ep_c11[:, t] = dep_c11
        ep_c22[:, t] = dep_c22
        ep_c12[:, t] = dep_c12
        ee_c11[:, t] = dee_c11
        ee_c22[:, t] = dee_c22
        ee_c12[:, t] = dee_c12
        eps_33[:, t] = deps_33
    else:
        ep_c11[:, t] = ep_c11[:, t - 1] + dep_c11
        ep_c22[:, t] = ep_c22[:, t - 1] + dep_c22
        ep_c12[:, t] = ep_c12[:, t - 1] + dep_c12
        ee_c11[:, t] = ee_c11[:, t - 1] + dee_c11
        ee_c22[:, t] = ee_c22[:, t - 1] + dee_c22
        ee_c12[:, t] = ee_c12[:, t - 1] + dee_c12
        eps_33[:, t] = eps_33[:, t - 1] + deps_33

    prev_plasticity_mask = plasticity_mask

# Write outputs
print(sigma_xx)
print(sigma_yy)
print(sigma_xy)
print(von_mises_stress)
