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

# 23 timesteps x 35708 values x 3 components
strain = np.stack((c11, c22, c12), axis=2).transpose((1, 0, 2))

# Output stresses
sigma_xx = np.zeros((35708, 23))
sigma_xy = np.zeros((35708, 23))
sigma_yy = np.zeros((35708, 23))
von_mises_stress = np.zeros((35708, 23))

young_modulus = 190000.0
# Poisson's ratio
nu = 0.28
# Valid for Engineering Shear Strain only!
elasticity_matrix = (young_modulus / (1 - nu**2)) * np.array(
    [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (0.5 * (1.0 - nu))]]
)

incremental_strain = np.empty_like(strain)

incremental_strain[0, :, :] = strain[0, :, :]
incremental_strain[1:, :, :] = np.diff(strain, axis=0)

stress = np.empty_like(strain)
# PEEQ
equivalent_plastic_strain = np.empty((35708, 23))
yield_strength = spatial_param_data["spatialParamData"]["param3"][0][0]["parameterMap"][
    0
][0]
hardening_modulus = spatial_param_data["spatialParamData"]["param4"][0][0][
    "parameterMap"
][0][0]

num_timesteps = 23
for t in range(num_timesteps):
    # Convert shear strain component from tensorial shear strain to engineering shear strain
    # Since by convention tensorial shear strain is half of engineering shear strain
    incremental_strain[t, :, 2] *= 2

    if t == 0:
        stress[0, :, :] = incremental_strain[0, :, :].dot(elasticity_matrix)
        # Flatten with order F to use column major ordering same as matlab
        yield_stress = yield_strength.flatten(order="F") + (
            hardening_modulus.flatten(order="F") * equivalent_plastic_strain[:, 0]
        )
    else:
        prev_stress = stress[t - 1, :, :]
        stress[t, :, :] = prev_stress + (
            incremental_strain[t, :, :].dot(elasticity_matrix)
        )
        yield_stress = yield_strength.flatten(order="F") + (
            hardening_modulus.flatten(order="F") * equivalent_plastic_strain[:, t - 1]
        )
        equivalent_plastic_strain[:, t] = equivalent_plastic_strain[:, t - 1]

    # Check yield criterion
    equivalent_stress = np.zeros(35708)
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

    plastic_multiplier = np.zeros(35708)
    plastic_multiplier[plasticity_mask] = (
        (1 / 6) * np.sum(stress[t, plasticity_mask, 0:2], axis=1) ** 2
        + 0.5 * (stress[t, plasticity_mask, 1] - stress[t, plasticity_mask, 0]) ** 2
        + 2 * (stress[t, plasticity_mask, 2]) ** 2
    )

    plastic_criterion = np.zeros(35708)
    plastic_criterion[plasticity_mask] = (1 / 2) * plastic_multiplier[
        plasticity_mask
    ] - (1 / 3) * (yield_stress[plasticity_mask] ** 2)
