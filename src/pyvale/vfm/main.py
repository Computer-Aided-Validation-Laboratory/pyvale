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

strain_data = loadmat("/Users/chris/work/vfmap-numerical-paper/scripts/strain.mat")
spatial_param_data = loadmat("/Users/chris/work/vfmap-numerical-paper/scripts/spatialParamData.mat")

# Contains componenets c11, c12, c22 arrays with 35708 values per column and 23 columns (timesteps)
strain = strain_data["strain"]
# 23 timesteps x 35708 values
c11 = strain["c11"][0][0]
c12 = strain["c12"][0][0]
c22 = strain["c22"][0][0]

# 23 timesteps x 3 components x 35708 values
strain = np.stack((c11, c22, c12), axis=2).transpose((1, 2, 0))

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
yield_strength = spatial_param_data["spatialParamData"]["param3"][0][0]["parameterMap"][0][0]
hardening_modulus = spatial_param_data["spatialParamData"]["param4"][0][0]["parameterMap"][0][0]

num_timesteps = 23
for i in range(num_timesteps):
    # Convert shear strain component from tensorial shear strain to engineering shear strain
    # Since by convention tensorial shear strain is half of engineering shear strain
    incremental_strain[i, 2, :] *= 2

    if i == 0:
        stress[0, :, :] = incremental_strain[0, :, :].T.dot(elasticity_matrix).T
        # Flatten with order F to use column major ordering same as matlab
        yield_stress_vec = yield_strength.flatten(order="F") + (hardening_modulus.flatten(order="F") * equivalent_plastic_strain[:, 0])
    else:
        prev_stress = stress[i-1, :, :]
        stress[i, :, :] = prev_stress + (incremental_strain[i, :, :].T.dot(elasticity_matrix).T)
        yield_stress_vec = yield_strength.flatten(order="F") + (hardening_modulus.flatten(order="F") * equivalent_plastic_strain[:, i-1])
        equivalent_plastic_strain[:, i] = equivalent_plastic_strain[:, i-1]

