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
# Contains c11, c12, c22 arrays with 35708 values per column and 23 columns (timesteps)
# c stands for component
strain = strain_data["strain"]
# 23 timesteps x 35708 values
c11 = strain["c11"][0][0]
c12 = strain["c12"][0][0]
c22 = strain["c22"][0][0]
# print(c12)

# 23 timesteps x 3 components x 35708 values
strain = np.stack((c11, c22, c12), axis=2).transpose((1, 2, 0))
# print(strain)

young_modulus = 190000.0
# Poisson's ratio
nu = 0.28
# Valid for Engineering Shear Strain only!
elasticity_matrix = (young_modulus / (1 - nu**2)) * np.array(
    [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (0.5 * (1.0 - nu))]]
)
# print(strain.shape)

incremental_strain = np.empty_like(strain)

incremental_strain[0, :, :] = strain[0, :, :]
incremental_strain[1:, :, :] = np.diff(strain, axis=0)

# Convert shear strain component from tensorial shear strain to engineering shear strain
# Since by convention tensorial shear strain is half of engineering shear strain
incremental_strain[:, 2, :] *= 2

# print(incremental_strain)

# Calculate elastic predictor
# no idea if this is the right name for this
incremental_elasticity = incremental_strain.transpose(0, 2, 1).dot(elasticity_matrix).transpose(0, 2, 1)
# print(incremental_elasticity)

stress = np.empty_like(strain)
stress[0, :, :] = incremental_elasticity[0, :, :]


# print(stress)

# The Y from hardeningfun
yield_stress = 0

yield_strength = spatial_param_data["spatialParamData"]["param3"][0][0]["parameterMap"][0][0]
hardening_modulus = spatial_param_data["spatialParamData"]["param4"][0][0]["parameterMap"][0][0]

equivalent_plastic_strain = np.empty((23, 35708))

for i in range(1, 23):
    stress[i, :, :] = stress[i-1, :, :] + incremental_elasticity[i, :, :]
    yield_stress_vec = yield_strength + (hardening_modulus * equivalent_plastic_strain)
    print(yield_stress_vec)
# For each timestep
# - there are 35k c11 components
# - there are 35k c12 components
# - there are 35k c22 components
#
# delta_strain = 0

# print(np.zeros((23, 35708, 3)))

# eps = np.stack((c11, c12, c22), axis=2)
# print(eps.shape)

# dEps = np.empty_like(eps)

# # First increment
# dEps[:, 0, :] = eps[:, 0, :]

# # Remaining increments
# dEps[:, 1:, :] = eps[:, 1:, :] - eps[:, :-1, :]

# print(dEps)
