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

data = loadmat("/Users/chris/work/vfmap-numerical-paper/scripts/strain.mat")
# Contains c11, c12, c22 arrays with 35708 values per column and 23 columns (timesteps)
# c stands for component
strain = data["strain"]
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
elasticity_matrix = (young_modulus / (1 - nu**2)) * np.array(
    [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (0.5 * (1.0 - nu))]]
)
# print(strain.shape)

delta_strain = np.empty_like(strain)

delta_strain[0, :, :] = strain[0, :, :]
delta_strain[1:, :, :] = strain[1:, :, :] - strain[:-1, :, :]
print(delta_strain)


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
