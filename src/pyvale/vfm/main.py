import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from pyvale.vfm import mechanical_properties
from pyvale.vfm.mechanical_properties import *
from pyvale.vfm.radial_return import radial_return
from pyvale.vfm.virtual_fields_mesh import generate_virtual_fields_mesh
# from pyvale.vfm.stress_sensitivity import calculate_stress_sensitivity

# data = loadmat(
#     "/home/robh/1_Projects/vfmap-numerical-paper/data/notchedButtWeld_bilin_lin360420S_hom3700H_imDef_1.5/5-testData/testData.mat",
#     struct_as_record=False,
#     squeeze_me=True,
#     simplify_cells=True
# )
data = loadmat(
    "/Users/chris/work/vfmap-numerical-paper/data/5-testData/testData.mat",
    struct_as_record=False,
    squeeze_me=True,
    simplify_cells=True
)
test_data = data["testData"]

# Coordinate axis convention:
# - y coordinate increases downwards (as row number increases)
# - x increases to the right (as col num increases)

# Strain convention is (timestep, component, y, x)
# Component convention is 0 -> c11, 1 -> c22, 2 -> c12
strain = test_data["strain"]

num_cols = 316
num_rows = 113
num_timesteps = 23

# Strain comes in as flattened vector x timesteps
# Matlab uses column major ordering, so using order="F" unpack this into
# rows and cols representing y and x coords
strain_c11 = strain["c11"].reshape((num_rows, num_cols, num_timesteps), order="F")
strain_c22 = strain["c22"].reshape((num_rows, num_cols, num_timesteps), order="F")
strain_c12 = strain["c12"].reshape((num_rows, num_cols, num_timesteps), order="F")

# transpose and add a new axis to get this component of strain into the form we want
strain_c11 = np.transpose(strain_c11, (2, 0, 1))
# strain_c11 = strain_c11[:, np.newaxis, :, :] # (23, 1, 113, 316)
assert(strain_c11.shape == (23, 113, 316))

strain_c22 = np.transpose(strain_c22, (2, 0, 1))
# strain_c22 = strain_c22[:, np.newaxis, :, :]
assert(strain_c22.shape == (23, 113, 316))

strain_c12 = np.transpose(strain_c12, (2, 0, 1))
# strain_c12 = strain_c12[:, np.newaxis, :, :]
assert(strain_c12.shape == (23, 113, 316))

# Merging components
strain_4d = np.stack((strain_c11, strain_c22, strain_c12), axis=1)
# Flipping to conform to y increasing downwards convention
strain = np.flip(strain_4d, axis=2)

# STRAIN PLOT
# strain_slice_2d = strain[20, 1, :, :]
# plt.figure()
# plt.imshow(slice_slice_2d, aspect='auto')
# plt.colorbar()
# plt.xlabel('Index (316)')
# plt.ylabel('Index (113)')
# plt.title('Slice strain_slice_2d[20, 1, :, :]')
# plt.show()

x = test_data["X"]
x_vals = np.nanmean(x, axis=0, keepdims=True) # gives us a (1, 316)
x = np.tile(x_vals, (x.shape[0], 1)) # gives us a (113, 316)
assert(x.shape == (113, 316))

y = test_data["Y"]
y_vals = np.nanmean(y, axis=1, keepdims=True) # gives us a (113, 1)
y = np.tile(y_vals, (1, y.shape[1])) # gives us a (113, 316)
# Flipping to conform to y increasing downwards convention
y = np.flip(y, axis=0)
assert(y.shape == (113, 316))

# The specimen mask is true if the datapoint is a valid material datapoint
# of the specimen, and nan if it doesn't exist (e.g. a hole/notch).
# Ideally this would be defined from the DIC Region of Interest, but for now
# we're extracting the data from the x data as below
specimen_mask = np.zeros(x.shape, dtype=bool)
nan_mask = np.isnan(test_data["X"])
specimen_mask = ~nan_mask
# Flipping to conform to y increasing downwards convention
specimen_mask = np.flip(specimen_mask, axis=0)

area = test_data["area"].reshape((num_rows, num_cols), order="F")
area = np.flip(area, axis=0)
assert(area.shape == (113, 316))

# The x and y components of force at each timestep
# Shape is (timestep, component)
force = test_data["FGlob"] # (23, 2)

# The time at each time step
time = test_data["time"]["time"]

yield_strength = HomogeneousParameter(
    IdentificationType.Unknown,
    ParameterBounds(200, 800),
    ScalarValue(400)
)

hardening_modulus = HomogeneousParameter(
    IdentificationType.Unknown,
    ParameterBounds(1000, 10_000),
    ScalarValue(3000)
)

elastic_modulus = HomogeneousParameter(
    IdentificationType.Unknown,
    ParameterBounds(1000, 10_000),
    ScalarValue(3000)
)

poissons_ratio = HomogeneousParameter(
    IdentificationType.Unknown,
    ParameterBounds(1000, 10_000),
    ScalarValue(3000)
)

mechanical_properties = MechanicalProperties(
    ConstituitiveLaw.LinearHardening,
    {
        ParameterName.ElasticModulus: elastic_modulus,
        ParameterName.PoissonsRatio: poissons_ratio,
        ParameterName.YieldStrength: yield_strength,
        ParameterName.HardeningModulus: hardening_modulus,
    }
)

# TODO: need to figure out where this check should happen and
# deliver useful error messages
if not check_validity(mechanical_properties):
    print("mechanical properties invalid")

stress = radial_return(strain, mechanical_properties)
# virtual_fields_mesh = generate_virtual_fields_mesh()
print("break")
