import numpy as np
from scipy.io import loadmat

from pyvale.vfm.dic_config import DICConfig
from pyvale.vfm.material_properties import MaterialProperties
from pyvale.vfm.stress_sensitivity import calculate_stress_sensitivity

input = loadmat("/Users/chris/work/vfmap-numerical-paper/test_data/compute_stress_sensetivity_input.mat")

parameter_map = input["spatialParamData"][0][0]
stress = input["stressRef"][0][0]
test_data = input["testData"][0][0]
dic_config = DICConfig(316, 116, test_data["time"]["time"][0][0])
material_properties = MaterialProperties(190000, 0.28, np.full((116, 316), 300), np.full((116, 316), 3000))
strain = test_data["strain"][0][0][0]
print(strain)

# strain: npt.NDArray[np.float64],
# material_properties: MaterialProperties,

# test_data = input["testData"][0][0]
# options = input["options"][0][0]
# print(spatial_param_data["parDof"].shape)

# calculate_stress_sensitivities(stress, parameter_map)
test = calculate_stress_sensitivity(stress, strain, material_properties, dic_config)
print("break")
