# import numpy as np
# import numpy.testing as np_test
# from scipy.io import loadmat

# from pyvale.vfm import MaterialProperties, radial_return


# def test_radial_return():
#     # R- I wonder if we can sort this so we share a very lightweight python datafile in git (after initial translation from matlab) so we can both use the same input data and not require maintaining data on both individual PCs.
#     # e.g. for this function I don't image just the raw strain data is very large? But would need to look into how best to manage this.
#     strain_data = loadmat("/Users/chris/work/vfmap-numerical-paper/scripts/strain.mat")
#     spatial_param_data = loadmat(
#         "/Users/chris/work/vfmap-numerical-paper/scripts/spatialParamData.mat"
#     )

#     # R - currently merged hardening function with main loop. Once happy, be sure to disentangle again
#     yield_strength = spatial_param_data["spatialParamData"]["param3"][0][0]["parameterMap"][
#         0
#     ][0]
#     hardening_modulus = spatial_param_data["spatialParamData"]["param4"][0][0][
#         "parameterMap"
#     ][0][0]

#     # Contains componenets c11, c12, c22 arrays with 35708 values per column and 23 columns (timesteps)
#     strain = strain_data["strain"]
#     # 23 timesteps x 35708 values
#     c11 = strain["c11"][0][0]
#     c12 = strain["c12"][0][0]
#     c22 = strain["c22"][0][0]

#     # 23 timesteps x 35708 values x 3 components          # R - need to decide on convention and ensure consistent (it may be currently, unsure.). Prob best to go npts x nsteps x ncomp for 3d, where npts is consistently wrapped / unwrapped to and from x by y grid
#     strain = np.stack((c11, c22, c12), axis=2).transpose((1, 0, 2))

#     matprops = MaterialProperties(youngs_modulus=190000, poissons_ratio=0.28, yield_strength=yield_strength, hardening_modulus=hardening_modulus)

#     (sigma_xx, sigma_xy, sigma_yy, von_mises_stress) = radial_return(strain, matprops)

#     output_data = loadmat(
#         "/Users/chris/work/vfmap-numerical-paper/scripts/vmplasticity_output.mat"
#     )
#     c11 = output_data["stressOutput"][0][0]["c11"]
#     c22 = output_data["stressOutput"][0][0]["c22"]
#     c12 = output_data["stressOutput"][0][0]["c12"]
#     vm = output_data["stressOutput"][0][0]["vm"]

#     np_test.assert_allclose(c11, sigma_xx, rtol=1e-12, atol=1e-12)
#     np_test.assert_allclose(c22, sigma_yy, rtol=1e-12, atol=1e-12)
#     np_test.assert_allclose(c12, sigma_xy, rtol=1e-12, atol=1e-12)
#     np_test.assert_allclose(vm, von_mises_stress, rtol=1e-12, atol=1e-12)
