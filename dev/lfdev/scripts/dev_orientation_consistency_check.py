'''
================================================================================
example: displacement sensors on a 2d plate

pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
'''
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import pyvale.mooseherder as mh
import pyvale

def main() -> None:
    """Basics
    """
    data_path = Path('src/data/case17_out.e')
    sim_data = mh.ExodusReader(data_path).read_all_sim_data()
    # Scale m to mm to make 3D visualisation scaling correct for pyvista
    sim_data = pyv.scale_length_units(1000.0,sim_data)

    descriptor = pyvale.SensorDescriptorFactory.displacement_descriptor()

    spat_dims = 2
    field_key = 'disp'
    components = ('disp_x','disp_y')
    disp_field = pyvale.FieldVector(sim_data,field_key,components,spat_dims)

    n_sens = (2,2,1)
    x_lims = (0.0,100.0)
    y_lims = (0.0,150.0)
    z_lims = (0.0,0.0)
    sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)

    orientations = sens_pos.shape[0] * \
        (R.from_euler("zyx", [45, 0, 0], degrees=True),)

    disp_sens_array = pyvale.SensorArrayPoint(sens_pos,
                                              disp_field,
                                              None,
                                              descriptor,
                                              None,
                                              orientations)

    measurements = disp_sens_array.get_measurements()

    print(80*"=")
    print("Transformation consistency check: scipy.rotation to numpy.matmul")
    print()
    print("ROTATION= object rotates with coords fixed.")
    print("For Z rotation: sin negative in row 1.")
    print()
    print("TRANSFORMATION= coords rotate with object fixed")
    print("For Z transformation: sin negative in row 2, transpose scipy mat.")
    print()
    print(80*"-")
    print("Need to add a third dimension of zeros (Z component) to allow scipy")
    print("to do the rotation with R.apply().")
    print(f"{measurements.shape=}")
    meas_shape = measurements.shape
    measurements = np.concatenate(
        (measurements,np.zeros((meas_shape[0],1,meas_shape[2]))),
        axis=1)

    print("Goes to:")
    print(f"{measurements.shape=}")
    print()

    print(80*"-")
    print("Now we extract a sensor and all components for checking.")
    sens_num = -1
    sens = np.squeeze(measurements[sens_num,:,:])
    print(f"{sens.shape=}")
    print()

    print(80*"-")
    print("SCIPY.ROTATION")
    print("Rotating the sensor data with scipy:")
    rot = R.from_euler("zyx", [30, 0, 0], degrees=True)
    check_scipy = rot.apply(sens.T).T
    print(f"{check_scipy.shape=}")
    print()

    print(80*"-")
    print("NUMPY MATMUL")
    rmat = rot.as_matrix()
    print(f"{rmat=}")
    rmat_x_sens1 = np.matmul(rmat,sens)
    print(f"{rmat_x_sens1.shape=}")
    print()
    check_np_1 = np.allclose(rmat_x_sens1,check_scipy)
    print(f"{np.allclose(rmat_x_sens1,check_scipy)=}")
    print()

    print(80*"-")
    print("NUMPY ALONG AXIS MAT MUL")
    np_mul_along_axis = np.apply_along_axis(np.matmul,0,rmat.T,measurements)
    check_sens_along_axis = np_mul_along_axis[sens_num,:,:].T
    print(f"{np_mul_along_axis.shape=}")
    print(f"{check_sens_along_axis.shape=}")
    print()
    check_np_2 = np.allclose(check_sens_along_axis,check_scipy)
    print(f"{np.allclose(check_sens_along_axis,check_scipy)=}")
    print()
    print(80*"=")
    print(f"Check np to sp = {check_np_1}")
    print(f"Check np along ax to sp = {check_np_2}")
    print(80*"=")
    print()

    np_swap = np.swapaxes(np_mul_along_axis,1,2)
    print(f"{np_swap.shape=}")
    np_swap_sens = np_swap[sens_num,:,:]
    print(f"{np.allclose(np_swap_sens,check_scipy)=}")
    print()




if __name__ == "__main__":
    main()