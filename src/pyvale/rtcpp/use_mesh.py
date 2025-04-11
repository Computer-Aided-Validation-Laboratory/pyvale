from rt import *
import pyvale
import mooseherder as mh


data_path = pyvale.DataSet.thermal_2d_output_path()
sim_data = mh.ExodusReader(data_path).read_all_sim_data()
field_key = "temperature"
# Scale to mm to make 3D visualisation scaling easier
sim_data.coords = sim_data.coords*1000.0 # type: ignore

n_sens = (3,2,1)
x_lims = (0.0,100.0)
y_lims = (0.0,50.0)
z_lims = (0.0,0.0)
sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)
sens_data = pyvale.SensorData(positions=sens_pos)

# loop through 
for i, _ in enumerate(sim_data.connect[0]):
    # get the ids
    i0 = sim_data.connect[0][i]
    i1 = sim_data.connect[1][i]
    i2 = sim_data.connect[2][i]
    i3 = sim_data.connect[3][i]

    # get the locs
    p0 = sim_data.coords[i0]
    p1 = sim_data.coords[i1]
    p2 = sim_data.coords[i2]
    p3 = sim_data.coords[i3]

    quad = 