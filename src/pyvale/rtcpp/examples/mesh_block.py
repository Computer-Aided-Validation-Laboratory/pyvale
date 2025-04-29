from pyvale.rtcpp.rt import *
import pyvale
import mooseherder as mh
import numpy as np

point3 = vec3

def get_mesh_spat_dim(sim_data: mh.SimData) -> int:
    spat_dim = 2 # all values in z are 0
    nodes = sim_data.coords
    check_if_2d = np.count_nonzero(nodes, axis=0)
    if check_if_2d[2] == 0:
        spat_dim = 2
    else:
        spat_dim = 3
    return spat_dim

data_path = pyvale.DataSet.render_mechanical_3d_path()
sim_data = mh.ExodusReader(data_path).read_all_sim_data()
field_key = "disp_x"
spat_dim = get_mesh_spat_dim(sim_data)
# Scale and move to make 3D visualisation scaling easier
sim_data.coords = sim_data.coords*1000.0 - np.array([-200, -100, -200])

n_sens = (3,2,1)
x_lims = (0.0,100.0)
y_lims = (0.0,50.0)
z_lims = (0.0,0.0)
sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)
sens_data = pyvale.SensorData(positions=sens_pos)

(pv_grid, _) = pyvale.simdata_to_pyvista(sim_data,
                                                [field_key],
                                                spat_dim)
pv_surf =  pv_grid.extract_surface()


vertices = pv_surf.points

def get_faces(pv_surf):
    # read in first number of faces
    faces = []
    index = 0
    while index < len(pv_surf.faces):
        num_faces = pv_surf.faces[index]
        faces.append(pv_surf.faces[index+1:index+num_faces+1])
        index += num_faces+1
    return faces

faces = get_faces(pv_surf)

tri_dicts = []
for f in faces:
    indices = f
    locs = vertices[f]

    a = locs[0]
    b = locs[1]
    c = locs[2]
    
    tri_dict = {
        "Q": point3(a[0], a[1], a[2]),
        "u": vec3(b[0]-a[0], b[1]-a[1], b[2]-a[2]),
        "v": vec3(c[0]-a[0], c[1]-a[1], c[2]-a[2]),
    }
    tri_dicts.append(tri_dict)



scene = Scene()


red   = Diffuse(solid_color(.65, .05, .05))
white = Diffuse(solid_color(.73, .73, .73))
green = Diffuse(solid_color(.12, .45, .15))
light = Diffuse_light(solid_color(13, 13, 13))
glass = Refractive(1.5)

scene.add(Plane_yz(0, 555, 0, 555, 555, green))
scene.add(Plane_yz(0, 555, 0, 555, 0, red))
scene.add(Plane_xz(213, 343, 227, 332, 554, light) , importance_sampled = True)
scene.add(Plane_xz(0, 555, 0, 555, 0, white))
scene.add(Plane_xz(0, 555, 0, 555, 555, white))
scene.add(Plane_xy(0, 555, 0, 555, 555, white))



box1 = Box(point3(265, 0, 295), point3(430, 330, 460), white)
box1 = Rotate_y(box1, -15)
scene.add(box1)

s = Sphere(vec3(160, 165/2, +65+185/2), 165/2, Refractive(1.5))
s2 = Translate(s, vec3(130, 0, 65))
scene.add(s2)

for t in tri_dicts:
    t = Tri(t["Q"], t["u"], t["v"], green)
    t = Rotate_y(t, 45)
    scene.add(t)



scene.add_Camera(lookfrom = point3(278, 278, -800),
				  lookat = point3(278,278,0),
				  screen_width = 200, 
				  screen_height = 200,
				  field_of_view = 40,
				  focus_distance  = 10.0,
				  aperture  = 0.01)


img = scene.render(samples_per_pixel = 100, max_depth = 5)

img.save("box.png")

img.show()

