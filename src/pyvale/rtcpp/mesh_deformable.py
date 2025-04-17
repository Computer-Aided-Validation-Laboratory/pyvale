from rt import *
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

def get_simulation_components(sim_data: mh.SimData) -> tuple | None:
    node_vars = sim_data.node_vars
    node_vars_names = list(node_vars.keys())
    components = []
    if 'disp_x' in node_vars_names:
        components.append('disp_x')
    if 'disp_y' in node_vars_names:
        components.append('disp_y')
    if 'disp_z' in node_vars_names:
        components.append('disp_z')
    components = tuple(components)
    if len(components) == 0:
        components = None
    return components

data_path = pyvale.DataSet.thermomechanical_2d_path()
sim_data = mh.ExodusReader(data_path).read_all_sim_data()
field_key = get_simulation_components(sim_data)
spat_dim = get_mesh_spat_dim(sim_data)
# Scale and move to make 3D visualisation scaling easier
sim_data.coords = sim_data.coords*4000.0 - np.array([-200, -100, -200])
times = sim_data.time

n_sens = (3,2,1)
x_lims = (0.0,100.0)
y_lims = (0.0,50.0)
z_lims = (0.0,0.0)
sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)
sens_data = pyvale.SensorData(positions=sens_pos)

(pv_grid, _) = pyvale.simdata_to_pyvista(sim_data,
                                                field_key,
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

x_disps = sim_data.node_vars["disp_x"]
y_disps = sim_data.node_vars["disp_y"]

def generate_quad_primatives_info(timestep_idx):
    quad_dicts = []

    # perturb the displacements for this timestep
    disp_verts = vertices.copy()
    disp_verts[:, 0] += x_disps[:, timestep_idx]
    disp_verts[:, 1] += y_disps[:, timestep_idx]
    
    for f in faces:
        indices = f
        locs = disp_verts[f]

        # Are they rectangles? - they are, and are always winding, never crossing - ASSUMPTION!
        a = locs[0]
        b = locs[1]
        c = locs[2]
        d = locs[3]

        quad_dict = {
            "Q": point3(a[0], a[1], a[2]),
            "u": vec3(b[0]-a[0], b[1]-a[1], b[2]-a[2]),
            "v": vec3(d[0]-a[0], d[1]-a[1], d[2]-a[2]),
        }
        quad_dicts.append(quad_dict)
    return quad_dicts

def cornel_scene():
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


    scene.add_Camera(lookfrom = point3(278, 278, -800),
                    lookat = point3(278,278,0),
                    screen_width = 200, 
                    screen_height = 200,
                    field_of_view = 40,
                    focus_distance  = 10.0,
                    aperture  = 0.01)
    return scene

for i, _ in enumerate(times):
    green = Diffuse(solid_color(.12, .45, .15))

    if i == 0 or i == 30 or i == 60:
        scene = cornel_scene()

        quad_dicts = generate_quad_primatives_info(i)
        for t in quad_dicts:
            t = Quad(t["Q"], t["u"], t["v"], green)
            # t = Rotate_y(t, 45)
            scene.add(t)


        img = scene.render(samples_per_pixel = 100, max_depth = 5)

        img.save("box.png")

        img.show()

