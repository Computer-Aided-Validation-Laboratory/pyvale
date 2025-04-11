from rt import *
import pyvale
import mooseherder as mh
import numpy as np

point3 = vec3

# tests if angle abc is a right angle
def IsOrthogonal(a, b, c) -> bool:
    return (b[0] - a[0]) * (b[0] - c[0]) + (b[1] - a[1]) * (b[1] - c[1]) + (b[2] - a[2]) * (b[2] - c[2]) == 0

def IsRectangle(a, b, c, d) -> bool:
    return IsOrthogonal(a, b, c) and IsOrthogonal(b, c, d) and IsOrthogonal(c, d, a)

def IsRectangleAnyOrder(a, b, c, d):
    if IsRectangle(a, b, c, d):
        return (a, b, c, d)
    if IsRectangle(b, c, a, d):
        return (b, c, a, d)
    if IsRectangle(c, a, b, d):
        return (c, a, b, d)
    else:
        return None
    # return IsRectangle(a, b, c, d) or IsRectangle(b, c, a, d) or IsRectangle(c, a, b, d)


data_path = pyvale.DataSet.thermal_2d_output_path()
sim_data = mh.ExodusReader(data_path).read_all_sim_data()
field_key = "temperature"
# Scale and move to make 3D visualisation scaling easier
sim_data.coords = sim_data.coords*6000.0 - np.array([-300, -200, -200])

n_sens = (3,2,1)
x_lims = (0.0,100.0)
y_lims = (0.0,50.0)
z_lims = (0.0,0.0)
sens_pos = pyvale.create_sensor_pos_array(n_sens,x_lims,y_lims,z_lims)
sens_data = pyvale.SensorData(positions=sens_pos)

conne = sim_data.connect["connect1"]
quad_dicts = []
missed = 0
count = 0
# loop through 
for i, _ in enumerate(conne[0]):
    # get the ids
    i0 = conne[0][i] - 1
    i1 = conne[1][i] - 1
    i2 = conne[2][i] - 1
    i3 = conne[3][i] - 1

    # get the locs
    p0 = sim_data.coords[i0]
    p1 = sim_data.coords[i1]
    p2 = sim_data.coords[i2]
    p3 = sim_data.coords[i3]

    # Are they rectangles?
    if IsRectangleAnyOrder(p0, p1, p2, p3) is not None:
        count += 1
        a, b, c, d = IsRectangleAnyOrder(p0, p1, p2, p3)
    else:
        missed += 1
        continue
    
    quad_dict = {
        "Q": point3(a[0], a[1], a[2]),
        "u": vec3(b[0]-a[0], b[1]-a[1], b[2]-a[2]),
        "v": vec3(c[0]-a[0], c[1]-a[1], c[2]-a[2]),
    }
    quad_dicts.append(quad_dict)
print("missed: ", missed)
print("count: ", count)


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
scene.add(box1)
scene.add(Sphere(vec3(160, 165/2, +65+185/2), 165/2, Refractive(1.5)))

for t in quad_dicts:
    scene.add(Quad(t["Q"], t["u"], t["v"], green))



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

