import numpy as np
from rt import *

point3 = vec3
color = vec3


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

