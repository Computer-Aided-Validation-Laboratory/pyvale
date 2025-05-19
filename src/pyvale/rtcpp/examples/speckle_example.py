import numpy as np

# Add parent directory (where .so file is) to sys.path
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)
import pyray

from PIL import Image
import io

# Check module imports okay
print(pyray)


scene = pyray.Hittable_list()
lights = pyray.Hittable_list()


point3 = pyray.vec3
color = pyray.vec3


red   = pyray.Lambertian(color(.65, .05, .05))
white = pyray.Lambertian(color(.73, .73, .73))
green = pyray.Lambertian(color(.12, .45, .15))
light = pyray.Diffuse_light(color(13, 13, 13))

camera = pyray.Camera(pyray.vec3(0, 0, 5),
                    pyray.vec3(0, 0, 0),
                    40.0,
                    float(200.0 / 200.0),
                    0.01,
                    10.0,
                    pyray.vec3(0.,1.,0.),
                    200,
                    200,
                    color(0.6, 0.6,0.6))

sphere = pyray.Sphere(point3(0, -152, 0), 150, white)
scene.add(sphere)

val = (0.016362695380085047, 0.016362695380085047)
speck = pyray.Dot_texture(30,30, val[0]/2., val[1]/2.)
nmat = pyray.Lambertian(speck)
scene.add(pyray.Quad(point3(-2, -2, 0), pyray.vec3(4,0,0), pyray.vec3(0,4,0), nmat))


l = pyray.Quad(point3(0, 5, 0), pyray.vec3(1,0,0), pyray.vec3(0,0,1), red)
lights.add(l)

ppm_data = camera.render(scene, lights)
image = Image.open(io.BytesIO(ppm_data.encode('utf-8')))
image.show()


