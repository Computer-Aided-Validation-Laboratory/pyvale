import numpy as np
from PIL import Image
from .render import *
import time

point3 = vec3
color = vec3

class Scene():
    def __init__(self,ambient_color = color(0.0, 0.0, 0.0), n = vec3(1.0,1.0,1.0)) :
        # n = index of refraction (by default index of refraction of air n = 1.)
        self.ambient_color = ambient_color
        self.n = n
        self.collider_list = Hittable_list()
        self.importance_sampled_list = Hittable_list()

    def add_Camera(self, lookfrom, lookat, screen_width, screen_height, field_of_view, focus_distance, aperture, vup= vec3(0.,1.,0.) ):
        self.screen_width = screen_width
        self.screen_height = screen_height

        self.camera = Camera(lookfrom, lookat,field_of_view, screen_width/screen_height, aperture, focus_distance, vup)


    def add(self,hittable, importance_sampled = False):
        self.collider_list.add(hittable)
        if (importance_sampled):
            self.importance_sampled_list.add(hittable)

                   
    def render(self,samples_per_pixel, max_depth):

        print ("Rendering...")

        sc_info = Scene_info(self.ambient_color, self.collider_list, self.camera, self.importance_sampled_list)


        r = np.empty([self.screen_height,self.screen_width], dtype=np.double)
        g = np.empty([self.screen_height,self.screen_width], dtype=np.double)
        b = np.empty([self.screen_height,self.screen_width], dtype=np.double)


        t0 = time.time()

        render_image(r, g, b, samples_per_pixel, max_depth, sc_info)

        print ("Took", time.time() - t0)

        img_RGB = []
        for c in [r,g,b]:
            # average ray colors that fall in the same pixel. (antialiasing) 
            img_RGB += [Image.fromarray((255 * np.clip(c, 0, 1)).astype(np.uint8), "L") ]

        return Image.merge("RGB", img_RGB)

