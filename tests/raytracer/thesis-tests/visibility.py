# Resolution 1024 x 1024
#Unlit
# Rabbits in pairs TRI3-TRI6, QUAD4-QUAD8, QUAD4-QUAD9, QUAD8-QUAD9
# Front rabbit is 1.0, behind it is 0.5
# Slight overlap
# Images
# Both rabbits
# Front rabbit only
# Difference image

import numpy as np

from global_utils import *
from pyvale.raytracer.rtmesh import *
from pyvale.raytracer.rtmeshvisuals import *
from pyvale.raytracer.rtcamera import *
from pyvale.raytracer.rtscene import *
from pyvale.raytracer.rtmain import *
from pyvale.raytracer.rtoutputformat import *

# thesis-data -> visibility -> rabbit_elem

def get_rabbit_path(rabbit_access: str, element: Element):
    return full_path(rabbit_access + element.label.lower())

class ElementPairs:
    TRIS = (Elements.TRI3, Elements.TRI6)
    QUADS1 = (Elements.QUAD4, Elements.QUAD8)
    QUADS2 = (Elements.QUAD4, Elements.QUAD9),
    QUADS3 = (Elements.QUAD8, Elements.QUAD9)

def visibility_test():
    # Data is stored in thesis-data/visibility/rabbit_ELEMENT, so build the base
    rabbit_access = "thesis-data/visibility/rabbit_"
    # Get rabbits
    rabbit_front_path = get_rabbit_path(rabbit_access, Elements.QUAD4)
    rabbit_back_path = get_rabbit_path(rabbit_access, Elements.QUAD8)
    import os
    print(os.path.exists(rabbit_front_path))
    print(rabbit_front_path)
    # Use SimData openers for connectivity, coords, and uvs
    bunny_front = simdata_csv_to_rtmesh(rabbit_front_path, world_position = np.array([-2.5, -0.5, -10]),target_size=110, size_axis = Axis.X)
    bunny_back = simdata_csv_to_rtmesh(rabbit_back_path, world_position = np.array([2.5, 1.0, -10.5]), target_size=110, size_axis=Axis.X)

    # Camera
    image_width = 400
    image_height = 400
    output_format = output_format_phs6
    camera_center = np.array([0.0, 0.0, 50])
    camera_target = np.array([0.0, 0.0, 0.0])
    angle_vfov = 20
    cam = Camera(image_width, image_height, camera_center, camera_target, angle_vfov)

    #SceneVisualiser([bunny_front, bunny_back], cam)

visibility_test()