from ..utils.constants import *
from ..utils.vector3 import vec3, rgb, extract
from functools import reduce as reduce 
from ..ray import Ray, get_raycolor
from .. import lights
from ..utils.image_functions import load_image, load_image_as_linear_sRGB
import numpy as np
from abc import abstractmethod 

class Material():

    def get_Normal(self, hit):
        N_coll = hit.collider.get_Normal(hit)
        return N_coll*hit.orientation

    @abstractmethod   
    def get_color(self, scene, ray, hit):
        pass
