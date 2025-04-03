from ..utils.constants import *
from ..utils.vector3 import vec3, rgb
from ..ray import Ray, get_raycolor
from ..utils.image_functions import load_image, load_image_as_linear_sRGB
import numpy as np
from abc import abstractmethod 

class texture():

    @abstractmethod   
    def __init__(self):
    	pass

    @abstractmethod  
    def get_color(self, hit):
    	pass


class solid_color(texture):
	 
    def __init__(self,color):
    	self.color = color

    def get_color(self, hit):
    	return self.color
