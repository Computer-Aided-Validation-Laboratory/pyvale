import numpy as np
from interval import Interval
from ray import Ray
from material import Material
from aabb import AABB

class HitRecord:
    p: np.ndarray # location in 3D space
    normal: np.ndarray
    t: float
    front_face: bool
    mat: Material

    def set_face_normal(self, r: Ray, outward_normal):        
        #  Sets the hit record normal vector.
        #  NOTE: the parameter `outward_normal` is assumed to have unit length.

        self.front_face = np.dot(r.direction, outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal

class Hittable:
    bbox: AABB = AABB()

    def __init__(self):
        pass
    
    def hit(self, r: Ray, ray_t: Interval) -> HitRecord:
        pass

    