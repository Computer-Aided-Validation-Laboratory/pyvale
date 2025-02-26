from dataclasses import dataclass, field
import numpy as np
from interval import Interval
from ray import Ray
from material import Material
from aabb import AABB

class HitRecord:
    def __init__(self, p: np.ndarray, t: float, mat: Material, r: Ray, outward_normal: np.ndarray, u: float = None, v: float = None) -> None:
        self.p = p
        self.t = t
        self.mat = mat
        self.set_face_normal(r, outward_normal)

        # texture u v coords
        self.u: float
        self.v: float

    def set_face_normal(self, r: Ray, outward_normal):        
        #  Sets the hit record normal vector.
        #  NOTE: the parameter `outward_normal` is assumed to have unit length.

        self.front_face = np.dot(r.direction, outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal

@dataclass
class Hittable:
    bbox: AABB = field(default_factory=AABB)
    
    def hit(self, r: Ray, ray_t: Interval) -> HitRecord:
        pass

    