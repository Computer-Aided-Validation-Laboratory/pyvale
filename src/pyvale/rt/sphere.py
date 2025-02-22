import math
import numpy as np
from ray import Ray
from interval import Interval
from hittable import HitRecord#, Hittable
from material import Material

class Sphere:
    centre: np.ndarray
    radius: float
    mat: Material
    
    def __init__(self, centre, radius: float, material: Material) -> None:
        self.centre = centre
        self.radius = max(radius,0)
        self.mat = material
    
    def hit(self, r: Ray, ray_t: Interval) -> HitRecord:
        oc = self.centre - r.origin
        a = np.sum(r.direction**2)
        h = np.dot(r.direction, oc)
        c = np.sum(oc**2) - self.radius**2

        discriminant = h*h - a*c
        if discriminant < 0:
            return None
        
        sqrtd = math.sqrt(discriminant)

        # Find the nearest root that lies in the acceptable range
        root = (h-sqrtd) / a
        if not ray_t.surrounds(root):
            root = (h + sqrtd) / a
            if not ray_t.surrounds(root):
                return None
        
        rec: HitRecord = HitRecord()   
        rec.t = root
        rec.p = r.at(rec.t)
        outward_normal = (rec.p - self.centre) / self.radius
        rec.set_face_normal(r, outward_normal)
        rec.mat = self.mat

        return rec