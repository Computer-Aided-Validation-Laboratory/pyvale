from math import sqrt
from typing import Tuple
import numpy as np
from pyvale.rt.ray import Ray
from pyvale.rt.interval import Interval
from pyvale.rt.hittable import Hittable, HitRecord
from pyvale.rt.material import Material
from pyvale.rt.aabb import AABB

# A quad. Q is a corner of the quad. u is the vector to one touching corner. v is the vector to the other
class Quad(Hittable):
    # _q: np.ndarray
    # _u: np.ndarray
    # _v: np.ndarray
    # _w: np.ndarray
    # _mat: Material
    # # bbox: AABB
    # _normal: np.ndarray
    # _d: float

    def __init__(self, q: np.ndarray, u: np.ndarray, v: np.ndarray, mat: Material):
        # super.__init__()
        self._q = q
        self._u = u
        self._v = v
        self._mat = mat

        n = np.cross(self._u, self._v)
        self._normal = n / np.sqrt(np.sum(n**2))
        self._d = np.dot(self._normal, q)
        self._w = n / np.dot(n,n)

        self.set_bounding_box()

    def set_bounding_box(self):
        # Compute the bounding box of all four vertices
        bbox_diagonal1 = AABB.from_arrays(a=self._q, b=self._q+self._u + self._v)
        bbox_diagonal2 = AABB.from_arrays(a=self._q + self._u, b=self._q + self._v)
        self.bbox = AABB.from_bbox(box0=bbox_diagonal1, box1=bbox_diagonal2)
    
    def hit(self, r: Ray, ray_t: Interval) -> HitRecord:
        denom = np.dot(self._normal, r.direction)

        # No hit if the ray is parallel to the plane
        if np.isclose(denom, 0):
            return None
        
        # No hit if the hit point parameter t is outside the ray interval
        t = (self._d - np.dot(self._normal, r.origin)) / denom
        if not ray_t.contains(t):
            return None
        
        # Determine if the hit point lies within the planar shape using its plane coordinates
        intersection = r.at(t)
        planar_htpt_vector = intersection - self._q
        alpha = np.dot(self._w, np.cross(planar_htpt_vector, self._v))
        beta = np.dot(self._w, np.cross(self._u, planar_htpt_vector))

        retu = self.is_interior(alpha, beta)
        if retu is None:
            return None
        
        # Ray hits the 2D shape; set the rest of the hit record and return true.
        rec: HitRecord = HitRecord(p = intersection, t= t, mat = self._mat, r=r, outward_normal=self._normal, u = retu[0], v = retu[1])
        return rec

    def is_interior(self, a: float, b: float) -> Tuple[float, float]:
        unit_interval = Interval(0,1)
        # Given the hit point in plane coordinates, return false if it is outside the primitive, otherwise set the hit record UV coordinates.

        if (not unit_interval.contains(a)) or (not unit_interval.contains(b)):
            return None
        
        return (a, b)

# A triangle 
class Tri(Quad):
    def __init__(self, q: np.ndarray, u: np.ndarray, v: np.ndarray, mat: Material):
        super().__init__(q, u, v, mat)
    
    def is_interior(self, a: float, b: float) -> Tuple[float, float]:
        if a > 0 and b > 0 and a+b < 1:
            return (a,b)
        else:
            return None


# A flat disk with radius
class Disk(Quad):
    def __init__(self, q: np.ndarray, u: np.ndarray, v: np.ndarray, mat: Material, radius: float):
        super().__init__(q, u, v, mat)
        self.radius = radius
    
    def is_interior(self, a: float, b: float) -> Tuple[float, float]:
        if sqrt(a**2 + b**2) < self.radius:
            return (a,b)
        else:
            return None
