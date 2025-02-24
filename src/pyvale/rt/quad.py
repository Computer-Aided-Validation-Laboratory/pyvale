import numpy as np
from ray import Ray
from interval import Interval
from hittable import Hittable, HitRecord
from material import Material
from aabb import AABB

class Quad(Hittable):
    _q: np.ndarray
    _u: np.ndarray
    _v: np.ndarray
    _w: np.ndarray
    _mat: Material
    # bbox: AABB
    _normal: np.ndarray
    _d: float

    def __init__(self, q, u, v, mat: Material):
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
        bbox_diagonal1 = AABB(a=self._q, b=self._q+self._u, z=self._v)
        bbox_diagonal2 = AABB(a=self._q + self._u, b=self._q + self._v)
        self.bbox = AABB(box0=bbox_diagonal1, box1=bbox_diagonal2)
    
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

        rec: HitRecord = self.is_interior(alpha, beta)
        if not rec:
            return None

        # Ray hits the 2D shape; set the rest of the hit record and return true.
        rec.t = t
        rec.p = intersection
        rec.mat = self._mat
        rec.set_face_normal(r, self._normal)
        return rec

    def is_interior(self, a: float, b: float) -> HitRecord:
        unit_interval = Interval(0,1)
        # Given the hit point in plane coordinates, return false if it is outside the primitive, otherwise set the hit record UV coordinates and return true.

        if (not unit_interval.contains(a)) or (not unit_interval.contains(b)):
            return None
        
        rec: HitRecord = HitRecord()
        rec.u = a
        rec.v = b
        return rec