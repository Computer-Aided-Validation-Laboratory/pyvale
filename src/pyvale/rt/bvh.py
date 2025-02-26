import math
import random
from typing import List
import numpy as np
from quad import Quad
from ray import Ray
from interval import Interval
from hittable import HitRecord, Hittable
from material import Material
from aabb import AABB

class BVH_Node(Hittable):
    
    def __init__(self, objects: List[Hittable]):
        end = len(objects)
           
        object_span = end

        if object_span == 1:
            self._left = self._right = objects[0]
        elif object_span == 2:
            self._left = objects[0]
            self._right = objects[1]
        else:
            # axis = random.randrange(3)
            axis = 0

            sub_obs = list(objects)
            sub_obs.sort(key=lambda obj: obj.bbox.axis_interval(axis).min)

            mid = int(end/2)
            self._left = BVH_Node(sub_obs[0:mid])
            self._right = BVH_Node(sub_obs[mid:end])

        self.bbox = AABB.from_bbox(box0 = self._left.bbox, box1 = self._right.bbox)
    
    def hit(self, r: Ray, ray_t: Interval) -> HitRecord:
        if not self.bbox.hit(r, ray_t):
            # Ray does not hit this box
            return None
        
        hit_left = self._left.hit(r, ray_t)
        hit_right = self._right.hit(r, Interval.from_floats(ray_t.min, hit_left.t if hit_left is not None else ray_t.max))

        if hit_right is not None:
            return hit_right
        elif hit_left is not None:
            return hit_left
        
        return None
