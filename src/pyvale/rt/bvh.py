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

random.seed(237)

class BVH_Node(Hittable):
    
    def __init__(self, objects: List[Hittable]):
        end = len(objects)
        
        axis = random.randrange(3)
      
        object_span = end

        if object_span == 1:
            self._left = self._right = objects[0]
        elif object_span == 2:
            self._left = objects[0]
            self._right = objects[1]
        else:
            objects.sort(key=lambda obj: obj.bbox.axis_interval(axis).min)

            mid = int(end/2)
            self._left = BVH_Node(objects[0:mid])
            self._right = BVH_Node(objects[mid:end])
        
        self.bbox = AABB(box0 = self._left.bbox, box1 = self._right.bbox)
    
    def hit(self, r: Ray, ray_t: Interval) -> HitRecord:
        if not self.bbox.hit(r, ray_t):
            # Ray does not hit this box
            return None
        if isinstance(self._left, Quad) or isinstance(self._right, Quad):
            print("this one")
        
        hit_left = self._left.hit(r, ray_t)
        hit_right = self._right.hit(r, Interval(ray_t.min, hit_left.t if hit_left is not None else ray_t.max))

        if hit_right is not None:
            print("right")
            return hit_right
        elif hit_left is not None:
            print("left")
            return hit_left
        
        return None
