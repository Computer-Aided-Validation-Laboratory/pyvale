import math
from random import randrange
from typing import List
import numpy as np
from ray import Ray
from interval import Interval
from hittable import HitRecord, Hittable
from material import Material
from aabb import AABB

class BVH_Node(Hittable):
    _left: Hittable
    _right: Hittable
    bbox: AABB
    
    def __init__(self, objects: List[Hittable], start: int = 0, end: int = None):
        if end is None:
            end = len(objects)
        
        axis = randrange(3)

        comparator = self.box_x_compare if axis == 0 else self.box_y_compare if axis == 1 else self.box_z_compare
        
        object_span = end - start

        if object_span == 1:
            self._left = self._right = objects[start]
        elif object_span == 2:
            self._left = objects[start]
            self._right = objects[start+1]
        else:
            sublist = objects[start:end].sort()
            sublist.sort(key=comparator)

            mid = start + object_span/2
            self._left = BVH_Node(objects, start, mid)
            self._right = BVH_Node(objects, mid, end)
        
        self.bbox = AABB(box0 = self._left.bbox, box1 = self._right.bbox)
    
    def hit(self, r: Ray, ray_t: Interval) -> HitRecord:
        if not self.bbox.hit(r, ray_t):
            return None
        
        hit_left = self._left.hit(r, ray_t)
        hit_right = self._right.hit(r, Interval(ray_t.min, hit_left.t if hit_left else ray_t.max))

        return hit_left or hit_right
    
    @staticmethod
    def box_compare(a: Hittable, b: Hittable, axis_index) -> bool:
        a_axis_interval = a.bbox.axis_interval(axis_index)
        b_axis_interval = b.bbox.axis_interval(axis_index)
        return a_axis_interval.min < b_axis_interval.min

    @staticmethod
    def box_x_compare(a: Hittable, b: Hittable):
        return AABB.box_compare(a, b, 0)
    @staticmethod
    def box_y_compare(a: Hittable, b: Hittable):
        return AABB.box_compare(a, b, 1)
    @staticmethod
    def box_z_compare(a: Hittable, b: Hittable):
        return AABB.box_compare(a, b, 2)