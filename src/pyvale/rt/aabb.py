import numpy as np
from ray import Ray
from interval import Interval

class AABB:
    x: Interval
    y: Interval
    z: Interval

    def __init__(self, *, x: Interval = None, y: Interval = None, z: Interval = None, a: np.ndarray = None, b: np.ndarray = None, box0: "AABB" = None, box1: "AABB" = None) -> None:
        # initialize using either none, (x,y,z) or (a,b) or (box0,box1), never combined.
        assert not (x and a and box0)
        if x is not None:
            self.x = x
            self.y = y
            self.z = z
        elif a is not None:
            self.x = Interval(a[0], b[0]) if a[0] < b[0] else Interval(b[0], a[0])
            self.y = Interval(a[1], b[1]) if a[1] < b[2] else Interval(b[1], a[1])
            self.z = Interval(a[2], b[2]) if a[1] < b[2] else Interval(b[2], a[2])
        elif box0 is not None:
            self.x = Interval(a = box0.x, b = box1.x)
            self.y = Interval(a = box0.y, b = box1.y)
            self.z = Interval(a = box0.z, b = box1.z)
        else:
            self.x = Interval()
            self.y = Interval()
            self.z = Interval()

    def axis_interval(self, n: int) -> Interval:
        if n == 1:
            return self.y
        if n == 2:
            return self.z
        return self.x
    
    def hit(self, r: Ray, ray_t: Interval) -> bool:
        for axis in [0, 1, 2]:
            ax = self.axis_interval(axis)
            adinv = 1 / r.direction[axis]

            t0 = (ax.min - r.origin[axis]) * adinv
            t1 = (ax.max - r.origin[axis]) * adinv

            if t0 < t1:
                if t0 > ray_t.min:
                    ray_t.min = t0
                if t1 < ray_t.max:
                    ray_t.max = t1
            else:
                if t1 > ray_t.min:
                    ray_t.min = t1
                if t0 < ray_t.max:
                    ray_t.max = t0
            
            if ray_t.max <= ray_t.min:
                return False
        return True