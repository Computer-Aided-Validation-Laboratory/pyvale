from dataclasses import dataclass, field
import numpy as np
from ray import Ray
from interval import Interval

@dataclass
class AABB:
    x: Interval = field(default_factory=Interval)
    y: Interval = field(default_factory=Interval)
    z: Interval = field(default_factory=Interval)

    def __init__(self, x: Interval = None, y: Interval = None, z: Interval = None) -> None:
        # don't pad values if initialising an empty AABB
        bdefaults = True

        if x is None:
            self.x = Interval()
        else:
            self.x = x
            bdefaults = False
        if y is None:
            self.y = Interval()
        else:
            self.y = y
            bdefaults = False
        if z is None:
            self.z = Interval()
        else:
            self.z = z
            bdefaults = False
        
        if not bdefaults:
            self._pad_to_minimums()

        

    @classmethod
    def from_arrays(cls, a: np.ndarray, b: np.ndarray) -> "AABB":
        x = Interval.from_floats(a[0], b[0])
        y = Interval.from_floats(a[1], b[1])
        z = Interval.from_floats(a[2], b[2])
        cls = AABB(x, y, z)
        return cls
    
    @classmethod
    def from_bbox(cls, box0: "AABB", box1: "AABB") -> "AABB":
        x = Interval.from_intervals(box0.x, box1.x)
        y = Interval.from_intervals(box0.y, box1.y)
        z = Interval.from_intervals(box0.z, box1.z)
        cls = AABB(x, y, z)
        return cls
    
    def _pad_to_minimums(self) -> "AABB":
        # Adjust the AABB so that no side is narrower than some delta, padding if necessary
        delta = 0.001
        if self.x.size() < delta:
            self.x = self.x.expand(delta)
        if self.y.size() < delta:
            self.y = self.y.expand(delta)
        if self.z.size() < delta:
            self.z = self.z.expand(delta)

    def axis_interval(self, n: int) -> Interval:
        if n == 1:
            return self.y
        if n == 2:
            return self.z
        return self.x
    
    def hit(self, r: Ray, ray_t: Interval) -> bool:
        temp_ray_t: Interval = Interval.from_floats(ray_t.min, ray_t.max)

        for axis in [0, 1, 2]:
            ax = self.axis_interval(axis)
            adinv = 1 / r.direction[axis]

            t0 = (ax.min - r.origin[axis]) * adinv
            t1 = (ax.max - r.origin[axis]) * adinv

            if t0 < t1:
                if t0 > temp_ray_t.min:
                    temp_ray_t.min = t0
                if t1 < temp_ray_t.max:
                    temp_ray_t.max = t1
            else:
                if t1 > temp_ray_t.min:
                    temp_ray_t.min = t1
                if t0 < temp_ray_t.max:
                    temp_ray_t.max = t0
            
            if temp_ray_t.max <= temp_ray_t.min:
                return False
        return True
