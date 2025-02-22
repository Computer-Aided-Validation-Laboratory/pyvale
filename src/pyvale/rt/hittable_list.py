from aabb import AABB
from ray import Ray
from interval import Interval
from hittable import HitRecord, Hittable

class HittableList(Hittable):
    _objects = []
    # bbox: AABB

    def __init__(self, object: Hittable = None):
        if object:
            self.add(object)
    
    def add(self, object: Hittable):
        self._objects.append(object)
        self.bbox = AABB( box0=self.bbox, box1=object.bbox)

    def hit(self, r: Ray, ray_t: Interval) -> HitRecord:
        rec: HitRecord = HitRecord()
        temp_record: HitRecord = HitRecord()
        hit_anything = False
        closest_so_far = ray_t.max

        for object in self._objects:
            temp_record = object.hit(r, Interval(ray_t.min, closest_so_far))
            if temp_record:
                hit_anything = True
                closest_so_far = temp_record.t
                rec = temp_record

        if hit_anything:
            return rec
        else:
            return None