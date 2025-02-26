from aabb import AABB
from ray import Ray
from interval import Interval
from hittable import HitRecord, Hittable


class HittableList(Hittable):
    def __init__(self, object: Hittable = None):
        super().__init__()

        self._objects = []

        if object is not None:
            self.add(object)
    
    def add(self, obj: Hittable):
        self._objects.append(obj)
        self.bbox = AABB.from_bbox(box0=self.bbox, box1=obj.bbox)

    def hit(self, r: Ray, ray_t: Interval) -> HitRecord:
        rec: HitRecord = None
        closest_so_far = ray_t.max

        for object in self._objects:
            temp_record: HitRecord = object.hit(r, Interval.from_floats(ray_t.min, closest_so_far))
            if temp_record is not None:
                closest_so_far = temp_record.t
                rec = temp_record

        return rec
