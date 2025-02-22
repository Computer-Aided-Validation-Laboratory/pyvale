from ray import Ray
from interval import Interval
from hittable import HitRecord
from sphere import Sphere

class HittableList:
    objects = []

    def __init__(self, object: Sphere = None):
        if object:
            self.objects.append(object)

    def hit(self, r: Ray, ray_t: Interval) -> HitRecord:
        rec: HitRecord = HitRecord()
        temp_record: HitRecord = HitRecord()
        hit_anything = False
        closest_so_far = ray_t.max

        for object in self.objects:
            temp_record = object.hit(r, Interval(ray_t.min, closest_so_far))
            if temp_record:
                hit_anything = True
                closest_so_far = temp_record.t
                rec = temp_record

        if hit_anything:
            return rec
        else:
            return None