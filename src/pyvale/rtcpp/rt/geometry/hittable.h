#ifndef HITTABLE_H
#define HITTABLE_H
#include "../util.h"
#include "aabb.h"


class Material;



struct Hit_record {
    point3 position;
    vec3 normal;
    shared_ptr<Material> material_ptr;
    double distance; //last ray collision distance
    bool orientation;
    double u;
    double v;
    
    inline void set_face_normal(const Ray& r, const vec3& outward_normal) {
        orientation = dot(r.get_direction(), outward_normal) < 0;
        normal = orientation == UPWARDS ? outward_normal :-outward_normal;
    }
};


class Hittable {
    public:
        virtual bool hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const = 0;
        virtual bool bounding_box(double dis_min, double dis_max, aabb& output_box) const = 0;
        virtual double pdf_value(const vec3& o, const vec3& v) const {
            return 0.0;
        }

        virtual vec3 random(const vec3& o) const {
            return vec3(1, 0, 0);
        }
};

class Translate : public Hittable {
    public:
    Translate(shared_ptr<Hittable> object, const vec3& offset)
                                    : object(object), offset(offset)
            {
                aabb b;
                object->bounding_box(0, 0, b);
                bbox = b + offset;
            }

      bool hit(const Ray& r, double dis_min, double dis_max, Hit_record& rec) const override {
          // Move the ray backwards by the offset
          Ray offset_r(r.get_origin() - offset, r.get_direction(), r.get_depth(), r.get_time());
  
          // Determine whether an intersection exists along the offset ray (and if so, where)
          if (!object->hit(offset_r, dis_min, dis_max, rec))
              return false;
  
          // Move the intersection point forwards by the offset
          rec.position += offset;
  
          return true;
      }

        bool bounding_box(double dis_min, double dis_max, aabb& output_box) const override
        { 
            output_box = bbox;
            return true;
        }
  
    private:
      shared_ptr<Hittable> object;
      vec3 offset;
      aabb bbox;
  };

  class Rotate_y : public Hittable {
    public:
        Rotate_y(shared_ptr<Hittable> object, double angle) : object(object) {
            auto radians = degrees_to_radians(angle);
            sin_theta = std::sin(radians);
            cos_theta = std::cos(radians);
            object->bounding_box(0, 0, bbox);

            point3 min( infinity,  infinity,  infinity);
            point3 max(-infinity, -infinity, -infinity);

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < 2; k++) {
                        auto x = i*bbox.max().x() + (1-i)*bbox.min().x();
                        auto y = j*bbox.max().y() + (1-j)*bbox.min().y();
                        auto z = k*bbox.max().z() + (1-k)*bbox.min().z();

                        auto newx =  cos_theta*x + sin_theta*z;
                        auto newz = -sin_theta*x + cos_theta*z;

                        vec3 tester(newx, y, newz);

                        for (int c = 0; c < 3; c++) {
                            min[c] = std::fmin(min[c], tester[c]);
                            max[c] = std::fmax(max[c], tester[c]);
                        }
                    }
                }
            }

            bbox = aabb(min, max);
        }

  
      bool hit(const Ray& r, double dis_min, double dis_max, Hit_record& rec) const override {
  
          // Transform the ray from world space to object space.
  
          auto origin = point3(
              (cos_theta * r.get_origin().x()) - (sin_theta * r.get_origin().z()),
              r.get_origin().y(),
              (sin_theta * r.get_origin().x()) + (cos_theta * r.get_origin().z())
          );
  
          auto direction = vec3(
              (cos_theta * r.get_direction().x()) - (sin_theta * r.get_direction().z()),
              r.get_direction().y(),
              (sin_theta * r.get_direction().x()) + (cos_theta * r.get_direction().z())
          );
  
          Ray rotated_r(origin, direction, r.get_depth(), r.get_time());
  
          // Determine whether an intersection exists in object space (and if so, where).
  
          if (!object->hit(rotated_r, dis_min, dis_max, rec))
              return false;
  
          // Transform the intersection from object space back to world space.
  
          rec.position = point3(
              (cos_theta * rec.position.x()) + (sin_theta * rec.position.z()),
              rec.position.y(),
              (-sin_theta * rec.position.x()) + (cos_theta * rec.position.z())
          );
  
          rec.normal = vec3(
              (cos_theta * rec.normal.x()) + (sin_theta * rec.normal.z()),
              rec.normal.y(),
              (-sin_theta * rec.normal.x()) + (cos_theta * rec.normal.z())
          );
  
          return true;
      }

      bool bounding_box(double dis_min, double dis_max, aabb& output_box) const override
      { 
          output_box = bbox;
          return true;
      }

      private:
        shared_ptr<Hittable> object;
        double sin_theta;
        double cos_theta;
        aabb bbox;
  };





#endif
