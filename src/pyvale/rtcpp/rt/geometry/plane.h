#ifndef PLANE_H
#define PLANE_H
#include "../util.h"

#include "hittable.h"

#include "aabb.h"

#include "../interval.h"

class Quad : public Hittable {
    public:
      Quad(const point3& Q, const vec3& u, const vec3& v, shared_ptr<Material> mat)
        : Q(Q), u(u), v(v), mat(mat)
      {
          auto n = cross(u, v);
          normal = normalize(n);
          D = dot(normal, Q);
          w = n / dot(n,n);
  
          area = n.length();
  
          set_bounding_box();
      }
  
      virtual void set_bounding_box() {
          // Compute the bounding box of all four vertices.
          auto bbox_diagonal1 = aabb(Q, Q + u + v);
          auto bbox_diagonal2 = aabb(Q + u, Q + v);
          bbox = aabb(bbox_diagonal1, bbox_diagonal2);
      }
  
    //   aabb bounding_box() const override { return bbox; }
        bool bounding_box(double dis_min, double dis_max, aabb& output_box) const override
        {
            output_box = bbox;
            return true;
        }
  
      bool hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const override {
          auto denom = dot(normal, r.get_direction());
  
          // No hit if the ray is parallel to the plane.
          if (std::fabs(denom) < 1e-8)
              return false;
  
          // Return false if the hit point parameter t is outside the ray interval.
          auto t = (D - dot(normal, r.get_origin())) / denom;
        //   if (!ray_t.contains(t))
        //       return false;
        if (t < dis_min || t > dis_max)
            return false;
  
          // Determine if the hit point lies within the planar shape using its plane coordinates.
          auto intersection = r.at(t);
          vec3 planar_hitpt_vector = intersection - Q;
          auto alpha = dot(w, cross(planar_hitpt_vector, v));
          auto beta = dot(w, cross(u, planar_hitpt_vector));
  
          if (!is_interior(alpha, beta, hit))
              return false;
  
          // Ray hits the 2D shape; set the rest of the hit record and return true.
          hit.distance = t;
          hit.position = intersection;
          hit.material_ptr = mat;
          hit.set_face_normal(r, normal);
  
          return true;
      }
  
      virtual bool is_interior(double a, double b, Hit_record& rec) const {
          Interval unit_interval = Interval(0, 1);
          // Given the hit point in plane coordinates, return false if it is outside the
          // primitive, otherwise set the hit record UV coordinates and return true.
  
          if (!unit_interval.contains(a) || !unit_interval.contains(b))
              return false;
  
          rec.u = a;
          rec.v = b;
          return true;
      }
  
      double pdf_value(const point3& origin, const vec3& direction) const override {
          Hit_record hit;
          if (!this->hit(Ray(origin, direction, 1, 0.), 0.001, infinity, hit))
              return 0;
  
          auto distance_squared = hit.distance * hit.distance * direction.length_squared();
          auto cosine = std::fabs(dot(direction, hit.normal) / direction.length());
  
          return distance_squared / (cosine * area);
      }
  
      vec3 random(const point3& origin) const override {
          auto p = Q + (random_double() * u) + (random_double() * v);
          return p - origin;
      }
  
    private:
      point3 Q;
      vec3 u, v;
      vec3 w;
      shared_ptr<Material> mat;
      aabb bbox;
      vec3 normal;
      double D;
      double area;
  };


class Tri : public Quad {
    public:
        Tri(const point3& Q, const vec3& u, const vec3& v, shared_ptr<Material> mat)
        : Quad(Q, u, v, mat) {};

        bool is_interior(double a, double b, Hit_record& rec) const override{
            if (a > 0 && b > 0 && a+b < 1){
                rec.u = a;
                rec.v = b;
                return true;
            }
            return false;
        }
};


class Plane_xy: public Hittable {
    public:

        Plane_xy(
            double _x0, double _x1, double _y0, double _y1, double _z, shared_ptr<Material> mat
        ) : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_z), mp(mat) {};

        virtual bool hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const;

        virtual bool bounding_box(double dis_min, double dis_max, aabb& output_box) const {
            // The bounding box must have non-zero width in each dimension, so pad the Z
            // dimension a small amount.
            output_box = aabb(point3(x0,y0, k-0.0001), point3(x1, y1, k+0.0001));
            return true;
        }

    public:
        shared_ptr<Material> mp;
        double x0, x1, y0, y1, k;
};

class Plane_xz: public Hittable {
    public:

        Plane_xz(
            double _x0, double _x1, double _z0, double _z1, double _y, shared_ptr<Material> mat
        ) : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_y), mp(mat) {};

        virtual bool hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const;

        virtual bool bounding_box(double dis_min, double dis_max, aabb& output_box) const {
            // The bounding box must have non-zero width in each dimension, so pad the Y
            // dimension a small amount.
            output_box = aabb(point3(x0,k-0.0001,z0), point3(x1, k+0.0001, z1));
            return true;
        }

        virtual double pdf_value(const point3& o, const vec3& v) const {
            Hit_record hit;
            if (!this->hit(Ray(o, v, 1, 0.), 0.0001, infinity, hit))
                return 0;

            double area = (x1 - x0) * (z1 - z0);
            double distance_squared = hit.distance * hit.distance;
            double cosine = (dot(-v, hit.normal));
            vec3 center = vec3((x0 + x1) / 2, k, (z0 + z1) / 2);


            return clamp(distance_squared / (cosine * area), 0.01, 10000);
        }

        virtual vec3 random(const point3& origin) const {

            auto random_point = point3(random_double(x0, x1), k, random_double(z0, z1));
            return normalize(random_point - origin);
        }

        /*virtual double pdf_value(const point3& o, const vec3& scatter_direction) const {
            Hit_record hit;

            vec3 center = vec3((x0 + x1) / 2, k, (z0 + z1) / 2);
            double radius = vec3((x0 - x1) / 2, 0, (z0 - z1) / 2).length();
            double target_distance = (center - o).length();
            double cos_theta_max = sqrt(1 - clamp(radius * radius / (target_distance * target_distance), 0., 1.));

            if (dot(scatter_direction, normalize(center - o)) > cos_theta_max) {
                return  1 / ((1 - cos_theta_max) * 2 * pi);
            }
            else {
                return 0;
            }



            auto solid_angle = 2 * pi * (1 - cos_theta_max);

            return  1 / solid_angle;
        }

        virtual vec3 random(const point3& o) const {
            vec3 center = vec3((x0 + x1) / 2, k, (z0 + z1) / 2);
            double radius = vec3((x0 - x1) / 2, 0, (z0 - z1) / 2).length();
            double target_distance = (center - o).length();
            double cos_theta_max = sqrt(1 - clamp(radius * radius / (target_distance * target_distance), 0., 1.));

            onb uvw;
            uvw.build_from_w(center - o);
            double phi = random_double() * 2 * pi;
            double r2 = random_double();

            double z = 1. + r2 * (cos_theta_max - 1.);
            double x = cos(phi) * sqrt(1. - z * z);
            double y = sin(phi) * sqrt(1. - z * z);

            return uvw.local(vec3(x, y, z));


        }*/
    public:
        shared_ptr<Material> mp;
        double x0, x1, z0, z1, k;
};

class Plane_yz: public Hittable {
    public:

        Plane_yz(
            double _y0, double _y1, double _z0, double _z1, double _x, shared_ptr<Material> mat
        ) : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_x), mp(mat) {};

        virtual bool hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const;

        virtual bool bounding_box(double dis_min, double dis_max, aabb& output_box) const {
            // The bounding box must have non-zero width in each dimension, so pad the X
            // dimension a small amount.
            output_box = aabb(point3(k-0.0001, y0, z0), point3(k+0.0001, y1, z1));
            return true;
        }

    public:
        shared_ptr<Material> mp;
        double y0, y1, z0, z1, k;
};

bool Plane_xy::hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const {
    auto dis = (k-r.get_origin().z()) / r.get_direction().z();
    if (dis < dis_min || dis > dis_max)
        return false;

    auto x = r.get_origin().x() + dis*r.get_direction().x();
    auto y = r.get_origin().y() + dis*r.get_direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1)
        return false;

    hit.u = (x-x0)/(x1-x0);
    hit.v = (y-y0)/(y1-y0);
    hit.distance = dis;
    auto outward_normal = vec3(0, 0, 1);
    hit.set_face_normal(r, outward_normal);
    hit.material_ptr = mp;
    hit.position = r.at(dis);

    return true;
}

bool Plane_xz::hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const {
    auto dis = (k-r.get_origin().y()) / r.get_direction().y();
    if (dis < dis_min || dis > dis_max)
        return false;

    auto x = r.get_origin().x() + dis*r.get_direction().x();
    auto z = r.get_origin().z() + dis*r.get_direction().z();
    if (x < x0 - 0. || x > x1 + 0. || z < z0 - 0. || z > z1 + 0.)

        return false;

    hit.u = (x-x0)/(x1-x0);
    hit.v = (z-z0)/(z1-z0);
    hit.distance = dis;
    auto outward_normal = vec3(0, 1, 0);
    hit.set_face_normal(r, outward_normal);
    hit.material_ptr = mp;
    hit.position = r.at(dis);

    return true;
}

bool Plane_yz::hit(const Ray& r, double dis_min, double dis_max, Hit_record& hit) const {
    auto dis = (k-r.get_origin().x()) / r.get_direction().x();
    if (dis < dis_min || dis > dis_max)
        return false;

    auto y = r.get_origin().y() + dis*r.get_direction().y();
    auto z = r.get_origin().z() + dis*r.get_direction().z();
    if (y < y0 || y > y1 || z < z0 || z > z1)
        return false;

    hit.u = (y-y0)/(y1-y0);
    hit.v = (z-z0)/(z1-z0);
    hit.distance = dis;
    auto outward_normal = vec3(1, 0, 0);
    hit.set_face_normal(r, outward_normal);
    hit.material_ptr = mp;
    hit.position = r.at(dis);

    return true;
}








#endif
