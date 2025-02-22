from typing import Tuple
import numpy as np
from ray import Ray
# from hittable import HitRecord

class Material:
    def scatter(self, r_in: Ray, rec) -> Tuple[np.ndarray, np.ndarray]:
        pass

class Lambertian(Material):
    albedo: np.ndarray

    def __init__(self, albedo: np.ndarray) -> None:
        super().__init__()
        self.albedo = albedo
    
    def scatter(self, r_in: Ray, rec) -> Tuple[np.ndarray, np.ndarray]:
        scatter_direction = rec.normal + random_unit_vector()

        #  Catch degenerate scatter direction
        if np.allclose(scatter_direction, 0):
            scatter_direction = rec.normal
        
        scattered = Ray(rec.p, scatter_direction)
        attenuation = self.albedo
        return (attenuation, scattered)

# class Metal(Material):
#     albedo: np.ndarray
#     fuzz: float

#     def __init__(self, color, fuzz: float) -> None:
#         self.albedo = color
#         self.fuzz = fuzz if fuzz < 1 else 1
    
#     def scatter(self, r_in: Ray, rec) -> Tuple[np.ndarray, np.ndarray]:
#         reflected = reflect
        

def random_on_hemisphere(normal: np.ndarray):
    vec = np.random.randn(3)
    vec /= np.linalg.norm(vec)
    if np.dot(vec, normal) > 0.0: # same hemesphere
        return vec
    else:
        return -vec

def random_unit_vector():
    v = np.random.randn(3)
    return v / np.linalg.norm(v)