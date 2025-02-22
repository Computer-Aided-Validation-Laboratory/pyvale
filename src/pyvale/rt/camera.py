import sys
import numpy as np
import math
from pyvale.rt.hittable import HitRecord
from pyvale.rt.hittable_list import HittableList
from interval import Interval

from ray import Ray

class Camera:
	# image_pixels: tuple[int, int] = (500, 400)
	image_width = 500
	image_height = 400
	position = (0,0,0)
	vfov = 90
	look_from = np.array([0,0,0])
	look_at = np.array([0,0,-1])
	v_up = (0,1,0)
	samples_per_pixel: int = 1
	max_depth: int = 5

	_u: np.ndarray
	_v: np.ndarray
	_w: np.ndarray
	_centre: np.ndarray
	_pixel00_loc: np.ndarray
	_pixel_delta_u: np.ndarray
	_pixel_delta_v: np.ndarray
	_pixel_samples_scale: float

	def __init__(self):
		self.pixel_samples_scale = 1.0 / self.samples_per_pixel
		self._centre = self.look_from
		focal_length = np.linalg.norm(self.look_from - self.look_at)
		theta = math.radians(self.vfov)
		h = math.tan(theta/2)
		viewport_height = 2* h * focal_length
		viewport_width = viewport_height * self.image_width / self.image_height

		# Calculate the u,v,w unit basis vectors for the camera coordinate frame.
		self._w = unit_vector(self.look_from - self.look_at)
		self._u = unit_vector(np.cross(self.v_up, self._w))
		self._v = np.cross(self._w, self._u)

		# Calculate the vectors across the horizontal and down the vertical viewport edges.
		viewport_u = viewport_width * self._u    # Vector across viewport horizontal edge
		viewport_v = viewport_height * -self._v

		# Calculate the horizontal and vertical delta vectors from pixel to pixel.
		self._pixel_delta_u = viewport_u / self.image_width
		self._pixel_delta_v = viewport_v / self.image_height

		# Calculate the location of the upper left pixel.
		viewport_upper_left = self._centre - (focal_length * self._w) - viewport_u/2 - viewport_v/2
		self._pixel00_loc = viewport_upper_left + 0.5 * (self._pixel_delta_u + self._pixel_delta_v)
	
	def _get_ray(self, i: int, j: int) -> Ray:
		# Construct a camera ray originating from the origin and directed at randomly sampled point around the pixel location i, j

		offset = np.random.random(2) - 0.5

		pixel_sample = self._pixel00_loc \
			+ ((i + offset[0]) * self._pixel_delta_u) \
			+ ((j + offset[1]) * self._pixel_delta_v)

		ray_origin = self._centre
		ray_direction = pixel_sample - ray_origin

		return Ray(ray_origin, ray_direction)
	
	def render(self, world: HittableList):
		# ensure initialized okay
		self.__init__()
		
		with open("im.ppm", "w") as f:
			f.write("P3\n")
			f.write(str(self.image_width) + " " + str(self.image_height) + "\n")
			f.write("255\n")

			for j in range(self.image_height):
				print("Scanlines remaining: ", (self.image_height - j), "\r", file=sys.stderr)
				for i in range(self.image_width):
					pixel_color: np.ndarray = np.array([0.0, 0.0, 0.0])
					for sample in range(self.samples_per_pixel):
						r: Ray = self._get_ray(i, j)
						pixel_color += self._ray_colour(r, self.max_depth, world)
					
					col = self.pixel_samples_scale * pixel_color
					r = col[0]
					g = col[1]
					b = col[2]
					r = 255 if r >= 1.0 else math.floor(r*256)
					g = 255 if g >= 1.0 else math.floor(g*256)
					b = 255 if b >= 1.0 else math.floor(b*256)
					col_str = str(r) + " " + str(g) + " " + str(b)
					f.write(col_str + "\n")
	
	def _ray_colour(self, r: Ray, depth: int, world: HittableList) -> np.ndarray:
		# If we've exceeded the ray bounce limit, no more light is gathered.
		if depth <= 0:
			return np.array([0, 0, 0])

		rec: HitRecord = world.hit(r, Interval(0.001, math.inf))

		if rec:
			# scattered: Ray = Ray()
			# attenuation: np.ndarray
			attenuation, scattered = rec.mat.scatter(r, rec)
			# if attenuation:
			return attenuation * self._ray_colour(scattered, depth-1, world)
			# return np.array([0, 0, 0])
		
		unit_direction = unit_vector(r.direction)
		a = 0.5*unit_direction[1] + 1
		return (1-a)*np.array([1,1,1]) + a*np.array([0.5, 0.7, 1])


def unit_vector(v):
	norm = np.linalg.norm(v)
	if norm == 0: 
	   return v
	return v / norm