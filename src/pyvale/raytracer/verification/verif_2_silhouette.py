from os import name

import numpy as np
from pathlib import Path

from pyvale.raytracer.rtblender import *
from pyvale.raytracer.rtcamera import Camera
from pyvale.raytracer.rtscene import *
from pyvale.raytracer.rtmain import render_scene
from pyvale.raytracer.rtmesh import *
from pyvale.raytracer.rtpresets import *

import verifconstants

from dataclasses import dataclass
from PIL import Image
import csv

from PIL import Image

base_dir = Path.cwd() / "verif" / "verif_2"
if not base_dir.is_dir():
    base_dir.mkdir(parents=True, exist_ok=True)

number_of_samples = 1; # for anti-aliasing
angle_vertical_view_default = 90  # degrees

def centroid(coords: np.ndarray) -> np.ndarray:
    return coords.mean(axis=0)

def default_rotation():
    return np.eye(3)

@dataclass
class CameraInput:
    pixels_num: tuple
    pixels_size: float
    pos_world: np.ndarray
    rot_world: np.ndarray
    roi_cent_world: np.ndarray
    focal_length: float
    sub_sample: int = 1
    distortion: str = "none"

@dataclass
class CameraPrepared:
    pixels_num: tuple
    pixels_size: float
    pos_world: np.ndarray
    rot_world: object
    roi_cent_world: np.ndarray
    focal_length: float

    sub_sample: int
    sensor_size: np.ndarray
    angle_vertical_view: float
    image_dims: np.ndarray
    image_dist: float

    distortion: object
    coord_sys: object

    ideal_pixel_centers: object
    pixel_center_jac: object

@dataclass
class ScalarMap:
    rows_num: int
    cols_num: int
    vals: np.ndarray

@dataclass
class CentroidStats:
    ideal_x: np.ndarray
    ideal_y: np.ndarray
    calc_x: np.ndarray
    calc_y: np.ndarray
    diff_x: np.ndarray
    diff_y: np.ndarray
    dist: float

@dataclass
class FOVScaling:
    plane_dist: float
    plane_size: np.ndarray
    leng_per_pixel: np.ndarray
    pixel_per_leng: np.ndarray

def buildCentroidCameraInput(
    mesh: RTMesh,
    pixels_num=(1024, 1024),
    pixel_size=1.0,
    focal_length=1.0,
    fov_scale=1.05,
) -> CameraInput:

    rot = default_rotation()

    coords = mesh.node_coords  # single frame

    roi_cent_world = centroid(coords)

    radius = np.linalg.norm(coords - roi_cent_world, axis=1).max()
    distance = fov_scale * radius
    pos_world = roi_cent_world + np.array([0.0, 0.0, distance])

    return CameraInput(
        pixels_num=pixels_num,
        pixels_size=pixel_size,
        pos_world=pos_world,
        rot_world=rot,
        roi_cent_world=roi_cent_world,
        focal_length=focal_length,
        sub_sample=1,
        distortion="none",
    )

def buildCentroidCameraInputOverFrames(
    mesh: RTMesh,
    pixels_num=(1024, 1024),
    pixel_size=1.0,
    focal_length=1.0,
    fov_scale=1.05,
) -> CameraInput:

    rot = default_rotation()

    # Reference centroid (frame 0)
    if mesh.node_coords_over_time is not None:
        base_coords = mesh.node_coords_over_time[0]
    else:
        base_coords = mesh.node_coords

    roi_cent_world = centroid(base_coords)

    # Worst-case extent over time
    max_radius = 0.0

    if mesh.node_coords_over_time is not None:
        for t in range(mesh.timestep_count):
            coords = mesh.node_coords_over_time[t]
            r = np.linalg.norm(coords - roi_cent_world, axis=1).max()
            max_radius = max(max_radius, r)
    else:
        # Fallback static mesh
        max_radius = np.linalg.norm(mesh.node_coords - roi_cent_world, axis=1).max()

    distance = fov_scale * max_radius

    pos_world = roi_cent_world + np.array([0.0, 0.0, distance])

    return CameraInput(
        pixels_num=pixels_num,
        pixels_size=pixel_size,
        pos_world=pos_world,
        rot_world=rot,
        roi_cent_world=roi_cent_world,
        focal_length=focal_length,
        sub_sample=1,
        distortion="none",
    )


def calcSensorSize(pixels_num, pixels_size):
    return np.array([
        pixels_num[0] * pixels_size,
        pixels_num[1] * pixels_size
    ], dtype=float)




def initForSubPixelCenterMap(
    input,
    subpixel_center_map
) -> CameraPrepared:

    actual_sub_sample = input.sub_sample if input.sub_sample != 0 else 2
    sensor_size = calcSensorSize(input.pixels_num, input.pixels_size)
    angle_vertical_view = angle_vertical_view_default

    pos_w = input.pos_world

    image_dist = np.linalg.norm(pos_w - input.roi_cent_world)

    image_dims = np.zeros(2, dtype=float)
    image_dims[0] = (image_dist / input.focal_length) * sensor_size[0]
    image_dims[1] = (image_dist / input.focal_length) * sensor_size[1]

    # Ideal pixel centers
    if subpixel_center_map == "full_in_mem":
        sub_samp_u = int(actual_sub_sample)

        dims = (
            input.pixels_num[1] * sub_samp_u,
            input.pixels_num[0] * sub_samp_u,
            2
        )
        ideal_pixel_centers = np.zeros(dims, dtype=float)

    else:
        ideal_pixel_centers = np.zeros((0, 0, 2), dtype=float)

    # Pixel jacobian
    if subpixel_center_map == "affine_jac":
        dims = (
            input.pixels_num[1],
            input.pixels_num[0],
            6
        )
        pixel_center_jac = np.zeros(dims, dtype=float)
    else:
        pixel_center_jac = np.zeros((0, 0, 6), dtype=float)

    return CameraPrepared(
        pixels_num=input.pixels_num,
        pixels_size=input.pixels_size,
        pos_world=input.pos_world,
        rot_world=input.rot_world,
        roi_cent_world=input.roi_cent_world,
        focal_length=input.focal_length,

        sub_sample=actual_sub_sample,
        sensor_size=sensor_size,
        angle_vertical_view=angle_vertical_view,
        image_dims=image_dims,
        image_dist=image_dist,

        distortion=input.distortion,
        coord_sys=getattr(input, "coord_sys", None),

        ideal_pixel_centers=ideal_pixel_centers,
        pixel_center_jac=pixel_center_jac,
    )

def extractScalarMapBmp(path: str) -> dict:

    img = Image.open(path).convert("F")  # 32-bit float grayscale

    arr = np.array(img, dtype=np.float64)

    rows_num, cols_num = arr.shape

    vals = arr.reshape(rows_num * cols_num)

    return ScalarMap(
        rows_num=rows_num, 
        cols_num=cols_num,
        vals=vals
    )


def calcCentroidStats(camera_input, rows_num, cols_num, vals):

    # Ideal center (image center in pixels)
    ideal_x = 0.5 * float(camera_input.pixels_num[0])
    ideal_y = 0.5 * float(camera_input.pixels_num[1])

    sum_w = 0.0
    sum_x = 0.0
    sum_y = 0.0

    for rr in range(rows_num):
        for cc in range(cols_num):

            # weight = vals[rr * cols_num + cc]
            weight = 1.0 if vals[rr * cols_num + cc] > 0 else 0.0

            if not (weight > 0.0):
                continue

            x = float(cc) + 0.5
            y = float(rr) + 0.5

            sum_w += weight
            sum_x += x * weight
            sum_y += y * weight

    if sum_w == 0.0:
        raise ValueError("Empty silhouette (no positive pixels found)")

    calc_x = sum_x / sum_w
    calc_y = sum_y / sum_w

    diff_x = calc_x - ideal_x
    diff_y = calc_y - ideal_y

    dist = np.sqrt(diff_x * diff_x + diff_y * diff_y)

    return CentroidStats(
        ideal_x=ideal_x,
        ideal_y=ideal_y,
        calc_x=calc_x,
        calc_y=calc_y,
        diff_x=diff_x,
        diff_y=diff_y,
        dist=dist,
    )


def writeStatsCsv(
    out_dir: Path,
    file_name: str,
    stats: dict,
    centroid_world: np.ndarray,
    scaling,
    camera_prepared
):
    file_path = out_dir / file_name

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header: key,unit,value
        writer.writerow(["key", "unit", "value"])

        writer.writerow(["cent_ideal_x", "px", f"{stats.ideal_x:.17f}"])
        writer.writerow(["cent_ideal_y", "px", f"{stats.ideal_y:.17f}"])
        writer.writerow(["cent_calc_x", "px", f"{stats.calc_x:.17f}"])
        writer.writerow(["cent_calc_y", "px", f"{stats.calc_y:.17f}"])
        writer.writerow(["diff_x", "px", f"{stats.diff_x:.17f}"])
        writer.writerow(["diff_y", "px", f"{stats.diff_y:.17f}"])
        writer.writerow(["dist", "px", f"{stats.dist:.17f}"])

        writer.writerow(["sensor_pixels_x", "px", camera_prepared.pixels_num[0]])
        writer.writerow(["sensor_pixels_y", "px", camera_prepared.pixels_num[1]])

        writer.writerow(["centroid_x", "length", f"{centroid_world[0]:.17f}"])
        writer.writerow(["centroid_y", "length", f"{centroid_world[1]:.17f}"])
        writer.writerow(["centroid_z", "length", f"{centroid_world[2]:.17f}"])

        writer.writerow(["plane_dist", "length", f"{scaling.plane_dist:.17f}"])
        writer.writerow(["plane_size_x", "length", f"{scaling.plane_size[0]:.17f}"])
        writer.writerow(["plane_size_y", "length", f"{scaling.plane_size[1]:.17f}"])

        writer.writerow([
            "leng_per_pixel_x",
            "length/px",
            f"{scaling.leng_per_pixel[0]:.17f}",
        ])
        writer.writerow([
            "leng_per_pixel_y",
            "length/px",
            f"{scaling.leng_per_pixel[1]:.17f}",
        ])

        writer.writerow([
            "pixel_per_leng_x",
            "px/length",
            f"{scaling.pixel_per_leng[0]:.17f}",
        ])
        writer.writerow([
            "pixel_per_leng_y",
            "px/length",
            f"{scaling.pixel_per_leng[1]:.17f}",
        ])


def calcFOVScaling(camera_input, plane_cent_world):
    # Camera Z axis (third column of rotation matrix)
    cam_z_axis = camera_input.rot_world[:, 2]

    # Vector from camera to plane center
    plane_vec = camera_input.pos_world - plane_cent_world

    # Distance along camera forward axis
    plane_dist = abs(np.dot(plane_vec, cam_z_axis))

    # Sensor size in world units
    sensor_size = calcSensorSize(
        camera_input.pixels_num,
        camera_input.pixels_size
    )

    # Projected plane size
    plane_size = np.zeros(2, dtype=float)
    plane_size[0] = (plane_dist / camera_input.focal_length) * sensor_size[0]
    plane_size[1] = (plane_dist / camera_input.focal_length) * sensor_size[1]

    # Length per pixel
    leng_per_pixel = np.zeros(2, dtype=float)
    leng_per_pixel[0] = plane_size[0] / float(camera_input.pixels_num[0])
    leng_per_pixel[1] = plane_size[1] / float(camera_input.pixels_num[1])

    # Pixel per length
    pixel_per_leng = np.zeros(2, dtype=float)
    pixel_per_leng[0] = 1.0 / leng_per_pixel[0]
    pixel_per_leng[1] = 1.0 / leng_per_pixel[1]

    return FOVScaling(
        plane_dist=plane_dist,
        plane_size=plane_size,
        leng_per_pixel=leng_per_pixel,
        pixel_per_leng=pixel_per_leng,
    )

def make_test_dir(base_dir: Path, test_name: str):
    """
    Small helper function to make separate directories for each test to avoid overwriting data.
    """
    test_dir = base_dir.joinpath(test_name)
    if not test_dir.is_dir():
        test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


def writeScalarMapCsv(out_dir: Path,
                      file_name: str,
                      scalar_map):

    path = out_dir / file_name

    with open(path, "w") as f:
        for r in range(scalar_map.rows_num):
            base = r * scalar_map.cols_num

            for c in range(scalar_map.cols_num):
                f.write(str(scalar_map.vals[base + c]))
                f.write(",")

            f.write("\n")


def verif_case2_test() -> None:

    for case_spec in verifconstants.distort_cases:

        data_path = Path(Path().resolve().joinpath(case_spec["data_dir"]))
        out_dir = make_test_dir(base_dir, "b_" + str(case_spec["mesh_type"].name  + "_" + case_spec["case_name"]))
        # camera_target = np.array(case_spec["camera_input"]["pos_world"])
        # camera_center = np.array(case_spec["camera_input"]["roi_cent_world"])

        camera_target = np.array(case_spec["camera_input"]["roi_cent_world"])
        camera_center = np.array(case_spec["camera_input"]["pos_world"])

        test_rtmesh = simdata_csv_to_rtmesh(directory = data_path, 
                                            spatial_dim = sens.EDim.TWOD, 
                                            world_position = np.array([0.0, 0.0, 0.0]))
        
        # print(test_rtmesh.timestep_count)

        # print(test_rtmesh.node_coords_over_time)
        
        if case_spec["case_name"] == "rot":
            base_camera_input = buildCentroidCameraInputOverFrames(
                test_rtmesh,
                pixels_num=(1024, 1024),
                pixel_size=1.0,
                focal_length=1.0,
                fov_scale=1.05,
            )
        else:
            base_camera_input = buildCentroidCameraInput(
                test_rtmesh,
                pixels_num=(1024, 1024),
                pixel_size=1.0,
                focal_length=1.0,
                fov_scale=1.05,
            )

        camera_prepared = initForSubPixelCenterMap(base_camera_input, "full_in_mem")

        scene = Scene()

        image_width = camera_prepared.pixels_num[0]
        image_height = camera_prepared.pixels_num[1]
        # camera_center = camera_prepared.roi_cent_world
        # camera_target = camera_prepared.pos_world

        camera_center = camera_prepared.pos_world
        camera_target = camera_prepared.roi_cent_world
        angle_vertical_view = camera_prepared.angle_vertical_view

        # print(image_width, image_height)
        # print(camera_center)
        # print(camera_target)
        # print(angle_vertical_view)

        dic_cam = Camera(image_width, image_height, camera_center, camera_target, angle_vertical_view)

        scene.add_camera(dic_cam)
        
        # print(test_rtmesh.node_coords_over_time)
    
    
        test_rtmesh.set_surface(surface_type = SurfType.FIELD_COLOR,
                                surface_fill = np.ones(3) * 1.0, # White
                                material_type = MaterialType.UNLIT)
        scene.add_rtmesh(test_rtmesh)
        scene.set_background(np.ones(3) * 0.0) # Black

        # render_scene(image_height, image_width, scene, number_of_samples, test_dir, RenderType.STATIC, frames_to_render = 10)
        render_scene(image_height, image_width, scene, number_of_samples, out_dir, RenderType.DYNAMIC)

        for frame_idx in range(test_rtmesh.timestep_count):
            img_path = out_dir / Path(f"rtimage_{frame_idx}_cam0.bmp")
            scalar_map = extractScalarMapBmp(img_path)
            stats = calcCentroidStats(base_camera_input, 
                                        scalar_map.rows_num, 
                                        scalar_map.cols_num, 
                                        scalar_map.vals)
            
            fov_scaling = calcFOVScaling(base_camera_input, base_camera_input.roi_cent_world)
            
            
            writeScalarMapCsv(out_dir=out_dir,
                              file_name=f"cam0_frame{frame_idx}_field0.csv",
                              scalar_map=scalar_map)
            writeStatsCsv(out_dir=out_dir,
                          file_name=f"cam0_frame{frame_idx}_field0_stats.csv",
                          stats=stats,
                          centroid_world=base_camera_input.roi_cent_world,
                          scaling = fov_scaling,
                          camera_prepared=camera_prepared)
            
            mesh_name = case_spec["mesh_type"].name
            case_name = case_spec["case_name"]
            print(
                f"b_{mesh_name}_{case_name} frame {frame_idx}: "
                f"centroid dist={stats.dist:.6e}"
            )
def verif_case2_generate_gifs() -> None:

    for case_spec in verifconstants.distort_cases:
        data_path = Path(Path().resolve().joinpath(case_spec["data_dir"]))
        out_dir = make_test_dir(base_dir, "b_" + str(case_spec["mesh_type"].name  + "_" + case_spec["case_name"]))
        test_rtmesh = simdata_csv_to_rtmesh(directory = data_path, 
                                    spatial_dim = sens.EDim.TWOD, 
                                    world_position = np.array([0.0, 0.0, 0.0]))
        frames = []
        for frame_idx in range(test_rtmesh.timestep_count):
            img_path = out_dir / Path(f"rtimage_{frame_idx}_cam0.bmp")
            frames.append(Image.open(img_path))

        # frames[0].save(
        #     out_dir / "animation.gif",
        #     save_all=True,
        #     append_images=frames[1:] + frames[-2::-1],
        #     duration=300,   # milliseconds per frame
        #     loop=0,
        # )


        ping_pong_frames = frames + frames[-2:0:-1]
        ping_pong_frames[0].save(
            out_dir / "animation.gif",
            save_all=True,
            append_images=ping_pong_frames[1:],
            duration=200,
            loop=0,
        )
        
def main():
    verif_case2_test()
    # verif_case2_generate_gifs()


if __name__ == "__main__":
    main()


