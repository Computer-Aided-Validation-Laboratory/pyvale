"""
Application test 4: Mechanical plate with FEA data - stereo DIC
Based on application_2_fea.py.

It renders the image sets needed by a stereo-DIC workflow,
following the structure used by ecomass_pres_dic/demo_dic.py:

1. Stereo calibration images: a planar dot target viewed by both cameras at
   several poses (translations and small rotations).
2. Stereo DIC images: an undeformed reference pair and a deformed pair, i.e.
   one image per camera for each state.

The specimen, pipe and water set-up is identical to application_2_fea.py and
the same three optical cases are available: AIR_DIFFUSE, PIPE, WATER. By
default the calibration target is rendered through the same optical path as
the specimen (pass cal_through_media=False to render it in air) so that the
calibration also captures any refraction introduced by the pipe and the water
fill.

The camera rig is a two-camera stereo pair. Camera 0 is the nominal
application_2_fea.py camera; camera 1 is obtained by rotating camera 0 about
the region of interest by STEREO_ANGLE_DEG about the world Y axis.

Images are 8-bit mono BMPs named rtimage_{frame}_cam{camera}.bmp, which is the
naming convention produced by the ray tracer for multi-camera scenes.

Outputs (relative to thesis-output/):
  app4_fea/{case}/stereocal/rtimage_{pose}_cam{0,1}.bmp
  app4_fea/{case}/stereodic/rtimage_{0,1}_cam{0,1}.bmp
  app4_fea/{case}/calplate/...      SimData CSVs for the calibration target
  app4_fea/{case}/stereo_geometry.csv
"""
import csv
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from global_utils import *

from pyvale.sensorsim import EDim
from pyvale.sensorsim.imagetools import ImageTools
from pyvale.raytracer.rtmesh import *
from pyvale.raytracer.rtcamera import *
from pyvale.raytracer.rtscene import *
from pyvale.raytracer.rtpresets import *
from pyvale.raytracer.rtmain import *
from pyvale.raytracer.rtoutputformat import *
from pyvale.raytracer.rtmeshvisuals import SeamSelector, SceneVisualiser


# ================================================================================
# Data path helpers
# ================================================================================
# These mirror the helpers in convergence_common. They are repeated here so that
# this application script only depends on global_utils and pyvale.raytracer.

def get_tank_path(tank_access: str, element: Element) -> Path:
    return full_path(tank_access + "/tank_surface_" + element.label + ".vtk")


def get_fill_path(tank_access: str, element: Element) -> Path:
    return full_path(tank_access + "/fill_surface_" + element.label + ".vtk")


# ================================================================================
# Stereo and calibration settings
# ================================================================================

STEREO_ANGLE_DEG = 5.0 # Total angle between the two camera axes, in degrees

PLATE_HEIGHT_MM = 35.0 # Physical height of the (25, 35, 1) mm plate

CAL_TARGET_ACCESS = "thesis-data/texture/cal_target-simple.tiff"
CAL_TARGET_ASPECT = 2660.0 / 1774.0 # Width / height of the dot target texture
CAL_TARGET_FOV_FRACTION = 0.90 # Target size as a fraction of the rendered FOV
CAL_PIPE_CLEARANCE_MM = 0.5 # Keep the target this far inside the pipe inner wall
CAL_TRANS_MM = 1.0 # Out-of-plane/in-plane translation of the target poses
CAL_TILT_DEG = 5.0 # Tilt of the target poses

CASE_BASE_DIR = "app4_fea"
CALPLATE_DIRNAME = "calplate"
STEREOCAL_DIRNAME = "stereocal"
STEREODIC_DIRNAME = "stereodic"


@dataclass(frozen=True)
class CalibrationPose:
    """Small rigid-body pose of the calibration target, relative to the ROI."""
    dx_mm: float = 0.0
    dy_mm: float = 0.0
    dz_mm: float = 0.0
    rx_deg: float = 0.0
    ry_deg: float = 0.0
    rz_deg: float = 0.0


def calibration_poses() -> list[CalibrationPose]:
    """
    Returns the zero pose followed by symmetric single-axis translations and
    tilts. Thirteen poses in total: 1 zero + 3 x 2 translations + 3 x 2 tilts.
    """
    poses = [CalibrationPose()]
    for axis in ("dx_mm", "dy_mm", "dz_mm"):
        for sign in (-1.0, 1.0):
            poses.append(CalibrationPose(**{axis: sign * CAL_TRANS_MM}))
    for axis in ("rx_deg", "ry_deg", "rz_deg"):
        for sign in (-1.0, 1.0):
            poses.append(CalibrationPose(**{axis: sign * CAL_TILT_DEG}))
    return poses


def calibration_target_size(fov_width_mm: float,
                            fov_height_mm: float,
                            max_length_mm: float | None = None
                            ) -> tuple[float, float]:
    """
    Picks the largest target that keeps the texture aspect ratio, fits inside
    the rendered field of view, and (optionally) stays inside the pipe.
    """
    length = CAL_TARGET_FOV_FRACTION * fov_width_mm
    if max_length_mm is not None:
        length = min(length, max_length_mm)
    height = length / CAL_TARGET_ASPECT
    max_height = CAL_TARGET_FOV_FRACTION * fov_height_mm
    if height > max_height:
        height = max_height
        length = height * CAL_TARGET_ASPECT
    return length, height


def write_calplate_simdata(directory: Path,
                           length_mm: float,
                           height_mm: float,
                           poses: list[CalibrationPose]) -> None:
    """
    Writes the SimData CSVs for a planar TRI3 calibration target built from a
    single rectangle split along one diagonal, in the same layout read by
    simdata_csv_to_rtmesh().
    """
    directory.mkdir(parents=True, exist_ok=True)

    # simdata_csv_to_rtmesh() infers the timestep axis from the CSV shape and
    # cannot represent a single timestep, so require at least two poses.
    if len(poses) < 2:
        raise ValueError("At least two calibration poses are required by "
                         "simdata_csv_to_rtmesh().")

    half_l = 0.5 * length_mm
    half_h = 0.5 * height_mm
    coords = np.array([
        [-half_l, -half_h, 0.0],
        [ half_l, -half_h, 0.0],
        [ half_l,  half_h, 0.0],
        [-half_l,  half_h, 0.0],
    ], dtype=np.float64)
    connect = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int64)
    uvs = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float64)

    # Displacement of each node from the zero pose, for every pose. The
    # rotations are about the target centre because coords are centred on zero.
    displacements = np.zeros((coords.shape[0], len(poses), 3), dtype=np.float64)
    for frame, pose in enumerate(poses):
        rotation = Rotation.from_euler(
            "xyz",
            [pose.rx_deg, pose.ry_deg, pose.rz_deg],
            degrees=True,
        ).as_matrix()
        rotated = coords @ rotation.T
        translation = np.array([pose.dx_mm, pose.dy_mm, pose.dz_mm])
        displacements[:, frame, :] = (rotated - coords) + translation

    np.savetxt(directory / "coords.csv", coords, delimiter=",", fmt="%.10f")
    np.savetxt(directory / "connect.csv", connect, delimiter=",", fmt="%d")
    np.savetxt(directory / "uvs.csv", uvs, delimiter=",", fmt="%.10f")
    np.savetxt(directory / "field_disp_x.csv", displacements[:, :, 0],
               delimiter=",", fmt="%.10f")
    np.savetxt(directory / "field_disp_y.csv", displacements[:, :, 1],
               delimiter=",", fmt="%.10f")
    np.savetxt(directory / "field_disp_z.csv", displacements[:, :, 2],
               delimiter=",", fmt="%.10f")


# ================================================================================
# Scene helpers
# ================================================================================

# TODO: Fix this bug in simdata_csv_to_rtmesh()
def sync_visual_surface(rtmesh: RTMesh) -> None:
    """
    Rebuilds the PyVista surface used by SceneVisualiser from the transformed
    nodal coordinates.

    simdata_csv_to_rtmesh() orients the mesh before create_rtmesh() triangulates
    it. Until then pyvista_surface.points aliases node_coords, so fit_size() and
    place_at() transform it twice. Rendering uses node_coords_over_time and is
    unaffected, but SceneVisualiser would draw the mesh at the wrong scale and
    position.
    """
    if rtmesh.tri_node_mapping is not None:
        rtmesh.pyvista_surface.points = rtmesh.node_coords[rtmesh.tri_node_mapping]
    else:
        rtmesh.pyvista_surface.points = rtmesh.node_coords


def configure_case_media(test_case: TestCaseApp, pipe: RTMesh, water: RTMesh) -> None:
    """Sets the refractive surfaces for the requested optical case."""
    if test_case == TestCaseApp.PIPE:
        pipe.set_surface(SurfType.FIELD_COLOR,
                         material_type=MaterialType.REFRACTIVE,
                         material=MaterialPresets.PLASTIC_ACRYLIC)
    elif test_case == TestCaseApp.WATER:
        pipe.set_surface(SurfType.FIELD_COLOR,
                         material_type=MaterialType.REFRACTIVE,
                         material=MaterialPresets.PLASTIC_ACRYLIC,
                         priority=1)
        water.set_surface(SurfType.FIELD_COLOR,
                          material_type=MaterialType.REFRACTIVE,
                          material=MaterialPresets.WATER,
                          priority=0)


def add_case_media(scene: Scene, test_case: TestCaseApp,
                   pipe: RTMesh, water: RTMesh) -> None:
    """Adds the already configured pipe/water meshes for the requested case."""
    if test_case == TestCaseApp.PIPE:
        scene.add_rtmesh(pipe)
    elif test_case == TestCaseApp.WATER:
        scene.add_rtmesh(pipe)
        scene.add_rtmesh(water)


def build_stereo_cameras(image_width: int,
                         image_height: int,
                         roi_center: np.ndarray,
                         cam0_center: np.ndarray,
                         angle_vertical_view: float,
                         stereo_angle_deg: float) -> tuple[Camera, Camera]:
    """
    Builds a stereo pair. Camera 0 is the nominal camera; camera 1 is obtained
    by rotating camera 0 about the ROI around the world Y axis.
    """
    stereo_angle = np.deg2rad(stereo_angle_deg)
    rotation_y = np.array([
        [ np.cos(stereo_angle), 0.0, np.sin(stereo_angle)],
        [ 0.0,                  1.0, 0.0                ],
        [-np.sin(stereo_angle), 0.0, np.cos(stereo_angle)],
    ])

    cam1_center = roi_center + rotation_y @ (cam0_center - roi_center)

    cam0 = Camera(image_width, image_height, cam0_center, roi_center,
                  angle_vertical_view)
    cam1 = Camera(image_width, image_height, cam1_center, roi_center,
                  angle_vertical_view)
    return cam0, cam1


def export_stereo_geometry(filepath: Path,
                           cam0: Camera,
                           cam1: Camera,
                           roi_center: np.ndarray) -> None:
    """
    Writes a small key,value CSV with the nominal stereo rig geometry. This is
    bookkeeping only: the calibration images are the intended source of the
    camera intrinsics and relative pose.
    """
    c0 = np.asarray(cam0.camera_center, dtype=float)
    c1 = np.asarray(cam1.camera_center, dtype=float)
    baseline = c1 - c0

    v0 = roi_center - c0
    v1 = roi_center - c1
    stereo_angle_deg = np.degrees(np.arccos(np.clip(
        np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)),
        -1.0, 1.0)))

    rows = [
        ("roi_cent_x_mm", roi_center[0]),
        ("roi_cent_y_mm", roi_center[1]),
        ("roi_cent_z_mm", roi_center[2]),
        ("cam0_x_mm", c0[0]),
        ("cam0_y_mm", c0[1]),
        ("cam0_z_mm", c0[2]),
        ("cam1_x_mm", c1[0]),
        ("cam1_y_mm", c1[1]),
        ("cam1_z_mm", c1[2]),
        ("baseline_x_mm", baseline[0]),
        ("baseline_y_mm", baseline[1]),
        ("baseline_z_mm", baseline[2]),
        ("baseline_len_mm", np.linalg.norm(baseline)),
        ("stereo_angle_deg", stereo_angle_deg),
    ]

    with open(filepath, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


# ================================================================================
# Rendering
# ================================================================================

def render_stereo_calibration(test_case: TestCaseApp,
                              cam0: Camera,
                              cam1: Camera,
                              roi_center: np.ndarray,
                              cal_texture: np.ndarray,
                              fov_width_mm: float,
                              fov_height_mm: float,
                              max_target_length_mm: float | None,
                              render_height: int,
                              render_width: int,
                              anti_alias: int,
                              image_format: ImageFormat,
                              pipe: RTMesh,
                              water: RTMesh,
                              through_media: bool = True) -> None:
    """
    Renders the calibration target from both cameras at every calibration
    pose. The target plane is placed on the specimen front face at the ROI and
    the same pipe/water media are optionally included.
    """
    case_dir = test_dir(BASE_TEST_DIR, f"{CASE_BASE_DIR}/{test_case.value}")
    cal_data_dir = test_dir(case_dir, CALPLATE_DIRNAME)
    out_dir = test_dir(case_dir, STEREOCAL_DIRNAME)

    poses = calibration_poses()
    length_mm, height_mm = calibration_target_size(
        fov_width_mm, fov_height_mm, max_target_length_mm)
    write_calplate_simdata(cal_data_dir, length_mm, height_mm, poses)
    print(f"Calibration target: {length_mm:.2f} x {height_mm:.2f} mm, "
          f"{len(poses)} poses")

    cal = simdata_csv_to_rtmesh(cal_data_dir, EDim.TWOD,
                                world_position=roi_center)
    sync_visual_surface(cal)
    cal.set_surface(SurfType.TEXTURE, surface_fill=cal_texture,
                    material_type=MaterialType.UNLIT)

    scene = Scene()
    if through_media:
        add_case_media(scene, test_case, pipe, water)
    scene.add_rtmesh(cal)
    scene.add_camera(cam0)
    scene.add_camera(cam1)
    scene.set_background(np.zeros(3)) # Black background for the dot target

    # SceneVisualiser([cal, pipe, water], cam0)
    # SceneVisualiser([cal, pipe], cam0)

    render_scene(render_height, render_width, scene, anti_alias, out_dir,
                 RenderType.DYNAMIC,
                 texture_sampler=TextureSampler.CATMULL_ROM,
                 shading_type=ShadingType.FLAT,
                 image_format=image_format,
                 omp_thread_count=None)


def render_stereo_dic(scene: Scene,
                      out_dir: Path,
                      render_height: int,
                      render_width: int,
                      anti_alias: int,
                      displaced_frame_idx: int,
                      image_format: ImageFormat) -> None:
    """
    Renders the undeformed reference and the deformed state from both
    cameras.

    Static renders always write rtimage_0_cam* because the scene is clipped to
    a single frame before it reaches the renderer. The deformed render is
    therefore written to a temporary directory and moved to rtimage_1_cam*.
    """
    reference_scene = deepcopy(scene)
    # render_scene(render_height, render_width, reference_scene, anti_alias,
    #              out_dir, RenderType.STATIC, frames_to_render=0,
    #              texture_sampler=TextureSampler.CATMULL_ROM,
    #              shading_type=ShadingType.FLAT,
    #              image_format=image_format,
    #              omp_thread_count=None)

    deformed_dir = test_dir(out_dir, "_deformed")
    deformed_scene = deepcopy(scene)
    # render_scene(render_height, render_width, deformed_scene, anti_alias,
    #              deformed_dir, RenderType.STATIC,
    #              frames_to_render=displaced_frame_idx,
    #              texture_sampler=TextureSampler.CATMULL_ROM,
    #              shading_type=ShadingType.FLAT,
    #              image_format=image_format,
    #              omp_thread_count=None)

    for cam_idx in range(2):
        source = deformed_dir / f"rtimage_0_cam{cam_idx}.bmp"
        target = out_dir / f"rtimage_1_cam{cam_idx}.bmp"
        source.replace(target)
    deformed_dir.rmdir()


# ================================================================================
# Specimen scene and frame selection
# ================================================================================

def plate_stereo_test(test_case: TestCaseApp,
                      aa_subsamples: int = 1,
                      render_cal: bool = False,
                      render_dic: bool = False,
                      crop_px: bool = False,
                      cal_through_media: bool = True,
                      stereo_angle_deg: float = STEREO_ANGLE_DEG) -> None:
    # 1. Paths and access to all data used in the scene
    # Object = main mesh that moves
    object_access = "thesis-data/plate_hole/platehole3d_2mr_63f"
    simdata_path = full_path(object_access)
    ref_texture = full_path("thesis-data/texture/speckle.tiff")
    object_texture = ImageTools.load_image_greyscale(ref_texture)
    cal_texture = ImageTools.load_image_greyscale(full_path(CAL_TARGET_ACCESS))
    # Pipe - we borrow data from application_1_rbm because it is shorter
    pipe_access = "thesis-data/pipe_plate"
    pipe_path = get_tank_path(pipe_access, Elements.TRI6)
    water_path = get_fill_path(pipe_access, Elements.TRI6)

    # 2. Set up the meshes
    object = simdata_csv_to_rtmesh(
        simdata_path, EDim.THREED,
        target_size=PLATE_HEIGHT_MM, size_axis=Axis.Y
    ) # Physical plate size is (25, 35, 1) mm in x,y,z spans
    # The SimData CSVs are stored in metres, while the pipe meshes and the
    # camera setup use millimetres. target_size scales the base coordinates to
    # the physical plate height, but the displacement frames are appended
    # afterwards, so rescale them by the same factor. This puts the plate and
    # the camera in the pipe's world units, with the camera outside the pipe as
    # in application_2_fea.py.
    base_position = object.node_coords_over_time[0]
    object.node_coords_over_time[1:] = (
        base_position
        + (object.node_coords_over_time[1:] - base_position) * object.scale
    )
    # Position the object so it touches the bottom of the pipe
    pipe_bottom_inner_y = - 23 + 4 # Position of the inner edge of the pipe tank, based on the geometry
    object.place_at(np.array([0.0, pipe_bottom_inner_y, 0.0]), anchor=Anchor.BASE)
    sync_visual_surface(object)
    pipe = any_mesh_to_rtmesh(pipe_path)
    water = any_mesh_to_rtmesh(water_path)
    configure_case_media(test_case, pipe, water)

    # 3. Camera and output settings
    output_format = ImageFormat(OutputFormat.IMG_BMP_8BIT, BitDepth.BIT_8,
                                ChannelCount.MONO, True)
    anti_alias = aa_subsamples
    print(f"Anti-aliasing subsamples: {anti_alias}")
    focal_length = 50 # mm
    sensor_height_mm = sensor_height_phs6
    # Derived camera parameters
    fov_height = object.get_size()[1] + 5 # See the entire height of the plate + some extra to get the edges
    camera_distance = camera_working_distance(focal_length, fov_height, sensor_height_mm)
    camera_y_position = pipe_bottom_inner_y + object.get_size()[1]/2 # Lower the y-position of the camera to match that of the plate
    print(f"Camera distance: {camera_distance:.2f} mm, "
          f"pipe outer radius: {0.5*pipe.get_size()[0]:.2f} mm")

    if not crop_px:
        camera_x = 0.0
        render_width = image_width_phs6
        render_height = image_height_phs6
    else:
        if test_case == TestCaseApp.WATER:
            crop_vertical = 20
            crop_horizontal = 20
        else:
            crop_vertical = 40
            crop_horizontal = 160
        # y offset: shift by n px down * 0.0390625 (px/mm scale)
        camera_y_offset = crop_vertical * 0.0390625
        # x offset: shift m px right * 0.0390625 (px/mm scale)
        camera_x = crop_horizontal * 0.0390625
        # Texture and everything else remain scaled exactly the same way. Win-win
        camera_y_position = camera_y_position - camera_y_offset # Lower the y-position of the camera to match that of the plate
        render_width = image_width_phs6 - 2 * crop_horizontal
        render_height = image_height_phs6 - 2 * crop_vertical

    angle_vertical_view = vertical_fov_from_sensor(sensor_height=sensor_height_mm,
                                                   focal_length=focal_length)
    mm_per_px = spatial_scale(fov_height, image_height_phs6)

    # The ROI is the centre of the specimen front face (the plate is 1 mm
    # thick and centred on z = 0, so its camera-facing face is at z = +0.5).
    roi_center = np.array([camera_x, camera_y_position, 0.5])
    cam0_center = np.array([camera_x, camera_y_position, camera_distance])
    cam0, cam1 = build_stereo_cameras(image_width_phs6, image_height_phs6,
                                      roi_center, cam0_center,
                                      angle_vertical_view, stereo_angle_deg)

    # Keep the calibration target inside the pipe inner wall. The fill mesh
    # gives the inner radius; the pipe axis is x = z = 0. The target also
    # translates by up to CAL_TRANS_MM in each pose, so that excursion and a
    # small clearance are removed from the available radius. For the air case
    # there is no pipe, so only the FOV limits the target size.
    max_cal_target_length_mm = None
    if test_case != TestCaseApp.AIR_DIFFUSE:
        water_size = water.get_size()
        pipe_inner_radius = 0.5 * min(water_size[0], water_size[2])
        radial_clearance = np.sqrt(max(pipe_inner_radius**2 - roi_center[2]**2, 0.0))
        radial_clearance -= abs(roi_center[0]) + CAL_TRANS_MM + CAL_PIPE_CLEARANCE_MM
        max_cal_target_length_mm = 2.0 * max(radial_clearance, 0.1)

    # 4. Texture and speckle pattern information for the plate
    # The loaded texture is 2464 x 2056 px (5MPx), 8-bit .tiff; speckles sampled by 5 pixels
    object.set_surface(SurfType.TEXTURE, surface_fill=object_texture,
                       material_type=MaterialType.DIFFUSE)
    uv_scale = speckle_scaling(image_width_phs6, image_height_phs6, 2464, 2056, 5, 5)
    object.uvs *= uv_scale

    # 5. Pick frames to render
    # We only want 2:
    # 1. Undeformed, so at t=0
    # 2. Deformed, somewhere s.t., displacement is <= 1 px
    scale = spatial_scale(fov_height, image_height_phs6) # mm/px, so 1 px = this in mm; 0.0390625 in this case
    print(f"Spatial scale: {scale}")
    node_coords_over_time = object.node_coords_over_time
    node_displacements = np.zeros(node_coords_over_time.shape) # Array for nodal displacements over time
    displaced_frame_idx = 0
    max_displacement_mm = 0
    for t in range(1, object.timestep_count):
        node_displacements[t] = node_coords_over_time[t] - node_coords_over_time[0]
        curr_max_disp_mm = np.max(node_displacements[t])
        if scale >= curr_max_disp_mm > max_displacement_mm:
            max_displacement_mm = curr_max_disp_mm
            displaced_frame_idx = t
    print(f"Displaced frame idx: {displaced_frame_idx}, with maximum displacement of {max_displacement_mm} mm, which is less than the scale 1 px = {scale} mm")

    # If max displacement is below 0.5 px, we scale it to 1.0 px
    if max_displacement_mm <= 0.5 * scale:
        target_displ_px = 1.0 # Target displacement in px
        target_mm = scale * target_displ_px
        factor = target_mm / max_displacement_mm
        print(f"Scaling factor = {factor}")
        disp = node_displacements[displaced_frame_idx]
        scaled_disp = disp * factor
        node_displacements[displaced_frame_idx] = scaled_disp
        object.node_coords_over_time[displaced_frame_idx] = (
            object.node_coords_over_time[0] + scaled_disp)
        # Recompute displacement to verify appropriate scaling
        disp_check = object.node_coords_over_time[displaced_frame_idx] - object.node_coords_over_time[0]
        max_disp_mm_check = np.max(np.linalg.norm(disp_check, axis=-1))
        max_disp_px_check = max_disp_mm_check / scale
        print(f"Scaled max displacement: {max_disp_mm_check} mm, {max_disp_px_check} px")

    case_dir = test_dir(BASE_TEST_DIR, f"{CASE_BASE_DIR}/{test_case.value}")

    # 6. Calibration images
    if render_cal:
        render_stereo_calibration(
            test_case=test_case,
            cam0=cam0,
            cam1=cam1,
            roi_center=roi_center,
            cal_texture=cal_texture,
            fov_width_mm=render_width * mm_per_px,
            fov_height_mm=render_height * mm_per_px,
            max_target_length_mm=max_cal_target_length_mm,
            render_height=render_height,
            render_width=render_width,
            anti_alias=anti_alias,
            image_format=output_format,
            pipe=pipe,
            water=water,
            through_media=cal_through_media,
        )

    # 7. Stereo DIC reference/deformed images
    if render_dic:
        scene = Scene()
        add_case_media(scene, test_case, pipe, water)
        scene.add_rtmesh(object)
        scene.add_camera(cam0)
        scene.add_camera(cam1)

        # SceneVisualiser([object, water, pipe], cam0)
        # SceneVisualiser([water, pipe], cam0)

        out_dir = test_dir(case_dir, STEREODIC_DIRNAME)
        render_stereo_dic(scene, out_dir, render_height, render_width,
                          anti_alias, displaced_frame_idx, output_format)

    # 8. Nominal rig geometry (bookkeeping for downstream checks)
    export_stereo_geometry(case_dir / "stereo_geometry.csv",
                           cam0, cam1, roi_center)


if __name__ == "__main__":
    # Edit this tuple to render only the cases you need. Rendering the water
    # case is the slowest because of the additional refractive interface.
    #cases = (TestCaseApp.AIR_DIFFUSE, TestCaseApp.PIPE, TestCaseApp.WATER)
    cases = (TestCaseApp.WATER, TestCaseApp.PIPE)
    for case in cases:
        plate_stereo_test(
            case,
            aa_subsamples=2**2,
            render_cal=True,
            render_dic=True,
            crop_px=True,
        )
