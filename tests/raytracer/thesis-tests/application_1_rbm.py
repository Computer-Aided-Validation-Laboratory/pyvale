"""
Application test 1: Rigid body motion (RBM)
Target: 0.1 px displacement per frame over 10 images
Images: 8-bit BMP, as small as possible (not targeting a particular camera, so what renders fastest)
Cases: AIR_DIFFUSE, PIPE, WATER
Do DIC on these
"""
from enum import StrEnum
import numpy as np
from pathlib import Path
from global_utils import *
from convergence_common import *

from pyvale.sensorsim.imagetools import ImageTools
from pyvale.dataset.dataset import dic_pattern_5mpx_path
import os
from pyvale.raytracer.rtmesh import *
from pyvale.raytracer.rtmeshvisuals import *
from pyvale.raytracer.rtcamera import *
from pyvale.raytracer.rtscene import *
from pyvale.raytracer.rtpresets import *
from pyvale.raytracer.rtmain import *
from pyvale.raytracer.rtoutputformat import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
# No Blender imports => Should work on Linux

# ================================================================================
# Turn SVG into a mesh
# "Borrowed" and adapted from https://github.com/Computer-Aided-Validation-Laboratory/riley-raster/blob/main/data/rabbits/main_meshbunnies.py
# ================================================================================
import os
from svgpathtools import svg2paths
import meshio
from matplotlib.collections import PolyCollection
import gmsh

TARGET_LENGTH = 1.0
EDGE_FRACTION = 0.07
MIN_SEGMENT_LENGTH = 0.005
STEPS_ON_PATH = 500
TRI_FACTOR = 0.6
QUAD_FACTOR = 0.5
QUAD_ALGORITHM = 8
QUAD_HIGH_ORDER_OPTIMIZE = 2

pet_name = "duck"
pet_svg_path = full_path(f"thesis-data/app1_rbm/{pet_name}.svg")
target_dir = os.path.dirname(pet_svg_path)

def load_outline_path(svg_path):
    paths, _ = svg2paths(svg_path)
    return paths[0]


def get_path_bbox(path):
    sample_points = np.array([path.point(tt / 500) for tt in range(501)])
    min_x = np.min(sample_points.real)
    max_x = np.max(sample_points.real)
    min_y = np.min(sample_points.imag)
    max_y = np.max(sample_points.imag)
    return min_x, max_x, min_y, max_y


def build_simplified_points(path, target_length):
    min_x, max_x, min_y, _ = get_path_bbox(path)
    scale = target_length / (max_x - min_x)

    def transform(point):
        x_coord = (point.real - min_x) * scale
        y_coord = -(point.imag - min_y) * scale
        return x_coord, y_coord

    simplified_points = []
    last_point = None

    for step_index in range(STEPS_ON_PATH):
        path_param = step_index / STEPS_ON_PATH
        current_point = path.point(path_param)
        transformed_point = transform(current_point)

        if last_point is None:
            simplified_points.append(transformed_point)
            last_point = transformed_point
            continue

        distance = np.sqrt(
            (transformed_point[0] - last_point[0]) ** 2
            + (transformed_point[1] - last_point[1]) ** 2
        )
        if distance > MIN_SEGMENT_LENGTH:
            simplified_points.append(transformed_point)
            last_point = transformed_point

    return flip_points_horizontal(simplified_points)


def flip_points_horizontal(points):
    x_coords = [point[0] for point in points]
    min_x = min(x_coords)
    max_x = max(x_coords)

    flipped_points = []
    for x_coord, y_coord in points:
        flipped_x_coord = max_x + min_x - x_coord
        flipped_points.append((flipped_x_coord, y_coord))

    return flipped_points


def is_tri(element):
    return element.label in ["TRI3", "TRI6"]


def is_quad(element):
    return element.label in ["QUAD4", "QUAD8", "QUAD9"]


def get_mesh_size(target_length, edge_fraction, element):
    mesh_size = target_length * edge_fraction
    if is_tri(element):
        return mesh_size * TRI_FACTOR
    if is_quad(element):
        return mesh_size * QUAD_FACTOR
    return mesh_size

def get_element_order(element):
    if element.label in ["TRI6", "QUAD8", "QUAD9"]:
        return 2
    return 1

def get_num_corners(element):
    if is_tri(element):
        return 3
    return 4

def get_meshio_cell_type(element):
    mapping = {
        "TRI3": "triangle",
        "TRI6": "triangle6",
        "QUAD4": "quad",
        "QUAD8": "quad8",
        "QUAD9": "quad9"}
    return mapping[element.label]


def build_polygons(coords, connectivity, element):
    num_corners = get_num_corners(element)
    polygons = []
    corner_indices = set()

    for element_nodes in connectivity:
        polygon_nodes = [coords[node_index][:2]
            for node_index in element_nodes[:num_corners]]
        polygons.append(polygon_nodes)
        for node_index in element_nodes[:num_corners]:
            corner_indices.add(node_index)

    return polygons, corner_indices


def draw_mesh(ax, coords, connectivity, element, title=None):
    polygons, corner_indices = build_polygons(coords, connectivity, element)
    collection = PolyCollection(
        polygons,
        facecolors="none",
        edgecolors="black",
        linewidths=0.8)
    ax.add_collection(collection)

    all_indices = np.arange(len(coords))
    is_corner = np.array([node_index in corner_indices for node_index in all_indices])

    ax.scatter(
        coords[is_corner, 0],
        coords[is_corner, 1],
        s=16,
        c="limegreen",
        marker="o",
        edgecolors="black",
        linewidths=0.6,
        zorder=4)

    if not np.all(is_corner):
        ax.scatter(
            coords[~is_corner, 0],
            coords[~is_corner, 1],
            s=8,
            c="white",
            marker="o",
            edgecolors="black",
            linewidths=0.3,
            zorder=3)

    ax.set_aspect("equal")
    ax.axis("off")
    if title is not None:
        ax.set_title(title)

def export_vtk(out_dir, stem, element, coords, connectivity):
    vtk_path = os.path.join(out_dir, f"{stem}_{element.label}.vtk")

    points = np.asarray(coords, dtype=float)
    cells = [(get_meshio_cell_type(element),np.asarray(connectivity, dtype=int))]

    mesh = meshio.Mesh(points=points, cells=cells)
    meshio.write(vtk_path, mesh)

    return vtk_path

def extract_mesh_data():
    elem_types, _, elem_node_tags = gmsh.model.mesh.getElements(2)

    all_connectivity = []
    used_node_tags = set()

    for type_index, gmsh_elem_type in enumerate(elem_types):
        _, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(gmsh_elem_type)
        element_nodes = np.asarray(elem_node_tags[type_index]).reshape((-1, num_nodes))
        for element_node_tags_row in element_nodes:
            all_connectivity.append(element_node_tags_row.tolist())
            used_node_tags.update(element_node_tags_row.tolist())

    sorted_node_tags = sorted(used_node_tags)
    node_map = {node_tag: node_index
        for node_index, node_tag in enumerate(sorted_node_tags)}

    coords_by_tag = {}
    for node_tag in sorted_node_tags:
        coords, _, _, _ = gmsh.model.mesh.getNode(node_tag)
        coords_by_tag[node_tag] = coords

    final_coords = np.array([coords_by_tag[node_tag] for node_tag in sorted_node_tags], dtype=float)

    final_connectivity = [
        [node_map[node_tag] for node_tag in element_node_tags_row]
        for element_node_tags_row in all_connectivity]

    return {"coords": final_coords, "connectivity": final_connectivity}


def mesh_svg(svg_path, element, target_length=TARGET_LENGTH, edge_fraction=EDGE_FRACTION):
    gmsh.initialize()
    stem = os.path.splitext(os.path.basename(svg_path))[0]
    gmsh.model.add(f"{stem}_{element.label}")

    try:
        outline_path = load_outline_path(svg_path)
        simplified_points = build_simplified_points(outline_path, target_length)

        mesh_size = get_mesh_size(target_length, edge_fraction, element)

        gmsh_point_tags = []
        for point in simplified_points:
            gmsh_point_tags.append(gmsh.model.geo.addPoint(point[0], point[1], 0.0, mesh_size))

        gmsh_point_tags.append(gmsh_point_tags[0])
        spline_tag = gmsh.model.geo.addSpline(gmsh_point_tags)
        loop_tag = gmsh.model.geo.addCurveLoop([spline_tag])
        surface_tag = gmsh.model.geo.addPlaneSurface([loop_tag])

        gmsh.model.geo.synchronize()

        if is_quad(element):
            gmsh.model.mesh.setRecombine(2, surface_tag)
            gmsh.option.setNumber("Mesh.Algorithm", QUAD_ALGORITHM)
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
        else:
            gmsh.option.setNumber("Mesh.Algorithm", 1)

        if element.label == "QUAD8":
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
        elif element.label == "QUAD9":
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)

        gmsh.model.mesh.generate(2)

        if is_quad(element):
            gmsh.model.mesh.optimize("Relocate2D")

        order = get_element_order(element)
        gmsh.model.mesh.setOrder(order)

        if is_quad(element) and order == 2:
            gmsh.option.setNumber(
                "Mesh.HighOrderOptimize",
                QUAD_HIGH_ORDER_OPTIMIZE,
            )
            gmsh.model.mesh.optimize("HighOrder")

        out_dir = os.path.join(target_dir, f"{stem}_{element.label}")
        os.makedirs(out_dir, exist_ok=True)

        mesh_data = extract_mesh_data()
        vtk_path = export_vtk(
            out_dir,
            stem,
            element,
            mesh_data["coords"],
            mesh_data["connectivity"])

        print(f"Exported VTK mesh to {vtk_path}")
        return {"coords": mesh_data["coords"],
            "connectivity": mesh_data["connectivity"],
            "vtk_path": vtk_path}

    finally:
        gmsh.finalize()

# Turn SVG into mesh
#mesh_svg(pet_svg_path, Elements.TRI3)


# Unwrap the mesh to avoid Blender dependency in the main test
def uv_unwrap(fallback=False):
    from pyvale.raytracer.rtblender import BlenderUnwrapper
    blender_uv = BlenderUnwrapper()
    object_path = full_path(f"thesis-data/app1_rbm/{pet_name}_TRI3/{pet_name}_TRI3.vtk")
    save_path = full_path(f"thesis-data/app1_rbm/{pet_name}_TRI3/{pet_name}_TRI3_uvs.csv")
    if fallback:
         object_path = full_path("thesis-data/app1_rbm/cube_QUAD9/hex27_cube.vtk")
         save_path = full_path("thesis-data/app1_rbm/cube_QUAD9/cube_QUAD9_uvs.csv")
    object = any_mesh_to_rtmesh(object_path, world_position = np.array([0.0, 0.0, 0.0]), anchor = Anchor.CENTER,
                                target_size=50, size_axis = Axis.Y) # Shark 50 mm long => 30 mm wide
    blender_uv.add_rtmesh(object)
    blender_uv.smart_unwrap()
    object.export_uvs(save_path)
   
#uv_unwrap(True)

# ================================================================================
# Rigid body motion image render
# ================================================================================
#calplate_dict_names = ["quad4_calplate3d", "quad8_calplate3d", "quad9_calplate3d", "tri3_calplate3d", "tri6_calplate3d"]

def rmb_test(test_case: TestCaseApp, aa_samples: int = 1, fallback:bool = False, crop_px: bool = False, frame_idx: int | None = None):
    # 1. Camera and output settings
    pet_height = 51 # mm
    pet_displacement = 2 # mm; we don't really use it in practice, but it is to mock some tiny experimental ROI
    if fallback:
        pet_height = 29 # now it will be sphere diameter
    # Swap height and width ("rotated" camera) as our ROI is more vertical than horizontal
    # Then scale down by 10 - we still see something, but the image is small
    image_width = int(image_height_cx5 / 10)
    image_height = int(image_width_cx5 / 10)
    pixel_pitch = pixel_pitch_cx5
    focal_length = 50 # mm
    # Derived camera parameters
    active_sensor_side_length = active_sensor_height(image_height, pixel_pitch)
    angle_vertical_view = vertical_fov_from_sensor(sensor_height=active_sensor_side_length, focal_length=focal_length)
    #angle_vertical_view = 1 # For rendering images further away (sanity checks)
    fov_height = pet_height + pet_displacement + 1 # We want to see the shark's motion + have some safety of 1 mm
    #camera_distance = camera_working_distance(focal_length, fov_height, active_sensor_side_length) / 50 # Slightly nicer distance for checking the dielectric overlap in SceneVisualiser
    camera_distance = camera_working_distance(focal_length, fov_height, active_sensor_side_length)
    # Camera positioning
    target_distance = camera_distance - focal_length
    camera_target = np.array([0, 0, target_distance])
    camera_center = np.array([0, 0, camera_distance])
    cam = Camera(image_width, image_height, camera_center, camera_target, angle_vertical_view)
    # Output directory for the renders
    base_data_dir = f"app1_rmb/{test_case.value}"
    if fallback:
        base_data_dir = f"app1_rmb/fallback/{test_case.value}"
    target_path = test_dir(BASE_TEST_DIR, base_data_dir)
    # Output format
    output_format = output_format_cx5
    # Anti-aliasing
    anti_alias = aa_samples; # for anti-aliasing
    print(f"VFOV angle: {angle_vertical_view} with camera distance: {camera_distance}")

    #cam.print_view_dims()

    # 2. Paths and access to all data used in the scene
    # Object = main mesh that moves
    object_path = full_path(f"thesis-data/app1_rbm/{pet_name}_TRI3/{pet_name}_TRI3.vtk")
    if fallback:
        object_path = full_path(f"thesis-data/app1_rbm/cube_QUAD9/hex27_cube.vtk")
    # Pipe and water - as in convergence_rt, but shorter and wider to reduce unnecessary computations
    pipe_access = "thesis-data/pipe_shark"
    pipe_path = get_tank_path(pipe_access, Elements.TRI6) # TRI3 or TRI6 only for pipe
    water_path = get_fill_path(pipe_access, Elements.TRI6)

    # 3. Set up the meshes
    scene = Scene()
    obj_size = 29
    if fallback:
        obj_size=20
    object = any_mesh_to_rtmesh(object_path, world_position = np.array([0.0, 0.0, 0.0]), anchor = Anchor.CENTER, 
                                target_size=obj_size, size_axis = Axis.Y) # Shark 50 mm long => 30 mm wide (hammer); 32 mm (fatshark); pipe is 35 mm ID
    object.rotate(rotation=Rotation.from_euler('z', 90, degrees=True))
    print(object.get_size())
    pipe = any_mesh_to_rtmesh(pipe_path)
    water = any_mesh_to_rtmesh(water_path)

    SceneVisualiser([object, pipe]) # Helper display

    if test_case == TestCaseApp.AIR_DIFFUSE:
        print(f"--------------------------------\nTESTED CASE: AIR DIFFUSE\n--------------------------------")
    elif test_case == TestCaseApp.PIPE:
        print(f"--------------------------------\nTESTED CASE: EMPTY PIPE\n--------------------------------")
        pipe.set_surface(SurfType.FIELD_COLOR, material_type=MaterialType.REFRACTIVE, material=MaterialPresets.PLASTIC_ACRYLIC)
        scene.add_rtmesh(pipe)
    elif test_case == TestCaseApp.WATER:
        print(f"--------------------------------\nTESTED CASE: PIPE WITH WATER\n----------------------------")
        pipe.set_surface(SurfType.FIELD_COLOR, material_type=MaterialType.REFRACTIVE, material=MaterialPresets.PLASTIC_ACRYLIC, priority=1)
        scene.add_rtmesh(pipe)
        water.set_surface(SurfType.FIELD_COLOR, material_type=MaterialType.REFRACTIVE, material=MaterialPresets.WATER, priority=0)
        scene.add_rtmesh(water)

    # 4. Create mock displacement for the object
    frame_count = 10 # How many frames we want to render
    # We need 0.1 px displacement per frame, so get the spatial scale first
    #Total displacement over 10 frames is 0.22131147540983606 mm
    #Scaling: 0.1 px = 0.022131147540983605 mm
    scale = spatial_scale(fov_height, image_height) # mm/px, so 1 px = this in mm
    disp_per_frame = scale / 10 # We get 0.1 px = yyy mm = our delta y per frame
    total_displacement = disp_per_frame * frame_count
    print(f"Total displacement over {frame_count} frames is {total_displacement} mm\n\tScaling: 0.1 px = {disp_per_frame} mm")
    object_nodal_displacements = create_rigid_linear_translation(object.node_count, frame_count, total_displacement, Axis.Y)
    object.add_temporal_displacement(object_nodal_displacements)

    # 5. Texture and speckle pattern information for the shark
    # The loaded texture is 2464 x 2056 px (5MPx), 8-bit .tiff; speckles sampled by 5 pixels
    # This is huge compared to the target resolution, so we need to change it or the speckles will be just noise
    # We could downsample the texture OR, since we're scaling down and the UV's will not go over the [0,1] range
    # simply rescale those without altering the image
    if fallback:
        object.import_uvs(Path.with_name(object_path, f"cube_QUAD9_uvs.csv")) # Load pre-processed UVs
    else:
        object.import_uvs(Path.with_name(object_path, f"{pet_name}_TRI3_uvs.csv"))
    object_texture = ImageTools.load_image_greyscale(dic_pattern_5mpx_path())
    object.set_surface(SurfType.TEXTURE, surface_fill=object_texture, material_type=MaterialType.DIFFUSE)
    # Scale the UVs to get 3.5 px speckles in the rendered images
    uv_scale = speckle_scaling(image_width, image_height, 2464, 2056, 5, 3.5) # Returns [delta_u, delta_v] array
    object.uvs = object.uvs * uv_scale
    scene.add_rtmesh(object)

    # 6. Render
    """
    if crop_px:
        if test_case == TestCaseApp.WATER:
            vertical_crop_top_px = 20 # per side
            vertical_crop_bottom_px = 35 # per side
            horizontal_crop_px = 10 # per side
        else:
            vertical_crop_top_px = 30 # per side
            vertical_crop_bottom_px = 35 # per side
            horizontal_crop_px = 25 # per side
        y_offset = scale * vertical_crop_top_px
        x_offset = scale * horizontal_crop_px
        camera_target = np.array([x_offset, -y_offset, target_distance])
        camera_center = np.array([x_offset, -y_offset, camera_distance])
        cam = Camera(image_width, image_height, camera_center, camera_target, angle_vertical_view)
        image_height = image_height - vertical_crop_top_px - vertical_crop_bottom_px
        image_width = image_width - 2 * horizontal_crop_px
    scene.add_camera(cam)
    if frame_idx is None:
        render_scene(image_height, image_width, scene, anti_alias, target_path, RenderType.DYNAMIC, texture_sampler = TextureSampler.CATMULL_ROM, shading_type = ShadingType.FLAT, image_format = output_format, omp_thread_count = None)
    else:
        target_path = test_dir(BASE_TEST_DIR, base_data_dir + f"/frame_{frame_idx}")
        if frame_idx <= 9:
            render_scene(image_height, image_width, scene, anti_alias, target_path, RenderType.STATIC, frames_to_render=frame_idx, texture_sampler = TextureSampler.CATMULL_ROM, shading_type = ShadingType.FLAT, image_format = output_format, omp_thread_count = None)
        else:
            raise ValueError(f"Wrong frame index: {frame_idx}")
    """

#rmb_test(TestCaseApp.WATER, aa_samples=2**0, fallback=False, crop_px=True, frame_idx = 2)

# ================================================================================
# DIC
# ================================================================================
ROI_FILENAME = "roi.dat"
DIC_RESULTS_PREFIX = "dic_results_"
# These params ran succesfully in Zeiss Correlate
SUBSET_SIZE = 21
STEP_SIZE = 5

def run_dic_rmb(test_case: TestCaseApp, save_plot: bool = True, convert_to_mm: bool = False):
    """
    Runs DIC on the experimental images.
    """
    import pyvale.dic as dic
    SCALE_PX_MM = 0.022131147540983605 # from rmb_test

    # Open the reference image
    base_data_dir = f"app1_rmb/{test_case.value}"
    target_path = test_dir(BASE_TEST_DIR, base_data_dir)
    image_basename = "rtimage_"
    suffix = "_cam0.tiff"
    ref_img_path = target_path / f"{image_basename}0{suffix}"
    ref_img = ImageTools.load_image_greyscale(ref_img_path)
    def_img_count = 10

    # Define ROI
    print(f"target path: {target_path}")
    roi = dic.RegionOfInterest(ref_image=ref_img)
    roi_file = target_path / f"{test_case.value}_{ROI_FILENAME}"
    dic_results_prefix = f"{test_case.value}_{DIC_RESULTS_PREFIX}"
    if not os.path.exists(roi_file):
        # Select and save ROI if file doesn't exist
        roi.interactive_selection(subset_size=SUBSET_SIZE)
        roi.save_array(filename=roi_file,binary=False)
    
    dic_files = target_path / f"{dic_results_prefix}*.csv"
    # The above is a wildcard, so it will not work for the os.path.exists condition below
    dic_filename_check = target_path / f"{dic_results_prefix}def_img_0000.csv"

    if not os.path.exists(dic_filename_check):
        # Run DIC analysis if it doesn't exist 
        roi.read_array(filename=roi_file, binary=False)
        # Go over frames 1-9 (inclusive) and do DIC on them
        def_images = np.ndarray((def_img_count, ref_img.shape[0], ref_img.shape[1]))
        for i in range(1, def_img_count):
            def_img_path = target_path / f"{image_basename}{i}{suffix}"
            def_img = ImageTools.load_image_greyscale(def_img_path)
            def_images[i] = def_img
        dic.calculate_2d(reference=ref_img,
                        deformed=def_images,
                        roi_mask=roi.mask,
                        seed=[123, 99], # Works for both pipe and air
                        subset_size=SUBSET_SIZE,
                        subset_step=STEP_SIZE,
                        shape_function="AFFINE",
                        correlation_criteria="ZNSSD",
                        output_basepath=target_path,
                        output_delimiter=",",
                        output_prefix=dic_results_prefix,
                        max_displacement=2,
                        method="IMAGE_SCAN")
            
    # Plotting
    # Read data
    
    dicdata = dic.import_2d(data=dic_files, delimiter=",", binary=False)

    for i in range(def_img_count):
        horizontal_displacement = dicdata.u[i]
        vertical_displacement = dicdata.v[i]

        unit = "[px]"
        combined_filename = f"{test_case.value}_rmb_dic_plot_px_{i}.png"
        ux_filename = f"{test_case.value}_rmb_dic_plot_ux_px_{i}.png"
        uy_filename = f"{test_case.value}_rmb_dic_plot_uy_px_{i}.png"

        if convert_to_mm:
            horizontal_displacement = horizontal_displacement / SCALE_PX_MM
            vertical_displacement = vertical_displacement / SCALE_PX_MM
            unit = "[mm]"
            combined_filename = f"{test_case.value}_rmb_dic_plot_mm_{i}.png"
            ux_filename = f"{test_case.value}_rmb_dic_plot_ux_mm_{i}.png"
            uy_filename = f"{test_case.value}_rmb_dic_plot_uy_mm_{i}.png"

        cmap = "magma"

        # -----------------------------
        # Combined figure: ux and uy
        # -----------------------------
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig_combined, axes = plt.subplots(1, 2, figsize=(15, 10))
        axes = axes.flatten()

        im1 = axes[0].pcolor(dicdata.ss_x, dicdata.ss_y, horizontal_displacement, cmap=cmap)
        im2 = axes[1].pcolor(dicdata.ss_x, dicdata.ss_y, vertical_displacement, cmap=cmap)

        fig_combined.suptitle(f"2D DIC results for frame {i}\nTest case: {test_case.value}",fontsize=FONT_SIZES["suptitle"])
        axes[0].set_title(f"$u_x$ {unit}", fontsize=FONT_SIZES["subtitle"])
        axes[1].set_title(f"$u_y$ {unit}", fontsize=FONT_SIZES["subtitle"])

        for aa in axes:
            aa.set_aspect("equal")
            aa.invert_yaxis()
            #aa.tick_params(axis="both", labelsize=25)
            aa.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

        # This makes cbars the same height as the figure axes
        divider1 = make_axes_locatable(axes[0])
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        fig_combined.colorbar(im1, cax=cax1)
        divider2 = make_axes_locatable(axes[1])
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        fig_combined.colorbar(im2, cax=cax2)

        fig_combined.tight_layout()

        if save_plot:
            fig_combined.savefig(target_path / combined_filename, dpi=300, bbox_inches="tight")
        plt.close(fig_combined)

        # -----------------------------
        # Separate figure: ux only
        # -----------------------------
        fig_ux, ax_ux = plt.subplots(figsize=(8, 7))

        im_ux = ax_ux.pcolor(dicdata.ss_x, dicdata.ss_y, horizontal_displacement, cmap=cmap)
        #fig_ux.suptitle(f"2D DIC results for frame {i}\nTest case: {test_case.value}", fontsize=FONT_SIZES["suptitle"])
        #ax_ux.set_title(f"$u_x$ {unit}", fontsize=FONT_SIZES["subtitle"])
        ax_ux.set_aspect("equal")
        ax_ux.invert_yaxis()
        #ax_ux.tick_params(axis="both", labelsize=20)
        ax_ux.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        divider3 = make_axes_locatable(ax_ux)
        cax3 = divider3.append_axes("right", size="5%", pad=0.1)
        cbar_x = fig_ux.colorbar(im_ux, cax=cax3)
        cbar_x.set_label(f"$u_x$ {unit}", fontsize=30, labelpad=10)
        # Tick label size (numbers next to the bar)
        cbar_x.ax.tick_params(labelsize=25) 
        


        fig_ux.tight_layout()

        if save_plot:
            fig_ux.savefig(target_path / ux_filename, dpi=300, bbox_inches="tight")
        plt.close(fig_ux)

        # -----------------------------
        # Separate figure: uy only
        # -----------------------------
        fig_uy, ax_uy = plt.subplots(figsize=(8, 7))

        im_uy = ax_uy.pcolor(dicdata.ss_x, dicdata.ss_y, vertical_displacement, cmap=cmap)
        #fig_uy.suptitle(f"2D DIC results for frame {i}\nTest case: {test_case.value}",fontsize=FONT_SIZES["suptitle"])
        #ax_uy.set_title(f"$u_y$ {unit}", fontsize=40)
        ax_uy.set_aspect("equal")
        ax_uy.invert_yaxis()
        #ax_uy.tick_params(axis="both", labelsize=20)
        ax_uy.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        divider4 = make_axes_locatable(ax_uy)
        cax4 = divider4.append_axes("right", size="5%", pad=0.1)
        cbar_y = fig_uy.colorbar(im_uy, cax=cax4)
        cbar_y.set_label(f"$u_y$ {unit}", fontsize=30, labelpad=10)
        # Tick label size (numbers next to the bar)
        cbar_y.ax.tick_params(labelsize=25)

        #cbar_y = fig_uy.colorbar(im_uy, ax=ax_uy, shrink=0.85)
        #cbar_y.set_label(f"$u_y$ {unit}", fontsize=30)
        #cbar_y.ax.tick_params(labelsize=25)

        fig_uy.tight_layout()

        if save_plot:
            fig_uy.savefig(target_path / uy_filename, dpi=300, bbox_inches="tight")
        plt.close(fig_uy)



def compare_converged_dic(save_plot: bool = True, convert_to_mm: bool = False):
    """
    Compare converged and converged_to_lsb DIC results for the air_diffuse case.
    Computes (conv - lsb) for ux and uy, crops mismatched fields to their common
    shape if necessary, and saves combined / ux-only / uy-only heatmaps into:
        app1_rmb/air_diffuse/
    """
    import pyvale.dic as dic

    SCALE_PX_MM = 0.022131147540983605  # from rmb_test

    case_name = "air_diffuse"
    base_data_dir_lsb = "app1_rmb/air_diffuse/converged_to_lsb"
    base_data_dir_conv = "app1_rmb/air_diffuse/fully_conv_1m"
    output_dir = test_dir(BASE_TEST_DIR, "app1_rmb/air_diffuse/lsbvsfull")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_basename = "rtimage_"
    suffix = "_cam0.tiff"
    def_img_count = 10

    def ensure_dic_results(base_data_dir: str):
        target_path = test_dir(BASE_TEST_DIR, base_data_dir)

        ref_img_path = target_path / f"{image_basename}0{suffix}"
        ref_img = ImageTools.load_image_greyscale(ref_img_path)

        print(f"target path: {target_path}")

        roi = dic.RegionOfInterest(ref_image=ref_img)
        roi_file = target_path / f"{case_name}_{ROI_FILENAME}"
        dic_results_prefix = f"{case_name}_{DIC_RESULTS_PREFIX}"

        if not os.path.exists(roi_file):
            roi.interactive_selection(subset_size=SUBSET_SIZE)
            roi.save_array(filename=roi_file, binary=False)

        dic_files = target_path / f"{dic_results_prefix}*.csv"
        dic_filename_check = target_path / f"{dic_results_prefix}def_img_0000.csv"

        if not os.path.exists(dic_filename_check):
            roi.read_array(filename=roi_file, binary=False)

            def_images = np.ndarray((def_img_count, ref_img.shape[0], ref_img.shape[1]))
            def_images[0] = ref_img

            for i in range(1, def_img_count):
                def_img_path = target_path / f"{image_basename}{i}{suffix}"
                def_img = ImageTools.load_image_greyscale(def_img_path)
                def_images[i] = def_img

            dic.calculate_2d(
                reference=ref_img,
                deformed=def_images,
                roi_mask=roi.mask,
                seed=[123, 99],
                subset_size=SUBSET_SIZE,
                subset_step=STEP_SIZE,
                shape_function="AFFINE",
                correlation_criteria="ZNSSD",
                output_basepath=target_path,
                output_delimiter=",",
                output_prefix=dic_results_prefix,
                max_displacement=2,
                method="IMAGE_SCAN",
            )

        return dic.import_2d(data=dic_files, delimiter=",", binary=False)

    def crop_to_common_shape(a, b):
        nrows = min(a.shape[0], b.shape[0])
        ncols = min(a.shape[1], b.shape[1])
        return a[:nrows, :ncols], b[:nrows, :ncols], nrows, ncols

    dicdata_lsb = ensure_dic_results(base_data_dir_lsb)
    dicdata_conv = ensure_dic_results(base_data_dir_conv)

    for i in range(def_img_count):
        ux_conv = dicdata_conv.u[i]
        ux_lsb = dicdata_lsb.u[i]
        uy_conv = dicdata_conv.v[i]
        uy_lsb = dicdata_lsb.v[i]

        ux_conv_crop, ux_lsb_crop, nrows_x, ncols_x = crop_to_common_shape(ux_conv, ux_lsb)
        uy_conv_crop, uy_lsb_crop, nrows_y, ncols_y = crop_to_common_shape(uy_conv, uy_lsb)

        horizontal_displacement = ux_conv_crop - ux_lsb_crop
        vertical_displacement = uy_conv_crop - uy_lsb_crop

        ss_x_ux = dicdata_conv.ss_x[:nrows_x, :ncols_x]
        ss_y_ux = dicdata_conv.ss_y[:nrows_x, :ncols_x]
        ss_x_uy = dicdata_conv.ss_x[:nrows_y, :ncols_y]
        ss_y_uy = dicdata_conv.ss_y[:nrows_y, :ncols_y]

        unit = "[px]"
        combined_filename = f"{case_name}_rmb_dic_plot_px_lsbvsconv_{i}.png"
        ux_filename = f"{case_name}_rmb_dic_plot_ux_px_lsbvsconv_{i}.png"
        uy_filename = f"{case_name}_rmb_dic_plot_uy_px_lsbvsconv_{i}.png"

        if convert_to_mm:
            horizontal_displacement = horizontal_displacement / SCALE_PX_MM
            vertical_displacement = vertical_displacement / SCALE_PX_MM
            unit = "[mm]"
            combined_filename = f"{case_name}_rmb_dic_plot_mm_lsbvsconv_{i}.png"
            ux_filename = f"{case_name}_rmb_dic_plot_ux_mm_lsbvsconv_{i}.png"
            uy_filename = f"{case_name}_rmb_dic_plot_uy_mm_lsbvsconv_{i}.png"

        cmap = "magma"

        # -----------------------------
        # Combined figure: ux and uy
        # -----------------------------
        fig_combined, axes = plt.subplots(1, 2, figsize=(15, 10))
        axes = axes.flatten()

        im1 = axes[0].pcolor(ss_x_ux, ss_y_ux, horizontal_displacement, cmap=cmap)
        im2 = axes[1].pcolor(ss_x_uy, ss_y_uy, vertical_displacement, cmap=cmap)

        fig_combined.suptitle(
            f"2D DIC difference results for frame {i}\nTest case: {case_name} (conv - lsb)",
            fontsize=FONT_SIZES["suptitle"],
        )
        axes[0].set_title(f"$u_x$ {unit}", fontsize=FONT_SIZES["subtitle"])
        axes[1].set_title(f"$u_y$ {unit}", fontsize=FONT_SIZES["subtitle"])

        for aa in axes:
            aa.set_aspect("equal")
            aa.invert_yaxis()
            aa.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )

        divider1 = make_axes_locatable(axes[0])
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        cbar1 = fig_combined.colorbar(im1, cax=cax1)
        cbar1.ax.tick_params(labelsize=20)
        cbar1.ax.yaxis.get_offset_text().set_fontsize(20)

        divider2 = make_axes_locatable(axes[1])
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        cbar2 = fig_combined.colorbar(im2, cax=cax2)
        cbar2.ax.tick_params(labelsize=20)
        cbar2.ax.yaxis.get_offset_text().set_fontsize(20)

        fig_combined.tight_layout()

        if save_plot:
            fig_combined.savefig(output_dir / combined_filename, dpi=300, bbox_inches="tight")
        plt.close(fig_combined)

        # -----------------------------
        # Separate figure: ux only
        # -----------------------------
        fig_ux, ax_ux = plt.subplots(figsize=(8, 7))

        im_ux = ax_ux.pcolor(ss_x_ux, ss_y_ux, horizontal_displacement, cmap=cmap)
        ax_ux.set_aspect("equal")
        ax_ux.invert_yaxis()
        ax_ux.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )

        divider3 = make_axes_locatable(ax_ux)
        cax3 = divider3.append_axes("right", size="5%", pad=0.1)
        cbar_x = fig_ux.colorbar(im_ux, cax=cax3)
        cbar_x.set_label(f"$u_x$ {unit}", fontsize=30, labelpad=10)
        cbar_x.ax.tick_params(labelsize=25)
        cbar_x.ax.yaxis.get_offset_text().set_fontsize(25)

        fig_ux.tight_layout()

        if save_plot:
            fig_ux.savefig(output_dir / ux_filename, dpi=300, bbox_inches="tight")
        plt.close(fig_ux)

        # -----------------------------
        # Separate figure: uy only
        # -----------------------------
        fig_uy, ax_uy = plt.subplots(figsize=(8, 7))

        im_uy = ax_uy.pcolor(ss_x_uy, ss_y_uy, vertical_displacement, cmap=cmap)
        ax_uy.set_aspect("equal")
        ax_uy.invert_yaxis()
        ax_uy.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )

        divider4 = make_axes_locatable(ax_uy)
        cax4 = divider4.append_axes("right", size="5%", pad=0.1)
        cbar_y = fig_uy.colorbar(im_uy, cax=cax4)
        cbar_y.set_label(f"$u_y$ {unit}", fontsize=30, labelpad=10)
        cbar_y.ax.tick_params(labelsize=25)
        cbar_y.ax.yaxis.get_offset_text().set_fontsize(25)

        fig_uy.tight_layout()

        if save_plot:
            print(f"Saving plot to: {output_dir / uy_filename}")
            fig_uy.savefig(output_dir / uy_filename, dpi=300, bbox_inches="tight")
        plt.close(fig_uy)
            
compare_converged_dic(save_plot=True, convert_to_mm=False)         
        
#run_dic_rmb(TestCaseApp.AIR_DIFFUSE, True, False)
#run_dic_rmb(TestCaseApp.PIPE, True, False)
#run_dic_rmb(TestCaseApp.WATER, True, False)