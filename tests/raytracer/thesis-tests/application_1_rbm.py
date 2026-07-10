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

def rmb_test(test_case: TestCaseApp, aa_samples: int = 1, fallback:bool = False):
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

    #SceneVisualiser([object, pipe]) # Helper display

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
    scene.add_camera(cam)
    render_scene(image_height, image_width, scene, anti_alias, target_path, RenderType.DYNAMIC, texture_sampler = TextureSampler.CATMULL_ROM, shading_type = ShadingType.FLAT, image_format = output_format, omp_thread_count = None)


#rmb_test(TestCaseApp.AIR_DIFFUSE, aa_samples=2**12, fallback=False)

# ================================================================================
# DIC
# ================================================================================
ROI_FILENAME = "roi.dat"
DIC_RESULTS_PREFIX = "dic_results_"
# These params ran succesfully in Zeiss Correlate
SUBSET_SIZE = 7
STEP_SIZE = 4

def run_dic_rmb(test_case: TestCaseApp, save_plot: bool = True, convert_to_mm: bool = False):
    """
    Runs DIC on the experimental images.
    """
    import pyvale.dic as dic
    # Unscaled max displacement: 9.999999999621423e-06 mm, which is less than the scale 1 px = 0.0390625 mm
    #Scaled max displacement: 0.0390625 mm, 1.0 px
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
                        seed=[123, 99],
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
        # Data for this deformation image
        horizontal_displacement = dicdata.u[i]
        vertical_displacement = dicdata.v[i]
        unit = "[px]"
        figure_filename = f"{test_case.value}_rmb_dic_plot_px_{i}.png"
        if convert_to_mm:
            horizontal_displacement /= SCALE_PX_MM
            vertical_displacement /= SCALE_PX_MM
            unit = "[mm]"
            figure_filename = f"{test_case.value}_rmb_dic_plot_mm_{i}.png"

        # Plot data
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        axes = axes.flatten()
        cmap = "magma"

        # First deformation image
        im1 = axes[0].pcolor(dicdata.ss_x, dicdata.ss_y, horizontal_displacement, cmap=cmap)
        im2 = axes[1].pcolor(dicdata.ss_x, dicdata.ss_y, vertical_displacement, cmap=cmap)

        # Titles
        fig.suptitle(f"2D DIC results for frame {i}\nTest case: {test_case.value}", fontsize=FONT_SIZES["suptitle"])
        axes[0].set_title(f"$u_x$ {unit}", fontsize=FONT_SIZES["subtitle"]) # Horizontal displacement
        axes[1].set_title(f"$u_y$ {unit}", fontsize=FONT_SIZES["subtitle"]) # Vertical displacement

        for aa in axes:
            aa.set_aspect('equal')
            aa.invert_yaxis() # Flip upside down because duck points the wrong way

        # Colorbars
        fig.colorbar(im1, ax=axes[0])
        fig.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        #plt.show()
        if save_plot:
            fig.savefig(target_path  / figure_filename, dpi=300, bbox_inches="tight")
        
#run_dic_rmb(TestCaseApp.AIR_DIFFUSE, True, False)