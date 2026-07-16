"""
Application test 2: Mechanical plate with FEA data
Target: 2 images so that the second one is within 0-1 px displacement
Images: 8-bit BMP, as small as possible (not targeting a particular camera, so what renders fastest)
Cases: AIR_DIFFUSE, PIPE, WATER
Do DIC on these
"""
from enum import StrEnum
import numpy as np
from pathlib import Path
from global_utils import *
from convergence_common import *
from copy import deepcopy
import matplotlib.ticker as ticker

from pyvale.sensorsim.imagetools import ImageTools
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
# DIC
# ================================================================================

ROI_FILENAME = "roi.dat"
DIC_RESULTS_PREFIX = "dic_results_"
SUBSET_SIZE = 25 # Riley paper used 51 x 51 at 2x res, with 10 px step size, so...
STEP_SIZE = 5

def save_dic_component_plot(ss_x: np.ndarray, ss_y: np.ndarray,
    values: np.ndarray, target_path: Path,
    filename: str, title: str,clim: tuple[float, float] | None = None):

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SINGLE_CMAP)

    im = ax.pcolor(ss_x, ss_y, values, cmap=CMAP_DIC_FEA,
        vmin=None if clim is None else clim[0],
        vmax=None if clim is None else clim[1])

    ax.set_aspect("equal")
    ax.invert_yaxis() # Needed to make the FEA and DIC coords match

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.08)  # 5% of axes height
    cbar.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((0, 0)) # "Always use scientific notation"
    cbar.update_normal(im)
    cbar.ax.set_title(title, pad=8, fontsize=FONT_SIZES["axis_labels"]-1)

    plt.tight_layout()
    #plt.show()

    fig.savefig(target_path / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_dic_fea(test_case: TestCaseApp, displaced_frame_idx: int,
                save_plot: bool = True, convert_to_mm: bool = True,
                ux_limits: tuple[float, float] | None = None,
                uy_limits: tuple[float, float] | None = None):
    """
    Runs DIC on the rendered images.
    """
    import pyvale.dic as dic
    # Unscaled FEA data: x: 1.5693988777967e-06, y:1e-05,1.5693988777967e-06
    # Scaled this would correspond to
    # Unscaled max displacement: 9.999999999621423e-06 mm, which is less than the scale 1 px = 0.0390625 mm
    #Scaled max displacement: 0.0390625 mm, 1.0 px
    SCALE_PX_MM = 0.0390625 # from plate_test

    # Open the deformed images (only 2, so we can hardcode this)
    base_data_dir = f"app2_fea/renders/{test_case.value}"
    target_path = test_dir(BASE_TEST_DIR, base_data_dir)
    ref_img_path = target_path / "rtimage_frame0.bmp"
    def_img_path = target_path / "rtimage_frame63.bmp"
    ref_img = ImageTools.load_image_greyscale(ref_img_path)
    def_img = ImageTools.load_image_greyscale(def_img_path)

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
        dic.calculate_2d(reference=ref_img,
                        deformed=def_img,
                        roi_mask=roi.mask,
                        seed=[377, 276],
                        subset_size=SUBSET_SIZE,
                        subset_step=STEP_SIZE,
                        shape_function="AFFINE",
                        correlation_criteria="ZNSSD",
                        output_basepath=target_path,
                        output_delimiter=",",
                        output_prefix=dic_results_prefix)
    
    # Read data
    dicdata = dic.import_2d(data=dic_files, delimiter=",", binary=False)

    # Data for the first deformation image (and the only one in this case)
    horizontal_displacement = dicdata.u[0]
    vertical_displacement = dicdata.v[0]
    unit = "[px]"
    figure_filename = f"{test_case.value}_fea_dic_plot_px"
    if convert_to_mm:
        horizontal_displacement *= SCALE_PX_MM
        vertical_displacement *= SCALE_PX_MM
        unit = "[mm]"
        figure_filename = f"{test_case.value}_fea_dic_plot_mm"

    # Convert to physical convention: y up (to match FEA)
    horizontal_displacement = horizontal_displacement
    vertical_displacement = -vertical_displacement

    if save_plot:
            save_dic_component_plot(
                dicdata.ss_x,
                dicdata.ss_y,
                horizontal_displacement,
                target_path,
                filename=figure_filename + "_ux.png",
                title=f"$u_x$ {unit}",
                clim=ux_limits)
            save_dic_component_plot(
                dicdata.ss_x,
                dicdata.ss_y,
                vertical_displacement,
                target_path,
                filename=figure_filename + "_uy.png",
                title=f"$u_y$ {unit}",
                clim=uy_limits)
    

# ================================================================================
# FEA plotter
# ================================================================================

def plot_node_disp_component(plate_rtmesh: RTMesh,
                    displacement: np.ndarray,
                    timestep: int,
                    output_path: Path,
                   component: str, # "u_x" or "u_y"
                    clim: tuple[float, float] | None = None,
                    use_deformed_coords: bool = False,
                    unit: str = "[mm]"):
    
    comp_idx = {"u_x": 0, "u_y": 1}[component]
    cells = plate_rtmesh.connectivity

    face_count, nodes_per_face = cells.shape
    cell_array = np.hstack([np.full((face_count, 1), nodes_per_face, dtype=np.int64), cells.astype(np.int64)]).ravel()
    celltypes = np.full(face_count, pv.CellType.QUAD, dtype=np.uint8)

     # Deformed nodal coordinates at this timestep
    points = (plate_rtmesh.node_coords_over_time[timestep]
        if use_deformed_coords else plate_rtmesh.node_coords)

    values = displacement[timestep, :, comp_idx]
    mesh = pv.UnstructuredGrid(cell_array, celltypes, points.copy())
    mesh.point_data[component] = values

    plotter = pv.Plotter(off_screen=True, window_size=(2312, 2905))

    label_str= f"{component} {unit}"
    plotter.add_mesh(
        mesh,
        scalars=component,
        clim=clim,
        cmap=CMAP_DIC_FEA,
        show_edges=False,
        lighting=False,
        scalar_bar_args=dict(
            title=label_str, # no title above cbar
            vertical=True,
            height=0.5, # 50% of window height
            width=0.08, # 3% of window width
            position_x=0.85,
            position_y=0.25,
            title_font_size=70, # They make no sense to me, but look good, so ok
            label_font_size=50,
            font_family="courier",
            fmt="%.1e"))

    # View front face rather than 3D mesh
    plotter.view_xy()
    plotter.enable_parallel_projection()

    fname = output_path / f"node_disp_{component}_t{timestep}.pdf"
    plotter.save_graphic(fname, title=f"FEA {component} {unit}", raster=True)
    plotter.close()

# ================================================================================
# Rendering images and plotting combined
# ================================================================================

def plate_test(test_case: TestCaseApp, aa_subsamples: int = 1, render: bool = False, plot: bool = False, crop_px: bool = False):
    # 1. Paths and access to all data used in the scene
    # Object = main mesh that moves
    object_access = "thesis-data/plate_hole/platehole3d_2mr_63f"
    simdata_path = full_path(object_access) # full path to e.g., tank_surface_TRI3.vtk
    ref_texture = full_path("thesis-data/texture/speckle.tiff")
    object_texture = ImageTools.load_image_greyscale(ref_texture) 
    # Pipe - we borrow data from application_1_rbm because it is shorter
    pipe_access = "thesis-data/pipe_plate"
    pipe_path = get_tank_path(pipe_access, Elements.TRI6) # TRI3 or TRI6 only for pipe
    water_path = get_fill_path(pipe_access, Elements.TRI6)

    # 2. Set up the meshes
    scene = Scene()
    object = simdata_csv_to_rtmesh(simdata_path, sens.EDim.THREED) # Plate is (25, 35, 1) mm in x,y,z spans
    # Position the object so it touches the bottom of the pipe
    pipe_bottom_inner_y = - 23 + 4 # Position of the inner edge of the pipe tank, based on the geometry
    object.place_at(np.array([0.0, pipe_bottom_inner_y, 0.0]), anchor=Anchor.BASE)
    pipe = any_mesh_to_rtmesh(pipe_path)
    water = any_mesh_to_rtmesh(water_path)

    # 3. Camera and output settings
    # Output settings and directory
    output_format = ImageFormat(OutputFormat.IMG_BMP_8BIT, BitDepth.BIT_8, ChannelCount.MONO, True)
    base_data_dir = f"app2_fea/renders/{test_case.value}"
    target_path = test_dir(BASE_TEST_DIR, base_data_dir)
    # Anti-aliasing
    anti_alias = aa_subsamples; # for anti-aliasing
    print(f"Anti-aliasing subsamples: {anti_alias}")
    image_width = image_width_phs6 # px
    image_height = image_width_phs6 # px; height = width for Novas
    pixel_pitch = pixel_pitch_ph6
    focal_length = 50 # mm
    sensor_height_mm = sensor_height_phs6
    # Derived camera parameters
    fov_height = object.get_size()[1] + 5 # See the entire height of the plate + some extra to get the edges
    camera_distance = camera_working_distance(focal_length, fov_height, sensor_height_mm)
    target_distance = camera_distance - focal_length
    camera_y_position = pipe_bottom_inner_y + object.get_size()[1]/2 # Lower the y-position of the camera to match that of the plate
    if not crop_px:
        camera_target = np.array([0, camera_y_position, target_distance])
        camera_center = np.array([0, camera_y_position, camera_distance])
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
        camera_target = np.array([camera_x, camera_y_position, target_distance])
        camera_center = np.array([camera_x, camera_y_position, camera_distance])

    angle_vertical_view = vertical_fov_from_sensor(sensor_height=sensor_height_mm, focal_length=focal_length)
    cam = Camera(image_width, image_height, camera_center, camera_target, angle_vertical_view)

    #SceneVisualiser([pipe, water], cam) # Check positioning
    
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

    # 4.Texture and speckle pattern information for the plate
    # The loaded texture is 2464 x 2056 px (5MPx), 8-bit .tiff; speckles sampled by 5 pixels
    # Rescale the texture since we're using 1024 x 1024 px res, so speckle size will be too small
    object.set_surface(SurfType.TEXTURE, surface_fill=object_texture, material_type=MaterialType.DIFFUSE)
    #uv_scale = speckle_scaling(image_width, image_height, 2464, 2056, 5, 5) # Aim to have 5 px speckles again
    uv_scale = speckle_scaling(image_width, image_height, 2464, 2056, 5, 5) # Aim to have 5 px speckles again
    object.uvs *= uv_scale
    # But if scaling is needed refer to application_1_rbm

    # 5. Pick frames to render
    # We only want 2:
    # 1. Undeformed, so at t=0
    # 2. Deformed, somewhere s.t., displacement is <= 1 px
    scale = spatial_scale(fov_height, image_height) # mm/px, so 1 px = this in mm; 0.0390625 in this case
    print(f"Spatial scale: {scale}")
    node_coords_over_time = object.node_coords_over_time
    node_displacements = np.ndarray(node_coords_over_time.shape) # Array for nodal displacements over time
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
        object.node_coords_over_time[displaced_frame_idx] = (object.node_coords_over_time[0] + scaled_disp)
        # Recompute displacement to verify appropriate scaling
        disp_check = object.node_coords_over_time[displaced_frame_idx] - object.node_coords_over_time[0]
        max_disp_mm_check = np.max(np.linalg.norm(disp_check, axis=-1))
        max_disp_px_check = max_disp_mm_check / scale
        print(f"Scaled max displacement: {max_disp_mm_check} mm, {max_disp_px_check} px")
    scene.add_rtmesh(object)

    # 5. Render
    scene.add_camera(cam)
    scene_deformed = deepcopy(scene)

    fresh_filename = "rtimage_0_cam0.bmp"
    # Render undeformed image
    if render:
        if crop_px:
            # Adjust rendered image size (but none of the scene dimensions) to chop a few px off to save on render time, while getting the same exact output for ROI
            image_width = image_width_phs6 - 2 * crop_horizontal
            image_height = image_width_phs6 - 2 * crop_vertical
        render_scene(image_height, image_width, scene, anti_alias, target_path, RenderType.STATIC, texture_sampler = TextureSampler.CATMULL_ROM, shading_type = ShadingType.FLAT, image_format = output_format, omp_thread_count = None)
        new_filename = "rtimage_frame0.bmp"
        os.rename(target_path.joinpath(fresh_filename), target_path.joinpath(new_filename))
        # Render deformed image
        render_scene(image_height, image_width, scene_deformed, anti_alias, target_path, RenderType.STATIC, frames_to_render=displaced_frame_idx, texture_sampler = TextureSampler.CATMULL_ROM, shading_type = ShadingType.FLAT, image_format = output_format, omp_thread_count = None)
        new_filename = f"rtimage_frame{displaced_frame_idx}.bmp"
        os.rename(target_path.joinpath(fresh_filename), target_path.joinpath(new_filename))
        
    if plot:
        # Plot FEA data
        from pyvale.raytracer.rtmesh import _read_nodal_displacements
        displacement = _read_nodal_displacements(
            simdata_path / "field_disp_x.csv",
            simdata_path / "field_disp_y.csv",
            simdata_path / "field_disp_z.csv")
        if max_displacement_mm <= 0.5 * scale:
            displacement_for_plot = displacement * factor
        else:
            displacement_for_plot = displacement
        
        ux_fe_phys = displacement_for_plot[displaced_frame_idx, :, 0] # x right
        uy_fe_phys = displacement_for_plot[displaced_frame_idx, :, 1] # y up

        # Symmetric limits around zero in physical convention to make DIC and FEA plots match
        # They agree otherwise, except for the signs
        #ux_abs = float(np.nanmax(np.abs(ux_fe_phys)))
        #uy_abs = float(np.nanmax(np.abs(uy_fe_phys)))
        #ux_limits = (-ux_abs, ux_abs)
        #uy_limits = (-uy_abs, uy_abs)
        # This looks better
        ux_limits = (float(np.nanmin(ux_fe_phys)), float(np.nanmax(ux_fe_phys)))
        uy_limits = (float(np.nanmin(uy_fe_phys)), float(np.nanmax(uy_fe_phys)))

        # FEA plots
        plot_node_disp_component(object, displacement_for_plot, displaced_frame_idx, target_path, 
                          component="u_x", clim=ux_limits, use_deformed_coords=False, unit="[mm]")
        plot_node_disp_component(object, displacement_for_plot, displaced_frame_idx, target_path, 
                          component="u_y", clim=uy_limits, use_deformed_coords=False, unit="[mm]")

        run_dic_fea(test_case, displaced_frame_idx=displaced_frame_idx, save_plot=True,
                    convert_to_mm=True, ux_limits=ux_limits, uy_limits=uy_limits)

# Run 3 cases, then shove them into DIC engine, that's it
plate_test(TestCaseApp.PIPE, 2**0, render = False, plot = True, crop_px=True)