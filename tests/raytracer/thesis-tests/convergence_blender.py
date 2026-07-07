"""
Blender (bpy) equivalent of convergence_rt.
We deliberately mirror, as closely as the Blender API allows:

  - The scene geometry (same meshes, same world placement);
  - The use of ambient lighting only (a uniform world background, no lamps);
  - The camera parameters (pinhole, sensor-height-derived vertical FOV, y-up view convention,
    look-at via the camera target);
  - The subsampling convergence logic (double the sample count until the bitwise RMSE between
    successive renders falls below a limit, or a maximum sample count is reached), logging every
    iteration to CSV.

Notes on shading
----------------
Pyvale ray tracer's ShadingType.FLAT uses the geometric normals found during ray-triangle
intersection (one constant normal per triangle), and that was used in convergece_rt.
Blender's shader engines always interpolate the mesh's stored vertex normals across a
face, so there is no direct switch to always use the geometic normals.
The closest equivalent is to mark every face as flat shaded (shade_flat), which makes Blender
store one normal per face and use it across the whole triangle <=> geometric normal for a planar
TRI3. 
If for some reason flat shading is unavailable, the documented fallback is to use
angle-averaged shading normals (shade_smooth with auto-smooth), which the shade_rtmesh_flat helper
falls back to.
"""
import csv
import os
import bpy
import numpy as np
import timeit

from pyvale.raytracer.rtblender import BlenderUnwrapper
from pyvale.raytracer.rtmesh import RTMesh, pyvista_faces_to_connectivity

from pyvale.raytracer.rtmesh import *
from pyvale.raytracer.rtmeshvisuals import *
from pyvale.raytracer.rtblender import *
from pyvale.raytracer.rtcamera import *
from pyvale.raytracer.rtscene import *
from pyvale.raytracer.rtpresets import *
from pyvale.raytracer.rtmain import *
from pyvale.raytracer.rtoutputformat import *
from convergence_rt import * # This also imports global positioning of the scene, so we don't need to copy it here
from global_utils import *

SUBSAMPLE_LIMIT_MAX = 2**22 # Overwrite the value from convergence_rt to set limits for Blender separately if needed

# ================================================================================
# CONVENIENCE CONVERTERS: pyvale scene objects -> Blender datablocks
# ================================================================================
def clear_blender_scene() -> None:
    """
    Remove all objects, meshes, materials, cameras and the world from the
    current Blender file so each render starts from a clean, deterministic state.
    """
    # Make sure we are in object mode before deleting anything
    if bpy.context.mode != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Purge datablocks left behind by the object deletion
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.images,
        bpy.data.objects):
        for datablock in list(collection):
            if getattr(datablock, "users", 0) == 0:
                collection.remove(datablock)


def convert_rtcamera_to_blender(camera: Camera) -> bpy.types.Object:
    """
    Translate a pyvale :class:`Camera` (pinhole model) into a Blender camera.

    The :class:`Camera` stores: the view-up convention (y is up), the focal
    length, the vertical field of view, the camera centre and target, and the
    associated transformation / viewport matrices. We reproduce the pinhole
    projection in Blender by:

      - Driving the lens from the sensor height and focal length with a
        vertical sensor fit, which makes Blender's intrinsic vertical FOV equal
        to vertical_fov_from_sensor(sensor_height, focal_length); i.e. the
        same camera.angle_vertical_view used by the ray tracer; and
      - Placing the camera at camera.camera_center and aiming it at
        camera.point_camera_target via a Track-To constraint, looking down -Z with
        +Y as up (the ray tracer's y-up convention).

    The FOV is the authoritative parameter (so the projection matches the ray
    tracer's frustum exactly); the sensor height and a FOV-consistent lens length
    are also set so the camera stays physically faithful to the Photron Nova S6.

    NOTE: Camera.angle_vertical_view is already in radians, so it is passed
    to Blender unchanged.

    Parameters
    ----------
    camera : Camera
        Source pinhole camera from the pyvale scene.

    Returns
    -------
    bpy.types.Object
        The created (and active) Blender camera object.
    """
    # 1. Create the camera data and object
    cam_data = bpy.data.cameras.new("PhotronCamera")

    # IMPORTANT: Camera.angle_vertical_view is ALREADY stored in radians, so do
    # not convert this or the image will be very zoomed in (incorrectly)
    vfov_rad = float(camera.angle_vertical_view)

    # Pinhole matching the physical dims of Photron Nova S6's
    cam_data.sensor_fit = "VERTICAL"
    # Vertical sensor fit with the real Photron sensor height
    cam_data.sensor_height = sensor_height_phs6
    # Derive the lens focal length from the same vertical FOV so
    # the millimetre and FOV representations are mutually consistent
    cam_data.lens = sensor_height_phs6 / (2.0 * np.tan(vfov_rad / 2.0))

    # Primary parameter: drive the projection directly from the FOV the ray
    # tracer uses, so the two pinhole frustums coincide. Blender expects radians
    cam_data.lens_unit = "FOV"
    cam_data.angle = vfov_rad

    # No depth of field in pinhole
    cam_data.dof.use_dof = False

    # Generous clip range so geometry far from the camera is never silently
    # culled. Blender's default clip_end (100-1000 depending on version) would
    # otherwise cut off objects pushed far down -Z (e.g., beam at z=-1000 disappeared)
    # Near 0.01 keeps things sharp without z-fighting at this scale
    cam_data.clip_start = 0.01
    cam_data.clip_end = 1.0e6

    # Create camera object
    cam_obj = bpy.data.objects.new("PhotronCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)

    # Set as the active scene camera and position it at the camera centre
    bpy.context.scene.camera = cam_obj
    cam_obj.location = tuple(float(v) for v in camera.camera_center)

    # 2. Create an Empty object to act as the look-at target
    target_obj = bpy.data.objects.new("CameraTarget", None)
    target_obj.location = tuple(float(v) for v in camera.point_camera_target)
    bpy.context.scene.collection.objects.link(target_obj)

    # 3. Add a Track-To constraint so the camera always aims at the target,
    # looking down its -Z axis with +Y up (y-up view convention)
    track_constraint = cam_obj.constraints.new(type="TRACK_TO")
    track_constraint.target = target_obj
    track_constraint.track_axis = "TRACK_NEGATIVE_Z"
    track_constraint.up_axis = "UP_Y"

    return cam_obj


def convert_rtmesh_to_blender(rtmesh: RTMesh, name: str = "RTObject") -> bpy.types.Object:
    """
    Translate an :class:`RTMesh` into a Blender mesh object.

    This reuses the same vertex/face extraction strategy as
    :class:`BlenderUnwrapper` (see BlenderUnwrapper.add_rtmesh / blender_load_mesh):
    For higher-order meshes we use the triangulated surface's connectivity and node coordinates;
    for meshes that are already TRI3 we use the original data directly as Blender only accepts
    TRI3's (and polygons).
    NB4, convergence test only expects TRI3's, but keeping this just in case.

    The mesh is created with the vertices already expressed in the world frame, so the Blender
    object transform is left at the identity and the geometry lines up 1:1 with the ray tracer's world.

    Parameters
    ----------
    rtmesh : RTMesh
        The mesh to convert.
    name : str
        Base name for the created Blender mesh/object datablocks.

    Returns
    -------
    bpy.types.Object
        The created Blender object (linked to the scene, set active).
    """
    # Same branch as BlenderUnwrapper: prefer the triangulated surface when a
    # higher-order -> tri mapping exists, otherwise use the native TRI3 data
    if rtmesh.tri_face_mapping is not None:
        faces = pyvista_faces_to_connectivity(rtmesh.pyvista_surface)
        vertices = rtmesh.pyvista_surface.points
    else:
        vertices = rtmesh.node_coords
        faces = rtmesh.connectivity

    blender_mesh = bpy.data.meshes.new(name + "Mesh")
    # Empty edge list -> edges are inferred from the faces
    blender_mesh.from_pydata(
        np.asarray(vertices, dtype=np.float64).tolist(),
        [], # edges = empty
        np.asarray(faces, dtype=np.int64).tolist())
    blender_mesh.update()

    blender_obj = bpy.data.objects.new(name, blender_mesh)
    bpy.context.collection.objects.link(blender_obj)
    bpy.context.view_layer.objects.active = blender_obj

    # Geometry is already in world coordinates; keep the object at the origin
    blender_obj.location = (0.0, 0.0, 0.0)

    # Ensure consistent, outward-facing normals. This matters a lot for the refractive (dielectric) cases:
    # Cycles relies on the normal direction to decide entering vs. exiting a medium,
    # and inconsistent winding from a surface-extract produces pure transmission noise
    blender_mesh.validate(verbose=False)
    try:
        # This still needs working - I've never managed to get bmesh to import and work
        import bmesh

        bm = bmesh.new()
        bm.from_mesh(blender_mesh)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(blender_mesh)
        bm.free()
        blender_mesh.update()
    except Exception as exc:
        print(f"Normal recalculation skipped for {name}: {exc}")

    return blender_obj


def write_uvs_to_blender_object(blender_obj: bpy.types.Object,
    rtmesh: RTMesh,
    uv_map_name: str = "UVMap") -> str:
    """
    Write the RTMesh's UVs (the same ones the native ray tracer samples) onto
    blender_obj as an explicit, active-render UV layer.

    Why this exists
    ---------------
    BlenderUnwrapper.smart_unwrap leaves a UV layer behind from
    bpy.ops.uv.smart_project + pack_islands, but that operator-left
    layer is not guaranteed to be the active render layer, and packing rescales
    islands. Relying on it might cause Blender to sample a near-degenerate UV layer.
    
    Instead we take rtmesh.uv [shape (element_count, nodes_per_element,2)]
    and write it per loop, then flag the layer as the active render layer.
    This guarantees Blender and the ray tracer sample identical texture coordinates.

    Parameters
    ----------
    blender_obj : bpy.types.Object
        The object whose loops receive the UVs. Its polygon order must match the
        element order of rtmesh.uvs (true when the object was built from the
        same connectivity used during unwrapping).
    rtmesh : RTMesh
        Mesh whose uvs attribute holds the per-element, per-corner UVs.
    uv_map_name : str
        Name for the created UV layer.

    Returns
    -------
    str
        The name of the UV layer that was created and set active for render.
    """
    mesh = blender_obj.data
    uvs = np.asarray(rtmesh.uvs, dtype=np.float64)

    if uvs.ndim != 3 or uvs.shape[2] != 2:
        raise ValueError(
            f"Expected rtmesh.uvs of shape (element_count, nodes_per_element, 2),"
            f"got {uvs.shape}."
        )
    if len(mesh.polygons) != uvs.shape[0]:
        raise ValueError(
            f"Polygon count ({len(mesh.polygons)}) does not match UV element "
            f"count ({uvs.shape[0]}). The Blender object and the unwrapped mesh "
            "must share the same connectivity/order.")

    # Remove any pre-existing layers so the operator-left one cannot win
    while mesh.uv_layers:
        mesh.uv_layers.remove(mesh.uv_layers[0])

    uv_layer = mesh.uv_layers.new(name=uv_map_name)
    uv_data = uv_layer.data

    for poly in mesh.polygons:
        elem_uvs = uvs[poly.index]  # (nodes_per_element, 2)
        for corner, loop_index in enumerate(poly.loop_indices):
            u, v = elem_uvs[corner]
            uv_data[loop_index].uv = (float(u), float(v))

    # Make this the active layer AND the active *render* layer;
    # The latter is what Cycles/EEVEE engines actually sample
    mesh.uv_layers.active = uv_layer
    uv_layer.active_render = True
    mesh.update()
    return uv_layer.name


def shade_rtmesh_flat(blender_obj: bpy.types.Object,
                      smooth: bool = False) -> None:
    """
    Set the shading mode on blender_obj.

    smooth=False is the Blender equivalent of the ray tracer's ShadingType.FLAT,
    which shades with the *geometric* triangle normal. For a TRI3 the Blender face
    normal equals the geometric normal, so flat shading reproduces the ray tracer's
    behaviour.

    smooth=True uses angle-averaged shading normals. We may use it as a fallback for
    shading; to be determined if we need it for the dielectric tank/water surfaces to
    avoid excessive noise or nonsensical outputs for TRI3's.

    Parameters
    ----------
    blender_obj : bpy.types.Object
        Mesh object to shade.
    smooth : bool
        False => flat (geometric) normals; True => angle-averaged (blended) normals.
    """
    mesh = blender_obj.data
    for poly in mesh.polygons:
        poly.use_smooth = bool(smooth)
    mesh.update()


def setup_ambient_world(strength: float = 1.0,
                        background_color: tuple = (0.5, 0.5, 0.5, 1.0))-> None:
    """
    Configure a uniform ambient world background and remove all light sources.

    The ray tracer has only ambient light at the time of writing these tests (no 
    extra lamps); Blender doesn't have this per se, so we mimic that by giving the Blender
    world a flat background of the given strength and ensuring there are no light objects
    in the scene. Under a uniform white environment a Lambertian (DIFFUSE) surface receives
    the same irradiance from every direction, which should match the ambient lighting in the ray tracer.
    """
    # Remove any lamps that may exist (should be none, but just in case)
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    bg_node = nodes.new(type="ShaderNodeBackground")
    bg_node.inputs["Color"].default_value = background_color
    bg_node.inputs["Strength"].default_value = float(strength)

    world_output = nodes.new(type="ShaderNodeOutputWorld")
    links.new(bg_node.outputs["Background"], world_output.inputs["Surface"])


def create_blender_material(name: str,
    rt_material: Material,
    mat_type: MaterialType,
    texture_path: str | None = None,
    uv_map_name: str = "UVMap") -> bpy.types.Material:
    """
    Build a Blender material that mirrors a pyvale :class:`Material` / surface.

    UNLIT
        Uses an Emission shader so the surface ignores lighting entirely (its
        appearance is its texture/colour); this is the analogue of the ray tracer's
        MaterialType.UNLIT. When a texture is supplied it is loaded as
        Non-Color data (no gamma curve) for faithful tracking and sampled using the
        UV map from :class:`BlenderUnwrapper`.
    DIFFUSE
        Principled BSDF, fully rough (Lambertian), specular off.

    REFRACTIVE
        Principled BSDF, smooth, full transmission, IOR taken from the preset's
        refractive index (Material.RI).

    Parameters
    ----------
    name : str
        Name for the new material datablock.
    rt_material : Material
        Source material (provides color and, for dielectrics, RI).
    mat_type : MaterialType
        Shading category (UNLIT / DIFFUSE / REFRACTIVE).
    texture_path : str | None
        Optional path to a greyscale texture image (used for UNLIT surfaces).

    Returns
    -------
    bpy.types.Material
        The created material.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()
    node_output = nodes.new(type="ShaderNodeOutputMaterial")
    node_output.location = (400, 0)

    base_color = (
        float(rt_material.color[0]),
        float(rt_material.color[1]),
        float(rt_material.color[2]),
        1.0,
    )

    # Build the texture node ONCE (if a texture is supplied). The same texture is
    # routed into whichever shader the material uses, so the speckle is present
    # for UNLIT and DIFFUSE beams
    has_texture = bool(texture_path) and os.path.exists(str(texture_path))
    node_tex = None
    if has_texture:
        node_tex = nodes.new(type="ShaderNodeTexImage")
        node_tex.location = (-300, 0)
        img_name = os.path.basename(str(texture_path))
        img = bpy.data.images.get(img_name)
        if not img:
            img = bpy.data.images.load(str(texture_path))
        # Prevent gamma-curve alterations
        img.colorspace_settings.name = "Non-Color"
        node_tex.image = img
        # Clamp (no tiling) + cubic interpolation (closest to Catmull-Rom used in tests)
        node_tex.extension = "EXTEND" # Original setting
        #node_tex.extension = "CLIP"
        node_tex.interpolation = "Cubic"
        # Bind to the exact UV layer written by write_uvs_to_blender_object
        node_uv = nodes.new(type="ShaderNodeUVMap")
        node_uv.location = (-500, 0)
        node_uv.uv_map = uv_map_name
        links.new(node_uv.outputs["UV"], node_tex.inputs["Vector"])

    if mat_type.name == "UNLIT":
        # Emission node => ambient/flat rendering independent of lights
        node_emission = nodes.new(type="ShaderNodeEmission")
        node_emission.location = (0, 0)
        node_emission.inputs["Strength"].default_value = 1.0
        if node_tex is not None:
            # Texture colour -> emission colour
            links.new(node_tex.outputs["Color"], node_emission.inputs["Color"])
        else:
            # Fallback to the flat colour when no texture is provided.
            node_emission.inputs["Color"].default_value = base_color

        links.new(node_emission.outputs["Emission"], node_output.inputs["Surface"])
    else:
        # DIFFUSE and REFRACTIVE both use a Principled BSDF
        node_bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
        node_bsdf.location = (0, 0)
        node_bsdf.inputs["Base Color"].default_value = base_color

        if mat_type.name == "REFRACTIVE":
            # Dielectrics (water, acrylic used in test scene): perfectly smooth, fully transmissive
            # Not textured - keep the flat tint colour
            node_bsdf.inputs["Roughness"].default_value = 0.0
            node_bsdf.inputs["Transmission Weight"].default_value = 1.0
            if rt_material.RI is not None:
                node_bsdf.inputs["IOR"].default_value = float(rt_material.RI)

        elif mat_type.name == "DIFFUSE":
            node_bsdf.inputs["Roughness"].default_value = 1.0  # Lambertian
            node_bsdf.inputs["Specular IOR Level"].default_value = 0.0
            # Take the BSDF base colour from the speckle texture when present
            # Without this diffuse beam renders as solid colour
            if node_tex is not None:
                links.new(node_tex.outputs["Color"], node_bsdf.inputs["Base Color"])

        links.new(node_bsdf.outputs["BSDF"], node_output.inputs["Surface"])

    return mat


def assign_material(blender_obj: bpy.types.Object, material: bpy.types.Material) -> None:
    """Replace the object's material slots with a single ``material``."""
    blender_obj.data.materials.clear()
    blender_obj.data.materials.append(material)


# ================================================================================
# RENDER SETTINGS + RENDER CALL (the Blender version of render_scene)
# ================================================================================
def configure_render_settings(image_width: int,
    image_height: int,
    subsamples: int,
    image_format: ImageFormat = output_format_phs6,
    flat_shading: bool = True,
    cpu: bool = False) -> None:
    """
    Configure Blender's render engine to match the ray tracer's pinhole +
    ambient + anti-aliasing-subsampling setup, and to write a comparable image.

    The key convergence parameter is subsamples - the number of
    anti-aliasing samples per pixel. In Cycles this maps directly to the render
    samples count, so doubling ``subsamples`` doubles the AA sampling exactly
    as in the native ray tracer => We can reuse the same convergence test.

    Parameters
    ----------
    image_width, image_height : int
        Output resolution in pixels.
    subsamples : int
        Anti-aliasing samples per pixel (the value the convergence loop doubles).
    image_format : ImageFormat
        Desired output format. We can use the TIFF 16-bit / mono / linear intent
        of output_format_phs6 as in the original tests.
    flat_shading : bool
        Whether the scene is being rendered with FLAT shading. Recorded for
        clarity; the actual per-object flat flag is set via shade_rtmesh_flat.
    cpu : bool
        Whether to render with CPU instead of GPU. This is to compare directly with
        the ray tracer. Defaults to false
    """
    scene = bpy.context.scene

    # Cycles gives us a physically-based path tracer whose samples count is
    # the AA subsample count we converge on
    scene.render.engine = "CYCLES"
    try:
        scene.cycles.samples = int(subsamples)
        # Deterministic sampling: disable the adaptive sampler and denoiser so a
        # given subsample count always produces the same image (essential for a
        # bitwise RMSE convergence test)
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.use_denoising = False
        scene.cycles.seed = 0
        #scene.cycles.use_guiding = True # TEST THIS; this is default and will do denoising even with denosing off
        # Allow enough refraction/transmission bounces for the dielectric cases (tank / water / nested dielectric)
        # Our ray-tracer goes to depth of 30 for this sample scene, so we apply the same cap here
        scene.cycles.max_bounces = 30
        scene.cycles.transmission_bounces = 30
        scene.cycles.transparent_max_bounces = 30

        # CPU settings if relevant
        if cpu:
            scene.cycles.device = "CPU"
            # Access Cycles add-on preferences
            prefs = bpy.context.preferences
            cprefs = prefs.addons["cycles"].preferences
            # CPU-only compute device type; "None" <=> CPU-only in newer Blender versions
            cprefs.compute_device_type = "NONE"
            print("Switched Blender to use CPU.")

    except AttributeError:
        print("Cycles engine is not available. Falling black to EEVEE.")
        # If Cycles is unavailable, fall back to EEVEE with TAA samples
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        scene.eevee.taa_render_samples = int(subsamples)

    # Resolution (100% scale so the pixel grid matches exactly)
    scene.render.resolution_x = int(image_width)
    scene.render.resolution_y = int(image_height)
    scene.render.resolution_percentage = 100

    # Output format: 16-bit TIFF, mono, linear (no view/display transform) so
    # the data is directly bit-comparable with the ray tracer output for this test
    scene.render.image_settings.file_format = "TIFF"
    scene.render.image_settings.color_mode = "BW"  # MONO
    scene.render.image_settings.color_depth = "16"
    scene.render.image_settings.tiff_codec = "NONE"

    # Linear workflow: standard/raw view transform so values are not tone-mapped
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        scene.display_settings.display_device = "sRGB"
    except (AttributeError, TypeError):
        pass

    # Film: keep transparent off so the ambient world is the background, matching
    # the ray tracer which sees the constant ambient environment behind geometry
    scene.render.film_transparent = False


def render_scene_blender(image_width: int,
    image_height: int,
    subsamples: int,
    target_dir,
    image_format: ImageFormat = output_format_phs6,
    flat_shading: bool = True,
    fresh_filename: str = "rtimage_0_cam0.tiff",
    cpu: bool = False) -> str:
    """
    Render the current Blender scene to target_dir / fresh_filename.

    This is the Blender equivalent of the ray tracer's render_scene(...). To
    keep the convergence loop identical to the native one, it always writes to
    the same base filename (rtimage_0_cam0.tiff); the caller then renames the
    file to encode the subsample count, exactly as in conv_test.

    Parameters
    ----------
    image_width, image_height : int
        Output resolution in pixels.
    subsamples : int
        AA samples per pixel for this render.
    target_dir : pathlib.Path
        Directory the image is written to.
    image_format : ImageFormat
        Output format intent (TIFF 16-bit mono linear).
    flat_shading : bool
        Whether FLAT shading is in effect (passed to settings for clarity).
    fresh_filename : str
        Base output filename (kept constant so the loop can rename it).
    cpu : bool
        Whether to render with CPU instead of GPU. This is to compare directly with
        the ray tracer. Defaults to false

    Returns
    -------
    str
        Absolute path of the rendered image.
    """
    configure_render_settings(
        image_width,
        image_height,
        subsamples,
        image_format=image_format,
        flat_shading=flat_shading,
        cpu = cpu) # Set to true to compare directly to ray-tracer, otherwise keep at False for speed

    out_path = os.fspath(target_dir / fresh_filename)
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    return out_path


# ================================================================================
# MAIN: Blender equivalent of conv_test
# ================================================================================
def conv_test_blender(test_case: TestCase,
    resolution: Resolution = Resolution.HIGH,
    starting_subsamples: int | None = None,
    cpu: bool = False) -> None:
    """
    Blender re-implementation of the ray tracer's conv_test/

    Step-by-step:
      1. Set up all mesh data (TRI3 only; remove element-specific checks, since Blender renders triangles).
      2. Build the pinhole camera from the Photron Nova S6 parameters.
      3. Select per-case surfaces/materials (UNLIT / DIFFUSE / REFRACTIVE) and
         add the tank / water dielectrics where relevant.
      4. Unwrap + texture the beam sample via :class:`BlenderUnwrapper`,
      5. Iteratively double the anti-aliasing subsample count, rendering each
         time and logging the bitwise RMSE between successive images to CSV,
         terminating on RMSE convergence or the maximum subsample count.

    Only ambient lighting is used (a uniform world background, no lamps), the
    camera parameters follow the ray tracer, and the subsampling-verification
    logic is preserved.
    """
    # Start from a clean Blender file every run
    clear_blender_scene()

    # 1. Mesh data (point to the correct mesh locations, etc.); TRI3 only
    tank_access = "thesis-data/" + Tank.RECTANGLE + "/" + Refinement.COARSE
    tank_path = get_tank_path(tank_access, Elements.TRI3)
    water_path = get_fill_path(tank_access, Elements.TRI3)
    sample_path = full_path("thesis-data/beam/" + Refinement.COARSE + "/beam_surface_" + Elements.TRI3.label + ".vtk")

    # We need to make tank RTMesh regardless of the case to snap the beam position correctly
    tank = any_mesh_to_rtmesh(tank_path, world_position = TANK_POSITION, anchor = Anchor.CENTER)

    # Sample (speckle) texture for the beam; no need to load it into numpy array
    ref_texture = full_path("thesis-data/texture/speckle.tiff")

    # BlenderUnwrapper reused for the beam's UVs (same as ray tracer)
    blender_uv = BlenderUnwrapper()

    # 2. Camera: Photron Nova S6, pinhole
    image_width = resolution
    image_height = resolution
    output_format = output_format_phs6
    camera_center = CAMERA_POSITION
    camera_target = CAMERA_TARGET
    # Angle vfov is in degrees
    angle_vfov = vertical_fov_from_resolution(resolution, SCALE_PX_PER_MM, CAMERA_DISTANCE)
    #angle_vfov = 20 # Uncomment to sanity-check nested dielectric set-up
    cam = Camera(image_width, image_height, camera_center, camera_target, angle_vfov)

    # Translate the camera into Blender and set the ambient-only world
    convert_rtcamera_to_blender(cam)
    setup_ambient_world(strength=1.0)

    # 3. Settings based on the selected case
    first_criterion_hit = False # Flag to mark when we hit the MaxAE/RMSE criterion for the first time, to run one more time for sureness and only then terminate
    roi_path = None
    # ROI defined only for high res - for low, the entire image is our ROI
    if resolution == Resolution.HIGH:
        roi_path_access = f"thesis-data/roi_1024_{test_case.value}.csv" 
        roi_path = full_path(roi_path_access)

    mat_type = MaterialType.UNLIT # Beam material (default)
    if test_case == TestCase.AIR_UNLIT:
        print(f"--------------------------------\nTESTED CASE: AIR UNLIT\n--------------------------------")
        mat_type = MaterialType.UNLIT
        if starting_subsamples is None:
            starting_subsamples = 1
    else:
        # This helps us speed up
        if starting_subsamples is None or starting_subsamples < 1:
            raise ValueError("Please base your starting subsample count on the UNLIT case, otherwise this will run for ages.")
        if test_case == TestCase.AIR_DIFFUSE:
            print(f"--------------------------------\nTESTED CASE: AIR DIFFUSE\n--------------------------------")
            mat_type = MaterialType.DIFFUSE
        # For these cases we just accept that we do full shading
        elif test_case == TestCase.TANK:
            print(f"--------------------------------\nTESTED CASE: EMPTY TANK\n--------------------------------")
            mat_type = MaterialType.DIFFUSE
            # Set tank data
            tank_obj = convert_rtmesh_to_blender(tank, name="Tank")
            # Equivalent of RT's "set_surface"
            tank_mat = create_blender_material(
                "TankAcrylic",
                MaterialPresets.PLASTIC_ACRYLIC,
                MaterialType.REFRACTIVE)
            assign_material(tank_obj, tank_mat)
        elif test_case == TestCase.WATER:
            print(f"--------------------------------\nTESTED CASE: TANK WITH WATER\n--------------------------------")
            mat_type = MaterialType.DIFFUSE
            # Set tank (would have priority 0 in RT)
            tank_obj = convert_rtmesh_to_blender(tank, name="Tank")
            tank_mat = create_blender_material("TankAcrylic", MaterialPresets.PLASTIC_ACRYLIC, MaterialType.REFRACTIVE)
            assign_material(tank_obj, tank_mat)
            # Set water (would have priority 1 in RT for nested dielectrics)
            water = any_mesh_to_rtmesh(water_path, world_position = WATER_POSITION)
            # Equivalent of RT's "set_surface"
            water_obj = convert_rtmesh_to_blender(water, name="Water")
            water_mat = create_blender_material("Water", MaterialPresets.WATER, MaterialType.REFRACTIVE)
            assign_material(water_obj, water_mat)

    output_dir_name = "convergence_blender/res_" + str(resolution.value) + "/" + test_case.value

    # 4. Beam sample: load, unwrap via BlenderUnwrapper, texture
    beam = any_mesh_to_rtmesh(sample_path, world_position = BEAM_POSITION, anchor = Anchor.BASE)
    snap_to(beam, tank, Axis.Y, align = (Axis.Z), gap = -BEAM_LEN + 2.0, stack_above = True)
    # Unwrap (this also creates a Blender object internally inside the unwrapper)
    blender_uv.add_rtmesh(beam)
    blender_uv.smart_unwrap()

    # Use the object the unwrapper already created/loaded so the UV layer it generated is the one the material samples
    beam_obj = blender_uv.blender_obj
    beam_obj.name = "Beam"

    # Write the same UVs the ray tracer uses (rtmesh.uvs) onto the beam as the active render
    # UV layer. Relying on the operator-left UV layer from smart_project/pack_islands may cause
    # Blender to sample a near-degenerate layer
    beam_uv_name = write_uvs_to_blender_object(beam_obj, beam, uv_map_name="RTUVMap")

    # Equivalent of RT's "set_surface"
    
    beam_mat = create_blender_material("BeamSpeckle",
        MaterialPresets.OFFICE_PAPER, # Just to have the syntax agree, it is irrelevant for textured non-refractive beam
        mat_type,
        texture_path=ref_texture,
        uv_map_name=beam_uv_name)
    """
    beam_mat = create_blender_material("BeamSpeckle",
        Material(color=np.array([1.0, 0.0, 0.0]), RI=None),
        MaterialType.UNLIT,   # or DIFFUSE
        texture_path=None
    )
    """
    assign_material(beam_obj, beam_mat)

    # Set shading
    blended_names = {"Tank", "Water", "Beam"}
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            # Option 1: Blended shading (can choose materials for this above)
            #shade_rtmesh_flat(obj, smooth=(obj.name in blended_names))
            # Option 2: Flat for everything - preferred
            # Matches the ray tracer convergence test set-up, and renders better
            shade_rtmesh_flat(obj, smooth=False)

    # --------------------------------------------------------------------------
    # ITERATIVE ANTI-ALIASING SUBSAMPLE CONVERGENCE
    # --------------------------------------------------------------------------
    # Blender always writes to the same base name; we rename afterwards to keep each image, exactly as the ray tracer does
    fresh_filename = "rtimage_0_cam0.tiff"
    target = test_dir(BASE_TEST_DIR, output_dir_name + "/" + Elements.TRI3.label)
    csv_path = target / "convergence_log.csv"
    iteration_number = 0
    subsamples = starting_subsamples  # Anti-aliasing samples
    # For saving render time data - useful for CPU rendering to compare directly to the ray-tracer
    time_csv_path = target / "gpu_render_time_log.csv"
    if cpu:
        time_csv_path = target / "cpu_render_time_log.csv"
    time_log_exists = os.path.isfile(time_csv_path)
    time_mode = "a" if time_log_exists else "w" # Append if time log already exists
    
    times = defaultdict()
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile, \
        open(time_csv_path, mode=time_mode, newline="", encoding="utf-8") as timefile:
        writer = csv.DictWriter(csvfile, fieldnames=CONV_CSV_COLS)
        writer.writeheader()
        csvfile.flush()
        os.fsync(csvfile.fileno())

        # Baseline render
        # Need lambda in timeit since we're using local variables
        time = timeit.timeit(lambda: render_scene_blender(image_width, image_height, subsamples, target, image_format=output_format, flat_shading=True, fresh_filename=fresh_filename, cpu=cpu), number=1)
        time_writer = csv.DictWriter(timefile, fieldnames=["subsamples","time (s)"])
        if not time_log_exists:
            time_writer.writeheader()
        time_writer.writerow({"subsamples": subsamples, "time (s)": time})
        timefile.flush()
        os.fsync(timefile.fileno())

        new_filename = "rtimage_" + "subsamples_" + str(subsamples) + ".tiff"
        os.rename(target.joinpath(fresh_filename), target.joinpath(new_filename))

        # Refine until we converge or hit subsample ceiling
        while True:
            prev_filename = new_filename
            subsamples *= 2
            iteration_number += 1

            time = timeit.timeit(lambda: render_scene_blender(image_width, image_height, subsamples, target, image_format=output_format, flat_shading=True, fresh_filename=fresh_filename, cpu=cpu), number=1)
            # Rename file
            new_filename = "rtimage_" + "subsamples_" + str(subsamples) + ".tiff"
            os.rename(target / fresh_filename, target / new_filename)
            # Store time data
            time_writer.writerow({"subsamples": subsamples, "time (s)": time})
            timefile.flush()
            os.fsync(timefile.fileno())
            # Compare this render with the previous one
            rmse, max_ae, percentile_diff, identical_count, total_pixels = bitwise_compare(target / new_filename, target / prev_filename, roi_path, BitDepth.BIT_16)
            print(f"-------------------------------- \nCURRENT SUBSAMPLE COUNT: {subsamples}"
                f"\n\t RMSE: {rmse}"
                f"\n\t MAX ABS ERROR: {max_ae}\n--------------------------------")
            # Store data in CSV/log
            writer.writerow({
                        "iteration": iteration_number,
                        "subsamples": subsamples,
                        "rmse": rmse,
                        "max_ae": max_ae,
                        "99p_abs_error": percentile_diff,
                        "identical_px_count": identical_count,
                        "tot_px_roi": total_pixels})
            csvfile.flush()
            os.fsync(csvfile.fileno())

            # RMSE termination condition - the main one
            if max_ae <= MAX_ABS_ERR_THRESHOLD: # RMSE is max. 1.0 for each individual pixel - we fall within least significant bit convergence
                if not first_criterion_hit:
                    first_criterion_hit = True # True, so we terminate on the next case to make sure we've converged without weird behavIOUR
                    continue
                print("Images converged to the least significant bit. Terminating this case.")
                break
            # Fallback: subsample count
            if subsamples >= SUBSAMPLE_LIMIT_MAX:
                print(f"Exceeded the maximum subsample limit of {SUBSAMPLE_LIMIT_MAX}. Terminating this case.")
                break
        
        with open(time_csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["subsamples", "time (s)"])
            writer.writeheader()
            for subsample_count in times.keys():
                writer.writerow({
                    "subsamples": int(subsample_count),
                    "time (s)": times[subsample_count]})      

#conv_test_blender(TestCase.TANK, Resolution.HIGH, 2**14, False)
#conv_test_blender(TestCase.AIR_UNLIT, Resolution.HIGH, 2**18, False)
#conv_test_blender(TestCase.AIR_UNLIT, Resolution.LOW, 2**20, False)