import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from enum import StrEnum

from pyvale.raytracer.rtoutputformat import *

# ================================================================================
# GLOBAL UTILITIES AND SETTINGS
# ================================================================================
PARENT_DIR = Path(__file__).resolve().parent # Home directory to this specific file

# Choose output directory for the rendered images
BASE_TEST_DIR = Path(__file__).resolve().parent.parent / "thesis-output" # Home directory to this specific file
if not BASE_TEST_DIR.is_dir():
    BASE_TEST_DIR.mkdir(parents=True, exist_ok=True)

def full_path(data_location: str):
    """
    Convenience helper to point to specific datasets in a directory relative to the current one's PARENT.
    E.g., we are in pyvaleCom/thesis-tests and want to test pyvaleCom/tests/texture/cal_target.tiff => just pass "tests/texture/cal_target.tiff"
    """
    return (PARENT_DIR.parent / data_location)
    #return Path(Path().resolve().joinpath(data_location)) # This is ok if we aren't travelling through the parent directory
 
def test_dir(BASE_TEST_DIR: Path, test_name: str):
    """
    Small helper function to make separate directories for each test to avoid overwriting data.
    """
    test_dir = BASE_TEST_DIR.joinpath(test_name)
    if not test_dir.is_dir():
        test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir

# Elements for plots etc.
@dataclass(slots=True)
class Element:
    label: str = field(default_factory = None)
    color: str = field(default_factory= None)
    
class Elements:
    QUAD4 = Element("QUAD4", "#ead6c2") # Golden
    QUAD8 = Element("QUAD8", "#53424c") # Dark burgyndy-ish
    QUAD9 = Element("QUAD9", "#b9e1d8") # Mint
    TRI3 = Element("TRI3", "#c99fb6") # Pink
    TRI6 = Element("TRI6", "#826f99") # Purple

def iter_elements():
    # Iterates over elements above
    for name, value in vars(Elements).items():
        if isinstance(value, Element):
            yield name, value

def iter_elements_plot_order():
    """
    Iterate through the elements starting with triangles for neater plots.
    """
    order = ("TRI3", "TRI6", "QUAD4", "QUAD9", "QUAD8")
    for name in order:
        yield name, getattr(Elements, name)
    
# Use:
#for name, elem in iter_elements():
#   print(name, elem.label, elem.color)


# Plot settings
FONT_SIZES = {"suptitle": 25, "subtitle": 20, "axis_labels": 22, "ticks": 13, "subplot_labels": 16, "legend": 15}
FIGURE_SIZE = (12, 10)
FIGURE_SIZE_STACKED_HOR = (12, 16)
RESIZE_PLOT_FACTOR = 50
SUBPLOT_SPACING = 0.3

# ================================================================================
# CAMERA DATA: Photron Nova S6, pinhole
# ================================================================================
image_width_phs6 = 1024  # px; max resolution
image_height_phs6 = 1024  # px; max resolution, PH6 record squares
sensor_height_phs6 = 20.48 # mm
lens_focal_length_phs6 = 100 # mm; based on the lens I used
pixel_pitch_ph6 = 0.020 # mm; 20 um


# Need to set camera_target, camera_center, and angle_vertical_view depending on the test
output_format_phs6 = ImageFormat(OutputFormat.IMG_TIFF_16BIT, BitDepth.BIT_12, ChannelCount.MONO, True)
#output_format_phs6 = ImageFormat(OutputFormat.IMG_TIFF_16BIT, BitDepth.BIT_12, ChannelCount.RGB, False) # For specific tests where colour is needed

# Coloured output that is faster to write, to test if dielectrics are fine
output_format_test_diel = ImageFormat(OutputFormat.IMG_BMP_24BIT, BitDepth.BIT_8, ChannelCount.RGB, False)

# ================================================================================
# CAMERA DATA: LaVision Imager CX-5, pinhole
# ================================================================================
image_width_cx5 = 2440 # px
image_height_cx5 = 2040 # px
pixel_pitch_cx5 = 0.0027 # mm; 2.7 x 2.7 um pixel size
sensor_width_cx5 = 5.6 # mm
sensor_height_cs5 = 6.7 # mm

# Can do 8- and 12- bit, we keep 8-bit BMP to speed up computations
output_format_cx5 = ImageFormat(OutputFormat.IMG_TIFF_8BIT, BitDepth.BIT_8, ChannelCount.MONO, True)

# ================================================================================
# CAMERA FUNCTIONS
# ================================================================================

def vertical_fov_from_sensor(sensor_height: float,
                             focal_length: float) -> float:
    v_angle_rad = 2.0 * np.arctan(sensor_height / (2.0 * focal_length))
    return np.rad2deg(v_angle_rad)

def vertical_fov_from_resolution(resolution: float, scale_px_per_mm: float, distance_to_sample: float) -> float:
    """
    Compute vertical FOV (degrees) from target image dimensions

    Note: distance_to_sample is from lens (due to this being pinhole), not the camera itself
    """
    target_fov_mm = resolution / scale_px_per_mm  
    target_half_fov = target_fov_mm / 2.0
    return np.rad2deg(2 * np.arctan(target_half_fov / distance_to_sample))

def active_sensor_height(target_image_height: int, pixel_pitch: float) -> float:
    """
    Written only to keep in mind that at a reduced resolution, the sensor size of a camera is cropped.
    """
    return target_image_height * pixel_pitch

def camera_working_distance(focal_length: float | int, fov_height: float | int, active_sensor_height: float) -> float:
    return focal_length * fov_height / active_sensor_height

def spatial_scale(fov_size: float | int, image_size: int) -> float:
    """
    Finds the spatial scale for the given set-up (presumably in mm/px).
    NB4 FOV size and image size must be given along the same axis.
    """
    return fov_size/image_size

def speckle_scaling(image_width: int,
               image_height: int,
               texture_width: int,
               texture_height: int,
               speckle_size_texture: int | float,
               speckle_size_target: int | float) -> np.ndarray:
    """
    Finds the appropriate scaling factors for the mesh UVs to achieve the desired speckle size in px.

    All passed values are in px.

    Returns a (2,) numpy array ready to multiply the RTMesh's uvs.
    """
    # Mapping between texture pixels and the rendered output pixels
    scaling_horizontal = texture_width / image_width
    scaling_vertical = texture_height / image_height
    #print(f"Scaling horizontal: {scaling_horizontal}")
    
    # Speckle dimensions (diameter) in the rendered image
    rendered_speckle_hor = speckle_size_texture / scaling_horizontal
    rendered_speckle_vert = speckle_size_texture / scaling_vertical
    #print(f"Rendered speckle horizontal: {rendered_speckle_hor}")

    # Tile factor; for T < 0, the texture is zoomed in, which is what we are aiming for
    # T = 1 <=> Full texture covers the 0-1 UV range once (default after unwrapping)
    # T > 1 currently unsupported in the ray-tracer engine, but could work in principle
    # Find the tile factor
    tile_factor_hor = rendered_speckle_hor / speckle_size_target
    tile_factor_vert = rendered_speckle_vert / speckle_size_target

    # Retun tile factors that can be then used to either:
    # a) Downscale the texture as e.g., tex_width *= tile_factor_vert
    # b) Multiply the uvs, mesh_uvs *= tile_factor_array
    return np.array([tile_factor_hor, tile_factor_vert], dtype=np.float64)

# ================================================================================
# Other
# ================================================================================

# Enum with test cases for applications 1 and 2
class TestCaseApp(StrEnum):
    AIR_DIFFUSE = "air_diffuse"
    PIPE = "pipe"
    WATER = "water"