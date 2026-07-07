"""
Stores all common functions/data for the convergence tests, regardless if
they are dedicated for Linux, Blender, single image case, or others.
"""
from enum import StrEnum, IntEnum
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnchoredOffsetbox

import smplotlib # For nicer figures (imo), but no need to install if you don't want it

from pyvale.raytracer.rtoutputformat import *
from global_utils import *
from tiff12_reader import *

# ================================================================================
# Constants for convergence metrics
# ================================================================================

MAX_ABS_ERR_THRESHOLD = 1.0 # When Max Absolute Error < MAX_ABS_ERR_THRESHOLD, we know that all pixels have RMSE <= 1.0, which matches least significant pixel criterion for convergence

CONV_CSV_COLS = ["iteration", "subsamples", "rmse", "max_ae", "99p_abs_error", "identical_px_count", "tot_px_roi"]

# ================================================================================
# Positioning
# ================================================================================

TANK_OUTER_LEN = 48.0 # Used to shift the tank and water so their front is at 0.0; also use for pipe to keep the overall scene positioning correct
BEAM_LEN = 48.0 # Used to drop the beam so that it is suspended from the top tank edge
TANK_MID_Z = -(TANK_OUTER_LEN/2) # z-coordinate corresponding to the centre of the tank
TANK_POSITION = np.array([0.0, 0.0, TANK_MID_Z]) # Just set it so that the front face is at z = 0.0; the rest can stay
WATER_POSITION = np.array([0.0, -3, TANK_MID_Z]) # Similar to tank, we just offset y a little bit to ensure there's overlap at the bottom for nested dielectrics
BEAM_POSITION = np.array([0.0, 4, 0.0]) # Doesn't really matter; x = 0.0 is key, we snap y and z based on the tank to reproduce the experiments
# Beam was not perfectly centered in the tank in reality due to the position of the breadboard holes. It was slightly to the right and to the front, so we offset it
BEAM_OFFSET = np.array([1.5, 0.0, 3.5]) 
CAMERA_DISTANCE = 110 # From original set-up
CAMERA_HEIGHT = 10.0
#CAMERA_Z = CAMERA_DISTANCE + TANK_MID_Z + BEAM_OFFSET[2] # We account here for the fact that the sample is at some -z position because it's centered within the tank and the offset
SCALE_PX_PER_MM = 45.06
# Camera distances to for sanity-checking nested dielectrics - uncomment for debug only
# You also need to increase angle_vfov to something like 20 in the main rendering function
#CAMERA_DISTANCE = 180 # See tank edges (slightly zoomed out)
#CAMERA_DISTANCE = 300 # See full tank (top and bottom)
VIEWPORT_Z = CAMERA_DISTANCE - 1 # Viewport position
CAMERA_POSITION = np.array([BEAM_OFFSET[0] - 0.5, CAMERA_HEIGHT, CAMERA_DISTANCE]) # Camera was slightly moved right to center on the beam, too
CAMERA_TARGET = np.array([BEAM_OFFSET[0] - 0.5, CAMERA_HEIGHT, VIEWPORT_Z])


# Convenience enums for accessing the right meshes
class Tank(StrEnum):
    RECTANGLE = "rectangular-box"
    PIPE = "pipe"

class Refinement(StrEnum):
    COARSE = "coarse"
    MED_FINE = "med-fine"
    FINE = "fine"

class TestCase(StrEnum):
    AIR_UNLIT = "air_unlit"
    AIR_DIFFUSE = "air_diffuse"
    TANK = "tank"
    WATER = "water"

class Resolution(IntEnum):
    LOW = 128,
    HIGH = 1024

# ================================================================================
# Helpers
# ================================================================================

def iter_cases():
    # Iterates over test cases above for plotting
    for name, value in vars(TestCase).items():
        if isinstance(value, TestCase):
            yield name, value

def get_tank_path(tank_access: str, element: Element):
    # Tank access is sth like cwd/thesis-data/rectangular-box/coarse
    return full_path(tank_access + "/tank_surface_" + element.label + ".vtk") # full path to e.g., tank_surface_TRI3.vtk

def tank_uv_path(tank_path: Path, element: Element):
    return Path.with_name(tank_path, "tank_" + element.label + "_uvs.csv")

def get_fill_path(tank_access: str, element: Element):
    return full_path(tank_access + "/fill_surface_" + element.label + ".vtk")

def fill_uv_path(water_path: Path, element: Element):
    return Path.with_name(water_path, "fill_" + element.label + "_uvs.csv")

def sample_uv_path(sample_path: Path, element: Element):
    return Path.with_name(sample_path, "beam_" + element.label + "_uvs.csv")

# ================================================================================
# ROI selector
# ================================================================================
@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h
    
def make_display_preview(img: np.ndarray, bit_depth: BitDepth = BitDepth.BIT_12) -> np.ndarray:
    if bit_depth == BitDepth.BIT_12:
        max_value = 4095
    elif bit_depth == BitDepth.BIT_8:
        max_value = 255
    elif bit_depth == BitDepth.BIT_16:
        max_value = 65535
    else:
        max_value = int(img.max()) if img.size else 1

    img_clipped = np.clip(img, 0, max_value)
    preview = ((img_clipped.astype(np.float32) / max_value) * 255.0).astype(np.uint8)
    return preview

def preview_minmax(img: np.ndarray) -> np.ndarray:
    # Best for visibility, but contrast changes image-to-image
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def select_roi_from_path(image_path: Path, bit_depth: BitDepth = BitDepth.BIT_12, window_name: str = "Select ROI") -> ROI:
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    #preview = make_display_preview(img, bit_depth=bit_depth)
    preview = preview_minmax(img)

    x, y, w, h = cv2.selectROI(window_name, preview, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if w == 0 or h == 0:
        raise ValueError("No ROI selected.")

    return ROI(int(x), int(y), int(w), int(h))

def crop_roi(img: np.ndarray, roi: ROI) -> np.ndarray:
    return img[roi.y:roi.y2, roi.x:roi.x2]


def export_roi_mask(image_path: Path, roi: ROI, output_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[roi.y:roi.y + roi.h, roi.x:roi.x + roi.w] = 1

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        np.savetxt(output_path, mask, fmt="%d", delimiter=",")
    elif suffix == ".dat":
        np.savetxt(output_path, mask, fmt="%d")
    else:
        raise ValueError("Output must end in .csv or .dat")

    return mask

def get_roi(test_case: TestCase, resolution: Resolution = Resolution.HIGH, bit_depth: BitDepth = BitDepth.BIT_12):
    base_data_dir = "convergence_rt/res_" + str(resolution.value) + "/" + test_case.value
    data_path = test_dir(BASE_TEST_DIR, base_data_dir)
    image_base_name = "rtimage_subsamples_2.tiff"
    image_path = data_path / "QUAD4" / image_base_name
    csv_filename = "roi_" + str(resolution.value) + "_" + test_case.value + ".csv"
    target_path = data_path /csv_filename 
    roi = select_roi_from_path(image_path, bit_depth=bit_depth)
    roi_data = export_roi_mask(image_path, roi, target_path)
    #print(roi)
    #print(roi_data.shape)

def show_roi_overlay(image_path: Path, roi_path: Path, alpha: float = 0.35):
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    if img.ndim == 3:
        img = img[:, :, 0]

    mask = np.loadtxt(roi_path, delimiter="," if roi_path.suffix.lower() == ".csv" else None, dtype=np.uint8, ndmin=2)

    if mask.shape != img.shape:
        raise ValueError(f"Mask shape {mask.shape} != image shape {img.shape}")

    mask = mask.astype(bool)

    preview = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

    overlay = preview_bgr.copy()
    overlay[mask] = (0, 0, 255)  # red in BGR

    blended = cv2.addWeighted(overlay, alpha, preview_bgr, 1 - alpha, 0)

    ys, xs = np.where(mask)
    if len(xs) > 0 and len(ys) > 0:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("ROI overlay", blended)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#path_4 = full_path("thesis-output/convergence_rt_roitest/res_1024_roi/air_unlit/QUAD8/rtimage_subsamples_4096.tiff")
#roi_path = full_path("thesis-data/roi_1024_air_unlit.csv" )
#show_roi_overlay(path_4, roi_path)

# ================================================================================
# Convergence tester
# ================================================================================

def debug_image_stats(img: np.ndarray, name: str = "img") -> None:
    """
    Helper to see how the image is stored/read for troubleshooting.
    """
    print(
        f"{name}: dtype={img.dtype}, shape={img.shape}, "
        f"min={img.min()}, max={img.max()}, "
        f"unique_low_bits={np.unique(img & 0xF)[:16]}")

def _load_image(path: Path, bit_depth: BitDepth) -> np.ndarray:
    """
    Load a grayscale image as a uint16 array of logical codes.

    For BIT_12 the pyvale C++ writer packs samples tightly (BitsPerSample=12),
    which OpenCV/libtiff decode incorrectly. We use a dedicated unpacker.
    For other depths we fall back to OpenCV (16-bit / 8-bit TIFF, BMP, etc.).
    """
    if bit_depth == BitDepth.BIT_12:
        return read_packed_12bit_tiff(path)

    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if arr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr

def bitwise_compare(data_path_new: Path, data_path_prev: Path | None = None, roi: Path | None = None, bit_depth: BitDepth = BitDepth.BIT_12):
    """
    Checks if two images are bitwise identical, optionally within a ROI mask. If ROI is None, or loading fails for any reason, the whole image is used.

    Handles pyvale's tightly-packed 12-bit TIFFs correctly (OpenCV cannot).
    Assumes both images have the same dimensions.

    If data_path_prev is None, the image is compared to itself (sanity check, should yield 100% similarity and RMSE 0).

    Returns (rmse, similarity_rmse, similarity_identical).
    """
    max_value = {BitDepth.BIT_8: 255,
        BitDepth.BIT_10: 1023,
        BitDepth.BIT_12: 4095,
        BitDepth.BIT_16: 65535}[bit_depth]

    if data_path_prev is None:
        data_path_prev = data_path_new

    pixel_array_new = _load_image(data_path_new, bit_depth)
    pixel_array_prev = _load_image(data_path_prev, bit_depth)

    if pixel_array_new.shape != pixel_array_prev.shape:
        raise ValueError(
            f"Image shapes differ: {pixel_array_new.shape} vs {pixel_array_prev.shape}")

    # ROI mask
    roi_mask = None
    if roi is not None:
        try:
            delimiter = "," if roi.suffix.lower() == ".csv" else None
            roi_mask = np.loadtxt(roi, delimiter=delimiter, dtype=np.uint8)
            if roi_mask.shape != pixel_array_new.shape:
                raise ValueError(
                    f"ROI mask shape does not match image shape: "
                    f"{roi_mask.shape} vs {pixel_array_new.shape}"
                )
            roi_mask = roi_mask.astype(bool)
            if not np.any(roi_mask):
                raise ValueError("ROI mask contains no selected pixels.")
        except Exception as e:
            print(f"Warning: ROI load failed ({e}). Falling back to full image.")
            roi_mask = np.ones(pixel_array_new.shape, dtype=bool)
    else:
        roi_mask = np.ones(pixel_array_new.shape, dtype=bool)

    new_roi = pixel_array_new[roi_mask]
    prev_roi = pixel_array_prev[roi_mask]

    # Metrics (compute on the SAME masked region) 
    # Cast to int32 to avoid uint16 wraparound in the subtraction
    diff = new_roi.astype(np.int32) - prev_roi.astype(np.int32)

    total_pixels = new_roi.size
    different_count = int(np.count_nonzero(diff))
    identical_count = total_pixels - different_count

    rmse = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))
    max_diff = np.max(np.abs(diff))
    percentile_diff = np.percentile(np.abs(diff), 99.9)

    return rmse, max_diff, percentile_diff, identical_count, total_pixels

# Debug area
"""
path_0_c = full_path("thesis-output/convergence_rt/res_1024_roi/air_unlit/QUAD8/256_crop.tiff")
path_1_c = full_path("thesis-output/convergence_rt/res_1024_roi/air_unlit/QUAD8/512_crop.tiff")
path_2_c = full_path("thesis-output/convergence_rt/res_1024_roi/air_unlit/QUAD8/1024_crop.tiff")
path_3_c = full_path("thesis-output/convergence_rt/res_1024_roi/air_unlit/QUAD8/2048_crop.tiff")
path_4_c = full_path("thesis-output/convergence_rt/res_1024_roi/air_unlit/QUAD8/4096_crop.tiff")
print("MANUALLY CROPPED")
rmse, similarity_rmse, similarity_identical= bitwise_compare(path_0_c, path_1_c, bit_depth = BitDepth.BIT_16)
print(rmse)
rmse, similarity_rmse, similarity_identical= bitwise_compare(path_1_c, path_2_c, bit_depth = BitDepth.BIT_16)
print(rmse)
rmse, similarity_rmse, similarity_identical= bitwise_compare(path_2_c, path_3_c, bit_depth = BitDepth.BIT_16)
print(rmse)
rmse, similarity_rmse, similarity_identical= bitwise_compare(path_3_c, path_4_c, bit_depth = BitDepth.BIT_16)
print(rmse)
path_0 = full_path("thesis-output/convergence_rt/res_1024_roi/air_unlit/QUAD8/rtimage_subsamples_256.tiff")
path_1 = full_path("thesis-output/convergence_rt/res_1024_roi/air_unlit/QUAD8/rtimage_subsamples_512.tiff")
path_2 = full_path("thesis-output/convergence_rt/res_1024_roi/air_unlit/QUAD8/rtimage_subsamples_1024.tiff")
path_3 = full_path("thesis-output/convergence_rt/res_1024_roi/air_unlit/QUAD8/rtimage_subsamples_2048.tiff")
path_4 = full_path("thesis-output/convergence_rt/res_1024_roi/air_unlit/QUAD8/rtimage_subsamples_4096.tiff")
roi_path = full_path("thesis-data/roi_1024_air_unlit.csv" )
print("ROI DEFINED, UNCROPPED")
rmse, similarity_rmse, similarity_identical= bitwise_compare(path_0, path_1, roi_path)
print(rmse)
rmse, similarity_rmse, similarity_identical= bitwise_compare(path_1, path_2, roi_path)
print(rmse)
rmse, similarity_rmse, similarity_identical= bitwise_compare(path_2, path_3, roi_path)
print(rmse)
rmse, similarity_rmse, similarity_identical= bitwise_compare(path_3, path_4, roi_path)
print(rmse)
print("NO ROI DEFINED, UNCROPPED")
rmse, similarity_rmse, similarity_identical= bitwise_compare(path_0, path_1)
print(rmse)
rmse, similarity_rmse, similarity_identical= bitwise_compare(path_1, path_2)
print(rmse)
rmse, similarity_rmse, similarity_identical= bitwise_compare(path_2, path_3)
print(rmse)
rmse, similarity_rmse, similarity_identical= bitwise_compare(path_3, path_4)
print(rmse)

path_4 = full_path("thesis-output/convergence_blender/res_1024_roi/air_unlit - original/TRI3/rtimage_subsamples_4096.tiff")
blender_img = _load_image(path_4, BitDepth.BIT_12)
debug_image_stats(blender_img)
"""

# ================================================================================
# Post-processing: Convergence log
# ================================================================================

def fill_convergence_log(element: Element, test_case:TestCase, resolution: Resolution, start_subsamples: int, end_subsamples: int, blender: bool = False):
    """
    Fills the convergence log; useful if the rendering was interrupted or split across machines, etc.
    to get the data in the same csv effortlessly.
    """
    roi_path = None
    # ROI defined only for high res - for low, the entire image is our ROI
    if resolution == Resolution.HIGH and not blender:
        roi_path_access = f"thesis-data/roi_1024_{test_case.value}.csv" 
        roi_path = full_path(roi_path_access)

    base_data_dir = "convergence_rt/res_" + str(resolution.value) + "/" + test_case.value + "/"
    bit_depth = BitDepth.BIT_12
    if blender:
        base_data_dir = "convergence_blender/res_" + str(resolution.value) + "/" + test_case.value + "/"
        bit_depth = BitDepth.BIT_16
    elem_dir_name = base_data_dir + element.label
    data_path = test_dir(BASE_TEST_DIR, elem_dir_name)
    csv_path = data_path / "convergence_log.csv" # Full path to the csv with all numerical data
    image_base_name = "rtimage_subsamples_"
    image_suffix = ".tiff"
    subsamples = start_subsamples * 2 # Assuming we increase the subsample count in powers of 2
    iteration = 1
    prev_filename = image_base_name + str(start_subsamples) + image_suffix
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
            #writer = csv.DictWriter(csvfile, fieldnames=["iteration", "subsamples", "rmse", "sim_score_rmse", "sim_score_identical"])
            writer = csv.DictWriter(csvfile, fieldnames=CONV_CSV_COLS)
            writer.writeheader()
            while subsamples <= end_subsamples:
                current_filename = image_base_name + str(subsamples) + image_suffix
                #rmse, sim_score_rmse, sim_score_identical = bitwise_compare(data_path / current_filename, data_path / prev_filename, roi_path, bit_depth)
                rmse, max_ae, percentile_diff, identical_count, total_pixels = bitwise_compare(data_path / current_filename, data_path / prev_filename, roi_path, bit_depth)
                """
                writer.writerow({
                    "iteration": iteration,
                    "subsamples": subsamples,
                    "rmse": rmse,
                    "sim_score_rmse": sim_score_rmse,
                    "sim_score_identical": sim_score_identical})
                """
                writer.writerow({
                        "iteration": iteration,
                        "subsamples": subsamples,
                        "rmse": rmse,
                        "max_ae": max_ae,
                        "99p_abs_error": percentile_diff,
                        "identical_px_count": identical_count,
                        "tot_px_roi": total_pixels})
                prev_filename = current_filename
                subsamples *= 2
                iteration += 1

# ================================================================================
# Post-processing: Plotters for RT convergence
# ================================================================================

def plot_results_all(test_case: TestCase, resolution: Resolution, save: bool = False, show: bool = False):
    """
    Plots all convergence results on the same plot.
    """
    # Get the address of the directory with the data (assuming we haven't changed it)
    base_data_dir = "convergence_rt/res_" + str(resolution.value) + "/" + test_case.value + "/"
    target_path = test_dir(BASE_TEST_DIR, base_data_dir)
    filename = f"{test_case.value}_{resolution.value}_convergence_plot.png"


    # Create plot
    fig, (ax, ax_bottom) = plt.subplots(2, 1, figsize=FIGURE_SIZE_STACKED_HOR, sharex=True)
    #ax.set_title("Subsampling for high resolution/low resolution", fontsize=FONT_SIZES["suptitle"]) # If you want a title
    ax_bottom.set_xlabel("Subsamples per pixel", fontsize=FONT_SIZES["axis_labels"])
    ax.set_ylabel("RMSE [GL]", fontsize=FONT_SIZES["axis_labels"])
    ax_bottom.set_ylabel("Maximum absolute error [GL]", fontsize=FONT_SIZES["axis_labels"])
    # Format the y-axis as by default it just shows orders of magnitude
    ax.yaxis.set_major_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3g"))
    ax.tick_params(axis="y", which="minor", labelsize=FONT_SIZES["ticks"]) # Set label sizes on the axis
    ax_bottom.tick_params(axis="y", which="both", labelsize=FONT_SIZES["ticks"]) # Set label sizes on the axis
    # Format x-axis as well
    ax.set_xscale("log")
    ax_bottom.set_xscale("log")
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax_bottom.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="x", which="both", labelsize=FONT_SIZES["ticks"])
    ax_bottom.tick_params(axis="x", which="both", labelsize=FONT_SIZES["ticks"])
    label_x = None
    min_x = np.inf
    max_x = -np.inf
    title = f"Convergence for test case: {test_case.value} at {resolution.value}x{resolution.value} px resolution"
    ax.set_title(title, fontsize=FONT_SIZES["suptitle"])
    # Iterate over elements
    for name, element in iter_elements():
        elem_dir_name = base_data_dir + element.label  
        data_path = test_dir(BASE_TEST_DIR, elem_dir_name) / "convergence_log.csv" # Full path to the csv with all numerical data
        #print(data_path)
        # Convergence stores data as ["iteration", "subsamples", "rmse", "max_ae", "99p_abs_error", "identical_px_count", "tot_px_roi"]
        elem_data = np.loadtxt(data_path, delimiter=",", skiprows=1, unpack=True) # Full data
        all_x = np.unique(elem_data[1])
        curr_min_x = np.min(all_x)
        curr_max_x = np.max(all_x)
        # Handle x-labels in case some elements have different number of samples
        if label_x is None: # First element
            label_x = all_x
            min_x = curr_min_x
            max_x = curr_max_x
        else:
            if curr_min_x < min_x:
                min_x = curr_min_x
            if curr_max_x > max_x:
                max_x = curr_max_x
            # Extend x-labels to include all range of x-values across the dataset
            label_x = np.union1d(label_x, all_x)
        ax.plot(elem_data[1], elem_data[2],
                    label=name,
                    color=element.color,
                    marker="o",
                    linestyle="-",
                    linewidth=3,
                    markersize=10)
        ax_bottom.plot(elem_data[1], elem_data[3],
                    color=element.color,
                    marker="o",
                    linestyle="-",
                    linewidth=3,
                    markersize=10)
    ax_bottom.set_xticks(label_x)
    # Plot x-labels as 2^power for clarity
    ax_bottom.set_xticklabels([rf"$2^{{{int(np.log2(x))}}}$" for x in label_x])
    ax.legend(loc='upper right', fontsize=FONT_SIZES["axis_labels"])
    ax.grid(visible=True, which='both', axis='both')
    ax_bottom.grid(visible=True, which='both', axis='both')
    plt.tight_layout()


    if show:
     plt.show()
    if save:
        fig.savefig(Path.joinpath(target_path, filename), dpi=300, bbox_inches="tight")

def plot_results_subplots(test_case: TestCase, resolution: Resolution, rmse: bool = False, save: bool = False, show: bool = False):
    """
    Plots convergence results for all elements on separate subplots in one figure.

    """
    base_data_dir = f"convergence_rt/res_{resolution.value}/{test_case.value}/"
    target_path = test_dir(BASE_TEST_DIR, base_data_dir)
    filename = f"{test_case.value}_{resolution.value}_convergence_subplots"
    if rmse:
        data_index = 2 # Index corresponding to this data in elem_data
        y_label = "RMSE [GL]"
        filename = filename + "_rmse.png"
    else:
        data_index = 3
        y_label = "Maximum absolute error [GL]"
        filename = filename + "_maxae.png"

    # Define subplot layout
    # Elements are stacked neatly by type (triangles/quads), then on the left we have "usual" elements and higher order ones on the right
    fig, axes = plt.subplot_mosaic(
        [["TRI3", "TRI6"],
            ["QUAD4","QUAD8"],
            [".",  "QUAD9"],],
        figsize=(14, 14),
        constrained_layout=True)

    fig.suptitle(f"Convergence for test case: {test_case.value} at {resolution.value} px resolution", fontsize=FONT_SIZES["suptitle"])

    for name, element in iter_elements_plot_order():
        ax = axes[name]

        elem_dir_name = base_data_dir + element.label
        data_path = test_dir(BASE_TEST_DIR, elem_dir_name) / "convergence_log.csv"

        # Convergence stores data as [iteration, subsamples, rmse, sim_score_rmse, sim_score_identical]
        elem_data = np.loadtxt(data_path, delimiter=",", skiprows=1, unpack=True)

        # Values for ticks so we only display values from actual data
        x_data = elem_data[1]
        y_data = elem_data[data_index]

        all_x = np.unique(elem_data[1])

        ax.plot(x_data,y_data,
            color=element.color,
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=5)
            
        ax.set_title(name, fontsize=FONT_SIZES["subtitle"])
        ax.set_xlabel("Subsamples per pixel", fontsize=FONT_SIZES["subplot_labels"])
        ax.set_ylabel(y_label, fontsize=FONT_SIZES["subplot_labels"])

        # Y-axis formatting
        ax.tick_params(axis="y", which="both", labelsize=FONT_SIZES["ticks"])
        all_y = np.unique(y_data)
        # Small dataset (usually 131+k samples) => We can display actual RMSE values on the plot
        if all_y.shape[0] < 8:
            ax.set_yticks(all_y)
        # Big dataset (usually starting at 1 sample) => Matplotlib doesn't like it => Leave default major ticks
        # But point at min and max values as otherwise ~0 looks like exact 0 etc.
        else:
            # Min RMSE/MaxAE - important for big datasets
            y_last = np.min(y_data)
            idx_last = np.argmin(y_data) # Index of first occurrence of that minimum
            x_last = x_data[idx_last]
            annotation_min = " "
            if rmse:
                maxae_here = elem_data[3][idx_last] # MaxAE at this point
                annotation_min = f"RMSE: {y_last:.3g}\nMaxAE: {maxae_here:.3g}"
            else:
                rmse_here = elem_data[2][idx_last] # RMSE at this point
                annotation_min = f"MaxAE: {y_last:.3g}\nRMSE: {rmse_here:.3g}"
            # Display y-value above the marker in a box with arrow
            ax.annotate(annotation_min,
                xy=(x_last, y_last),
                xytext=(-1, 100), # Offset up vertically above the marker
                textcoords="offset pixels",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZES["subplot_labels"]-1,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=element.color, lw=1), # Box storing the text
                arrowprops=dict(arrowstyle="->", color=element.color, lw=2, shrinkA=0, shrinkB=0),
                annotation_clip=True)

            x_first = x_data[0]
            y_first = y_data[0]
            annotation_first = " "
            if rmse:
                maxae_here = elem_data[3][0] # MaxAE at this point
                annotation_first = f"RMSE: {y_first:.3g}\nMaxAE: {maxae_here:.3g}"
            else:
                rmse_here = elem_data[2][0] # RMSE at this point
                annotation_first = f"MaxAE: {y_first:.3g}\nRMSE: {rmse_here:.3g}"

            # Display y-value below the marker in a box with arrow
            ax.annotate(annotation_first,
                xy=(x_first, y_first), #(horizontal_offset, vertical_offset)
                xytext=(275, -55), # Offset up vertically above the marker
                textcoords="offset pixels",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZES["subplot_labels"]-1,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=element.color, lw=1), # Box storing the text
                arrowprops=dict(arrowstyle="->", color=element.color, lw=2, shrinkA=0, shrinkB=0),
                annotation_clip=True)

        # X-axis formatting
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(axis="x", which="both", labelsize=FONT_SIZES["ticks"])

        ax.set_xticks(all_x)
        ax.set_xticklabels([rf"$2^{{{int(np.log2(x))}}}$" for x in all_x])

        ax.grid(visible=True, which="major", axis="both")
    if show:
        plt.show()
    if save:
        fig.savefig(Path.joinpath(target_path, filename), dpi=300, bbox_inches="tight")

def difference_image(data_path_higher: Path, data_path_lower: Path, label: str, bit_depth: BitDepth = BitDepth.BIT_12):
    """
    Finds difference between images (higher = higher subsample count; lower = lower subsample count).
    """
    max_value = 4095 # Max integer value for 12-bit uint; assign by default
    if bit_depth == BitDepth.BIT_8:
        max_value = 255 # Max integer value for 8-bit uint
    elif bit_depth == BitDepth.BIT_16:
        max_value = 65535 # Max integer value for 16-bit uint

    # cv2.IMREAD_ANYDEPTH forces OpenCV to keep the 16-bit depth instead of downsampling it to 8-bit
    img_a = cv2.imread(str(data_path_higher), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    img_b = cv2.imread(str(data_path_lower), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)

    # If it loaded as a 3-channel image, grab just the first channel
    if len(img_a .shape) == 3:
        img_a  = img_a [:, :, 0]
    if len(img_b.shape) == 3:
        img_b = img_b[:, :, 0]

    # 1. Raw difference between the two images
    difference = cv2.absdiff(img_a, img_b).astype(np.uint16)
    output_dir = data_path_higher.parent
    output_name = f"difference_raw_{label}.tiff"
    raw_path = output_dir / output_name

    ok = cv2.imwrite(str(raw_path), difference)
    if not ok:
        raise IOError(f"Could not write output image: {raw_path}")
    
    # 2. Save stretched 8-bit visualization for human inspection
    diff_float = difference.astype(np.float32)

    dmax = diff_float.max()
    if dmax > 0:
        diff_vis = diff_float * (max_value / dmax)
    else:
        diff_vis = diff_float.copy()

    diff_vis_8 = np.clip(diff_vis * (255.0 / max_value), 0, 255).astype(np.uint8)

    vis_path = output_dir / f"difference_vis_{label}.png"
    ok = cv2.imwrite(str(vis_path), diff_vis_8)
    if not ok:
        raise IOError(f"Could not write visualized difference image: {vis_path}")

   # 3. Optional: binary mask of changed pixels
    threshold_value = 2
    mask = (difference > threshold_value).astype(np.uint8) * 255

    # Make changed pixels easier to see
    dilate_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    mask_vis = cv2.dilate(mask, kernel, iterations=1)

    mask_path = output_dir / f"difference_mask_{label}.png"
    ok = cv2.imwrite(str(mask_path), mask_vis)
    if not ok:
        raise IOError(f"Could not write mask image: {mask_path}")

    # 4. Overlay
    base_8 = np.clip(img_a.astype(np.float32) * (255.0 / max_value), 0, 255).astype(np.uint8)
    overlay = cv2.cvtColor(base_8, cv2.COLOR_GRAY2BGR)

    # Paint dilated mask pink (matching historgram), not the raw exact-difference pixels
    overlay[mask_vis == 255] = (194, 119, 227) # BGR, not RGB

    overlay_path = output_dir / f"difference_overlay_{label}.png"
    ok = cv2.imwrite(str(overlay_path), overlay)
    if not ok:
        raise IOError(f"Could not write overlay image: {overlay_path}")
    
    # 5. Histogram of absolute differences
    hist_path = output_dir / f"difference_hist_{label}.png"

    diff_1d = difference.ravel()
    bins = np.arange(0, int(diff_1d.max()) + 2) - 0.5

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(diff_1d, bins=bins, color="tab:pink", edgecolor="black")
    ax.set_title("Histogram of absolute pixel differences")
    ax.set_xlabel("Absolute pixel difference")
    ax.set_ylabel("Pixel count")

    # Uncomment if most pixels are identical and the zero bin dominates:
    # ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(hist_path, bbox_inches="tight")
    plt.close(fig)

    return raw_path, vis_path, mask_path

def difference_heatmap(data_path_higher: Path,
    data_path_lower: Path,
    label: str,
    bit_depth: BitDepth = BitDepth.BIT_12,
    vmax: float | None = None):
    """
    Plots difference heatmap.
    """
    max_value = 4095
    if bit_depth == BitDepth.BIT_8:
        max_value = 255
    elif bit_depth == BitDepth.BIT_16:
        max_value = 65535

    img1 = cv2.imread(str(data_path_higher), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    img2 = cv2.imread(str(data_path_lower), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)

    if len(img1.shape) == 3:
        img1 = img1[:, :, 0]
    if len(img2.shape) == 3:
        img2 = img2[:, :, 0]

    # Cast before subtracting
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    diff_norm = diff / max_value

    output_dir = data_path_higher.parent
    output_path = output_dir / f"difference_heatmap_{label}.png"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    im = ax.imshow(diff_norm, cmap="magma", vmin=0, vmax=0.01, origin="upper")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Absolute pixel difference (fraction of full scale)")

    ax.set_title("Per-pixel absolute difference")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return output_path

def check_difference(element: Element, test_case:TestCase, resolution: Resolution, start_subsamples: int, end_subsamples: int):
    """
    Plots all difference plots for the given two images.

    Useful for checking which pixels are different, etc.
    """
    base_data_dir = "convergence_rt/res_" + str(resolution.value) + "/" + test_case.value + "/"
    elem_dir_name = base_data_dir + element.label  
    data_path = test_dir(BASE_TEST_DIR, elem_dir_name)
    data_path_higher = data_path / ("rtimage_subsamples_" + str(end_subsamples) + ".tiff")
    data_path_lower = data_path / ("rtimage_subsamples_" + str(start_subsamples) + ".tiff")

    label = str(end_subsamples) + "_" + str(start_subsamples)
    difference_image(data_path_higher, data_path_lower, label)
    difference_heatmap(data_path_higher, data_path_lower, label)

# ================================================================================
# Post-processing: Plotters for Blender/ Blender vs RT comparison
# ================================================================================
def plot_results_blender(test_case: TestCase, resolution: Resolution, time: bool = False, save: bool = False, show: bool = False):
    """
    Plots Blender convergence plot with RMSE on the left and render time on the right.
    """
    # All access paths
    base_data_dir = "convergence_blender/res_" + str(resolution.value) + "/" + test_case.value + "/"
    elem_dir_name = base_data_dir + Elements.TRI3.label
    target_path = test_dir(BASE_TEST_DIR, base_data_dir)
    data_path = test_dir(BASE_TEST_DIR, elem_dir_name) / "convergence_log.csv"
    time_data_path = test_dir(BASE_TEST_DIR, elem_dir_name) / "cpu_render_time_log.csv"
    filename = f"{test_case.value}_{resolution.value}_blender_convergence_plot.png"

    # Load data
    # Convergence data: [iteration, subsamples, rmse, sim_score_rmse, sim_score_identical]
    elem_data = np.loadtxt(data_path, delimiter=",", skiprows=1, unpack=True)

    # Render time data: [subsamples, time(s)]
    time_data = np.loadtxt(time_data_path, delimiter=",", skiprows=1, unpack=True)

    label_x = np.unique(elem_data[1])

    title = f"Blender convergence for test case: {test_case.value} at {resolution.value}x{resolution.value} px resolution"

    # Create two side-by-side plots with shared x-axis if plotting CPU render time
    if time:
        fig, (ax_rmse, ax_time) = plt.subplots(1, 2, figsize=FIGURE_SIZE, sharex=True)
        fig.suptitle(title, fontsize=FONT_SIZES["suptitle"])

        # ------------------
        # Left plot: RMSE
        # ------------------
        ax_rmse.set_xlabel("Subsamples per pixel", fontsize=FONT_SIZES["axis_labels"])
        ax_rmse.set_ylabel("RMSE [GL]", fontsize=FONT_SIZES["axis_labels"])

        ax_rmse.yaxis.set_major_formatter(mticker.NullFormatter())
        ax_rmse.yaxis.set_minor_formatter(FormatStrFormatter("%.3g"))
        ax_rmse.tick_params(axis="y", which="minor", labelsize=FONT_SIZES["ticks"])
        ax_rmse.tick_params(axis="x", which="both", labelsize=FONT_SIZES["ticks"])

        ax_rmse.set_xscale("log")
        ax_rmse.xaxis.set_minor_locator(mticker.NullLocator())

        ax_rmse.plot(
            elem_data[1], elem_data[2],
            label=Elements.TRI3.label,
            color=Elements.TRI3.color,
            marker="o",
            linestyle="-",
            linewidth=3,
            markersize=10)

        ax_rmse.set_xticks(label_x)
        ax_rmse.set_xticklabels([rf"$2^{{{int(np.log2(x))}}}$" for x in label_x])
        ax_rmse.legend(loc="upper right", fontsize=FONT_SIZES["axis_labels"])
        ax_rmse.grid(visible=True, which="both", axis="both")

        # ------------------
        # Right plot: Render time
        # ------------------
        ax_time.set_xlabel("Subsamples per pixel", fontsize=FONT_SIZES["axis_labels"])
        ax_time.set_ylabel("Render time [s]", fontsize=FONT_SIZES["axis_labels"])
        ax_time.tick_params(axis="both", which="both", labelsize=FONT_SIZES["ticks"])

        ax_time.set_xscale("log")
        ax_time.xaxis.set_minor_locator(mticker.NullLocator())

        ax_time.plot(
            time_data[0], time_data[1],
            label="CPU render time",
            color="tab:red",
            marker="o",
            linestyle="-",
            linewidth=3,
            markersize=10)

        ax_time.set_xticks(label_x)
        ax_time.set_xticklabels([rf"$2^{{{int(np.log2(x))}}}$" for x in label_x])
        ax_time.legend(loc="upper left", fontsize=FONT_SIZES["axis_labels"])
        ax_time.grid(visible=True, which="both", axis="both")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
    # Just convergence
    else:
        fig, ax_rmse = plt.subplots(figsize=FIGURE_SIZE)
        fig.suptitle(title, fontsize=FONT_SIZES["suptitle"])

        # ------------------
        # Left plot: RMSE
        # ------------------
        ax_rmse.set_xlabel("Subsamples per pixel", fontsize=FONT_SIZES["axis_labels"])
        ax_rmse.set_ylabel("RMSE [GL]", fontsize=FONT_SIZES["axis_labels"])

        ax_rmse.yaxis.set_major_formatter(mticker.NullFormatter())
        ax_rmse.yaxis.set_minor_formatter(FormatStrFormatter("%.3g"))
        ax_rmse.tick_params(axis="y", which="minor", labelsize=FONT_SIZES["ticks"])
        ax_rmse.tick_params(axis="x", which="both", labelsize=FONT_SIZES["ticks"])

        ax_rmse.set_xscale("log")
        ax_rmse.xaxis.set_minor_locator(mticker.NullLocator())

        ax_rmse.plot(
            elem_data[1], elem_data[2],
            label=Elements.TRI3.label,
            color=Elements.TRI3.color,
            marker="o",
            linestyle="-",
            linewidth=3,
            markersize=10)

        ax_rmse.set_xticks(label_x)
        ax_rmse.set_xticklabels([rf"$2^{{{int(np.log2(x))}}}$" for x in label_x])
        ax_rmse.legend(loc="upper right", fontsize=FONT_SIZES["axis_labels"])
        ax_rmse.grid(visible=True, which="both", axis="both")

        plt.tight_layout()

    if show:
        plt.show()
    if save:
        fig.savefig(Path.joinpath(target_path, filename), dpi=300, bbox_inches="tight")

def format_spp_as_power_of_2(spp):
    return rf"$2^{{{int(np.log2(spp))}}}$"

def plot_results_blender_rt_single(test_case: TestCase, resolution: Resolution, save: bool = False,
    show: bool = False, as_percent: bool = False, rmse: bool = False):
    """
    Plots ray tracer and Blender convergence on a single plot, using
    NRMSE (RMSE normalized by each source's full-scale maximum) so the
    12-bit ray tracer and 16-bit Blender curves share a common y-axis.

    The convergence logs store RMSE in native code units:
        - Ray tracer : 12-bit codes -> divide by 4095
        - Blender    : 16-bit codes -> divide by 65535

    Parameters
    ----------
    as_percent : if True, plot NRMSE as a percentage of full scale (x100).
    rmse: whether to plot RMSE or MaxAE. If rmse, do not apply as_percent.
    """
    # Full-scale maxima for normalization
    RT_MAX = 4095.0        # 12-bit ray tracer
    BLENDER_MAX = 65535.0  # 16-bit Blender

    # Output directory
    base_data_dir_b = "convergence_blender/res_" + str(resolution.value) + "/" + test_case.value + "/"
    target_path = test_dir(BASE_TEST_DIR, base_data_dir_b)
    suffix = "_pct" if as_percent else ""
    filename = f"{test_case.value}_{resolution.value}_blender_rt_single_convergence_plot{suffix}.png"

    # Blender data path
    elem_dir_name_b = base_data_dir_b + Elements.TRI3.label
    data_path_b = test_dir(BASE_TEST_DIR, elem_dir_name_b) / "convergence_log.csv"

    # Ray tracer data path
    base_data_dir_rt = "convergence_rt/res_" + str(resolution.value) + "/" + test_case.value + "/"
    elem_dir_name_rt = base_data_dir_rt + Elements.TRI3.label
    data_path_rt = test_dir(BASE_TEST_DIR, elem_dir_name_rt) / "convergence_log.csv"

    # Colors
    rt_color = Elements.TRI3.color
    blender_color = Elements.TRI6.color

    # Load data
    # Convergence data: [iteration, subsamples, rmse, sim_score_rmse, sim_score_identical]
    elem_data_b = np.loadtxt(data_path_b, delimiter=",", skiprows=1, unpack=True)
    elem_data_rt = np.loadtxt(data_path_rt, delimiter=",", skiprows=1, unpack=True)

    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    if rmse:
        # Normalize RMSE (column 2) to NRMSE in [0, 1] (or %)
        data_index = 2 # Position of data in elem_data
        scale = 100.0 if as_percent else 1.0
        elem_data_rt[data_index] = elem_data_rt[data_index] / RT_MAX * scale
        elem_data_b[data_index] = elem_data_b[data_index] / BLENDER_MAX * scale
        y_unit = "%" if as_percent else "fraction of full scale"
        ax.set_ylabel(f"NRMSE [{y_unit}]", fontsize=FONT_SIZES["axis_labels"])
    else:
        data_index = 3
        ax.set_ylabel(f"Maximum absolute error [GL]", fontsize=FONT_SIZES["axis_labels"])
        # No scaling here (we care if it's within 1 bit or not), so display ceiling instead
        #ax.axhline(y=BLENDER_MAX, color=blender_color, linestyle="--", linewidth=2)
        #ax.axhline(y=RT_MAX, color=rt_color, linestyle="--", linewidth=2)
        
    

    # Shared x ticks
    label_x = np.unique(np.concatenate((elem_data_rt[1], elem_data_b[1])))

    # Minimum NRMSE values
    min_val_rt = np.min(elem_data_rt[data_index])
    min_val_b = np.min(elem_data_b[data_index ])

    # Also report where the minima occur
    min_idx_rt = np.argmin(elem_data_rt[data_index])
    min_idx_b = np.argmin(elem_data_b[data_index])
    min_x_rt = elem_data_rt[1][min_idx_rt]
    min_x_b = elem_data_b[1][min_idx_b]

    title = (f"Ray tracer and Blender convergence for TRI3\n"
        f"Test case: {test_case.value} at {resolution.value}x{resolution.value} px resolution")
    ax.set_title(title, fontsize=FONT_SIZES["suptitle"])

    ax.set_xlabel("Subsamples per pixel", fontsize=FONT_SIZES["axis_labels"])
    
    ax.yaxis.set_major_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3g"))
    ax.tick_params(axis="y", which="minor", labelsize=FONT_SIZES["ticks"])

    # X-axis formatting
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="x", which="both", labelsize=FONT_SIZES["ticks"])

    # Ray tracer
    ax.plot(elem_data_rt[1], elem_data_rt[data_index],
        label="Ray tracer",
        color=rt_color,
        marker="o",
        linestyle="-",
        linewidth=3,
        markersize=10)

    # Blender
    ax.plot(elem_data_b[1], elem_data_b[data_index],
        label="Blender Cycles",
        color=blender_color,
        marker="s",
        linestyle="-",
        linewidth=3,
        markersize=10)

    ax.set_xticks(label_x)
    ax.set_xticklabels([rf"$2^{{{int(np.log2(x))}}}$" for x in label_x])

    ax.legend(fontsize=FONT_SIZES["legend"])
    ax.grid(visible=True, which="both", axis="both")

    # Report the minimum values on the plot
    unit_str = "%" if as_percent and rmse else ""
    quantity_str = "Min NRMSE:" if rmse else "Min MaxAE:"

    line1 = TextArea(quantity_str,
        textprops=dict(color="black", fontsize=FONT_SIZES["legend"], ha="left"))

    line2 = HPacker(
        children=[
            TextArea("Ray tracer: ", textprops=dict(color=rt_color, fontsize=FONT_SIZES["legend"])),
            TextArea(f"{min_val_rt:.3g}{unit_str} at {format_spp_as_power_of_2(min_x_rt)} spp",
                     textprops=dict(color="black", fontsize=FONT_SIZES["legend"])),
        ],
        align="left", pad=0, sep=0)

    line3 = HPacker(
        children=[
            TextArea("Blender: ", textprops=dict(color=blender_color, fontsize=FONT_SIZES["legend"])),
            TextArea(f"{min_val_b:.3g}{unit_str} at {format_spp_as_power_of_2(min_x_b)} spp",
                     textprops=dict(color="black", fontsize=FONT_SIZES["legend"])),
        ],
        align="left", pad=0, sep=0)

    stats_box = VPacker(children=[line1, line2, line3], align="left", pad=0, sep=2)

    anchored_box = AnchoredOffsetbox(
        loc="upper right",
        child=stats_box,
        pad=0.3,
        frameon=True,
        bbox_to_anchor=(1.0, 0.9),
        bbox_transform=ax.transAxes,
        borderpad=0.6)

    anchored_box.patch.set_boxstyle("round")
    anchored_box.patch.set_facecolor("white")
    anchored_box.patch.set_alpha(0.85)
    anchored_box.patch.set_edgecolor("gray")

    ax.add_artist(anchored_box)

    plt.tight_layout()

    if show:
        plt.show()
    if save:
        fig.savefig(Path.joinpath(target_path, filename), dpi=300, bbox_inches="tight")


# ================================================================================
# Relative error distributions
# ================================================================================

def plot_relative_error_distribution(
    test_case: TestCase,
    resolution: Resolution,
    subsamples_1: int,
    subsamples_2: int,
    save: bool = False,
    show: bool = False,
    blender: bool = False,
    element: Elements = Elements.TRI3):
    """
    Plots relative pixel-wise error between two rendered images and show pixel coordinates on the axes.
    """
    subsamples_1 = str(subsamples_1)
    subsamples_2 = str(subsamples_2)

    base_data_dir = f"thesis-output/convergence_rt/res_{resolution.value}/{test_case.value}/"
    base_image_file = "/rtimage_subsamples_"

    if blender:
        base_data_dir = f"thesis-output/convergence_blender/res_{resolution.value}/{test_case.value}/"
        element = Elements.TRI3

    elem_dir_name = base_data_dir + element.label

    img1_path = full_path(elem_dir_name + base_image_file + subsamples_1 + ".tiff")
    img2_path = full_path(elem_dir_name + base_image_file + subsamples_2 + ".tiff")

    dynamic_range = 4095
    k = 10

    img1 = np.array(Image.open(str(img1_path)))
    img2 = np.array(Image.open(str(img2_path)))

    print("Shape1:", img1.shape)
    print("Min/Max1:", img1.min(), img1.max())
    print("Data type1:", img1.dtype)

    print("Shape2:", img2.shape)
    print("Min/Max2:", img2.min(), img2.max())
    print("Data type2:", img2.dtype)

    error = (img1.astype(np.int32) - img2.astype(np.int32)) * 100.0 / dynamic_range

    print("Error:", error.min(), error.max())

    max_val = np.max(error)
    max_positions = np.argwhere(error == max_val)
    print("Max value:", max_val)
    print("Locations:", max_positions[:10])

    flat_idx = np.argsort(error.ravel())[-k:]
    top_positions = np.array(np.unravel_index(flat_idx, error.shape)).T

    h, w = error.shape[:2]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    title = (f"Relative error for test case {test_case.value} at {resolution.value}x{resolution.value} px\n"
        f"Between {subsamples_1} and {subsamples_2} subsamples for {element.label}")
    if blender:
        title += " (Blender)"

    ax.set_title(title, fontsize=FONT_SIZES["suptitle"])

    im = ax.imshow(error, cmap="viridis", vmin=error.min(),vmax=error.max(),origin="upper",)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Relative error [%]")

    for (i, j) in top_positions:
        ax.scatter(j, i, color="red", s=50)
        ax.text(j, i, f"({j}, {i})\n{error[i, j]:.2f}", color="white", fontsize=9, ha="left",va="bottom")

    ax.set_xlabel("x pixel position")
    ax.set_ylabel("y pixel position")

    tick_step = max(1, resolution.value // 8)
    ax.set_xticks(np.arange(0, w, tick_step))
    ax.set_yticks(np.arange(0, h, tick_step))

    ax.tick_params(axis="x", rotation=45)

    ax.grid(False)
    fig.tight_layout()

    if save:
        filename = f"{test_case.value}_{resolution.value}_error_distr_{subsamples_1}_{subsamples_2}.png"
        save_path = full_path(elem_dir_name + "/" + filename)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

# ================================================================================
# Performance patch and NEE results
# ================================================================================
def get_patch_run_configs(test_case: TestCase, resolution: Resolution):
    """
    Returns the run configurations that are valid for the given test case/resolution.
    Each config contains folder, display label, and plot color.
    """
    all_runs = [
        {"folder": "thesis-output-bigpatch",
            "label": "Big perf patch",
            "color": "tab:blue"},
        {"folder": "thesis-output-baseline",
            "label": "Unmodified",
            "color": "tab:orange"},
        {"folder": "thesis-output-noneebutvarreduction",
            "label": "Variance reduction (no NEE)",
            "color": "tab:green"},
        {"folder": "thesis-output-smallpatch",
            "label": "Small perf patch",
            "color": "tab:red"},
        {"folder": "thesis-output-smallpatch-noneebutvarreduction",
            "label": "Small perf patch + variance reduction",
            "color": "tab:purple"}
    ]

    return [run for run in all_runs if run_is_applicable(run["folder"], test_case, resolution)]


def run_is_applicable(run_folder: str, test_case: TestCase, resolution: Resolution) -> bool:
    """
    Encodes availability rules from the comment block.
    """
    # HIGH/WATER exists for everything
    if test_case == TestCase.WATER and resolution == Resolution.HIGH:
        return True

    if test_case == TestCase.TANK and resolution == Resolution.LOW and run_folder == "thesis-output-bigpatch":
        return True

    # LOW/HIGH with AIR_DIFFUSE, AIR_UNLIT, TANK:
    # available for everything except bigpatch and noneebutvarreduction
    restricted_cases = {TestCase.AIR_DIFFUSE, TestCase.AIR_UNLIT, TestCase.TANK}
    restricted_resolutions = {Resolution.LOW, Resolution.HIGH}
    excluded_runs = {"thesis-output-bigpatch","thesis-output-noneebutvarreduction"}

    if test_case in restricted_cases and resolution in restricted_resolutions:
        return run_folder not in excluded_runs

    return False

def plot_results_subplots_patch(test_case: TestCase,
    resolution: Resolution,
    plot_time: bool = True,
    save: bool = False,
    show: bool = False,):
    """
    Plots render-time or convergence results for all element types on a subplot mosaic.
    Each subplot contains one line per run configuration.
    """
    patch_results_dir = Path(__file__).resolve().parent.parent / "patch_results"
    if not patch_results_dir.is_dir():
        patch_results_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{test_case.value}_{resolution.value}_rtime_subplots.png"
    suptitle = f"Render time for test case: {test_case.value} at {resolution.value} px resolution"
    csv_name = "render_time_log.csv"
    y_label = "Time [s]"

    if not plot_time:
        filename = f"{test_case.value}_{resolution.value}_rconv_subplots.png"
        suptitle = f"Convergence for test case: {test_case.value} at {resolution.value} px resolution"
        csv_name = "convergence_log.csv"
        y_label = "RMSE [GL]"

    fig, axes = plt.subplot_mosaic(
        [
            ["TRI3", "TRI6"],
            ["QUAD4", "QUAD8"],
            [".", "QUAD9"],
        ],
        figsize=(14, 14),
        constrained_layout=True)

    fig.suptitle(suptitle, fontsize=FONT_SIZES["suptitle"])

    run_configs = get_patch_run_configs(test_case, resolution)

    legend_handles = []
    legend_labels = []

    for name, element in iter_elements_plot_order():
        ax = axes[name]
        ax.set_title(name, fontsize=FONT_SIZES["subtitle"])
        ax.set_xlabel("Subsamples per pixel", fontsize=FONT_SIZES["subplot_labels"])
        ax.set_ylabel(y_label, fontsize=FONT_SIZES["subplot_labels"])

        plotted_any = False
        x_ticks_seen = set()
        y_values_seen = []

        for run in run_configs:
            case_base = f"convergence_rt/res_{resolution.value}/{test_case.value}"
            base_data_dir = Path(__file__).resolve().parent.parent / run["folder"] / case_base
            elem_dir_name = base_data_dir / element.label
            data_path = test_dir(BASE_TEST_DIR, elem_dir_name) / csv_name

            if not data_path.is_file():
                continue

            elem_data = np.loadtxt(data_path, delimiter=",", skiprows=1, unpack=True)

            if plot_time:
                # render_time_log.csv -> [subsamples, time]
                x_data = elem_data[0]
                y_data = elem_data[1]
            else:
                # convergence_log.csv -> [iteration, subsamples, rmse, sim_score_rmse, sim_score_identical]
                x_data = elem_data[1]
                y_data = elem_data[2]

            line, = ax.plot(
                x_data,
                y_data,
                color=run["color"],
                marker="o",
                linestyle="-",
                linewidth=2,
                markersize=5,
                label=run["label"])

            plotted_any = True
            x_ticks_seen.update(np.unique(x_data))
            y_values_seen.extend(np.unique(y_data))

            if run["label"] not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(run["label"])

        if not plotted_any:
            ax.set_visible(False)
            continue

        # Y-axis formatting
        ax.tick_params(axis="y", which="both", labelsize=FONT_SIZES["ticks"])
        all_y = np.unique(y_values_seen)
        if all_y.shape[0] < 8:
            ax.set_yticks(all_y)

        # X-axis formatting
        all_x = np.array(sorted(x_ticks_seen))
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(axis="x", which="both", labelsize=FONT_SIZES["ticks"])
        ax.set_xticks(all_x)
        ax.set_xticklabels([rf"$2^{{{int(np.log2(x))}}}$" for x in all_x])

        ax.grid(visible=True, which="major", axis="both")

    if legend_handles:
        fig.legend(
        legend_handles,
        legend_labels,
        loc="lower left",
        bbox_to_anchor=(0.08, 0.08),
        ncol=1,
        fontsize=FONT_SIZES["ticks"],
        frameon=True)

    if show:
        plt.show()

    if save:
        fig.savefig(patch_results_dir / filename, dpi=300, bbox_inches="tight")

# ================================================================================
# Convenience for data plotting/updating in one go
# ================================================================================

def _get_min_max_subsamples(elem_dir: Path, base_prefix: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Gets min/max subsample counts in a given folder based on the "rtimage_subsamples_SUBSAMPLES.tiff" filename pattern.
    """
    if not elem_dir.exists():
        return None, None

    counts = [int(file.stem.removeprefix(base_prefix)) for file in elem_dir.glob(f"{base_prefix}*.tiff")
        if file.stem.removeprefix(base_prefix).isdigit()]

    if not counts:
        return None, None

    return min(counts), max(counts)

def fill_all_convergence_logs(test_case: TestCase, resolution: Resolution, blender: bool = False):
    # Need to detect these from image names
    base_image_file = "rtimage_subsamples_"

    if blender:
        base_data_dir = f"thesis-output/convergence_blender/res_{resolution.value}/{test_case.value}/"
        elem_path = full_path(base_data_dir + Elements.TRI3.label)
        subsamples_min, subsamples_max = _get_min_max_subsamples(elem_path, base_image_file)
        fill_convergence_log(Elements.TRI3, test_case, resolution, subsamples_min, subsamples_max, True)
        try:
            plot_results_blender(test_case, resolution, False, True, False)
        except Exception as e: # Likely missing enough data for plots, so we just skip them
            print(f"Error plotting the results: {e}.\nLikely from missing sufficient data for some elements. Plotting skipped.")
        return
    
    base_data_dir = f"thesis-output/convergence_rt/res_{resolution.value}/{test_case.value}/"
    for name, element in iter_elements():
        # Detect the min/max subsamples in
        elem_path = full_path(base_data_dir + element.label)
        subsamples_min, subsamples_max = _get_min_max_subsamples(elem_path, base_image_file)
        fill_convergence_log(element, test_case, resolution, subsamples_min, subsamples_max, False)
    try:
        plot_results_all(test_case, resolution, True, False)
        plot_results_subplots(test_case, resolution, True, True, False) # RMSE plot
        plot_results_subplots(test_case, resolution, False, True, False) # Max AE plot
    except Exception as e: # Likely missing enough data for plots, so we just skip them
        print(f"Error plotting the results: {e}.\nLikely from missing sufficient data for some elements. Plotting skipped.")


#fill_all_convergence_logs(TestCase.AIR_UNLIT, Resolution.LOW, blender=True)
#fill_convergence_log(Elements.QUAD8, TestCase.TANK, Resolution.LOW, 131072, 2097152, False)

#get_roi(TestCase.AIR_UNLIT)

#plot_results_subplots_patch(TestCase.TANK, Resolution.LOW, plot_time = False, save = True, show = False)

#plot_results_all(TestCase.TANK, Resolution.LOW, True, True)
#plot_results_subplots(TestCase.AIR_UNLIT, Resolution.HIGH, True, True)

#check_difference(Elements.QUAD9, TestCase.AIR_DIFFUSE, Resolution.HIGH, 524288, 1048576)

#plot_results_blender(TestCase.TANK, Resolution.LOW, False, True, True)
#plot_results_blender_rt_single(TestCase.AIR_UNLIT, Resolution.LOW, save=True, show=True, rmse=False)


# From 2**14 (16k) to 2**18 (262k) for Blender
#plot_relative_error_distribution(TestCase.AIR_UNLIT, Resolution.HIGH, subsamples_1 = 2**14, subsamples_2=2**15, blender=True, save=True, show=True)

#plot_relative_error_distribution(TestCase.AIR_DIFFUSE, Resolution.HIGH, subsamples_1 = 2**17, subsamples_2=2**18, blender=False, save=True, show=True, element=Elements.TRI3)
