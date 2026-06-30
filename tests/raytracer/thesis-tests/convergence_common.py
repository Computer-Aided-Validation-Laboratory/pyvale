"""
Stores all common functions/data for the convergence tests, regardless if
they are dedicated for Linux, Blender, single image case, or others.
"""
from enum import StrEnum, IntEnum
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import FormatStrFormatter

import smplotlib # For nicer figures (imo), but no need to install if you don't want it

from pyvale.raytracer.rtoutputformat import *
from global_utils import *

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

RMSE_LIMIT_MIN = 1e-6 # When rmse < RMSE_LIMIT_MIN, we say it is converged

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
# Convergence tester
# ================================================================================

def bitwise_compare(data_path_new: Path, data_path_prev: Path | None = None, bit_depth: BitDepth = BitDepth.BIT_12):
    """
    Checks if the images are bitwise identical. Written for 16-bit TIFFs (storing 12-bit images), but should
    work with BMP etc. as well.
    Created for the convergence tests, so it assumes that the size of the images is the same and does not check the format.

    If data_path_prev is None, it assumes that we want to compare the same image to itself to verify that the bitmap
    comparison works correctly (i.e., we get 100% similarity.)

    1. Finds the absolute difference between the two images. In the difference array:
        - 0 => Identical
        - != 0 => Different
    2. Counts number of non-zero entries in the difference array.
    3. Verifies if the images are bitwise identical based on the difference array and the total pixel count.
    4. Returns the similarity score.
    """
    max_value = 4095 # Max integer value for 12-bit uint; assign by default
    if bit_depth == BitDepth.BIT_8:
        max_value = 255 # Max integer value for 8-bit uint
    elif bit_depth == BitDepth.BIT_16:
        max_value = 65535 # Max integer value for 16-bit uint

    if data_path_prev is None:
        data_path_prev = data_path_new

    # cv2.IMREAD_ANYDEPTH forces OpenCV to keep the 16-bit depth instead of downsampling it to 8-bit
    pixel_array_new = cv2.imread(str(data_path_new), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    pixel_array_prev = cv2.imread(str(data_path_prev), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)

    # If it loaded as a 3-channel image, grab just the first channel
    if len(pixel_array_new .shape) == 3:
        pixel_array_new  = pixel_array_new [:, :, 0]
    if len(pixel_array_prev.shape) == 3:
        pixel_array_prev = pixel_array_prev[:, :, 0]

    # Difference between the two images
    difference = cv2.absdiff(pixel_array_new, pixel_array_prev)
    total_pixels = pixel_array_new.shape[0] * pixel_array_new.shape[1] # Pixel count in the image
    num_different = cv2.countNonZero(difference) # How many pixels are different
    num_identical = total_pixels - num_different # How many are identical
    
    # Calculate our metrics
    similarity_identical = num_identical / total_pixels # Similarity score based on how many pixels are exactly identical
    # Cast to float since original data is uint16, so we may risk overflow
    rmse = np.sqrt(np.mean((pixel_array_new - pixel_array_prev) ** 2)) # Root mean square error
    # RMSE similarity - based on the RMSE and the max. integer value for this picture (less sensitive to tiny per-pixel differences)
    similarity_rmse = 1.0 - rmse / float(max_value)

    return rmse, similarity_rmse, similarity_identical


# ================================================================================
# Post-processing
# ================================================================================

def fill_convergence_log(element: Element, test_case:TestCase, resolution: Resolution, start_subsamples: int, end_subsamples: int):
    """
    Fills the convergence log; useful if the rendering was interrupted or split across machines, etc.
    to get the data in the same csv effortlessly.
    """
    base_data_dir = "convergence_rt/res_" + str(resolution.value) + "/" + test_case.value + "/"
    elem_dir_name = base_data_dir + element.label
    data_path = test_dir(BASE_TEST_DIR, elem_dir_name)
    csv_path = data_path / "convergence_log.csv" # Full path to the csv with all numerical data
    image_base_name = "rtimage_subsamples_"
    image_suffix = ".tiff"
    subsamples = start_subsamples * 2 # Assuming we increase the subsample count in powers of 2
    iteration = 1
    prev_filename = image_base_name + str(start_subsamples) + image_suffix
    with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["iteration", "subsamples", "rmse", "sim_score_rmse", "sim_score_identical"])
            writer.writeheader()
            while subsamples <= end_subsamples:
                current_filename = image_base_name + str(subsamples) + image_suffix
                rmse, sim_score_rmse, sim_score_identical = bitwise_compare(data_path / current_filename, data_path / prev_filename)
                writer.writerow({
                    "iteration": iteration,
                    "subsamples": subsamples,
                    "rmse": rmse,
                    "sim_score_rmse": sim_score_rmse,
                    "sim_score_identical": sim_score_identical})
                prev_filename = current_filename
                subsamples *= 2
                iteration += 1

def plot_results_all(test_case: TestCase, resolution: Resolution, save: bool = False, detailed = False):
    """
    Plots all convergence results on the same plot.
    """
    # Get the address of the directory with the data (assuming we haven't changed it)
    base_data_dir = "convergence_rt/res_" + str(resolution.value) + "/" + test_case.value + "/"
    target_path = test_dir(BASE_TEST_DIR, base_data_dir)
    filename = test_case.value + "_convergence_plot.png"
    if detailed:
        filename = test_case.value + "_convergence_plot_detailed.png"

    # Create plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    #ax.set_title("Subsampling for high resolution/low resolution", fontsize=FONT_SIZES["suptitle"]) # If you want a title
    ax.set_xlabel("Subsamples per pixel", fontsize=FONT_SIZES["axis_labels"])
    ax.set_ylabel("RMSE [GL]", fontsize=FONT_SIZES["axis_labels"])
    ax.set_yscale("log")
    # Format the y-axis as by default it just shows orders of magnitude
    ax.yaxis.set_major_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3g"))
    ax.tick_params(axis="y", which="minor", labelsize=FONT_SIZES["ticks"]) # Set label sizes on the axis
    # Format x-axis as well
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="x", which="both", labelsize=FONT_SIZES["ticks"])
    label_x = None
    title = f"Convergence for test case: {test_case.value} at {resolution.value} px resolution"
    ax.set_title(title, fontsize=FONT_SIZES["suptitle"])
    # Iterate over elements
    for name, element in iter_elements():
        elem_dir_name = base_data_dir + element.label  
        data_path = test_dir(BASE_TEST_DIR, elem_dir_name) / "convergence_log.csv" # Full path to the csv with all numerical data
        #print(data_path)
        # Convergence stores data as [iteration, subsamples, rmse, sim_score_rmse, sim_score_identical]
        elem_data = np.loadtxt(data_path, delimiter=",", skiprows=1, unpack=True) # Full data
        if label_x is None:
                # Option 1: Fetch data for x-values and plot those (all; not good if you start at 1 subsamples, because the smaller values are too close)
                all_x = np.unique(elem_data[1])
                label_x = all_x
                # Option 2: Fewer datapoints initially, then stack with the last X values 
                #extra = all_x[all_x > 32768]
                #label_x = np.array([1, 16384])
                #extra = all_x[all_x > 1024]
                #label_x = np.array([1, 1024])
                #label_x = np.concatenate((label_x, extra))
                #label_x = np.unique(label_x)  # sort + deduplicate
        # Sanity check to make sure all data is being read correctly and not just skipped
        #print(name, "path:", data_path, "shape:", elem_data.shape, "rmse_minmax:", np.nanmin(elem_data[2]), np.nanmax(elem_data[2]))
        if not detailed:
            ax.plot(elem_data[1], elem_data[2],
                    label=name,
                    color=element.color,
                    marker="o",
                    linestyle="-",
                    linewidth=3,
                    markersize=10)
            #ax.plot(elem_data[1], elem_data[2], label=name, color=element.color, marker="o", linestyle="-", linewidth=3, markersize=10)
        # More detailed plot (focus on the last x values)
        else: 
            elem_data = elem_data[:, -4:] # Keep last 4 rows of original data (CSV) - if we want it more detailed
            label_x = np.unique(elem_data[1])
        # Plot RMSE values above the markers; don't do it on the full plot as they overlap and look poorly
            ax.plot(elem_data[1], elem_data[2], label=name, color=element.color, marker="o", linestyle="-", linewidth=3, markersize=10)
            for x, y in zip(elem_data[1], elem_data[2]):
                ax.annotate(
                    f"{y:.3g}", # RMSE value
                    xy=(x, y),
                    xytext=(12, 8), # 20 points to the right, 8 points above (w.r.t. marker dot)
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=FONT_SIZES["axis_labels"] - 4)
    ax.set_xticks(label_x)
    ax.set_xticklabels([f"{x:g}" for x in label_x])
    ax.legend(loc='upper right', fontsize=FONT_SIZES["axis_labels"])
    ax.grid(visible=True, which='both', axis='both')
    plt.tight_layout()
    plt.show()
    if save:
        fig.savefig(Path.joinpath(target_path, "convergence_plot.png"), dpi=300)


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
    """
    base_data_dir = "convergence_rt/res_" + str(resolution.value) + "/" + test_case.value + "/"
    elem_dir_name = base_data_dir + element.label  
    data_path = test_dir(BASE_TEST_DIR, elem_dir_name)
    data_path_higher = data_path / ("rtimage_subsamples_" + str(end_subsamples) + ".tiff")
    data_path_lower = data_path / ("rtimage_subsamples_" + str(start_subsamples) + ".tiff")

    label = str(end_subsamples) + "_" + str(start_subsamples)
    difference_image(data_path_higher, data_path_lower, label)
    difference_heatmap(data_path_higher, data_path_lower, label)



#check_difference(Elements.QUAD9, TestCase.AIR_DIFFUSE, Resolution.HIGH, 524288, 1048576)
#check_difference(Elements.QUAD9, TestCase.AIR_DIFFUSE, Resolution.HIGH, 262144, 1048576)
#fill_convergence_log(Elements.TRI6, TestCase.AIR_UNLIT, Resolution.LOW, 1, 2097152)
#plot_results_all(TestCase.AIR_DIFFUSE, Resolution.HIGH, True, False)
