from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field

from global_utils import *
# pyvale modules
import pyvale.dic as dic

import smplotlib # For nicer figures (imo), but no need to install if you don't want it

from pyvale.sensorsim.imagetools import ImageTools
import os
from pyvale.raytracer.rtmesh import *
from pyvale.raytracer.rtmeshvisuals import *
from pyvale.raytracer.rtcamera import *
from pyvale.raytracer.rtscene import *
from pyvale.raytracer.rtpresets import *
from pyvale.raytracer.rtmain import *
from pyvale.raytracer.rtoutputformat import *

base_output_dir = f"app3_exp/"
OUTPUT_DIR_PATH = test_dir(BASE_TEST_DIR, base_output_dir)
#IMG_ACCESS = "thesis-data/app3_exp/experiment_data/"
START_IMG_NAME = "start_image.npy"
END_IMG_NAME = "end_image.npy"
STATIC_IMG_NAME = "static_image.npy"
ROI_FILENAME = "roi.dat"
ROI_YAML_FILENAME = "roi.yaml"
EXP_DIC_RESULTS_PREFIX = "exp_dic_results_"
RT_DIC_RESULTS_PREFIX = "rt_dic_results_"
base_input_dir = Path(__file__).resolve().parent.parent.parent.parent.parent / "030225sidDynamic/specimenC1"

@dataclass(slots=True)
class ExpTest:
    label_file: str = field(default_factory = None)
    label_plot: str = field(default_factory = None)
    color: str = field(default_factory = None)
    data_source_dir: Path = field(default_factory = None)
    output_save_dir: Path = field(default_factory = None)
    frame_range_start: tuple = field(default_factory=None) # For static frames before motion
    frame_range_end: tuple | None = field(default_factory=None)  # For static frames after motion; None is for static case where we consider all frames
    measured_displ_mm: float = 0.0 # Real mm displacement from the micrometer stage

class ExpTests:
    STATIC = ExpTest(label_file = "static", label_plot = "static", color = "#917457",
                     data_source_dir = base_input_dir / "Calibration/static01b_C001H001S0001/static01b_C001H001S0001",
                     output_save_dir = OUTPUT_DIR_PATH / "static",
                     frame_range_start = (1,300), frame_range_end = None, measured_displ_mm = 0.0)
    AIR_A = ExpTest(label_file = "air_a", label_plot = "air (Run A)", color = "#53424c",
                     data_source_dir = base_input_dir / "Dry/sidDyn01a_C001H001S0001/sidDyn01a_C001H001S0001",
                     output_save_dir = OUTPUT_DIR_PATH / "air",
                     frame_range_start = (1,40), frame_range_end = (1300, 1326), measured_displ_mm = 3.001)
    AIR_B = ExpTest(label_file = "air_b", label_plot = "air (Run B)", color = "#53424c",
                     data_source_dir = base_input_dir / "Dry/sidDyn01b_C001H001S0001/sidDyn01b_C001H001S0001",
                     output_save_dir = OUTPUT_DIR_PATH / "air",
                     frame_range_start = (1,30), frame_range_end = (1327,1372), measured_displ_mm = 3.000)
    AIR_PIPE_A = ExpTest(label_file = "air_pipe_a", label_plot = "empty pipe (Run A)", color = "#c99fb6",
                     data_source_dir = base_input_dir / "Dry/sidDyn02a_C001H001S0001/sidDyn02a_C001H001S0001",
                     output_save_dir = OUTPUT_DIR_PATH / "air_pipe",
                     frame_range_start = (1,40), frame_range_end = (1232, 1268), measured_displ_mm = 3.000)
    AIR_PIPE_B = ExpTest(label_file = "air_pipe_b", label_plot = "empty pipe (Run B)", color = "#c99fb6",
                     data_source_dir = base_input_dir / "Dry/sidDyn02b_C001H001S0001/sidDyn02b_C001H001S0001",
                     output_save_dir = OUTPUT_DIR_PATH / "air_pipe",
                     frame_range_start = (1,40), frame_range_end = (1096,1153), measured_displ_mm = 3.006)
    AIR_TANK_A = ExpTest(label_file = "air_tank_a", label_plot = "empty tank (Run A)", color = "#826f99",
                     data_source_dir = base_input_dir / "Dry/sidDyn03a_C001H001S0001/sidDyn03a_C001H001S0001",
                     output_save_dir = OUTPUT_DIR_PATH / "air_tank",
                     frame_range_start = (1,55), frame_range_end = (1210, 1285), measured_displ_mm = 3.003)
    AIR_TANK_B = ExpTest(label_file = "air_tank_b", label_plot = "empty tank (Run B)", color = "#826f99",
                     data_source_dir = base_input_dir / "Dry/sidDyn03b_C001H001S0001/sidDyn03b_C001H001S0001",
                     output_save_dir = OUTPUT_DIR_PATH / "air_tank",
                     frame_range_start = (1,36), frame_range_end = (1044,1116), measured_displ_mm = 3.000)
    AIR_BOTH_A = ExpTest(label_file = "air_both_a", label_plot = "empty tank and pipe (Run A)", color = "#ead6c2",
                     data_source_dir = base_input_dir / "Dry/sidDyn04a_C001H001S0001/sidDyn04a_C001H001S0001",
                     output_save_dir = OUTPUT_DIR_PATH / "air_both",
                     frame_range_start = (1,40), frame_range_end = (1220, 1266), measured_displ_mm = 3.0001)
    AIR_BOTH_B = ExpTest(label_file = "air_both_b", label_plot = "empty tank and pipe (Run B)", color = "#ead6c2",
                     data_source_dir = base_input_dir / "Dry/sidDyn04b_C001H001S0001/sidDyn04b_C001H001S0001",
                     output_save_dir = OUTPUT_DIR_PATH / "air_both",
                     frame_range_start = (1,42), frame_range_end = (883,919), measured_displ_mm = 3.007)
    FLUID_PIPE_A = ExpTest(label_file = "fluid_pipe_a", label_plot = "RI-matching fluid in pipe, air in tank (Run A)", color = "#b9e1d8",
                     data_source_dir = base_input_dir / "Fluid/sidDyn05a_C001H001S0001/sidDyn05a_C001H001S0001",
                     output_save_dir = OUTPUT_DIR_PATH / "fluid_pipe",
                     frame_range_start = (1,35), frame_range_end = (1384,1421), measured_displ_mm = 3.001)
    FLUID_PIPE_B = ExpTest(label_file = "fluid_pipe_b", label_plot = "RI-matching fluid in pipe, air in tank (Run B)", color = "#b9e1d8",
                     data_source_dir = base_input_dir / "Fluid/sidDyn05b_C001H001S0001/sidDyn05b_C001H001S0001",
                     output_save_dir = OUTPUT_DIR_PATH / "fluid_pipe",
                     frame_range_start = (1,15), frame_range_end = (1100,1169), measured_displ_mm = 3.000)
    FLUID_A = ExpTest(label_file = "fluid_a", label_plot = "RI-matching fluid in tank and pipe (Run A)", color = "#5f9ea0",
                     data_source_dir = base_input_dir / "Fluid/sidDyn06a_C001H001S0001/sidDyn06a_C001H001S0001",
                     output_save_dir = OUTPUT_DIR_PATH / "fluid",
                     frame_range_start = (1,50), frame_range_end = (928, 967), measured_displ_mm = 3.001)
    FLUID_B = ExpTest(label_file = "fluid_b", label_plot = "RI-matching fluid in tank and pipe (Run B)", color = "#5f9ea0",
                     data_source_dir = base_input_dir / "Fluid/sidDyn06b_C001H001S0001/sidDyn06b_C001H001S0001",
                     output_save_dir = OUTPUT_DIR_PATH / "fluid",
                     frame_range_start = (1,42), frame_range_end = (800,832), measured_displ_mm = 3.001)
    
def iter_exp_tests():
    # Iterates over experimental test cases
    for name, value in vars(ExpTests).items():
        if isinstance(value, ExpTest):
            yield name, value

@dataclass(slots=True)
class RTTest:
    label_file: str = field(default_factory = None)
    label_plot: str = field(default_factory = None)
    color: str = field(default_factory = None)
    output_save_dir: Path = field(default_factory = None)

class RTTests:
    AIR = RTTest(label_file = "air_rt", label_plot = "air (RT)", color = "#53424c",
                 output_save_dir = ExpTests.AIR_A.output_save_dir / "rt")
    AIR_PIPE = RTTest(label_file = "air_pipe_rt", label_plot = "empty pipe (RT)", color = "#c99fb6",
                 output_save_dir = ExpTests.AIR_PIPE_A.output_save_dir / "rt")
    AIR_TANK = RTTest(label_file = "air_tank_rt", label_plot = "empty tank (RT)", color = "#826f99",
                 output_save_dir = ExpTests.AIR_TANK_A.output_save_dir / "rt")
    AIR_BOTH = RTTest(label_file = "air_both_rt", label_plot = "empty tank and pipe (RT)", color = "#ead6c2",
                 output_save_dir = ExpTests.AIR_BOTH_A.output_save_dir / "rt")
    FLUID_PIPE = RTTest(label_file = "fluid_pipe_rt", label_plot = "RI-matching fluid in pipe, air in tank (RT)", color = "#b9e1d8",
                 output_save_dir = ExpTests.FLUID_PIPE_A.output_save_dir / "rt")
    FLUID = RTTest(label_file = "fluid_rt", label_plot = "RI-matching fluid in tank and pipe (RT)", color = "#5f9ea0",
                 output_save_dir = ExpTests.FLUID_A.output_save_dir / "rt")
        
# DIC params
SUBSET_SIZE = 29 # px
STEP_SIZE = 19 # px
SCALE_PX_MM = 20.24 # px/mm scaling factor for this particular test; 20.24 px/mm <=>  20.24 px = 1 mm; what I got out of real data
SCALE_PX_MM = 19.6 # what I need to set this to to get identical beam widths in pixels BUT then the displacements are much worse than real
# From experimental tests
OUTPUT_FORMAT = ImageFormat(output_format=OutputFormat.IMG_TIFF_8BIT, bit_depth = BitDepth.BIT_8, channel_count = ChannelCount.MONO, grayscale=True)
ri_matching_fluid = Material(np.zeros(3), 1.49) # Optimistically assuming it hasn't turned yellow yet
    
# ================================================================================
# Histograms from ROI on experimental data
# ================================================================================
def read_and_match_histogram():
    # Use static images for the texture histogram
    static_path_in = ExpTests.STATIC.data_source_dir
    static_path_out = ExpTests.STATIC.output_save_dir
    mask = np.loadtxt(static_path_out / "static_roi.dat") # shape (1024, 1024)
    mask = mask.astype(bool) # convert 0/1 -> False/True

    image = cv2.imread(str(static_path_in / "static01b_C001H001S0001000001.tif"), cv2.IMREAD_GRAYSCALE)

    # Sanity check
    assert image.shape == mask.shape

    roi_pixels = image[mask] # 1D array of pixel values in ROI only
    hist_roi, bin_edges_roi = np.histogram(roi_pixels, bins=256, range=(0, 255))
    plt.figure()
    plt.plot(bin_edges_roi[:-1], hist_roi)
    plt.xlabel("Intensity")
    plt.ylabel("Count in ROI")
    plt.title("ROI intensity histogram")
    #plt.show()

    # Load the pyvale speckle texture whose histogram I want to adjust
    ref_texture = full_path("thesis-data/texture/speckle.tiff")
    beam_texture = ImageTools.load_image_greyscale(ref_texture) # height, width np.array storing the texture

    from skimage.exposure import match_histograms
    # match_histograms expects a 2D array, so reshape
    N = roi_pixels.size
    side = int(np.ceil(np.sqrt(N)))
    ref = np.zeros((side * side,), dtype=roi_pixels.dtype)
    ref[:N] = roi_pixels
    ref = ref.reshape((side, side)) # Synthetic reference image

    # Match histograms
    tex_matched = match_histograms(beam_texture, ref, channel_axis=None)

    # Check 
    # Old texture
    hist_tex_old, bin_edges_tex_old = np.histogram(beam_texture.ravel(), bins=256, range=(0, 255))
    plt.figure()
    plt.plot(bin_edges_roi[:-1], hist_roi / hist_roi.sum(), label="ROI")
    plt.plot(bin_edges_tex_old[:-1], hist_tex_old / hist_tex_old.sum(), label="Texture (original)")
    plt.legend()
    plt.xlabel("Intensity")
    plt.ylabel("Probability")
    plt.title("Histogram comparison before matching")
    #plt.show()
    plt.savefig(OUTPUT_DIR_PATH / "unmatched_histograms.png")

    # New texture hist after matching
    hist_tex, bin_edges_tex = np.histogram(tex_matched.ravel(), bins=256, range=(0, 255))

    plt.figure()
    plt.plot(bin_edges_roi[:-1], hist_roi / hist_roi.sum(), label="ROI")
    plt.plot(bin_edges_tex[:-1], hist_tex / hist_tex.sum(), label="Texture matched")
    plt.legend()
    plt.xlabel("Intensity")
    plt.ylabel("Probability")
    plt.title("Histogram match check")
    #plt.show()
    plt.savefig(OUTPUT_DIR_PATH / "matched_histograms.png")

    np.savetxt(OUTPUT_DIR_PATH / "matched_texture.npy", tex_matched)

#read_and_match_histogram()

# ================================================================================
# Average experimental images
# ================================================================================

def average_image(directory_input: Path, directory_output: Path, frame_range_start: int, frame_range_end: int, img_string: str) -> np.ndarray:
    image_average = np.zeros((image_height_phs6, image_height_phs6), dtype=np.float64) #1024 X 1024 px image

    # Directory name is the base for the CSV filenames, e.g., name_C001H001S0001 is the directory, then files are name_C001H001S0001XXXXXX.tif
    source_dir_name = os.path.basename(directory_input)
    suffix = "tif"
    frame_count = 0
    for i in range(frame_range_start, frame_range_end):
        # DaVis encodes filenames in format 000000, so e.g., 0000001, 0099. We want to make sure we add a sufficient
        # number of 0s to the front (left) to get the appropriate filename:
        file_number = str(i).zfill(6)
        filepath = os.path.join(directory_input, f"{source_dir_name}{file_number}.{suffix}")
        img_arr = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        frame_count += 1
        image_average += img_arr
    image_average /= frame_count
    try:
        if not (directory_output).is_dir():
            os.mkdir(directory_output)
        out_img_path = directory_output / img_string
        np.save(out_img_path, image_average)
        print(f"Succesfully exported the average image to {directory_output}.")
    except Exception as e:
        print(f"Error exporting the averaged image: {e}.")
    finally:
        return image_average
    

def get_avg_experimental_images(test: ExpTest):
    # Get averaged frames
    if test.frame_range_end is None: # Static case - single image
        image_start = average_image(test.data_source_dir, test.output_save_dir, test.frame_range_start[0], test.frame_range_start[1], STATIC_IMG_NAME)
    else:
        # Before motion
        image_start = average_image(test.data_source_dir, test.output_save_dir, test.frame_range_start[0], test.frame_range_start[1], f"{test.label_file}_{START_IMG_NAME}")
        # Now image after motion (in final position)
        image_end = average_image(test.data_source_dir, test.output_save_dir, test.frame_range_end[0], test.frame_range_end[1], f"{test.label_file}_{END_IMG_NAME}")
    
    
#get_avg_experimental_images(ExpTests.AIR_BOTH_B)
    
# ================================================================================
# Noise floor heatmap plotter adapted to pyvale DIC from my old code for davis outputs (https://github.com/AnalogArnold/2D-DIC-heatmap)
# ================================================================================

def create_mean_and_std_heatmap(mean_u_map, std_u_map, output_path, tick_jump=8, title=""):
    fig, axs = plt.subplots(1, 2, figsize=(14, 14),
        gridspec_kw={"width_ratios": [1, 1], "hspace": 0.2}, layout="constrained")

    fig.suptitle(title, fontsize=FONT_SIZES["suptitle"], y=1)

    ax0 = sns.heatmap(mean_u_map,center=0.0, cmap="coolwarm", square=True, ax=axs[0],cbar_kws={"shrink": 0.6})
    ax1 = sns.heatmap(std_u_map, cmap="viridis", square=True, ax=axs[1], cbar_kws={"shrink": 0.6})

    axs[0].set_title("$\overline{u_{x}} \ [px]$", fontsize=FONT_SIZES["axis_labels"])
    axs[1].set_title("$SD(u_{x}) \ [px]$", fontsize=FONT_SIZES["axis_labels"])

    for ax in axs:
        ax.invert_yaxis()

    apply_reduced_ticks(axs[0], tick_jump=tick_jump)
    axs[0].set_xlabel("$x$ [px]", fontsize=FONT_SIZES["axis_labels"])
    axs[0].set_ylabel("$y$ [px]", fontsize=FONT_SIZES["axis_labels"])

    apply_reduced_ticks(axs[1], tick_jump=tick_jump)
    axs[1].set_xlabel("$x$ [px]", fontsize=FONT_SIZES["axis_labels"])
    axs[1].set_ylabel("")

    axs[1].set_yticks([])

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def create_mean_xy_heatmap(mean_u_map, mean_v_map, output_path, tick_jump=8, title=""):
    all_vals = np.concatenate([mean_u_map.to_numpy().ravel(), mean_v_map.to_numpy().ravel()])
    all_vals = all_vals[np.isfinite(all_vals)]

    cbar_min = all_vals.min()
    cbar_max = all_vals.max()

    fig, axs = plt.subplots(1, 3, figsize=(14, 14), gridspec_kw={"width_ratios": [1, 1, 0.08], "hspace": 0.2})
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.suptitle(title, fontsize=FONT_SIZES["suptitle"], y=0.98)

    ax0 = sns.heatmap(mean_u_map,
        vmin=cbar_min, vmax=cbar_max, center=0.0,
        cmap="coolwarm", square=True, ax=axs[0], cbar=False)
    
    ax1 = sns.heatmap(mean_v_map,
        vmin=cbar_min, vmax=cbar_max, center=0.0,
        cmap="coolwarm", square=True, ax=axs[1], cbar_ax=axs[2])

    axs[0].set_title("$\overline{u_{x}} \ [px]$", fontsize=FONT_SIZES["axis_labels"])
    axs[1].set_title("$\overline{u_{y}} \ [px]$", fontsize=FONT_SIZES["axis_labels"])

    for ax in axs[:2]:
        ax.invert_yaxis()

    apply_reduced_ticks(axs[0], tick_jump=tick_jump)
    axs[0].set_xlabel("$x$ [px]", fontsize=FONT_SIZES["axis_labels"])
    axs[0].set_ylabel("$y$ [px]", fontsize=FONT_SIZES["axis_labels"])

    apply_reduced_ticks(axs[1], tick_jump=tick_jump)
    axs[1].set_xlabel("$x$ [px]", fontsize=FONT_SIZES["axis_labels"])
    axs[1].set_ylabel("")
    axs[1].set_yticks([])

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def create_single_heatmap(data_map, output_path, tick_jump=8, title="", cbar_label=""):
    fig, ax = plt.subplots(figsize=(5, 7), layout="constrained")

    sns.heatmap(data_map, cmap="viridis", square=True, ax=ax, cbar_kws={"label": cbar_label, "shrink": 0.7})

    ax.set_title(title, fontsize=FONT_SIZES["suptitle"]-6)
    apply_reduced_ticks(ax, tick_jump=tick_jump)
    ax.set_xlabel("$x$ [px]")
    ax.set_ylabel("$y$ [px]")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def apply_reduced_ticks(ax, tick_jump=8):
    x_ticks = ax.get_xticks()
    x_labels = ax.get_xticklabels()
    y_ticks = ax.get_yticks()
    y_labels = ax.get_yticklabels()

    if len(x_ticks) > 0:
        new_x_ticks = list(x_ticks[::tick_jump])
        new_x_labels = [lbl.get_text() for lbl in x_labels[::tick_jump]]
        if x_ticks[-1] not in new_x_ticks:
            new_x_ticks.append(x_ticks[-1])
            new_x_labels.append(x_labels[-1].get_text())
        ax.set_xticks(new_x_ticks)
        ax.set_xticklabels(new_x_labels, rotation=0)

    if len(y_ticks) > 0:
        new_y_ticks = list(y_ticks[::tick_jump])
        new_y_labels = [lbl.get_text() for lbl in y_labels[::tick_jump]]
        if y_ticks[-1] not in new_y_ticks:
            new_y_ticks.append(y_ticks[-1])
            new_y_labels.append(y_labels[-1].get_text())
        ax.set_yticks(new_y_ticks)
        ax.set_yticklabels(new_y_labels, rotation=0)

def plot_dic_noise_floor_heatmaps(stats_df, output_dir, output_prefix, tick_jump=8):
    mean_u_map = stats_df.pivot(index="subset_y", columns="subset_x", values="mean_u")
    mean_v_map = stats_df.pivot(index="subset_y", columns="subset_x", values="mean_v")
    std_u_map  = stats_df.pivot(index="subset_y", columns="subset_x", values="std_u")
    std_v_map  = stats_df.pivot(index="subset_y", columns="subset_x", values="std_v")
    std_mag_map = stats_df.pivot(index="subset_y", columns="subset_x", values="std_mag")

    create_mean_and_std_heatmap(mean_u_map=mean_u_map,
        std_u_map=std_u_map,
        output_path=os.path.join(output_dir, f"{output_prefix}_u_mean_std_heatmap.png"),
        tick_jump=tick_jump,
        title="Average and temporal noise floor in $x$")

    create_mean_xy_heatmap(mean_u_map=mean_u_map,
        mean_v_map=mean_v_map,
        output_path=os.path.join(output_dir, f"{output_prefix}_uv_mean_heatmap.png"),
        tick_jump=tick_jump,
        title="Mean displacements across all static frames")
    
    create_single_heatmap(data_map=std_v_map,
        output_path=os.path.join(output_dir, f"{output_prefix}_v_std_heatmap.png"),
        tick_jump=tick_jump,
        title="Temporal noise floor in $y$",
        cbar_label="$SD(u_{y}) \ [px]$")

    create_single_heatmap(data_map=std_mag_map,
        output_path=os.path.join(output_dir, f"{output_prefix}_mag_std_heatmap.png"),
        tick_jump=tick_jump,
        title="Temporal noise floor\nin displacement magnitude",
        cbar_label="$SD(|u{x}|) \ [px]$")
        
def run_dic_experimental_noise_floor():
    test = ExpTests.STATIC
    source_dir_name = os.path.basename(test.data_source_dir)
    suffix = "tif"
    ref_image = os.path.join(test.data_source_dir, f"{source_dir_name}{str(1).zfill(6)}.{suffix}")
    roi = dic.RegionOfInterest(ref_image=ref_image)
    roi_file = test.output_save_dir / f"{test.label_file}_{ROI_FILENAME}"
    dic_results_prefix = f"{test.label_file}_{EXP_DIC_RESULTS_PREFIX}"
    if not os.path.exists(roi_file):
        # Select and save ROI if file doesn't exist
        roi.interactive_selection(subset_size=SUBSET_SIZE)
        roi.save_array(filename=roi_file,binary=False)
    # Now DIC on deformation images
    for i in range(2, 300):
        # DaVis encodes filenames in format 000000, so e.g., 0000001, 0099. We want to make sure we add a sufficient
        # number of 0s to the front (left) to get the appropriate filename:
        file_number = str(i).zfill(6)
        def_image = os.path.join(test.data_source_dir, f"{source_dir_name}{file_number}.{suffix}")
        roi.read_array(filename=roi_file, binary=False)
        dic.calculate_2d(reference=ref_image,
                        deformed=def_image,
                        roi_mask=roi.mask,
                        seed=[400,400],
                        subset_size=29,
                        subset_step=19,
                        shape_function="AFFINE",
                        correlation_criteria="ZNSSD",
                        output_basepath=test.output_save_dir,
                        output_delimiter=",",
                        output_prefix=dic_results_prefix)
               

# ================================================================================
# DIC runners and plotters
# ================================================================================

def process_dic_noise_floor(plot_heatmaps=True, heatmap_tick_jump=8):
    test = ExpTests.STATIC
    # DIC files are stored as static_exp_dic_results_static01b_C001H001S0001000002
    # Columns: subset_x,subset_y,displacement_u,displacement_v,displacement_mag,converged,cost,ftol,xtol,num_iterations
    test = ExpTests.STATIC
    source_dir_name = os.path.basename(test.data_source_dir)
    dic_results_prefix = f"{test.label_file}_{EXP_DIC_RESULTS_PREFIX}{source_dir_name}"
    suffix = "csv"
    required_columns = ["subset_x", "subset_y","displacement_u", "displacement_v", "displacement_mag","converged", "cost", "ftol", "xtol", "num_iterations"]
    csv_attempt_data = []
    # Read DIC data
    for i in range(2, 300):
        file_number = str(i).zfill(6)
        data_csv_path = os.path.join(test.output_save_dir, f"{dic_results_prefix}{file_number}.{suffix}")
        if not os.path.exists(data_csv_path):
            print(f"Missing file, skipping: {data_csv_path}")
            continue
        try:
            df = pd.read_csv(data_csv_path, usecols=required_columns)
            df["frame_id"] = i
            csv_attempt_data.append(df)
        except Exception as e:
            print(f"Error processing file {data_csv_path}: {e}")
            continue

    if not csv_attempt_data:
        raise ValueError("No DIC CSV files were loaded.")

    df_all = pd.concat(csv_attempt_data, ignore_index=True)

    # Optional: keep only converged subsets if your software encodes this as bool or 0/1
    if "converged" in df_all.columns:
        df_all = df_all[df_all["converged"].astype(bool)].copy()

    # Compute statistics per subset location across all static frames
    stats_df = (
        df_all.groupby(["subset_x", "subset_y"], as_index=False)
        .agg(
            mean_u=("displacement_u", "mean"),
            mean_v=("displacement_v", "mean"),
            mean_mag=("displacement_mag", "mean"),
            std_u=("displacement_u", "std"),
            std_v=("displacement_v", "std"),
            std_mag=("displacement_mag", "std"),
            n=("displacement_u", "size"),
            mean_cost=("cost", "mean"),
            std_cost=("cost", "std"),
            mean_iterations=("num_iterations", "mean")))

    # Global summary metrics for reporting
    global_stats = pd.DataFrame({
        "metric": ["mean(mean_u)", "mean(mean_v)", "mean(mean_mag)",
            "mean(std_u)", "mean(std_v)", "mean(std_mag)",
            "max(std_u)", "max(std_v)", "max(std_mag)",
            "mean_cost", "mean_iterations"],
        "value": [
            stats_df["mean_u"].mean(),
            stats_df["mean_v"].mean(),
            stats_df["mean_mag"].mean(),
            stats_df["std_u"].mean(),
            stats_df["std_v"].mean(),
            stats_df["std_mag"].mean(),
            stats_df["std_u"].max(),
            stats_df["std_v"].max(),
            stats_df["std_mag"].max(),
            stats_df["mean_cost"].mean(),
            stats_df["mean_iterations"].mean()]})

    stats_output_path = os.path.join(test.output_save_dir, f"{dic_results_prefix}_noise_floor_statistics.csv")
    stats_df.to_csv(stats_output_path, index=False)

    global_output_path = os.path.join(test.output_save_dir, f"{dic_results_prefix}_noise_floor_global_summary.csv")
    global_stats.to_csv(global_output_path, index=False)

    if plot_heatmaps:
        plot_dic_noise_floor_heatmaps(stats_df=stats_df, output_dir=test.output_save_dir,
            output_prefix=dic_results_prefix,tick_jump=heatmap_tick_jump)

    return df_all, stats_df, global_stats

#run_dic_experimental_noise_floor()
#process_dic_noise_floor()            

def convert_roi_to_yaml(test: ExpTest):
    """
    Converts roi.dat to roi.yaml, so it can be opened in GUI and used for the RT images.
    (They can't be used explicitly since the beam in experimental data was slightly shifted,
    whereas in RT it's exactly centered)
    """
    import textwrap
    import yaml  # pip install pyyaml if needed

    width = image_height_phs6  #image width in px
    height = image_height_phs6
    roi_file = test.output_save_dir / f"{test.label_file}_{ROI_FILENAME}"

    # Load the .dat mask (ASCII 0/1, whitespace-separated)
    with open(roi_file, "r") as f:
        # Read all integers
        vals = np.fromstring(f.read(), sep=" ", dtype=int)

    mask = vals.reshape((height, width))

    # Find the bounding box of 1s
    ys, xs = np.where(mask == 1)
    if ys.size == 0:
        raise ValueError("Mask contains no foreground (no 1s)")

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())

    w = x_max - x_min + 1
    h = y_max - y_min + 1

    roi = [{
            "type": "RectROI",
            "pos": [x_min, y_min],
            "size": [w, h],
            "add": True}]
    # Write
    yaml_str = yaml.dump(roi, sort_keys=False)
    yaml_path = test.output_save_dir / f"{test.label_file}_{ROI_YAML_FILENAME}"
    with open(yaml_path, "w", encoding="utf-8") as f:
        print(f"Writing roi.yaml to {yaml_path}")
        f.write(yaml_str)

def run_dic_experimental(test: ExpTest, save_plot: bool = True, convert_to_mm: bool = True):
    """
    Runs DIC on the experimental images.
    """

    ref_img_path = test.output_save_dir / f"{test.label_file}_{START_IMG_NAME}"
    def_img_path = test.output_save_dir / f"{test.label_file}_{END_IMG_NAME}"
    ref_img = np.load(ref_img_path)
    def_img = np.load(def_img_path)

    # Default pyvale DIC data to test if the problem is data or me using this
    #ref_img_path = dataset.dic_plate_with_hole_ref()
    #def_img_path = dataset.dic_plate_with_hole_def()

    roi = dic.RegionOfInterest(ref_image=ref_img)
    roi_file = test.output_save_dir / f"{test.label_file}_{ROI_FILENAME}"
    dic_results_prefix = f"{test.label_file}_{EXP_DIC_RESULTS_PREFIX}"
    if not os.path.exists(roi_file):
        # Select and save ROI if file doesn't exist
        roi.interactive_selection(subset_size=29)
        roi.save_array(filename=roi_file,binary=False)
    
    dic_files =  test.output_save_dir / f"{dic_results_prefix}*.csv"
    # The above is a wildcard, so it will not work for the os.path.exists condition below
    dic_filename_check = test.output_save_dir / f"{dic_results_prefix}def_img_0000.csv"

    if not os.path.exists(dic_filename_check):
        # Run DIC analysis if it doesn't exist 
        roi.read_array(filename=roi_file, binary=False)
        dic.calculate_2d(reference=ref_img,
                        deformed=def_img,
                        roi_mask=roi.mask,
                        seed=[400,350],
                        subset_size=SUBSET_SIZE,
                        subset_step=STEP_SIZE,
                        shape_function="AFFINE",
                        correlation_criteria="ZNSSD",
                        output_basepath=test.output_save_dir,
                        output_delimiter=",",
                        output_prefix=dic_results_prefix)
    
    # Read data
    dicdata = dic.import_2d(data=dic_files, delimiter=",", binary=False)

    # Data for the first deformation image (and the only one in this case)
    horizontal_displacement = dicdata.u[0]
    vertical_displacement = dicdata.v[0]
    unit = "[px]"
    figure_filename = f"{test.label_file}_exp_dic_plot_px.png"
    if convert_to_mm:
        horizontal_displacement /= SCALE_PX_MM
        vertical_displacement /= SCALE_PX_MM
        unit = "[mm]"
        figure_filename = f"{test.label_file}_exp_dic_plot_mm.png"

    # Plot data
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes = axes.flatten()
    cmap = "magma"

    # First deformation image
    im1 = axes[0].pcolor(dicdata.ss_x, dicdata.ss_y, horizontal_displacement, cmap=cmap)
    im2 = axes[1].pcolor(dicdata.ss_x, dicdata.ss_y, vertical_displacement, cmap=cmap)

    # Titles
    fig.suptitle(f"Experimental DIC results\nTest case: {test.label_plot}", fontsize=FONT_SIZES["suptitle"])
    axes[0].set_title(f"$u_x$ {unit}", fontsize=FONT_SIZES["subtitle"]) # Horizontal displacement
    axes[1].set_title(f"$u_y$ {unit}", fontsize=FONT_SIZES["subtitle"]) # Vertical displacement

    for aa in axes:
        aa.set_aspect('equal')

    # Colorbars
    fig.colorbar(im1, ax=axes[0])
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    #plt.show()
    if save_plot:
        fig.savefig(test.output_save_dir / figure_filename, dpi=300, bbox_inches="tight")

def do_dic_all_exp():
    for name, test_case in iter_exp_tests():
        if name == "STATIC": # Skip static as it does its own thing
            continue
        run_dic_experimental(test_case, True, False) # px plot
        run_dic_experimental(test_case, True, True) # mm plot
        convert_roi_to_yaml(test_case)

#do_dic_all_exp()

# ================================================================================
# Ray tracer rendering
# ================================================================================
from convergence_common import * # Some functions for path, and tank positioning

def get_tank_path_exp(element: Element):
    # Tank access is sth like cwd/thesis-data/rectangular-box/coarse
    return full_path(f"thesis-data/nested-dielectrics/tank_surface_" + element.label + ".vtk") # full path to e.g., tank_surface_TRI3.vtk

def get_tank_fill_path_exp(element: Element):
    return full_path(f"thesis-data/nested-dielectrics/box_fill_surface_" + element.label + ".vtk")

def get_pipe_path_exp(element: Element):
    # Tank access is sth like cwd/thesis-data/rectangular-box/coarse
    return full_path(f"thesis-data/nested-dielectrics/pipe_surface_" + element.label + ".vtk") # full path to e.g., tank_surface_TRI3.vtk

def get_pipe_fill_path_exp(element: Element):
    return full_path(f"thesis-data/nested-dielectrics/pipe_fill_surface_" + element.label + ".vtk")


def render_exp_images(test: RTTest, aa_samples: int = 1, min_refr_depth: int | None = 4, crop_px: bool = False,
                      frame_idx: int | None = None):
    # 1. Paths to data, etc.
    #pipe_access = "thesis-data/" + Tank.PIPE + "/" + Refinement.COARSE # Point the correct mesh locations
    #tank_access = "thesis-data/" + Tank.RECTANGLE + "/" + Refinement.COARSE # Point the correct mesh locations
    #pipe_path = get_tank_path(pipe_access, Elements.TRI6)
    #tank_path = get_tank_path(tank_access, Elements.TRI6)
    #water_tank_path = get_fill_path(tank_access, Elements.TRI6)
    #water_pipe_path = get_fill_path(pipe_access, Elements.TRI6)
    pipe_path = get_pipe_path_exp(Elements.TRI6)
    tank_path = get_tank_path_exp(Elements.TRI6)
    water_tank_path = get_tank_fill_path_exp(Elements.TRI6)
    water_pipe_path = get_pipe_fill_path_exp(Elements.TRI6)

    # Set the sample path
    sample_element = Elements.TRI3
    sample_name = "thesis-data/beam/exp_coarse/beam_surface_"
    sample_path = full_path(sample_name + sample_element.label + ".vtk")
    # Sample texture
    # Histogram-matched pyvale speckle
    beam_texture = np.loadtxt(OUTPUT_DIR_PATH / "matched_texture.npy")
    
    # 2. Mesh set-up
    scene = Scene()
    TANK_BOTTOM_HEIGHT = 4.0 # mm; height of the bottom wall of the tank
    TANK_HEIGHT = 95.0
    PIPE_INN_BOTTOM_Y = -43.0
    # Pipe and tank were created to begin at the same y, so now we need to move the pipe up so that:
    # 1. Dry case: It touches the bottom of the tank inside, rather than going through it
    PIPE_SHIFT_DRY = np.array([0.0, TANK_BOTTOM_HEIGHT, 0.0])
    # 2. Fluid case: There is slight overlap (and water needs to be shifted accordingly)
    PIPE_SHIFT_FLUID = np.array([0.0, TANK_BOTTOM_HEIGHT/1.5, 0.0])
    # Beam needs to be lifted up as well

    # Tank and pipe
    pipe = any_mesh_to_rtmesh(pipe_path)
    tank = any_mesh_to_rtmesh(tank_path)
    z_translation = np.array([0.0, 0.0, TANK_MID_Z])
    pipe.translate(z_translation)
    tank.translate(z_translation)
    #tank = any_mesh_to_rtmesh(tank_path, world_position = TANK_POSITION, anchor = Anchor.CENTER)
    #beam_position = np.array([0.0, PIPE_INN_BOTTOM_Y, TANK_MID_Z])
    beam_position = np.array([0.0, PIPE_INN_BOTTOM_Y+3, TANK_MID_Z])
    beam = any_mesh_to_rtmesh(sample_path, world_position = beam_position, anchor = Anchor.BASE) 
    beam.translate(PIPE_SHIFT_DRY)

    
    # Data for mock displacement for the beam (and everything else as sadly occured experimentally)
    frame_count = 2
    total_displacement = 3.000 # in mm
    beam_nodal_displacements = create_rigid_linear_translation(beam.node_count, frame_count, total_displacement, Axis.X)
    beam.add_temporal_displacement(beam_nodal_displacements)
    pipe_temporal_displacements = create_rigid_linear_translation(pipe.node_count, frame_count, total_displacement, Axis.X)
    tank_temporal_displacements = create_rigid_linear_translation(tank.node_count, frame_count, total_displacement, Axis.X)
    tank.add_temporal_displacement(tank_temporal_displacements)
    # Pipe will have to get the displacements added after being translated, which is case specific to model the dielectrics correctly
    
    if test == RTTests.AIR_PIPE:
        print(f"--------------------------------\nTESTED CASE: AIR PIPE\n--------------------------------")
        #pipe.translate(PIPE_SHIFT_DRY)
        pipe.add_temporal_displacement(pipe_temporal_displacements)
        pipe.set_surface(SurfType.FIELD_COLOR, material = MaterialPresets.PLASTIC_ACRYLIC,
                         material_type = MaterialType.REFRACTIVE, mesh_type = MeshType.SOLID)
        scene.add_rtmesh(pipe)
    elif test == RTTests.AIR_TANK:
        print(f"--------------------------------\nTESTED CASE: AIR TANK\n--------------------------------")
        tank.set_surface(SurfType.FIELD_COLOR, material = MaterialPresets.PLASTIC_ACRYLIC,
                         material_type = MaterialType.REFRACTIVE, mesh_type = MeshType.SOLID)
        scene.add_rtmesh(tank)
    elif test == RTTests.AIR_BOTH:
        print(f"--------------------------------\nTESTED CASE: AIR BOTH\n--------------------------------")
        # No nested dielectrics yet because there is air separating tank and pipe
        #pipe.translate(PIPE_SHIFT_DRY)
        #pipe.add_temporal_displacement(pipe_temporal_displacements)
        pipe.set_surface(SurfType.FIELD_COLOR, material = MaterialPresets.PLASTIC_ACRYLIC,
                         material_type = MaterialType.REFRACTIVE, mesh_type = MeshType.SOLID)
        tank.set_surface(SurfType.FIELD_COLOR, material = MaterialPresets.PLASTIC_ACRYLIC,
                                material_type = MaterialType.REFRACTIVE, mesh_type = MeshType.SOLID)
        scene.add_rtmesh([pipe, tank])
    elif test == RTTests.FLUID_PIPE:
        print(f"--------------------------------\nTESTED CASE: FLUID_PIPE\n--------------------------------")
        # First nested dielectric case
        #pipe.translate(PIPE_SHIFT_DRY)
        pipe.add_temporal_displacement(pipe_temporal_displacements)
        pipe.set_surface(SurfType.FIELD_COLOR, material = MaterialPresets.PLASTIC_ACRYLIC,
                         material_type = MaterialType.REFRACTIVE, mesh_type = MeshType.SOLID,
                         priority = 1)
        tank.set_surface(SurfType.FIELD_COLOR, material = MaterialPresets.PLASTIC_ACRYLIC,
                                material_type = MaterialType.REFRACTIVE, mesh_type = MeshType.SOLID,
                                priority = 0)
        # Add water fill to the pipe only
        #water_pipe = any_mesh_to_rtmesh(water_pipe_path, world_position = WATER_POSITION)
        water_pipe = any_mesh_to_rtmesh(water_pipe_path)
        water_pipe.translate(z_translation)
        #water_pipe.translate(PIPE_SHIFT_DRY)
        water_pipe_temporal_displacements = create_rigid_linear_translation(water_pipe.node_count, frame_count, total_displacement, Axis.X)
        water_pipe.add_temporal_displacement(water_pipe_temporal_displacements)
        water_pipe.set_surface(SurfType.FIELD_COLOR, material = ri_matching_fluid,
                          material_type = MaterialType.REFRACTIVE,
                          mesh_type = MeshType.SOLID,
                          priority = 0) # Pipe is open-ended => Water needs to have higher priority
        scene.add_rtmesh([pipe, tank, water_pipe])
    elif test == RTTests.FLUID:
        # Very nested case - tinker carefully
        # Order of dielectrics
        # tank -> water_tank -> pipe -> water_pipe, BUT all are open-ended, so we need to reverse it
        # Pipe might need to be moved down to intersect with the tank
        print(f"--------------------------------\nTESTED CASE: FLUID\n--------------------------------")
        #pipe.translate(PIPE_SHIFT_FLUID)
        pipe.add_temporal_displacement(pipe_temporal_displacements)
        pipe.set_surface(SurfType.FIELD_COLOR, material = MaterialPresets.PLASTIC_ACRYLIC,
                         material_type = MaterialType.REFRACTIVE, mesh_type = MeshType.SOLID,
                         priority = 1)
        tank.set_surface(SurfType.FIELD_COLOR, material = MaterialPresets.PLASTIC_ACRYLIC,
                        material_type = MaterialType.REFRACTIVE, mesh_type = MeshType.SOLID,
                        priority = 3)
        # Add water fills
        #water_pipe = any_mesh_to_rtmesh(water_pipe_path, world_position = WATER_POSITION)
        water_pipe = any_mesh_to_rtmesh(water_pipe_path)
        water_pipe.translate(z_translation)
        #water_pipe.translate(PIPE_SHIFT_FLUID)
        water_pipe_temporal_displacements = create_rigid_linear_translation(water_pipe.node_count, frame_count, total_displacement, Axis.X)
        water_pipe.add_temporal_displacement(water_pipe_temporal_displacements)
        water_pipe.set_surface(SurfType.FIELD_COLOR, material = ri_matching_fluid,
                          material_type = MaterialType.REFRACTIVE,
                          mesh_type = MeshType.SOLID,
                          priority = 0) # Pipe is open-ended => Water needs to have higher priority
        #water_tank = any_mesh_to_rtmesh(water_tank_path, world_position = WATER_POSITION)
        water_tank = any_mesh_to_rtmesh(water_tank_path)
        water_tank.translate(z_translation)
        water_tank_temporal_displacements = create_rigid_linear_translation(water_tank.node_count, frame_count, total_displacement, Axis.X)
        water_tank.add_temporal_displacement(water_tank_temporal_displacements)
        water_tank.set_surface(SurfType.FIELD_COLOR, material = ri_matching_fluid,
                          material_type = MaterialType.REFRACTIVE,
                          mesh_type = MeshType.SOLID,
                          priority = 2) # Pipe is open-ended => Water needs to have higher priority
        scene.add_rtmesh([pipe, tank, water_pipe, water_tank])

    # Check positioning - VERY IMPORTANT HERE
    #SceneVisualiser([water_tank, pipe, water_pipe, tank, beam])
    #SceneVisualiser([pipe, tank, beam])
    #SceneVisualiser([pipe, tank, water_pipe])
    #SceneVisualiser([pipe, tank, water_tank]) 
    # Data for Photron Nova S6
    image_width = image_width_phs6
    image_height = image_width_phs6
    output_format = output_format_cx5 # Not a mistake. 8-bit TIFF, seeing as I mistakenly exported my experimental data in this format instead of 12-bit :')
    #output_format = output_format_test_diel # TEST
    focal_length = 100 # mm
    CAMERA_Z = 280 + TANK_MID_Z # Camera was about 280 mm away from the beam, but beam isn't at z=0, so we need to account for that to get camera z
    TARGET_Z = CAMERA_Z - focal_length #mm
    beam_center_coords = beam._get_bounding_box()["center"]
    beam_center_y = beam_center_coords[1]
    CAMERA_Y = beam_center_y+1.5

    #target_distance = camera_distance - focal_length
    #camera_distance = camera_working_distance(focal_length, fov_height, sensor_height_phs6)
    # Sanity check for camera distance by reverse engineering it based on the beam size from experimental data
    #beam_width = 12.11 # mm
    #pixel_pitch = pixel_pitch_ph6
    #pixel_count_width = 248 # px
    #est_cam_dist = beam_width * focal_length / (pixel_pitch * pixel_count_width)
    #print(f"Estimated camera distance: {est_cam_dist}")
    # Beam was at line 280 mm on the table
    # Lens finished slightly before the 100 mm line
    beam_front_z = beam._get_bounding_box()["max_corner"][2] # -23.55 (face nearest to camera)
    camera_to_beam_dist = CAMERA_Z - beam_front_z # 256 - (-23.55) = 279.55 # Distance from camera to beam
    angle_vfov = vertical_fov_from_resolution(image_height, SCALE_PX_MM, camera_to_beam_dist)
    print(f"Angle vfov with LV calibration: {angle_vfov}")

    # 4. Texture and speckle pattern information for the beam
    # The loaded texture is 2464 x 2056 px (5MPx), 8-bit .tiff; speckles sampled by 5 pixels
    # Adjust it to our resolution of 1 MPx and the experimental speckle sizes by scaling the UVs
    beam.import_uvs(sample_uv_path(sample_path, sample_element)) # Load pre-processed UVs
    beam.set_surface(SurfType.TEXTURE, beam_texture, MaterialType.DIFFUSE)
    # Scale the UVs to get 3.5 px speckles in the rendered images
    uv_scale = speckle_scaling(image_width, image_height, 2464, 2056, 5, 3) # Returns [delta_u, delta_v] array
    beam.uvs = beam.uvs * uv_scale
    scene.add_rtmesh(beam)

    bb = beam._get_bounding_box()
    width_mm = bb["max_corner"][0] - bb["min_corner"][0]

    # 6. Render
    target_path = test.output_save_dir
    if not target_path.is_dir():
        target_path.mkdir(parents=True, exist_ok=True)
    if not crop_px:
        camera_center = np.array([0.0, CAMERA_Y, CAMERA_Z])
        camera_target = np.array([0.0, CAMERA_Y, TARGET_Z])
        cam = Camera(image_width, image_height, camera_center, camera_target, angle_vfov)
    else:
        # Adjust rendered image size (but none of the scene dimensions) to chop a few px off to save on render time, while getting the same exact output for ROI
        if test == RTTests.FLUID_PIPE:
            #In pipe, the beam gets much wider, so the side crop must be less to preserve the ROI
            # Horizontal crop can remain unchanged (still captures the ROI from experimental data)
            left_crop = 240 # px
            right_crop = 240
            top_crop = 15 # px, cropped from the top
        elif test == RTTests.FLUID:
            left_crop = 300 # px
            right_crop = 240
            top_crop = 15 # px, cropped from the top
        else: # Same crop is fine for all other cases
            left_crop = 375 # px, per side
            right_crop = 315
            top_crop = 25 # px, cropped from the top

        bottom_crop = 445 # px, cropped from the bottom <- no need to offset camera for this
        scale_mm_px = 1/SCALE_PX_MM # To convert these offsets into actual camera shifts, so our image still shows the desired FOV
        cam_y_offset = top_crop * scale_mm_px # Offset down
        cam_x_offset = left_crop * scale_mm_px # Offset to the right
        camera_center = np.array([cam_x_offset, CAMERA_Y-cam_y_offset, CAMERA_Z])
        camera_target = np.array([cam_x_offset, CAMERA_Y-cam_y_offset, TARGET_Z])
        cam = Camera(image_width, image_height, camera_center, camera_target, angle_vfov)
        image_width = image_width_phs6 - left_crop - right_crop
        image_height = image_width_phs6 - bottom_crop - top_crop
    #SceneVisualiser([pipe, tank, beam], cam)
    scene.add_camera(cam)
    # Render both images at once or one at a time (option mostly for fluid cases that might take ages otherwise)
    if frame_idx is None:
        render_scene(image_height, image_width, scene, aa_samples, target_path, RenderType.DYNAMIC, texture_sampler = TextureSampler.CATMULL_ROM, shading_type = ShadingType.FLAT, image_format = output_format, omp_thread_count = None, min_refractive_depth=min_refr_depth)
    else:
        fresh_filename = "rtimage_0_cam0.tiff"
        if frame_idx == 0:
        # Render frame 0 (undeformed)nan
            import timeit
            time = timeit.timeit(lambda:render_scene(image_height, image_width, scene, aa_samples, target_path, RenderType.STATIC, frames_to_render = 0, texture_sampler = TextureSampler.CATMULL_ROM, shading_type = ShadingType.FLAT, image_format = output_format, omp_thread_count = None, min_refractive_depth=min_refr_depth), number=1)
            print(f"Time taken:{time} s")
            #render_scene(image_height, image_width, scene, aa_samples, target_path, RenderType.STATIC, frames_to_render = 0, texture_sampler = TextureSampler.CATMULL_ROM, shading_type = ShadingType.FLAT, image_format = output_format, omp_thread_count = None, min_refractive_depth=min_refr_depth)
        elif frame_idx == 1:
        # Render frame 1 (deformed)
            temp_filename = None
            if os.path.exists(target_path.joinpath(fresh_filename)): # Check if frame 0 exists
                print(f"{fresh_filename} already exists. Temporarily overwriting.")
                temp_filename = "rtimage_0.tiff"
                os.rename(target_path.joinpath(fresh_filename), target_path.joinpath(temp_filename)) # Set a new temporary filename if it does
            render_scene(image_height, image_width, scene, aa_samples, target_path, RenderType.STATIC, frames_to_render = 1, texture_sampler = TextureSampler.CATMULL_ROM, shading_type = ShadingType.FLAT, image_format = output_format, omp_thread_count = None, min_refractive_depth=min_refr_depth)
            new_filename = f"rtimage_1_cam0.tiff"
            os.rename(target_path.joinpath(fresh_filename), target_path.joinpath(new_filename)) # Rename the rendered frame to indicate it is frame 1, i.e., displaced
            if temp_filename is not None:
                os.rename(target_path.joinpath(temp_filename), target_path.joinpath(fresh_filename)) # Restore the name of frame 0
        else:
            raise ValueError(f"Wrong frame index: {frame_idx}")

#render_exp_images(RTTests.BOTH, aa_samples=1, min_refr_depth=8) # This RR depth works best
#render_exp_images(RTTests.FLUID_PIPE, aa_samples=1, min_refr_depth=8) # This RR depth works best
#render_exp_images(RTTests.FLUID, aa_samples=1, min_refr_depth=4) # <= Looks kinda good, but need to fix modeling of water in tank. Awfully slow even at RR=4 and aa=1
#render_exp_images(RTTests.FLUID, aa_samples=1, min_refr_depth=4, crop_px = True)
#render_exp_images(RTTests.FLUID, aa_samples=1, min_refr_depth=2, crop_px = True, frame_idx = 0)

# Note to self (14.07):
# VERIFY THE CROP_PX FOR ALL CASES AFTER FIXING DIELECTRIC SCENE POSITIONING TO MAKE SURE THE BEAM STAYS VISIBLE
#render_exp_images(RTTests.FLUID, aa_samples=1, min_refr_depth=4, crop_px = False, frame_idx = 0)

# ================================================================================
# Ray tracer DIC
# ================================================================================

def run_dic_rt(test: RTTest, save_plot: bool = True, convert_to_mm: bool = True):
    """
    Runs DIC on the experimental images.
    """

    ref_img_path = test.output_save_dir / f"rtimage_0_cam0.tiff"
    def_img_path = test.output_save_dir / f"rtimage_1_cam0.tiff"
    ref_img = ImageTools.load_image_greyscale(ref_img_path)
    def_img = ImageTools.load_image_greyscale(def_img_path)

    roi = dic.RegionOfInterest(ref_image=ref_img)
    roi_file = test.output_save_dir / f"{test.label_file}_{ROI_FILENAME}"
    dic_results_prefix = f"{test.label_file}_{RT_DIC_RESULTS_PREFIX}"
    if not os.path.exists(roi_file):
        # Select and save ROI if file doesn't exist
        roi.interactive_selection(subset_size=SUBSET_SIZE)
        roi.save_array(filename=roi_file,binary=False)
    
    dic_files =  test.output_save_dir / f"{dic_results_prefix}*.csv"
    # The above is a wildcard, so it will not work for the os.path.exists condition below
    dic_filename_check = test.output_save_dir / f"{dic_results_prefix}def_img_0000.csv"

    if not os.path.exists(dic_filename_check):
        # Run DIC analysis if it doesn't exist 
        roi.read_array(filename=roi_file, binary=False)
        dic.calculate_2d(reference=ref_img,
                        deformed=def_img,
                        roi_mask=roi.mask,
                        seed=roi.seed,
                        #seed=[480,300], # For full images
                        #seed=[131,267] # Cropped images
                        subset_size=SUBSET_SIZE,
                        subset_step=STEP_SIZE,
                        shape_function="AFFINE",
                        correlation_criteria="ZNSSD",
                        output_basepath=test.output_save_dir,
                        output_delimiter=",",
                        output_prefix=dic_results_prefix)
    
    # Read data
    dicdata = dic.import_2d(data=dic_files, delimiter=",", binary=False)

    # Data for the first deformation image (and the only one in this case)
    horizontal_displacement = dicdata.u[0]
    vertical_displacement = dicdata.v[0]
    unit = "[px]"
    figure_filename = f"{test.label_file}_rt_dic_plot_px.png"
    if convert_to_mm:
        horizontal_displacement /= SCALE_PX_MM
        vertical_displacement /= SCALE_PX_MM
        unit = "[mm]"
        figure_filename = f"{test.label_file}_rt_dic_plot_mm.png"

    # Plot data
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes = axes.flatten()
    cmap = "magma"

    # First deformation image
    im1 = axes[0].pcolor(dicdata.ss_x, dicdata.ss_y, horizontal_displacement, cmap=cmap)
    im2 = axes[1].pcolor(dicdata.ss_x, dicdata.ss_y, vertical_displacement, cmap=cmap)

    # Titles
    fig.suptitle(f"RayTracer DIC results\nTest case: {test.label_plot}", fontsize=FONT_SIZES["suptitle"])
    axes[0].set_title(f"$u_x$ {unit}", fontsize=FONT_SIZES["subtitle"]) # Horizontal displacement
    axes[1].set_title(f"$u_y$ {unit}", fontsize=FONT_SIZES["subtitle"]) # Vertical displacement

    for aa in axes:
        aa.set_aspect('equal')

    # Colorbars
    fig.colorbar(im1, ax=axes[0])
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    #plt.show()
    if save_plot:
        fig.savefig(test.output_save_dir / figure_filename, dpi=300, bbox_inches="tight")

#run_dic_rt(RTTests.AIR, save_plot=True, convert_to_mm=True)
