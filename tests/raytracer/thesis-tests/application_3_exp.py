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

base_output_dir = f"app3_exp/dic/exp"
OUTPUT_DIR_PATH = test_dir(BASE_TEST_DIR, base_output_dir)
#IMG_ACCESS = "thesis-data/app3_exp/experiment_data/"
START_IMG_NAME = "start_image.npy"
END_IMG_NAME = "end_image.npy"
STATIC_IMG_NAME = "static_image.npy"
ROI_FILENAME = "roi.dat"
EXP_DIC_RESULTS_PREFIX = "exp_dic_results_"
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

ri_matching_fluid = Material(np.zeros(3), 1.49) # Optimistically assuming it hasn't turned yellow yet

class ExpTests:
    STATIC = ExpTest(label_file = "static", label_plot = "static", color = "#ead6c2",
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
                     frame_range_start = (1,40), frame_range_end = (1232, 1268), measured_displ_mm = 3.006)
    AIR_PIPE_B = ExpTest(label_file = "air_pipe_b", label_plot = "empty pipe (Run B)", color = "#c99fb6",
                     data_source_dir = base_input_dir / "Dry/sidDyn02b_C001H001S0001/sidDyn02b_C001H001S0001",
                     output_save_dir = OUTPUT_DIR_PATH / "air_pipe",
                     frame_range_start = (1,40), frame_range_end = (1096,1153), measured_displ_mm = 3.006)
    AIR_TANK_A = ExpTest(label_file = "air_tank_a", label_plot = "empty tank (Run A)", color = "#826f99",
                     data_source_dir = base_input_dir / "Dry/sidDyn03a_C001H001S0001/sidDyn03a_C001H001S0001",
                     output_save_dir = OUTPUT_DIR_PATH / "air_tank",
                     frame_range_start = (1,55), frame_range_end = (1210, 1285), measured_displ_mm = 3.0003)
    AIR_TANK_B = ExpTest(label_file = "air_tank_b", label_plot = "empty tank (Run B)", color = "#826f99",
                     data_source_dir = base_input_dir / "Dry/sidDyn03b_C001H001S0001/sidDyn03b_C001H001S0001",
                     output_save_dir = OUTPUT_DIR_PATH / "air_tank",
                     frame_range_start = (1,36), frame_range_end = (1044,1116), measured_displ_mm = 3.000)
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
    
# DIC params
subset_size = 29 # px
step_size = 19 # px
scale_factor = 20.24 # px
# From experimental tests
output_format = ImageFormat(output_format=OutputFormat.IMG_TIFF_8BIT, bit_depth = BitDepth.BIT_8, channel_count = ChannelCount.MONO, grayscale=True)
    
# ================================================================================
# Histograms from ROI on experimental data
# ================================================================================
"""
mask = np.loadtxt(OUTPUT_DIR_PATH / "roi.dat")        # shape (1024, 1024)
mask = mask.astype(bool)                 # convert 0/1 -> False/True

image = cv2.imread(str(full_path(IMG_ACCESS + "static01b_C001H001S0001000001.tif")), cv2.IMREAD_GRAYSCALE)

# Sanity check
assert image.shape == mask.shape

roi_pixels = image[mask]                 # 1D array of pixel values in ROI only
plt.hist(roi_pixels.ravel(), bins=256, range=(0, 255))
plt.xlabel("Intensity")
plt.ylabel("Count")
plt.title("Histogram of ROI")
plt.show()

mask_uint8 = mask.astype("uint8") * 255  # OpenCV mask is 0 or 255
hist = cv2.calcHist([image], [0], mask_uint8, [256], [0, 256])


image = cv2.imread("image.png", cv2.IMREAD_COLOR)
b, g, r = cv2.split(image)

roi_b = b[mask]
roi_g = g[mask]
roi_r = r[mask]

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.hist(roi_b.ravel(), bins=256, range=(0, 255), color='b')
plt.title("Blue ROI")

plt.subplot(1, 3, 2)
plt.hist(roi_g.ravel(), bins=256, range=(0, 255), color='g')
plt.title("Green ROI")

plt.subplot(1, 3, 3)
plt.hist(roi_r.ravel(), bins=256, range=(0, 255), color='r')
plt.title("Red ROI")

plt.tight_layout()
#plt.show()
"""

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
    
    
#get_avg_experimental_images(ExpTests.AIR_B)
    

# ================================================================================
# Noise floor heatmap plotter adapted to pyvale DIC from my old code for davis outputs (https://github.com/AnalogArnold/2D-DIC-heatmap)
# ================================================================================

def create_mean_and_std_heatmap(mean_u_map, std_u_map, output_path, tick_jump=8, title=""):
    fig, axs = plt.subplots(1, 2, figsize=(14, 14),
        gridspec_kw={"width_ratios": [1, 1], "hspace": 0.2}, layout="constrained")

    fig.suptitle(title, fontsize=FONT_SIZES["suptitle"], y=1)

    ax0 = sns.heatmap(mean_u_map,center=0.0, cmap="coolwarm", square=True, ax=axs[0],cbar_kws={"shrink": 0.6})
    ax1 = sns.heatmap(std_u_map, cmap="viridis", square=True, ax=axs[1], cbar_kws={"shrink": 0.6})

    axs[0].set_title("$\overline{U_{x}} \ [px]$", fontsize=FONT_SIZES["axis_labels"])
    axs[1].set_title("$SD(U_{x}) \ [px]$", fontsize=FONT_SIZES["axis_labels"])

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

    axs[0].set_title("$\overline{U_{x}} \ [px]$", fontsize=FONT_SIZES["axis_labels"])
    axs[1].set_title("$\overline{U_{y}} \ [px]$", fontsize=FONT_SIZES["axis_labels"])

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

    ax.set_title(title, fontsize=FONT_SIZES["suptitle"])
    ax.invert_yaxis()
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
        cbar_label="$SD(U_{y}) \ [px]$")

    create_single_heatmap(data_map=std_mag_map,
        output_path=os.path.join(output_dir, f"{output_prefix}_mag_std_heatmap.png"),
        tick_jump=tick_jump,
        title="Temporal noise floor\nin displacement magnitude",
        cbar_label="$SD(|U{x}|) \ [px]$")
        
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
        roi.interactive_selection(subset_size=29)
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
process_dic_noise_floor()            


def run_dic_experimental(test: ExpTest):
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

    roi.read_array(filename=roi_file, binary=False)
    dic.calculate_2d(reference=ref_img,
                    deformed=def_img,
                    roi_mask=roi.mask,
                    seed=[400,400],
                    subset_size=29,
                    subset_step=19,
                    shape_function="AFFINE",
                    correlation_criteria="ZNSSD",
                    output_basepath=test.output_save_dir,
                    output_delimiter=",",
                    output_prefix=dic_results_prefix)
    
    dic_files =  test.output_save_dir / f"{dic_results_prefix}*.csv"
    dicdata = dic.import_2d(data=dic_files, delimiter=",", binary=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes = axes.flatten()
    cmap = "magma"

    # First deformation image
    im1 = axes[0].pcolor(dicdata.ss_x, dicdata.ss_y, dicdata.u[0], cmap=cmap)
    im2 = axes[1].pcolor(dicdata.ss_x, dicdata.ss_y, dicdata.v[0], cmap=cmap)

    # Titles
    fig.suptitle(f"Experimental DIC results\nTest case: {test.label_plot}", fontsize=FONT_SIZES["suptitle"])
    axes[0].set_title("Horizontal displacement [px]", fontsize=FONT_SIZES["subtitle"])
    axes[1].set_title("Vertical displacement [px]", fontsize=FONT_SIZES["subtitle"])

    for aa in axes:
        aa.set_aspect('equal')

    # Colorbars
    fig.colorbar(im1, ax=axes[0])
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.show()
    fig.savefig(test.output_save_dir / "exp_dic_plot.png", dpi=300, bbox_inches="tight")

    #20 px/mm   20 px = 1 mm
    # 61 px displacement ~ 3 mm - ok makes senseee

#run_dic_experimental(ExpTests.AIR)