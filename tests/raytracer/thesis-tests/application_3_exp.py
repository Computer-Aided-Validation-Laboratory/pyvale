from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

from global_utils import *
# pyvale modules
import pyvale.dic as dic

base_output_dir = f"app3_exp/dic/exp"
OUTPUT_DIR_PATH = test_dir(BASE_TEST_DIR, base_output_dir)
IMG_ACCESS = "thesis-data/app3_exp/experiment_data/"

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
# DIC
# ================================================================================

def run_dic_experimental():
    """
    Runs DIC on the experimental images.
    """
    
    
    ref_img_path = full_path(IMG_ACCESS + "static01b_C001H001S0001000001.tif")
    def_img_path = full_path(IMG_ACCESS + "sidDyn01b_C001H001S0001001372.tif")

    # Default pyvale DIC data to test if the problem is data or me using this
    #ref_img_path = dataset.dic_plate_with_hole_ref()
    #def_img_path = dataset.dic_plate_with_hole_def()

    roi = dic.RegionOfInterest(ref_image=ref_img_path)
    roi_file = OUTPUT_DIR_PATH / "roi.dat"

    # Select and save ROI
    #oi.interactive_selection(subset_size=29)
    #roi.save_array(filename=roi_file,binary=False)

    # Read already existing ROI
    """
    roi.read_array(filename=roi_file, binary=False)
    dic.calculate_2d(reference=ref_img_path,
                    deformed=def_img_path,
                    roi_mask=roi.mask,
                    seed=[400,400],
                    subset_size=29,
                    subset_step=19,
                    shape_function="AFFINE",
                    correlation_criteria="ZNSSD",
                    output_basepath=OUTPUT_DIR_PATH,
                    output_delimiter=",",
                    output_prefix="dic_results_")
    """
    dic_files = OUTPUT_DIR_PATH / "dic_results_*.csv"
    dicdata = dic.import_2d(data=dic_files, delimiter=",", binary=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 10))
    axes = axes.flatten()

    # First deformation image
    im1 = axes[0].pcolor(dicdata.ss_x, dicdata.ss_y, dicdata.u[0])
    im2 = axes[1].pcolor(dicdata.ss_x, dicdata.ss_y, dicdata.v[0])
    im3 = axes[2].pcolor(dicdata.ss_x, dicdata.ss_y, dicdata.cost[0])


    # Titles
    axes[0].set_title('u component (def0000.tiff)')
    axes[1].set_title('v component (def0000.tiff)')
    axes[2].set_title('cost (def0000.tiff)')

    for aa in axes:
        aa.set_aspect('equal')

    # Colorbars
    fig.colorbar(im1, ax=axes[0])
    fig.colorbar(im2, ax=axes[1])
    fig.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.show()

    #20 px/mm   20 px = 1 mm
    # 61 px displacement ~ 3 mm - ok makes senseee

#run_dic_experimental()