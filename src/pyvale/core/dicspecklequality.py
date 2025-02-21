"""
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
"""


import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage as ndi
from numba import jit



class DICSpeckleQuality:

    def __init__(self, pattern: np.ndarray, subset_size: int, subset_step: int, gray_level: int):
        self.pattern = pattern
        self.subset_size = subset_size
        self.subset_step = subset_step
        self.gray_level = gray_level

        #TODO: regoin of interest for staticistics
        # this needs to be a 'sub' array of the overall image



    def mean_intensity_gradient(self) -> float:
        """ 
        Mean Intensity Gradient. Based on the below: 
        https://www.sciencedirect.com/science/article/abs/pii/S0143816613001103 
        """

        gradient_x, gradient_y = np.gradient(self.pattern)

        # mag
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        # plot for debugging
        plt.figure()
        plt.imshow(gradient_magnitude)
        plt.colorbar(label='Magnitude')
        plt.show()

        #get mean of 2d array.
        mean_gradient = np.mean(gradient_magnitude)

        return mean_gradient

    
    def shannon_entropy(self) -> float:
        """ 
        shannon entropy for speckle patterns. Based on the below: 
        https://www.sciencedirect.com/science/article/abs/pii/S0030402615007950 
        """

        #count occurances of each value. bincount doesn't like 2d arrays. flatten to 1d.
        bins = np.bincount(self.pattern.flatten()) / self.pattern.size

        # reset shannon_entropy
        shannon_entropy = 0.0

        # loop over gray leves
        for i in range(0,2):
            shannon_entropy -= bins[i] * math.log2(bins[i])

        return shannon_entropy

    def gray_level_histogram(self) -> None:
        """
        Count the number of occurrences of each gray value.
        plot results as a histogram
        """

        # Count occurrences of each gray value
        unique_values, counts = np.unique(self.pattern, return_counts=True)

        # Plot histogram
        plt.figure(figsize=(8, 5))
        plt.bar(unique_values, counts, width=1.0, color='gray', edgecolor='black')
        plt.title('Histogram of Gray Levels')
        plt.xlabel('Gray Level (0-255)')
        plt.ylabel('Count')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()




    def calculate_speckle_size(self) -> tuple[np.ndarray, np.ndarray, int]:

        # Convert speckle to binary img (https://learnopencv.com/otsu-thresholding-with-opencv/)
        #TODO: pass graylevel range to this function
        _, binary_image = cv2.threshold(self.pattern.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Label connected components (speckles)
        labeled_speckles, num_speckles = ndi.label(binary_image)
        
        # sizes of speckles
        speckle_sizes = np.array(ndi.sum(binary_image > 0, labeled_speckles, index=np.arange(1, num_speckles + 1)))
        
        # Compute equivalent diameter (circle with same area)
        equivalent_diameters = 2 * np.sqrt(speckle_sizes / np.pi)

        
        return labeled_speckles, equivalent_diameters, num_speckles


    def classify_speckles(self, labeled_speckles, speckle_sizes, num_speckles) -> np.ndarray:
    
    
    
        classifications = np.zeros_like(labeled_speckles, dtype=np.uint8)
        
        #TODO: Not sure whether to bin into three catagorories:
        # 0-3 kinda small, 3-5 ideal, 5 < kinda big.
        # I'm leaving the logic in to deal with this but going to assume continous is probs best
        for i in range(1, num_speckles + 1):
            size = speckle_sizes[i - 1]
            if size <= 3:
                classifications[labeled_speckles == i] = size #1 
            elif 3 < size <= 5:
                classifications[labeled_speckles == i] = size #3 
            else:
                classifications[labeled_speckles == i] = size #2 

        return ndi.gaussian_filter(classifications, 0.5)
    



    def plot_results(self, image_array, classifications) -> None:

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

        im1 = axes[0].imshow(image_array, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title("Speckle Pattern")
        axes[0].axis("off")
        fig.colorbar(im1,ax=axes[0],fraction=0.046, pad=0.04)


        im2 = axes[1].imshow(classifications, cmap="turbo", vmin=0, vmax=15)
        axes[1].set_title("Speckle Size")
        axes[1].axis("off")
        fig.colorbar(im2,ax=axes[1],fraction=0.046, pad=0.04)

        plt.show()

        return None






    def black_white_balance(self) -> None:


        # dont use subsets if rows/cols < edge_cutoff
        edge_cutoff = 100

        min_x = self.subset_size // 2
        min_y = self.subset_size // 2
        max_x = self.pattern.shape[1] - self.subset_size // 2
        max_y = self.pattern.shape[0] - self.subset_size // 2

        # image coordiantes array containing the central pixel for each subset
        x_values = np.arange(min_x+edge_cutoff, max_x-edge_cutoff, self.subset_step)
        y_values = np.arange(min_y+edge_cutoff, max_y-edge_cutoff, self.subset_step)
        
        # init array to store black/white balance value
        shape = (len(y_values), len(x_values))
        subset_average = np.zeros(shape)
        # ic(subset_average.shape)



        # looping over the subsets
        for i, x in enumerate(x_values):
            for j, y in enumerate(y_values):

                subset = extract_subset(self.pattern, x, y, self.subset_size)

                # plt.figure()
                # plt.imshow(subset)
                # plt.show()

                subset_average[j,i] = np.average(subset) / self.gray_level
                print(subset_average[j,i])

        plt.figure(figsize=(10, 10))
        plt.imshow(self.pattern, cmap='gray', interpolation='none')
        extent = [x_values[0], x_values[-1], y_values[-1], y_values[0]]  # Match coordinates
        # plt.imshow(subset_average, cmap='jet', vmin=0.0, vmax=1.0, alpha=0.3, extent=extent, interpolation='none')
        plt.imshow(subset_average, cmap='jet', alpha=0.3, extent=extent, interpolation='none')
        plt.xlim(0,self.pattern.shape[1])
        plt.ylim(self.pattern.shape[0],0)
        plt.colorbar(label='Normalized Subset Average')
        plt.title("Black/White Balance Overlay")
        plt.show()

        return None



#TODO: This is going to become c++ at some point.
# I think this is OK to keep in python for calculation of black/white balance
@jit(nopython=True)
def extract_subset(image: np.ndarray, x: int, y: int, subset_size: int) -> np.ndarray:
    """
    Parameters:
    x (int): x-coord of subset center in image
    y (int): y-coord of subset center in image

    """

    half_size = subset_size // 2

    # reference image subset
    x1, x2 = x - half_size, x + half_size + 1
    y1, y2 = y - half_size, y + half_size + 1

    # Ensure indices are within bounds
    #TODO: Update this when implementing ROI
    if (x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]):
        raise ValueError(f"Subset exceeds image boundaries.\nSubset Pixel Range:\n"
                        f"x1: {x1}\n"
                        f"x2: {x2}\n"
                        f"y1: {y1}\n"
                        f"y2: {y2}")

    # Extract subsets
    subset = image[y1:y2, x1:x2]

    return subset