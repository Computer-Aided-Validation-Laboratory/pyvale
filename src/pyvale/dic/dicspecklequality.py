# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================


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

        # Internal cache for speckle sizes
        self._speckle_sizes = None
        self._subset_average = None
        self._xvalues = None
        self._yvalues = None

        #TODO: regoin of interest for staticistics
        # this needs to be a 'sub' array of the overall image



    def mean_intensity_gradient(self) -> float:
        """ 
        Mean Intensity Gradient. Based on the below: 
        https://www.sciencedirect.com/science/article/abs/pii/S0143816613001103 

        Returns:
        mean_intensity_gradient (float): float value for mean_intensity gradient
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

        
        Returns:
        shannon_entropy (float): float value for shannon entropy
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

        return None





    def speckle_size(self) -> tuple[int, np.ndarray, np.ndarray]:
        """
        Calculates the Speckle sizes using a binary map calculaed from otsu threshholding
        (https://learnopencv.com/otsu-thresholding-with-opencv/)

        
        Returns:
        tuple containing:
        num_speckles                   (int): total number of speckles identified in the binary map
        equivalent_diameters    (np.ndarray): Speckle diameter if circle with same area
        labeled_speckles        (np.ndarray): Label of the connected elements within speckle
        """

        # calculate binary map using otsu thresholding with opencv
        _, binary_image = cv2.threshold(self.pattern,
                                        0,
                                        self.gray_level - 1, 
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Label connected components (speckles)
        labeled_speckles, num_speckles = ndi.label(binary_image)
        speckle_sizes = np.array(ndi.sum(binary_image > 0, 
                                         labeled_speckles, 
                                         index=np.arange(1, num_speckles + 1)))
        
        equivalent_diameters = 2 * np.sqrt(speckle_sizes / np.pi)

        # assign values to cached tuple
        self._speckle_sizes = (num_speckles, equivalent_diameters, labeled_speckles)

        # Raise exception if there's no speckles
        if num_speckles ==  0:
            raise ValueError("No speckles identified.")
        
        return self._speckle_sizes


    def speckle_size_plot(self) -> None:

        # get speckle sizes if not computed already
        if self._speckle_sizes is None:
            self.speckle_size()

        # assign each speckle to a classification group.
        # Group is jst the 'size' unsure whether to bin to discrete sizes
        classifications = self._classify_speckles()

        # plotting
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

        im1 = axes[0].imshow(self.pattern, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title("Speckle Pattern")
        axes[0].axis("off")
        fig.colorbar(im1,ax=axes[0],fraction=0.046, pad=0.04)


        im2 = axes[1].imshow(classifications, cmap="turbo", vmin=0, vmax=15)
        axes[1].set_title("Speckle Size")
        axes[1].axis("off")
        fig.colorbar(im2,ax=axes[1],fraction=0.046, pad=0.04)

        plt.show()

        return None


    def _classify_speckles(self) -> np.ndarray:
        """
        Calculates the Speckle sizes using a binary map calculaed from otsu threshholding
        (https://learnopencv.com/otsu-thresholding-with-opencv/)


        Returns:
        classifications  (np.ndarray): speckle sizes classified by bin sizes for plots.
                                       To discuss which bins are appropriate.
                                       My proposed bins:
                                       0-3 small, 3-5 ideal, 5 < big.
        """


        num_speckles, speckle_sizes, labeled_speckles = self._speckle_sizes
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

        return classifications


    def balance_subset(self) -> np.ndarray:

        # dont use subsets if rows/cols < edge_cutoff
        edge_cutoff = 100

        min_x = self.subset_size // 2
        min_y = self.subset_size // 2
        max_x = self.pattern.shape[1] - self.subset_size // 2
        max_y = self.pattern.shape[0] - self.subset_size // 2

        # image coordiantes array containing the central pixel for each subset
        self._xvalues = np.arange(min_x+edge_cutoff, max_x-edge_cutoff, self.subset_step)
        self._yvalues = np.arange(min_y+edge_cutoff, max_y-edge_cutoff, self.subset_step)
        
        # init array to store black/white balance value
        shape = (len(self._yvalues), len(self._xvalues))
        self._subset_average = np.zeros(shape)


        # looping over the subsets
        for i, x in enumerate(self._xvalues):
            for j, y in enumerate(self._yvalues):

                subset = extract_subset(self.pattern, x, y, self.subset_size)

                # plt.figure()
                # plt.imshow(subset)
                # plt.show()

                self._subset_average[j,i] = np.average(subset) / self.gray_level

        return self._subset_average
    

    def balance_image(self) -> float:

        avg = np.mean(self.pattern) / self.gray_level

        return avg

    def balance_subset_avg(self) -> float:

        if self._subset_average is None:
            self.balance_subset()

        subset_avg = np.mean(self._subset_average)

        return subset_avg



    def balance_subset_plot(self) -> None:

        if self._subset_average is None:
            self.balance_subset()

        plt.figure(figsize=(10, 10))
        plt.imshow(self.pattern, cmap='gray', interpolation='none')
        extent = [self._xvalues[0], self._xvalues[-1], self._yvalues[-1], self._yvalues[0]]  # Match coordinates
        plt.imshow(self._subset_average, cmap='jet', alpha=0.3, extent=extent, interpolation='none')
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
    Parameters
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
