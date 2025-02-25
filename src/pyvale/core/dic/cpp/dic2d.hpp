// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#include <iostream>
#include <cmath>
#include "spline.h"


namespace dic2d {

    void calculate_displacements() {


        // get the interpolation of the entire reference image
        interpolator = correlation.spline_interpolation_object(reference_image, 3)

        min_x = subset_size // 2
        min_y = subset_size // 2
        max_x = reference_image.shape[1] - subset_size // 2
        max_y = reference_image.shape[0] - subset_size // 2

        // dont use subsets if rows/cols < 10
        edge_cutoff = 50

        x_values = np.arange(min_x+edge_cutoff, max_x-edge_cutoff, subset_step)
        y_values = np.arange(min_y+edge_cutoff, max_y-edge_cutoff, subset_step)
        shape = (len(y_values), len(x_values), 6) 

        total_iterations = x_values.shape[0] * y_values.shape[0]

        // Initialize 2D arrays
        p_arr = np.zeros(shape)
        ssd_arr = np.zeros((len(y_values), len(x_values)))

        progress_bar = tqdm(total=total_iterations, desc=f"{'Searching for deformed subsets in the interpolated reference image using scipy.optimize.minimize':150}",position=0)


        p = np.array([0.0,0.0,0.0,0.0,0.0,0.0])

        // looping over the subsets
        for i, x in enumerate(x_values):
            for j, y in enumerate(y_values):

                subset = correlation.subset(deformed_image, x, y, subset_size)

                half_size = subset_size // 2

                // reference image subset
                x1, x2 = x - half_size, x + half_size + 1
                y1, y2 = y - half_size, y + half_size + 1

                // list of coordinates 
                coords_x = np.arange(x1,x2,1)
                coords_y = np.arange(y1,y2,1)

                //pixel coordinates of reference subset
                xx, yy = np.meshgrid(coords_x,coords_y)

                sol = minimize(subset_search_affine_minimizer, p, args=(subset,interpolator,xx,yy),bounds=bounds)
                p = sol.x
                ssd_val = sol.fun


                // value is negative because its deformed subset looking searching in reference image
                p_arr[j,i,0:6]  = p
                ssd_arr[j,i] = ssd_val

                progress_bar.update(1)

    return p_arr, ssd_arr



    }





}