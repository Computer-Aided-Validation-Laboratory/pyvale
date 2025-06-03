// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICFOURIER_H
#define DICFOURIER_H

// STD library Header files
#include <fftw3.h>

// Program Header files
#include "./defines.hpp"
#include "./dicutil.hpp"

namespace fourier {

    // A helper structure for sorting
    struct NeighborDist {
        int index;
        double dist_sq;
    };


    void init(std::vector<util::SubsetData> ssdata, 
              const bool *img_roi, const util::Config conf, 
              const std::vector<int> &windows);

    void mgwd(const std::vector<util::SubsetData> ssdata,
              const double *img_def, const double *img_ref, 
              const util::Config conf);

    void cleanup();

    void get_4nn(std::vector<int> &neighlist,
                 const util::SubsetData ssdata,
                 const util::SubsetData ssdata_prev);

    void get_neighlist(std::vector<int> &neighlist,
                       const util::SubsetData ssdata,
                       const util::SubsetData ssdata_prev);

    inline int fftshift(int peak, int w);

    // TODO: Bilinear Interpolation for seeding
    //void get_prev_shift_vals(double *sx, double *sy,
    //                         double &tx, double &ty,
    //                         const int ss_x, const int ss_y, 
    //                         const int i, const int prev_step, 
    //                         const int num_ss_x, const std::vector<bool> &mask);
    
    //void get_grid_vals(int ss_x, int ss_y, int step, int grid_vals[4]);
    //void get_grid_indx(const int grid_vals[4], int step, int num_ss_x, int grid_indx[4]);
    //void get_valid_neigh(std::vector<bool> &mask, const int grid_indx[4], int valid_neigh[4]);
    //void adjust_boundary_look(int ss_x, int ss_y, int step, int grid_vals[4]);

}

#endif // DICFOURIER_H
