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

    
    extern std::vector<std::vector<double>> shift_x;
    extern std::vector<std::vector<double>> shift_y;

    struct NeighborDist {
        int index;
        double dist_sq;
    };


    void init(std::vector<util::SubsetData> &ssdata, 
              const bool *img_roi, const util::Config conf);

    void mgwd(const std::vector<util::SubsetData> &ssdata,
              const double *img_def, const double *img_ref, 
              const util::Config conf);

    
    void get_4nn(std::vector<int> &neighlist,
                 const util::SubsetData ssdata,
                 const util::SubsetData ssdata_prev);

    void get_neighlist(std::vector<int> &neighlist,
                       const util::SubsetData ssdata,
                       const util::SubsetData ssdata_prev);

    inline int fftshift(int peak, int ss_size);
    
    inline void destroy_fftw_plans(std::vector<fftw_plan>& plans);
    inline void free_fftw_arrays(std::vector<fftw_complex*>& vec);
    double debugcost(util::Subset &ss_ref, util::Subset &ss_def);
}

#endif // DICFOURIER_H
