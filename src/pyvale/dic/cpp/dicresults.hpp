// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

#ifndef DICRESULTS_H
#define DICRESULTS_H

// STD library Header files
#include <vector>
#include <cstdint>
#include <cstddef>

// commoncpp Header files
#include "../../commoncpp/util.hpp"

// DIC Header files
#include "./dicsubset.hpp"
#include "./dicoptimizerbase.hpp"


class ResultArrays {

    private:


    public:

        int num_ss;
        int num_params;
        bool stereo;

        // result arrays.
        std::vector<int> niter;
        std::vector<double> u; 
        std::vector<double> v;
        std::vector<double> p;
        std::vector<double> ftol;
        std::vector<double> xtol;
        std::vector<double> cost;
        std::vector<uint8_t> conv;
        std::vector<uint8_t> valid_match;

        // incremental tracking;
        // std::vector<double> u_last_good;
        // std::vector<double> v_last_good;
        // std::vector<double> du_dt;
        // std::vector<double> dv_dt;
        // std::vector<int> last_success_frame;
        // std::vector<char> has_good_history;

        // world coordinates
        std::vector<double> x_world; 
        std::vector<double> y_world;
        std::vector<double> z_world;
        std::vector<double> u_world; 
        std::vector<double> v_world;
        std::vector<double> w_world;
        std::vector<double> epi_dist_px;

        // constructors
        ResultArrays() = default;
        ResultArrays(int num_ss,
                     int num_params,
                     bool stereo);


        void append(OptResult &res, const int ss);


        void reset();


        //void get_latest_matches(const ResultArrays &results_def, const int img_num_def);

        void write_to_disk_2d(const common_util::SaveConfig &saveconf,
                              const SubsetGrid &ss_grid,
                              const std::string &filename);


        void write_to_disk_stereo(const ResultArrays &stereo,
                                  const common_util::SaveConfig &saveconf,
                                  const SubsetGrid &ss_grid,
                                  const std::string &filename);


};


/** Owning frame-major result storage exposed through zero-copy NumPy views. */
class InMemoryResults {
public:
    std::size_t num_frames = 0;
    std::size_t num_ss = 0;
    std::size_t num_ss_x = 0;
    std::size_t num_ss_y = 0;
    bool stereo = false;

    std::vector<int32_t> ss_x;
    std::vector<int32_t> ss_y;
    std::vector<double> u, v, mag, cost, ftol, xtol;
    std::vector<int32_t> niter;
    std::vector<uint8_t> converged;

    std::vector<double> disparity_u, disparity_v, disparity_mag;
    std::vector<double> x_mm, y_mm, z_mm, u_mm, v_mm, w_mm;
    std::vector<uint8_t> stereo_converged;
    std::vector<double> stereo_cost, stereo_ftol, stereo_xtol, epi_dist_px;
    std::vector<int32_t> stereo_niter;

    InMemoryResults(std::size_t frames, std::size_t points, bool is_stereo);
    void set_coordinates(const SubsetGrid &grid);
    void store_frame(std::size_t frame, const ResultArrays &source,
                     const SubsetGrid &grid,
                     const common_util::SaveConfig &saveconf);
    void store_stereo_frame(std::size_t frame,
                            const ResultArrays &temporal,
                            const ResultArrays &stereo_results,
                            const SubsetGrid &grid,
                            const common_util::SaveConfig &saveconf);
};



#endif // DICRESULTS_H
