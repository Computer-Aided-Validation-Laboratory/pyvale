// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================

// STD library Header files
#include <cmath>
#include <omp.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <signal.h>
#include <functional>
#include <atomic>
#include <stdexcept>

// pybind header files
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// eigen header files
#include <Eigen/Dense>

// common_cpp header files
#include "../../common_cpp/dicsignalhandler.hpp"
#include "../../common_cpp/progressbar.hpp"
#include "../../common_cpp/defines.hpp"
#include "../../common_cpp/util.hpp"

// Program Header files
#include "./smooth.hpp"
#include "./strain.hpp"

namespace py = pybind11;

namespace strain {

    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    namespace {
        using SmoothFn = std::function<Eigen::VectorXd(const std::vector<double>&,
                                                       const std::vector<double>&,
                                                       const std::vector<double>&)>;

        Eigen::Vector2d eval_poly_gradient_at_centre(const int q, const Eigen::VectorXd &c,
                                                     const double x0, const double y0) {
            Eigen::Vector2d F = Eigen::Vector2d::Zero();

            if (q == 4) {
                F(0) = c[1] + c[3] * y0;
                F(1) = c[2] + c[3] * x0;
            }
            else if (q == 9) {
                F(0) = c[1] + c[3]*y0 + 2.0*c[4]*x0 + 2.0*c[6]*x0*y0
                        + c[7]*y0*y0 + 2.0*c[8]*x0*y0*y0;
                F(1) = c[2] + c[3]*x0 + 2.0*c[5]*y0 + c[6]*x0*x0
                        + 2.0*c[7]*x0*y0 + 2.0*c[8]*x0*x0*y0;
            }
            else {
                throw std::invalid_argument("Unsupported polynomial order");
            }

            return F;
        }

        Eigen::Matrix3d compute_tangent_fit_coordinates(Window &window,
                                                                     const int centre_idx) {
            const Eigen::Vector3d centre(window.x_mm[centre_idx],
                                         window.y_mm[centre_idx],
                                         window.z_mm[centre_idx]);

            Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
            for (size_t i = 0; i < window.x_mm.size(); ++i) {
                Eigen::Vector3d d(window.x_mm[i], window.y_mm[i], window.z_mm[i]);
                d -= centre;
                covariance += d * d.transpose();
            }

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
            if (solver.info() != Eigen::Success) {
                throw std::runtime_error("Tangent-basis eigen decomposition failed.");
            }

            Eigen::Matrix3d basis;
            basis.col(0) = solver.eigenvectors().col(2).normalized();
            basis.col(1) = solver.eigenvectors().col(1).normalized();
            basis.col(2) = basis.col(0).cross(basis.col(1));

            for (size_t i = 0; i < window.x_mm.size(); ++i) {
                Eigen::Vector3d d(window.x_mm[i], window.y_mm[i], window.z_mm[i]);
                d -= centre;
                window.x[i] = d.dot(basis.col(0));
                window.y[i] = d.dot(basis.col(1));
            }

            return basis;
        }

        void engine_impl(const py::array_t<int> &ss_x_arr,
                         const py::array_t<int> &ss_y_arr,
                         const py::array_t<double> &x_mm_arr,
                         const py::array_t<double> &y_mm_arr,
                         const py::array_t<double> &z_mm_arr,
                         const py::array_t<double> &u_arr,
                         const py::array_t<double> &v_arr,
                         const py::array_t<double> &w_arr,
                         const int nss_x, const int nss_y,
                         const int nimg, const int sw_size,
                         const int q, const std::string &form,
                         const std::vector<std::string> &filenames,
                         const common_util::SaveConfig &strain_save_conf,
                         const int debug_level,
                         const bool use_3d_coordinates) {

            signal(SIGINT, signalHandler);
            g_debug_level = debug_level;

            const int nwindows = nss_x * nss_y;

            int* ss_x = static_cast<int*>(ss_x_arr.request().ptr);
            int* ss_y = static_cast<int*>(ss_y_arr.request().ptr);
            double* x_mm = static_cast<double*>(x_mm_arr.request().ptr);
            double* y_mm = static_cast<double*>(y_mm_arr.request().ptr);
            double* z_mm = static_cast<double*>(z_mm_arr.request().ptr);
            double* u = static_cast<double*>(u_arr.request().ptr);
            double* v = static_cast<double*>(v_arr.request().ptr);
            double* w = static_cast<double*>(w_arr.request().ptr);

            SmoothFn smooth_window = (q == 4) ? SmoothFn(smooth::q4) : SmoothFn(smooth::q9);
            strain::Results results(nwindows);

            for (int img_num = 0; img_num < nimg; img_num++) {

                ProgressBar pbar(filenames[img_num], nwindows);
                std::atomic<int> current_progress(0);

                #pragma omp parallel for schedule(static)
                for (int sw = 0; sw < nwindows; sw++){

                    Window window(sw_size);

                    const int x0 = ss_x[sw];
                    const int y0 = ss_y[sw];
                    const int idx_3d_centre = nss_x*nss_y*img_num + sw;
                    results.x[sw] = x0;
                    results.y[sw] = y0;
                    results.x_mm[sw] = x_mm[idx_3d_centre];
                    results.y_mm[sw] = y_mm[idx_3d_centre];
                    results.z_mm[sw] = z_mm[idx_3d_centre];

                    if (use_3d_coordinates) {
                        results.valid_window[sw] = fill_window_3d(ss_x, ss_y, x_mm, y_mm, z_mm,
                                                                  u, v, w, img_num, sw, window,
                                                                  nss_x, nss_y, sw_size);
                    }
                    else {
                        results.valid_window[sw] = fill_window_2d(ss_x, ss_y, u, v, w, img_num,
                                                                  sw, window, nss_x, nss_y, sw_size);
                    }

                    Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
                    Eigen::Matrix3d eps = Eigen::Matrix3d::Zero();

                    if (results.valid_window[sw]){
                        if (use_3d_coordinates) {
                            const int centre_idx = (sw_size * sw_size) / 2;
                            Eigen::Matrix3d tangent_basis = compute_tangent_fit_coordinates(window, centre_idx);

                            Eigen::VectorXd uc = smooth_window(window.x, window.y, window.u);
                            Eigen::VectorXd vc = smooth_window(window.x, window.y, window.v);
                            Eigen::VectorXd wc = smooth_window(window.x, window.y, window.w);

                            F = compute_surface_F_3d(q, uc, vc, wc, tangent_basis);
                        }
                        else {
                            Eigen::VectorXd uc = smooth_window(window.x, window.y, window.u);
                            Eigen::VectorXd vc = smooth_window(window.x, window.y, window.v);
                            Eigen::VectorXd wc = smooth_window(window.x, window.y, window.w);

                            F = compute_F_2d(q, uc, vc, wc, 0.0, 0.0);
                        }

                        eps = compute_strain(form, F);
                        append_results(sw, results, x0, y0, F, eps, nwindows);
                    }

                    if (g_debug_level>0){
                        int progress = current_progress.fetch_add(1);
                        if (omp_get_thread_num() == 0) pbar.update(progress+1);
                    }
                }

                if(g_debug_level>0){
                    pbar.finish();
                }

                strain::save_to_disk(img_num, results, strain_save_conf, nwindows, nimg, filenames);
                
                if (stop_request) break;
            }
            
            raise_on_interrupt();
        }
    }

    void engine_2d(const py::array_t<int> &ss_x_arr,
                   const py::array_t<int> &ss_y_arr,
                   const py::array_t<double> &x_mm_arr,
                   const py::array_t<double> &y_mm_arr,
                   const py::array_t<double> &z_mm_arr,
                   const py::array_t<double> &u_arr,
                   const py::array_t<double> &v_arr,
                   const py::array_t<double> &w_arr,
                   const int nss_x, const int nss_y,
                   const int nimg, const int sw_size,
                   const int q, const std::string &form,
                   const std::vector<std::string> &filenames,
                   const common_util::SaveConfig &strain_save_conf,
                   const int debug_level) {
        engine_impl(ss_x_arr, ss_y_arr, x_mm_arr, y_mm_arr, z_mm_arr, u_arr, v_arr, w_arr,
                    nss_x, nss_y, nimg, sw_size, q, form, filenames, strain_save_conf,
                    debug_level, false);
    }

    void engine_3d(const py::array_t<int> &ss_x_arr,
                   const py::array_t<int> &ss_y_arr,
                   const py::array_t<double> &x_mm_arr,
                   const py::array_t<double> &y_mm_arr,
                   const py::array_t<double> &z_mm_arr,
                   const py::array_t<double> &u_arr,
                   const py::array_t<double> &v_arr,
                   const py::array_t<double> &w_arr,
                   const int nss_x, const int nss_y,
                   const int nimg, const int sw_size,
                   const int q, const std::string &form,
                   const std::vector<std::string> &filenames,
                   const common_util::SaveConfig &strain_save_conf,
                   const int debug_level) {
        engine_impl(ss_x_arr, ss_y_arr, x_mm_arr, y_mm_arr, z_mm_arr, u_arr, v_arr, w_arr,
                    nss_x, nss_y, nimg, sw_size, q, form, filenames, strain_save_conf,
                    debug_level, true);
    }

    void engine(const py::array_t<int> &ss_x_arr,
                const py::array_t<int> &ss_y_arr,
                const py::array_t<double> &x_mm_arr,
                const py::array_t<double> &y_mm_arr,
                const py::array_t<double> &z_mm_arr,
                const py::array_t<double> &u_arr,
                const py::array_t<double> &v_arr,
                const py::array_t<double> &w_arr,
                const int nss_x, const int nss_y,
                const int nimg, const int sw_size,
                const int q, const std::string &form,
                const std::vector<std::string> &filenames,
                const common_util::SaveConfig &strain_save_conf,
                const int debug_level) {
        engine_2d(ss_x_arr, ss_y_arr, x_mm_arr, y_mm_arr, z_mm_arr, u_arr, v_arr, w_arr,
                  nss_x, nss_y, nimg, sw_size, q, form, filenames, strain_save_conf,
                  debug_level);
    }

    bool fill_window_2d(int *ss_x, int *ss_y, double *u, double *v, double *w,
                        int img, int sw, Window &window,
                        int nss_x, int nss_y, int sw_size){

        const int swr = sw_size / 2;
        const int x0_idx = sw % nss_x;
        const int y0_idx = sw / nss_x;
        const int xmin = x0_idx - swr;
        const int xmax = x0_idx + swr;
        const int ymin = y0_idx - swr;
        const int ymax = y0_idx + swr;

        // check centre of strain window is within mask bounds
        if ((xmin < 0) || (xmax >= nss_x) || (ymin < 0) || (ymax >= nss_y)) return false;
        
        int widx = 0;
        for (int j = ymin; j <= ymax; j++){
            for (int i = xmin; i <= xmax; i++){

                // index in 3d results array
                int idx_2d = nss_x*j + i;
                int idx_3d = nss_x*nss_y*img + idx_2d;

                // check if all subsets in the strain window are not nan
                if (std::isnan(u[idx_3d]) || std::isnan(v[idx_3d]) || std::isnan(w[idx_3d])) return false;

                window.x[widx] = static_cast<double>(ss_x[idx_2d]);
                window.y[widx] = static_cast<double>(ss_y[idx_2d]);
                window.u[widx] = u[idx_3d];
                window.v[widx] = v[idx_3d];
                window.w[widx] = w[idx_3d];
                widx++;
            }
        }
        return true;
    }

    bool fill_window_3d(int *ss_x, int *ss_y, double *x_mm, double *y_mm, double *z_mm,
                        double *u, double *v, double *w,
                        int img, int sw, Window &window,
                        int nss_x, int nss_y, int sw_size){

        const int swr = sw_size / 2;
        const int x0_idx = sw % nss_x;
        const int y0_idx = sw / nss_x;
        const int xmin = x0_idx - swr;
        const int xmax = x0_idx + swr;
        const int ymin = y0_idx - swr;
        const int ymax = y0_idx + swr;

        if ((xmin < 0) || (xmax >= nss_x) || (ymin < 0) || (ymax >= nss_y)) return false;

        int widx = 0;
        for (int j = ymin; j <= ymax; j++){
            for (int i = xmin; i <= xmax; i++){
                int idx_2d = nss_x*j + i;
                int idx_3d = nss_x*nss_y*img + idx_2d;

                if (std::isnan(x_mm[idx_3d]) || std::isnan(y_mm[idx_3d]) || std::isnan(z_mm[idx_3d]) ||
                    std::isnan(u[idx_3d]) || std::isnan(v[idx_3d]) || std::isnan(w[idx_3d])) return false;

                window.x_mm[widx] = x_mm[idx_3d];
                window.y_mm[widx] = y_mm[idx_3d];
                window.z_mm[widx] = z_mm[idx_3d];
                window.u[widx] = u[idx_3d];
                window.v[widx] = v[idx_3d];
                window.w[widx] = w[idx_3d];
                widx++;
            }
        }
        return true;
    }

    Eigen::Matrix3d compute_F_2d(const int q,
                                        const Eigen::VectorXd &uc,
                                        const Eigen::VectorXd &vc,
                                        const Eigen::VectorXd &wc,
                                        const double x0,
                                        const double y0) {

        Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
        Eigen::Vector2d gu = eval_poly_gradient_at_centre(q, uc, x0, y0);
        Eigen::Vector2d gv = eval_poly_gradient_at_centre(q, vc, x0, y0);
        Eigen::Vector2d gw = eval_poly_gradient_at_centre(q, wc, x0, y0);

        F(0,0) = 1.0 + gu(0);
        F(0,1) = gu(1);
        F(1,0) = gv(0);
        F(1,1) = 1.0 + gv(1);
        F(2,0) = gw(0);
        F(2,1) = gw(1);
        F(2,2) = 1.0;

        return F;
    }

    Eigen::Matrix3d compute_surface_F_3d(const int q,
                                                const Eigen::VectorXd &uc,
                                                const Eigen::VectorXd &vc,
                                                const Eigen::VectorXd &wc,
                                                const Eigen::Matrix3d &tangent_basis) {

        Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
        F.block<1,2>(0,0) = eval_poly_gradient_at_centre(q, uc, 0.0, 0.0).transpose();
        F.block<1,2>(1,0) = eval_poly_gradient_at_centre(q, vc, 0.0, 0.0).transpose();
        F.block<1,2>(2,0) = eval_poly_gradient_at_centre(q, wc, 0.0, 0.0).transpose();


        return I + F * tangent_basis.transpose();
    }

    Eigen::Matrix3d compute_strain(const std::string& form, const Eigen::Matrix3d& F) {
        if (form == "GREEN")        return green(F);
        else if (form == "ALMANSI") return almansi(F);
        else if (form == "HENCKY")  return hencky(F);
        else if (form == "BIOT_EULER") return biot_euler(F);
        else if (form == "BIOT_LAGRANGE") return biot_lagrange(F);

        std::cerr << "Unknown Strain formulation: '" << form << "'." << std::endl;
        return Eigen::Matrix3d::Zero();
    }


    inline Eigen::Matrix3d green(const Eigen::Matrix3d &F){
        return 0.5 * (F.transpose() * F - I);
    }


    inline Eigen::Matrix3d hencky(const Eigen::Matrix3d &F){
        Eigen::Matrix3d C = F.transpose() * F;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(C);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Eigen decomposition failed.");

        // Get eigenvectors and sqrt-eigenvalues
        const Eigen::Matrix3d Q = solver.eigenvectors();
        const Eigen::Vector3d eigvals = solver.eigenvalues();

        return Q * (0.5 * eigvals.array().log().matrix().asDiagonal()) * Q.transpose();
    }




    inline Eigen::Matrix3d almansi(const Eigen::Matrix3d &F){
        Eigen::Matrix3d B = F * F.transpose();
        Eigen::Matrix3d B_inv = B.inverse();
        return 0.5 * (I - B_inv); 
    }





    inline Eigen::Matrix3d biot_euler(const Eigen::Matrix3d &F){

        Eigen::Matrix3d C = F * F.transpose();

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(C);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Eigen decomposition failed.");

        // U = sqrt(C) = Q * sqrt(D) * Q^T
        Eigen::Matrix3d D_sqrt = solver.eigenvalues().cwiseSqrt().asDiagonal();
        Eigen::Matrix3d U = solver.eigenvectors() * D_sqrt * solver.eigenvectors().transpose();

        return U - I;

    }




    inline Eigen::Matrix3d biot_lagrange(const Eigen::Matrix3d &F){

        Eigen::Matrix3d C = F.transpose() * F;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(C);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Eigen decomposition failed.");

        // U = sqrt(C) = Q * sqrt(D) * Q^T
        Eigen::Matrix3d D_sqrt = solver.eigenvalues().cwiseSqrt().asDiagonal();
        Eigen::Matrix3d U = solver.eigenvectors() * D_sqrt * solver.eigenvectors().transpose();

        return U - I;

    }


    void append_results(int sw, strain::Results &results,
                        const int x0, const int y0,
                        const Eigen::Matrix3d &F,
                        const Eigen::Matrix3d &eps,
                        const int nwindows){

        results.F[6*sw+0] = F(0,0);
        results.F[6*sw+1] = F(0,1);
        results.F[6*sw+2] = F(1,0);
        results.F[6*sw+3] = F(1,1);
        results.F[6*sw+4] = F(2,0);
        results.F[6*sw+5] = F(2,1);

        results.strain[4*sw+0] = eps(0,0);
        results.strain[4*sw+1] = eps(0,1);
        results.strain[4*sw+2] = eps(1,0);
        results.strain[4*sw+3] = eps(1,1);
    }

    void save_to_disk(int img_num,
                      const strain::Results &results,
                      const common_util::SaveConfig &strain_save_conf,
                      const int nwindows,
                      const int nimg,
                      const std::vector<std::string> filenames)
    {
        const std::string delimiter = strain_save_conf.delimiter;

        std::stringstream outfile_str;
        std::ofstream outfile;

        std::string file_ext;
        if (strain_save_conf.binary) file_ext = ".dic3d";
        else file_ext = ".csv";

        std::string full_filename = filenames[img_num];
        size_t dot_pos = full_filename.find(".");
        if (dot_pos != std::string::npos) {
            full_filename = full_filename.substr(0, dot_pos);
        }

        outfile_str << strain_save_conf.basepath << "/"
                    << strain_save_conf.prefix
                    << full_filename
                    << file_ext;


        outfile << std::fixed << std::setprecision(8);

        const int def_size = 6;
        const int tensor_size = 4;

        if (strain_save_conf.binary)
        {
            outfile.open(outfile_str.str(), std::ios::binary);

            for (int i = 0; i < nwindows; ++i)
            {
                common_util::write_int(outfile, results.x[i]);
                common_util::write_int(outfile, results.y[i]);
                common_util::write_dbl(outfile, results.x_mm[i]);
                common_util::write_dbl(outfile, results.y_mm[i]);
                common_util::write_dbl(outfile, results.z_mm[i]);

                for (int k = 0; k < def_size; ++k)
                    common_util::write_dbl(outfile, results.F[def_size * i + k]);

                for (int k = 0; k < tensor_size; ++k)
                    common_util::write_dbl(outfile, results.strain[tensor_size * i + k]);
            }

            outfile.close();
        }
        else
        {
            outfile.open(outfile_str.str());

            outfile << "\"window_x\"" << delimiter
                    << "\"window_y\"" << delimiter
                    << "\"x_mm\"" << delimiter
                    << "\"y_mm\"" << delimiter
                    << "\"z_mm\"" << delimiter
                    << "\"def_grad_00\"" << delimiter
                    << "\"def_grad_01\"" << delimiter
                    << "\"def_grad_10\"" << delimiter
                    << "\"def_grad_11\"" << delimiter
                    << "\"def_grad_20\"" << delimiter
                    << "\"def_grad_21\"" << delimiter
                    << "\"eps_00\"" << delimiter
                    << "\"eps_01\"" << delimiter
                    << "\"eps_10\"" << delimiter
                    << "\"eps_11\"\n";

            for (int i = 0; i < nwindows; i++)
            {
                if (results.valid_window[i])
                {
                    outfile << results.x[i] << delimiter;
                    outfile << results.y[i] << delimiter;
                    outfile << results.x_mm[i] << delimiter;
                    outfile << results.y_mm[i] << delimiter;
                    outfile << results.z_mm[i] << delimiter;

                    for (int k = 0; k < def_size; ++k)
                    {
                        outfile << results.F[def_size * i + k] << delimiter;
                    }

                    for (int k = 0; k < tensor_size; ++k)
                    {
                        outfile << results.strain[tensor_size * i + k];
                        if (k != tensor_size - 1) outfile << delimiter;
                    }

                    outfile << "\n";
                }
            }

            outfile.close();
        }
    }

} // namespace strain
