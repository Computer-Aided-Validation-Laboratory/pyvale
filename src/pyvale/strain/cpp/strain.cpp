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

// commoncpp header files
#include "../../commoncpp/dicsignalhandler.hpp"
#include "../../commoncpp/progressbar.hpp"
#include "../../commoncpp/defines.hpp"
#include "../../commoncpp/util.hpp"

// Program Header files
#include "./smooth.hpp"
#include "./strain.hpp"

namespace py = pybind11;

namespace strain {

    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    namespace {
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
                                                                     const Eigen::Vector3d &centre) {

            Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
            for (size_t i = 0; i < window.x_mm.size(); ++i) {
                Eigen::Vector3d d(window.x_mm[i], window.y_mm[i], window.z_mm[i]);
                d -= centre;
                covariance += d * d.transpose();
            }

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(covariance);
            if (solver.info() != Eigen::Success) {
                return Eigen::Matrix3d::Constant(NAN);
            }

            Eigen::Vector3d normal = solver.eigenvectors().col(0).normalized();
            if (normal.dot(Eigen::Vector3d::UnitZ()) < 0.0) {
                normal = -normal;
            }

            const auto project_to_surface = [&normal](const Eigen::Vector3d &axis) {
                return axis - normal * normal.dot(axis);
            };

            Eigen::Vector3d tangent_x = project_to_surface(Eigen::Vector3d::UnitX());
            constexpr double MIN_TANGENT_NORM = 0.0017453283658983088; // sin(0.1 degrees)

            if (tangent_x.norm() < MIN_TANGENT_NORM) {
                tangent_x = project_to_surface(Eigen::Vector3d::UnitZ());
            }

            if (tangent_x.norm() < MIN_TANGENT_NORM) {
                return Eigen::Matrix3d::Constant(NAN);
            }

            tangent_x.normalize();

            Eigen::Vector3d tangent_y = normal.cross(tangent_x).normalized();

            Eigen::Matrix3d basis;
            basis.col(0) = tangent_x;
            basis.col(1) = tangent_y;
            basis.col(2) = normal;

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
                         const bool use_3d_coordinates,
                         const double partial_window) {

            if (!(partial_window >= 0.0 && partial_window <= 1.0))
                throw std::invalid_argument("partial_window must be between 0 and 1 inclusive.");

            if (sw_size <= 0 || sw_size % 2 == 0 || (q != 4 && q != 9))
                throw std::invalid_argument("Expected a positive odd window size and Q4 or Q9.");

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


            for (int img_num = 0; img_num < nimg; img_num++) {
                strain::Results results(nwindows);

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
                                                                  nss_x, nss_y, sw_size, partial_window);
                    }
                    else {
                        results.valid_window[sw] = fill_window_2d(ss_x, ss_y, u, v, w, img_num,
                                                                  sw, window, nss_x, nss_y, sw_size, partial_window);
                    }

                    Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
                    Eigen::Matrix3d eps = Eigen::Matrix3d::Zero();

                    if (results.valid_window[sw]) {
                        Eigen::Matrix3d basis = Eigen::Matrix3d::Identity();
                        if (use_3d_coordinates) {
                            const Eigen::Vector3d centre(x_mm[idx_3d_centre], y_mm[idx_3d_centre],
                                                         z_mm[idx_3d_centre]);
                            basis = compute_tangent_fit_coordinates(window, centre);
                        }

                        Eigen::MatrixXd coefficients;

                        const bool basis_valid = basis.allFinite();

                        const bool fit_valid = smooth::fit_displacements(window.x,
                                                                         window.y,
                                                                         window.u,
                                                                         window.v,
                                                                         window.w,
                                                                         q,
                                                                         coefficients);

                        results.valid_window[sw] = basis_valid && fit_valid;

                        if (results.valid_window[sw]) {

                            if (use_3d_coordinates) {
                                F = compute_surface_F_3d(q, coefficients.col(0), coefficients.col(1),
                                                         coefficients.col(2), basis);
                            }
                            else {
                                F = compute_F_2d(q, coefficients.col(0), coefficients.col(1),
                                                 coefficients.col(2), 0.0, 0.0);
                            }

                            eps = compute_strain(form, F);
                            results.valid_window[sw] = F.allFinite() && eps.allFinite();

                            if (results.valid_window[sw]) {
                                append_results(sw, results, x0, y0, F, eps, nwindows);
                            }
                        }
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
                   const int debug_level,
                   const double partial_window) {

        engine_impl(ss_x_arr, ss_y_arr, x_mm_arr, y_mm_arr, z_mm_arr, u_arr, v_arr, w_arr,
                    nss_x, nss_y, nimg, sw_size, q, form, filenames, strain_save_conf,
                    debug_level, false, partial_window);

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
                   const int debug_level,
                   const double partial_window) {

        engine_impl(ss_x_arr, ss_y_arr, x_mm_arr, y_mm_arr, z_mm_arr, u_arr, v_arr, w_arr,
                    nss_x, nss_y, nimg, sw_size, q, form, filenames, strain_save_conf,
                    debug_level, true, partial_window);

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
                const int debug_level,
                const double partial_window) {

        engine_2d(ss_x_arr, ss_y_arr, x_mm_arr, y_mm_arr, z_mm_arr, u_arr, v_arr, w_arr,
                  nss_x, nss_y, nimg, sw_size, q, form, filenames, strain_save_conf,
                  debug_level, partial_window);

    }

    bool fill_window_2d(int *ss_x, int *ss_y, double *u, double *v, double *w,
                        int img, int sw, Window &window,
                        int nss_x, int nss_y, int sw_size, double partial_window){

        const int swr = sw_size / 2;
        const int x0_idx = sw % nss_x;
        const int y0_idx = sw / nss_x;
        const int xmin = x0_idx - swr;
        const int xmax = x0_idx + swr;
        const int ymin = y0_idx - swr;
        const int ymax = y0_idx + swr;


        
        int widx = 0;
        for (int j = std::max(0, ymin); j <= std::min(nss_y - 1, ymax); j++){
            for (int i = std::max(0, xmin); i <= std::min(nss_x - 1, xmax); i++){

                // index in 3d results array
                int idx_2d = nss_x*j + i;
                int idx_3d = nss_x*nss_y*img + idx_2d;

                // check if all subsets in the strain window are not nan
                if (!std::isfinite(u[idx_3d]) || !std::isfinite(v[idx_3d]) || !std::isfinite(w[idx_3d])) continue;

                window.x[widx] = static_cast<double>(ss_x[idx_2d]) - ss_x[sw];
                window.y[widx] = static_cast<double>(ss_y[idx_2d]) - ss_y[sw];
                window.u[widx] = u[idx_3d];
                window.v[widx] = v[idx_3d];
                window.w[widx] = w[idx_3d];
                widx++;
            }
        }
        window.x.resize(widx);
        window.y.resize(widx);
        window.x_mm.resize(widx);
        window.y_mm.resize(widx);
        window.z_mm.resize(widx);
        window.u.resize(widx);
        window.v.resize(widx);
        window.w.resize(widx);
        return widx >= std::ceil(partial_window * sw_size * sw_size);
    }

    bool fill_window_3d(int *ss_x, int *ss_y, double *x_mm, double *y_mm, double *z_mm,
                        double *u, double *v, double *w,
                        int img, int sw, Window &window,
                        int nss_x, int nss_y, int sw_size, double partial_window){

        const int centre = nss_x * nss_y * img + sw;
        if (!std::isfinite(x_mm[centre]) || !std::isfinite(y_mm[centre]) ||
            !std::isfinite(z_mm[centre])) return false;

        const int swr = sw_size / 2;
        const int x0_idx = sw % nss_x;
        const int y0_idx = sw / nss_x;
        const int xmin = x0_idx - swr;
        const int xmax = x0_idx + swr;
        const int ymin = y0_idx - swr;
        const int ymax = y0_idx + swr;



        int widx = 0;
        for (int j = std::max(0, ymin); j <= std::min(nss_y - 1, ymax); j++){
            for (int i = std::max(0, xmin); i <= std::min(nss_x - 1, xmax); i++){
                int idx_2d = nss_x*j + i;
                int idx_3d = nss_x*nss_y*img + idx_2d;

                if (!std::isfinite(x_mm[idx_3d]) || !std::isfinite(y_mm[idx_3d]) || !std::isfinite(z_mm[idx_3d]) ||
                    !std::isfinite(u[idx_3d]) || !std::isfinite(v[idx_3d]) || !std::isfinite(w[idx_3d])) continue;

                window.x_mm[widx] = x_mm[idx_3d];
                window.y_mm[widx] = y_mm[idx_3d];
                window.z_mm[widx] = z_mm[idx_3d];
                window.u[widx] = u[idx_3d];
                window.v[widx] = v[idx_3d];
                window.w[widx] = w[idx_3d];
                widx++;
            }
        }
        window.x.resize(widx);
        window.y.resize(widx);
        window.x_mm.resize(widx);
        window.y_mm.resize(widx);
        window.z_mm.resize(widx);
        window.u.resize(widx);
        window.v.resize(widx);
        window.w.resize(widx);
        return widx >= std::ceil(partial_window * sw_size * sw_size);
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
