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

// pybind header files
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// eigen header files
#include <Eigen/Dense>

// common_cpp header files
#include "../../common_cpp/progressbar.hpp"
#include "../../common_cpp/defines.hpp"
#include "../../common_cpp/util.hpp"

// Program Header files
#include "./smooth.hpp"
#include "./strain.hpp"



namespace py = pybind11;

namespace strain {

    Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

    void engine(const py::array_t<int> &ss_x_arr,
                const py::array_t<int> &ss_y_arr,
                const py::array_t<double> &u_arr,
                const py::array_t<double> &v_arr,
                const py::array_t<double> &w_arr,
                const int nss_x, const int nss_y, 
                const int nimg, const int sw_size, 
                const int q, const std::string &form,
                const std::vector<std::string> &filenames,
                const common_util::SaveConfig &strain_save_conf){

        TITLE("Strain Config");
        INFO_OUT("Number of Images:", nimg)
        INFO_OUT("Strain window size:", sw_size)
        INFO_OUT("Strain element (4 = bilinear, 9 is biquadratic):", q);
        INFO_OUT("Strain formulation:", form);
        INFO_OUT("Saving data to folder:", strain_save_conf.basepath)
        INFO_OUT("Saving data as binary ", strain_save_conf.binary)


        const int nwindows = nss_x*nss_y;

        // get raw pointers for numpy arrays
        int* ss_x = static_cast<int*>(ss_x_arr.request().ptr);
        int* ss_y = static_cast<int*>(ss_y_arr.request().ptr);
        double* u = static_cast<double*>(u_arr.request().ptr);
        double* v = static_cast<double*>(v_arr.request().ptr);
        double* w = static_cast<double*>(w_arr.request().ptr);

        // function wrapper for bilinear or biquadratic element
        std::function<Eigen::VectorXd(std::vector<int>&, std::vector<int>&, std::vector<double>&)> smooth_window = (q == 4) ? smooth::q4 : smooth::q9;


        strain::Window window(sw_size);
        strain::Results results(nwindows);


        TITLE("Deformation Gradient and Strain Calculation")

        // loop over the displacement images
        for (int img_num = 0; img_num < nimg; img_num++) {

            ProgressBar pbar(filenames[img_num], nwindows);

            // loop over strain windows within the image
            for (int sw = 0; sw < nwindows; sw++){

                int x0 = ss_x[sw];
                int y0 = ss_y[sw];
                results.x[sw] = x0;
                results.y[sw] = y0;

                // TODO: through a warning. NAN out the entire window.
                // it should be up to the user whether they correct with
                // outlier removal / smoothing.
                results.valid_window[sw] = fill_window(ss_x, ss_y, u, v, w, img_num,
                                                sw, window, nss_x,
                                                nss_y, sw_size);

                // element coefficients
                Eigen::VectorXd uc;
                Eigen::VectorXd vc;
                Eigen::VectorXd wc;

                // 2D deformation gradient matrix and identity matrix
                Eigen::Matrix3d deform_grad = Eigen::Matrix3d::Zero();
                Eigen::Matrix3d eps = Eigen::Matrix3d::Zero();

                if (results.valid_window[sw]){
                    uc = smooth_window(window.x, window.y, window.u);
                    vc = smooth_window(window.x, window.y, window.v);
                    wc = smooth_window(window.x, window.y, window.w);
                    deform_grad = compute_def_grad(q, uc, vc, wc, x0, y0);
                    eps = compute_strain(form, deform_grad);
                    append_results(sw, results, x0, y0, 
                                   deform_grad, eps, nwindows, img_num);
                }

                if (omp_get_thread_num() == 0) pbar.update(sw+1);

            }

            // finish up progress bar
            pbar.finish();

            strain::save_to_disk(img_num, results, strain_save_conf, nwindows, nimg, filenames);
        }
    }



    bool fill_window(int *ss_x, int *ss_y, double *u, double *v, double *w,
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

                // populate subset window
                window.x[widx] = ss_x[idx_2d];
                window.y[widx] = ss_y[idx_2d];
                window.u[widx] = u[idx_3d];
                window.v[widx] = v[idx_3d];
                window.w[widx] = w[idx_3d];

                //std::cout << window.x[idx] << " " << window.y[idx] << " ";
                //std::cout << window.u[idx] << " " << window.v[idx] << std::endl;
                widx++;
            }
        }
        return true;
    }


    Eigen::Matrix3d compute_def_grad(const int q, 
                                 const Eigen::VectorXd &uc, 
                                 const Eigen::VectorXd &vc, 
                                 const Eigen::VectorXd &wc, 
                                 const double x0, 
                                 const double y0) {

        Eigen::Matrix3d grad = Eigen::Matrix3d::Identity();

        if (q == 4) {
            grad(0,0) = 1.0 + uc[1] + uc[3]*y0;
            grad(0,1) =        uc[2] + uc[3]*x0;
            grad(0,2) = 0.0;

            grad(1,0) =        vc[1] + vc[3]*y0;
            grad(1,1) = 1.0 + vc[2] + vc[3]*x0;
            grad(1,2) = 0.0;

            grad(2,0) =        wc[1] + wc[3]*y0;
            grad(2,1) =        wc[2] + wc[3]*x0;
            grad(2,2) = 1.0;
        }
        else if (q == 9) {
            grad(0,0) = 1.0 + uc[1] + uc[3]*y0 + 2.0*uc[4]*x0 + 2.0*uc[6]*x0*y0 + uc[7]*y0*y0 + 2.0*uc[8]*x0*y0*y0;
            grad(0,1) =        uc[2] + uc[3]*x0 + 2.0*uc[5]*y0 + uc[6]*x0*x0 + 2.0*uc[7]*x0*y0 + 2.0*uc[8]*x0*x0*y0;
            grad(0,2) = 0.0;

            grad(1,0) =        vc[1] + vc[3]*y0 + 2.0*vc[4]*x0 + 2.0*vc[6]*x0*y0 + vc[7]*y0*y0 + 2.0*vc[8]*x0*y0*y0;
            grad(1,1) = 1.0 + vc[2] + vc[3]*x0 + 2.0*vc[5]*y0 + vc[6]*x0*x0 + 2.0*vc[7]*x0*y0 + 2.0*vc[8]*x0*x0*y0;
            grad(1,2) = 0.0;

            grad(2,0) =        wc[1] + wc[3]*y0 + 2.0*wc[4]*x0 + 2.0*wc[6]*x0*y0 + wc[7]*y0*y0 + 2.0*wc[8]*x0*y0*y0;
            grad(2,1) =        wc[2] + wc[3]*x0 + 2.0*wc[5]*y0 + wc[6]*x0*x0 + 2.0*wc[7]*x0*y0 + 2.0*wc[8]*x0*x0*y0;
            grad(2,2) = 1.0;
        }

        return grad;
    }

    Eigen::Matrix3d compute_strain(const std::string& form, const Eigen::Matrix3d& deform_grad) {
        if (form == "GREEN")        return green(deform_grad);
        else if (form == "ALMANSI") return almansi(deform_grad);
        else if (form == "HENCKY")  return hencky(deform_grad);
        else if (form == "BIOT_EULER") return biot_euler(deform_grad);
        else if (form == "BIOT_LAGRANGE") return biot_lagrange(deform_grad);

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
                        const Eigen::Matrix3d &deform_grad,
                        const Eigen::Matrix3d &eps,
                        const int nwindows, const int img){

        results.def_grad[9*sw+0] = deform_grad(0,0);
        results.def_grad[9*sw+1] = deform_grad(0,1);
        results.def_grad[9*sw+2] = deform_grad(0,2);

        results.def_grad[9*sw+3] = deform_grad(1,0);
        results.def_grad[9*sw+4] = deform_grad(1,1);
        results.def_grad[9*sw+5] = deform_grad(1,2);

        results.def_grad[9*sw+6] = deform_grad(2,0);
        results.def_grad[9*sw+7] = deform_grad(2,1);
        results.def_grad[9*sw+8] = deform_grad(2,2);

        results.strain[9*sw+0] = eps(0,0);
        results.strain[9*sw+1] = eps(0,1);
        results.strain[9*sw+2] = eps(0,2);

        results.strain[9*sw+3] = eps(1,0);
        results.strain[9*sw+4] = eps(1,1);
        results.strain[9*sw+5] = eps(1,2);

        results.strain[9*sw+6] = eps(2,0);
        results.strain[9*sw+7] = eps(2,1);
        results.strain[9*sw+8] = eps(2,2);
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

    // file extension
    std::string file_ext;
    if (strain_save_conf.binary) file_ext = ".dic3d";
    else file_ext = ".csv";

    // Extract base filename
    std::string full_filename = filenames[img_num];
    size_t dot_pos = full_filename.find(".");
    if (dot_pos != std::string::npos) {
        full_filename = full_filename.substr(0, dot_pos);
    }

    outfile_str << strain_save_conf.basepath << "/"
                << strain_save_conf.prefix
                << full_filename
                << file_ext;

    // reset img index (as in your original logic)
    img_num = 0;

    int tensor_size = 9;

    // =========================
    // BINARY SAVE
    // =========================
    if (strain_save_conf.binary)
    {
        outfile.open(outfile_str.str(), std::ios::binary);

        for (int i = 0; i < nwindows; ++i)
        {
            int idx = img_num * nwindows + i;

            common_util::write_int(outfile, results.x[idx]);
            common_util::write_int(outfile, results.y[idx]);

            // deformation gradient (3x3)
            for (int k = 0; k < tensor_size; ++k)
                common_util::write_dbl(outfile, results.def_grad[tensor_size * idx + k]);

            // strain tensor (3x3)
            for (int k = 0; k < tensor_size; ++k)
                common_util::write_dbl(outfile, results.strain[tensor_size * idx + k]);
        }

        outfile.close();
    }

    // =========================
    // ASCII / CSV SAVE
    // =========================
    else
    {
        outfile.open(outfile_str.str());

        // header
        outfile << "window_x" << delimiter
                << "window_y" << delimiter
                << "def_grad_00" << delimiter
                << "def_grad_01" << delimiter
                << "def_grad_02" << delimiter
                << "def_grad_10" << delimiter
                << "def_grad_11" << delimiter
                << "def_grad_12" << delimiter
                << "def_grad_20" << delimiter
                << "def_grad_21" << delimiter
                << "def_grad_22" << delimiter
                << "eps_00" << delimiter
                << "eps_01" << delimiter
                << "eps_02" << delimiter
                << "eps_10" << delimiter
                << "eps_11" << delimiter
                << "eps_12" << delimiter
                << "eps_20" << delimiter
                << "eps_21" << delimiter
                << "eps_22\n";

        for (int i = 0; i < nwindows; i++)
        {
            int idx = img_num * nwindows + i;

            if (results.valid_window[idx])
            {
                outfile << results.x[idx] << delimiter;
                outfile << results.y[idx] << delimiter;

                // deformation gradient
                for (int k = 0; k < tensor_size; ++k)
                {
                    outfile << results.def_grad[tensor_size * idx + k] << delimiter;
                }

                // strain tensor
                for (int k = 0; k < tensor_size; ++k)
                {
                    outfile << results.strain[tensor_size * idx + k];
                    if (k != tensor_size - 1) outfile << delimiter;
                }

                outfile << "\n";
            }
        }

        outfile.close();
    }
}



} // namespace strain
