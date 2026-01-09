// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <vector>
#include <iostream>
#include <omp.h>

// common_cpp header files
#include "../../common_cpp/util.hpp"
#include "../../common_cpp/defines.hpp"
#include "../../common_cpp/progressbar.hpp"
#include "../../common_cpp/dicsignalhandler.hpp"

// DIC Header files
#include "./dicinterpolator.hpp"


inline int idx_from_2d(const int x, const int y, const int length){
    return y*length+x;
}



Interpolator::Interpolator(double*img, int px_hori, int px_vert){

    //Timer timer("interpolator initialisation");

    // intitialise vars used globally within Interpolator.
    this->image = img;
    this->px_vert = px_vert;
    this->px_hori = px_hori;

    // allocate memory for pixel coordinate arrays
    px_y.resize(px_vert);
    px_x.resize(px_hori);

    // allocate memory for image derivatives
    dx.resize(px_vert*px_hori);
    dy.resize(px_vert*px_hori);
    dxy.resize(px_vert*px_hori);

    // setting pixel values for internal vectors
    for (int i = 0; i < px_hori; i++) {
        px_x[i] = i;
    }
    for (int j = 0; j < px_vert; j++) {
        px_y[j] = j;

    }

    //interpolator data
    //std::vector<double> data(px_hori,0);


    std::atomic<int> current_progress = 0;
    int niters = px_vert+px_hori+px_vert;
    ProgressBar pbar("Interpolator Initialisation:", niters);

    #ifdef _MSC_VER
        // Windows/MSVC - no explicit sharing of member variables
        #pragma omp parallel for shared(px_hori, px_vert, stop_request)
    #else
        // Linux/GCC - explicit sharing works
        #pragma omp parallel for shared(px_x, image, dx, px_hori, px_vert, stop_request)
    #endif
    for (int j = 0; j < px_vert; j++){

        // exit if ctrl+C
        if (stop_request) continue;

        // thread local data
        std::vector<double> data(px_hori, 0.0);
        std::vector<double> local_tridiag_sol(px_hori,0.0);

        // populate thread local image data
        for (int i = 0; i < px_hori; i++) {
            data[i] = image[j*px_hori + i];
        }

        cspline_init(px_x, data, local_tridiag_sol);
        for (int i = 0; i < px_hori; i++){
            dx[j*px_hori + i] = cspline_eval_deriv(px_x, data, local_tridiag_sol, px_x[i], px_hori);
        }

        // update progress bar if enabled
        if (g_debug_level>1){
            int progress = current_progress.fetch_add(1);
            if (omp_get_thread_num() == 0) pbar.update(progress);
        }

    }

    #ifdef _MSC_VER
        // Windows/MSVC - no explicit sharing of member variables
        #pragma omp parallel for shared(px_hori, px_vert, stop_request)
    #else
        // Linux/GCC - explicit sharing works
        #pragma omp parallel for shared(px_x, image, dx, px_hori, px_vert, stop_request)
    #endif
    for (int i = 0; i < px_hori; ++i) {

        // exit if ctrl+C
        if (stop_request) continue;

        // thread local data
        std::vector<double> data(px_vert, 0.0);
        std::vector<double> local_tridiag_sol(px_vert,0.0);

        // get 1D data
        for (int j = 0; j < px_vert; j++){
            data[j] = image[j*px_hori + i];
        }

        cspline_init(px_y, data, local_tridiag_sol);
        for (int j = 0; j < px_vert; j++){
            dy[j*px_hori + i] = cspline_eval_deriv(px_y, data, local_tridiag_sol, px_y[j], px_vert);
        }

        // update progress bar if enabled
        if (g_debug_level>1){
            int progress = current_progress.fetch_add(1);
            if (omp_get_thread_num() == 0) pbar.update(progress);
        }
    }


    //data.resize(px_hori,0);
    //
    #ifdef _MSC_VER
    // Windows/MSVC - no explicit sharing of member variables
        #pragma omp parallel for shared(px_hori, px_vert, stop_request)
    #else
        // Linux/GCC - explicit sharing works
        #pragma omp parallel for shared(px_x, image, dx, px_hori, px_vert, stop_request)
    #endif 
    for (int j = 0; j < px_vert; j++){

        // exit if ctrl+C
        if (stop_request) continue;

        // thread local data
        std::vector<double> data(px_hori, 0.0);
        std::vector<double> local_tridiag_sol(px_hori,0.0);

        // get 1D data
        for (int i = 0; i < px_hori; i++) {
            data[i] = dy[j*px_hori + i];
        }

        cspline_init(px_x, data, local_tridiag_sol);
        for (int i = 0; i < px_hori; i++){
            dxy[j*px_hori + i] = cspline_eval_deriv(px_x, data, local_tridiag_sol, px_x[i], px_hori);
        }

        // update progress bar if enabled
        if (g_debug_level>1){
            int progress = current_progress.fetch_add(1);
            if (omp_get_thread_num() == 0) pbar.update(progress);
        }
    }


    if (g_debug_level>1){
        pbar.update(current_progress);
        pbar.finish();
    }
}

double Interpolator::eval_bicubic(const int ss_x, const int ss_y, const double subpx_x, const double subpx_y) const {

    // get indices
    size_t xi,yi;
    index_lookup_xy(ss_x, ss_y, xi, yi, subpx_x, subpx_y);

    // 4 neighbouring grid points
    int idx00 = idx_from_2d(xi, yi, px_hori);
    int idx01 = idx_from_2d(xi, yi + 1, px_hori);
    int idx10 = idx_from_2d(xi + 1, yi, px_hori);
    int idx11 = idx_from_2d(xi + 1, yi + 1, px_hori);

    double d00 = image[idx00];
    double d01 = image[idx01];
    double d10 = image[idx10];
    double d11 = image[idx11];

    double dx00 = dx[idx00];
    double dx01 = dx[idx01];
    double dx10 = dx[idx10];
    double dx11 = dx[idx11];

    double dy00 = dy[idx00];
    double dy01 = dy[idx01];
    double dy10 = dy[idx10];
    double dy11 = dy[idx11];

    double dxy00 = dxy[idx00];
    double dxy01 = dxy[idx01];
    double dxy10 = dxy[idx10];
    double dxy11 = dxy[idx11];

    // polynomial terms
    double t0 = 1;
    double u0 = 1;
    double t1 = (subpx_x - px_x[xi]);
    double u1 = (subpx_y - px_y[yi]);
    double t2 = t1*t1;
    double u2 = u1*u1;  
    double t3 = t1*t2;
    double u3 = u1*u2;

    /* Perform bicubic interpolation */
    double result = 0.0;
    result += d00*t0*u0;
    result += dy00*t0*u1;
    result += (-3*d00 + 3*d01 - 2*dy00 - dy01)*t0*u2;
    result += (2*d00 - 2*d01 + dy00 + dy01)*t0*u3;

    result += dx00*t1*u0;
    result += dxy00*t1*u1;
    result += (-3*dx00 + 3*dx01 - 2*dxy00 - dxy01)*t1*u2;
    result += (2*dx00 - 2*dx01 + dxy00 + dxy01)*t1*u3;

    result += (-3*d00 + 3*d10 - 2*dx00 - dx10)*t2*u0;
    result += (-3*dy00 + 3*dy10 - 2*dxy00 - dxy10)*t2*u1;
    result += (9*d00 - 9*d10 + 9*d11 - 9*d01 + 6*dx00 + 3*dx10 - 3*dx11 - 6*dx01 + 6*dy00 - 6*dy10 - 3*dy11 + 3*dy01 + 4*dxy00 + 2*dxy10 + dxy11 + 2*dxy01)*t2*u2;
    result += (-6*d00 + 6*d10 - 6*d11 + 6*d01 - 4*dx00 - 2*dx10 + 2*dx11 + 4*dx01 - 3*dy00 + 3*dy10 + 3*dy11 - 3*dy01 - 2*dxy00 - dxy10 - dxy11 - 2*dxy01)*t2*u3;

    result += (2*d00 - 2*d10 + dx00 + dx10)*t3*u0;
    result += (2*dy00 - 2*dy10 + dxy00 + dxy10)*t3*u1;
    result += (-6*d00 + 6*d10 - 6*d11 + 6*d01 - 3*dx00 - 3*dx10 + 3*dx11 + 3*dx01 - 4*dy00 + 4*dy10 + 2*dy11 - 2*dy01 - 2*dxy00 - 2*dxy10 - dxy11 - dxy01)*t3*u2;
    result += (4*d00 - 4*d10 + 4*d11 - 4*d01 + 2*dx00 + 2*dx10 - 2*dx11 - 2*dx01 + 2*dy00 - 2*dy10 - 2*dy11 + 2*dy01 + dxy00 + dxy10 + dxy11 + dxy01)*t3*u3;

    return result;
}




double Interpolator::eval_bicubic_dx(const int ss_x, const int ss_y, const double subpx_x, const double subpx_y) const{

    /* first compute the indices into the data arrays where we are interpolating */ 
    size_t xi,yi;
    index_lookup_xy(ss_x, ss_y, xi, yi, subpx_x, subpx_y);

    int idx00 = idx_from_2d(xi, yi, px_hori);
    int idx01 = idx_from_2d(xi, yi + 1, px_hori);
    int idx10 = idx_from_2d(xi + 1, yi, px_hori);
    int idx11 = idx_from_2d(xi + 1, yi + 1, px_hori);

    double d00 = image[idx00];
    double d01 = image[idx01];
    double d10 = image[idx10];
    double d11 = image[idx11];

    double dx00 = dx[idx00];
    double dx01 = dx[idx01];
    double dx10 = dx[idx10];
    double dx11 = dx[idx11];
    double dy00 = dy[idx00];
    double dy01 = dy[idx01];
    double dy10 = dy[idx10];
    double dy11 = dy[idx11];
    double dxy00 = dxy[idx00];
    double dxy01 = dxy[idx01];
    double dxy10 = dxy[idx10];
    double dxy11 = dxy[idx11];

    // polynomial terms
    double t0 = 1;
    double u0 = 1;
    double t1 = (subpx_x - px_x[xi]);
    double u1 = (subpx_y - px_y[yi]);
    double t2 = t1*t1;
    double u2 = u1*u1;
    double u3 = u1*u2;

    double result = 0.0;
    result += dx00 *t0*u0;
    result += dxy00*t0*u1;
    result += (-3*dx00 + 3*dx01 - 2*dxy00 - dxy01) *t0*u2;
    result += (2*dx00 - 2*dx01 + dxy00 + dxy01)*t0*u3;
    result += 2*(-3*d00 + 3*d10 - 2*dx00 - dx10)*t1*u0;
    result += 2*(-3*dy00 + 3*dy10 - 2*dxy00 - dxy10)*t1*u1;
    result += 2*(9*d00 - 9*d10 + 9*d11 - 9*d01 + 6*dx00 + 3*dx10 - 3*dx11 - 6*dx01 + 6*dy00 - 6*dy10 - 3*dy11 + 3*dy01 + 4*dxy00 + 2*dxy10 + dxy11 + 2*dxy01)*t1*u2;
    result += 2*(-6*d00 + 6*d10 - 6*d11 + 6*d01 - 4*dx00 - 2*dx10 + 2*dx11 + 4*dx01 - 3*dy00 + 3*dy10 + 3*dy11 - 3*dy01 - 2*dxy00 - dxy10 - dxy11 - 2*dxy01)*t1*u3;
    result += 3*(2*d00 - 2*d10 + dx00 + dx10)*t2 *u0;
    result += 3*(2*dy00 - 2*dy10 + dxy00 + dxy10)*t2*u1;
    result += 3*(-6*d00 + 6*d10 - 6*d11 + 6*d01 - 3*dx00 - 3*dx10 + 3*dx11 + 3*dx01 - 4*dy00 + 4*dy10 + 2*dy11 - 2*dy01 - 2*dxy00 - 2*dxy10 - dxy11 - dxy01)*t2*u2;
    result += 3*(4*d00 - 4*d10 + 4*d11 - 4*d01 + 2*dx00 + 2*dx10 - 2*dx11 - 2*dx01 + 2*dy00 - 2*dy10 - 2*dy11 + 2*dy01 + dxy00 + dxy10 + dxy11 + dxy01)*t2*u3;
    return result;

}


double Interpolator::eval_bicubic_dy(const int ss_x, const int ss_y, const double subpx_x, const double subpx_y) const{

    size_t xi,yi;
    index_lookup_xy(ss_x, ss_y, xi, yi, subpx_x, subpx_y);

    // precompute indices of surrounding pixel values
    size_t idx00 = idx_from_2d(xi, yi, px_hori);
    size_t idx01 = idx_from_2d(xi, yi + 1, px_hori);
    size_t idx10 = idx_from_2d(xi + 1, yi, px_hori);
    size_t idx11 = idx_from_2d(xi + 1, yi + 1, px_hori);

    double d00 = image[idx00];
    double d01 = image[idx01];
    double d10 = image[idx10];
    double d11 = image[idx11];

    double dx00 = dx[idx00];
    double dx01 = dx[idx01];
    double dx10 = dx[idx10];
    double dx11 = dx[idx11];
    double dy00 = dy[idx00];
    double dy01 = dy[idx01];
    double dy10 = dy[idx10];
    double dy11 = dy[idx11];
    double dxy00 = dxy[idx00];
    double dxy01 = dxy[idx01];
    double dxy10 = dxy[idx10];
    double dxy11 = dxy[idx11];

    // polynomial terms
    double t0 = 1;
    double u0 = 1;
    double t1 = (subpx_x - px_x[xi]);
    double u1 = (subpx_y - px_y[yi]);
    double t2 = t1*t1;
    double u2 = u1*u1;
    double t3 = t1*t2;

    double result = 0.0;
    result += dy00*t0*u0;
    result += 2*(-3*d00 + 3*d01 - 2*dy00 - dy01)*t0*u1;
    result += 3*(2*d00-2*d01 + dy00 + dy01)*t0*u2;
    result += dxy00*t1*u0;
    result += 2*(-3*dx00 + 3*dx01 - 2*dxy00 - dxy01)*t1*u1;
    result += 3*(2*dx00 - 2*dx01 + dxy00 + dxy01)*t1*u2;
    result += (-3*dy00 + 3*dy10 - 2*dxy00 - dxy10)*t2*u0;
    result += 2*(9*d00 - 9*d10 + 9*d11 - 9*d01 + 6*dx00 + 3*dx10 - 3*dx11 - 6*dx01 + 6*dy00 - 6*dy10 - 3*dy11 + 3*dy01 + 4*dxy00 + 2*dxy10 + dxy11 + 2*dxy01)*t2*u1;
    result += 3*(-6*d00 + 6*d10 - 6*d11 + 6*d01 - 4*dx00 - 2*dx10 + 2*dx11 + 4*dx01 - 3*dy00 + 3*dy10 + 3*dy11 - 3*dy01 - 2*dxy00 - dxy10 - dxy11 - 2*dxy01)*t2*u2;
    result += (2*dy00 - 2*dy10 + dxy00 + dxy10)*t3*u0;
    result += 2*(-6*d00 + 6*d10 - 6*d11 + 6*d01 - 3*dx00 - 3*dx10 + 3*dx11 + 3*dx01 - 4*dy00 + 4*dy10 + 2*dy11 - 2*dy01 - 2*dxy00 - 2*dxy10 - dxy11 - dxy01)*t3*u1;
    result += 3*(4*d00 - 4*d10 + 4*d11 - 4*d01 + 2*dx00 + 2*dx10 - 2*dx11 - 2*dx01 + 2*dy00 - 2*dy10 - 2*dy11 + 2*dy01 + dxy00 + dxy10 + dxy11 + dxy01)*t3*u2;

    return result;
}


InterpVals Interpolator::eval_bicubic_and_derivs(const int ss_x, const int ss_y, const double subpx_x, const double subpx_y) const{

    // pixel floor of x and y 
    size_t xi,yi;
    index_lookup_xy(ss_x, ss_y, xi, yi, subpx_x, subpx_y);

    // precompute indices of surrounding pixel values
    size_t idx00 = idx_from_2d(xi, yi, px_hori);
    size_t idx01 = idx_from_2d(xi, yi + 1, px_hori);
    size_t idx10 = idx_from_2d(xi + 1, yi, px_hori);
    size_t idx11 = idx_from_2d(xi + 1, yi + 1, px_hori);

    double d00 = image[idx00];
    double d01 = image[idx01];
    double d10 = image[idx10];
    double d11 = image[idx11];

    double dx00 = dx[idx00];
    double dx01 = dx[idx01];
    double dx10 = dx[idx10];
    double dx11 = dx[idx11];
    double dy00 = dy[idx00];
    double dy01 = dy[idx01];
    double dy10 = dy[idx10];
    double dy11 = dy[idx11];
    double dxy00 = dxy[idx00];
    double dxy01 = dxy[idx01];
    double dxy10 = dxy[idx10];
    double dxy11 = dxy[idx11];

    // polynomial terms
    double t0 = 1;
    double u0 = 1;
    double t1 = (subpx_x - px_x[xi]);
    double u1 = (subpx_y - px_y[yi]);
    double t2 = t1*t1;
    double u2 = u1*u1;  
    double t3 = t1*t2;
    double u3 = u1*u2;

    double result = 0.0;
    double result_dx = 0.0;
    double result_dy = 0.0;

    result += d00*t0*u0;
    result += dy00*t0*u1;
    result += (-3*d00 + 3*d01 - 2*dy00 - dy01)*t0*u2;
    result += (2*d00 - 2*d01 + dy00 + dy01)*t0*u3;
    result += dx00*t1*u0;
    result += dxy00*t1*u1;
    result += (-3*dx00 + 3*dx01 - 2*dxy00 - dxy01)*t1*u2;
    result += (2*dx00 - 2*dx01 + dxy00 + dxy01)*t1*u3;
    result += (-3*d00 + 3*d10 - 2*dx00 - dx10)*t2*u0;
    result += (-3*dy00 + 3*dy10 - 2*dxy00 - dxy10)*t2*u1;
    result += (9*d00 - 9*d10 + 9*d11 - 9*d01 + 6*dx00 + 3*dx10 - 3*dx11 - 6*dx01 + 6*dy00 - 6*dy10 - 3*dy11 + 3*dy01 + 4*dxy00 + 2*dxy10 + dxy11 + 2*dxy01)*t2*u2;
    result += (-6*d00 + 6*d10 - 6*d11 + 6*d01 - 4*dx00 - 2*dx10 + 2*dx11 + 4*dx01 - 3*dy00 + 3*dy10 + 3*dy11 - 3*dy01 - 2*dxy00 - dxy10 - dxy11 - 2*dxy01)*t2*u3;
    result += (2*d00 - 2*d10 + dx00 + dx10)*t3*u0;
    result += (2*dy00 - 2*dy10 + dxy00 + dxy10)*t3*u1;
    result += (-6*d00 + 6*d10 - 6*d11 + 6*d01 - 3*dx00 - 3*dx10 + 3*dx11 + 3*dx01 - 4*dy00 + 4*dy10 + 2*dy11 - 2*dy01 - 2*dxy00 - 2*dxy10 - dxy11 - dxy01)*t3*u2;
    result += (4*d00 - 4*d10 + 4*d11 - 4*d01 + 2*dx00 + 2*dx10 - 2*dx11 - 2*dx01 + 2*dy00 - 2*dy10 - 2*dy11 + 2*dy01 + dxy00 + dxy10 + dxy11 + dxy01)*t3*u3;

    result_dx += dx00 *t0*u0;
    result_dx += dxy00*t0*u1;
    result_dx += (-3*dx00 + 3*dx01 - 2*dxy00 - dxy01) *t0*u2;
    result_dx += (2*dx00 - 2*dx01 + dxy00 + dxy01)*t0*u3;
    result_dx += 2*(-3*d00 + 3*d10 - 2*dx00 - dx10)*t1*u0;
    result_dx += 2*(-3*dy00 + 3*dy10 - 2*dxy00 - dxy10)*t1*u1;
    result_dx += 2*(9*d00 - 9*d10 + 9*d11 - 9*d01 + 6*dx00 + 3*dx10 - 3*dx11 - 6*dx01 + 6*dy00 - 6*dy10 - 3*dy11 + 3*dy01 + 4*dxy00 + 2*dxy10 + dxy11 + 2*dxy01)*t1*u2;
    result_dx += 2*(-6*d00 + 6*d10 - 6*d11 + 6*d01 - 4*dx00 - 2*dx10 + 2*dx11 + 4*dx01 - 3*dy00 + 3*dy10 + 3*dy11 - 3*dy01 - 2*dxy00 - dxy10 - dxy11 - 2*dxy01)*t1*u3;
    result_dx += 3*(2*d00 - 2*d10 + dx00 + dx10)*t2 *u0;
    result_dx += 3*(2*dy00 - 2*dy10 + dxy00 + dxy10)*t2*u1;
    result_dx += 3*(-6*d00 + 6*d10 - 6*d11 + 6*d01 - 3*dx00 - 3*dx10 + 3*dx11 + 3*dx01 - 4*dy00 + 4*dy10 + 2*dy11 - 2*dy01 - 2*dxy00 - 2*dxy10 - dxy11 - dxy01)*t2*u2;
    result_dx += 3*(4*d00 - 4*d10 + 4*d11 - 4*d01 + 2*dx00 + 2*dx10 - 2*dx11 - 2*dx01 + 2*dy00 - 2*dy10 - 2*dy11 + 2*dy01 + dxy00 + dxy10 + dxy11 + dxy01)*t2*u3;

    result_dy += dy00*t0*u0;
    result_dy += 2*(-3*d00 + 3*d01 - 2*dy00 - dy01)*t0*u1;
    result_dy += 3*(2*d00-2*d01 + dy00 + dy01)*t0*u2;
    result_dy += dxy00*t1*u0;
    result_dy += 2*(-3*dx00 + 3*dx01 - 2*dxy00 - dxy01)*t1*u1;
    result_dy += 3*(2*dx00 - 2*dx01 + dxy00 + dxy01)*t1*u2;
    result_dy += (-3*dy00 + 3*dy10 - 2*dxy00 - dxy10)*t2*u0;
    result_dy += 2*(9*d00 - 9*d10 + 9*d11 - 9*d01 + 6*dx00 + 3*dx10 - 3*dx11 - 6*dx01 + 6*dy00 - 6*dy10 - 3*dy11 + 3*dy01 + 4*dxy00 + 2*dxy10 + dxy11 + 2*dxy01)*t2*u1;
    result_dy += 3*(-6*d00 + 6*d10 - 6*d11 + 6*d01 - 4*dx00 - 2*dx10 + 2*dx11 + 4*dx01 - 3*dy00 + 3*dy10 + 3*dy11 - 3*dy01 - 2*dxy00 - dxy10 - dxy11 - 2*dxy01)*t2*u2;
    result_dy += (2*dy00 - 2*dy10 + dxy00 + dxy10)*t3*u0;
    result_dy += 2*(-6*d00 + 6*d10 - 6*d11 + 6*d01 - 3*dx00 - 3*dx10 + 3*dx11 + 3*dx01 - 4*dy00 + 4*dy10 + 2*dy11 - 2*dy01 - 2*dxy00 - 2*dxy10 - dxy11 - dxy01)*t3*u1;
    result_dy += 3*(4*d00 - 4*d10 + 4*d11 - 4*d01 + 2*dx00 + 2*dx10 - 2*dx11 - 2*dx01 + 2*dy00 - 2*dy10 - 2*dy11 + 2*dy01 + dxy00 + dxy10 + dxy11 + dxy01)*t3*u2;

    return {result, result_dx, result_dy};
}




inline void Interpolator::coeff_calc(std::vector<double> &tridiag_solution, double dy, double dx, size_t i, double *b, double *c, double *d) {
    
    const double s_i = tridiag_solution[i];
    const double s_ip1 = tridiag_solution[i + 1];
    const double dx_inv = 1.0 / dx;

    *b = (dy*dx_inv) - dx*(s_ip1 + 2.0*s_i) / 3.0;
    *c = s_i;
    *d = (s_ip1 - s_i) / (3.0*dx);
}


inline void Interpolator::index_lookup_xy(const int ss_x, const int ss_y, size_t &xi, size_t &yi, const double subpx_x, const double subpx_y) const {
    
    if (subpx_x < px_x[0]) 
        xi = 0;
    else if (subpx_x > px_x[px_hori - 2]) 
        xi = px_hori - 2;
    else
        xi = static_cast<size_t>(subpx_x);

    if (subpx_y < px_y[0])
        yi = 0;
    else if (subpx_y > px_y[px_vert - 2])
        yi = px_vert - 2;
    else
        yi = static_cast<size_t>(subpx_y);

    //if (subpx_x >= px_x[0] && subpx_x <= px_x[px_hori-1]) {
    //    xi = static_cast<size_t>(subpx_x); // Return x as the index
    //}
    //else {
    //    std::cerr << "ERROR in \'" << __FILE__ << "\' at line \'" << __LINE__ << "\' \n";
    //    std::cerr << "Interpolator went out of bounds for subset (" << ss_x << ", " << ss_y << ")" << std::endl;
    //    std::cerr << "value is out of bounds: (" << subpx_x << ", " << subpx_y << ")" << std::endl;
    //    std::cerr << "Image bounds: (0,0) to (" << px_hori-1 << ", " << px_vert-1 << ")" << std::endl;
    //    exit(EXIT_FAILURE);
    //}

    //if (subpx_y >= px_y[0] && subpx_y <= px_y[px_vert-1]) {
    //    yi = static_cast<size_t>(subpx_y); // Return x as the index
    //}
    //else {
    //    std::cerr << "ERROR in \'" << __FILE__ << "\' at line \'" << __LINE__ << "\' \n";
    //    std::cerr << "Interpolator went out of bounds for subset (" << ss_x << ", " << ss_y << ")" << std::endl;
    //    std::cerr << "value is out of bounds: (" << subpx_x << ", " << subpx_y << ")" << std::endl;
    //    std::cerr << "Image bounds: (0,0) to (" << px_hori-1 << ", " << px_vert-1 << ")" << std::endl;
    //    exit(EXIT_FAILURE);
    //}
}


inline int Interpolator::index_lookup(const std::vector<double> &px, double x) const {
    
    // Clamp coordinates to valid range
    // double clamped_x = std::max(static_cast<double>(index_lo), std::min(static_cast<double>(index_hi), x));

    // if (x >= px[index_lo] && x <= px[index_hi]) {
    //     // return static_cast<int>(x); // Return x as the index
    // }
    // else {
    //     // std::cout << "ERROR in \'" << __FILE__ << "\' at line \'" << __LINE__ << "\' \n";
    //     // std::cout << "value is out of bounds. value = " << x << std::endl;
    //     // exit(EXIT_FAILURE);
    // }
    // return static_cast<int>(clamped_x);

    if (x >= px.front() && x <= px.back()) {
        return static_cast<int>(x); // Return x as the index
    }
    else {
        std::cerr << "ERROR in \'" << __FILE__ << "\' at line \'" << __LINE__ << "\' \n";
        std::cerr << "value is out of bounds. value = " << x << std::endl;
        exit(EXIT_FAILURE);
    }

}



void Interpolator::cspline_init(const std::vector<double> &px, const std::vector<double> &data, 
                                std::vector<double> &tridiag_solution){


    int num_points = px.size();
    int max_index = num_points - 1;  
    int sys_size = max_index - 1;
    
    std::vector<double> diagonal(num_points);
    std::vector<double> off_diagonal(num_points);
    std::vector<double> rhs(num_points);

    for (int i = 0; i < sys_size; i++)
    {
        const double h_i   = px[i + 1] - px[i];
        const double h_ip1 = px[i + 2] - px[i + 1];
        const double ydiff_i   = data[i + 1] - data[i];
        const double ydiff_ip1 = data[i + 2] - data[i + 1];
        const double g_i = (h_i != 0.0) ? 1.0 / h_i : 0.0;
        const double g_ip1 = (h_ip1 != 0.0) ? 1.0 / h_ip1 : 0.0;
        off_diagonal[i] = h_ip1;
        diagonal[i] = 2.0*(h_ip1 + h_i);
        rhs[i] = 3.0*(ydiff_ip1*g_ip1 -  ydiff_i*g_i);

    }

    std::vector<double> gamma(sys_size);
    std::vector<double> alpha(sys_size);
    std::vector<double> c(sys_size);
    std::vector<double> z(sys_size);
    alpha[0] = diagonal[0];
    gamma[0] = off_diagonal[0] / alpha[0];

    if (alpha[0] == 0) {
        std::cerr << __FILE__ << " " << __LINE__ << "ERROR: div by zero" << std::endl;
        exit(1);
    }

    for (int i = 1; i < sys_size - 1; i++) {

        alpha[i] = diagonal[i] - off_diagonal[i - 1]*gamma[i - 1];
        gamma[i] = off_diagonal[i] / alpha[i];
        if (alpha[i] == 0) {
            std::cerr << __FILE__ << " " << __LINE__ << "ERROR: div by zero" << std::endl;
            exit(1);
        }

    }

    if (sys_size > 1) {
        alpha[sys_size - 1] = diagonal[(sys_size - 1)] - off_diagonal[(sys_size - 2)]*gamma[sys_size - 2];
    }

    // RHS of equation
    z[0] = rhs[0];
    for (int i = 1; i < sys_size; i++) {
        z[i] = rhs[i] - gamma[i - 1]*z[i - 1];
    }

    for (int i = 0; i < sys_size; i++){
        c[i] = z[i] / alpha[i];
    }

    // back substitution
    tridiag_solution[sys_size] = c[sys_size - 1];
    if (sys_size >= 2) {
        for (int i = sys_size - 2; i >= 0; i--) {
            tridiag_solution[i+1] = c[i] - gamma[i]*tridiag_solution[i + 2];
        }
    }  
}

double Interpolator::cspline_eval_deriv(std::vector<double> &px, std::vector<double> &data,
                                        std::vector<double> &local_tridiag_sol, double value, int length) {

    // Find the interval containing the evaluation point
    int index = index_lookup(px, value);

    // Get interval boundaries
    double px_min = px[index];
    double px_max = px[index + 1];
    double dx = px_max - px_min;

    // Handle degenerate case where interval has zero width
    if (dx <= 0.0) {
        return 0.0;
    }

    // Get y-values at interval endpoints
    double y_lo = data[index];
    double y_hi = data[index + 1];
    double dy = y_hi - y_lo;

    // Calculate distance from left endpoint
    double delx = value - px_min;

    // Calculate cubic spline coefficients for this interval
    double b_i, c_i, d_i;
    coeff_calc(local_tridiag_sol, dy, dx, index, &b_i, &c_i, &d_i);

    // Evaluate derivative: dy/dx = b + 2c*delx + 3d*delx^2
    double dydx = b_i + delx*(2.0*c_i + 3.0*d_i*delx);

    return dydx;
}
