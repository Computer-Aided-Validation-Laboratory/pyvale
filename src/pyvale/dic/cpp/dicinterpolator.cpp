// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <vector>
#include <iostream>

// Program Header files
#include "./dicinterpolator.hpp"



inline int idx_from_2d(const int x, const int y, const int length){
    return y*length+x;
}



Interpolator::Interpolator(double * img, int px_hori, int px_vert){

    //util::Timer timer("interpolator initialisation");

    // intitialise vars used globally within Interpolator.
    this->image = img;
    this->px_vert = px_vert;
    this->px_hori = px_hori;

    // allocate memory for pixel coordinate arrays
    px_y.resize(px_vert);
    px_x.resize(px_hori);

    // allocate memory for image derivatives
    zx.resize(px_vert*px_hori);
    zy.resize(px_vert*px_hori);
    zxy.resize(px_vert*px_hori);

    // setting pixel values for internal vectors
    for (int i = 0; i < px_hori; ++i) {
        px_x[i] = i;
    }
    for (int j = 0; j < px_vert; ++j) {
        px_y[j] = j;

    }

    //interpolator data
    std::vector<double> data(px_hori,0);
    for (int j = 0; j < px_vert; j++){

        // get 1D data
        for (int i = 0; i < px_hori; ++i) {
            data[i] = image[j * px_hori + i];
        }

        cspline_init(px_x, data);
        for (int i = 0; i < px_hori; i++){
            zx[j * px_hori + i] = cspline_eval_deriv(px_x, data, px_x[i], px_hori);
        }
    }

    data.resize(px_vert,0);
    for (int i = 0; i < px_hori; ++i) {

        // get 1D data
        for (int j = 0; j < px_vert; j++){
            data[j] = image[j * px_hori + i];
        }

        cspline_init(px_y, data);
        for (int j = 0; j < px_vert; j++){
            zy[j * px_hori + i] = cspline_eval_deriv(px_y, data, px_y[j], px_vert);
        }
    }


    data.resize(px_hori,0);
    for (int j = 0; j < px_vert; j++){

        // get 1D data
        for (int i = 0; i < px_hori; ++i) {
            data[i] = zy[j * px_hori + i];
        }

        cspline_init(px_x, data);
        for (int i = 0; i < px_hori; i++){
            zxy[j * px_hori + i] = cspline_eval_deriv(px_x, data, px_x[i], px_hori);
        }
    }
}

double Interpolator::eval_bicubic(const int ss_x, const int ss_y, const double subpx_x, const double subpx_y) const {

    // get indices
    size_t xi,yi;
    index_lookup_xy(ss_x, ss_y, xi, yi, subpx_x, subpx_y);

    int idx00 = idx_from_2d(xi, yi, px_hori);
    int idx01 = idx_from_2d(xi, yi + 1, px_hori);
    int idx10 = idx_from_2d(xi + 1, yi, px_hori);
    int idx11 = idx_from_2d(xi + 1, yi + 1, px_hori);

    double zminmin = image[idx00];
    double zminmax = image[idx01];
    double zmaxmin = image[idx10];
    double zmaxmax = image[idx11];

    double zxminmin = zx[idx00];
    double zxminmax = zx[idx01];
    double zxmaxmin = zx[idx10];
    double zxmaxmax = zx[idx11];

    double zyminmin = zy[idx00];
    double zyminmax = zy[idx01];
    double zymaxmin = zy[idx10];
    double zymaxmax = zy[idx11];

    double zxyminmin = zxy[idx00];
    double zxyminmax = zxy[idx01];
    double zxymaxmin = zxy[idx10];
    double zxymaxmax = zxy[idx11];

    // polynomial terms
    double t0 = 1;
    double u0 = 1;
    double t1 = (subpx_x - px_x[xi]);
    double u1 = (subpx_y - px_y[yi]);
    double t2 = t1 * t1;
    double u2 = u1 * u1;  
    double t3 = t1 * t2;
    double u3 = u1 * u2;

    /* Perform bicubic interpolation */
    double result = 0.0;
    result += zminmin * t0 * u0;
    result += zyminmin * t0 * u1;
    result += (-3 * zminmin + 3 * zminmax - 2 * zyminmin - zyminmax) * t0 * u2;
    result += (2 * zminmin - 2 * zminmax + zyminmin + zyminmax) * t0 * u3;

    result += zxminmin * t1 * u0;
    result += zxyminmin * t1 * u1;
    result += (-3 * zxminmin + 3 * zxminmax - 2 * zxyminmin - zxyminmax) * t1 * u2;
    result += (2 * zxminmin - 2 * zxminmax + zxyminmin + zxyminmax) * t1 * u3;

    result += (-3 * zminmin + 3 * zmaxmin - 2 * zxminmin - zxmaxmin) * t2 * u0;
    result += (-3 * zyminmin + 3 * zymaxmin - 2 * zxyminmin - zxymaxmin) * t2 * u1;
    result += (9 * zminmin - 9 * zmaxmin + 9 * zmaxmax - 9 * zminmax + 6 * zxminmin + 3 * zxmaxmin - 3 * zxmaxmax - 6 * zxminmax + 6 * zyminmin - 6 * zymaxmin - 3 * zymaxmax + 3 * zyminmax + 4 * zxyminmin + 2 * zxymaxmin + zxymaxmax + 2 * zxyminmax) * t2 * u2;
    result += (-6 * zminmin + 6 * zmaxmin - 6 * zmaxmax + 6 * zminmax - 4 * zxminmin - 2 * zxmaxmin + 2 * zxmaxmax + 4 * zxminmax - 3 * zyminmin + 3 * zymaxmin + 3 * zymaxmax - 3 * zyminmax - 2 * zxyminmin - zxymaxmin - zxymaxmax - 2 * zxyminmax) * t2 * u3;

    result += (2 * zminmin - 2 * zmaxmin + zxminmin + zxmaxmin) * t3 * u0;
    result += (2 * zyminmin - 2 * zymaxmin + zxyminmin + zxymaxmin) * t3 * u1;
    result += (-6 * zminmin + 6 * zmaxmin - 6 * zmaxmax + 6 * zminmax - 3 * zxminmin - 3 * zxmaxmin + 3 * zxmaxmax + 3 * zxminmax - 4 * zyminmin + 4 * zymaxmin + 2 * zymaxmax - 2 * zyminmax - 2 * zxyminmin - 2 * zxymaxmin - zxymaxmax - zxyminmax) * t3 * u2;
    result += (4 * zminmin - 4 * zmaxmin + 4 * zmaxmax - 4 * zminmax + 2 * zxminmin + 2 * zxmaxmin - 2 * zxmaxmax - 2 * zxminmax + 2 * zyminmin - 2 * zymaxmin - 2 * zymaxmax + 2 * zyminmax + zxyminmin + zxymaxmin + zxymaxmax + zxyminmax) * t3 * u3;

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

    double zminmin = image[idx00];
    double zminmax = image[idx01];
    double zmaxmin = image[idx10];
    double zmaxmax = image[idx11];

    double zxminmin = zx[idx00];
    double zxminmax = zx[idx01];
    double zxmaxmin = zx[idx10];
    double zxmaxmax = zx[idx11];
    double zyminmin = zy[idx00];
    double zyminmax = zy[idx01];
    double zymaxmin = zy[idx10];
    double zymaxmax = zy[idx11];
    double zxyminmin = zxy[idx00];
    double zxyminmax = zxy[idx01];
    double zxymaxmin = zxy[idx10];
    double zxymaxmax = zxy[idx11];

    // polynomial terms
    double t0 = 1;
    double u0 = 1;
    double t1 = (subpx_x - px_x[xi]);
    double u1 = (subpx_y - px_y[yi]);
    double t2 = t1 * t1;
    double u2 = u1 * u1;
    double u3 = u1 * u2;

    double result = 0.0;
    result += zxminmin *t0 * u0;
    result += zxyminmin * t0 * u1;
    result += (-3*zxminmin + 3*zxminmax - 2*zxyminmin - zxyminmax) *t0 * u2;
    result += (2*zxminmin - 2*zxminmax + zxyminmin + zxyminmax) * t0 * u3;
    result += 2 * (-3*zminmin + 3*zmaxmin - 2*zxminmin - zxmaxmin)*t1*u0;
    result += 2 * (-3*zyminmin + 3*zymaxmin - 2*zxyminmin - zxymaxmin)*t1*u1;
    result += 2 * (9*zminmin - 9*zmaxmin + 9*zmaxmax - 9*zminmax + 6*zxminmin + 3*zxmaxmin - 3*zxmaxmax - 6*zxminmax + 6*zyminmin - 6*zymaxmin - 3*zymaxmax + 3*zyminmax + 4*zxyminmin + 2*zxymaxmin + zxymaxmax + 2*zxyminmax)*t1*u2;
    result += 2 * (-6*zminmin + 6*zmaxmin - 6*zmaxmax + 6*zminmax - 4*zxminmin - 2*zxmaxmin + 2*zxmaxmax + 4*zxminmax - 3*zyminmin + 3*zymaxmin + 3*zymaxmax - 3*zyminmax - 2*zxyminmin - zxymaxmin - zxymaxmax - 2*zxyminmax)*t1*u3;
    result += 3 * (2*zminmin - 2*zmaxmin + zxminmin + zxmaxmin) * t2 *u0;
    result += 3 * (2*zyminmin - 2*zymaxmin + zxyminmin + zxymaxmin) * t2 * u1;
    result += 3 * (-6*zminmin + 6*zmaxmin - 6*zmaxmax + 6*zminmax - 3*zxminmin - 3*zxmaxmin + 3*zxmaxmax + 3*zxminmax - 4*zyminmin + 4*zymaxmin + 2*zymaxmax - 2*zyminmax - 2*zxyminmin - 2*zxymaxmin - zxymaxmax - zxyminmax) * t2 * u2;
    result += 3 * (4*zminmin - 4*zmaxmin + 4*zmaxmax - 4*zminmax + 2*zxminmin + 2*zxmaxmin - 2*zxmaxmax - 2*zxminmax + 2*zyminmin - 2*zymaxmin - 2*zymaxmax + 2*zyminmax + zxyminmin + zxymaxmin + zxymaxmax + zxyminmax) * t2 * u3;
    return result;

}


double Interpolator::eval_bicubic_dy(const int ss_x, const int ss_y, const double subpx_x, const double subpx_y) const{

    /* first compute the indices into the data arrays where we are interpolating */
    size_t xi,yi;
    index_lookup_xy(ss_x, ss_y, xi, yi, subpx_x, subpx_y);

    // precompute indices of surrounding pixel values
    size_t idx00 = idx_from_2d(xi, yi, px_hori);
    size_t idx01 = idx_from_2d(xi, yi + 1, px_hori);
    size_t idx10 = idx_from_2d(xi + 1, yi, px_hori);
    size_t idx11 = idx_from_2d(xi + 1, yi + 1, px_hori);

    double zminmin = image[idx00];
    double zminmax = image[idx01];
    double zmaxmin = image[idx10];
    double zmaxmax = image[idx11];

    double zxminmin = zx[idx00];
    double zxminmax = zx[idx01];
    double zxmaxmin = zx[idx10];
    double zxmaxmax = zx[idx11];
    double zyminmin = zy[idx00];
    double zyminmax = zy[idx01];
    double zymaxmin = zy[idx10];
    double zymaxmax = zy[idx11];
    double zxyminmin = zxy[idx00];
    double zxyminmax = zxy[idx01];
    double zxymaxmin = zxy[idx10];
    double zxymaxmax = zxy[idx11];

    // distance between interpolation point and pixel value 

    // polynomial terms
    double t0 = 1;
    double u0 = 1;
    double t1 = (subpx_x - px_x[xi]);
    double u1 = (subpx_y - px_y[yi]);
    double t2 = t1 * t1;
    double u2 = u1 * u1;  
    double t3 = t1 * t2;

    double result = 0.0;
    result += zyminmin * t0 * u0;
    result += 2 * (-3*zminmin + 3*zminmax - 2*zyminmin - zyminmax) * t0 * u1;
    result += 3 * (2*zminmin-2*zminmax + zyminmin + zyminmax) * t0 * u2;
    result += zxyminmin*t1*u0;
    result += 2 * (-3*zxminmin + 3*zxminmax - 2*zxyminmin - zxyminmax) * t1 * u1;
    result += 3 * (2*zxminmin - 2*zxminmax + zxyminmin + zxyminmax) * t1 * u2;
    result += (-3*zyminmin + 3*zymaxmin - 2*zxyminmin - zxymaxmin) * t2 * u0;
    result += 2 * (9*zminmin - 9*zmaxmin + 9*zmaxmax - 9*zminmax + 6*zxminmin + 3*zxmaxmin - 3*zxmaxmax - 6*zxminmax + 6*zyminmin - 6*zymaxmin - 3*zymaxmax + 3*zyminmax + 4*zxyminmin + 2*zxymaxmin + zxymaxmax + 2*zxyminmax) * t2 * u1;
    result += 3 * (-6*zminmin + 6*zmaxmin - 6*zmaxmax + 6*zminmax - 4*zxminmin - 2*zxmaxmin + 2*zxmaxmax + 4*zxminmax - 3*zyminmin + 3*zymaxmin + 3*zymaxmax - 3*zyminmax - 2*zxyminmin - zxymaxmin - zxymaxmax - 2*zxyminmax) * t2 * u2;
    result += (2*zyminmin - 2*zymaxmin + zxyminmin + zxymaxmin) * t3 * u0;
    result += 2 * (-6*zminmin + 6*zmaxmin - 6*zmaxmax + 6*zminmax - 3*zxminmin - 3*zxmaxmin + 3*zxmaxmax + 3*zxminmax - 4*zyminmin + 4*zymaxmin + 2*zymaxmax - 2*zyminmax - 2*zxyminmin - 2*zxymaxmin - zxymaxmax - zxyminmax) * t3 * u1;
    result += 3 * (4*zminmin - 4*zmaxmin + 4*zmaxmax - 4*zminmax + 2*zxminmin + 2*zxmaxmin - 2*zxmaxmax - 2*zxminmax + 2*zyminmin - 2*zymaxmin - 2*zymaxmax + 2*zyminmax + zxyminmin + zxymaxmin + zxymaxmax + zxyminmax) * t3 * u2;

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

    double zminmin = image[idx00];
    double zminmax = image[idx01];
    double zmaxmin = image[idx10];
    double zmaxmax = image[idx11];

    double zxminmin = zx[idx00];
    double zxminmax = zx[idx01];
    double zxmaxmin = zx[idx10];
    double zxmaxmax = zx[idx11];
    double zyminmin = zy[idx00];
    double zyminmax = zy[idx01];
    double zymaxmin = zy[idx10];
    double zymaxmax = zy[idx11];
    double zxyminmin = zxy[idx00];
    double zxyminmax = zxy[idx01];
    double zxymaxmin = zxy[idx10];
    double zxymaxmax = zxy[idx11];

    // distance between interpolation point and pixel value 

    // polynomial terms
    double t0 = 1;
    double u0 = 1;
    double t1 = (subpx_x - px_x[xi]);
    double u1 = (subpx_y - px_y[yi]);
    double t2 = t1 * t1;
    double u2 = u1 * u1;  
    double t3 = t1 * t2;
    double u3 = u1 * u2;

    double result = 0.0;
    double result_dx = 0.0;
    double result_dy = 0.0;

    result += zminmin * t0 * u0;
    result += zyminmin * t0 * u1;
    result += (-3 * zminmin + 3 * zminmax - 2 * zyminmin - zyminmax) * t0 * u2;
    result += (2 * zminmin - 2 * zminmax + zyminmin + zyminmax) * t0 * u3;
    result += zxminmin * t1 * u0;
    result += zxyminmin * t1 * u1;
    result += (-3 * zxminmin + 3 * zxminmax - 2 * zxyminmin - zxyminmax) * t1 * u2;
    result += (2 * zxminmin - 2 * zxminmax + zxyminmin + zxyminmax) * t1 * u3;
    result += (-3 * zminmin + 3 * zmaxmin - 2 * zxminmin - zxmaxmin) * t2 * u0;
    result += (-3 * zyminmin + 3 * zymaxmin - 2 * zxyminmin - zxymaxmin) * t2 * u1;
    result += (9 * zminmin - 9 * zmaxmin + 9 * zmaxmax - 9 * zminmax + 6 * zxminmin + 3 * zxmaxmin - 3 * zxmaxmax - 6 * zxminmax + 6 * zyminmin - 6 * zymaxmin - 3 * zymaxmax + 3 * zyminmax + 4 * zxyminmin + 2 * zxymaxmin + zxymaxmax + 2 * zxyminmax) * t2 * u2;
    result += (-6 * zminmin + 6 * zmaxmin - 6 * zmaxmax + 6 * zminmax - 4 * zxminmin - 2 * zxmaxmin + 2 * zxmaxmax + 4 * zxminmax - 3 * zyminmin + 3 * zymaxmin + 3 * zymaxmax - 3 * zyminmax - 2 * zxyminmin - zxymaxmin - zxymaxmax - 2 * zxyminmax) * t2 * u3;
    result += (2 * zminmin - 2 * zmaxmin + zxminmin + zxmaxmin) * t3 * u0;
    result += (2 * zyminmin - 2 * zymaxmin + zxyminmin + zxymaxmin) * t3 * u1;
    result += (-6 * zminmin + 6 * zmaxmin - 6 * zmaxmax + 6 * zminmax - 3 * zxminmin - 3 * zxmaxmin + 3 * zxmaxmax + 3 * zxminmax - 4 * zyminmin + 4 * zymaxmin + 2 * zymaxmax - 2 * zyminmax - 2 * zxyminmin - 2 * zxymaxmin - zxymaxmax - zxyminmax) * t3 * u2;
    result += (4 * zminmin - 4 * zmaxmin + 4 * zmaxmax - 4 * zminmax + 2 * zxminmin + 2 * zxmaxmin - 2 * zxmaxmax - 2 * zxminmax + 2 * zyminmin - 2 * zymaxmin - 2 * zymaxmax + 2 * zyminmax + zxyminmin + zxymaxmin + zxymaxmax + zxyminmax) * t3 * u3;

    result_dx += zxminmin *t0 * u0;
    result_dx += zxyminmin * t0 * u1;
    result_dx += (-3*zxminmin + 3*zxminmax - 2*zxyminmin - zxyminmax) *t0 * u2;
    result_dx += (2*zxminmin - 2*zxminmax + zxyminmin + zxyminmax) * t0 * u3;
    result_dx += 2 * (-3*zminmin + 3*zmaxmin - 2*zxminmin - zxmaxmin)*t1*u0;
    result_dx += 2 * (-3*zyminmin + 3*zymaxmin - 2*zxyminmin - zxymaxmin)*t1*u1;
    result_dx += 2 * (9*zminmin - 9*zmaxmin + 9*zmaxmax - 9*zminmax + 6*zxminmin + 3*zxmaxmin - 3*zxmaxmax - 6*zxminmax + 6*zyminmin - 6*zymaxmin - 3*zymaxmax + 3*zyminmax + 4*zxyminmin + 2*zxymaxmin + zxymaxmax + 2*zxyminmax)*t1*u2;
    result_dx += 2 * (-6*zminmin + 6*zmaxmin - 6*zmaxmax + 6*zminmax - 4*zxminmin - 2*zxmaxmin + 2*zxmaxmax + 4*zxminmax - 3*zyminmin + 3*zymaxmin + 3*zymaxmax - 3*zyminmax - 2*zxyminmin - zxymaxmin - zxymaxmax - 2*zxyminmax)*t1*u3;
    result_dx += 3 * (2*zminmin - 2*zmaxmin + zxminmin + zxmaxmin) * t2 *u0;
    result_dx += 3 * (2*zyminmin - 2*zymaxmin + zxyminmin + zxymaxmin) * t2 * u1;
    result_dx += 3 * (-6*zminmin + 6*zmaxmin - 6*zmaxmax + 6*zminmax - 3*zxminmin - 3*zxmaxmin + 3*zxmaxmax + 3*zxminmax - 4*zyminmin + 4*zymaxmin + 2*zymaxmax - 2*zyminmax - 2*zxyminmin - 2*zxymaxmin - zxymaxmax - zxyminmax) * t2 * u2;
    result_dx += 3 * (4*zminmin - 4*zmaxmin + 4*zmaxmax - 4*zminmax + 2*zxminmin + 2*zxmaxmin - 2*zxmaxmax - 2*zxminmax + 2*zyminmin - 2*zymaxmin - 2*zymaxmax + 2*zyminmax + zxyminmin + zxymaxmin + zxymaxmax + zxyminmax) * t2 * u3;

    result_dy += zyminmin * t0 * u0;
    result_dy += 2 * (-3*zminmin + 3*zminmax - 2*zyminmin - zyminmax) * t0 * u1;
    result_dy += 3 * (2*zminmin-2*zminmax + zyminmin + zyminmax) * t0 * u2;
    result_dy += zxyminmin*t1*u0;
    result_dy += 2 * (-3*zxminmin + 3*zxminmax - 2*zxyminmin - zxyminmax) * t1 * u1;
    result_dy += 3 * (2*zxminmin - 2*zxminmax + zxyminmin + zxyminmax) * t1 * u2;
    result_dy += (-3*zyminmin + 3*zymaxmin - 2*zxyminmin - zxymaxmin) * t2 * u0;
    result_dy += 2 * (9*zminmin - 9*zmaxmin + 9*zmaxmax - 9*zminmax + 6*zxminmin + 3*zxmaxmin - 3*zxmaxmax - 6*zxminmax + 6*zyminmin - 6*zymaxmin - 3*zymaxmax + 3*zyminmax + 4*zxyminmin + 2*zxymaxmin + zxymaxmax + 2*zxyminmax) * t2 * u1;
    result_dy += 3 * (-6*zminmin + 6*zmaxmin - 6*zmaxmax + 6*zminmax - 4*zxminmin - 2*zxmaxmin + 2*zxmaxmax + 4*zxminmax - 3*zyminmin + 3*zymaxmin + 3*zymaxmax - 3*zyminmax - 2*zxyminmin - zxymaxmin - zxymaxmax - 2*zxyminmax) * t2 * u2;
    result_dy += (2*zyminmin - 2*zymaxmin + zxyminmin + zxymaxmin) * t3 * u0;
    result_dy += 2 * (-6*zminmin + 6*zmaxmin - 6*zmaxmax + 6*zminmax - 3*zxminmin - 3*zxmaxmin + 3*zxmaxmax + 3*zxminmax - 4*zyminmin + 4*zymaxmin + 2*zymaxmax - 2*zyminmax - 2*zxyminmin - 2*zxymaxmin - zxymaxmax - zxyminmax) * t3 * u1;
    result_dy += 3 * (4*zminmin - 4*zmaxmin + 4*zmaxmax - 4*zminmax + 2*zxminmin + 2*zxmaxmin - 2*zxmaxmax - 2*zxminmax + 2*zyminmin - 2*zymaxmin - 2*zymaxmax + 2*zyminmax + zxyminmin + zxymaxmin + zxymaxmax + zxyminmax) * t3 * u2;

    return {result, result_dx, result_dy};
}



inline void Interpolator::coeff_calc(std::vector<double> &tridiag_solution, double dy, double dx, size_t index, double * b, double * c, double * d){

    *b = (dy / dx) - dx * (tridiag_solution[index + 1] + 2.0 * tridiag_solution[index]) / 3.0;
    *c = tridiag_solution[index];
    *d = (tridiag_solution[index+1] - tridiag_solution[index]) / (3.0 * dx);

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



void Interpolator::cspline_init(std::vector<double> &px, std::vector<double> &data){


    int num_points = px.size();
    int max_index = num_points - 1;  
    int sys_size = max_index - 1;
    
    std::vector<double> diagonal(num_points);
    std::vector<double> off_diagonal(num_points);
    std::vector<double> rhs(num_points);
    tridiag_solution.resize(num_points,0.0);

    for (int i = 0; i < sys_size; i++)
    {
        const double h_i   = px[i + 1] - px[i];
        const double h_ip1 = px[i + 2] - px[i + 1];
        const double ydiff_i   = data[i + 1] - data[i];
        const double ydiff_ip1 = data[i + 2] - data[i + 1];
        const double g_i = (h_i != 0.0) ? 1.0 / h_i : 0.0;
        const double g_ip1 = (h_ip1 != 0.0) ? 1.0 / h_ip1 : 0.0;
        off_diagonal[i] = h_ip1;
        diagonal[i] = 2.0 * (h_ip1 + h_i);
        rhs[i] = 3.0 * (ydiff_ip1 * g_ip1 -  ydiff_i * g_i);

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

        alpha[i] = diagonal[i] - off_diagonal[i - 1] * gamma[i - 1];
        gamma[i] = off_diagonal[i] / alpha[i];
        if (alpha[i] == 0) {
            std::cerr << __FILE__ << " " << __LINE__ << "ERROR: div by zero" << std::endl;
            exit(1);            
        }

    }

    if (sys_size > 1) {
        alpha[sys_size - 1] = diagonal[(sys_size - 1)] - off_diagonal[(sys_size - 2)] * gamma[sys_size - 2];
    }

    // RHS of equation
    z[0] = rhs[0];
    for (int i = 1; i < sys_size; i++) {
        z[i] = rhs[i] - gamma[i - 1] * z[i - 1];
    }

    for (int i = 0; i < sys_size; i++){
        c[i] = z[i] / alpha[i];
    }

    // back substitution
    tridiag_solution[sys_size] = c[sys_size - 1];
    if (sys_size >= 2) {
        for (int i = sys_size - 2; i >= 0; i--) {
            tridiag_solution[i+1] = c[i] - gamma[i] * tridiag_solution[i + 2];
        }
    }  
}

double Interpolator::cspline_eval_deriv(std::vector<double> &px, std::vector<double> &data, double value, int length) {

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
    coeff_calc(tridiag_solution, dy, dx, index, &b_i, &c_i, &d_i);

    // Evaluate derivative: dy/dx = b + 2c*delx + 3d*delx^2
    double dydx = b_i + delx * (2.0 * c_i + 3.0 * d_i * delx);

    return dydx;
}
