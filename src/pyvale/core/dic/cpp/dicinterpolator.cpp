// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>

// Program Header files
#include "./dicinterpolator.hpp"
#include "./dicutil.hpp"



#define IDX2D(i, j, w) ((j) * (w) + (i))


namespace interpolator {

    std::vector<double> zx;
    std::vector<double> zy;
    std::vector<double> zxy;
    std::vector<double> tridiag_solution;
    std::vector<double> px_y;
    std::vector<double> px_x;
    std::vector<double> *image;
    int px_vertical;
    int px_horizontal;

    void bicubic_init(util::Image *img){

        // intitialise vars used globally within interpolator.
        image = &img->vals;
        px_vertical = img->px_vertical;
        px_horizontal = img->px_horizontal;

        // allocate memory for pixel coordinate arrays
        px_y.resize(px_vertical);
        px_x.resize(px_horizontal);

        // allocate memory for image derivatives
        zx.resize(px_vertical*px_horizontal);
        zy.resize(px_vertical*px_horizontal);
        zxy.resize(px_vertical*px_horizontal);


        for (int i = 0; i < px_horizontal; ++i) {
            px_x[i] = i; 
        }
        for (int j = 0; j < px_vertical; ++j) {
            px_y[j] = j; 

        }

        std::vector<double> data(px_horizontal,0);
        for (int j = 0; j < px_vertical; j++){

            // get 1D data
            for (int i = 0; i < px_horizontal; ++i) {
                data[i] = (*image)[j * px_horizontal + i];
            }

            cspline_init(px_x, data);
            for (int i = 0; i < px_horizontal; i++){
                zx[j * px_horizontal + i] = cspline_eval_deriv(px_x, data, px_x[i], px_horizontal);
            }
        }

        data.resize(px_vertical,0);
        for (int i = 0; i < px_horizontal; ++i) {

            // get 1D data
            for (int j = 0; j < px_vertical; j++){
                data[j] = (*image)[j * px_horizontal + i];
            }

            cspline_init(px_y, data);
            for (int j = 0; j < px_vertical; j++){
                zy[j * px_horizontal + i] = cspline_eval_deriv(px_y, data, px_y[j], px_vertical);
            }
        }


        data.resize(px_horizontal,0);
        for (int j = 0; j < px_vertical; j++){

            // get 1D data
            for (int i = 0; i < px_horizontal; ++i) {
                data[i] = zy[j * px_horizontal + i];
            }

            cspline_init(px_x, data);
            for (int i = 0; i < px_horizontal; i++){
                zxy[j * px_horizontal + i] = cspline_eval_deriv(px_x, data, px_x[i], px_horizontal);
            }
        }




    }

    double eval_bicubic(double x, double y){

        // get indices
        size_t xi = index_lookup(px_x, x, 0, px_horizontal - 1);
        size_t yi = index_lookup(px_y, y, 0, px_vertical - 1);

        // precompute indices of surrounding pixel values
        size_t idx00 = IDX2D(xi, yi, px_horizontal);
        size_t idx01 = IDX2D(xi, yi + 1, px_horizontal);
        size_t idx10 = IDX2D(xi + 1, yi, px_horizontal);
        size_t idx11 = IDX2D(xi + 1, yi + 1, px_horizontal);

        /* Precompute values for the grid points */
        double zminmin = (*image)[idx00];
        double zminmax = (*image)[idx01];
        double zmaxmin = (*image)[idx10];
        double zmaxmax = (*image)[idx11];

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
        double t1 = (x - px_x[xi]);
        double u1 = (y - px_y[yi]);
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




    double eval_bicubic_dx(double x, double y){

        /* first compute the indices into the data arrays where we are interpolating */
        size_t xi = index_lookup(px_x, x, 0, px_horizontal - 1);
        size_t yi = index_lookup(px_y, y, 0, px_vertical - 1);

        // precompute indices of surrounding pixel values
        size_t idx00 = IDX2D(xi, yi, px_horizontal);
        size_t idx01 = IDX2D(xi, yi + 1, px_horizontal);
        size_t idx10 = IDX2D(xi + 1, yi, px_horizontal);
        size_t idx11 = IDX2D(xi + 1, yi + 1, px_horizontal);

        double zminmin = (*image)[idx00];
        double zminmax = (*image)[idx01];
        double zmaxmin = (*image)[idx10];
        double zmaxmax = (*image)[idx11];

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
        double t1 = (x - px_x[xi]);
        double u1 = (y - px_y[yi]);
        double t2 = t1 * t1;
        double u2 = u1 * u1;  
        double u3 = u1 * u2;


        double result = 0.0;
        result = 0;
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


    double eval_bicubic_dy(double x, double y){

        /* first compute the indices into the data arrays where we are interpolating */
        size_t xi = index_lookup(px_x, x, 0, px_horizontal - 1);
        size_t yi = index_lookup(px_y, y, 0, px_vertical - 1);

        // precompute indices of surrounding pixel values
        size_t idx00 = IDX2D(xi, yi, px_horizontal);
        size_t idx01 = IDX2D(xi, yi + 1, px_horizontal);
        size_t idx10 = IDX2D(xi + 1, yi, px_horizontal);
        size_t idx11 = IDX2D(xi + 1, yi + 1, px_horizontal);

        double zminmin = (*image)[idx00];
        double zminmax = (*image)[idx01];
        double zmaxmin = (*image)[idx10];
        double zmaxmax = (*image)[idx11];

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
        double t1 = (x - px_x[xi]);
        double u1 = (y - px_y[yi]);
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


    Data eval_bicubic_and_derivs(double x, double y){

        // pixel floor of x and y 
        size_t xi = index_lookup(px_x, x, 0, px_horizontal - 1);
        size_t yi = index_lookup(px_y, y, 0, px_vertical - 1);

        // precompute indices of surrounding pixel values
        size_t idx00 = IDX2D(xi, yi, px_horizontal);
        size_t idx01 = IDX2D(xi, yi + 1, px_horizontal);
        size_t idx10 = IDX2D(xi + 1, yi, px_horizontal);
        size_t idx11 = IDX2D(xi + 1, yi + 1, px_horizontal);

        double zminmin = (*image)[idx00];
        double zminmax = (*image)[idx01];
        double zmaxmin = (*image)[idx10];
        double zmaxmax = (*image)[idx11];

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
        double t1 = (x - px_x[xi]);
        double u1 = (y - px_y[yi]);
        double t2 = t1 * t1;
        double u2 = u1 * u1;  
        double t3 = t1 * t2;
        double u3 = u1 * u2;

        double result, result_dx, result_dy;

        result = 0.0;
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

        result_dx = 0;
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

        result_dy = 0.0;
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



    inline void coeff_calc(std::vector<double> &tridiag_solution, double dy, double dx, size_t index, double * b, double * c, double * d){

        *b = (dy / dx) - dx * (tridiag_solution[index + 1] + 2.0 * tridiag_solution[index]) / 3.0;
        *c = tridiag_solution[index];
        *d = (tridiag_solution[index+1] - tridiag_solution[index]) / (3.0 * dx);

    }

    inline int index_lookup(std::vector<double> &px, double x, size_t index_lo, size_t index_hi){
        
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

        
        
        if (x >= px[index_lo] && x <= px[index_hi]) {
            return static_cast<int>(x); // Return x as the index
        }
        else {
            std::cerr << "ERROR in \'" << __FILE__ << "\' at line \'" << __LINE__ << "\' \n";
            std::cerr << "value is out of bounds. value = " << x << std::endl;
            exit(EXIT_FAILURE);
        }

    }


    void cspline_init(std::vector<double> &px, std::vector<double> &data){


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


        //--------------------------------------------------------------------------------
        //Cholesky decomposition
        // A = L.D.L^t
        // lower_diag(L) = gamma
        // diag(D) = alpha
        //--------------------------------------------------------------------------------


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

    double cspline_eval_deriv (std::vector<double> &px, std::vector<double> &data, double value, int length){

        double dx;
        double dydx;

        int index = index_lookup(px, value, 0, length-1);

        /* evaluate */
        double px_max = px[index + 1];
        double px_min = px[index];
        dx = px_max - px_min;

        if (dx > 0.0)
        {
            const double y_lo = data[index];
            const double y_hi = data[index + 1];
            const double dy = y_hi - y_lo;
            double delx = value - px_min;
            double b_i, c_i, d_i; 
            coeff_calc(tridiag_solution, dy, dx, index,  &b_i, &c_i, &d_i);
            dydx = b_i + delx * (2.0 * c_i + 3.0 * d_i * delx);
            // std::cout << dydx << std::endl;
        }
        else
        {
            dydx = 0.0;
        }
        return dydx;
        exit(0);

    }

}