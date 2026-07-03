// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================



#include <vector>
#include <cmath>
#include <algorithm>

#include "./dicinterpBspline.hpp"

#include "../../common_cpp/util.hpp"



Bspline::Bspline(const Image &img) {

    common_util::Timer time("to init " + img.filename + " interp:", 2);

    this->px_hori = img.width;
    this->px_vert = img.height;

    padded_hori = px_hori + 4;
    padded_vert = px_vert + 4;
    coeff_padded.resize(padded_hori * padded_vert, 0.0);

    // Lambda to get clamped pixel value regardless of type
    auto getpix = [&](int x, int y) -> double {
        x = std::clamp(x, 0, px_hori - 1);
        y = std::clamp(y, 0, px_vert - 1);
        if (img.type == PixelType::UINT8)  return img.data8 [y * px_hori + x];
        if (img.type == PixelType::UINT16) return img.data16[y * px_hori + x];
        if (img.type == PixelType::UINT32) return img.data32[y * px_hori + x];
        throw std::runtime_error("Unsupported pixel type");
    };

    // Fill entire padded array using clamped reads — handles interior, edges, and corners
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < padded_vert; y++)
        for (int x = 0; x < padded_hori; x++)
            coeff_padded[y * padded_hori + x] = getpix(x - 2, y - 2);

    prefilter_x();
    prefilter_y();
}

// 1D cubic B-spline basis and derivatives
inline void Bspline::basis(double t, double B[4]) {
    double tt = t*t, ttt = tt*t;

    B[0] = (1.0 - 3.0*t + 3.0*tt - ttt) / 6.0;
    B[1] = (4.0 - 6.0*tt + 3.0*ttt) / 6.0;
    B[2] = (1.0 + 3.0*t + 3.0*tt - 3.0*ttt) / 6.0;
    B[3] = ttt / 6.0;
}

inline void Bspline::basis_d(double t, double Bd[4]) {
    double tt = t*t;

    Bd[0] = (-3.0 + 6.0*t - 3.0*tt) / 6.0;
    Bd[1] = (-12.0*t + 9.0*tt) / 6.0;
    Bd[2] = (3.0 + 6.0*t - 9.0*tt) / 6.0;
    Bd[3] = (3.0*tt) / 6.0;
}


void Bspline::prefilter_x() {

    const double z = std::sqrt(3.0) - 2.0;
    const double lambda = (1.0 - z)*(1.0 - 1.0/z);

    // Normalize
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < padded_vert; y++)
        for (int x = 0; x < padded_hori; x++)
            coeff_padded[y*padded_hori + x] *= lambda;

    // Causal
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < padded_vert; y++) {
        double* row = &coeff_padded[y*padded_hori];
        for (int x = 1; x < padded_hori; x++)
            row[x] += z * row[x-1];
    }

    // Anticausal
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < padded_vert; y++) {
        double* row = &coeff_padded[y*padded_hori];
        row[padded_hori-1] = z/(z*z - 1.0) * row[padded_hori-1];
        for (int x = padded_hori-2; x >= 0; x--)
            row[x] = z*(row[x+1] - row[x]);
    }
}

// -------------------------------------------------------
// Prefilter along each column
// -------------------------------------------------------
void Bspline::prefilter_y() {
    const double z = std::sqrt(3.0) - 2.0;
    const double lambda = (1.0 - z)*(1.0 - 1.0/z);

    // Normalize
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < padded_vert; y++)
        for (int x = 0; x < padded_hori; x++)
            coeff_padded[y*padded_hori + x] *= lambda;

    // Causal
    #pragma omp parallel for schedule(static)
    for (int x = 0; x < padded_hori; x++) {
        for (int y = 1; y < padded_vert; y++)
            coeff_padded[y*padded_hori + x] += z * coeff_padded[(y-1)*padded_hori + x];
    }

    // Anticausal
    #pragma omp parallel for schedule(static)
    for (int x = 0; x < padded_hori; x++) {
        coeff_padded[(padded_vert-1)*padded_hori + x] = z/(z*z - 1.0) * coeff_padded[(padded_vert-1)*padded_hori + x];
        for (int y = padded_vert-2; y >= 0; y--)
            coeff_padded[y*padded_hori + x] = z*(coeff_padded[(y+1)*padded_hori + x] - coeff_padded[y*padded_hori + x]);
    }
}


double Bspline::eval(const int ss_x, const int ss_y, const double subpx_x, const double subpx_y) const {

    double x = std::clamp(subpx_x, 0.0, (double)(px_hori - 1));
    double y = std::clamp(subpx_y, 0.0, (double)(px_vert - 1));
    const int ix = (int)x + 2;
    const int iy = (int)y + 2;
    const double tx = x - (ix-2);
    const double ty = y - (iy-2);

    double Bx[4], By[4];
    basis(tx, Bx);
    basis(ty, By);

    // Row pointers
    const double* r0 = coeff_padded.data() + (iy-1)*padded_hori;
    const double* r1 = coeff_padded.data() + (iy  )*padded_hori;
    const double* r2 = coeff_padded.data() + (iy+1)*padded_hori;
    const double* r3 = coeff_padded.data() + (iy+2)*padded_hori;

    // Column indices
    int xx0 = ix-1, xx1 = ix, xx2 = ix+1, xx3 = ix+2;

    // Compute row sums
    double sum0 = r0[xx0]*Bx[0] + r0[xx1]*Bx[1] + r0[xx2]*Bx[2] + r0[xx3]*Bx[3];
    double sum1 = r1[xx0]*Bx[0] + r1[xx1]*Bx[1] + r1[xx2]*Bx[2] + r1[xx3]*Bx[3];
    double sum2 = r2[xx0]*Bx[0] + r2[xx1]*Bx[1] + r2[xx2]*Bx[2] + r2[xx3]*Bx[3];
    double sum3 = r3[xx0]*Bx[0] + r3[xx1]*Bx[1] + r3[xx2]*Bx[2] + r3[xx3]*Bx[3];

    // Final value
    return sum0*By[0] + sum1*By[1] + sum2*By[2] + sum3*By[3];
}

double Bspline::eval_dx(const int ss_x, const int ss_y, const double subpx_x, double subpx_y) const {
    int ix = (int)floor(subpx_x);
    int iy = (int)floor(subpx_y);

    double tx = subpx_x - ix;
    double ty = subpx_y - iy;

    double By[4], dBx[4];
    basis(ty, By);
    basis_d(tx, dBx);

    double dfdx = 0.0;
    for (int j = 0; j < 4; j++) {
        int yy = std::clamp(iy + j - 1, 0, px_vert-1);
        for (int i = 0; i < 4; i++) {
            int xx = std::clamp(ix + i - 1, 0, px_hori-1);
            double c = coeff[yy*px_hori + xx];
            dfdx += c * dBx[i] * By[j];
        }
    }
    return dfdx;
}


double Bspline::eval_dy(const int ss_x, const int ss_y, const double subpx_x, double subpx_y) const {
    int ix = (int)floor(subpx_x);
    int iy = (int)floor(subpx_y);

    double tx = subpx_x - ix;
    double ty = subpx_y - iy;

    double Bx[4], dBy[4];
    basis(tx, Bx);
    basis_d(ty, dBy);

    double dfdy = 0.0;
    for (int j = 0; j < 4; j++) {
        int yy = std::clamp(iy + j - 1, 0, px_vert-1);
        for (int i = 0; i < 4; i++) {
            int xx = std::clamp(ix + i - 1, 0, px_hori-1);
            double c = coeff[yy*px_hori + xx];
            dfdy += c * Bx[i] * dBy[j];
        }
    }
    return dfdy;
}

InterpVals Bspline::eval_and_derivs(const int ss_x, const int ss_y, const double subpx_x, const double subpx_y) const {

    double x = std::clamp(subpx_x, 0.0, (double)(px_hori - 1));
    double y = std::clamp(subpx_y, 0.0, (double)(px_vert - 1));
    const int ix = (int)x + 2;
    const int iy = (int)y + 2;
    const double tx = x - (ix-2);
    const double ty = y - (iy-2);

    double Bx[4], By[4], dBx[4], dBy[4];
    basis(tx, Bx);
    basis(ty, By);
    basis_d(tx, dBx);
    basis_d(ty, dBy);

    // Precompute row pointers
    const double* r0 = coeff_padded.data() + (iy-1)*padded_hori;
    const double* r1 = coeff_padded.data() + (iy  )*padded_hori;
    const double* r2 = coeff_padded.data() + (iy+1)*padded_hori;
    const double* r3 = coeff_padded.data() + (iy+2)*padded_hori;

    // Column indices
    int xx0 = ix-1;
    int xx1 = ix;
    int xx2 = ix+1;
    int xx3 = ix+2;

    // Row sums
    double sum_f0 = r0[xx0]*Bx[0] + r0[xx1]*Bx[1] + r0[xx2]*Bx[2] + r0[xx3]*Bx[3];
    double sum_f1 = r1[xx0]*Bx[0] + r1[xx1]*Bx[1] + r1[xx2]*Bx[2] + r1[xx3]*Bx[3];
    double sum_f2 = r2[xx0]*Bx[0] + r2[xx1]*Bx[1] + r2[xx2]*Bx[2] + r2[xx3]*Bx[3];
    double sum_f3 = r3[xx0]*Bx[0] + r3[xx1]*Bx[1] + r3[xx2]*Bx[2] + r3[xx3]*Bx[3];

    // Compute value and derivatives
    double f    = sum_f0*By[0] + sum_f1*By[1] + sum_f2*By[2] + sum_f3*By[3];
    double dfdx = (r0[xx0]*dBx[0] + r0[xx1]*dBx[1] + r0[xx2]*dBx[2] + r0[xx3]*dBx[3])*By[0] +
                  (r1[xx0]*dBx[0] + r1[xx1]*dBx[1] + r1[xx2]*dBx[2] + r1[xx3]*dBx[3])*By[1] +
                  (r2[xx0]*dBx[0] + r2[xx1]*dBx[1] + r2[xx2]*dBx[2] + r2[xx3]*dBx[3])*By[2] +
                  (r3[xx0]*dBx[0] + r3[xx1]*dBx[1] + r3[xx2]*dBx[2] + r3[xx3]*dBx[3])*By[3];

    double dfdy = sum_f0*dBy[0] + sum_f1*dBy[1] + sum_f2*dBy[2] + sum_f3*dBy[3];

    return {f, dfdx, dfdy};
}
