// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================



#include <vector>
#include <cmath>
#include <algorithm>

#include "./dicinterpBspline.hpp"

Bspline::Bspline(double* img, int px_hori, int px_vert){

    // intitialise vars used globally within Interpolator.
    this->image = img;
    this->px_vert = px_vert;
    this->px_hori = px_hori;
    coeff.resize(px_vert*px_hori);

    for (int i = 0; i < px_hori*px_vert; i++){
        coeff[i] = img[i];
    }

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
    for (int y = 0; y < px_vert; y++)
        for (int x = 0; x < px_hori; x++)
            coeff[y*px_hori + x] *= lambda;

    // Causal
    for (int y = 0; y < px_vert; y++) {
        double* row = &coeff[y*px_hori];
        for (int x = 1; x < px_hori; x++)
            row[x] += z * row[x-1];
    }

    // Anticausal
    for (int y = 0; y < px_vert; y++) {
        double* row = &coeff[y*px_hori];
        row[px_hori-1] = z/(z*z - 1.0) * row[px_hori-1];
        for (int x = px_hori-2; x >= 0; x--)
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
    for (int y = 0; y < px_vert; y++)
        for (int x = 0; x < px_hori; x++)
            coeff[y*px_hori + x] *= lambda;

    // Causal
    for (int x = 0; x < px_hori; x++) {
        for (int y = 1; y < px_vert; y++)
            coeff[y*px_hori + x] += z * coeff[(y-1)*px_hori + x];
    }

    // Anticausal
    for (int x = 0; x < px_hori; x++) {
        coeff[(px_vert-1)*px_hori + x] = z/(z*z - 1.0) * coeff[(px_vert-1)*px_hori + x];
        for (int y = px_vert-2; y >= 0; y--)
            coeff[y*px_hori + x] = z*(coeff[(y+1)*px_hori + x] - coeff[y*px_hori + x]);
    }
}


double Bspline::eval(const int ss_x, const int ss_y, const double subpx_x, double subpx_y) const {
    int ix = (int)floor(subpx_x);
    int iy = (int)floor(subpx_y);

    double tx = subpx_x - ix;
    double ty = subpx_y - iy;

    double Bx[4], By[4];
    basis(tx, Bx);
    basis(ty, By);

    double f = 0.0;
    for (int j = 0; j < 4; j++) {
        int yy = std::clamp(iy + j - 1, 0, px_vert-1);

        for (int i = 0; i < 4; i++) {
            int xx = std::clamp(ix + i - 1, 0, px_hori-1);
            double c = coeff[yy*px_hori + xx];
            f += c * Bx[i] * By[j];
        }
    }
    return f;
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

InterpVals Bspline::eval_and_derivs(const int ss_x, const int ss_y, const double subpx_x, double subpx_y) const {
    int ix = (int)floor(subpx_x);
    int iy = (int)floor(subpx_y);

    double tx = subpx_x - ix;
    double ty = subpx_y - iy;

    double Bx[4], By[4], dBx[4], dBy[4];
    basis(tx, Bx);
    basis(ty, By);
    basis_d(tx, dBx);
    basis_d(ty, dBy);

    InterpVals out {0,0,0};

    for (int j = 0; j < 4; j++) {
        int yy = std::clamp(iy + j - 1, 0, px_vert-1);
        for (int i = 0; i < 4; i++) {
            int xx = std::clamp(ix + i - 1, 0, px_hori-1);
            double c = coeff[yy*px_hori + xx];
            out.f += c * Bx[i] * By[j];
            out.dfdx += c * dBx[i] * By[j];
            out.dfdy += c * Bx[i] * dBy[j];
        }
    }
    return out;
}
