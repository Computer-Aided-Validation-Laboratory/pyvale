// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

// Program Header files
#include "./dicshapefunc.hpp"

    // Shape function declarations
    void Affine::get_pixel(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p){
        x_new = p[0] + (1.0+p[2]) * x + p[3] * y;
        y_new = p[1] + (1.0+p[5]) * y + p[4] * x;
    }

    void Rigid::get_pixel(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p){
        x_new = p[0] + x;
        y_new = p[1] + y;
    }

    void Quad::get_pixel(double &x_new, double &y_new, const double x, const double y, const std::vector<double> &p){
        x_new = p[0] + (1.0+p[2])*x + p[3]*y + p[6]*x*x + p[7]*x*y + p[8]*y*y;
        y_new = p[1] + (1.0+p[5])*y + p[4]*x + p[9]*x*x + p[10]*x*y + p[11]*y*y;
    }

    void Quad::get_displacement(double &u, double &v, const double x, const double y, const std::vector<double> &p){
        double x_new = p[0] + (1.0+p[2])*x + p[3]*y + p[6]*x*x + p[7]*x*y + p[8]*y*y;
        double y_new = p[1] + (1.0+p[5])*y + p[4]*x + p[9]*x*x + p[10]*x*y + p[11]*y*y;
        u = x_new - x;
        v = y_new - y;
    }

    void Affine::get_displacement(double &u, double &v, const double x, const double y, const std::vector<double> &p){
        double x_new = p[0] + (1.0+p[2]) * x + p[3] * y;
        double y_new = p[1] + (1.0+p[5]) * y + p[4] * x;
        u = x_new - x;
        v = y_new - y;
    }

    void Rigid::get_displacement(double &u, double &v, const double x, const double y, const std::vector<double> &p){
        u = p[0];
        v = p[1];
    }


    void Affine::get_dshape_dp(std::vector<double> &dfdp, const double x, const double y, const double dfdx, const double dfdy){
        dfdp[0] = dfdx;
        dfdp[1] = dfdy;
        dfdp[2] = dfdx * x;
        dfdp[3] = dfdx * y;
        dfdp[4] = dfdy * x;
        dfdp[5] = dfdy * y;
    }

    void Rigid::get_dshape_dp(std::vector<double> &dfdp, const double x, const double y,  const double dfdx, const double dfdy){
            dfdp[0] = dfdx;
            dfdp[1] = dfdy;
    }

    void Quad::get_dshape_dp(std::vector<double> &dfdp, const double x, const double y, const double dfdx, const double dfdy){
        dfdp[0]  = dfdx;
        dfdp[1]  = dfdy;
        dfdp[2]  = dfdx * x;
        dfdp[3]  = dfdx * y;
        dfdp[4]  = dfdy * x;
        dfdp[5]  = dfdy * y;
        dfdp[6]  = dfdx * x*x;
        dfdp[7]  = dfdx * x*y;
        dfdp[8]  = dfdx * y*y;
        dfdp[9]  = dfdy * x*x;
        dfdp[10] = dfdy * x*y;
        dfdp[11] = dfdy * y*y;
    }

    void Rigid::compose(std::vector<double> &pC, const std::vector<double> &pA, const std::vector<double> &pB){
        pC[0] = pA[0] + pB[0];
        pC[1] = pA[1] + pB[1];
    }

    void Affine::compose(std::vector<double> &pC, const std::vector<double> &pA, const std::vector<double> &pB){
        pC[0] = pB[0] + (1.0+pB[2])*pA[0] + pB[3]*pA[1];
        pC[1] = pB[1] + (1.0+pB[5])*pA[1] + pB[4]*pA[0];
        pC[2] = (1.0+pB[2])*(1.0+pA[2]) + pB[3]*pA[4]  - 1.0;
        pC[3] = (1.0+pB[2])*pA[3]       + pB[3]*(1.0+pA[5]);
        pC[4] = (1.0+pB[5])*pA[4]       + pB[4]*(1.0+pA[2]);
        pC[5] = (1.0+pB[5])*(1.0+pA[5]) + pB[4]*pA[3]  - 1.0;
    }

    void Quad::compose(std::vector<double> &pC, const std::vector<double> &pA, const std::vector<double> &pB){
        pC[0] = pB[0] + (1.0+pB[2])*pA[0] + pB[3]*pA[1];
        pC[1] = pB[1] + (1.0+pB[5])*pA[1] + pB[4]*pA[0];
        pC[2] = (1.0+pB[2])*(1.0+pA[2]) + pB[3]*pA[4]  - 1.0;
        pC[3] = (1.0+pB[2])*pA[3]       + pB[3]*(1.0+pA[5]);
        pC[4] = (1.0+pB[5])*pA[4]       + pB[4]*(1.0+pA[2]);
        pC[5] = (1.0+pB[5])*(1.0+pA[5]) + pB[4]*pA[3]  - 1.0;
        pC[6]  = (1.0+pB[2])*pA[6]  + pB[3]*pA[9]  + pB[6]*(1.0+pA[2])*(1.0+pA[2]) + pB[7]*(1.0+pA[2])*pA[4]          + pB[9]*pA[4]*pA[4];
        pC[7]  = (1.0+pB[2])*pA[7]  + pB[3]*pA[10] + pB[6]*2.0*(1.0+pA[2])*pA[3]   + pB[7]*((1.0+pA[2])*(1.0+pA[5]) + pA[3]*pA[4]) + pB[9]*2.0*pA[4]*(1.0+pA[5]);
        pC[8]  = (1.0+pB[2])*pA[8]  + pB[3]*pA[11] + pB[6]*pA[3]*pA[3]             + pB[7]*pA[3]*(1.0+pA[5])           + pB[9]*(1.0+pA[5])*(1.0+pA[5]);
        pC[9]  = pB[4]*pA[6]  + (1.0+pB[5])*pA[9]  + pB[9]*(1.0+pA[2])*(1.0+pA[2]) + pB[10]*(1.0+pA[2])*pA[4]         + pB[11]*pA[4]*pA[4];
        pC[10] = pB[4]*pA[7]  + (1.0+pB[5])*pA[10] + pB[9]*2.0*(1.0+pA[2])*pA[3]   + pB[10]*((1.0+pA[2])*(1.0+pA[5]) + pA[3]*pA[4]) + pB[11]*2.0*pA[4]*(1.0+pA[5]);
        pC[11] = pB[4]*pA[8]  + (1.0+pB[5])*pA[11] + pB[9]*pA[3]*pA[3]             + pB[10]*pA[3]*(1.0+pA[5])          + pB[11]*(1.0+pA[5])*(1.0+pA[5]);
    }

void Affine::compose_inverse(std::vector<double>& p_new,
                     const std::vector<double>& p,
                     const std::vector<double>& dp)
{
    // Build 2×2 warp matrices  A = I + Jp,  B = I + Jdp
    const double A[2][2] = { {1.0 + p[2],        p[3]  },
                              {       p[4],  1.0 + p[5] } };
    const double B[2][2] = { {1.0 + dp[2],        dp[3] },
                              {       dp[4], 1.0 + dp[5] } };

    // Invert B
    const double detB = B[0][0]*B[1][1] - B[0][1]*B[1][0];
    const double invB[2][2] = { { B[1][1]/detB, -B[0][1]/detB },
                                 {-B[1][0]/detB,  B[0][0]/detB } };

    // C = A · invB
    const double C[2][2] = {
        { A[0][0]*invB[0][0] + A[0][1]*invB[1][0],
          A[0][0]*invB[0][1] + A[0][1]*invB[1][1] },
        { A[1][0]*invB[0][0] + A[1][1]*invB[1][0],
          A[1][0]*invB[0][1] + A[1][1]*invB[1][1] }
    };

    // Translation:  t_new = t_p - C · t_dp
    p_new[0] = p[0] - (C[0][0]*dp[0] + C[0][1]*dp[1]);
    p_new[1] = p[1] - (C[1][0]*dp[0] + C[1][1]*dp[1]);

    // Deformation gradient
    p_new[2] = C[0][0] - 1.0;
    p_new[3] = C[0][1];
    p_new[4] = C[1][0];
    p_new[5] = C[1][1] - 1.0;
}

void Rigid::compose_inverse(std::vector<double>& p_new,
                     const std::vector<double>& p,
                     const std::vector<double>& dp)
{
    // Rigid motion has only two parameters: horizontal and vertical
    // translation. Do not access affine parameters here.
    p_new[0] = p[0] - dp[0];
    p_new[1] = p[1] - dp[1];
}


void Quad::compose_inverse(std::vector<double>& p_new,
                     const std::vector<double>& p,
                     const std::vector<double>& dp)
{
    // ICGN requires p_new = p composed with the inv of dp. A quad
    // warp has no exact quad inv, so retain terms through second
    // order. This is exact for the affine case and updates all 12 parameters.

    // Find c such that dp(c) = 0. The affine inv is an excellent initial
    // guess; Newton refinement accounts for the quad terms in dp.
    //
    double c[2] = {0.0, 0.0};
    const double affine_dp[2][2] = {{1.0 + dp[2], dp[3]},
                                    {dp[4], 1.0 + dp[5]}};

    double inv_affine_dp[2][2] = {};

    if (invert_2x2(affine_dp, inv_affine_dp)) {
        c[0] = -(inv_affine_dp[0][0] * dp[0] + inv_affine_dp[0][1] * dp[1]);
        c[1] = -(inv_affine_dp[1][0] * dp[0] + inv_affine_dp[1][1] * dp[1]);
    }

    for (int iteration = 0; iteration < 8; ++iteration) {

        double value[2] = {};

        eval_quad(dp, c[0], c[1], value[0], value[1]);

        if (std::max(std::abs(value[0]), std::abs(value[1])) < 1e-14) break;

        double jac[2][2] = {};
        quad_jac(dp, c[0], c[1], jac);
        double inv_jac[2][2] = {};
        if (!invert_2x2(jac, inv_jac)) break;

        c[0] -= inv_jac[0][0]*value[0] + inv_jac[0][1]*value[1];
        c[1] -= inv_jac[1][0]*value[0] + inv_jac[1][1]*value[1];
    }

    double dp_jac[2][2] = {};
    quad_jac(dp, c[0], c[1], dp_jac);
    double inv_dp_jac[2][2] = {};
    if (!invert_2x2(dp_jac, inv_dp_jac)) {
        std::fill(p_new.begin(), p_new.end(), 0.0);
        return;
    }

    // G(z) = c + Bz + R(z), where R(z) = -B Q_dp(Bz).
    double dp_quad[2][3] = {};
    transform_quad(dp[6], dp[7], dp[8], inv_dp_jac, dp_quad[0]);
    transform_quad(dp[9], dp[10], dp[11], inv_dp_jac, dp_quad[1]);

    double inv_quad[2][3] = {};
    for (int coef = 0; coef < 3; ++coef) {
        inv_quad[0][coef] = -(inv_dp_jac[0][0] * dp_quad[0][coef] + inv_dp_jac[0][1] * dp_quad[1][coef]);
        inv_quad[1][coef] = -(inv_dp_jac[1][0] * dp_quad[0][coef] + inv_dp_jac[1][1] * dp_quad[1][coef]);
    }

    double p_at_c[2] = {};
    eval_quad(p, c[0], c[1], p_at_c[0], p_at_c[1]);

    double p_jac[2][2] = {};
    quad_jac(p, c[0], c[1], p_jac);

    double linear[2][2] = {};
    double quad[2][3] = {};

    for (int row = 0; row < 2; ++row) {

        for (int col = 0; col < 2; ++col) {
            linear[row][col] = p_jac[row][0] * inv_dp_jac[0][col] + p_jac[row][1] * inv_dp_jac[1][col];
        }

        for (int coef = 0; coef < 3; ++coef) {
            quad[row][coef] = p_jac[row][0] * inv_quad[0][coef] + p_jac[row][1] * inv_quad[1][coef];
        }
    }

    double p_quad[2][3] = {};
    transform_quad(p[6], p[7], p[8], inv_dp_jac, p_quad[0]);
    transform_quad(p[9], p[10], p[11], inv_dp_jac, p_quad[1]);

    for (int coef = 0; coef < 3; ++coef) {
        quad[0][coef] += p_quad[0][coef];
        quad[1][coef] += p_quad[1][coef];
    }

    p_new[0] = p_at_c[0];
    p_new[1] = p_at_c[1];
    p_new[2] = linear[0][0] - 1.0;
    p_new[3] = linear[0][1];
    p_new[4] = linear[1][0];
    p_new[5] = linear[1][1] - 1.0;
    p_new[6] = quad[0][0];
    p_new[7] = quad[0][1];
    p_new[8] = quad[0][2];
    p_new[9] = quad[1][0];
    p_new[10] = quad[1][1];
    p_new[11] = quad[1][2];
}

void Quad::eval_quad(const std::vector<double> &p,
                          const double x,
                          const double y,
                          double &u,
                          double &v) {
    u = p[0] + (1.0+p[2])*x + p[3]*y + p[6]*x*x +  p[7]*x*y + p[8]*y*y;
    v = p[1] + p[4]*x + (1.0+p[5])*y + p[9]*x*x + p[10]*x*y + p[11]*y*y;
}

bool Quad::invert_2x2(const double matrix[2][2], double inv[2][2]) {

    const double det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    if (std::abs(det) < 1e-14) return false;

    inv[0][0] = matrix[1][1] / det;
    inv[0][1] = -matrix[0][1] / det;
    inv[1][0] = -matrix[1][0] / det;
    inv[1][1] = matrix[0][0] / det;
    return true;
}

void Quad::transform_quad(const double qxx,
                                const double qxy,
                                const double qyy,
                                const double matrix[2][2],
                                double coefs[3]) {

    const double ax = matrix[0][0];
    const double ay = matrix[0][1];
    const double bx = matrix[1][0];
    const double by = matrix[1][1];

    coefs[0] = qxx * ax * ax + qxy * ax * bx + qyy * bx * bx;
    coefs[1] = 2.0 * qxx * ax * ay + qxy * (ax * by + ay * bx) + 2.0 * qyy * bx * by;
    coefs[2] = qxx * ay * ay + qxy * ay * by + qyy * by * by;
}

void Quad::quad_jac(const std::vector<double> &p,
                          const double x,
                          const double y,
                          double jac[2][2]) {

    jac[0][0] = 1.0 + p[2] + 2.0 * p[6] * x + p[7] * y;
    jac[0][1] = p[3] + p[7] * x + 2.0 * p[8] * y;
    jac[1][0] = p[4] + 2.0 * p[9] * x + p[10] * y;
    jac[1][1] = 1.0 + p[5] + p[10] * x + 2.0 * p[11] * y;

}
