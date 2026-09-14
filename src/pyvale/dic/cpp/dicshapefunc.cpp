// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


// STD library Header files
#include <vector>
#include <string>
#include <cmath>

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


void Quad::compose_inverse(std::vector<double>& p_new,
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
