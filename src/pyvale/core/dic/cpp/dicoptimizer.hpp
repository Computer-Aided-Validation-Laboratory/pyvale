// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef DICOPTIMIZER_H
#define DICOPTIMIZER_H

// STD library Header files
#include <vector>
#include <iostream>
#include <chrono>

// Program Header files



namespace optimizer {



    struct Parameters {
        int iter; // number of iterations for each subset optimization
        double ftol; // tolerance for termination by the change of the cost function
        double xtol; // tolerance for termination by the change of the independent variables
        double lambda; // damping
        double costp; // cost function for current P values
        double costpdp; // cost function for P+deltaP values
        std::vector<double> g; // gradient
        std::vector<double> dfdp; // derivative of shape function with repsect to parameters
        std::vector<double> H; // Hessian ( becomes (H + lambda * diag(H)) )
        std::vector<double> invH; // Used for inverse of (H + lambda * diag(H))
        std::vector<double> ss_ref; // gray level values in reference subset
        std::vector<double> ss_ref_x; // x coordinate values of ref susbet
        std::vector<double> ss_ref_y; // y coordinate values of ref subset
        std::vector<double> p; // hard coded affine parameters
        std::vector<double> dp; // deltaP
        std::vector<double> pdp; // P + deltaP
    };



    struct Results {
        std::vector<double> p;
        double u;
        double v;
        double mag;
        double ftol;
        double xtol;
        int iter;
    };


    // intitialisation and debugging
    void init(std::string &, std::string &);
    void setCostFunction(const std::string& corr_crit);
    void setShapeFunction(const std::string& shape_func);
    void debugPrint(int iter, double ftol, double xtol, const std::vector<double>& p);
    void init_parameters(Parameters *params, int ss_size);


    // Optimization routine
    Results solve(double, double, std::vector<double> &, std::vector<double> &,  std::vector<double> &, int, double, int, Parameters *params);

    // choice of cost function
    void   ssd(std::vector<double> &, std::vector<double> &,  std::vector<double> &, int, Parameters *params);
    void  nssd(std::vector<double> &, std::vector<double> &,  std::vector<double> &, int, Parameters *params);
    void znssd(std::vector<double> &, std::vector<double> &,  std::vector<double> &, int, Parameters *params);


    // brute force optimizations
    void   brute_force_ssd(int ss_x, int ss_y, std::vector<double> &ss_def, std::vector<double> &ss_def_coords_x, std::vector<double> &ss_def_coords_y, int *image_ref, int px_vertical, int px_horizontal, int num_px_ss, int range, double tol);
    void  brute_force_nssd(int ss_x, int ss_y, std::vector<double> &ss_def, std::vector<double> &ss_def_coords_x, std::vector<double> &ss_def_coords_y, int *image_ref, int px_vertical, int px_horizontal, int num_px_ss, int range, double tol);
    void brute_force_znssd(int ss_x, int ss_y, std::vector<double> &ss_def, std::vector<double> &ss_def_coords_x, std::vector<double> &ss_def_coords_y, int *image_ref, int px_vertical, int px_horizontal, int num_px_ss, int range, double tol);


    // optimizer functions
    bool invertMatrix(const std::vector<double>& matrix, std::vector<double>& inverse);
    void update_shapefunc_parameters(std::vector<double> &pdp, std::vector<double> &p, std::vector<double> &dp, std::vector<double> &invH, std::vector<double> &gradient);
    void update_lambda(double costfunc_p, double costfunc_pdp, std::vector<double> &p, std::vector<double> &pdp, double &lambda);
    double computeMagnitude(const std::vector<double>& vec);
    void populate_hessian_lower_tri(std::vector<double> &H, double lambda);
    void affine_parameters_to_displacement(Results *results, double ss_x, double ss_y, std::vector<double> &p);

    // shape functions and their derivatives with respect to optimization parameters
    void affine(double &x_new, double &y_new, double x, double y, std::vector<double> &p);
    void rigid(double &x_new, double &y_new, double x, double y, std::vector<double> &p);
    void quad(double &x_new, double &y_new, double x, double y, std::vector<double> &p);
    void daffine_dp(std::vector<double> &dfdp, double x, double y, double dfdx, double dfdy, int n);
    void drigid_dp(std::vector<double> &dfdp, double x, double y, double dfdx, double dfdy, int n);
    void dquad_dp(double &x_new, double &y_new, double x, double y, std::vector<double> &p);
}

#endif //DICOPTIMIZER_H
