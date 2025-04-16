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
#include <array>

// Program Header files
#include "./dicutil.hpp"


namespace optimizer {



    struct Parameters {
        int num_params;
        double lambda; // damping
        double costp; // cost function for current P values
        double costpdp; // cost function for P+deltaP values
        std::vector<double> g; // gradient
        std::vector<double> dfdp; // derivative of shape function with respect to parameters
        std::vector<double> H; // Hessian ( becomes (H + lambda * diag(H)) )
        std::vector<double> invH; // Used for inverse of (H + lambda * diag(H))
        std::vector<double> p; // hard coded affine parameters
        std::vector<double> dp; // deltaP
        std::vector<double> pdp; // P + deltaP
        std::vector<double> augmented;
        int max_iter;
        double precision;
        double threshold_lm;
        int px_vertical;
        int px_horizontal;


        // Constructor to initialize vectors and other parameters
        Parameters(int num_params_, int max_iter_, double precision_, double threshold_lm_, int px_vertical_, int px_horizontal_)
            :
            num_params(num_params_),
            lambda(0.001),
            costp(0.0),
            costpdp(0.0),
            g(num_params, 0.0),
            dfdp(num_params, 0.0),
            H(num_params*num_params, 0.0),
            invH(num_params*num_params, 0.0),
            p(num_params, 0.0),
            dp(num_params, 0.0),
            pdp(num_params, 0.0),
            augmented(num_params*num_params*2, 0.0),
            max_iter(max_iter_),
            precision(precision_),
            threshold_lm(threshold_lm_),
            px_vertical(px_vertical_),
            px_horizontal(px_horizontal_) {}
    };



    struct Results {
        std::vector<double> p;
        double u;
        double v;
        double mag;
        double ftol;
        double xtol;
        int iter;
        double cost;
    };


    // intitialisation and debugging
    void init(std::string &, std::string &);
    void setCostFunction(const std::string& corr_crit);
    void setShapeFunction(const std::string& shape_func);
    void debugPrint(int ss_x, int ss_y, int iter, double costp, double ftol, double xtol, const std::vector<double>& p);
    void init_parameters(optimizer::Parameters *opt, int ss_size);


    // Optimization routine
    Results solve(double ss_x, double ss_y, util::Subset *ss_def, util::Subset *ss_ref, optimizer::Parameters *opt);

    // choice of cost function
    void   ssd(double ss_x, double ss_y, util::Subset *ss_def, util::Subset *ss_ref, optimizer::Parameters *opt);
    void  nssd(double ss_x, double ss_y, util::Subset *ss_def, util::Subset *ss_ref, optimizer::Parameters *opt);
    void znssd(double ss_x, double ss_y, util::Subset *ss_def, util::Subset *ss_ref, optimizer::Parameters *opt);


    // optimizer functions
    bool invertMatrix(const std::vector<double>& matrix, std::vector<double>& inverse, std::vector<double>& augmented, int num_params);
    void update_shapefunc_parameters(std::vector<double> &pdp, std::vector<double> &p, std::vector<double> &dp, std::vector<double> &invH, std::vector<double> &g, int num_params);
    void update_lambda(double costp, double costpdp, std::vector<double> &p, std::vector<double> &pdp, double &lambda, int num_params);
    void populate_hessian_lower_tri(std::vector<double> &H, double lambda, int num_params);

    // shape functions and their derivatives with respect to optimization parameters
    void affine(double &x_new, double &y_new, double x, double y, std::vector<double> &p);
    void rigid(double &x_new, double &y_new, double x, double y, std::vector<double> &p);
    void quad(double &x_new, double &y_new, double x, double y, std::vector<double> &p);
    void daffine_dp(std::vector<double> &dfdp, double x, double y, double dfdx, double dfdy, int n);
    void drigid_dp(std::vector<double> &dfdp, double x, double y, double dfdx, double dfdy, int n);
    void dquad_dp(double &x_new, double &y_new, double x, double y, std::vector<double> &p);
    void affine_parameters_to_displacement(Results *results, double ss_x, double ss_y, std::vector<double> &p);
    void rigid_parameters_to_displacement(Results *results, double ss_x, double ss_y, std::vector<double> &p);

}

#endif //DICOPTIMIZER_H
