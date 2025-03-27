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


    // externally accessible varaibles
    extern std::vector<double> p;
    extern double ftol;
    extern double xtol;
    extern int iter;


    // intitialisation and debugging
    void init(std::string &, std::string &, int);
    void setCostFunction(const std::string& corr_crit);
    void setShapeFunction(const std::string& shape_func);
    void debugPrint(int iter, double ftol, double xtol, const std::vector<double>& p);


    // Optimization routine
    void solve(std::vector<double> &, std::vector<double> &,  std::vector<double> &, int, double, int);

    // choice of cost function
    void   ssd(std::vector<double> &, std::vector<double> &,  std::vector<double> &, int);
    void  nssd(std::vector<double> &, std::vector<double> &,  std::vector<double> &, int);
    void znssd(std::vector<double> &, std::vector<double> &,  std::vector<double> &, int);


    // optimizer functions
    bool invertMatrix(const std::vector<std::vector<double>>& matrix, std::vector<std::vector<double>>& inverse);
    void update_shapefunc_parameters(std::vector<double> &pdp, std::vector<std::vector<double>> &invH, std::vector<double> &gradient);
    void update_lambda(double costfunc_p, double costfunc_pdp, std::vector<double> &p, std::vector<double> &pdp, double &lambda);
    double computeMagnitude(const std::vector<double>& vec);


    // shape functions and their derivatives with respect to optimization parameters
    void affine(double &x_new, double &y_new, double x, double y, std::vector<double> &p);
    void rigid(double &x_new, double &y_new, double x, double y, std::vector<double> &p);
    void quad(double &x_new, double &y_new, double x, double y, std::vector<double> &p);
    void daffine_dp(std::vector<double> &dfdp, double x, double y, double dfdx, double dfdy, int n);
    void drigid_dp(std::vector<double> &dfdp, double x, double y, double dfdx, double dfdy, int n);
    void dquad_dp(double &x_new, double &y_new, double x, double y, std::vector<double> &p);
}

#endif //DICOPTIMIZER_H
