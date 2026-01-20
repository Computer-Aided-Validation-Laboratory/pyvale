// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef DICOPTIMIZER_H
#define DICOPTIMIZER_H

// STD library Header files
#include <vector>

// Program Header files
#include "./dicresults.hpp"
#include "./dicinterpolator.hpp"


namespace optimizer {


    /**
     * @brief 
     * 
     */
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
        double threshold;
        int px_vert;
        int px_hori;


        // Constructor to initialize vectors and other parameters
        Parameters(int num_params_, int max_iter_, double precision_, 
                   double threshold_, int px_vert_, int px_hori_,
                   const std::string& corr_crit)
            :
            num_params(num_params_),
            lambda(0.01),
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
            threshold(threshold_),
            px_vert(px_vert_),
            px_hori(px_hori_) {
            }
    };

    /**
     * @brief This function gets called before the corrolation optimization starts. Sets the function pointer for the user specified shape function.
     * 
     * @param[in] corr_crit string for the correlation criteria, e.g. "SSD", "NSSD", "ZNSSD".
     */
    void set_cost_function(const std::string& corr_crit);

    /**
     * @brief 
     * 
     * @param ss_x 
     * @param ss_y 
     * @param iter 
     * @param costp 
     * @param ftol 
     * @param xtol 
     * @param p 
     */
    void debug_print(int ss_x, int ss_y, int iter, double costp, double ftol, double xtol, const std::vector<double>& p);


    /**
     * @brief 
     * 
     * @param ss_x 
     * @param ss_y 
     * @param ss_ref 
     * @param ss_def 
     * @param interp_ref 
     * @param opt 
     * @return OptResult 
     */
    OptResult solve(const double ss_x, const double ss_y, subset::Pixels &ss_ref, subset::Pixels &ss_def, const Interpolator &interp_ref, optimizer::Parameters &opt, const std::string &corr_crit);

    /**
     * @brief calcutes the Sum of Squared Differences (SSD) between reference and deformed subsets.
     * 
     * @param[in] ss_ref reference subset
     * @param[in,out] ss_def deformed subset
     * @param[in] interp_def interpolator for deformed image 
     * @param[in,out] opt Optimization parameters including gradient, Hessian, etc. 
     */
    void   ssd(const subset::Pixels &ss_ref, subset::Pixels &ss_def, const Interpolator &interp_def, optimizer::Parameters &opt);

    /**
     * @brief calcutes the Normalized Sum of Squared Differences (NSSD) between reference and deformed subsets.
     * 
     * @param[in] ss_ref reference subset
     * @param[in,out] ss_def deformed subset
     * @param[in] interp_def interpolator for deformed image 
     * @param[in,out] opt Optimization parameters including gradient, Hessian, etc. 
     */
    void  nssd(const subset::Pixels &ss_ref, subset::Pixels &ss_def, const Interpolator &interp_def, optimizer::Parameters &opt);

    /**
     * @brief calcutes the Zero Normalized Sum of Squared Differences (ZNSSD) between reference and deformed subsets.
     * 
     * @param[in] ss_ref reference subset
     * @param[in,out] ss_def deformed subset
     * @param[in] interp_def interpolator for deformed image 
     * @param[in,out] opt Optimization parameters including gradient, Hessian, etc. 
     */
    void znssd(const subset::Pixels &ss_ref, subset::Pixels &ss_def, const Interpolator &interp_def, optimizer::Parameters &opt);


    /**
     * @brief Inverts square matrix using Gauss-Jordan elimination.
     * 
     * @param[in] matrix 
     * @param[out] inverse 
     * @param[in] augmented 
     * @param[in] num_params Number of shape function parameters (2 for rigid, 6 for affine, ...)
     * @return true Matrix inversion was successful
     * @return false Matrix inversion failed
     */
    bool invertMatrix(const std::vector<double>& matrix, std::vector<double>& inverse, std::vector<double>& augmented, int num_params);

    /**
     * @brief Updates the shape function parameters based on the current and updated parameters.
     * 
     * @param[out] pdp shape function parameters for P+deltaP
     * @param[in] p current shape function parameters P
     * @param[out] dp the change in shape function for based on the Hessian and gradient
     * @param[in] invH inverse of the Hessian matrix
     * @param[in] g gradient vector
     * @param[in] num_params Number of shape function parameters (2 for rigid, 6 for affine, ...)
     */
    void update_shapefunc_parameters(std::vector<double> &pdp, std::vector<double> &p, std::vector<double> &dp, std::vector<double> &invH, std::vector<double> &g, int num_params);

    /**
     * @brief 
     * 
     * @param[in] costp cost value for current shape function parameters P
     * @param[in] costpdp cost value for updated shape function parameters P+deltaP
     * @param[out] p shape function parameters for P
     * @param[in] pdp shape function parameters for P+deltaP
     * @param lambda Optimization damping factor
     * @param[in] num_params Number of shape function parameters (2 for rigid, 6 for affine, ...)
     */
    void update_lambda(double costp, double costpdp, std::vector<double> &p, std::vector<double> &pdp, double &lambda, int num_params);

    /**
     * @brief Populates the lower triangular part of the Hessian matrix, H.
     * 
     * @param[in,out] H Hessian matrix
     * @param[out] lambda Optimization damping factor
     * @param[in] num_params Number of shape function parameters (2 for rigid, 6 for affine, ...)
     */
    void populate_hessian_lower_tri(std::vector<double> &H, double lambda, int num_params);

    /**
     * @brief 
     * 
     * @param[out] x_new 
     * @param[out] y_new 
     * @param[in] x 
     * @param[in] y 
     * @param[in] p 
     */
    inline void affine(double &x_new, double &y_new, double x, double y, std::vector<double> &p);

    /**
     * @brief 
     * 
     * @param[out] x_new 
     * @param[out] y_new 
     * @param[in] x 
     * @param[in] y 
     * @param[in] p 
     */
    inline void rigid(double &x_new, double &y_new, double x, double y, std::vector<double> &p);

    /**
     * @brief 
     * 
     * @param[out] x_new 
     * @param[out] y_new 
     * @param[in] x 
     * @param[in] y 
     * @param[in] p 
     */
    inline void quad(double &x_new, double &y_new, double x, double y, std::vector<double> &p);

    /**
     * @brief calculates the derivative of the affine function with respect to each parameter
     * 
     * @param[out dfdp 
     * @param[in] x 
     * @param[in] y 
     * @param[in] dfdx 
     * @param[in] dfdy 
     */
    inline void daffine_dp(std::vector<double> &dfdp, double x, double y, double dfdx, double dfdy);

    /**
     * @brief calculates the derivative of the rigid shape function with respect to each parameter
     * 
     * @param[out dfdp 
     * @param[in] x 
     * @param[in] y 
     * @param[in] dfdx 
     * @param[in] dfdy 
     */
    inline void drigid_dp(std::vector<double> &dfdp, double x, double y, double dfdx, double dfdy);

    /**
     * @brief calculates the derivative of the quadratic function with respect to each parameter
     * 
     * @param[out] x_new 
     * @param[out] y_new 
     * @param[in] x 
     * @param[in] y 
     * @param[in] p 
     */
    inline void dquad_dp(std::vector<double> &dfdp, double x, double y, double dfdy);

    /**
     * @brief Funcion to convert affine shape function parameters to displacement values
     * 
     * @param[out] displacements values (u,v, magnitude) are added to results
     * @param[in] ss_x subset x coordinate
     * @param[in] ss_y subset y coordinate
     * @param[in] p shape function parameters
     */
    void quad_parameters_to_displacement(OptResult &results, double ss_x, double ss_y, std::vector<double> &p);

    /**
     * @brief Funcion to convert affine shape function parameters to displacement values
     * 
     * @param[out] displacements values (u,v, magnitude) are added to results
     * @param[in] ss_x subset x coordinate
     * @param[in] ss_y subset y coordinate
     * @param[in] p shape function parameters
     */
    void affine_parameters_to_displacement(OptResult &results, double ss_x, double ss_y, std::vector<double> &p);

    /**
     * @brief Funcion to convert affine shape function parameters to displacement values
     * 
     * @param[out] displacements values (u,v, magnitude) are added to results
     * @param[in] ss_x subset x coordinate
     * @param[in] ss_y subset y coordinate
     * @param[in] p shape function parameters
     */
    void rigid_parameters_to_displacement(OptResult &results, double ss_x, double ss_y, std::vector<double> &p);

}

#endif //DICOPTIMIZER_H
