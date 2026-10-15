// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================


#ifndef OPTIMIZERFALM_HPP
#define OPTIMIZERFALM_HPP

// STD library Header files
#include <vector>
#include <string>

// Program Header files
#include "./dicoptimizerbase.hpp"
#include "./dicinterp.hpp"
#include "./dicutil.hpp"


class OptimizerFALM : public OptimizerBase {

    public:
        // Constructor
        OptimizerFALM(util::EShapeFunc shape_func,
                 util::ECorrCrit cost_func,
                 int max_iter,
                 double precision,
                 double threshold,
                 int ss_size);
        
        // Main solve method
        OptResult solve(const double cx, 
                  const double cy, 
                  Subset<double> &ss_ref, 
                  Subset<double> &ss_def, 
                  const Interpolator &interp_def,
                  const bool check_on_thresh=false) override;
        
    private:

        util::ECorrCrit criteria;
        double costp;
        double costpdp;
        std::vector<double> g;          // Gradient
        std::vector<double> dfdp;       // Derivative of shape function wrt parameters
        std::vector<double> dfdx;       // Derivative of shape function wrt parameters
        std::vector<double> dfdy;       // Derivative of shape function wrt parameters
        std::vector<double> H;          // Hessian
        std::vector<double> invH;       // Inverse Hessian
        std::vector<double> augmented;  // For matrix inversion
        double lambda;              // Damping parameter
        
        // Settings
        const double precision;
        const double threshold;
        int px_vert;
        int px_hori;
        
        // Points to the selected FALM cost function.
        void (OptimizerFALM::*optimize_cost)(const Subset<double>&, Subset<double>&, const Interpolator&, const double, const double);
    
        /**
        * @brief This function gets called before the corrolation optimization starts. Sets the function pointer for the user specified shape function.
        * 
        * @param[in] corr_crit correlation criteria enum.
        */
        void set_cost_function(util::ECorrCrit corr_crit);

        /**
        * @brief 
        * 
        * @param ss_x 
        * @param ss_y 
        */
        void debug_print(const int ss_x, const int ss_y, int iter, double costp, double ftol, double xtol);


        /**
        * @brief calcutes the Sum of Squared Differences (SSD) between reference and deformed subsets.
        * 
        * @param[in] ss_ref reference subset
        * @param[in,out] ss_def deformed subset
        * @param[in] interp_def interpolator for deformed image 
        * @param[in] cx x coordinate at subset centre
        * @param[in] cy y coordinate at subset centre
        */ 
        void ssd(const Subset<double> &ss_ref,
             Subset<double> &ss_def,
             const Interpolator &interp_def,
             const double cx,
             const double cy);

        /**
        * @brief calcutes the Normalized Sum of Squared Differences (NSSD) between reference and deformed subsets.
        * 
        * @param[in] ss_ref reference subset
        * @param[in,out] ss_def deformed subset
        * @param[in] interp_def interpolator for deformed image 
        * @param[in] cx x coordinate at subset centre
        * @param[in] cy y coordinate at subset centre
        */
        void nssd(const Subset<double> &ss_ref,
                  Subset<double> &ss_def,
                  const Interpolator &interp_def,
                  const double cx,
                  const double cy);

        /**
        * @brief calcutes the Zero Normalized Sum of Squared Differences (ZNSSD) between reference and deformed subsets.
        * 
        * @param[in] ss_ref reference subset
        * @param[in,out] ss_def deformed subset
        * @param[in] interp_def interpolator for deformed image 
        * @param[in] cx x coordinate at subset centre
        * @param[in] cy y coordinate at subset centre
        */
         void znssd(const Subset<double> &ss_ref,
                    Subset<double> &ss_def,
                    const Interpolator &interp_def,
                    const double cx,
                    const double cy);


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
        bool invert_matrix(const std::vector<double>& matrix, std::vector<double>& inverse, std::vector<double>& augmented, int num_params);

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


};

#endif // OPTIMIZERFALM_HPP
