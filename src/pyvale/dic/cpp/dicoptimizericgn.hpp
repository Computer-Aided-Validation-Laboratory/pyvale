#ifndef DICOPTIMIZERICGN_HPP
#define DICOPTIMIZERICGN_HPP

#include <cstdint>
#include <vector>

#include "./dicinterp.hpp"
#include "./dicoptimizer.hpp"
#include "./dicsubset.hpp"
#include "./dicutil.hpp"

// Inverse-compositional Gauss-Newton optimizer using the approximations in
// Appendix A1 of Blaber et al.  The reference steepest-descent images and
// Gauss-Newton Hessian are fixed for the duration of solve().
class OptimizerICGN {
public:
    OptimizerICGN(util::EShapeFunc shape_func,
                  util::ECorrCrit cost_func,
                  int max_iter,
                  double precision,
                  double threshold,
                  int num_px);

    OptResult solve(const double cx,
                    const double cy,
                    Subset<double> &ss_ref,
                    Subset<double> &ss_def,
                    const Interpolator &interp_def,
                    bool check_on_thresh = false);

    const int num_params;
    int max_iter;
    std::vector<double> p;
    std::vector<double> dp;
    std::vector<double> pdp;

    void set_rigid_displacement(double dx, double dy);
    void copy_params_from_fft(const int idx,
                              const std::vector<double> &shift_x,
                              const std::vector<double> &shift_y);
    void copy_params_from_neigh(const std::vector<double> &results_p,
                                const int idx);
    void copy_params_from_neigh(const std::vector<double> &results_p,
                                const std::vector<double> &results_cost,
                                const std::vector<uint8_t> &results_above_thresh,
                                const std::vector<int> &neigh,
                                const int fallback_idx);
    int average_params_from_neigh(const std::vector<double> &results_p,
                                  const std::vector<uint8_t> &successful,
                                  const std::vector<int> &neigh);
    void reset_params();

private:
    using PixelFn = void (*)(double &, double &, double, double,
                             const std::vector<double> &);
    using JacobianFn = void (*)(std::vector<double> &, double, double,
                                double, double);
    using DisplacementFn = void (*)(double &, double &, double, double,
                                    const std::vector<double> &);
    using ComposeInverseFn = void (*)(std::vector<double> &,
                                      const std::vector<double> &,
                                      const std::vector<double> &);

    util::ECorrCrit criteria;
    const double precision;
    const double threshold;
    int num_px;

    PixelFn get_pixel;
    JacobianFn get_dfdp;
    DisplacementFn get_displacement;
    ComposeInverseFn compose_inverse;

    std::vector<double> dfdx_ref;
    std::vector<double> dfdy_ref;
    std::vector<double> dfdp;
    std::vector<double> steepest_descent;
    std::vector<double> H;
    std::vector<double> invH;
    std::vector<double> augmented;

    bool hessian_valid;

    static int get_num_params(util::EShapeFunc shape_func);
    void set_shape(util::EShapeFunc shape_func);

    void calculate_reference_gradients(const Subset<double> &ss_ref);
    void precompute_hessian(const Subset<double> &ss_ref);
    void calculate_current_cost(const Subset<double> &ss_ref,
                                Subset<double> &ss_def,
                                const Interpolator &interp_def,
                                double cx,
                                double cy,
                                double &cost);
    void calculate_trial_cost(const Subset<double> &ss_ref,
                              Subset<double> &ss_def,
                              const Interpolator &interp_def,
                              double cx,
                              double cy,
                              const std::vector<double> &trial,
                              double &cost);
    void calculate_gradient(const Subset<double> &ss_ref,
                            const Subset<double> &ss_def,
                            std::vector<double> &gradient) const;
    void calculate_statistics(const Subset<double> &ss,
                              double &mean,
                              double &norm) const;
    double normalized_value(double value, double mean, double norm) const;
    double calculate_zncc(const Subset<double> &ss_ref,
                          const Subset<double> &ss_def) const;
    bool invert_matrix(const std::vector<double> &matrix,
                       std::vector<double> &inverse,
                       std::vector<double> &augmented,
                       int n);
};

#endif
