#ifndef OPTIMIZERICGN_HPP
#define OPTIMIZERICGN_HPP

#include <vector>

#include "./dicoptimizerbase.hpp"
#include "./dicinterp.hpp"

// Inverse-compositional Gauss-Newton optimizer using the approximations in
// Appendix A1 of Blaber et al.  The reference steepest-descent images and
// Gauss-Newton Hessian are fixed for the duration of solve().
class OptimizerICGN : public OptimizerBase {
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
                    bool check_on_thresh = false) override;

private:
    util::ECorrCrit criteria;
    const double precision;
    const double threshold;
    int num_px;

    std::vector<double> dfdx_ref;
    std::vector<double> dfdy_ref;
    std::vector<double> dfdp;
    std::vector<double> steepest_descent;
    std::vector<double> H;
    std::vector<double> invH;
    std::vector<double> augmented;

    bool hessian_valid;

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

#endif // OPTIMIZERICGN_HPP
