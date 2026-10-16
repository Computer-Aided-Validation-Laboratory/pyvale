#ifndef OPTIMIZERBASE_HPP
#define OPTIMIZERBASE_HPP

#include <cstdint>
#include <vector>

#include "./dicsubset.hpp"
#include "./dicshapefunc.hpp"
#include "./dicutil.hpp"

struct OptResult {
    std::vector<double> p;
    double u = 0.0;
    double v = 0.0;
    double mag = 0.0;
    double ftol = 0.0;
    double xtol = 0.0;
    int iter = 0;
    double cost = 0.0;
    uint8_t converged = false;
    uint8_t above_thresh = false;

    explicit OptResult(size_t num_params) : p(num_params, 0.0) {}
};

class OptimizerBase {
public:
    virtual ~OptimizerBase() = default;

    virtual OptResult solve(double cx,
                            double cy,
                            Subset<double> &ss_ref,
                            Subset<double> &ss_def,
                            const Interpolator &interp_def,
                            bool check_on_thresh = false) = 0;

    void set_rigid_displacement(double dx, double dy);
    void copy_params_from_fft(int idx,
                              const std::vector<double> &shift_x,
                              const std::vector<double> &shift_y);
    void copy_params_from_neigh(const std::vector<double> &results_p, int idx);
    void copy_params_from_neigh(const std::vector<double> &results_p,
                                const std::vector<double> &results_cost,
                                const std::vector<uint8_t> &results_above_thresh,
                                const std::vector<int> &neigh,
                                int fallback_idx);
    int average_params_from_neigh(const std::vector<double> &results_p,
                                  const std::vector<uint8_t> &successful,
                                  const std::vector<int> &neigh);
    void reset_params();

    const int num_params;
    int max_iter;
    std::vector<double> p;
    std::vector<double> dp;
    std::vector<double> pdp;

protected:
    using PixelFn = void (*)(double &, double &, double, double,
                             const std::vector<double> &);
    using JacobianFn = void (*)(std::vector<double> &, double, double,
                                double, double);
    using DisplacementFn = void (*)(double &, double &, double, double,
                                    const std::vector<double> &);
    using ComposeInverseFn = void (*)(std::vector<double> &,
                                      const std::vector<double> &,
                                      const std::vector<double> &);

    OptimizerBase(util::EShapeFunc shape_func, int max_iter, int num_px);

    double calculate_zncc(const Subset<double> &ss_ref,
                          const Subset<double> &ss_def) const;

    static int get_num_params(util::EShapeFunc shape_func);
    void set_shape(util::EShapeFunc shape_func);

    PixelFn get_pixel = nullptr;
    JacobianFn get_dfdp = nullptr;
    DisplacementFn get_displacement = nullptr;
    ComposeInverseFn compose_inverse = nullptr;
};

#endif // OPTIMIZERBASE_HPP
