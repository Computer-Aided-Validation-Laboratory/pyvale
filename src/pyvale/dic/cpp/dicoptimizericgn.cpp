#include "./dicoptimizericgn.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

OptimizerICGN::OptimizerICGN(util::EShapeFunc shape_func,
                             util::ECorrCrit cost_func,
                             int max_iter_,
                             double precision_,
                             double threshold_,
                             int num_px_)
    : OptimizerBase(shape_func, max_iter_, num_px_),
      criteria(cost_func),
      precision(precision_),
      threshold(threshold_),
      num_px(num_px_),
      dfdx_ref(num_px_, 0.0),
      dfdy_ref(num_px_, 0.0),
      dfdp(num_params, 0.0),
      steepest_descent(num_px_ * num_params, 0.0),
      H(num_params * num_params, 0.0),
      invH(num_params * num_params, 0.0),
      augmented(num_params * num_params * 2, 0.0),
      hessian_valid(false) {
}

void OptimizerICGN::calculate_reference_gradients(const Subset<double> &ss_ref) {

    for (int y = 0; y < ss_ref.size_y; ++y) {
        for (int x = 0; x < ss_ref.size_x; ++x) {

            const int i = y * ss_ref.size_x + x;
            if (x == 0) dfdx_ref[i] = ss_ref.vals[i + 1] - ss_ref.vals[i];
            else if (x == ss_ref.size_x - 1) dfdx_ref[i] = ss_ref.vals[i] - ss_ref.vals[i - 1];
            else dfdx_ref[i] = 0.5 * (ss_ref.vals[i + 1] - ss_ref.vals[i - 1]);

            if (y == 0) dfdy_ref[i] = ss_ref.vals[i + ss_ref.size_x] - ss_ref.vals[i];
            else if (y == ss_ref.size_y - 1) dfdy_ref[i] = ss_ref.vals[i] - ss_ref.vals[i - ss_ref.size_x];
            else dfdy_ref[i] = 0.5 * (ss_ref.vals[i + ss_ref.size_x] - ss_ref.vals[i - ss_ref.size_x]);
        }
    }
}

void OptimizerICGN::calculate_statistics(const Subset<double> &ss,
                                         double &mean,
                                         double &norm) const {
    mean = std::accumulate(ss.vals.begin(), ss.vals.end(), 0.0) / ss.num_px;
    double sum = 0.0;
    for (double value : ss.vals) {
        const double zero = value - mean;
        sum += zero * zero;
    }
    norm = std::sqrt(sum);
}

double OptimizerICGN::normalized_value(double value, double mean, double norm) const {
    return (value - mean) / norm;
}

void OptimizerICGN::precompute_hessian(const Subset<double> &ss_ref) {
    calculate_reference_gradients(ss_ref);
    std::fill(H.begin(), H.end(), 0.0);

    double mean_ref = 0.0;
    double norm_ref = 1.0;
    if (criteria == util::ECorrCrit::ZNSSD) {
        calculate_statistics(ss_ref, mean_ref, norm_ref);
    }
    else if (criteria == util::ECorrCrit::NSSD) {
        double sum = 0.0;
        for (double value : ss_ref.vals) sum += value * value;
        norm_ref = std::sqrt(sum);
    }

    hessian_valid = norm_ref > 1e-12;
    if (!hessian_valid) {
        std::fill(invH.begin(), invH.end(), 0.0);
        return;
    }

    for (int i = 0; i < ss_ref.num_px; ++i) {

        get_dfdp(dfdp, ss_ref.x[i], ss_ref.y[i], dfdx_ref[i], dfdy_ref[i]);

        const double scale = (criteria == util::ECorrCrit::SSD) ? 1.0 : 1.0 / norm_ref;

        for (int row = 0; row < num_params; ++row) {

            const double value = scale * dfdp[row];
            steepest_descent[i * num_params + row] = value;

            for (int col = row; col < num_params; ++col) {
                H[row * num_params + col] += value * scale * dfdp[col];
            }
        }
    }

    for (int row = 0; row < num_params; ++row)
        for (int col = row + 1; col < num_params; ++col)
            H[col * num_params + row] = H[row * num_params + col];

    hessian_valid = invert_matrix(H, invH, augmented, num_params);
}

void OptimizerICGN::calculate_current_cost(const Subset<double> &ss_ref,
                                           Subset<double> &ss_def,
                                           const Interpolator &interp_def,
                                           double cx, double cy,
                                           double &cost) {
    for (int i = 0; i < ss_def.num_px; ++i) {
        get_pixel(ss_def.x[i], ss_def.y[i], ss_ref.x[i], ss_ref.y[i], p);
        ss_def.vals[i] = interp_def.eval(cx, cy, cx + ss_def.x[i], cy + ss_def.y[i]);
    }

    double mean_ref = 0.0, norm_ref = 1.0;
    double mean_def = 0.0, norm_def = 1.0;
    if (criteria == util::ECorrCrit::ZNSSD) {
        calculate_statistics(ss_ref, mean_ref, norm_ref);
        calculate_statistics(ss_def, mean_def, norm_def);
    } else if (criteria == util::ECorrCrit::NSSD) {
        double sum_ref = 0.0, sum_def = 0.0;
        for (int i = 0; i < ss_def.num_px; ++i) {
            sum_ref += ss_ref.vals[i] * ss_ref.vals[i];
            sum_def += ss_def.vals[i] * ss_def.vals[i];
        }
        norm_ref = std::sqrt(sum_ref);
        norm_def = std::sqrt(sum_def);
    }

    cost = 0.0;
    if (norm_ref <= 1e-12 || norm_def <= 1e-12) {
        cost = 2.0;
        return;
    }
    for (int i = 0; i < ss_def.num_px; ++i) {
        const double ref = criteria == util::ECorrCrit::SSD ? ss_ref.vals[i] : normalized_value(ss_ref.vals[i], mean_ref, norm_ref);
        const double def = criteria == util::ECorrCrit::SSD ? ss_def.vals[i] : normalized_value(ss_def.vals[i], mean_def, norm_def);
        const double residual = def - ref;
        cost += residual * residual;
    }
}

void OptimizerICGN::calculate_trial_cost(const Subset<double> &ss_ref,
                                         Subset<double> &ss_def,
                                         const Interpolator &interp_def,
                                         double cx, double cy,
                                         const std::vector<double> &trial,
                                         double &cost) {
    const std::vector<double> saved = p;
    p = trial;
    calculate_current_cost(ss_ref, ss_def, interp_def, cx, cy, cost);
    p = saved;
}

void OptimizerICGN::calculate_gradient(const Subset<double> &ss_ref,
                                       const Subset<double> &ss_def,
                                       std::vector<double> &gradient) const {

    std::fill(gradient.begin(), gradient.end(), 0.0);
    double mean_ref = 0.0, norm_ref = 1.0;
    double mean_def = 0.0, norm_def = 1.0;

    if (criteria == util::ECorrCrit::ZNSSD) {

        calculate_statistics(ss_ref, mean_ref, norm_ref);
        calculate_statistics(ss_def, mean_def, norm_def);

    } 
    else if (criteria == util::ECorrCrit::NSSD) {

        double sum_ref = 0.0, sum_def = 0.0;
        for (int i = 0; i < ss_def.num_px; ++i) {
            sum_ref += ss_ref.vals[i] * ss_ref.vals[i];
            sum_def += ss_def.vals[i] * ss_def.vals[i];
        }
        norm_ref = std::sqrt(sum_ref);
        norm_def = std::sqrt(sum_def);
    }

    for (int i = 0; i < ss_def.num_px; ++i) {

        const double ref = criteria == util::ECorrCrit::SSD ? ss_ref.vals[i] : normalized_value(ss_ref.vals[i], mean_ref, norm_ref);
        const double def = criteria == util::ECorrCrit::SSD ? ss_def.vals[i] : normalized_value(ss_def.vals[i], mean_def, norm_def);
        const double residual = def - ref;

        for (int j = 0; j < num_params; ++j) {
            gradient[j] += steepest_descent[i * num_params + j] * residual;
        }
    }
}

OptResult OptimizerICGN::solve(const double cx, const double cy,
                               Subset<double> &ss_ref,
                               Subset<double> &ss_def,
                               const Interpolator &interp_def,
                               const bool check_on_thresh) {
    precompute_hessian(ss_ref);

    int iter = 0;
    double ftol = 0.0;
    double xtol = 0.0;
    bool converged = false;
    double cost = 0.0;

    while (iter < max_iter && hessian_valid) {

        calculate_current_cost(ss_ref, ss_def, interp_def, cx, cy, cost);

        std::vector<double> gradient(num_params, 0.0);
        calculate_gradient(ss_ref, ss_def, gradient);

        for (int i = 0; i < num_params; ++i) {

            dp[i] = 0.0;
            for (int j = 0; j < num_params; ++j) {
                dp[i] += invH[i * num_params + j] * gradient[j];
            }
        }

        compose_inverse(pdp, p, dp);
        double trial_cost = 0.0;
        calculate_trial_cost(ss_ref, ss_def, interp_def, cx, cy, pdp, trial_cost);

        int backtrack = 0;
        while (trial_cost > cost && backtrack < 4) {
            for (double &value : dp) value *= 0.5;
            compose_inverse(pdp, p, dp);
            calculate_trial_cost(ss_ref, ss_def, interp_def, cx, cy, pdp, trial_cost);
            ++backtrack;
        }

        const double dp_norm = std::sqrt(std::inner_product(dp.begin(), dp.end(), dp.begin(), 0.0));
        const double p_norm = std::sqrt(std::inner_product(p.begin(), p.end(), p.begin(), 0.0));

        xtol = dp_norm / (p_norm + 1e-10);
        ftol = std::abs(trial_cost - cost) / (std::abs(cost) + 1e-10);
        p = pdp;
        cost = trial_cost;

        converged = xtol < precision && ftol < precision;
        const bool good_enough = 1.0 - 0.5 * cost > threshold;
        ++iter;
        if (converged && (criteria == util::ECorrCrit::SSD || !check_on_thresh || good_enough))
            break;
    }

    calculate_current_cost(ss_ref, ss_def, interp_def, cx, cy, cost);

    OptResult result(num_params);

    get_displacement(result.u, result.v, 0.0, 0.0, p);

    result.iter = iter;
    result.ftol = ftol;
    result.xtol = xtol;
    result.p = p;
    result.cost = calculate_zncc(ss_ref, ss_def);
    result.converged = converged;

    const bool in_bounds = std::isfinite(result.u) && std::isfinite(result.v) &&
                           std::abs(result.u) <= static_cast<double>(interp_def.px_hori) &&
                           std::abs(result.v) <= static_cast<double>(interp_def.px_vert);

    result.valid_match = in_bounds && result.cost >= threshold;

    return result;
}

bool OptimizerICGN::invert_matrix(const std::vector<double> &matrix,
                                  std::vector<double> &inverse,
                                  std::vector<double> &augmented,
                                  int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            augmented[i * 2 * n + j] = matrix[i * n + j];
            augmented[i * 2 * n + n + j] = i == j ? 1.0 : 0.0;
        }
    }
    for (int i = 0; i < n; ++i) {
        int pivot_row = i;
        double pivot_abs = std::abs(augmented[i * 2 * n + i]);
        for (int r = i + 1; r < n; ++r) {
            const double value = std::abs(augmented[r * 2 * n + i]);
            if (value > pivot_abs) {
                pivot_abs = value;
                pivot_row = r;
            }
        }
        if (pivot_abs < 1e-20) {
            std::fill(inverse.begin(), inverse.end(), 0.0);
            return false;
        }
        if (pivot_row != i)
            for (int j = 0; j < 2 * n; ++j)
                std::swap(augmented[i * 2 * n + j], augmented[pivot_row * 2 * n + j]);
        const double pivot = augmented[i * 2 * n + i];
        for (int j = 0; j < 2 * n; ++j) augmented[i * 2 * n + j] /= pivot;
        for (int r = 0; r < n; ++r) {
            if (r == i) continue;
            const double factor = augmented[r * 2 * n + i];
            for (int j = 0; j < 2 * n; ++j)
                augmented[r * 2 * n + j] -= factor * augmented[i * 2 * n + j];
        }
    }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            inverse[i * n + j] = augmented[i * 2 * n + n + j];
    return true;
}

