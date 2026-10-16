#include "./dicoptimizerbase.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

OptimizerBase::OptimizerBase(util::EShapeFunc shape_func,
                             int max_iter_,
                             int num_px)
    : num_params(get_num_params(shape_func)),
      max_iter(max_iter_),
      p(num_params, 0.0),
      dp(num_params, 0.0),
      pdp(num_params, 0.0) {
    (void)num_px;
    set_shape(shape_func);
}

int OptimizerBase::get_num_params(util::EShapeFunc shape_func) {
    switch (shape_func) {
        case util::EShapeFunc::RIGID: return Rigid::num_params;
        case util::EShapeFunc::AFFINE: return Affine::num_params;
        case util::EShapeFunc::QUAD: return Quad::num_params;
    }
    throw std::invalid_argument("Unknown shape function");
}

void OptimizerBase::set_shape(util::EShapeFunc shape_func) {
    switch (shape_func) {

        case util::EShapeFunc::RIGID:
            get_pixel = &Rigid::get_pixel;
            get_dfdp = &Rigid::get_dshape_dp;
            get_displacement = &Rigid::get_displacement;
            compose_inverse = &Rigid::compose_inverse;
            break;

        case util::EShapeFunc::AFFINE:
            get_pixel = &Affine::get_pixel;
            get_dfdp = &Affine::get_dshape_dp;
            get_displacement = &Affine::get_displacement;
            compose_inverse = &Affine::compose_inverse;
            break;

        case util::EShapeFunc::QUAD:
            get_pixel = &Quad::get_pixel;
            get_dfdp = &Quad::get_dshape_dp;
            get_displacement = &Quad::get_displacement;
            compose_inverse = &Quad::compose_inverse;
            break;

    }
}

void OptimizerBase::set_rigid_displacement(double dx, double dy) {
    std::fill(p.begin(), p.end(), 0.0);
    p[0] = dx;
    p[1] = dy;
}

void OptimizerBase::copy_params_from_fft(int idx,
                                         const std::vector<double> &shift_x,
                                         const std::vector<double> &shift_y) {
    std::fill(p.begin(), p.end(), 0.0);
    p[0] = shift_x[idx];
    p[1] = shift_y[idx];
}

void OptimizerBase::copy_params_from_neigh(const std::vector<double> &results_p,
                                           int idx) {
    std::copy_n(results_p.begin() + idx * num_params, num_params, p.begin());
}

void OptimizerBase::copy_params_from_neigh(
    const std::vector<double> &results_p,
    const std::vector<double> &results_cost,
    const std::vector<uint8_t> &results_above_thresh,
    const std::vector<int> &neigh,
    int fallback_idx) {
    std::fill(p.begin(), p.end(), 0.0);
    double weight_sum = 0.0;
    int count = 0;

    for (int nidx : neigh) {
        if (count == 4) break;
        if (!results_above_thresh[nidx] || results_cost[nidx] <= 0.0) continue;

        const double weight = results_cost[nidx];
        for (int i = 0; i < num_params; ++i)
            p[i] += weight * results_p[nidx * num_params + i];
        weight_sum += weight;
        ++count;
    }

    if (weight_sum <= 0.0) {
        copy_params_from_neigh(results_p, fallback_idx);
        return;
    }
    for (double &value : p) value /= weight_sum;
}

int OptimizerBase::average_params_from_neigh(
    const std::vector<double> &results_p,
    const std::vector<uint8_t> &successful,
    const std::vector<int> &neigh) {
    std::fill(p.begin(), p.end(), 0.0);
    int count = 0;

    for (int nidx : neigh) {
        if (!successful[nidx]) continue;
        for (int i = 0; i < num_params; ++i)
            p[i] += results_p[nidx * num_params + i];
        ++count;
    }

    if (count > 0)
        for (double &value : p) value /= count;
    return count;
}

void OptimizerBase::reset_params() {
    std::fill(p.begin(), p.end(), 0.0);
}

double OptimizerBase::calculate_zncc(const Subset<double> &ss_ref,
                                     const Subset<double> &ss_def) const {
    double mean_ref = 0.0;
    double mean_def = 0.0;

    for (int i = 0; i < ss_ref.num_px; ++i) {
        mean_ref += ss_ref.vals[i];
        mean_def += ss_def.vals[i];
    }

    mean_ref /= ss_ref.num_px;
    mean_def /= ss_def.num_px;

    double ref_norm = 0.0;
    double def_norm = 0.0;

    for (int i = 0; i < ss_ref.num_px; ++i) {
        const double ref = ss_ref.vals[i] - mean_ref;
        const double def = ss_def.vals[i] - mean_def;
        ref_norm += ref * ref;
        def_norm += def * def;
    }

    if (ref_norm < 1e-12 || def_norm < 1e-12) return 0.0;

    double corr = 0.0;
    for (int i = 0; i < ss_ref.num_px; ++i) {
        corr += (ss_ref.vals[i] - mean_ref) *
                (ss_def.vals[i] - mean_def);
    }

    return corr / std::sqrt(ref_norm * def_norm);
}
