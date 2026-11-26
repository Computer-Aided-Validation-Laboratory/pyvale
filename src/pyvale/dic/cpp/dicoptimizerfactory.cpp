#include "./dicoptimizerfactory.hpp"

#include "./dicoptimizerfalm.hpp"
#include "./dicoptimizericgn.hpp"

std::unique_ptr<OptimizerBase> make_optimizer(
    util::EOptimizer optimizer,
    util::EShapeFunc shape_func,
    util::ECorrCrit corr_crit,
    int max_iter,
    double precision,
    double threshold,
    int num_px) {
    if (optimizer == util::EOptimizer::ICGN) {
        return std::make_unique<OptimizerICGN>(
            shape_func, corr_crit, max_iter, precision, threshold, num_px);
    }
    return std::make_unique<OptimizerFALM>(
        shape_func, corr_crit, max_iter, precision, threshold, num_px);
}
