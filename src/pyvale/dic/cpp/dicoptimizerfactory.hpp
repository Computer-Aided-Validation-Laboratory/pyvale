#ifndef DICOPTIMIZERFACTORY_HPP
#define DICOPTIMIZERFACTORY_HPP

#include <memory>

#include "./dicoptimizerbase.hpp"

std::unique_ptr<OptimizerBase> make_optimizer(
    util::EOptimizer optimizer,
    util::EShapeFunc shape_func,
    util::ECorrCrit corr_crit,
    int max_iter,
    double precision,
    double threshold,
    int num_px);

#endif // DICOPTIMIZERFACTORY_HPP
