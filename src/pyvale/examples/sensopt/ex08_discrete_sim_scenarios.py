# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Place sensors for three named physics scenarios
================================================

A warm peak shifts across a rectangular plate with a fixed cool trough.
Compare reconstruction references and the order of the two reductions.
"""

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

import pyvale.dataio as io
import pyvale.sensoropt as opt
import pyvale.sensorsim as sens
from pyvale.examples.sensopt.dual_blob import (
    build_dual_blob_case,
    save_dual_blob_layout,
)
from pyvale.verif.analyticsimdatafactory import scalar_linear_2d

# %%
# 1. Define three input scenarios and a shared candidate grid
# -----------------------------------------------------------

mesh, _ = scalar_linear_2d()
coords = cast(npt.NDArray[np.float64], mesh.coords)
scenario_keys: tuple[str, ...] = ("left", "nominal", "right")
peak_xs: tuple[float, ...] = (2.0, 3.5, 5.0)
sim_scenarios: dict[str, io.SimData] = {}
for key, peak_x in zip(scenario_keys, peak_xs, strict=True):
    sim_scenarios[key] = build_dual_blob_case(
        mesh, peak=(peak_x, 5.0), trough=(7.5, 2.0)
    )

sites: np.ndarray = sens.gen_pos_grid_inside(
    num_sensors=(5, 3, 1),
    x_lims=(0.0, 10.0),
    y_lims=(0.0, 7.5),
    z_lims=(0.0, 0.0),
)
array: sens.SensorsPoint = sens.SensorFactory.scalar_point(
    sim_scenarios["nominal"],
    sens.SensorData(positions=sites[:3]),
    comp_key="temperature",
    spatial_dims=sens.EDim.TWOD,
    descriptor=sens.DescriptorFactory.temperature(),
)
array.set_error_chain([sens.ErrRandGen(sens.GenNormal(std=0.5, seed=123))])

# %%
# 2. Define reductions and the reconstruction target
# --------------------------------------------------

scenario_weights: dict[str, float] = dict(
    zip(scenario_keys, (0.2, 0.5, 0.3), strict=True)
)
worst_experiment: opt.EvalReduction = opt.EvalReduction(
    axis=opt.EEvalAxis.VIRTUAL_EXP, method=opt.EEvalReduce.MAX
)
expected_scenario: opt.EvalReduction = opt.EvalReduction(
    axis=opt.EEvalAxis.SIM_SCENARIO,
    method=opt.EEvalReduce.MEAN,
    weights=scenario_weights,
)
experiment_first: opt.EvalStrat = opt.EvalStrat(
    reductions=(worst_experiment, expected_scenario)
)
scenario_first: opt.EvalStrat = opt.EvalStrat(
    reductions=(expected_scenario, worst_experiment)
)
recon: opt.ObjFieldRecon = opt.ObjFieldRecon(
    target_positions=coords,
    reference_field="temperature",
    sensor_keys=("temperature",),
    component_index=0,
)

# %%
# 3. Compare ordered reductions and all three reference fields
# -------------------------------------------------------------

cases: dict[str, opt.ObjSpec] = {
    "scenario_reference": opt.ObjSpec(objective=recon, eval_strat=experiment_first),
    "scenario_first": opt.ObjSpec(objective=recon, eval_strat=scenario_first),
    "mean_reference": opt.ObjSpec(
        objective=replace(
            recon,
            reference_mode=opt.EReconRef.MEAN_SCENARIO,
            reference_weights=scenario_weights,
        ),
        eval_strat=experiment_first,
    ),
    "median_reference": opt.ObjSpec(
        objective=replace(recon, reference_mode=opt.EReconRef.MEDIAN_SCENARIO),
        eval_strat=experiment_first,
    ),
}
design_space: opt.DesignSpace = opt.DesignSpace(
    positions={"temperature": sites},
    position_axes={"temperature": opt.EDesignAxis.DISC},
)
output_path: Path = Path.cwd() / "pyvale-output" / "sensopt_ex08_discrete_sim_scenarios"
output_path.mkdir(parents=True, exist_ok=True)

results: dict[str, opt.OptResult] = {}
for name, objective in cases.items():
    result: opt.OptResult = opt.SensorOptEngine(
        sensor_arrays={"temperature": array},
        design_space=design_space,
        sim_scenarios=sim_scenarios,
        num_virtual_exps=4,
        objectives=(objective,),
        optimiser=opt.OptGreedyForward(num_sensors=3),
        seed=123,
    ).optimise()
    results[name] = result
    print(f"{name}: reduced RMSE {result.objective_values[0]:.6f}")
    save_dual_blob_layout(result, array, output_path / f"{name}.png", title=name)

# %%
# 4. Sample the named library with and without replacement
# --------------------------------------------------------

# Both runs use 24 virtual experiments per candidate. The nested run revisits
# two distinct fields with fresh sensor noise. The paired run draws a field
# for each experiment, with the library weights as draw probabilities.
library_runs: dict[str, tuple[opt.SimSamplerLibrary, opt.ESampling, int]] = {
    "nested_distinct": (
        opt.SimSamplerLibrary(
            library=sim_scenarios, num_sim_scenarios=2, replace=False
        ),
        opt.ESampling.NESTED,
        12,
    ),
    "paired_weighted": (
        opt.SimSamplerLibrary(
            library=sim_scenarios, num_sim_scenarios=24, weights=scenario_weights
        ),
        opt.ESampling.PAIRED,
        1,
    ),
}
scenario_mean: opt.EvalReduction = opt.EvalReduction(
    axis=opt.EEvalAxis.SIM_SCENARIO, method=opt.EEvalReduce.MEAN
)
experiment_mean: opt.EvalReduction = opt.EvalReduction(
    axis=opt.EEvalAxis.VIRTUAL_EXP, method=opt.EEvalReduce.MEAN
)
library_results: dict[str, opt.OptResult] = {}
for name, (sampler, sampling, num_virtual_exps) in library_runs.items():
    reductions = (scenario_mean,)
    if sampling is opt.ESampling.NESTED:
        reductions = (experiment_mean, scenario_mean)
    result = opt.SensorOptEngine(
        sensor_arrays={"temperature": array},
        design_space=design_space,
        sim_scenarios=sampler,
        sampling=sampling,
        num_virtual_exps=num_virtual_exps,
        objectives=(
            opt.ObjSpec(
                objective=recon, eval_strat=opt.EvalStrat(reductions=reductions)
            ),
        ),
        optimiser=opt.OptGreedyForward(num_sensors=3),
        seed=123,
    ).optimise()
    library_results[name] = result
    print(
        f"{name}: {sampler.num_sim_scenarios} draws × {num_virtual_exps} experiments, "
        f"estimated expected RMSE {result.objective_values[0]:.6f}"
    )
    save_dual_blob_layout(result, array, output_path / f"{name}.png", title=name)
