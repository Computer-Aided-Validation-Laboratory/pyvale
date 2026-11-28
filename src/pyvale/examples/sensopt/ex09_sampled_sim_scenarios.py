# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""
Place sensors for randomly shifted peaks and troughs
====================================================

Compare nested and paired sampling at the same virtual experiment budget.
"""

from dataclasses import dataclass, replace
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


@dataclass(slots=True, kw_only=True)
class DualBlobSampler(opt.ISimSampler):
    """Shift both blob centres independently in X and Y."""

    mesh: io.SimData
    plate_lengths: tuple[float, float]
    num_sim_scenarios: int

    def sample(self, *, rng: np.random.Generator) -> tuple[io.SimData, ...]:
        shift_limits = 0.2 * np.asarray(self.plate_lengths)
        cases: list[io.SimData] = []
        for _ in range(self.num_sim_scenarios):
            shifts = rng.uniform(-shift_limits, shift_limits, size=(2, 2))
            cases.append(
                build_dual_blob_case(
                    self.mesh,
                    peak=(3.5 + float(shifts[0, 0]), 5.0 + float(shifts[0, 1])),
                    trough=(7.5 + float(shifts[1, 0]), 2.0 + float(shifts[1, 1])),
                )
            )
        return tuple(cases)


# %%
# 1. Build the nominal array, candidate sites and the scenario sampler
# -------------------------------------------------------------------

mesh, _ = scalar_linear_2d()
coords = cast(npt.NDArray[np.float64], mesh.coords)
plate_lengths: tuple[float, float] = (10.0, 7.5)
nominal: io.SimData = build_dual_blob_case(mesh, peak=(3.5, 5.0), trough=(7.5, 2.0))
sites: np.ndarray = sens.gen_pos_grid_inside(
    num_sensors=(6, 4, 1),
    x_lims=(0.0, plate_lengths[0]),
    y_lims=(0.0, plate_lengths[1]),
    z_lims=(0.0, 0.0),
)
array_field = sens.FieldScalar(
    nominal,
    comp_key="temperature",
    spatial_dims=sens.EDim.TWOD,
)
array: sens.SensorsPoint = sens.SensorsPoint(
    sensor_data=sens.SensorData(positions=sites[:3]),
    field=array_field,
    descriptor=sens.DescriptorFactory.temperature(),
)
array.set_error_chain([sens.ErrRandGen(sens.GenNormal(std=0.5, seed=123))])

# %%
# 2. Minimise expected reconstruction RMSE using the same measurement budget
# --------------------------------------------------------------------------

objective: opt.ObjSpec = opt.ObjSpec(
    objective=opt.ObjFieldRecon(
        target_positions=coords,
        reference_field="temperature",
        sensor_keys=("temperature",),
        component_index=0,
    ),
    eval_strat=opt.EvalStrat(
        reductions=(
            opt.EvalReduction(
                axis=opt.EEvalAxis.VIRTUAL_EXP, method=opt.EEvalReduce.MEAN
            ),
            opt.EvalReduction(
                axis=opt.EEvalAxis.SIM_SCENARIO, method=opt.EEvalReduce.MEAN
            ),
        )
    ),
)
# Paired sampling has one virtual experiment per scenario, so only the
# simulation scenario axis needs a reduction.
paired_objective: opt.ObjSpec = replace(
    objective,
    eval_strat=opt.EvalStrat(
        reductions=(
            opt.EvalReduction(
                axis=opt.EEvalAxis.SIM_SCENARIO, method=opt.EEvalReduce.MEAN
            ),
        )
    ),
)
design_space: opt.DesignSpace = opt.DesignSpace(
    positions={"temperature": sites},
    position_axes={"temperature": opt.EDesignAxis.DISC},
)
budgets: dict[opt.ESampling, tuple[int, int]] = {
    opt.ESampling.NESTED: (4, 3),
    opt.ESampling.PAIRED: (12, 1),
}
output_path: Path = Path.cwd() / "pyvale-output" / "sensopt_ex09_sampled_sim_scenarios"
output_path.mkdir(parents=True, exist_ok=True)

results: dict[opt.ESampling, opt.OptResult] = {}
for sampling, (num_sim_scenarios, num_virtual_exps) in budgets.items():
    sampler: DualBlobSampler = DualBlobSampler(
        mesh=mesh, plate_lengths=plate_lengths, num_sim_scenarios=num_sim_scenarios
    )
    result: opt.OptResult = opt.SensorOptEngine(
        sensor_arrays={"temperature": array},
        design_space=design_space,
        sim_scenarios=sampler,
        num_virtual_exps=num_virtual_exps,
        sampling=sampling,
        objectives=(
            paired_objective if sampling is opt.ESampling.PAIRED else objective,
        ),
        optimiser=opt.OptGreedyForward(num_sensors=3),
        seed=123,
    ).optimise()
    results[sampling] = result
    print(
        f"{sampling.name}: {num_sim_scenarios} scenarios × "
        f"{num_virtual_exps} experiments, "
        f"estimated expected RMSE {result.objective_values[0]:.6f}"
    )
    save_dual_blob_layout(
        result, array, output_path / f"{sampling.name.lower()}.png", title=sampling.name
    )

# %%
# 3. Compare the selected layouts on the same independent scenarios
# ----------------------------------------------------------------

# Reuse the same validation seed for both layouts so the held-out scenarios
# and sensor draws are identical. The validation runs perform no greedy additions.
validation_scores: dict[opt.ESampling, float] = {}
validation_sampler: DualBlobSampler = DualBlobSampler(
    mesh=mesh, plate_lengths=plate_lengths, num_sim_scenarios=24
)
site_lookup = dict(zip(map(tuple, sites), range(len(sites)), strict=True))
for sampling, result in results.items():
    positions = cast(
        npt.NDArray[np.float64], result.sensor_data["temperature"].positions
    )
    selected_indices: tuple[int, ...] = tuple(
        site_lookup[tuple(position)] for position in positions
    )
    validation: opt.OptResult = opt.SensorOptEngine(
        sensor_arrays={"temperature": array},
        design_space=design_space,
        sim_scenarios=validation_sampler,
        sampling=opt.ESampling.PAIRED,
        objectives=(paired_objective,),
        optimiser=opt.OptGreedyForward(
            num_sensors=len(selected_indices),
            start=opt.OptStartLayout(sites={"temperature": selected_indices}),
        ),
        seed=321,
    ).optimise()
    validation_scores[sampling] = float(validation.objective_values[0])
    print(f"{sampling.name}: held-out RMSE {validation_scores[sampling]:.6f}")
