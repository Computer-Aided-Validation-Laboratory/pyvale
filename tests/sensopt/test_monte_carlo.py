# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Monte Carlo placement tests with analytic fields."""

import runpy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import pyvale.dataio as io
import pyvale.sensoropt as opt
import pyvale.sensorsim as sens
from pyvale.verif import pointsensscalar


def test_named_reduction_order_and_weights() -> None:
    losses = np.array([[2.0, 0.0], [0.0, 4.0]])
    scenarios = ("left", "right")
    worst = opt.EvalReduction(
        axis=opt.EEvalAxis.VIRTUAL_EXP, method=opt.EEvalReduce.MAX
    )
    expected = opt.EvalReduction(
        axis=opt.EEvalAxis.SIM_SCENARIO,
        method=opt.EEvalReduce.MEAN,
        weights={"right": 3.0, "left": 1.0},
    )
    for steps, score in (((worst, expected), 3.5), ((expected, worst), 3.0)):
        np.testing.assert_allclose(
            opt.EvalStrat(reductions=steps).reduce(losses, scenario_keys=scenarios),
            [[score]],
        )


@pytest.fixture
def constant_field_setup() -> tuple[io.SimData, sens.SensorsPoint, opt.DesignSpace]:
    plate = pointsensscalar.simdata_2d_analytic()
    sites = np.array([[0.25, 0.25, 0.0], [0.5, 0.5, 0.0]])
    array = pointsensscalar.sens_array_noerrs(
        plate, sens.SensorData(positions=sites), sens.EDim.TWOD
    )
    design = opt.DesignSpace(
        positions={"temperature": sites},
        position_axes={"temperature": opt.EDesignAxis.DISC},
    )
    return plate, array, design


def build_constant_case(plate: io.SimData, value: float) -> io.SimData:
    assert plate.coords is not None and plate.time is not None
    return replace(
        plate,
        node_vars={"temperature": np.full((len(plate.coords), len(plate.time)), value)},
    )


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (opt.EReconRef.SIM_SCENARIO, 0.0),
        (opt.EReconRef.MEAN_SCENARIO, 35.0 / 9.0),
        (opt.EReconRef.MEDIAN_SCENARIO, 8.0 / 3.0),
    ],
)
def test_automatic_reference_modes(
    constant_field_setup: tuple[io.SimData, sens.SensorsPoint, opt.DesignSpace],
    mode: opt.EReconRef,
    expected: float,
) -> None:
    plate, array, design = constant_field_setup
    cases = dict(
        zip(
            ("cold", "warm", "hot"),
            (
                build_constant_case(plate, 0.0),
                build_constant_case(plate, 2.0),
                build_constant_case(plate, 8.0),
            ),
            strict=True,
        )
    )
    # A weighted mean target of 17/3 gives mean absolute offsets of 35/9.
    weights = {"cold": 1.0, "warm": 1.0, "hot": 4.0}
    assert plate.coords is not None
    objective = opt.ObjFieldRecon(
        target_positions=plate.coords[:5],
        reference_field="temperature",
        reference_mode=mode,
        reference_weights=weights if mode is opt.EReconRef.MEAN_SCENARIO else None,
        sensor_keys=("temperature",),
        component_index=0,
    )
    result = opt.SensorOptEngine(
        sensor_arrays={"temperature": array},
        design_space=design,
        sim_scenarios=cases,
        objectives=(
            opt.ObjSpec(
                objective=objective,
                eval_strat=opt.EvalStrat(
                    reductions=(
                        opt.EvalReduction(
                            axis=opt.EEvalAxis.SIM_SCENARIO, method=opt.EEvalReduce.MEAN
                        ),
                    )
                ),
            ),
        ),
        optimiser=opt.OptExhaustive(num_sensors=1),
    ).optimise()
    assert result.success
    np.testing.assert_allclose(result.objective_values, [expected], atol=1e-12)


class TestPlateSampler(opt.ISimSampler):
    __test__ = False

    def __init__(self, plate: io.SimData) -> None:
        self.plate = plate
        self.num_sim_scenarios: int = 2
        self.draws: list[float] = []

    def sample(self, *, rng: np.random.Generator) -> tuple[io.SimData, ...]:
        temperatures = rng.uniform(0.0, 5.0, size=self.num_sim_scenarios).tolist()
        self.draws.extend(temperatures)
        return tuple(build_constant_case(self.plate, value) for value in temperatures)


@pytest.mark.parametrize(
    ("sampling", "num_virtual_exps"),
    [(opt.ESampling.NESTED, 3), (opt.ESampling.PAIRED, 1)],
)
def test_physics_is_sampled_once_per_search_and_seed_is_repeatable(
    constant_field_setup: tuple[io.SimData, sens.SensorsPoint, opt.DesignSpace],
    sampling: opt.ESampling,
    num_virtual_exps: int,
) -> None:
    plate, array, design = constant_field_setup
    sampler = TestPlateSampler(plate)
    array.set_error_chain([sens.ErrRandGen(sens.GenNormal(std=1.0, seed=99))])
    assert plate.coords is not None
    engine = opt.SensorOptEngine(
        sensor_arrays={"temperature": array},
        design_space=design,
        sim_scenarios=sampler,
        num_virtual_exps=num_virtual_exps,
        sampling=sampling,
        objectives=(
            opt.ObjSpec(
                objective=opt.ObjFieldRecon(
                    target_positions=plate.coords,
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
            ),
        ),
        optimiser=opt.OptExhaustive(num_sensors=1),
        seed=123,
    )
    first = engine.optimise()
    assert first.success and len(sampler.draws) == 2 and first.num_iterations == 2
    draws = tuple(sampler.draws)
    sampler.draws.clear()
    second = engine.optimise()
    assert tuple(sampler.draws) == draws
    np.testing.assert_array_equal(first.objective_values, second.objective_values)
    assert first.objective_values[0] > 0.0


def test_library_sampler_supports_weights_and_replacement(
    constant_field_setup: tuple[io.SimData, sens.SensorsPoint, opt.DesignSpace],
) -> None:
    plate, _, _ = constant_field_setup
    warm = build_constant_case(plate, 5.0)
    library = {"cold": plate, "warm": warm}
    sampler = opt.SimSamplerLibrary(
        library=library,
        num_sim_scenarios=3,
        weights={"cold": 0.0, "warm": 4.0},
    )
    assert all(case is warm for case in sampler.sample(rng=np.random.default_rng(123)))
    sampler.replace = False
    sampler.num_sim_scenarios = 2
    with pytest.raises(ValueError, match="Too few eligible"):
        sampler.sample(rng=np.random.default_rng(123))
    sampler.weights = None
    first = sampler.sample(rng=np.random.default_rng(123))
    second = sampler.sample(rng=np.random.default_rng(123))
    assert {id(case) for case in first} == {id(case) for case in library.values()}
    assert all(a is b for a, b in zip(first, second, strict=True))


def test_dual_blob_examples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    discrete = runpy.run_module("pyvale.examples.sensopt.ex08_discrete_sim_scenarios")
    results = discrete["results"]
    assert all(result.success for result in results.values())
    np.testing.assert_allclose(
        [result.objective_values[0] for result in results.values()],
        [2.189585, 2.180162, 1.897222, 1.781270],
        atol=1e-6,
    )
    for key in results:
        assert (discrete["output_path"] / f"{key}.png").stat().st_size > 0
    library_results = discrete["library_results"]
    assert all(result.success for result in library_results.values())
    np.testing.assert_allclose(
        [result.objective_values[0] for result in library_results.values()],
        [2.188426, 2.128307],
        atol=1e-6,
    )
    for key in library_results:
        assert (discrete["output_path"] / f"{key}.png").stat().st_size > 0
    sampled = runpy.run_module("pyvale.examples.sensopt.ex09_sampled_sim_scenarios")
    assert set(sampled["validation_scores"]) == set(opt.ESampling)
    np.testing.assert_allclose(
        [sampled["validation_scores"][mode] for mode in opt.ESampling],
        [2.417068, 2.455619],
        atol=1e-6,
    )
    for mode, result in sampled["results"].items():
        assert result.success
        assert np.isfinite(sampled["validation_scores"][mode])
        assert (sampled["output_path"] / f"{mode.name.lower()}.png").stat().st_size > 0
