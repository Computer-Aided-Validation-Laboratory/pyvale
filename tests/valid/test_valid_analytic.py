# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Analytic metric answers for deterministic and odd/even aleatory sample counts.

Interleaved samples are sim = [0, 2, ..., 2(n-1)] and exp = [1, 3, ..., 2m-1].
The table contains exact rational results, not outputs of another metric library:

* AVM sums unit-width rectangles |F_sim - F_exp| between integer supports.
* KS takes the largest of those exact empirical probability differences.
* CvM uses pooled integer ranks: U/(nm(n+m)) - (4nm-1)/(6(n+m)).
* MAVM at alpha=1/2 has zero confidence shift. Its signed areas integrate
  Q_exp(p)-Q_sim(p) on the rational grid {i/n} union {j/m}.

For example n=2, m=1 gives CDF differences 1/2 and -1/2: AVM=1, KS=1/2,
MAVM d+=d-=1/2. CvM is undefined whenever either sample count is below two.
Equal counts necessarily have matching parity; mixed parity applies to both
orientations of unequal counts. Paired metrics reject every unequal count.
"""

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pytest
from scipy import stats

import pyvale.sensorsim as sens
import pyvale.validation as vl
from pyvale.dataio.simdata import SimData
from pyvale.validation.metrics import (
    EMAVMDuplicates,
    calc_avm_1d,
    calc_mavm_1d,
)


@dataclass(slots=True)
class AnalyticCase:
    sim_count: int
    exp_count: int
    avm: float
    ks: float
    cvm: float
    d_plus: float
    d_minus: float


# Counts: deterministic/deterministic; each deterministic/aleatory orientation;
# equal even/even and odd/odd; then all four parities in each unequal direction.
ANALYTIC_CASES = (
    AnalyticCase(
        sim_count=1,
        exp_count=1,
        avm=1,
        ks=1,
        cvm=np.nan,
        d_plus=1,
        d_minus=0,
    ),
    AnalyticCase(
        sim_count=1,
        exp_count=2,
        avm=2,
        ks=1,
        cvm=np.nan,
        d_plus=2,
        d_minus=0,
    ),
    AnalyticCase(
        sim_count=1,
        exp_count=3,
        avm=3,
        ks=1,
        cvm=np.nan,
        d_plus=3,
        d_minus=0,
    ),
    AnalyticCase(
        sim_count=2,
        exp_count=1,
        avm=1,
        ks=1 / 2,
        cvm=np.nan,
        d_plus=1 / 2,
        d_minus=1 / 2,
    ),
    AnalyticCase(
        sim_count=3,
        exp_count=1,
        avm=5 / 3,
        ks=2 / 3,
        cvm=np.nan,
        d_plus=1 / 3,
        d_minus=4 / 3,
    ),
    AnalyticCase(
        sim_count=2,
        exp_count=2,
        avm=1,
        ks=1 / 2,
        cvm=1 / 8,
        d_plus=1,
        d_minus=0,
    ),
    AnalyticCase(
        sim_count=3,
        exp_count=3,
        avm=1,
        ks=1 / 3,
        cvm=1 / 12,
        d_plus=1,
        d_minus=0,
    ),
    AnalyticCase(
        sim_count=4,
        exp_count=2,
        avm=3 / 2,
        ks=1 / 2,
        cvm=7 / 72,
        d_plus=1 / 4,
        d_minus=5 / 4,
    ),
    AnalyticCase(
        sim_count=4,
        exp_count=3,
        avm=1,
        ks=1 / 4,
        cvm=1 / 21,
        d_plus=1 / 2,
        d_minus=1 / 2,
    ),
    AnalyticCase(
        sim_count=5,
        exp_count=2,
        avm=12 / 5,
        ks=3 / 5,
        cvm=1 / 7,
        d_plus=1 / 5,
        d_minus=11 / 5,
    ),
    AnalyticCase(
        sim_count=5,
        exp_count=3,
        avm=23 / 15,
        ks=2 / 5,
        cvm=19 / 240,
        d_plus=4 / 15,
        d_minus=19 / 15,
    ),
    AnalyticCase(
        sim_count=2,
        exp_count=4,
        avm=3,
        ks=3 / 4,
        cvm=19 / 72,
        d_plus=3,
        d_minus=0,
    ),
    AnalyticCase(
        sim_count=3,
        exp_count=4,
        avm=2,
        ks=1 / 2,
        cvm=13 / 84,
        d_plus=2,
        d_minus=0,
    ),
    AnalyticCase(
        sim_count=2,
        exp_count=5,
        avm=4,
        ks=4 / 5,
        cvm=11 / 35,
        d_plus=4,
        d_minus=0,
    ),
    AnalyticCase(
        sim_count=3,
        exp_count=5,
        avm=3,
        ks=3 / 5,
        cvm=11 / 48,
        d_plus=3,
        d_minus=0,
    ),
)
CASE_IDS = tuple(
    f"sim-{test_case.sim_count}_exp-{test_case.exp_count}"
    for test_case in ANALYTIC_CASES
)


@pytest.mark.parametrize("test_case", ANALYTIC_CASES, ids=CASE_IDS)
def test_interleaved_distribution_metrics_have_exact_answers(
    test_case: AnalyticCase,
) -> None:
    sim = 2.0 * np.arange(test_case.sim_count)
    exp = 1.0 + 2.0 * np.arange(test_case.exp_count)
    assert vl.calc_avm_1d(sim, exp) == pytest.approx(test_case.avm, rel=0, abs=1e-12)
    assert vl.calc_ks_1d(sim, exp) == pytest.approx(test_case.ks, rel=0, abs=1e-12)
    if np.isnan(test_case.cvm):
        with pytest.warns(RuntimeWarning, match="sample arguments is too small"):
            cvm = vl.calc_cvm_1d(sim, exp)
        assert np.isnan(cvm)
    else:
        cvm = vl.calc_cvm_1d(sim, exp)
        assert cvm == pytest.approx(test_case.cvm, rel=0, abs=1e-12)
    mavm = vl.calc_mavm_1d(sim, exp, alpha=0.5)
    assert mavm.d_plus == pytest.approx(test_case.d_plus, rel=0, abs=1e-12)
    assert mavm.d_minus == pytest.approx(test_case.d_minus, rel=0, abs=1e-12)
    np.testing.assert_allclose(mavm.exp_conf_lower, exp, rtol=0, atol=1e-12)
    np.testing.assert_allclose(mavm.exp_conf_upper, exp, rtol=0, atol=1e-12)


# Fixed 95th percentiles of Student's t for df=1..4. df=0 has no confidence
# shift for a singleton. These reference constants are independent of SciPy's
# quantile evaluation in the implementation. The population variance of the
# consecutive exp supports 1,3,...,2m-1 is (m^2-1)/3.
STUDENT_T_95 = (
    0.0,
    6.313751514800932,
    2.919985580355516,
    2.3533634348018264,
    2.131846786326649,
)


@pytest.mark.parametrize("test_case", ANALYTIC_CASES, ids=CASE_IDS)
@pytest.mark.parametrize("sim_shift", (-20.0, 20.0), ids=("sim-below", "sim-above"))
def test_separated_metrics_and_mavm_confidence_have_closed_form_answers(
    test_case: AnalyticCase,
    sim_shift: float,
) -> None:
    sim = sim_shift + 2.0 * np.arange(test_case.sim_count)
    exp = 1.0 + 2.0 * np.arange(test_case.exp_count)
    # Entire distributions, including confidence bounds, are strictly separated.
    mean_gap = test_case.exp_count - (sim_shift + test_case.sim_count - 1)
    confidence_shift = STUDENT_T_95[test_case.exp_count - 1] * sqrt(
        (test_case.exp_count**2 - 1) / (3 * test_case.exp_count)
    )
    assert vl.calc_avm_1d(sim, exp) == pytest.approx(abs(mean_gap), rel=0, abs=1e-12)
    assert vl.calc_ks_1d(sim, exp) == pytest.approx(1.0, rel=0, abs=1e-12)
    if min(test_case.sim_count, test_case.exp_count) == 1:
        with pytest.warns(RuntimeWarning, match="sample arguments is too small"):
            cvm = vl.calc_cvm_1d(sim, exp)
        assert np.isnan(cvm)
    else:
        cvm = vl.calc_cvm_1d(sim, exp)
        # Disjoint pooled ranks reduce the rank statistic to this expression.
        expected_cvm = (2 * test_case.sim_count * test_case.exp_count + 1) / (
            6 * (test_case.sim_count + test_case.exp_count)
        )
        assert cvm == pytest.approx(expected_cvm, rel=0, abs=1e-12)
    mavm = vl.calc_mavm_1d(sim, exp)
    expected_d_plus = mean_gap + confidence_shift if sim_shift < 0 else 0.0
    expected_d_minus = -mean_gap + confidence_shift if sim_shift > 0 else 0.0
    assert mavm.d_plus == pytest.approx(expected_d_plus, rel=0, abs=1e-10)
    assert mavm.d_minus == pytest.approx(expected_d_minus, rel=0, abs=1e-10)
    np.testing.assert_allclose(
        mavm.exp_conf_lower, exp - confidence_shift, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        mavm.exp_conf_upper, exp + confidence_shift, rtol=0, atol=1e-10
    )


@pytest.mark.parametrize("test_case", ANALYTIC_CASES, ids=CASE_IDS)
def test_paired_metrics_use_pairs_and_reject_unequal_counts(
    test_case: AnalyticCase,
) -> None:
    sim = 2.0 * np.arange(test_case.sim_count)
    exp = 1.0 + 3.0 * np.arange(test_case.exp_count)
    if test_case.sim_count != test_case.exp_count:
        with pytest.raises(ValueError, match="paired"):
            vl.MetricRMSE().calc(sim, exp)
        with pytest.raises(ValueError, match="paired"):
            vl.MetricRelErr().calc(sim, exp)
        with pytest.raises(ValueError, match="paired"):
            vl.calc_deterministic_metrics_1d(sim, exp)
        return
    # Residual magnitudes are 1,2,...,n, with known first and second moments.
    count = test_case.sim_count
    mean_absolute_error = (count + 1) / 2
    mean_square_error = (count + 1) * (2 * count + 1) / 6
    exp_variance = 3 * (count**2 - 1) / 4
    expected_relative = mean_absolute_error / (3 * count - 2)
    errors = vl.calc_deterministic_metrics_1d(sim, exp)
    assert errors.abs_err == pytest.approx(mean_absolute_error, rel=0, abs=1e-12)
    assert errors.rmse == pytest.approx(sqrt(mean_square_error), rel=0, abs=1e-12)
    assert errors.rel_err == pytest.approx(expected_relative, rel=0, abs=1e-12)
    if count == 1:
        assert np.isnan(errors.nmse)
    else:
        assert errors.nmse == pytest.approx(
            mean_square_error / exp_variance, rel=0, abs=1e-12
        )
    assert vl.MetricRMSE().calc(sim, exp).value == pytest.approx(
        sqrt(mean_square_error), rel=0, abs=1e-12
    )
    for normalisation, denominator in (
        (vl.ERelErrNorm.EXP_MAX, 3 * count - 2),
        (vl.ERelErrNorm.SIM_MAX, 2 * (count - 1)),
        (vl.ERelErrNorm.CONSTANT, 10.0),
    ):
        constant = 10.0 if normalisation is vl.ERelErrNorm.CONSTANT else None
        metric = vl.MetricRelErr(normalisation=normalisation, constant=constant)
        relative = metric.calc(sim, exp).value
        errors = vl.calc_deterministic_metrics_1d(
            sim,
            exp,
            normalisation=normalisation,
            constant=constant,
        )
        if denominator == 0:
            assert np.isnan(relative)
            assert np.isnan(errors.rel_err)
        else:
            expected = mean_absolute_error / denominator
            assert relative == pytest.approx(expected, rel=0, abs=1e-12)
            assert errors.rel_err == pytest.approx(expected, rel=0, abs=1e-12)


@pytest.mark.parametrize("test_case", ANALYTIC_CASES, ids=CASE_IDS)
def test_engine_outputs_match_analytic_sample_count_cases(
    test_case: AnalyticCase,
) -> None:
    sim = 2.0 * np.arange(test_case.sim_count)
    exp = 1.0 + 2.0 * np.arange(test_case.exp_count)
    axes = (
        vl.EValidAxis.POINT,
        vl.EValidAxis.COMPONENT,
        vl.EValidAxis.TIME,
        vl.EValidAxis.ALEATORY,
    )
    valid_case = vl.ValidCase(
        {
            "response": vl.ValidObservable(
                components=("response",), geometry=vl.EValidGeometry.GLOBAL
            )
        },
        {"response": vl.ValidArray(sim.reshape(1, 1, 1, -1), axes)},
        {"response": vl.ValidArray(exp.reshape(1, 1, 1, -1), axes)},
    )
    metrics: list[vl.IValidMetric] = [
        vl.MetricAVM(),
        vl.MetricKS(),
        vl.MetricCVM(),
        vl.MetricMAVM(alpha=0.5),
    ]
    if np.isnan(test_case.cvm):
        with pytest.warns(RuntimeWarning, match="sample arguments is too small"):
            result = vl.ValidEngine(metrics).run(valid_case)
    else:
        result = vl.ValidEngine(metrics).run(valid_case)
    expected_outputs = (
        (vl.EMetricType.AVM, vl.EMetricOut.DISTANCE, test_case.avm),
        (vl.EMetricType.KS, vl.EMetricOut.STATISTIC, test_case.ks),
        (vl.EMetricType.CVM, vl.EMetricOut.STATISTIC, test_case.cvm),
        (vl.EMetricType.MAVM, vl.EMetricOut.D_PLUS, test_case.d_plus),
        (vl.EMetricType.MAVM, vl.EMetricOut.D_MINUS, test_case.d_minus),
    )
    for metric_type, output, expected in expected_outputs:
        field = result.metrics["response"][metric_type]
        assert field.axes == (
            vl.EValidResAxis.POINT,
            vl.EValidResAxis.COMPONENT,
            vl.EValidResAxis.TIME,
        )
        assert field.metric_vals[output].shape == (1, 1, 1)
        actual = field.metric_vals[output][0, 0, 0]
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert actual == pytest.approx(expected, rel=0, abs=1e-12)
    if test_case.sim_count != test_case.exp_count:
        for metric in (vl.MetricRMSE(), vl.MetricRelErr()):
            with pytest.raises(ValueError, match="equal aleatory counts"):
                vl.ValidEngine([metric]).run(valid_case)


# Epistemic toolbox references: all probabilities and distances below are
# hand-derived. No production profiling function supplies an expected answer.


def create_epistemic_case(sim: np.ndarray, exp: np.ndarray) -> vl.ValidCase:
    """Create a scalar physical cell with (transferable, nuisance, aleatory) axes."""
    axes = (
        vl.EValidAxis.EPISTEMIC_TRANS,
        vl.EValidAxis.EPISTEMIC_NUIS,
        vl.EValidAxis.ALEATORY,
    )
    observable = vl.ValidObservable(
        components=("length",), geometry=vl.EValidGeometry.GLOBAL, phys_units="m"
    )
    return vl.ValidCase(
        {"length": observable},
        {"length": vl.ValidArray(sim, axes)},
        {"length": vl.ValidArray(exp, axes)},
    )


@pytest.mark.parametrize(
    "reduction, probability, lower, upper",
    [
        (vl.EECDFReduce.KEEP, None, 1.0, 90.0),
        (vl.EECDFReduce.MEAN, None, 4.0, 33.0),
        (vl.EECDFReduce.MEDIAN, None, 3.0, 8.0),
        (vl.EECDFReduce.QUANTILE, 0.0, 1.0, 4.0),
        (vl.EECDFReduce.QUANTILE, 0.25, 2.5, 4.5),
        (vl.EECDFReduce.QUANTILE, 0.5, 3.0, 8.0),
        (vl.EECDFReduce.QUANTILE, 0.75, 5.0, 49.0),
        (vl.EECDFReduce.QUANTILE, 1.0, 6.0, 90.0),
    ],
)
def test_epis_bounds_nuisance_treatments_have_analytic_quantiles(
    reduction: vl.EECDFReduce,
    probability: float | None,
    lower: float,
    upper: float,
) -> None:
    # Nuisance shifts per transferable case: [1,90,8], [2,3,7], [4,6,5].
    shifts = np.array([[1.0, 90.0, 8.0], [2.0, 3.0, 7.0], [4.0, 6.0, 5.0]])
    quantity = vl.ValidArray(
        shifts[..., None] + np.array([0.0, 2.0]),
        (
            vl.EValidAxis.EPISTEMIC_TRANS,
            vl.EValidAxis.EPISTEMIC_NUIS,
            vl.EValidAxis.ALEATORY,
        ),
    )
    bounds = vl.find_ecdf_epis_bounds(
        quantity, nuis_reduce=reduction, nuis_quantile=probability
    )
    np.testing.assert_allclose(bounds.view.probs, [0.5, 1.0])
    np.testing.assert_allclose(bounds.lower, [lower, lower + 2], rtol=0, atol=1e-12)
    np.testing.assert_allclose(bounds.upper, [upper, upper + 2], rtol=0, atol=1e-12)
    if reduction is vl.EECDFReduce.KEEP:
        np.testing.assert_array_equal(bounds.lower_indices[:, :2], [[0, 0], [0, 0]])
        np.testing.assert_array_equal(bounds.upper_indices[:, :2], [[0, 1], [0, 1]])
    else:
        assert np.all(bounds.lower_indices[:, 1] == -1)
        assert np.all(bounds.upper_indices[:, 1] == -1)
    conditioned = vl.find_ecdf_epis_bounds(quantity, nuis_idxs=(1,))
    np.testing.assert_array_equal(conditioned.lower, [3.0, 5.0])
    np.testing.assert_array_equal(conditioned.upper, [90.0, 92.0])


def test_avm_bounds_preserve_unequal_empirical_probability_mass() -> None:
    sim = vl.ValidArray(
        np.array([[0.0, 10.0, np.nan], [2.0, 3.0, 20.0]]),
        (vl.EValidAxis.EPISTEMIC_TRANS, vl.EValidAxis.ALEATORY),
    )
    exp = vl.ValidArray(np.array([5.0]), (vl.EValidAxis.ALEATORY,))
    sim_bounds = vl.find_ecdf_epis_bounds(sim)
    exp_bounds = vl.find_ecdf_epis_bounds(exp)
    # The grid has interval weights [1/3,1/6,1/6,1/3], not four equal masses.
    np.testing.assert_allclose(sim_bounds.view.probs, [1 / 3, 0.5, 2 / 3, 1.0])
    np.testing.assert_array_equal(sim_bounds.lower, [0.0, 0.0, 3.0, 10.0])
    np.testing.assert_array_equal(sim_bounds.upper, [2.0, 3.0, 10.0, 20.0])
    supports = vl.calc_avm_over_epis_bounds(sim_bounds, exp_bounds)
    assert supports.metric_outputs is not None
    np.testing.assert_allclose(
        supports.metric_outputs[vl.EMetricOut.DISTANCE],
        [[9 / 2, 9 / 2], [43 / 6, 43 / 6]],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        supports.lower, [-31 / 6, -9 / 2, -3 / 2, 11 / 2], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        supports.upper, [55 / 6, 61 / 6, 103 / 6, 163 / 6], rtol=0, atol=1e-12
    )


def test_mavm_over_bounds_checks_both_sim_anchors_in_each_direction() -> None:
    sim = vl.ValidArray(
        np.array([[0.0, 100.0], [40.0, 100.0]]),
        (vl.EValidAxis.EPISTEMIC_TRANS, vl.EValidAxis.ALEATORY),
    )
    exp = vl.ValidArray(np.array([100.0]), (vl.EValidAxis.ALEATORY,))
    supports = vl.calc_mavm_over_epis_bounds(
        vl.find_ecdf_epis_bounds(sim), vl.find_ecdf_epis_bounds(exp)
    )
    # d+ is 50 on the lower boundary and 30 on the upper boundary. Thus the
    # upper boundary controls p=.5 (70), but the lower controls p=1 (150).
    np.testing.assert_allclose(supports.upper, [70.0, 150.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(supports.lower, [0.0, 100.0], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(supports.upper_indices[:, 0], [1, 0])
    assert supports.metric_outputs is not None
    np.testing.assert_allclose(
        supports.metric_outputs[vl.EMetricOut.D_PLUS],
        [[50.0, 50.0], [30.0, 30.0]],
        rtol=0,
        atol=1e-12,
    )


def test_mavm_confidence_is_constructed_before_enveloping_original_exp_cases() -> None:
    sim = vl.ValidArray(np.array([-20.0]), (vl.EValidAxis.ALEATORY,))
    exp = vl.ValidArray(
        np.array([[0.0, 10.0], [1.0, 5.0]]),
        (vl.EValidAxis.EPISTEMIC_TRANS, vl.EValidAxis.ALEATORY),
    )
    # The second curve owns the nominal upper quantile at p=0.5. With
    # t(0.75, 1) = 1, the first curve shifts by 5/sqrt(2) and takes over.
    shift_wide = 5 / sqrt(2)
    shift_narrow = 2 / sqrt(2)
    exp_bounds = vl.find_ecdf_epis_bounds(exp)
    np.testing.assert_array_equal(exp_bounds.upper, [1.0, 10.0])
    supports = vl.calc_mavm_over_epis_bounds(
        vl.find_ecdf_epis_bounds(sim), exp_bounds, alpha=0.25
    )
    np.testing.assert_allclose(
        supports.upper, [5 + shift_wide, 5 + shift_wide], rtol=0, atol=1e-10
    )
    assert supports.metric_outputs is not None
    np.testing.assert_allclose(
        supports.metric_outputs[vl.EMetricOut.D_PLUS][0],
        [22.5 - (shift_wide + shift_narrow) / 2, 25 + shift_wide],
        rtol=0,
        atol=1e-10,
    )


def test_bounds_route_and_original_scenario_route_are_distinct() -> None:
    test_case = create_epistemic_case(
        np.array([[[0.0, 100.0], [40.0, 60.0]]]), np.array([[[100.0]]])
    )
    result = vl.ValidEngine([vl.MetricMAVM(alpha=0.5)]).run(test_case)
    direct = vl.find_metric_epis_supports(
        test_case, result, "length", vl.EMetricType.MAVM
    )
    bounded = vl.calc_mavm_over_epis_bounds(
        vl.find_ecdf_epis_bounds(test_case.sim["length"]),
        vl.find_ecdf_epis_bounds(test_case.exp["length"]),
        alpha=0.5,
    )
    np.testing.assert_allclose(direct.upper[0, 0, 0], [90.0, 150.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(bounded.upper, [70.0, 130.0], rtol=0, atol=1e-12)


def test_nuisance_profiles_preserve_transferable_axes_and_original_arg_indices() -> (
    None
):
    sim = np.array([[0.0, 10.0], [2.0, 12.0]])[..., None] + np.array([0.0, 2.0])
    exp = np.array([[1.0, 4.0], [3.0, 6.0]])[..., None] + np.array([0.0, 2.0])
    test_case = create_epistemic_case(sim, exp)
    result = vl.ValidEngine([vl.MetricAVM(), vl.MetricMAVM(alpha=0.5)]).run(
        test_case
    )
    extrema = vl.profile_epis_nuisance_extrema(
        result, "length", vl.EMetricType.AVM
    )
    assert extrema.axes == (
        vl.EValidResAxis.POINT,
        vl.EValidResAxis.COMPONENT,
        vl.EValidResAxis.TIME,
        vl.EValidResAxis.SIM_TRANS,
        vl.EValidResAxis.EXP_TRANS,
    )
    assert extrema.reduced_axes == (
        vl.EValidResAxis.SIM_NUIS,
        vl.EValidResAxis.EXP_NUIS,
    )
    np.testing.assert_allclose(
        extrema.minimum[vl.EMetricOut.DISTANCE][0, 0, 0], [[1.0, 3.0], [1.0, 1.0]]
    )
    np.testing.assert_allclose(
        extrema.maximum[vl.EMetricOut.DISTANCE][0, 0, 0],
        [[9.0, 7.0], [11.0, 9.0]],
    )
    np.testing.assert_array_equal(
        extrema.maximum_indices[vl.EMetricOut.DISTANCE][0, 0, 0],
        np.tile([1, 0], (2, 2, 1)),
    )
    mavm_extrema = vl.profile_epis_nuisance_extrema(
        result, "length", vl.EMetricType.MAVM
    )
    np.testing.assert_allclose(
        mavm_extrema.maximum[vl.EMetricOut.D_PLUS][0, 0, 0],
        [[4.0, 6.0], [2.0, 4.0]],
    )
    np.testing.assert_allclose(
        mavm_extrema.maximum[vl.EMetricOut.D_MINUS][0, 0, 0],
        [[9.0, 7.0], [11.0, 9.0]],
    )
    supports = vl.profile_epis_nuisance_supports(
        test_case, result, "length", vl.EMetricType.MAVM
    )
    assert supports.axes == extrema.axes
    np.testing.assert_allclose(
        supports.lower[0, 0, 0, :, :, 0], [[0.0, 0.0], [1.0, 2.0]]
    )
    np.testing.assert_allclose(
        supports.upper[0, 0, 0, :, :, 1], [[12.0, 12.0], [14.0, 14.0]]
    )
    # For Ts=0 the largest d+=4 is on an interior curve. It adds no extension
    # beyond the original upper nuisance envelope [10,12].
    np.testing.assert_allclose(
        supports.upper[0, 0, 0, 0, 0], supports.sim_upper[0, 0, 0, 0, 0]
    )


@pytest.mark.parametrize(
    "metric",
    [
        vl.MetricAVM(),
        vl.MetricKS(),
        vl.MetricCVM(),
        vl.MetricRMSE(),
        vl.MetricRelErr(),
    ],
)
def test_other_metric_nuisance_extrema_match_hand_calculated_shift_cases(
    metric: vl.IValidMetric,
) -> None:
    # Same-sized paired two-point samples: residual shifts are 0 and 2.
    test_case = create_epistemic_case(
        np.array([[[0.0, 2.0], [2.0, 4.0]]]), np.array([[[0.0, 2.0]]])
    )
    result = vl.ValidEngine([metric]).run(test_case)
    profile = vl.profile_epis_nuisance_extrema(result, "length", metric.get_kind())
    output = metric.get_outputs()[0]
    expected_max = {
        vl.EMetricType.AVM: 2.0,
        vl.EMetricType.KS: 0.5,
        vl.EMetricType.CVM: 3 / 16,
        vl.EMetricType.RMSE: 2.0,
        vl.EMetricType.REL_ERR: 1.0,
    }[metric.get_kind()]
    np.testing.assert_allclose(profile.minimum[output], 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        profile.maximum[output], expected_max, rtol=0, atol=1e-12
    )
    if metric.get_kind() not in (vl.EMetricType.AVM, vl.EMetricType.MAVM):
        with pytest.raises(ValueError, match="only for AVM and MAVM"):
            vl.profile_epis_nuisance_supports(
                test_case, result, "length", metric.get_kind()
            )


def test_nuisance_profiles_keep_missing_cells_and_offset_selections() -> None:
    test_case = create_epistemic_case(
        np.array([[[0.0, 2.0], [np.nan, np.nan]]]), np.array([[[3.0, 5.0]]])
    )
    test_case.exp["length"].offsets = vl.EpistemicOffsets(1.0, 2.0)
    result = vl.ValidEngine([vl.MetricMAVM(alpha=0.5)]).run(test_case)
    extrema = vl.profile_epis_nuisance_extrema(
        result, "length", vl.EMetricType.MAVM
    )
    assert vl.EValidResAxis.EXP_OFFSET in extrema.axes
    np.testing.assert_allclose(
        extrema.maximum[vl.EMetricOut.D_PLUS][0, 0, 0], [3.0, 2.0, 5.0]
    )
    supports = vl.profile_epis_nuisance_supports(
        test_case,
        result,
        "length",
        vl.EMetricType.MAVM,
        selectors={vl.EValidResAxis.EXP_OFFSET: vl.EEpisOffset.UPPER},
    )
    np.testing.assert_allclose(supports.upper[0, 0, 0], [5.0, 7.0])
    # Explicitly selecting the missing original nuisance case leaves missing bounds.
    missing = vl.find_metric_epis_supports(
        test_case,
        result,
        "length",
        vl.EMetricType.MAVM,
        selectors={vl.EValidResAxis.SIM_NUIS: 1},
    )
    assert np.all(np.isnan(missing.lower)) and np.all(missing.lower_indices == -1)
    missing_values = vl.find_metric_epis_extrema(
        result,
        "length",
        vl.EMetricType.MAVM,
        selectors={vl.EValidResAxis.SIM_NUIS: 1},
    )
    assert np.all(np.isnan(missing_values.maximum[vl.EMetricOut.D_PLUS]))


def test_ranked_nuisance_cases_are_coherent_and_use_explicit_width_scores() -> None:
    probabilities = np.array([0.25, 1.0])
    shifted_width = np.array([[3.0, 10.0], [4.0, 8.0]])
    original_width = np.array([[1.0, 8.0], [1.0, 2.0]])
    upper = np.repeat(shifted_width[..., None], 2, axis=-1)
    upper[0, 0] = [1.5, 3.5]  # Weighted width is 3; the unweighted mean is 2.5.
    sim_upper = np.repeat(original_width[..., None], 2, axis=-1)
    origin = np.empty((*upper.shape, 0), dtype=int)
    supports = vl.ECDFMetricBounds(
        (vl.EValidResAxis.SIM_NUIS, vl.EValidResAxis.EXP_NUIS),
        probabilities,
        np.zeros_like(upper),
        upper,
        np.zeros_like(upper),
        sim_upper,
        (),
        origin,
        origin,
        {},
    )
    total = vl.rank_epis_nuisance_cases(supports)
    added = vl.rank_epis_nuisance_cases(
        supports, score=vl.EECDFNuisanceScore.ADDED_WIDTH
    )
    np.testing.assert_array_equal(total.cases, [[0, 0], [0, 1], [1, 0], [1, 1]])
    np.testing.assert_allclose(total.scores, [3.0, 10.0, 4.0, 8.0])
    np.testing.assert_array_equal(total.order, [0, 2, 3, 1])
    np.testing.assert_allclose(added.scores, [2.0, 2.0, 3.0, 6.0])
    np.testing.assert_array_equal(added.order, [0, 1, 2, 3])  # Stable tie order.
    assert total.valid_counts == 4
    # Minimum/median/maximum-ranked cases refer to rows in cases, not a curve
    # assembled from different cases at different cumulative probabilities.
    np.testing.assert_array_equal(
        total.cases[total.order[[0, 1, -1]]], [[0, 0], [1, 0], [0, 1]]
    )
    supports.upper[1, 1] = np.nan
    missing = vl.rank_epis_nuisance_cases(supports)
    assert missing.valid_counts == 3
    np.testing.assert_array_equal(missing.order, [0, 2, 1, -1])


@pytest.mark.parametrize("test_case", ANALYTIC_CASES, ids=CASE_IDS)
def test_bounds_and_profiles_cover_the_full_analytic_sample_count_matrix(
    test_case: AnalyticCase,
) -> None:
    sim = vl.ValidArray(
        2.0 * np.arange(test_case.sim_count), (vl.EValidAxis.ALEATORY,)
    )
    exp = vl.ValidArray(
        1.0 + 2.0 * np.arange(test_case.exp_count), (vl.EValidAxis.ALEATORY,)
    )
    sim_bounds = vl.find_ecdf_epis_bounds(sim)
    exp_bounds = vl.find_ecdf_epis_bounds(exp)
    avm = vl.calc_avm_over_epis_bounds(sim_bounds, exp_bounds)
    mavm = vl.calc_mavm_over_epis_bounds(sim_bounds, exp_bounds, alpha=0.5)
    assert avm.metric_outputs is not None and mavm.metric_outputs is not None
    np.testing.assert_allclose(
        avm.metric_outputs[vl.EMetricOut.DISTANCE], test_case.avm, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        mavm.metric_outputs[vl.EMetricOut.D_PLUS], test_case.d_plus, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        mavm.metric_outputs[vl.EMetricOut.D_MINUS],
        test_case.d_minus,
        rtol=0,
        atol=1e-12,
    )
    assert avm.lower[0] == pytest.approx(-test_case.avm, rel=0, abs=1e-12)
    assert avm.upper[-1] == pytest.approx(
        2 * (test_case.sim_count - 1) + test_case.avm, rel=0, abs=1e-12
    )
    assert mavm.lower[0] == pytest.approx(-test_case.d_minus, rel=0, abs=1e-12)
    assert mavm.upper[-1] == pytest.approx(
        2 * (test_case.sim_count - 1) + test_case.d_plus, rel=0, abs=1e-12
    )
    observable = vl.ValidObservable(
        components=("length",), geometry=vl.EValidGeometry.GLOBAL
    )
    valid_case = vl.ValidCase(
        {"length": observable}, {"length": sim}, {"length": exp}
    )
    result = vl.ValidEngine([vl.MetricAVM(), vl.MetricMAVM(alpha=0.5)]).run(
        valid_case
    )
    profile = vl.profile_epis_nuisance_extrema(
        result, "length", vl.EMetricType.AVM
    )
    assert profile.reduced_axes == ()  # Singleton/absent epistemic axes are valid.
    np.testing.assert_allclose(
        profile.minimum[vl.EMetricOut.DISTANCE], test_case.avm, rtol=0, atol=1e-12
    )
    supports = vl.find_metric_epis_supports(
        valid_case, result, "length", vl.EMetricType.MAVM
    )
    assert supports.lower[0, 0, 0, 0] == pytest.approx(
        -test_case.d_minus, rel=0, abs=1e-12
    )
    assert supports.upper[0, 0, 0, -1] == pytest.approx(
        2 * (test_case.sim_count - 1) + test_case.d_plus, rel=0, abs=1e-12
    )


def test_mavm_confidence_preserves_original_counts_before_nuisance_mean() -> None:
    sim = vl.ValidArray(np.array([0.0]), (vl.EValidAxis.ALEATORY,))
    exp = vl.ValidArray(
        np.array([[0.0, 2.0, np.nan], [10.0, 12.0, 14.0]]),
        (vl.EValidAxis.EPISTEMIC_NUIS, vl.EValidAxis.ALEATORY),
    )
    exp_bounds = vl.find_ecdf_epis_bounds(
        exp, nuis_reduce=vl.EECDFReduce.MEAN
    )
    # The mean quantile curve is [5,6,7,8] on intervals [1/3,1/6,1/6,1/3],
    # giving mean 6.5. Confidence shifts use the original counts two and three,
    # not four pseudo-observations reconstructed from this curve's vertices.
    confidence = (1 / sqrt(2) + 4 / (3 * sqrt(3))) / 2
    supports = vl.calc_mavm_over_epis_bounds(
        vl.find_ecdf_epis_bounds(sim), exp_bounds, alpha=0.25
    )
    assert supports.metric_outputs is not None
    np.testing.assert_allclose(
        supports.metric_outputs[vl.EMetricOut.D_PLUS][0],
        [6.5 - confidence, 6.5 + confidence],
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_allclose(supports.upper, 6.5 + confidence, rtol=0, atol=1e-10)


def test_spatial_and_temporal_bias_survives_nuisance_profiling_and_cell_selection() -> (
    None
):
    positions = np.arange(3.0)
    times = np.arange(2.0)
    baseline = 10 * positions[:, None] + times
    bias = 3 * positions[:, None] + 2 * times
    aleatory = np.array([0.0, 1.0])
    sim_samples = (
        baseline[:, None, :, None, None]
        + np.array([0.0, 2.0])[None, None, None, :, None]
        + aleatory
    )
    exp_samples = baseline[:, None, :, None] + bias[:, None, :, None] + aleatory
    observable = vl.ValidObservable(
        components=("length",),
        geometry=vl.EValidGeometry.POINT,
        coords=np.column_stack((positions, np.zeros(3), np.zeros(3))),
        times=times,
        phys_units="m",
    )
    test_case = vl.ValidCase(
        {"length": observable},
        {
            "length": vl.ValidArray(
                sim_samples,
                (
                    vl.EValidAxis.POINT,
                    vl.EValidAxis.COMPONENT,
                    vl.EValidAxis.TIME,
                    vl.EValidAxis.EPISTEMIC_NUIS,
                    vl.EValidAxis.ALEATORY,
                ),
            )
        },
        {
            "length": vl.ValidArray(
                exp_samples,
                (
                    vl.EValidAxis.POINT,
                    vl.EValidAxis.COMPONENT,
                    vl.EValidAxis.TIME,
                    vl.EValidAxis.ALEATORY,
                ),
            )
        },
    )
    result = vl.ValidEngine([vl.MetricAVM(), vl.MetricMAVM(alpha=0.5)]).run(
        test_case
    )
    values = vl.profile_epis_nuisance_extrema(
        result, "length", vl.EMetricType.AVM
    )
    np.testing.assert_allclose(
        values.minimum[vl.EMetricOut.DISTANCE][:, 0],
        [[0.0, 0.0], [1.0, 3.0], [4.0, 6.0]],
    )
    np.testing.assert_allclose(
        values.maximum[vl.EMetricOut.DISTANCE][:, 0],
        [[2.0, 2.0], [3.0, 5.0], [6.0, 8.0]],
    )
    supports = vl.profile_epis_nuisance_supports(
        test_case, result, "length", vl.EMetricType.MAVM
    )
    np.testing.assert_allclose(supports.lower[:, 0, :, 0], baseline)
    np.testing.assert_allclose(
        supports.upper[:, 0, :, -1], baseline + np.maximum(2.0, bias) + 1
    )
    selected = vl.profile_epis_nuisance_supports(
        test_case,
        result,
        "length",
        vl.EMetricType.MAVM,
        selectors={vl.EValidResAxis.POINT: 2, vl.EValidResAxis.TIME: 1},
    )
    assert selected.axes == (vl.EValidResAxis.COMPONENT,)
    np.testing.assert_allclose(selected.lower[0], [21.0, 22.0])
    np.testing.assert_allclose(selected.upper[0], [29.0, 30.0])
    np.testing.assert_array_equal(
        selected.upper_indices[0], [[2, 0, 1, 0], [2, 0, 1, 0]]
    )


@pytest.mark.parametrize("reference", [0.0, 1e-15, -1e-15])
def test_relative_error_zero_and_near_zero(reference: float) -> None:
    sim_data, exp = (np.array([1.0]), np.array([reference]))
    standalone = vl.calc_deterministic_metrics_1d(sim_data, exp)
    strategy = vl.MetricRelErr().calc(sim_data, exp).value
    if reference == 0:
        assert np.isnan(standalone.rel_err) and np.isnan(strategy)
    else:
        assert standalone.rel_err == pytest.approx(
            abs(1 - reference) / abs(reference)
        )
        assert strategy == standalone.rel_err
    assert np.isnan(standalone.nmse)


def test_mavm_frequency_sensitive_legacy_confidence() -> None:
    model = np.array([0.0, 2.0])
    unique = np.array([0.0, 2.0])
    repeated = np.array([0.0, 0.0, 0.0, 2.0])
    first = vl.calc_mavm_1d(model, unique, duplicates=vl.EMAVMDuplicates.INCLUDE)
    second = vl.calc_mavm_1d(
        model, repeated, duplicates=vl.EMAVMDuplicates.INCLUDE
    )
    expected_half_width = stats.t.ppf(0.95, 1) / np.sqrt(2)
    np.testing.assert_allclose(first.exp_conf_upper - unique, expected_half_width)
    np.testing.assert_array_equal(first.exp_conf_upper, second.exp_conf_upper)
    np.testing.assert_allclose(second.exp_probs, [0.75, 1.0])
    assert second.d_plus == pytest.approx(first.d_plus - 0.5)
    assert second.d_minus == pytest.approx(first.d_minus + 0.5)


def test_cvm_dimensionless_statistic_is_not_bounded_by_one() -> None:
    sim = np.arange(10.0)
    exp = sim + 20.0
    assert vl.MetricCVM().calc(sim, exp).value == pytest.approx(201 / 120)


def test_deterministic_metrics_preserve_observation_order() -> None:
    sim = np.array([0.0, 2.0])
    exp = np.array([4.0, -1.0])
    expected_rmse = sqrt(12.5)
    errors = vl.calc_deterministic_metrics_1d(sim, exp)
    assert errors.abs_err == pytest.approx(3.5)
    assert errors.rmse == pytest.approx(expected_rmse)
    assert errors.rel_err == pytest.approx(0.875)
    assert errors.nmse == pytest.approx(2.0)

    axes = (vl.EValidAxis.ALEATORY,)
    observable = vl.ValidObservable(
        components=("response",), geometry=vl.EValidGeometry.GLOBAL
    )
    case = vl.ValidCase(
        {"response": observable},
        {"response": vl.ValidArray(sim, axes)},
        {"response": vl.ValidArray(exp, axes)},
    )
    for metric, expected in (
        (vl.MetricRMSE(), expected_rmse),
        (vl.MetricRelErr(), 0.875),
    ):
        assert metric.calc(sim, exp).value == pytest.approx(expected)
        result = vl.ValidEngine([metric]).run(case)
        field = result.metrics["response"][metric.get_kind()]
        assert field.metric_vals[vl.EMetricOut.ERROR].item() == pytest.approx(expected)


def calculate_rectangle_signed_areas(
    model_quantiles: np.ndarray,
    model_probs: np.ndarray,
    exp_quantiles: np.ndarray,
    exp_probs: np.ndarray,
) -> tuple[float, float]:
    """Integrate quantile-function differences as finite rectangles.

    Merging the two empirical-CDF probability grids gives a direct,
    independent area calculation for the MAVM integration loop.
    """
    model_idx = 0
    exp_idx = 0
    prob = 0.0
    d_plus = 0.0
    d_minus = 0.0
    while model_idx < len(model_quantiles) and exp_idx < len(exp_quantiles):
        next_prob = min(model_probs[model_idx], exp_probs[exp_idx])
        area = (exp_quantiles[exp_idx] - model_quantiles[model_idx]) * (
            next_prob - prob
        )
        if area > 0.0:
            d_plus += area
        else:
            d_minus -= area
        prob = next_prob
        if np.isclose(prob, model_probs[model_idx]):
            model_idx += 1
        if np.isclose(prob, exp_probs[exp_idx]):
            exp_idx += 1
    return (d_plus, d_minus)


def calculate_rectangle_mavm_areas(
    model_data: np.ndarray, exp_data: np.ndarray, alpha: float = 0.05
) -> tuple[float, float]:
    """Calculate MAVM using only merged empirical-CDF rectangles."""
    model_cdf = stats.ecdf(model_data).cdf
    exp_cdf = stats.ecdf(exp_data).cdf
    model_quantiles = model_cdf.quantiles
    model_probs = model_cdf.probabilities
    exp_quantiles = exp_cdf.quantiles
    exp_probs = exp_cdf.probabilities
    t_value = stats.t.ppf(1.0 - alpha, len(exp_quantiles) - 1)
    standard_error = np.std(exp_quantiles) / np.sqrt(len(exp_quantiles))
    lower = calculate_rectangle_signed_areas(
        model_quantiles,
        model_probs,
        exp_quantiles - t_value * standard_error,
        exp_probs,
    )
    upper = calculate_rectangle_signed_areas(
        model_quantiles,
        model_probs,
        exp_quantiles + t_value * standard_error,
        exp_probs,
    )
    return (max(lower[0], upper[0]), max(lower[1], upper[1]))


@pytest.mark.parametrize(
    ("model_data", "exp_data"),
    [
        (np.array([0.0, 2.0]), np.array([0.0, 2.0])),
        (np.array([2.0, 4.0]), np.array([0.0, 2.0])),
        (np.array([0.0, 2.0]), np.array([2.0, 4.0])),
        (np.array([0.0, 4.0]), np.array([1.0, 2.0, 3.0])),
        (np.array([0.0, 2.0, 4.0]), np.array([1.0, 3.0])),
        (np.array([0.0, 0.0, 2.0]), np.array([1.0, 1.0, 3.0, 3.0])),
        (np.array([0.0, 0.0, 2.0]), np.array([1.0, 3.0, 3.0, 3.0])),
        (np.array([0.0, 2.0, 4.0, 6.0, 8.0]), np.array([1.0, 5.0, 9.0])),
    ],
)
def test_mavm_matches_independent_rectangle_integration(
    model_data: np.ndarray, exp_data: np.ndarray
) -> None:
    """Cover odd/even, unequal-size, and repeated-value ECDF cases."""
    expected_d_plus, expected_d_minus = calculate_rectangle_mavm_areas(
        model_data, exp_data
    )
    result = calc_mavm_1d(model_data, exp_data, duplicates=EMAVMDuplicates.INCLUDE)
    assert result.d_plus == pytest.approx(expected_d_plus, abs=1e-12)
    assert result.d_minus == pytest.approx(expected_d_minus, abs=1e-12)


@pytest.mark.parametrize(
    ("model_data", "exp_data", "expected_area"),
    [
        (np.array([-1.0, 1.0]), np.array([-1.0, 1.0]), 0.0),
        (np.array([1.0, 3.0]), np.array([0.0, 2.0]), 1.0),
        (np.array([0.0, 4.0]), np.array([1.0, 2.0, 3.0]), 4.0 / 3.0),
        (np.array([0.0, 2.0, 4.0]), np.array([1.0, 3.0]), 1.0),
        (np.array([0.0, 0.0, 2.0]), np.array([1.0, 3.0, 3.0, 3.0]), 11.0 / 6.0),
    ],
)
def test_avm_matches_hand_calculated_ecdf_rectangles(
    model_data: np.ndarray, exp_data: np.ndarray, expected_area: float
) -> None:
    """Verify AVM against manually summed empirical-CDF rectangles."""
    assert calc_avm_1d(model_data, exp_data) == pytest.approx(expected_area, abs=1e-12)


def test_crossing_cdfs_distinguish_pointwise_from_support_sum_bounds() -> None:
    scenarios = np.array([[0.0, 10.0], [4.0, 5.0]])
    quantity = vl.ValidArray(
        scenarios, (vl.EValidAxis.EPISTEMIC_TRANS, vl.EValidAxis.ALEATORY)
    )
    bounds = vl.find_ecdf_epis_bounds(quantity)
    np.testing.assert_allclose(bounds.view.probs, [0.5, 1.0])
    np.testing.assert_allclose(bounds.lower, [0.0, 5.0])
    np.testing.assert_allclose(bounds.upper, [4.0, 10.0])
    assert vl.select_ecdf_support_cases(bounds.view.samples) == (1, 0)


def test_quantile_bounds_preserve_duplicate_mass_and_unequal_counts() -> None:
    samples = np.array([[0.0, 0.0, 2.0], [1.0, 3.0, np.nan]])
    quantity = vl.ValidArray(
        samples, (vl.EValidAxis.EPISTEMIC_TRANS, vl.EValidAxis.ALEATORY)
    )
    bounds = vl.find_ecdf_epis_bounds(quantity)
    np.testing.assert_allclose(bounds.view.probs, [0.5, 2.0 / 3.0, 1.0])
    np.testing.assert_allclose(bounds.lower, [0.0, 0.0, 2.0])
    np.testing.assert_allclose(bounds.upper, [1.0, 3.0, 3.0])
    np.testing.assert_allclose(bounds.view.quants.mean(axis=(0, 1, 2)), [0.5, 1.5, 2.5])
    cdf = vl.calc_ecdf(samples[0])
    np.testing.assert_allclose(
        vl.evaluate_ecdf_quantiles(cdf, np.array([0.0, 2.0 / 3.0, 1.0])),
        [0.0, 0.0, 2.0],
    )


def test_endpoint_corrections_are_anchored_to_model_limits() -> None:
    correction = vl.calc_mavm_bound_pairs(
        np.array([0.0]), np.array([10.0]), np.array([-2.0]), np.array([13.0])
    )
    assert correction.lower.d_minus == pytest.approx(2.0)
    assert correction.upper.d_plus == pytest.approx(3.0)
    assert correction.lower_exp_idx == 0
    assert correction.upper_exp_idx == 1
    assert correction.comparisons[2].d_plus == pytest.approx(13.0)
    np.testing.assert_allclose(
        correction.lower.sim_quantiles - correction.lower.d_minus, [-2.0]
    )
    np.testing.assert_allclose(
        correction.upper.sim_quantiles + correction.upper.d_plus, [13.0]
    )


def create_plot_epistemic_case() -> vl.ValidCase:
    shifts = np.array([[1.0, 9.0, 8.0], [2.0, 3.0, 7.0], [4.0, 6.0, 5.0]])
    samples = shifts[..., None] + np.array([0.0, 2.0])
    quantity = vl.ValidArray(
        samples,
        (
            vl.EValidAxis.EPISTEMIC_TRANS,
            vl.EValidAxis.EPISTEMIC_NUIS,
            vl.EValidAxis.ALEATORY,
        ),
    )
    observable = vl.ValidObservable(
        components=("length",), geometry=vl.EValidGeometry.GLOBAL, phys_units="m"
    )
    exp = vl.ValidArray(np.array([2.0, 5.0, 8.0]), (vl.EValidAxis.ALEATORY,))
    return vl.ValidCase(
        {"length": observable}, {"length": quantity}, {"length": exp}
    )


def test_quantile_reductions_preserve_axes_and_use_nuisance_first() -> None:
    test_case = create_plot_epistemic_case()
    quantity = test_case.sim["length"]
    curves = vl.calc_ecdf_view(quantity)
    assert curves.quants.shape == (3, 3, 1, 2)
    np.testing.assert_array_equal(curves.probs, [0.5, 1.0])
    np.testing.assert_array_equal(curves.quants[..., 0, :], quantity.data)
    median_nuis = vl.calc_ecdf_view(
        quantity, view=vl.ECDFViewOpts(nuis_reduce=vl.EECDFReduce.MEDIAN)
    )
    np.testing.assert_array_equal(
        median_nuis.quants[:, 0, 0], [[8.0, 10.0], [3.0, 5.0], [5.0, 7.0]]
    )
    median_both = vl.calc_ecdf_view(
        quantity,
        view=vl.ECDFViewOpts(
            nuis_reduce=vl.EECDFReduce.MEDIAN,
            trans_reduce=vl.EECDFReduce.MEDIAN,
        ),
    )
    np.testing.assert_array_equal(median_both.quants, [[[[5.0, 7.0]]]])
    mean_both = vl.calc_ecdf_view(
        quantity,
        view=vl.ECDFViewOpts(
            nuis_reduce=vl.EECDFReduce.MEAN,
            trans_reduce=vl.EECDFReduce.MEAN,
        ),
    )
    np.testing.assert_allclose(mean_both.quants, [[[[5.0, 7.0]]]])


def test_selection_precedes_reduction_and_original_joint_overlays_are_preserved() -> (
    None
):
    quantity = create_plot_epistemic_case().sim["length"]
    view = vl.ECDFViewOpts(
        trans_idxs=(2, 0),
        nuis_idxs=(1, 2),
        nuis_reduce=vl.EECDFReduce.MEAN,
        scenario_idxs=((0, 2), (2, 1)),
    )
    curves = vl.calc_ecdf_view(quantity, view=view)
    np.testing.assert_array_equal(curves.quants[:, 0, 0], [[5.5, 7.5], [8.5, 10.5]])
    assert tuple(curves.orig_epis_offsets[idx] for idx in curves.overlay_indices) == (
        (0, 2, vl.EEpisOffset.NOMINAL),
        (2, 1, vl.EEpisOffset.NOMINAL),
    )
    original = curves.orig_quants.reshape(-1, curves.probs.size)
    np.testing.assert_array_equal(
        original[list(curves.overlay_indices)], [[8.0, 10.0], [6.0, 8.0]]
    )


def test_pointwise_ecdf_envelope_handles_crossing_distributions() -> None:
    support, lower, upper = vl.calc_ecdf_envelope(
        [np.array([0.0, 3.0]), np.array([1.0, 2.0])]
    )
    np.testing.assert_array_equal(support, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(lower, [0.0, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(upper, [0.5, 0.5, 1.0, 1.0])


def create_linear_sim(shift: float = 0.0) -> SimData:
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    times = np.array([0.0, 1.0])
    temperature = coords[:, :1] + 2 * coords[:, 1:2] + 3 * times + shift
    return SimData(
        num_spat_dims=2,
        coords=coords,
        time=times,
        node_vars={"temperature": temperature},
        glob_vars={"mean_temperature": 1.5 + 3 * times + shift},
    )


def test_sample_bound_field_matches_linear_space_and_time() -> None:
    sim_data = create_linear_sim()
    field = sens.FieldScalar(sim_data, "temperature", sens.EDim.TWOD)
    observable = vl.ValidObservable(
        components=("temperature",),
        geometry=vl.EValidGeometry.POINT,
        coords=np.array([[0.25, 0.5, 0.0], [0.75, 0.5, 0.0]]),
        times=np.array([0.25, 0.75]),
    )
    sampled = vl.sample_field_for_valid(field, observable)
    expected = np.array([[2.0, 3.5], [2.5, 4.0]])[:, None, :]
    assert sampled.axes == (
        vl.EValidAxis.POINT,
        vl.EValidAxis.COMPONENT,
        vl.EValidAxis.TIME,
    )
    np.testing.assert_allclose(sampled.data, expected, rtol=0, atol=1e-12)


def test_sample_sweep_infers_single_axis_shape_from_flat_runs() -> None:
    observable = vl.ValidObservable(
        components=("mean_temperature",),
        geometry=vl.EValidGeometry.GLOBAL,
        times=np.array([0.25, 0.75]),
    )
    sampled = vl.sample_sim_sweep_for_valid(
        fields={},
        observables={"mean_temperature": observable},
        runs=[create_linear_sim(5), create_linear_sim(1)],
        sample_axes=(vl.EValidAxis.EPISTEMIC_TRANS,),
    )["mean_temperature"]

    assert sampled.axes[-1] is vl.EValidAxis.EPISTEMIC_TRANS
    np.testing.assert_allclose(sampled.data[0, 0], [[7.25, 3.25], [8.75, 4.75]])


def test_sample_sweep_reuses_field_and_preserves_named_axes_and_globals() -> None:
    field = sens.FieldScalar(create_linear_sim(), "temperature", sens.EDim.TWOD)
    times = np.array([0.25, 0.75])
    observables = {
        "temperature": vl.ValidObservable(
            components=("temperature",),
            geometry=vl.EValidGeometry.POINT,
            coords=np.array([[0.25, 0.5, 0.0]]),
            times=times,
        ),
        "mean_temperature": vl.ValidObservable(
            components=("mean_temperature",),
            geometry=vl.EValidGeometry.GLOBAL,
            times=times,
        ),
    }
    indices = ((1, 2), (0, 1), (1, 0), (0, 2), (1, 1), (0, 0))
    runs = ((index, create_linear_sim(10 * index[0] + index[1])) for index in indices)
    sampled = vl.sample_sim_sweep_for_valid(
        {"temperature": field},
        observables,
        runs,
        sample_axes=(vl.EValidAxis.EPISTEMIC_TRANS, vl.EValidAxis.ALEATORY),
        sample_shape=(2, 3),
    )
    shifts = 10 * np.arange(2)[:, None] + np.arange(3)
    expected = np.array([2.0, 3.5])[:, None, None] + shifts
    np.testing.assert_allclose(sampled["temperature"].data[0, 0], expected, atol=1e-12)
    np.testing.assert_allclose(
        sampled["mean_temperature"].data[0, 0], expected + 0.25, atol=1e-12
    )
    assert sampled["temperature"].axes[-2:] == (
        vl.EValidAxis.EPISTEMIC_TRANS,
        vl.EValidAxis.ALEATORY,
    )
    np.testing.assert_array_equal(
        field.get_sim_data().node_vars["temperature"],
        create_linear_sim().node_vars["temperature"],
    )


def test_crossing_transferable_supports_have_exact_analytic_limits() -> None:
    axes = (vl.EValidAxis.EPISTEMIC_TRANS, vl.EValidAxis.ALEATORY)
    test_case = vl.ValidCase(
        {
            "response": vl.ValidObservable(
                components=("response",), geometry=vl.EValidGeometry.GLOBAL
            )
        },
        {"response": vl.ValidArray(np.array([[0.0, 100.0], [40.0, 60.0]]), axes)},
        {"response": vl.ValidArray(np.array([[-10.0, 0.0], [100.0, 110.0]]), axes)},
    )
    sim_bounds = vl.find_ecdf_epis_bounds(test_case.sim["response"])
    exp_bounds = vl.find_ecdf_epis_bounds(test_case.exp["response"])
    boundary_supports = vl.calc_mavm_over_epis_bounds(
        sim_bounds, exp_bounds, alpha=0.5
    )
    result = vl.ValidEngine([vl.MetricMAVM(alpha=0.5)]).run(test_case)
    original_supports = vl.find_metric_epis_supports(
        test_case, result, "response", vl.EMetricType.MAVM
    )
    np.testing.assert_allclose(
        original_supports.lower[0, 0, 0], [-55.0, 5.0], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        original_supports.upper[0, 0, 0], [95.0, 155.0], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        boundary_supports.lower, [-35.0, 25.0], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        boundary_supports.upper, [75.0, 135.0], rtol=0, atol=1e-12
    )
