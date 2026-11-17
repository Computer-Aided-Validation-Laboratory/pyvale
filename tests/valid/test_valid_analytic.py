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
* U-pooling integrates |Q_U(p)-p| after mapping each exp observation through
  the right-continuous sim ECDF. Each probability interval is a triangle or
  trapezoid, including the intervals at zero and one.

For example n=2, m=1 gives CDF differences 1/2 and -1/2: AVM=1, KS=1/2,
MAVM d+=d-=1/2. Its pooled observation is 1/2, giving two triangles of total
area 1/4. CvM is undefined whenever either sample count is below two.
Equal counts necessarily have matching parity; mixed parity applies to both
orientations of unequal counts. Paired metrics reject every unequal count.
"""

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pytest

from pyvale import valid


@dataclass(slots=True)
class AnalyticCase:
    sim_count: int
    exp_count: int
    avm: float
    ks: float
    cvm: float
    d_plus: float
    d_minus: float
    u_pooling: float


# Counts: deterministic/deterministic; each deterministic/aleatory orientation;
# equal even/even and odd/odd; then all four parities in each unequal direction.
ANALYTIC_CASES = (
    AnalyticCase(1, 1, 1, 1, np.nan, 1, 0, 1 / 2),
    AnalyticCase(1, 2, 2, 1, np.nan, 2, 0, 1 / 2),
    AnalyticCase(1, 3, 3, 1, np.nan, 3, 0, 1 / 2),
    AnalyticCase(2, 1, 1, 1 / 2, np.nan, 1 / 2, 1 / 2, 1 / 4),
    AnalyticCase(3, 1, 5 / 3, 2 / 3, np.nan, 1 / 3, 4 / 3, 5 / 18),
    AnalyticCase(2, 2, 1, 1 / 2, 1 / 8, 1, 0, 1 / 4),
    AnalyticCase(3, 3, 1, 1 / 3, 1 / 12, 1, 0, 1 / 6),
    AnalyticCase(4, 2, 3 / 2, 1 / 2, 7 / 72, 1 / 4, 5 / 4, 3 / 16),
    AnalyticCase(4, 3, 1, 1 / 4, 1 / 21, 1 / 2, 1 / 2, 7 / 72),
    AnalyticCase(5, 2, 12 / 5, 3 / 5, 1 / 7, 1 / 5, 11 / 5, 6 / 25),
    AnalyticCase(5, 3, 23 / 15, 2 / 5, 19 / 240, 4 / 15, 19 / 15, 13 / 90),
    AnalyticCase(2, 4, 3, 3 / 4, 19 / 72, 3, 0, 3 / 8),
    AnalyticCase(3, 4, 2, 1 / 2, 13 / 84, 2, 0, 1 / 4),
    AnalyticCase(2, 5, 4, 4 / 5, 11 / 35, 4, 0, 2 / 5),
    AnalyticCase(3, 5, 3, 3 / 5, 11 / 48, 3, 0, 3 / 10),
)
CASE_IDS = tuple(
    f"sim-{case.sim_count}_exp-{case.exp_count}" for case in ANALYTIC_CASES
)


@pytest.mark.parametrize("case", ANALYTIC_CASES, ids=CASE_IDS)
def test_interleaved_distribution_metrics_have_exact_answers(
    case: AnalyticCase,
) -> None:
    sim = 2.0 * np.arange(case.sim_count)
    exp = 1.0 + 2.0 * np.arange(case.exp_count)
    assert valid.calc_avm_1d(sim, exp) == pytest.approx(case.avm, rel=0, abs=1e-12)
    assert valid.calc_ks_1d(sim, exp) == pytest.approx(case.ks, rel=0, abs=1e-12)
    if np.isnan(case.cvm):
        with pytest.warns(RuntimeWarning, match="sample arguments is too small"):
            cvm = valid.calc_cvm_1d(sim, exp)
        assert np.isnan(cvm)
    else:
        cvm = valid.calc_cvm_1d(sim, exp)
        assert cvm == pytest.approx(case.cvm, rel=0, abs=1e-12)
    mavm = valid.calc_mavm_1d(sim, exp, alpha=0.5)
    assert mavm.d_plus == pytest.approx(case.d_plus, rel=0, abs=1e-12)
    assert mavm.d_minus == pytest.approx(case.d_minus, rel=0, abs=1e-12)
    np.testing.assert_allclose(mavm.exp_conf_lower, exp, rtol=0, atol=1e-12)
    np.testing.assert_allclose(mavm.exp_conf_upper, exp, rtol=0, atol=1e-12)

    # Explicit ECDF construction keeps the pooling reference independent of calc_ecdf.
    sim_cdf = valid.ValidECDF(sim, np.arange(1, case.sim_count + 1) / case.sim_count)
    assert valid.calc_u_pooling_1d([sim_cdf] * case.exp_count, exp) == pytest.approx(
        case.u_pooling, rel=0, abs=1e-12
    )


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


@pytest.mark.parametrize("case", ANALYTIC_CASES, ids=CASE_IDS)
@pytest.mark.parametrize("sim_shift", (-20.0, 20.0), ids=("sim-below", "sim-above"))
def test_separated_metrics_and_mavm_confidence_have_closed_form_answers(
    case: AnalyticCase,
    sim_shift: float,
) -> None:
    sim = sim_shift + 2.0 * np.arange(case.sim_count)
    exp = 1.0 + 2.0 * np.arange(case.exp_count)
    # Entire distributions, including confidence bounds, are strictly separated.
    mean_gap = case.exp_count - (sim_shift + case.sim_count - 1)
    confidence_shift = STUDENT_T_95[case.exp_count - 1] * sqrt(
        (case.exp_count**2 - 1) / (3 * case.exp_count)
    )
    assert valid.calc_avm_1d(sim, exp) == pytest.approx(abs(mean_gap), rel=0, abs=1e-12)
    assert valid.calc_ks_1d(sim, exp) == pytest.approx(1.0, rel=0, abs=1e-12)
    if min(case.sim_count, case.exp_count) == 1:
        with pytest.warns(RuntimeWarning, match="sample arguments is too small"):
            cvm = valid.calc_cvm_1d(sim, exp)
        assert np.isnan(cvm)
    else:
        cvm = valid.calc_cvm_1d(sim, exp)
        # Disjoint pooled ranks reduce the rank statistic to this expression.
        expected_cvm = (2 * case.sim_count * case.exp_count + 1) / (
            6 * (case.sim_count + case.exp_count)
        )
        assert cvm == pytest.approx(expected_cvm, rel=0, abs=1e-12)
    mavm = valid.calc_mavm_1d(sim, exp)
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
    # A degenerate model p-box must give exactly the single-model MAVM.
    pbox = valid.calc_mavm_pbox_1d(sim, sim, exp)
    assert pbox.d_plus == pytest.approx(expected_d_plus, rel=0, abs=1e-10)
    assert pbox.d_minus == pytest.approx(expected_d_minus, rel=0, abs=1e-10)
    sim_cdf = valid.ValidECDF(sim, np.arange(1, case.sim_count + 1) / case.sim_count)
    assert valid.calc_u_pooling_1d([sim_cdf] * case.exp_count, exp) == pytest.approx(
        1 / 2, rel=0, abs=1e-12
    )


@pytest.mark.parametrize("case", ANALYTIC_CASES, ids=CASE_IDS)
def test_paired_metrics_use_pairs_and_reject_unequal_counts(case: AnalyticCase) -> None:
    sim = 2.0 * np.arange(case.sim_count)
    exp = 1.0 + 3.0 * np.arange(case.exp_count)
    if case.sim_count != case.exp_count:
        with pytest.raises(ValueError, match="paired"):
            valid.MetricRMSE().calc(sim, exp)
        with pytest.raises(ValueError, match="paired"):
            valid.MetricRelErr().calc(sim, exp)
        with pytest.raises(ValueError, match="paired"):
            valid.calc_deterministic_metrics_1d(sim, exp)
        return
    # Residual magnitudes are 1,2,...,n, with known first and second moments.
    count = case.sim_count
    mean_absolute_error = (count + 1) / 2
    mean_square_error = (count + 1) * (2 * count + 1) / 6
    exp_variance = 3 * (count**2 - 1) / 4
    expected_relative = mean_absolute_error / (3 * count - 2)
    errors = valid.calc_deterministic_metrics_1d(sim, exp)
    assert errors.absolute_error == pytest.approx(mean_absolute_error, rel=0, abs=1e-12)
    assert errors.rmse == pytest.approx(sqrt(mean_square_error), rel=0, abs=1e-12)
    assert errors.relative_error == pytest.approx(expected_relative, rel=0, abs=1e-12)
    if count == 1:
        assert np.isnan(errors.nmse)
    else:
        assert errors.nmse == pytest.approx(
            mean_square_error / exp_variance, rel=0, abs=1e-12
        )
    assert valid.MetricRMSE().calc(sim, exp).value == pytest.approx(
        sqrt(mean_square_error), rel=0, abs=1e-12
    )
    for normalisation, denominator in (
        (valid.ERelErrNorm.EXP_MAX, 3 * count - 2),
        (valid.ERelErrNorm.SIM_MAX, 2 * (count - 1)),
        (valid.ERelErrNorm.CONSTANT, 10.0),
    ):
        constant = 10.0 if normalisation is valid.ERelErrNorm.CONSTANT else None
        metric = valid.MetricRelErr(normalisation=normalisation, constant=constant)
        relative = metric.calc(sim, exp).value
        errors = valid.calc_deterministic_metrics_1d(
            sim,
            exp,
            normalisation=normalisation,
            constant=constant,
        )
        if denominator == 0:
            assert np.isnan(relative)
            assert np.isnan(errors.relative_error)
        else:
            expected = mean_absolute_error / denominator
            assert relative == pytest.approx(expected, rel=0, abs=1e-12)
            assert errors.relative_error == pytest.approx(expected, rel=0, abs=1e-12)


@pytest.mark.parametrize("case", ANALYTIC_CASES, ids=CASE_IDS)
def test_engine_outputs_match_analytic_sample_count_cases(case: AnalyticCase) -> None:
    sim = 2.0 * np.arange(case.sim_count)
    exp = 1.0 + 2.0 * np.arange(case.exp_count)
    axes = (
        valid.EValidAxis.POINT,
        valid.EValidAxis.COMPONENT,
        valid.EValidAxis.TIME,
        valid.EValidAxis.ALEATORY,
    )
    valid_case = valid.ValidCase(
        {
            "response": valid.ValidObservable(
                components=("response",), geometry=valid.EValidGeometry.GLOBAL
            )
        },
        {"response": valid.ValidArray(sim.reshape(1, 1, 1, -1), axes)},
        {"response": valid.ValidArray(exp.reshape(1, 1, 1, -1), axes)},
    )
    metrics: list[valid.IValidMetric] = [
        valid.MetricAVM(),
        valid.MetricKS(),
        valid.MetricCVM(),
        valid.MetricMAVM(alpha=0.5),
    ]
    if case.sim_count == case.exp_count:
        metrics.extend([valid.MetricRMSE(), valid.MetricRelErr()])
    if np.isnan(case.cvm):
        with pytest.warns(RuntimeWarning, match="sample arguments is too small"):
            result = valid.ValidEngine(metrics).run(valid_case)
    else:
        result = valid.ValidEngine(metrics).run(valid_case)
    expected_outputs = (
        (valid.EMetricType.AVM, valid.EMetricOutput.DISTANCE, case.avm),
        (valid.EMetricType.KS, valid.EMetricOutput.STATISTIC, case.ks),
        (valid.EMetricType.CVM, valid.EMetricOutput.STATISTIC, case.cvm),
        (valid.EMetricType.MAVM, valid.EMetricOutput.D_PLUS, case.d_plus),
        (valid.EMetricType.MAVM, valid.EMetricOutput.D_MINUS, case.d_minus),
    )
    for metric_type, output, expected in expected_outputs:
        field = result.metrics["response"][metric_type]
        assert field.axes == (
            valid.EValidResAxis.POINT,
            valid.EValidResAxis.COMPONENT,
            valid.EValidResAxis.TIME,
        )
        assert field.outputs[output].shape == (1, 1, 1)
        actual = field.outputs[output][0, 0, 0]
        if np.isnan(expected):
            assert np.isnan(actual)
        else:
            assert actual == pytest.approx(expected, rel=0, abs=1e-12)
    if case.sim_count == case.exp_count:
        assert result.metrics["response"][valid.EMetricType.RMSE].outputs[
            valid.EMetricOutput.ERROR
        ][0, 0, 0] == pytest.approx(1.0, rel=0, abs=1e-12)
        assert result.metrics["response"][valid.EMetricType.RELATIVE_ERROR].outputs[
            valid.EMetricOutput.ERROR
        ][0, 0, 0] == pytest.approx(1 / (2 * case.exp_count - 1), rel=0, abs=1e-12)
    else:
        for metric in (valid.MetricRMSE(), valid.MetricRelErr()):
            with pytest.raises(ValueError, match="equal aleatory counts"):
                valid.ValidEngine([metric]).run(valid_case)


# Epistemic toolbox references: all probabilities and distances below are
# hand-derived. No production profiling function supplies an expected answer.


def create_epistemic_case(sim: np.ndarray, exp: np.ndarray) -> valid.ValidCase:
    """Create a scalar physical cell with (transferable, nuisance, aleatory) axes."""
    axes = (
        valid.EValidAxis.EPISTEMIC_TRANS,
        valid.EValidAxis.EPISTEMIC_NUIS,
        valid.EValidAxis.ALEATORY,
    )
    observable = valid.ValidObservable(
        components=("length",), geometry=valid.EValidGeometry.GLOBAL, phys_units="m"
    )
    return valid.ValidCase(
        {"length": observable},
        {"length": valid.ValidArray(sim, axes)},
        {"length": valid.ValidArray(exp, axes)},
    )


@pytest.mark.parametrize(
    "reduction, probability, lower, upper",
    [
        (valid.EValidECDFReduce.KEEP, None, 1.0, 90.0),
        (valid.EValidECDFReduce.MEAN, None, 4.0, 33.0),
        (valid.EValidECDFReduce.MEDIAN, None, 3.0, 8.0),
        (valid.EValidECDFReduce.QUANTILE, 0.0, 1.0, 4.0),
        (valid.EValidECDFReduce.QUANTILE, 0.25, 2.5, 4.5),
        (valid.EValidECDFReduce.QUANTILE, 0.5, 3.0, 8.0),
        (valid.EValidECDFReduce.QUANTILE, 0.75, 5.0, 49.0),
        (valid.EValidECDFReduce.QUANTILE, 1.0, 6.0, 90.0),
    ],
)
def test_epis_bounds_nuisance_treatments_have_analytic_quantiles(
    reduction: valid.EValidECDFReduce,
    probability: float | None,
    lower: float,
    upper: float,
) -> None:
    # Nuisance shifts per transferable case: [1,90,8], [2,3,7], [4,6,5].
    shifts = np.array([[1.0, 90.0, 8.0], [2.0, 3.0, 7.0], [4.0, 6.0, 5.0]])
    quantity = valid.ValidArray(
        shifts[..., None] + np.array([0.0, 2.0]),
        (
            valid.EValidAxis.EPISTEMIC_TRANS,
            valid.EValidAxis.EPISTEMIC_NUIS,
            valid.EValidAxis.ALEATORY,
        ),
    )
    bounds = valid.find_ecdf_epis_bounds(
        quantity, nuis_reduce=reduction, nuis_quantile=probability
    )
    np.testing.assert_allclose(bounds.probabilities, [0.5, 1.0])
    np.testing.assert_allclose(bounds.lower, [lower, lower + 2], rtol=0, atol=1e-12)
    np.testing.assert_allclose(bounds.upper, [upper, upper + 2], rtol=0, atol=1e-12)
    if reduction is valid.EValidECDFReduce.KEEP:
        np.testing.assert_array_equal(bounds.lower_indices[:, :2], [[0, 0], [0, 0]])
        np.testing.assert_array_equal(bounds.upper_indices[:, :2], [[0, 1], [0, 1]])
    else:
        assert np.all(bounds.lower_indices[:, 1] == -1)
        assert np.all(bounds.upper_indices[:, 1] == -1)
    conditioned = valid.find_ecdf_epis_bounds(quantity, nuis_idxs=(1,))
    np.testing.assert_array_equal(conditioned.lower, [3.0, 5.0])
    np.testing.assert_array_equal(conditioned.upper, [90.0, 92.0])


def test_avm_bounds_preserve_unequal_empirical_probability_mass() -> None:
    sim = valid.ValidArray(
        np.array([[0.0, 10.0, np.nan], [2.0, 3.0, 20.0]]),
        (valid.EValidAxis.EPISTEMIC_TRANS, valid.EValidAxis.ALEATORY),
    )
    exp = valid.ValidArray(np.array([5.0]), (valid.EValidAxis.ALEATORY,))
    sim_bounds = valid.find_ecdf_epis_bounds(sim)
    exp_bounds = valid.find_ecdf_epis_bounds(exp)
    # The grid has interval weights [1/3,1/6,1/6,1/3], not four equal masses.
    np.testing.assert_allclose(sim_bounds.probabilities, [1 / 3, 0.5, 2 / 3, 1.0])
    np.testing.assert_array_equal(sim_bounds.lower, [0.0, 0.0, 3.0, 10.0])
    np.testing.assert_array_equal(sim_bounds.upper, [2.0, 3.0, 10.0, 20.0])
    supports = valid.calc_avm_over_epis_bounds(sim_bounds, exp_bounds)
    assert supports.metric_outputs is not None
    np.testing.assert_allclose(
        supports.metric_outputs[valid.EMetricOutput.DISTANCE],
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
    sim = valid.ValidArray(
        np.array([[0.0, 100.0], [40.0, 100.0]]),
        (valid.EValidAxis.EPISTEMIC_TRANS, valid.EValidAxis.ALEATORY),
    )
    exp = valid.ValidArray(np.array([100.0]), (valid.EValidAxis.ALEATORY,))
    supports = valid.calc_mavm_over_epis_bounds(
        valid.find_ecdf_epis_bounds(sim), valid.find_ecdf_epis_bounds(exp)
    )
    # d+ is 50 on the lower boundary and 30 on the upper boundary. Thus the
    # upper boundary controls p=.5 (70), but the lower controls p=1 (150).
    np.testing.assert_allclose(supports.upper, [70.0, 150.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(supports.lower, [0.0, 100.0], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(supports.upper_indices[:, 0], [1, 0])
    assert supports.metric_outputs is not None
    np.testing.assert_allclose(
        supports.metric_outputs[valid.EMetricOutput.D_PLUS],
        [[50.0, 50.0], [30.0, 30.0]],
        rtol=0,
        atol=1e-12,
    )


def test_mavm_confidence_is_constructed_before_enveloping_original_exp_cases() -> None:
    sim = valid.ValidArray(np.array([-20.0]), (valid.EValidAxis.ALEATORY,))
    exp = valid.ValidArray(
        np.array([[0.0, 2.0], [10.0, 14.0]]),
        (valid.EValidAxis.EPISTEMIC_TRANS, valid.EValidAxis.ALEATORY),
    )
    # For alpha=.25 and df=1 the t quantile is exactly 1. Original standard
    # errors are 1/sqrt(2) and 2/sqrt(2); no synthetic bound sample count exists.
    shift_lower = 1 / sqrt(2)
    shift_upper = 2 / sqrt(2)
    supports = valid.calc_mavm_over_epis_bounds(
        valid.find_ecdf_epis_bounds(sim), valid.find_ecdf_epis_bounds(exp), alpha=0.25
    )
    expected_shift = 32 + shift_upper  # mean upper confidence support minus -20.
    np.testing.assert_allclose(
        supports.upper, [-20 + expected_shift, -20 + expected_shift], rtol=0, atol=1e-10
    )
    assert supports.metric_outputs is not None
    np.testing.assert_allclose(
        supports.metric_outputs[valid.EMetricOutput.D_PLUS][0],
        [21 - shift_lower, 32 + shift_upper],
        rtol=0,
        atol=1e-10,
    )


def test_bounds_route_and_original_scenario_route_are_distinct() -> None:
    case = create_epistemic_case(
        np.array([[[0.0, 100.0], [40.0, 60.0]]]), np.array([[[100.0]]])
    )
    result = valid.ValidEngine([valid.MetricMAVM(alpha=0.5)]).run(case)
    direct = valid.find_metric_epis_supports(
        case, result, "length", valid.EMetricType.MAVM
    )
    bounded = valid.calc_mavm_over_epis_bounds(
        valid.find_ecdf_epis_bounds(case.sim["length"]),
        valid.find_ecdf_epis_bounds(case.exp["length"]),
        alpha=0.5,
    )
    np.testing.assert_allclose(direct.upper[0, 0, 0], [90.0, 150.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(bounded.upper, [70.0, 130.0], rtol=0, atol=1e-12)


def test_nuisance_profiles_preserve_transferable_axes_and_original_arg_indices() -> (
    None
):
    sim = np.array([[0.0, 10.0], [2.0, 12.0]])[..., None] + np.array([0.0, 2.0])
    exp = np.array([[1.0, 4.0], [3.0, 6.0]])[..., None] + np.array([0.0, 2.0])
    case = create_epistemic_case(sim, exp)
    result = valid.ValidEngine([valid.MetricAVM(), valid.MetricMAVM(alpha=0.5)]).run(
        case
    )
    extrema = valid.profile_epis_nuisance_extrema(
        result, "length", valid.EMetricType.AVM
    )
    assert extrema.axes == (
        valid.EValidResAxis.POINT,
        valid.EValidResAxis.COMPONENT,
        valid.EValidResAxis.TIME,
        valid.EValidResAxis.SIM_TRANS,
        valid.EValidResAxis.EXP_TRANS,
    )
    assert extrema.reduced_axes == (
        valid.EValidResAxis.SIM_NUIS,
        valid.EValidResAxis.EXP_NUIS,
    )
    np.testing.assert_allclose(
        extrema.minimum[valid.EMetricOutput.DISTANCE][0, 0, 0], [[1.0, 3.0], [1.0, 1.0]]
    )
    np.testing.assert_allclose(
        extrema.maximum[valid.EMetricOutput.DISTANCE][0, 0, 0],
        [[9.0, 7.0], [11.0, 9.0]],
    )
    np.testing.assert_array_equal(
        extrema.maximum_indices[valid.EMetricOutput.DISTANCE][0, 0, 0],
        np.tile([1, 0], (2, 2, 1)),
    )
    mavm_extrema = valid.profile_epis_nuisance_extrema(
        result, "length", valid.EMetricType.MAVM
    )
    np.testing.assert_allclose(
        mavm_extrema.maximum[valid.EMetricOutput.D_PLUS][0, 0, 0],
        [[4.0, 6.0], [2.0, 4.0]],
    )
    np.testing.assert_allclose(
        mavm_extrema.maximum[valid.EMetricOutput.D_MINUS][0, 0, 0],
        [[9.0, 7.0], [11.0, 9.0]],
    )
    supports = valid.profile_epis_nuisance_supports(
        case, result, "length", valid.EMetricType.MAVM
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
        valid.MetricAVM(),
        valid.MetricKS(),
        valid.MetricCVM(),
        valid.MetricRMSE(),
        valid.MetricRelErr(),
    ],
)
def test_other_metric_nuisance_extrema_match_hand_calculated_shift_cases(
    metric: valid.IValidMetric,
) -> None:
    # Same-sized paired two-point samples: residual shifts are 0 and 2.
    case = create_epistemic_case(
        np.array([[[0.0, 2.0], [2.0, 4.0]]]), np.array([[[0.0, 2.0]]])
    )
    result = valid.ValidEngine([metric]).run(case)
    profile = valid.profile_epis_nuisance_extrema(result, "length", metric.get_kind())
    output = metric.get_outputs()[0]
    expected_max = {
        valid.EMetricType.AVM: 2.0,
        valid.EMetricType.KS: 0.5,
        valid.EMetricType.CVM: 3 / 16,
        valid.EMetricType.RMSE: 2.0,
        valid.EMetricType.RELATIVE_ERROR: 1.0,
    }[metric.get_kind()]
    np.testing.assert_allclose(profile.minimum[output], 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        profile.maximum[output], expected_max, rtol=0, atol=1e-12
    )
    if metric.get_kind() not in (valid.EMetricType.AVM, valid.EMetricType.MAVM):
        with pytest.raises(ValueError, match="only for AVM and MAVM"):
            valid.profile_epis_nuisance_supports(
                case, result, "length", metric.get_kind()
            )


def test_nuisance_profiles_keep_missing_cells_and_offset_selections() -> None:
    case = create_epistemic_case(
        np.array([[[0.0, 2.0], [np.nan, np.nan]]]), np.array([[[3.0, 5.0]]])
    )
    case.exp["length"].offsets = valid.EpistemicOffsets(1.0, 2.0)
    result = valid.ValidEngine([valid.MetricMAVM(alpha=0.5)]).run(case)
    extrema = valid.profile_epis_nuisance_extrema(
        result, "length", valid.EMetricType.MAVM
    )
    assert valid.EValidResAxis.EXP_OFFSET in extrema.axes
    np.testing.assert_allclose(
        extrema.maximum[valid.EMetricOutput.D_PLUS][0, 0, 0], [3.0, 2.0, 5.0]
    )
    supports = valid.profile_epis_nuisance_supports(
        case,
        result,
        "length",
        valid.EMetricType.MAVM,
        selectors={valid.EValidResAxis.EXP_OFFSET: valid.EEpisOffset.UPPER},
    )
    np.testing.assert_allclose(supports.upper[0, 0, 0], [5.0, 7.0])
    # Explicitly selecting the missing original nuisance case leaves missing bounds.
    missing = valid.find_metric_epis_supports(
        case,
        result,
        "length",
        valid.EMetricType.MAVM,
        selectors={valid.EValidResAxis.SIM_NUIS: 1},
    )
    assert np.all(np.isnan(missing.lower)) and np.all(missing.lower_indices == -1)
    missing_values = valid.find_metric_epis_extrema(
        result,
        "length",
        valid.EMetricType.MAVM,
        selectors={valid.EValidResAxis.SIM_NUIS: 1},
    )
    assert np.all(np.isnan(missing_values.maximum[valid.EMetricOutput.D_PLUS]))


def test_ranked_nuisance_cases_are_coherent_and_use_explicit_width_scores() -> None:
    probabilities = np.array([0.5, 1.0])
    shifted_width = np.array([[3.0, 10.0], [4.0, 8.0]])
    original_width = np.array([[1.0, 8.0], [1.0, 2.0]])
    upper = np.repeat(shifted_width[..., None], 2, axis=-1)
    sim_upper = np.repeat(original_width[..., None], 2, axis=-1)
    origin = np.empty((*upper.shape, 0), dtype=int)
    supports = valid.ValidSupportBounds(
        (valid.EValidResAxis.SIM_NUIS, valid.EValidResAxis.EXP_NUIS),
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
    total = valid.rank_epis_nuisance_cases(supports)
    added = valid.rank_epis_nuisance_cases(
        supports, score=valid.EValidNuisanceScore.ADDED_WIDTH
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
    missing = valid.rank_epis_nuisance_cases(supports)
    assert missing.valid_counts == 3
    np.testing.assert_array_equal(missing.order, [0, 2, 1, -1])


@pytest.mark.parametrize("case", ANALYTIC_CASES, ids=CASE_IDS)
def test_bounds_and_profiles_cover_the_full_analytic_sample_count_matrix(
    case: AnalyticCase,
) -> None:
    sim = valid.ValidArray(
        2.0 * np.arange(case.sim_count), (valid.EValidAxis.ALEATORY,)
    )
    exp = valid.ValidArray(
        1.0 + 2.0 * np.arange(case.exp_count), (valid.EValidAxis.ALEATORY,)
    )
    sim_bounds = valid.find_ecdf_epis_bounds(sim)
    exp_bounds = valid.find_ecdf_epis_bounds(exp)
    avm = valid.calc_avm_over_epis_bounds(sim_bounds, exp_bounds)
    mavm = valid.calc_mavm_over_epis_bounds(sim_bounds, exp_bounds, alpha=0.5)
    assert avm.metric_outputs is not None and mavm.metric_outputs is not None
    np.testing.assert_allclose(
        avm.metric_outputs[valid.EMetricOutput.DISTANCE], case.avm, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        mavm.metric_outputs[valid.EMetricOutput.D_PLUS], case.d_plus, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        mavm.metric_outputs[valid.EMetricOutput.D_MINUS],
        case.d_minus,
        rtol=0,
        atol=1e-12,
    )
    assert avm.lower[0] == pytest.approx(-case.avm, rel=0, abs=1e-12)
    assert avm.upper[-1] == pytest.approx(
        2 * (case.sim_count - 1) + case.avm, rel=0, abs=1e-12
    )
    assert mavm.lower[0] == pytest.approx(-case.d_minus, rel=0, abs=1e-12)
    assert mavm.upper[-1] == pytest.approx(
        2 * (case.sim_count - 1) + case.d_plus, rel=0, abs=1e-12
    )
    observable = valid.ValidObservable(
        components=("length",), geometry=valid.EValidGeometry.GLOBAL
    )
    valid_case = valid.ValidCase(
        {"length": observable}, {"length": sim}, {"length": exp}
    )
    result = valid.ValidEngine([valid.MetricAVM(), valid.MetricMAVM(alpha=0.5)]).run(
        valid_case
    )
    profile = valid.profile_epis_nuisance_extrema(
        result, "length", valid.EMetricType.AVM
    )
    assert profile.reduced_axes == ()  # Singleton/absent epistemic axes are valid.
    np.testing.assert_allclose(
        profile.minimum[valid.EMetricOutput.DISTANCE], case.avm, rtol=0, atol=1e-12
    )
    supports = valid.find_metric_epis_supports(
        valid_case, result, "length", valid.EMetricType.MAVM
    )
    assert supports.lower[0, 0, 0, 0] == pytest.approx(-case.d_minus, rel=0, abs=1e-12)
    assert supports.upper[0, 0, 0, -1] == pytest.approx(
        2 * (case.sim_count - 1) + case.d_plus, rel=0, abs=1e-12
    )


def test_mavm_confidence_preserves_original_counts_before_nuisance_mean() -> None:
    sim = valid.ValidArray(np.array([0.0]), (valid.EValidAxis.ALEATORY,))
    exp = valid.ValidArray(
        np.array([[0.0, 2.0, np.nan], [10.0, 12.0, 14.0]]),
        (valid.EValidAxis.EPISTEMIC_NUIS, valid.EValidAxis.ALEATORY),
    )
    exp_bounds = valid.find_ecdf_epis_bounds(
        exp, nuis_reduce=valid.EValidECDFReduce.MEAN
    )
    # The mean quantile curve is [5,6,7,8] on intervals [1/3,1/6,1/6,1/3],
    # giving mean 6.5. Confidence shifts use the original counts two and three,
    # not four pseudo-observations reconstructed from this curve's vertices.
    confidence = (1 / sqrt(2) + 4 / (3 * sqrt(3))) / 2
    supports = valid.calc_mavm_over_epis_bounds(
        valid.find_ecdf_epis_bounds(sim), exp_bounds, alpha=0.25
    )
    assert supports.metric_outputs is not None
    np.testing.assert_allclose(
        supports.metric_outputs[valid.EMetricOutput.D_PLUS][0],
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
    observable = valid.ValidObservable(
        components=("length",),
        geometry=valid.EValidGeometry.POINT,
        coords=np.column_stack((positions, np.zeros(3), np.zeros(3))),
        times=times,
        phys_units="m",
    )
    case = valid.ValidCase(
        {"length": observable},
        {
            "length": valid.ValidArray(
                sim_samples,
                (
                    valid.EValidAxis.POINT,
                    valid.EValidAxis.COMPONENT,
                    valid.EValidAxis.TIME,
                    valid.EValidAxis.EPISTEMIC_NUIS,
                    valid.EValidAxis.ALEATORY,
                ),
            )
        },
        {
            "length": valid.ValidArray(
                exp_samples,
                (
                    valid.EValidAxis.POINT,
                    valid.EValidAxis.COMPONENT,
                    valid.EValidAxis.TIME,
                    valid.EValidAxis.ALEATORY,
                ),
            )
        },
    )
    result = valid.ValidEngine([valid.MetricAVM(), valid.MetricMAVM(alpha=0.5)]).run(
        case
    )
    values = valid.profile_epis_nuisance_extrema(
        result, "length", valid.EMetricType.AVM
    )
    np.testing.assert_allclose(
        values.minimum[valid.EMetricOutput.DISTANCE][:, 0],
        [[0.0, 0.0], [1.0, 3.0], [4.0, 6.0]],
    )
    np.testing.assert_allclose(
        values.maximum[valid.EMetricOutput.DISTANCE][:, 0],
        [[2.0, 2.0], [3.0, 5.0], [6.0, 8.0]],
    )
    supports = valid.profile_epis_nuisance_supports(
        case, result, "length", valid.EMetricType.MAVM
    )
    np.testing.assert_allclose(supports.lower[:, 0, :, 0], baseline)
    np.testing.assert_allclose(
        supports.upper[:, 0, :, -1], baseline + np.maximum(2.0, bias) + 1
    )
    selected = valid.profile_epis_nuisance_supports(
        case,
        result,
        "length",
        valid.EMetricType.MAVM,
        selectors={valid.EValidResAxis.POINT: 2, valid.EValidResAxis.TIME: 1},
    )
    assert selected.axes == (valid.EValidResAxis.COMPONENT,)
    np.testing.assert_allclose(selected.lower[0], [21.0, 22.0])
    np.testing.assert_allclose(selected.upper[0], [29.0, 30.0])
    np.testing.assert_array_equal(
        selected.upper_indices[0], [[2, 0, 1, 0], [2, 0, 1, 0]]
    )
