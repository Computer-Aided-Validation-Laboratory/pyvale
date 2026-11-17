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
