"""Gather and plot persisted workflow results."""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .result import CaseResult, WorkflowDataset
from .selector import EStrainComponent, SignalExtraction


@dataclass(frozen=True, slots=True)
class ConvergenceMetric:
    """Define the metrics and signal extraction used by one convergence plot."""

    component: EStrainComponent
    signal_extraction: SignalExtraction
    signal_metric: str = "signal_mean"
    noise_metric: str = "noise_floor_mean"


class WorkflowGatherer:
    """Gather compact workflow summaries from memory or disk."""

    @staticmethod
    def from_results(results: tuple[CaseResult, ...]) -> WorkflowDataset:
        """Convert completed and failed case results into NumPy columns."""
        parameter_keys = sorted({key for item in results for key in item.case.values})
        metric_keys = sorted({key for item in results for key in item.metrics})
        parameters = {
            key: np.asarray([item.case.values.get(key) for item in results])
            for key in parameter_keys
        }
        metrics = {
            key: np.asarray(
                [item.metrics.get(key, np.nan) for item in results],
                dtype=float,
            )
            for key in metric_keys
        }
        return WorkflowDataset(
            parameters=parameters,
            metrics=metrics,
            statuses=np.asarray([item.status.value for item in results]),
            case_dirs=tuple(
                item.artifacts[0]
                if item.artifacts and item.artifacts[0].is_dir()
                else Path()
                for item in results
            ),
        )

    @staticmethod
    def gather(output_dir: Path) -> WorkflowDataset:
        """Read persisted case parameters and summaries from one workflow run."""
        results: list[CaseResult] = []
        for case_dir in sorted((Path(output_dir) / "cases").glob("*")):
            with (case_dir / "parameters.json").open(encoding="utf-8") as file:
                values = json.load(file)
            with (case_dir / "summary.json").open(encoding="utf-8") as file:
                summary = json.load(file)
            from .case import WorkflowCase
            from .result import ECaseStatus

            results.append(
                CaseResult(
                    WorkflowCase(int(case_dir.name), values, 0),
                    summary["metrics"],
                    (case_dir,),
                    ECaseStatus(summary["status"]),
                    summary["elapsed_seconds"],
                    summary["error"],
                ),
            )
        return WorkflowGatherer.from_results(tuple(results))

    @staticmethod
    def aggregate_repeats(
        dataset: WorkflowDataset,
        repeat_parameter: str = "repeat",
    ) -> WorkflowDataset:
        """Aggregate repeat cases while retaining numeric and categorical keys."""
        if repeat_parameter not in dataset.parameters:
            raise KeyError(f"Unknown repeat parameter: {repeat_parameter}.")
        keys = tuple(key for key in dataset.parameters if key != repeat_parameter)
        groups: dict[tuple[object, ...], list[int]] = defaultdict(list)
        for index in range(len(dataset.statuses)):
            groups[tuple(dataset.parameters[key][index] for key in keys)].append(index)
        parameters = {
            key: np.asarray([
                dataset.parameters[key][indices[0]]
                for indices in groups.values()
            ])
            for key in keys
        }
        metrics: dict[str, np.ndarray] = {}
        for key, values in dataset.metrics.items():
            metrics[f"{key}_mean"] = np.asarray([
                np.nanmean(values[indices]) for indices in groups.values()
            ])
            metrics[f"{key}_sd"] = np.asarray([
                np.nanstd(values[indices]) for indices in groups.values()
            ])
        return WorkflowDataset(
            parameters=parameters,
            metrics=metrics,
            statuses=np.asarray(["completed"] * len(groups)),
            case_dirs=tuple(Path() for _ in groups),
        )


def plot_signal_to_noise(
    dataset: WorkflowDataset,
    line_parameters: tuple[str, ...],
    point_parameter: str,
    metric: ConvergenceMetric | None = None,
    noise_metric: str = "noise_floor_mean",
    signal_metric: str = "signal_mean",
) -> plt.Figure:
    """Plot grouped signal-versus-noise lines from gathered workflow data."""
    if metric is not None:
        noise_metric = metric.noise_metric
        signal_metric = metric.signal_metric
    for name in (*line_parameters, point_parameter):
        if name not in dataset.parameters:
            raise KeyError(f"Unknown workflow parameter: {name}.")
    if noise_metric not in dataset.metrics or signal_metric not in dataset.metrics:
        raise KeyError("Requested signal or noise metric is unavailable.")
    point_values = dataset.parameters[point_parameter]
    if not np.issubdtype(point_values.dtype, np.number):
        raise TypeError("The plot point parameter must be numeric.")
    groups: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for index in range(len(point_values)):
        key = tuple(dataset.parameters[name][index] for name in line_parameters)
        groups[key].append(index)
    figure, axis = plt.subplots()
    for key, indices in groups.items():
        order = sorted(indices, key=lambda index: point_values[index])
        label = ", ".join(
            f"{name}={value}" for name, value in zip(line_parameters, key)
        )
        axis.plot(
            dataset.metrics[noise_metric][order],
            dataset.metrics[signal_metric][order],
            marker="o",
            label=label,
        )
    axis.set_xlabel(noise_metric)
    axis.set_ylabel(signal_metric)
    axis.legend()
    return figure
