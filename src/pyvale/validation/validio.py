# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Save and load validation records as NumPy archives."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from numpy.lib.npyio import NpzFile

from pyvale.validation.imetric import (
    EMetricOut,
    EMetricType,
    IValidMetric,
)
from pyvale.validation.metrics import ERelErrNorm
from pyvale.validation.metricstrategy import (
    MetricAVM,
    MetricCVM,
    MetricKS,
    MetricMAVM,
    MetricRelErr,
    MetricRMSE,
)
from pyvale.validation.validtypes import (
    EEpisOffset,
    EMAVMDuplicates,
    EpistemicOffsets,
    EValidAxis,
    EValidGeometry,
    EValidResAxis,
    MetricField,
    ValidArray,
    ValidCase,
    ValidMetricExtrema,
    ValidObservable,
    ValidResult,
    ValidSampleInputs,
)

_SCHEMA_VERSION = 1
_METRIC_CLASSES: dict[EMetricType, type[IValidMetric]] = {
    EMetricType.AVM: MetricAVM,
    EMetricType.CVM: MetricCVM,
    EMetricType.KS: MetricKS,
    EMetricType.MAVM: MetricMAVM,
    EMetricType.RELATIVE_ERROR: MetricRelErr,
    EMetricType.RMSE: MetricRMSE,
}


def _store_array(arrays: dict[str, np.ndarray], array: np.ndarray) -> str:
    if array.dtype.hasobject:
        raise TypeError("Validation archives cannot store object arrays.")
    key = f"array_{len(arrays)}"
    arrays[key] = array
    return key


def _save_archive(
    path: str | Path,
    kind: str,
    metadata: dict[str, object],
    arrays: dict[str, np.ndarray],
) -> None:
    destination = Path(path)
    manifest = np.array(
        json.dumps({"version": _SCHEMA_VERSION, "kind": kind, "metadata": metadata})
    )
    with NamedTemporaryFile(dir=destination.parent, delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        try:
            np.savez(temp_file, manifest=manifest, **arrays)
            temp_file.close()
            temp_path.replace(destination)
        finally:
            temp_file.close()
            temp_path.unlink(missing_ok=True)


def _load_metadata(archive: NpzFile, kind: str) -> dict:
    if "manifest" not in archive:
        raise ValueError("Validation archive has no manifest.")
    manifest = json.loads(archive["manifest"].item())
    if manifest.get("version") != _SCHEMA_VERSION:
        raise ValueError("Unsupported validation archive schema version.")
    if manifest.get("kind") != kind:
        raise ValueError(f"Expected a {kind} validation archive.")
    return manifest["metadata"]


def _pack_observable(
    observable: ValidObservable, arrays: dict[str, np.ndarray]
) -> dict[str, object]:
    return {
        "components": observable.components,
        "geometry": observable.geometry.name,
        "coords": None
        if observable.coords is None
        else _store_array(arrays, observable.coords),
        "times": None
        if observable.times is None
        else _store_array(arrays, observable.times),
        "connect": (
            None
            if observable.connect is None
            else {
                tag: _store_array(arrays, array)
                for tag, array in observable.connect.items()
            }
        ),
        "phys_units": observable.phys_units,
        "coord_units": observable.coord_units,
        "coord_frame": observable.coord_frame,
        "time_units": observable.time_units,
    }


def _unpack_observable(metadata: dict, archive: NpzFile) -> ValidObservable:
    return ValidObservable(
        components=tuple(metadata["components"]),
        geometry=EValidGeometry[metadata["geometry"]],
        coords=None if metadata["coords"] is None else archive[metadata["coords"]],
        times=None if metadata["times"] is None else archive[metadata["times"]],
        connect=(
            None
            if metadata["connect"] is None
            else {tag: archive[key] for tag, key in metadata["connect"].items()}
        ),
        phys_units=metadata["phys_units"],
        coord_units=metadata["coord_units"],
        coord_frame=metadata["coord_frame"],
        time_units=metadata["time_units"],
    )


def _pack_quantity(
    quantity: ValidArray, arrays: dict[str, np.ndarray]
) -> dict[str, object]:
    offsets = quantity.offsets
    return {
        "data": _store_array(arrays, quantity.data),
        "axes": [axis.name for axis in quantity.axes],
        "offsets": (
            None
            if offsets is None
            else {
                "axes": [axis.name for axis in offsets.axes],
                "lower": _store_array(arrays, np.asarray(offsets.lower)),
                "upper": _store_array(arrays, np.asarray(offsets.upper)),
            }
        ),
    }


def _unpack_quantity(metadata: dict, archive: NpzFile) -> ValidArray:
    offsets = metadata["offsets"]
    return ValidArray(
        archive[metadata["data"]],
        tuple(EValidAxis[name] for name in metadata["axes"]),
        offsets=(
            None
            if offsets is None
            else EpistemicOffsets(
                lower=archive[offsets["lower"]],
                upper=archive[offsets["upper"]],
                axes=tuple(EValidAxis[name] for name in offsets["axes"]),
            )
        ),
    )


def _pack_inputs(
    inputs: ValidSampleInputs | None, arrays: dict[str, np.ndarray]
) -> dict[str, object] | None:
    if inputs is None:
        return None
    return {
        "parameters": (
            None
            if inputs.input_params is None
            else {
                tag: _pack_quantity(value, arrays)
                for tag, value in inputs.input_params.items()
            }
        ),
        "run_ids": None
        if inputs.run_ids is None
        else _pack_quantity(inputs.run_ids, arrays),
    }


def _unpack_inputs(metadata: dict | None, archive: NpzFile) -> ValidSampleInputs | None:
    if metadata is None:
        return None
    parameters = metadata["parameters"]
    run_ids = metadata["run_ids"]
    return ValidSampleInputs(
        input_params=(
            None
            if parameters is None
            else {
                tag: _unpack_quantity(value, archive)
                for tag, value in parameters.items()
            }
        ),
        run_ids=None if run_ids is None else _unpack_quantity(run_ids, archive),
    )


def save_valid_case(path: str | Path, valid_case: ValidCase) -> None:
    """Save prepared samples, axes, observables and optional sample inputs."""
    arrays: dict[str, np.ndarray] = {}
    metadata = {
        "observables": {
            tag: _pack_observable(value, arrays)
            for tag, value in valid_case.observables.items()
        },
        "sim": {
            tag: _pack_quantity(value, arrays) for tag, value in valid_case.sim.items()
        },
        "exp": {
            tag: _pack_quantity(value, arrays) for tag, value in valid_case.exp.items()
        },
        "sim_inputs": _pack_inputs(valid_case.sim_inputs, arrays),
        "exp_inputs": _pack_inputs(valid_case.exp_inputs, arrays),
    }
    _save_archive(path, "case", metadata, arrays)


def load_valid_case(path: str | Path) -> ValidCase:
    """Load the prepared case, including the samples needed for ECDF plots."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = _load_metadata(archive, "case")
        return ValidCase(
            {
                tag: _unpack_observable(value, archive)
                for tag, value in metadata["observables"].items()
            },
            {
                tag: _unpack_quantity(value, archive)
                for tag, value in metadata["sim"].items()
            },
            {
                tag: _unpack_quantity(value, archive)
                for tag, value in metadata["exp"].items()
            },
            sim_inputs=_unpack_inputs(metadata["sim_inputs"], archive),
            exp_inputs=_unpack_inputs(metadata["exp_inputs"], archive),
        )


def _pack_metric_config(kind: EMetricType, metric: IValidMetric) -> dict[str, object]:
    if type(metric) is not _METRIC_CLASSES.get(kind):
        raise TypeError(
            "Validation archives support only built-in metric configurations."
        )
    if isinstance(metric, MetricMAVM):
        return {
            "alpha": float(metric.alpha),
            "duplicates": metric.duplicates.name,
            "tol": float(metric.tol),
        }
    if isinstance(metric, MetricRelErr):
        return {
            "normalisation": metric.normalisation.name,
            "constant": None if metric.constant is None else float(metric.constant),
        }
    return {}


def _unpack_metric_config(kind: EMetricType, config: dict) -> IValidMetric:
    if kind is EMetricType.MAVM:
        return MetricMAVM(
            alpha=config["alpha"],
            duplicates=EMAVMDuplicates[config["duplicates"]],
            tol=config["tol"],
        )
    if kind is EMetricType.RELATIVE_ERROR:
        return MetricRelErr(
            normalisation=ERelErrNorm[config["normalisation"]],
            constant=config["constant"],
        )
    return _METRIC_CLASSES[kind]()


def _pack_outputs(
    outputs: dict[EMetricOut, np.ndarray], arrays: dict[str, np.ndarray]
) -> dict[str, str]:
    return {
        output.name: _store_array(arrays, values) for output, values in outputs.items()
    }


def _unpack_outputs(
    outputs: dict[str, str], archive: NpzFile
) -> dict[EMetricOut, np.ndarray]:
    return {EMetricOut[name]: archive[key] for name, key in outputs.items()}


def save_valid_result(path: str | Path, result: ValidResult) -> None:
    """Save metric fields, observable metadata and built-in metric settings."""
    arrays: dict[str, np.ndarray] = {}
    fields = {}
    for tag, metrics in result.metrics.items():
        fields[tag] = {}
        for kind, metric_field in metrics.items():
            epis_offsets = metric_field.epis_offsets
            fields[tag][kind.name] = {
                "axes": [axis.name for axis in metric_field.axes],
                "outputs": _pack_outputs(metric_field.metric_vals, arrays),
                "scenarios": (
                    None
                    if epis_offsets is None
                    else {
                        axis.name: [offset.name for offset in offsets]
                        for axis, offsets in epis_offsets.items()
                    }
                ),
            }
    metadata = {
        "observables": {
            tag: _pack_observable(value, arrays)
            for tag, value in result.observables.items()
        },
        "metrics": fields,
        "metric_configs": {
            kind.name: _pack_metric_config(kind, metric)
            for kind, metric in result.metric_configs.items()
        },
    }
    _save_archive(path, "result", metadata, arrays)


def load_valid_result(path: str | Path) -> ValidResult:
    """Load previously calculated metric fields without running validation again."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = _load_metadata(archive, "result")
        metrics: dict[str, dict[EMetricType, MetricField]] = {}
        for tag, fields in metadata["metrics"].items():
            metrics[tag] = {}
            for kind, stored in fields.items():
                epis_offsets = stored["scenarios"]
                metrics[tag][EMetricType[kind]] = MetricField(
                    axes=tuple(EValidResAxis[name] for name in stored["axes"]),
                    metric_vals=_unpack_outputs(stored["outputs"], archive),
                    epis_offsets=(
                        None
                        if epis_offsets is None
                        else {
                            EValidResAxis[axis]: tuple(
                                EEpisOffset[name] for name in offsets
                            )
                            for axis, offsets in epis_offsets.items()
                        }
                    ),
                )
        return ValidResult(
            {
                tag: _unpack_observable(value, archive)
                for tag, value in metadata["observables"].items()
            },
            metrics,
            {
                EMetricType[kind]: _unpack_metric_config(EMetricType[kind], config)
                for kind, config in metadata["metric_configs"].items()
            },
        )


def save_metric_extrema(path: str | Path, extrema: ValidMetricExtrema) -> None:
    """Save extrema, their controlling indices and the associated axis selections."""
    arrays: dict[str, np.ndarray] = {}
    metadata = {
        "axes": [axis.name for axis in extrema.axes],
        "reduced_axes": [axis.name for axis in extrema.reduced_axes],
        "selectors": {
            axis.name: int(index) for axis, index in extrema.selectors.items()
        },
        "minimum": _pack_outputs(extrema.minimum, arrays),
        "maximum": _pack_outputs(extrema.maximum, arrays),
        "minimum_indices": _pack_outputs(extrema.minimum_indices, arrays),
        "maximum_indices": _pack_outputs(extrema.maximum_indices, arrays),
    }
    _save_archive(path, "extrema", metadata, arrays)


def load_metric_extrema(path: str | Path) -> ValidMetricExtrema:
    """Load a profile. Use the original result for metadata and case for samples."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = _load_metadata(archive, "extrema")
        return ValidMetricExtrema(
            axes=tuple(EValidResAxis[name] for name in metadata["axes"]),
            reduced_axes=tuple(
                EValidResAxis[name] for name in metadata["reduced_axes"]
            ),
            minimum=_unpack_outputs(metadata["minimum"], archive),
            maximum=_unpack_outputs(metadata["maximum"], archive),
            minimum_indices=_unpack_outputs(metadata["minimum_indices"], archive),
            maximum_indices=_unpack_outputs(metadata["maximum_indices"], archive),
            selectors={
                EValidResAxis[axis]: index
                for axis, index in metadata["selectors"].items()
            },
        )
