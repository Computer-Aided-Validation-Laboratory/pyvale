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
    EMAVMDupes,
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
    EMetricType.REL_ERR: MetricRelErr,
    EMetricType.RMSE: MetricRMSE,
}


def _store_array(arrays: dict[str, np.ndarray], array: np.ndarray) -> str:
    """Register an array under the next available archive key.

    Parameters
    ----------
    arrays : dict[str, ndarray]
        Mapping from archive keys to stored NumPy arrays.
    array : ndarray
        NumPy array of arbitrary rank. Object arrays are unsupported.

    Returns
    -------
    str
        Assigned key used by the metadata to refer to the array.
    """
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
    """Write the versioned manifest and arrays and publish the archive atomically.

    Parameters
    ----------
    path : str or Path
        Path to the NPZ archive.
    kind : str
        Archive or metric identifier for the operation.
    metadata : dict[str, object]
        JSON-compatible metadata describing the corresponding arrays.
    arrays : dict[str, ndarray]
        Mapping from archive keys to stored NumPy arrays.
    """
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
    """Read the JSON manifest and check the archive kind and version.

    Parameters
    ----------
    archive : NpzFile
        Open NPZ archive with a JSON manifest and named arrays.
    kind : str
        Archive or metric identifier for the operation.

    Returns
    -------
    dict
        Metadata payload from the checked manifest.
    """
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
    """Encode observable metadata and register its coordinate, time and connectivity arrays.

    Parameters
    ----------
    observable : ValidObservable
        Physical observable metadata to encode.
    arrays : dict[str, ndarray]
        Mapping from archive keys to stored NumPy arrays.

    Returns
    -------
    dict[str, object]
        JSON-compatible observable metadata with keys for stored arrays.
    """
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
    """Reconstruct observable metadata from the manifest and array data.

    Parameters
    ----------
    metadata : dict
        JSON-compatible metadata describing the corresponding arrays.
    archive : NpzFile
        Open NPZ archive with a JSON manifest and named arrays.

    Returns
    -------
    ValidObservable
        Observable with coordinate, connectivity and time axes restored.
    """
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
    """Encode observation axes, values and coherent offsets.

    Parameters
    ----------
    quantity : ValidArray
        Observations paired with their canonical named axes.
    arrays : dict[str, ndarray]
        Mapping from archive keys to stored NumPy arrays.

    Returns
    -------
    dict[str, object]
        Stored array keys and JSON-compatible metadata for the quantity.
    """
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
    """Load a named array and its optional coherent offsets.

    Parameters
    ----------
    metadata : dict
        JSON-compatible metadata describing the corresponding arrays.
    archive : NpzFile
        Open NPZ archive with a JSON manifest and named arrays.

    Returns
    -------
    ValidArray
        Array with the original dimension order and offset metadata.
    """
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
    """Encode optional input parameter and run ID arrays.

    Parameters
    ----------
    inputs : ValidSampleInputs or None
        Optional sampled input parameters and run identifiers.
    arrays : dict[str, ndarray]
        Mapping from archive keys to stored NumPy arrays.

    Returns
    -------
    dict[str, object] or None
        Provenance metadata referring to archive arrays, or None.
    """
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
    """Load optional input parameter and run ID arrays.

    Parameters
    ----------
    metadata : dict or None
        JSON-compatible metadata describing the corresponding arrays.
    archive : NpzFile
        Open NPZ archive with a JSON manifest and named arrays.

    Returns
    -------
    ValidSampleInputs or None
        Reconstructed sample inputs, or None when no inputs were stored.
    """
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
    """Save prepared samples, axes, observables and optional sample inputs.

    Parameters
    ----------
    path : str or Path
        Path to the NPZ archive.
    valid_case : ValidCase
        Aligned simulation and experiment observations with their named axes.
    """
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
    """Load the prepared case, including the samples needed for ECDF plots.

    Parameters
    ----------
    path : str or Path
        Path to the NPZ archive.

    Returns
    -------
    ValidCase
        Case with observable metadata, named observation arrays and optional
        provenance restored.
    """

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
    """Encode a built-in metric's calculation settings.

    Parameters
    ----------
    kind : EMetricType
        Archive or metric identifier for the operation.
    metric : IValidMetric
        Metric to select or evaluate.

    Returns
    -------
    dict[str, object]
        JSON-compatible options for reconstructing the metric.
    """
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
    """Reconstruct a built-in metric strategy from its stored settings.

    Parameters
    ----------
    kind : EMetricType
        Archive or metric identifier for the operation.
    config : dict
        Stored options for the selected metric kind.

    Returns
    -------
    IValidMetric
        Metric configured as when the result was saved.
    """
    if kind is EMetricType.MAVM:
        return MetricMAVM(
            alpha=config["alpha"],
            duplicates=EMAVMDupes[config["duplicates"]],
            tol=config["tol"],
        )

    if kind is EMetricType.REL_ERR:
        return MetricRelErr(
            normalisation=ERelErrNorm[config["normalisation"]],
            constant=config["constant"],
        )

    return _METRIC_CLASSES[kind]()


def _pack_outputs(
    outputs: dict[EMetricOut, np.ndarray], arrays: dict[str, np.ndarray]
) -> dict[str, str]:
    """Register metric output arrays without changing their axes.

    Parameters
    ----------
    outputs : dict[EMetricOut, ndarray]
        Metric outputs keyed by output tag or stored tag name.
    arrays : dict[str, ndarray]
        Mapping from archive keys to stored NumPy arrays.

    Returns
    -------
    dict[str, str]
        Stored output tag names mapped to array keys.
    """
    return {
        output.name: _store_array(arrays, values) for output, values in outputs.items()
    }


def _unpack_outputs(
    outputs: dict[str, str], archive: NpzFile
) -> dict[EMetricOut, np.ndarray]:
    """Load output arrays with their original shapes.

    Parameters
    ----------
    outputs : dict[str, str]
        Metric outputs keyed by output tag or stored tag name.
    archive : NpzFile
        Open NPZ archive with a JSON manifest and named arrays.

    Returns
    -------
    dict[EMetricOut, ndarray]
        Metric output tags mapped to arrays from the archive.
    """
    return {EMetricOut[name]: archive[key] for name, key in outputs.items()}


def _pack_metric_kind(kind: EMetricType) -> str:
    # Keep the schema v1 key when the public enum member is renamed.
    """Encode the archive name of a metric kind.

    Parameters
    ----------
    kind : EMetricType
        Archive or metric identifier for the operation.

    Returns
    -------
    str
        Schema-compatible metric name.
    """
    return "RELATIVE_ERROR" if kind is EMetricType.REL_ERR else kind.name


def _unpack_metric_kind(name: str) -> EMetricType:
    """Resolve a stored metric name to its enum member.

    Parameters
    ----------
    name : str
        Stored name of a metric kind.

    Returns
    -------
    EMetricType
        Metric type represented by the archive name.
    """
    return EMetricType.REL_ERR if name == "RELATIVE_ERROR" else EMetricType[name]


def save_valid_result(path: str | Path, result: ValidResult) -> None:
    """Save metric fields, observable metadata and built-in metric settings.

    Parameters
    ----------
    path : str or Path
        Path to the NPZ archive.
    result : ValidResult
        Validation results containing the requested observable and metric.
    """
    arrays: dict[str, np.ndarray] = {}
    fields = {}

    for tag, metrics in result.metrics.items():
        fields[tag] = {}

        for kind, metric_field in metrics.items():
            epis_offsets = metric_field.epis_offsets
            fields[tag][_pack_metric_kind(kind)] = {
                "axes": [axis.name for axis in metric_field.axes],
                "outputs": _pack_outputs(metric_field.metric_vals, arrays),
                "epis_offsets": (
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
            _pack_metric_kind(kind): _pack_metric_config(kind, metric)
            for kind, metric in result.metric_configs.items()
        },
    }
    _save_archive(path, "result", metadata, arrays)


def load_valid_result(path: str | Path) -> ValidResult:
    """Load previously calculated metric fields without running validation again.

    Parameters
    ----------
    path : str or Path
        Path to the NPZ archive.

    Returns
    -------
    ValidResult
        Metric fields, named result axes and available built-in metric settings.
    """

    with np.load(path, allow_pickle=False) as archive:
        metadata = _load_metadata(archive, "result")
        metrics: dict[str, dict[EMetricType, MetricField]] = {}

        for tag, fields in metadata["metrics"].items():
            metrics[tag] = {}

            for kind, stored in fields.items():
                # Older result archives stored these offset labels under "scenarios".
                epis_offsets = (
                    stored["epis_offsets"]
                    if "epis_offsets" in stored
                    else stored["scenarios"]
                )
                metrics[tag][_unpack_metric_kind(kind)] = MetricField(
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

        metric_configs = {}

        for name, config in metadata["metric_configs"].items():
            kind = _unpack_metric_kind(name)
            metric_configs[kind] = _unpack_metric_config(kind, config)

        return ValidResult(
            {
                tag: _unpack_observable(value, archive)
                for tag, value in metadata["observables"].items()
            },
            metrics,
            metric_configs,
        )


def save_metric_extrema(path: str | Path, extrema: ValidMetricExtrema) -> None:
    """Save extrema, their controlling indices and the associated axis selections.

    Parameters
    ----------
    path : str or Path
        Path to the NPZ archive.
    extrema : ValidMetricExtrema
        Extrema values and controlling indices on their retained axes.
    """
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
    """Load a profile. Use the original result for metadata and case for samples.

    Parameters
    ----------
    path : str or Path
        Path to the NPZ archive.

    Returns
    -------
    ValidMetricExtrema
        Extrema arrays and original comparison indices with their axis labels.
    """

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
