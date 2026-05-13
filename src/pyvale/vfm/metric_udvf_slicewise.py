from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from pyvale.vfm.global_virtual_fields_cost_function import global_vf_cost_function
from pyvale.vfm.metric_sensitivity_based_vf import SensitivityBasedVirtualFields
from pyvale.vfm.metrics import BaseMetric, MetricContext, MetricResult
from pyvale.vfm.parameterisation_slice import (
    SlicePartition,
    SliceWiseParameterisation,
    slice_partitions_match,
)
from pyvale.vfm.project_definition import MetricSpec, TestData


@dataclass(slots=True)
class UDVFSlicewiseMetric(BaseMetric):
    options: dict[str, Any] = field(default_factory=dict)
    partition: SlicePartition | None = None
    virtual_component: str | None = None
    traction_edge: int | None = None
    kind: str = "udvf_slicewise"

    def prepare(
        self,
        test_data: TestData,
        context: MetricContext | None = None,
    ) -> None:
        if context is None or context.parameter_states is None:
            raise ValueError("Slice-wise UDVF preparation requires parameter states.")

        self.partition = _extract_common_partition(context)
        self.virtual_component = _resolve_virtual_component(
            self.options,
            self.partition,
        )
        self.traction_edge = int(
            self.options.get(
                "traction_edge",
                3 if self.virtual_component == "xx" else 0,
            )
        )

    def evaluate(
        self,
        stress: npt.NDArray[np.float64],
        test_data: TestData,
        context: MetricContext | None = None,
    ) -> MetricResult:
        if self.partition is None or self.virtual_component is None or self.traction_edge is None:
            raise ValueError("Slice-wise UDVF metric was not prepared before evaluation.")

        slice_virtual_fields: dict[str, SensitivityBasedVirtualFields] = {}
        for slice_index, (slice_mask, slice_width) in enumerate(
            zip(
                self.partition.slice_masks,
                self.partition.slice_widths,
                strict=True,
            )
        ):
            slice_virtual_fields[f"slice_{slice_index}"] = _build_slice_virtual_field(
                stress_shape=stress.shape,
                slice_mask=slice_mask,
                virtual_component=self.virtual_component,
                slice_width=float(slice_width),
                traction_edge=self.traction_edge,
            )

        cost_result = global_vf_cost_function(
            stress=stress,
            sensitivity_based_virtual_fields=slice_virtual_fields,
            force=test_data.force,
            area=test_data.area,
            thickness=float(self.options.get("thickness", test_data.thickness)),
            traction_edge=self.traction_edge,
            scaling=bool(self.options.get("scaling", False)),
            scale_fraction=float(self.options.get("scale_fraction", 0.05)),
        )

        return MetricResult(
            name="udvf_slicewise",
            value=cost_result.cost,
            details={
                "residual_vector": cost_result.residual_vector,
                "num_slices": self.partition.num_slices,
                "slice_widths": self.partition.slice_widths.copy(),
                "virtual_component": self.virtual_component,
                "traction_edge": self.traction_edge,
            },
        )

    def evaluate_single_slice(
        self,
        stress: npt.NDArray[np.float64],
        test_data: TestData,
        slice_width: float,
        slice_index: int,
    ) -> MetricResult:
        if self.virtual_component is None or self.traction_edge is None:
            raise ValueError("Slice-wise UDVF metric was not prepared before slice evaluation.")

        slice_virtual_field = _build_slice_virtual_field(
            stress_shape=stress.shape,
            slice_mask=np.asarray(test_data.specimen_mask, dtype=bool),
            virtual_component=self.virtual_component,
            slice_width=float(slice_width),
            traction_edge=self.traction_edge,
        )
        cost_result = global_vf_cost_function(
            stress=stress,
            sensitivity_based_virtual_fields={f"slice_{slice_index}": slice_virtual_field},
            force=test_data.force,
            area=test_data.area,
            thickness=float(self.options.get("thickness", test_data.thickness)),
            traction_edge=self.traction_edge,
            scaling=bool(self.options.get("scaling", False)),
            scale_fraction=float(self.options.get("scale_fraction", 0.05)),
        )

        return MetricResult(
            name=f"udvf_slicewise.slice_{slice_index}",
            value=cost_result.cost,
            details={
                "residual_vector": cost_result.residual_vector,
                "slice_width": float(slice_width),
                "virtual_component": self.virtual_component,
                "traction_edge": self.traction_edge,
            },
        )


def build_udvf_slicewise_metric(metric_spec: MetricSpec) -> BaseMetric:
    return UDVFSlicewiseMetric(options=metric_spec.options)


def _extract_common_partition(context: MetricContext) -> SlicePartition:
    if context.parameter_states is None:
        raise ValueError("Slice-wise UDVF requires parameter states in the metric context.")

    common_partition: SlicePartition | None = None
    for parameter_state in context.parameter_states.values():
        for parameterisation in parameter_state.parameterisations:
            if not isinstance(parameterisation, SliceWiseParameterisation):
                continue
            if parameterisation.partition is None:
                raise ValueError("Slicewise parameterisations must be prepared before metric preparation.")

            if common_partition is None:
                common_partition = parameterisation.partition
                continue

            if not slice_partitions_match(common_partition, parameterisation.partition):
                raise ValueError(
                    "All slicewise parameterisations in a phase must share the same "
                    "slice layout."
                )

    if common_partition is None:
        raise ValueError(
            "Slice-wise UDVF requires at least one slicewise parameterisation."
        )

    return common_partition


def _resolve_virtual_component(
    options: dict[str, Any],
    partition: SlicePartition,
) -> str:
    requested_component = options.get("virtual_component")
    if requested_component is not None:
        component = str(requested_component).strip().lower()
        if component in {"xx", "yy"}:
            return component
        raise ValueError(
            "Slice-wise UDVF option 'virtual_component' must be 'xx' or 'yy'."
        )

    return "yy" if partition.constant_coordinate == "x" else "xx"


def _build_slice_virtual_field(
    stress_shape: tuple[int, ...],
    slice_mask: npt.NDArray[np.bool_],
    virtual_component: str,
    slice_width: float,
    traction_edge: int,
) -> SensitivityBasedVirtualFields:
    if len(stress_shape) != 4:
        raise ValueError("stress must have shape (timesteps, components, y, x).")

    component_index = 0 if virtual_component == "xx" else 1
    n_timesteps = stress_shape[0]

    virtual_strain = np.zeros(stress_shape, dtype=np.float64)
    virtual_strain[:, component_index, :, :] = slice_mask[np.newaxis, :, :]

    edge_displacement = np.zeros((n_timesteps, 2, 4), dtype=np.float64)
    edge_displacement[:, component_index, traction_edge] = slice_width

    full_displacement = np.full(
        (n_timesteps, 2, stress_shape[2], stress_shape[3]),
        np.nan,
        dtype=np.float64,
    )

    return SensitivityBasedVirtualFields(
        virtual_strain=virtual_strain,
        edge_displacement=edge_displacement,
        full_displacement=full_displacement,
    )
