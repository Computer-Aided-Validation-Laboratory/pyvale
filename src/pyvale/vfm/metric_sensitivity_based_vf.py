from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pyvale.vfm.global_virtual_fields_cost_function import global_vf_cost_function
from pyvale.vfm.metrics import BaseMetric, MetricContext, MetricResult
from pyvale.vfm.project_definition import MetricSpec, TestData
from pyvale.vfm.sensitivity_based_virtual_fields import generate_sensitivity_based_virtual_fields
from pyvale.vfm.stress_sensitivity import calculate_stress_sensitivity
from pyvale.vfm.virtual_fields_mesh import (
    generate_virtual_fields_mesh,
    plot_virtual_fields_mesh,
)


@dataclass(slots=True)
class SensitivityBasedVFMetric(BaseMetric):
    """Sensitivity-based virtual-field cost using the current active DOFs.
    
    TODO: I removed defaults from below to prevent hidden bugs. Default definition should be centralised
    and implemented with proper validation.
    """

    options: dict[str, Any] = field(default_factory=dict)
    kind: str = "sensitivity_based_vf"
    virtual_fields_mesh: object | None = None

    def prepare(
        self,
        test_data: TestData,
        context: MetricContext | None = None,
    ) -> None:
        if context is not None and not context.active_dofs:
            self.virtual_fields_mesh = None
            return

        mesh_size = np.asarray(
            self.options.get("virtual_mesh_size"),
            dtype=np.uint32,
        )
        if mesh_size.shape != (2,):
            raise ValueError("SBVF option 'virtual_mesh_size' must have two entries.")

        self.virtual_fields_mesh = generate_virtual_fields_mesh(
            test_data.x,
            test_data.y,
            test_data.specimen_mask,
            test_data.boundary_conditions,
            mesh_size,
        )

        if self.options.get("plot_virtual_mesh", False) and self.options.get("plot_virtual_mesh_path") is not None:
            output_path = _resolve_virtual_mesh_plot_path(
                test_data=test_data,
                plot_path_option=self.options.get("plot_virtual_mesh_path"),
            )
            plot_virtual_fields_mesh(
                data_x=test_data.x,
                data_y=test_data.y,
                specimen_mask=test_data.specimen_mask,
                virtual_fields_mesh=self.virtual_fields_mesh,
                output_path=output_path,
                show=self.options.get("show_virtual_mesh_plot", False),
            )

    def evaluate(
        self,
        stress,
        test_data: TestData,
        context: MetricContext | None = None,
    ) -> MetricResult:
        if context is None:
            raise ValueError("SBVF evaluation requires a metric context.")
        if not context.active_dofs:
            return MetricResult(
                name="sbvf",
                value=0.0,
                details={"num_virtual_fields": 0, "residual_vector": np.zeros(0)},
            )
        if context.base_mechanical_properties is None:
            raise ValueError("SBVF evaluation requires base mechanical properties.")
        if context.parameter_states is None:
            raise ValueError("SBVF evaluation requires parameter states.")
        if self.virtual_fields_mesh is None:
            raise ValueError("SBVF metric was not prepared before evaluation.")

        use_incremental = str(self.options.get("stress_sensitivity", "total")) == "incremental"
        perturbation_factor = float(self.options.get("perturbation_factor", 0.15))
        thickness = float(self.options.get("thickness", test_data.thickness))
        traction_edge = int(self.options.get("traction_edge", 3))
        scaling = bool(self.options.get("scaling", True))
        scale_fraction = float(self.options.get("scale_fraction", 0.3))

        stress_sensitivities = calculate_stress_sensitivity(
            stress_reference=stress,
            test_data=test_data,
            base_mechanical_properties=context.base_mechanical_properties,
            parameter_states=context.parameter_states,
            active_dofs=context.active_dofs,
            perturbation_factor=perturbation_factor,
        )
        virtual_fields = generate_sensitivity_based_virtual_fields(
            stress_sensitivities,
            self.virtual_fields_mesh,
            use_incremental=use_incremental,
        )
        cost_result = global_vf_cost_function(
            stress=stress,
            sensitivity_based_virtual_fields=virtual_fields,
            force=test_data.force,
            area=test_data.area,
            thickness=thickness,
            traction_edge=traction_edge,
            scaling=scaling,
            scale_fraction=scale_fraction,
        )

        return MetricResult(
            name="sbvf",
            value=cost_result.cost,
            details={
                "num_virtual_fields": len(virtual_fields),
                "residual_vector": cost_result.residual_vector,
            },
        )


def build_sensitivity_based_vf_metric(metric_spec: MetricSpec) -> BaseMetric:
    return SensitivityBasedVFMetric(options=metric_spec.options)


def _coerce_option_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_virtual_mesh_plot_path(
    test_data: TestData,
    plot_path_option: Any,
) -> Path:
    if plot_path_option is not None and str(plot_path_option).strip() != "":
        return Path(str(plot_path_option))

    if test_data.source_path is not None:
        return test_data.source_path.with_name("virtual_fields_mesh.png")

    return Path.cwd() / "virtual_fields_mesh.png"
