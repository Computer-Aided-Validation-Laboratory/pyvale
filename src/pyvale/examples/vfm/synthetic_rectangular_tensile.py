"""Synthetic tensile data shared by the VFM examples and tests.

The specimen is rectangular, under plane stress, and loaded monotonically in
the x direction. Elastic properties and linear hardening are homogeneous,
while yield strength may vary between longitudinal slices. All geometry,
loading, strain, stress, and material fields are generated in memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pyvale.vfm.constparam import ConstitutiveParameter
from pyvale.vfm.experimentdata import (
    BoundaryConditions,
    Edge,
    EdgeConditions,
    EEdgeCondition,
    ExperimentData,
    SpecimenGeometry,
)
from pyvale.vfm.roi import RoiDefinition, RoiShape, VfmRegionOfInterest
from pyvale.vfm.slicewise_utils import (
    SliceConfig,
    resolve_cell_aligned_slice_boundaries,
)


SYNTHETIC_PARAMETER_NAMES: tuple[str, ...] = (
    "elastic_modulus",
    "poissons_ratio",
    "yield_strength",
    "hardening_modulus",
)


@dataclass(slots=True, frozen=True)
class SyntheticTensileMaterial:
    """Material properties used by the synthetic tensile case."""

    elastic_modulus: float = 210_000.0
    poissons_ratio: float = 0.3
    yield_strengths: tuple[float, ...] = (180.0, 230.0, 280.0, 330.0)
    hardening_modulus: float = 7_000.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.elastic_modulus) or self.elastic_modulus <= 0.0:
            raise ValueError("elastic_modulus must be finite and positive")
        if (
            not np.isfinite(self.poissons_ratio)
            or self.poissons_ratio < 0.0
            or self.poissons_ratio >= 0.5
        ):
            raise ValueError("poissons_ratio must be finite and in [0, 0.5)")
        if not self.yield_strengths or any(
            not np.isfinite(value) or value <= 0.0
            for value in self.yield_strengths
        ):
            raise ValueError(
                "yield_strengths must contain positive finite values"
            )
        if (
            not np.isfinite(self.hardening_modulus)
            or self.hardening_modulus <= 0.0
        ):
            raise ValueError(
                "hardening_modulus must be finite and positive"
            )


@dataclass(slots=True, frozen=True)
class SyntheticTensileCase:
    """Synthetic experiment and its known reference solution."""

    experiment_data: ExperimentData
    known_parameter_maps: dict[str, npt.NDArray[np.float64]]
    known_stress: npt.NDArray[np.float64]
    prescribed_longitudinal_stress: npt.NDArray[np.float64]
    slice_boundaries: npt.NDArray[np.float64]
    material: SyntheticTensileMaterial


def build_synthetic_tensile_case(
    *,
    num_grid_rows: int = 4,
    num_grid_columns: int = 16,
    length: float = 16.0,
    height: float = 4.0,
    thickness: float = 0.8,
    prescribed_longitudinal_stress: npt.ArrayLike | None = None,
    material: SyntheticTensileMaterial = SyntheticTensileMaterial(),
) -> SyntheticTensileCase:
    """Build a mechanically consistent rectangular tensile experiment.

    Coordinates are cell centres on a regular grid. Their constant pixel areas
    sum to the physical specimen area, keeping SBVF internal virtual work
    consistent with the applied boundary force. The number of grid columns
    must be divisible by the number of yield-strength regions so their
    boundaries coincide with support-cell edges.
    """

    _validate_case_dimensions(
        num_grid_rows=num_grid_rows,
        num_grid_columns=num_grid_columns,
        length=length,
        height=height,
        thickness=thickness,
        num_slices=len(material.yield_strengths),
    )
    stress_history = _resolve_prescribed_longitudinal_stress(
        prescribed_longitudinal_stress
    )

    delta_x = length / num_grid_columns
    delta_y = height / num_grid_rows
    x_axis = (
        np.arange(num_grid_columns, dtype=np.float64) + 0.5
    ) * delta_x
    y_axis = (
        np.arange(num_grid_rows, dtype=np.float64) + 0.5
    ) * delta_y
    x, y = np.meshgrid(x_axis, y_axis)
    pixel_area = np.full(
        x.shape,
        delta_x * delta_y,
        dtype=np.float64,
    )

    region_of_interest = VfmRegionOfInterest.from_definition(
        RoiDefinition(
            shapes=(
                RoiShape(
                    shape_type="rectangle",
                    index=0,
                    is_cutting=False,
                    rectangle=(0.0, 0.0, float(length), float(height)),
                ),
            )
        )
    )
    specimen_geometry = SpecimenGeometry(
        x=x,
        y=y,
        pixel_area=pixel_area,
        thickness=float(thickness),
        region_of_interest=region_of_interest,
    )
    slice_boundaries = resolve_cell_aligned_slice_boundaries(
        specimen_geometry,
        SliceConfig(
            axis="x",
            num_slices=len(material.yield_strengths),
        ),
    )
    known_parameter_maps = _build_parameter_maps(
        x,
        slice_boundaries=slice_boundaries,
        material=material,
    )
    strain = _build_uniaxial_strain_history(
        stress_history,
        known_parameter_maps["yield_strength"],
        material=material,
    )

    known_stress = np.zeros_like(strain)
    known_stress[:, 0] = stress_history[:, np.newaxis, np.newaxis]
    force_x = stress_history * height * thickness
    force = np.column_stack((force_x, np.zeros_like(force_x)))
    timesteps = np.linspace(
        0.1,
        1.0,
        stress_history.size,
        dtype=np.float64,
    )

    experiment_data = ExperimentData(
        strain=strain,
        specimen_geometry=specimen_geometry,
        boundary_conditions=BoundaryConditions(
            edge_conditions=_tensile_edge_conditions(),
            force=force,
        ),
        timesteps=timesteps,
    )
    return SyntheticTensileCase(
        experiment_data=experiment_data,
        known_parameter_maps=known_parameter_maps,
        known_stress=known_stress,
        prescribed_longitudinal_stress=stress_history,
        slice_boundaries=slice_boundaries,
        material=material,
    )


def build_synthetic_identification_parameters(
    map_size: npt.NDArray[np.uint32],
    *,
    material: SyntheticTensileMaterial = SyntheticTensileMaterial(),
    initial_yield_strength: float = 250.0,
) -> dict[str, ConstitutiveParameter]:
    """Return bounded starting parameters suitable for the synthetic case."""

    return {
        "elastic_modulus": ConstitutiveParameter(
            material.elastic_modulus,
            100_000.0,
            300_000.0,
            map_size,
        ),
        "poissons_ratio": ConstitutiveParameter(
            material.poissons_ratio,
            0.2,
            0.4,
            map_size,
        ),
        "yield_strength": ConstitutiveParameter(
            initial_yield_strength,
            100.0,
            500.0,
            map_size,
        ),
        "hardening_modulus": ConstitutiveParameter(
            material.hardening_modulus,
            1_000.0,
            20_000.0,
            map_size,
        ),
    }


def _tensile_edge_conditions() -> EdgeConditions:
    return EdgeConditions(
        min_x_edge=Edge(EEdgeCondition.Fixed, EEdgeCondition.Free),
        max_x_edge=Edge(EEdgeCondition.Traction, EEdgeCondition.Free),
        min_y_edge=Edge(EEdgeCondition.Free, EEdgeCondition.Free),
        max_y_edge=Edge(EEdgeCondition.Free, EEdgeCondition.Free),
    )


def _build_parameter_maps(
    x: npt.NDArray[np.float64],
    *,
    slice_boundaries: npt.NDArray[np.float64],
    material: SyntheticTensileMaterial,
) -> dict[str, npt.NDArray[np.float64]]:
    slice_indices = (
        np.searchsorted(slice_boundaries, x, side="right") - 1
    )
    slice_indices = np.clip(
        slice_indices,
        0,
        len(material.yield_strengths) - 1,
    )
    yield_values = np.asarray(
        material.yield_strengths,
        dtype=np.float64,
    )

    return {
        "elastic_modulus": np.full(
            x.shape,
            material.elastic_modulus,
            dtype=np.float64,
        ),
        "poissons_ratio": np.full(
            x.shape,
            material.poissons_ratio,
            dtype=np.float64,
        ),
        "yield_strength": yield_values[slice_indices],
        "hardening_modulus": np.full(
            x.shape,
            material.hardening_modulus,
            dtype=np.float64,
        ),
    }


def _build_uniaxial_strain_history(
    prescribed_longitudinal_stress: npt.NDArray[np.float64],
    yield_strength: npt.NDArray[np.float64],
    *,
    material: SyntheticTensileMaterial,
) -> npt.NDArray[np.float64]:
    stress = prescribed_longitudinal_stress[:, np.newaxis, np.newaxis]
    equivalent_plastic_strain = np.maximum(
        (stress - yield_strength[np.newaxis])
        / material.hardening_modulus,
        0.0,
    )
    strain = np.zeros(
        (
            prescribed_longitudinal_stress.size,
            3,
            *yield_strength.shape,
        ),
        dtype=np.float64,
    )
    strain[:, 0] = (
        stress / material.elastic_modulus
        + equivalent_plastic_strain
    )
    strain[:, 1] = (
        -material.poissons_ratio
        * stress
        / material.elastic_modulus
        - 0.5 * equivalent_plastic_strain
    )
    return strain


def _resolve_prescribed_longitudinal_stress(
    prescribed_longitudinal_stress: npt.ArrayLike | None,
) -> npt.NDArray[np.float64]:
    target = (
        np.asarray(
            prescribed_longitudinal_stress,
            dtype=np.float64,
        )
        if prescribed_longitudinal_stress is not None
        else np.linspace(40.0, 420.0, 9, dtype=np.float64)
    )
    if target.ndim != 1 or target.size < 2:
        raise ValueError(
            "prescribed_longitudinal_stress must be one-dimensional "
            "with at least two values"
        )
    if not np.all(np.isfinite(target)) or np.any(target < 0.0):
        raise ValueError(
            "prescribed_longitudinal_stress must contain finite "
            "non-negative values"
        )
    if np.any(np.diff(target) <= 0.0):
        raise ValueError(
            "prescribed_longitudinal_stress must be strictly increasing"
        )
    return target.copy()


def _validate_case_dimensions(
    *,
    num_grid_rows: int,
    num_grid_columns: int,
    length: float,
    height: float,
    thickness: float,
    num_slices: int,
) -> None:
    if num_grid_rows < 2 or num_grid_columns < 2:
        raise ValueError(
            "num_grid_rows and num_grid_columns must both be at least two"
        )
    if num_grid_columns % num_slices != 0:
        raise ValueError(
            "num_grid_columns must be divisible by the number of "
            "yield-strength slices. This is only a requirement for this "
            "synthetic case, not for the VFM implementation."
        )
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in (length, height, thickness)
    ):
        raise ValueError(
            "length, height and thickness must be finite and positive"
        )
