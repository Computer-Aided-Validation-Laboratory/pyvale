from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# from pyvale.vfm.sensitivity_based_virtual_fields import SensitivityBasedVirtualFields


@dataclass(slots=True)
class GlobalVirtualFieldCostResult:
    """Scalar SBVF cost plus a few useful diagnostic arrays."""

    cost: float
    residual_vector: npt.NDArray[np.float64]
    internal_virtual_work: dict[str, npt.NDArray[np.float64]]
    external_virtual_work: dict[str, npt.NDArray[np.float64]]


def global_vf_cost_function(
    stress: npt.NDArray[np.float64],
    sensitivity_based_virtual_fields: dict[str, SensitivityBasedVirtualFields],
    force: npt.NDArray[np.float64],
    area: npt.NDArray[np.float64],
    thickness: float = 1.0,
    traction_edge: int = 0,
    scaling: bool = True,
    scale_fraction: float = 0.05,
) -> GlobalVirtualFieldCostResult:
    """Evaluate the global virtual-fields residual in a direct, explicit way."""

    area_4d = _coerce_area(area, stress.shape[0])
    residuals: list[npt.NDArray[np.float64]] = []
    internal_virtual_work: dict[str, npt.NDArray[np.float64]] = {}
    external_virtual_work: dict[str, npt.NDArray[np.float64]] = {}

    for name, sbvf in sensitivity_based_virtual_fields.items():
        ivw_4d = stress * sbvf.virtual_strain * area_4d * thickness
        ivw_4d = np.nan_to_num(ivw_4d, nan=0.0)
        ivw = np.sum(ivw_4d, axis=(1, 2, 3))

        evw = (
            force[:, 0] * sbvf.edge_displacement[:, 0, traction_edge]
            + force[:, 1] * sbvf.edge_displacement[:, 1, traction_edge]
        )

        alpha = 1.0
        if scaling:
            alpha = _compute_scaling(ivw, scale_fraction)

        residual = alpha * (ivw - evw)
        residuals.append(residual)
        internal_virtual_work[name] = ivw
        external_virtual_work[name] = evw

    if residuals:
        residual_vector = np.concatenate(residuals)
    else:
        residual_vector = np.zeros(0, dtype=np.float64)

    cost = float(residual_vector @ residual_vector)
    return GlobalVirtualFieldCostResult(
        cost=cost,
        residual_vector=residual_vector,
        internal_virtual_work=internal_virtual_work,
        external_virtual_work=external_virtual_work,
    )


def _coerce_area(
    area: npt.NDArray[np.float64],
    num_timesteps: int,
) -> npt.NDArray[np.float64]:
    if area.ndim == 2:
        return area[np.newaxis, np.newaxis, :, :]
    if area.ndim == 4:
        return area
    raise ValueError("area must be a 2D grid or a 4D array compatible with stress.")


def _compute_scaling(ivw: npt.NDArray[np.float64], scale_fraction: float) -> float:
    num_steps = max(1, int(np.floor(scale_fraction * ivw.size)))
    sorted_ivw = np.sort(np.abs(ivw))[::-1]
    mean_ivw = float(np.mean(sorted_ivw[:num_steps]))
    if mean_ivw == 0.0:
        return 1.0
    return 1.0 / mean_ivw
