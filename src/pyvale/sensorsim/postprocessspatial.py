# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Spatial array signal processors for reconstructing strain tensors and
continuous surface deflection profiles from discrete sensor grids.
"""

from typing import Literal
import numpy as np
from scipy import integrate

from pyvale.sensorsim.measurementdata import MeasurementData
from pyvale.sensorsim.postprocessor import IMeasurementProcessor
from pyvale.sensorsim.sensordescriptor import SensorDescriptor


class ProcessSpatialStrain(IMeasurementProcessor):
    """Reconstructs the full 2D or 3D infinitesimal strain tensor fields
    from a spatial array of discrete displacement sensor probes.
    """

    __slots__ = (
        "_source",
        "_poly_degree",
        "_eval_positions",
        "_spatial_dims",
    )

    def __init__(
        self,
        source: str,
        poly_degree: int = 2,
        eval_positions: np.ndarray | None = None,
        spatial_dims: Literal["2D", "3D"] = "2D",
    ) -> None:
        self._source = source
        self._poly_degree = int(poly_degree)
        self._eval_positions = eval_positions
        self._spatial_dims = spatial_dims

    def get_source_keys(self) -> tuple[str, ...]:
        return (self._source,)

    def get_output_components(
        self,
        input_metadata: dict[str, MeasurementData],
    ) -> tuple[str, ...]:
        if self._spatial_dims == "2D":
            return ("strain_xx", "strain_yy", "strain_xy")
        return (
            "strain_xx",
            "strain_yy",
            "strain_zz",
            "strain_xy",
            "strain_xz",
            "strain_yz",
        )

    def process(
        self,
        inputs: dict[str, MeasurementData],
    ) -> MeasurementData:
        data = inputs[self._source]
        vals = data.values
        pos = data.positions

        if pos is None or pos.shape[0] < 3:
            raise ValueError(
                "ProcessSpatialStrain requires sensor positions with at "
                f"least 3 points, but received: {pos}"
            )

        if self._eval_positions is not None:
            eval_pts = self._eval_positions
        else:
            eval_pts = pos

        is_mc = vals.ndim == 4
        if is_mc:
            n_trials, n_sens, n_comps, n_times = vals.shape
        else:
            n_sens, n_comps, n_times = vals.shape
            n_trials = 1
            vals = np.expand_dims(vals, axis=0)

        n_eval = eval_pts.shape[0]

        if self._spatial_dims == "2D":
            out_comps = ("strain_xx", "strain_yy", "strain_xy")
            n_out_comps = 3
            strains = np.empty((n_trials, n_eval, n_out_comps, n_times))

            x_s = pos[:, 0]
            y_s = pos[:, 1]
            x_e = eval_pts[:, 0]
            y_e = eval_pts[:, 1]

            if self._poly_degree == 1 or n_sens < 6:
                # Linear basis: [1, x, y]
                A_s = np.column_stack([np.ones_like(x_s), x_s, y_s])
                A_e_dx = np.column_stack(
                    [np.zeros_like(x_e), np.ones_like(x_e), np.zeros_like(x_e)]
                )
                A_e_dy = np.column_stack(
                    [np.zeros_like(y_e), np.zeros_like(y_e), np.ones_like(y_e)]
                )
            else:
                # Quadratic basis: [1, x, y, x^2, xy, y^2]
                A_s = np.column_stack([
                    np.ones_like(x_s),
                    x_s,
                    y_s,
                    x_s**2,
                    x_s * y_s,
                    y_s**2,
                ])
                A_e_dx = np.column_stack([
                    np.zeros_like(x_e),
                    np.ones_like(x_e),
                    np.zeros_like(x_e),
                    2.0 * x_e,
                    y_e,
                    np.zeros_like(y_e),
                ])
                A_e_dy = np.column_stack([
                    np.zeros_like(y_e),
                    np.zeros_like(y_e),
                    np.ones_like(y_e),
                    np.zeros_like(x_e),
                    x_e,
                    2.0 * y_e,
                ])

            pinv_A = np.linalg.pinv(A_s)

            for tr in range(n_trials):
                u_x = vals[tr, :, 0, :]
                u_y = vals[tr, :, 1, :] if n_comps > 1 else np.zeros_like(u_x)

                coeff_ux = pinv_A @ u_x
                coeff_uy = pinv_A @ u_y

                dux_dx = A_e_dx @ coeff_ux
                dux_dy = A_e_dy @ coeff_ux
                duy_dx = A_e_dx @ coeff_uy
                duy_dy = A_e_dy @ coeff_uy

                eps_xx = dux_dx
                eps_yy = duy_dy
                eps_xy = 0.5 * (dux_dy + duy_dx)

                strains[tr, :, 0, :] = eps_xx
                strains[tr, :, 1, :] = eps_yy
                strains[tr, :, 2, :] = eps_xy

        else:
            raise NotImplementedError(
                "3D spatial strain fitting is not yet implemented"
            )

        if not is_mc:
            strains = strains[0]

        desc = SensorDescriptor(
            name="Derived Strain Tensor", tag="STRAIN", units="με"
        )

        return MeasurementData(
            values=strains,
            sample_times=data.sample_times,
            positions=eval_pts,
            components=out_comps,
            units="με",
            descriptor=desc,
        )


class ProcessIntegrateSpatial(IMeasurementProcessor):
    """Integrates spatial slope or deflection angles along a sensor path
    to reconstruct continuous surface deflection profiles.
    """

    __slots__ = (
        "_source",
        "_initial_value",
        "_label",
        "_units",
    )

    def __init__(
        self,
        source: str,
        initial_value: float = 0.0,
        label: str = "displacement",
        units: str = "mm",
    ) -> None:
        self._source = source
        self._initial_value = float(initial_value)
        self._label = label
        self._units = units

    def get_source_keys(self) -> tuple[str, ...]:
        return (self._source,)

    def get_output_components(
        self,
        input_metadata: dict[str, MeasurementData],
    ) -> tuple[str, ...]:
        return (self._label,)

    def process(
        self,
        inputs: dict[str, MeasurementData],
    ) -> MeasurementData:
        data = inputs[self._source]
        vals = data.values
        pos = data.positions

        if pos is None or pos.shape[0] < 2:
            raise ValueError(
                "ProcessIntegrateSpatial requires at least 2 spatial points, "
                f"but received: {pos}"
            )

        diffs = np.diff(pos, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        s_coords = np.insert(np.cumsum(seg_lengths), 0, 0.0)

        # Integrate along the sensor positions axis (axis -3 in standard shape)
        # Shape: (..., num_sensors, num_components, num_times)
        integrated = integrate.cumulative_trapezoid(
            vals, x=s_coords, axis=-3, initial=0.0
        )
        integrated = integrated + self._initial_value

        desc = SensorDescriptor(
            name=self._label, tag="DISP_SPAT", units=self._units
        )

        return MeasurementData(
            values=integrated,
            sample_times=data.sample_times,
            positions=pos,
            components=(self._label,),
            units=self._units,
            descriptor=desc,
        )
