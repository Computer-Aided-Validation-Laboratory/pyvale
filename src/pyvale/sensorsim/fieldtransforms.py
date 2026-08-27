# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from scipy.spatial.transform import Rotation
import pyvista as pv
from pyvale.dataio.simdata import SimData
from pyvale.sensorsim.field import IField


class IFieldTransform(ABC):
    """Abstract base class (interface) for field transformation operators."""

    @abstractmethod
    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Returns the names of output components given input components."""

    @abstractmethod
    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        """Transforms raw interpolated field tensor into derived quantity.

        Parameters
        ----------
        raw_samples : np.ndarray
            Array of raw field values with shape
            (n_points, n_in_comps, n_times).
        points : np.ndarray
            Evaluation points with shape (n_points, 3).
        times : np.ndarray
            Evaluation time steps with shape (n_times,).
        angles : tuple[Rotation, ...] | None
            Local sensor rotation frames for directional/traction projections.

        Returns
        -------
        np.ndarray
            Transformed field array with shape (n_points, n_out_comps, n_times).
        """


class FieldTransformCustom(IFieldTransform):
    """Applies a user-provided function to transform sampled field values."""

    __slots__ = ("_func", "_component_names")

    def __init__(
        self,
        func: Callable[
            [
                np.ndarray,
                np.ndarray,
                np.ndarray,
                tuple[Rotation, ...] | None,
            ],
            np.ndarray,
        ],
        component_names: tuple[str, ...] = ("custom",),
    ) -> None:
        self._func = func
        self._component_names = component_names

    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        return self._component_names

    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        return self._func(raw_samples, points, times, angles)


class FieldTransformVonMises(IFieldTransform):
    """Computes the scalar von Mises stress invariant from 2D or 3D
    stress/strain tensors."""

    __slots__ = ("_component_name",)

    def __init__(self, component_name: str = "von_mises") -> None:
        self._component_name = component_name

    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        return (self._component_name,)

    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        n_comps = raw_samples.shape[1]
        if n_comps == 3:
            s_xx = raw_samples[:, 0:1, :]
            s_yy = raw_samples[:, 1:2, :]
            s_xy = raw_samples[:, 2:3, :]
            vm_sq = s_xx**2 - s_xx * s_yy + s_yy**2 + 3.0 * (s_xy**2)
            vm = np.sqrt(np.maximum(0.0, vm_sq))
            return vm
        elif n_comps >= 6:
            s_xx = raw_samples[:, 0:1, :]
            s_yy = raw_samples[:, 1:2, :]
            s_zz = raw_samples[:, 2:3, :]
            s_xy = raw_samples[:, 3:4, :]
            s_yz = raw_samples[:, 4:5, :]
            s_xz = raw_samples[:, 5:6, :]
            vm_sq = 0.5 * (
                (s_xx - s_yy) ** 2
                + (s_yy - s_zz) ** 2
                + (s_zz - s_xx) ** 2
                + 6.0 * (s_xy**2 + s_yz**2 + s_xz**2)
            )
            vm = np.sqrt(np.maximum(0.0, vm_sq))
            return vm
        else:
            raise ValueError(
                f"Von Mises transform expects 3 (2D) or 6 (3D) components,"
                f" but got {n_comps} components."
            )


class FieldTransformPrincipal(IFieldTransform):
    """Computes ordered principal eigenvalues and maximum shear from symmetric
    2D or 3D tensors."""

    __slots__ = ("_return_max_shear",)

    def __init__(self, return_max_shear: bool = True) -> None:
        self._return_max_shear = return_max_shear

    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        n_in = len(input_components)
        prefix = "eps" if "strain" in input_components[0].lower() else "sigma"
        shear_key = "gamma_max" if prefix == "eps" else "tau_max"

        if n_in == 3:
            names = [f"{prefix}_1", f"{prefix}_2"]
            if self._return_max_shear:
                names.append(shear_key)
            return tuple(names)
        else:
            names = [f"{prefix}_1", f"{prefix}_2", f"{prefix}_3"]
            if self._return_max_shear:
                names.append(shear_key)
            return tuple(names)

    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        n_pts, n_comps, n_times = raw_samples.shape
        if n_comps == 3:
            s_xx = raw_samples[:, 0:1, :]
            s_yy = raw_samples[:, 1:2, :]
            s_xy = raw_samples[:, 2:3, :]

            center = 0.5 * (s_xx + s_yy)
            radius = np.sqrt((0.5 * (s_xx - s_yy)) ** 2 + s_xy**2)
            p1 = center + radius
            p2 = center - radius
            out_list = [p1, p2]
            if self._return_max_shear:
                out_list.append(2.0 * radius)
            return np.concatenate(out_list, axis=1)

        elif n_comps >= 6:
            mat = np.zeros((n_pts, n_times, 3, 3), dtype=np.float64)
            s_xx = raw_samples[:, 0, :]
            s_yy = raw_samples[:, 1, :]
            s_zz = raw_samples[:, 2, :]
            s_xy = raw_samples[:, 3, :]
            s_xz = raw_samples[:, 4, :]
            s_yz = raw_samples[:, 5, :]

            mat[:, :, 0, 0] = s_xx
            mat[:, :, 1, 1] = s_yy
            mat[:, :, 2, 2] = s_zz
            mat[:, :, 0, 1] = s_xy
            mat[:, :, 1, 0] = s_xy
            mat[:, :, 1, 2] = s_yz
            mat[:, :, 2, 1] = s_yz
            mat[:, :, 0, 2] = s_xz
            mat[:, :, 2, 0] = s_xz

            eigvals = np.linalg.eigvalsh(mat)
            p1 = eigvals[:, :, 2]
            p2 = eigvals[:, :, 1]
            p3 = eigvals[:, :, 0]

            p1 = p1[:, np.newaxis, :]
            p2 = p2[:, np.newaxis, :]
            p3 = p3[:, np.newaxis, :]
            out_list = [p1, p2, p3]
            if self._return_max_shear:
                tau_max = 0.5 * (p1 - p3)
                out_list.append(tau_max)
            return np.concatenate(out_list, axis=1)
        else:
            raise ValueError(
                f"Principal transform expects 3 (2D) or 6 (3D) components,"
                f" but got {n_comps} components."
            )


class FieldTransformHydrostatic(IFieldTransform):
    """Computes the mean hydrostatic scalar from 2D or 3D symmetric tensors."""

    __slots__ = ("_component_name",)

    def __init__(self, component_name: str = "hydrostatic") -> None:
        self._component_name = component_name

    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        return (self._component_name,)

    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        n_comps = raw_samples.shape[1]
        if n_comps == 3:
            s_xx = raw_samples[:, 0:1, :]
            s_yy = raw_samples[:, 1:2, :]
            return 0.5 * (s_xx + s_yy)
        elif n_comps >= 6:
            s_xx = raw_samples[:, 0:1, :]
            s_yy = raw_samples[:, 1:2, :]
            s_zz = raw_samples[:, 2:3, :]
            return (s_xx + s_yy + s_zz) / 3.0
        else:
            raise ValueError(
                f"Hydrostatic transform expects 3 or 6 components,"
                f" got {n_comps}."
            )


class FieldTransformTraction(IFieldTransform):
    """Computes the 3D surface traction vector t = sigma . n, normal traction
    t_n, and shear traction magnitude t_s from Cauchy stress tensors."""

    __slots__ = ("_include_scalar_projections",)

    def __init__(self, include_scalar_projections: bool = True) -> None:
        self._include_scalar_projections = include_scalar_projections

    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        names = ["traction_x", "traction_y", "traction_z"]
        if self._include_scalar_projections:
            names.extend(["traction_normal", "traction_shear"])
        return tuple(names)

    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        n_pts, n_comps, n_times = raw_samples.shape
        normals = np.zeros((n_pts, 3), dtype=np.float64)
        if angles is not None:
            if len(angles) == 1:
                n_vec = angles[0].apply(np.array([0.0, 0.0, 1.0]))
                normals[:] = n_vec
            else:
                for idx, rot in enumerate(angles):
                    normals[idx, :] = rot.apply(np.array([0.0, 0.0, 1.0]))
        else:
            normals[:, 2] = 1.0

        if n_comps == 3:
            s_xx = raw_samples[:, 0, :]
            s_yy = raw_samples[:, 1, :]
            s_xy = raw_samples[:, 2, :]
            nx = normals[:, 0, np.newaxis]
            ny = normals[:, 1, np.newaxis]

            tx = s_xx * nx + s_xy * ny
            ty = s_xy * nx + s_yy * ny
            tz = np.zeros_like(tx)
        elif n_comps >= 6:
            s_xx = raw_samples[:, 0, :]
            s_yy = raw_samples[:, 1, :]
            s_zz = raw_samples[:, 2, :]
            s_xy = raw_samples[:, 3, :]
            s_xz = raw_samples[:, 4, :]
            s_yz = raw_samples[:, 5, :]

            nx = normals[:, 0, np.newaxis]
            ny = normals[:, 1, np.newaxis]
            nz = normals[:, 2, np.newaxis]

            tx = s_xx * nx + s_xy * ny + s_xz * nz
            ty = s_xy * nx + s_yy * ny + s_yz * nz
            tz = s_xz * nx + s_yz * ny + s_zz * nz
        else:
            raise ValueError(
                f"Traction transform expects 3 or 6 components, got {n_comps}."
            )

        tx = tx[:, np.newaxis, :]
        ty = ty[:, np.newaxis, :]
        tz = tz[:, np.newaxis, :]
        out_list = [tx, ty, tz]

        if self._include_scalar_projections:
            nx = normals[:, 0, np.newaxis, np.newaxis]
            ny = normals[:, 1, np.newaxis, np.newaxis]
            nz = normals[:, 2, np.newaxis, np.newaxis]

            t_n = tx * nx + ty * ny + tz * nz
            t_sq = tx**2 + ty**2 + tz**2
            t_s = np.sqrt(np.maximum(0.0, t_sq - t_n**2))
            out_list.extend([t_n, t_s])

        return np.concatenate(out_list, axis=1)


class FieldTransformFlux(IFieldTransform):
    """Computes the normal scalar flux q_n = q . n from a vector field."""

    __slots__ = ("_component_name",)

    def __init__(self, component_name: str = "flux_normal") -> None:
        self._component_name = component_name

    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        return (self._component_name,)

    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        n_pts, n_comps, n_times = raw_samples.shape
        normals = np.zeros((n_pts, 3), dtype=np.float64)
        if angles is not None:
            if len(angles) == 1:
                normals[:] = angles[0].apply(np.array([0.0, 0.0, 1.0]))
            else:
                for idx, rot in enumerate(angles):
                    normals[idx, :] = rot.apply(np.array([0.0, 0.0, 1.0]))
        else:
            normals[:, 2] = 1.0

        if n_comps == 2:
            qx = raw_samples[:, 0, :]
            qy = raw_samples[:, 1, :]
            nx = normals[:, 0, np.newaxis]
            ny = normals[:, 1, np.newaxis]
            qn = qx * nx + qy * ny
        elif n_comps >= 3:
            qx = raw_samples[:, 0, :]
            qy = raw_samples[:, 1, :]
            qz = raw_samples[:, 2, :]
            nx = normals[:, 0, np.newaxis]
            ny = normals[:, 1, np.newaxis]
            nz = normals[:, 2, np.newaxis]
            qn = qx * nx + qy * ny + qz * nz
        else:
            raise ValueError(
                f"Flux transform expects 2 or 3 components, got {n_comps}."
            )

        return qn[:, np.newaxis, :]


class FieldTransformMagnitude(IFieldTransform):
    """Computes the Euclidean magnitude ||u|| of a vector field."""

    __slots__ = ("_component_name",)

    def __init__(self, component_name: str = "magnitude") -> None:
        self._component_name = component_name

    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        return (self._component_name,)

    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        mag = np.sqrt(np.sum(raw_samples**2, axis=1, keepdims=True))
        return mag


class FieldTransformDirectional(IFieldTransform):
    """Projects a vector field onto a specified direction vector u . d."""

    __slots__ = ("_direction", "_component_name")

    def __init__(
        self,
        direction: tuple[float, float, float] | np.ndarray,
        component_name: str = "directional_projection",
    ) -> None:
        dir_arr = np.asarray(direction, dtype=np.float64)
        self._direction = dir_arr / np.linalg.norm(dir_arr)
        self._component_name = component_name

    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        return (self._component_name,)

    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        n_comps = raw_samples.shape[1]
        d = self._direction[:n_comps]
        proj = np.tensordot(d, raw_samples, axes=(0, 1))
        return proj[:, np.newaxis, :]


class FieldTransformChain(IFieldTransform):
    """Chains multiple IFieldTransform operators sequentially."""

    __slots__ = ("_transforms",)

    def __init__(self, transforms: list[IFieldTransform]) -> None:
        self._transforms = transforms

    def get_component_names(
        self, input_components: tuple[str, ...]
    ) -> tuple[str, ...]:
        current_names = input_components
        for tr in self._transforms:
            current_names = tr.get_component_names(current_names)
        return current_names

    def transform(
        self,
        raw_samples: np.ndarray,
        points: np.ndarray,
        times: np.ndarray,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        current_data = raw_samples
        for tr in self._transforms:
            current_data = tr.transform(current_data, points, times, angles)
        return current_data


class FieldTransformed(IField):
    """Wraps any IField and intercepts sample_field() to apply an
    IFieldTransform."""

    __slots__ = ("_underlying_field", "_transform")

    def __init__(self, field: IField, transform: IFieldTransform) -> None:
        self._underlying_field = field
        self._transform = transform

    def set_sim_data(self, sim_data: SimData) -> None:
        self._underlying_field.set_sim_data(sim_data)

    def get_sim_data(self) -> SimData:
        return self._underlying_field.get_sim_data()

    def get_time_steps(self) -> np.ndarray:
        return self._underlying_field.get_time_steps()

    def get_visualiser(self) -> pv.UnstructuredGrid:
        return self._underlying_field.get_visualiser()

    def get_all_components(self) -> tuple[str, ...]:
        in_comps = self._underlying_field.get_all_components()
        return self._transform.get_component_names(in_comps)

    def get_component_index(self, comp_key: str) -> int:
        comps = self.get_all_components()
        return comps.index(comp_key)

    def sample_field(
        self,
        points: np.ndarray,
        times: np.ndarray | None = None,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        raw_samples = self._underlying_field.sample_field(
            points, times, angles=None
        )
        sample_times = (
            times
            if times is not None
            else self._underlying_field.get_time_steps()
        )
        return self._transform.transform(
            raw_samples, points, sample_times, angles
        )


class FieldMultiTransformed(IField):
    """Fuses multiple IField instances through a multi-field transform
    function."""

    __slots__ = ("_fields", "_transform_func", "_component_names")

    def __init__(
        self,
        fields: dict[str, IField],
        transform_func: Callable[
            [dict[str, np.ndarray], np.ndarray, np.ndarray], np.ndarray
        ],
        component_names: tuple[str, ...],
    ) -> None:
        self._fields = fields
        self._transform_func = transform_func
        self._component_names = component_names

    def set_sim_data(self, sim_data: SimData) -> None:
        for f in self._fields.values():
            f.set_sim_data(sim_data)

    def get_sim_data(self) -> SimData:
        first_field = next(iter(self._fields.values()))
        return first_field.get_sim_data()

    def get_time_steps(self) -> np.ndarray:
        first_field = next(iter(self._fields.values()))
        return first_field.get_time_steps()

    def get_visualiser(self) -> pv.UnstructuredGrid:
        first_field = next(iter(self._fields.values()))
        return first_field.get_visualiser()

    def get_all_components(self) -> tuple[str, ...]:
        return self._component_names

    def get_component_index(self, comp_key: str) -> int:
        return self._component_names.index(comp_key)

    def sample_field(
        self,
        points: np.ndarray,
        times: np.ndarray | None = None,
        angles: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        samples = {}
        for k, f in self._fields.items():
            samples[k] = f.sample_field(points, times, angles)

        first_field = next(iter(self._fields.values()))
        sample_times = (
            times if times is not None else first_field.get_time_steps()
        )
        return self._transform_func(samples, points, sample_times)
