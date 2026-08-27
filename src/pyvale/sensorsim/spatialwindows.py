# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Spatial integration windows for 0D point, 1D line, 2D area, and 3D volume
transducer support domains.
"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.transform import Rotation

from pyvale.sensorsim.enums import EIntegrationMode
from pyvale.sensorsim.integrationrules import (
    IIntegrationRule,
    IntegrationGaussLegendre,
)
from pyvale.sensorsim.spatialkernels import (
    ISpatialKernel,
    SpatialKernelUniform,
)


class ISpatialWindow(ABC):
    """Abstract interface for spatial sensing support windows."""

    @abstractmethod
    def get_spatial_dims(self) -> int:
        """Intrinsic spatial dimension of support window (0, 1, 2, or 3)."""

    @abstractmethod
    def get_measure(self) -> float:
        """Physical geometric measure (1.0 for point, length for 1D,
        area for 2D, volume for 3D).
        """

    @abstractmethod
    def get_effective_measure(self) -> float:
        """Integrated effective spatial measure (integral of sensitivity kernel
        over the geometric support domain).
        """

    @abstractmethod
    def get_local_points_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculates quadrature points and weights in the sensor local frame.

        Parameters
        ----------
        mode : EIntegrationMode, optional
            Integration mode (AVERAGE or ACCUMULATE), by default AVERAGE.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of:
            - local_points: shape (n_quad_pts, 3) in sensor local frame
            - weights: shape (n_quad_pts,), weighting factors
        """

    def to_global_points(
        self,
        sensor_positions: np.ndarray,
        sensor_rotations: tuple[Rotation, ...] | None = None,
    ) -> np.ndarray:
        """Transforms local window integration points to 3D global simulation
        coordinates for all sensors in an array.

        Parameters
        ----------
        sensor_positions : np.ndarray
            Sensor center anchor positions, shape=(n_sensors, 3).
        sensor_rotations : tuple[Rotation, ...] | None, optional
            Rotations for each sensor, by default None. If None, sensor frames
            are aligned with global axes.

        Returns
        -------
        np.ndarray
            Global evaluation points, shape=(n_sensors, n_quad_pts, 3).
        """
        local_pts, _ = self.get_local_points_and_weights()
        n_sensors = sensor_positions.shape[0]
        n_quad = local_pts.shape[0]

        if local_pts.shape[1] == 1:
            local_3d = np.zeros((n_quad, 3))
            local_3d[:, 0] = local_pts[:, 0]
        elif local_pts.shape[1] == 2:
            local_3d = np.zeros((n_quad, 3))
            local_3d[:, :2] = local_pts
        else:
            local_3d = local_pts

        if sensor_rotations is None:
            # Broadcast directly
            delta = local_3d[np.newaxis, :, :]
            global_pts = sensor_positions[:, np.newaxis, :] + delta
            return global_pts

        if len(sensor_rotations) == 1:
            # Single uniform rotation applied to all sensors
            rot = sensor_rotations[0]
            rotated_local = rot.apply(local_3d)
            delta = rotated_local[np.newaxis, :, :]
            global_pts = sensor_positions[:, np.newaxis, :] + delta
            return global_pts

        # Per-sensor rotation
        global_pts = np.zeros((n_sensors, n_quad, 3))
        for ss in range(n_sensors):
            rot_s = sensor_rotations[ss]
            rotated_s = rot_s.apply(local_3d)
            global_pts[ss, :, :] = sensor_positions[ss, :] + rotated_s

        return global_pts


class SpatialWindowPoint(ISpatialWindow):
    """0D point sensor support window (infinitesimal point or spot)."""

    __slots__ = ("_kernel",)

    def __init__(self, kernel: ISpatialKernel | None = None) -> None:
        if kernel is None:
            kernel = SpatialKernelUniform()
        self._kernel = kernel

    def get_spatial_dims(self) -> int:
        return 0

    def get_measure(self) -> float:
        return 1.0

    def get_effective_measure(self) -> float:
        return 1.0

    def get_local_points_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        pts = np.zeros((1, 3), dtype=float)
        wts = np.ones(1, dtype=float)
        return (pts, wts)


class SpatialWindowLine(ISpatialWindow):
    """1D line sensor support window (e.g. optical fiber Bragg grating,
    linear thermocouple probe).
    """

    __slots__ = ("_length", "_axis", "_integ_rule", "_kernel")

    def __init__(
        self,
        length: float,
        axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
        integ_rule: IIntegrationRule | None = None,
        kernel: ISpatialKernel | None = None,
    ) -> None:
        """
        Parameters
        ----------
        length : float
            Length of the line sensor.
        axis : tuple[float, float, float], optional
            Direction vector in the sensor's local frame, by default (1, 0, 0).
        integ_rule : IIntegrationRule | None, optional
            Numerical quadrature rule, by default IntegrationGaussLegendre(3).
        kernel : ISpatialKernel | None, optional
            Spatial weighting kernel, by default SpatialKernelUniform().
        """
        self._length = float(length)
        unit_axis = np.array(axis, dtype=float)
        norm = np.linalg.norm(unit_axis)
        if norm > 0.0:
            unit_axis = unit_axis / norm
        self._axis = unit_axis

        if integ_rule is None:
            integ_rule = IntegrationGaussLegendre(3)
        self._integ_rule = integ_rule

        if kernel is None:
            kernel = SpatialKernelUniform()
        self._kernel = kernel

    def get_spatial_dims(self) -> int:
        return 1

    def get_length(self) -> float:
        return self._length

    def get_axis(self) -> np.ndarray:
        return self._axis

    def get_measure(self) -> float:
        return self._length

    def get_effective_measure(self) -> float:
        _, weights = self.get_local_points_and_weights(
            mode=EIntegrationMode.ACCUMULATE
        )
        return float(np.sum(weights))

    def get_local_points_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        nodes_can, weights_can = self._integ_rule.get_nodes_and_weights(dims=1)
        jacobian = 0.5 * self._length
        s_coords = nodes_can.ravel() * jacobian
        raw_weights = weights_can * jacobian

        local_pts = s_coords[:, np.newaxis] * self._axis[np.newaxis, :]
        kernel_weights = self._kernel.eval_weights(local_pts)
        composite_weights = raw_weights * kernel_weights

        if mode == EIntegrationMode.AVERAGE:
            tot = np.sum(composite_weights)
            norm_w = (
                composite_weights / tot if tot > 0.0 else composite_weights
            )
            return (local_pts, norm_w)

        return (local_pts, composite_weights)


class SpatialWindowRectangle(ISpatialWindow):
    """2D rectangular planar support window in sensor's local XY plane
    [-length_x/2, length_x/2] x [-length_y/2, length_y/2].
    """

    __slots__ = ("_length_x", "_length_y", "_integ_rule", "_kernel")

    def __init__(
        self,
        length_x: float,
        length_y: float,
        integ_rule: IIntegrationRule | None = None,
        kernel: ISpatialKernel | None = None,
    ) -> None:
        """
        Parameters
        ----------
        length_x : float
            Length along local X axis.
        length_y : float
            Length along local Y axis.
        integ_rule : IIntegrationRule | None, optional
            Integration rule, by default IntegrationGaussLegendre(2).
        kernel : ISpatialKernel | None, optional
            Spatial weighting kernel, by default SpatialKernelUniform().
        """
        self._length_x = float(length_x)
        self._length_y = float(length_y)

        if integ_rule is None:
            integ_rule = IntegrationGaussLegendre(2)
        self._integ_rule = integ_rule

        if kernel is None:
            kernel = SpatialKernelUniform()
        self._kernel = kernel

    def get_spatial_dims(self) -> int:
        return 2

    def get_length_x(self) -> float:
        return self._length_x

    def get_length_y(self) -> float:
        return self._length_y

    def get_measure(self) -> float:
        return self._length_x * self._length_y

    def get_effective_measure(self) -> float:
        _, weights = self.get_local_points_and_weights(
            mode=EIntegrationMode.ACCUMULATE
        )
        return float(np.sum(weights))

    def get_local_points_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        nodes_can, weights_can = self._integ_rule.get_nodes_and_weights(dims=2)
        jac_x = 0.5 * self._length_x
        jac_y = 0.5 * self._length_y
        jacobian = jac_x * jac_y

        local_pts = np.zeros((nodes_can.shape[0], 3), dtype=float)
        local_pts[:, 0] = nodes_can[:, 0] * jac_x
        local_pts[:, 1] = nodes_can[:, 1] * jac_y

        raw_weights = weights_can * jacobian
        kernel_weights = self._kernel.eval_weights(local_pts)
        composite_weights = raw_weights * kernel_weights

        if mode == EIntegrationMode.AVERAGE:
            tot = np.sum(composite_weights)
            norm_w = (
                composite_weights / tot if tot > 0.0 else composite_weights
            )
            return (local_pts, norm_w)

        return (local_pts, composite_weights)


class SpatialWindowDisk(ISpatialWindow):
    """2D circular disk support window in sensor local XY plane (r <= radius).
    """

    __slots__ = ("_radius", "_integ_rule", "_kernel")

    def __init__(
        self,
        radius: float,
        integ_rule: IIntegrationRule | None = None,
        kernel: ISpatialKernel | None = None,
    ) -> None:
        """
        Parameters
        ----------
        radius : float
            Disk radius.
        integ_rule : IIntegrationRule | None, optional
            Integration rule, by default IntegrationGaussLegendre((3, 6)).
        kernel : ISpatialKernel | None, optional
            Spatial weighting kernel, by default SpatialKernelUniform().
        """
        self._radius = float(radius)

        if integ_rule is None:
            integ_rule = IntegrationGaussLegendre((3, 6))
        self._integ_rule = integ_rule

        if kernel is None:
            kernel = SpatialKernelUniform()
        self._kernel = kernel

    def get_spatial_dims(self) -> int:
        return 2

    def get_radius(self) -> float:
        return self._radius

    def get_measure(self) -> float:
        return np.pi * (self._radius**2)

    def get_effective_measure(self) -> float:
        _, weights = self.get_local_points_and_weights(
            mode=EIntegrationMode.ACCUMULATE
        )
        return float(np.sum(weights))

    def get_local_points_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        nodes_can, weights_can = self._integ_rule.get_nodes_and_weights(dims=2)
        r_vals = 0.5 * self._radius * (nodes_can[:, 0] + 1.0)
        theta_vals = np.pi * (nodes_can[:, 1] + 1.0)

        jac = r_vals * (0.5 * self._radius) * np.pi
        raw_weights = weights_can * jac

        local_pts = np.zeros((nodes_can.shape[0], 3), dtype=float)
        local_pts[:, 0] = r_vals * np.cos(theta_vals)
        local_pts[:, 1] = r_vals * np.sin(theta_vals)

        kernel_weights = self._kernel.eval_weights(local_pts)
        composite_weights = raw_weights * kernel_weights

        if mode == EIntegrationMode.AVERAGE:
            tot = np.sum(composite_weights)
            norm_w = (
                composite_weights / tot if tot > 0.0 else composite_weights
            )
            return (local_pts, norm_w)

        return (local_pts, composite_weights)


class SpatialWindowBox(ISpatialWindow):
    """3D rectangular cuboid (box) support window in sensor's local frame
    [-Lx/2, Lx/2] x [-Ly/2, Ly/2] x [-Lz/2, Lz/2].
    """

    __slots__ = (
        "_length_x",
        "_length_y",
        "_length_z",
        "_integ_rule",
        "_kernel",
    )

    def __init__(
        self,
        length_x: float,
        length_y: float,
        length_z: float,
        integ_rule: IIntegrationRule | None = None,
        kernel: ISpatialKernel | None = None,
    ) -> None:
        """
        Parameters
        ----------
        length_x : float
            Length along local X axis.
        length_y : float
            Length along local Y axis.
        length_z : float
            Length along local Z axis.
        integ_rule : IIntegrationRule | None, optional
            Integration rule, by default IntegrationGaussLegendre(2).
        kernel : ISpatialKernel | None, optional
            Spatial weighting kernel, by default SpatialKernelUniform().
        """
        self._length_x = float(length_x)
        self._length_y = float(length_y)
        self._length_z = float(length_z)

        if integ_rule is None:
            integ_rule = IntegrationGaussLegendre(2)
        self._integ_rule = integ_rule

        if kernel is None:
            kernel = SpatialKernelUniform()
        self._kernel = kernel

    def get_spatial_dims(self) -> int:
        return 3

    def get_length_x(self) -> float:
        return self._length_x

    def get_length_y(self) -> float:
        return self._length_y

    def get_length_z(self) -> float:
        return self._length_z

    def get_measure(self) -> float:
        return self._length_x * self._length_y * self._length_z

    def get_effective_measure(self) -> float:
        _, weights = self.get_local_points_and_weights(
            mode=EIntegrationMode.ACCUMULATE
        )
        return float(np.sum(weights))

    def get_local_points_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        nodes_can, weights_can = self._integ_rule.get_nodes_and_weights(dims=3)
        jac_x = 0.5 * self._length_x
        jac_y = 0.5 * self._length_y
        jac_z = 0.5 * self._length_z
        jacobian = jac_x * jac_y * jac_z

        local_pts = np.zeros((nodes_can.shape[0], 3), dtype=float)
        local_pts[:, 0] = nodes_can[:, 0] * jac_x
        local_pts[:, 1] = nodes_can[:, 1] * jac_y
        local_pts[:, 2] = nodes_can[:, 2] * jac_z

        raw_weights = weights_can * jacobian
        kernel_weights = self._kernel.eval_weights(local_pts)
        composite_weights = raw_weights * kernel_weights

        if mode == EIntegrationMode.AVERAGE:
            tot = np.sum(composite_weights)
            norm_w = (
                composite_weights / tot if tot > 0.0 else composite_weights
            )
            return (local_pts, norm_w)

        return (local_pts, composite_weights)


class SpatialWindowCylinder(ISpatialWindow):
    """3D cylindrical support window along sensor's local Z axis
    (r <= radius, z in [-height/2, height/2]).
    """

    __slots__ = ("_radius", "_height", "_integ_rule", "_kernel")

    def __init__(
        self,
        radius: float,
        height: float,
        integ_rule: IIntegrationRule | None = None,
        kernel: ISpatialKernel | None = None,
    ) -> None:
        """
        Parameters
        ----------
        radius : float
            Cylinder radius.
        height : float
            Cylinder height along local Z axis.
        integ_rule : IIntegrationRule | None, optional
            Integration rule, by default IntegrationGaussLegendre((3, 6, 2)).
        kernel : ISpatialKernel | None, optional
            Spatial weighting kernel, by default SpatialKernelUniform().
        """
        self._radius = float(radius)
        self._height = float(height)

        if integ_rule is None:
            integ_rule = IntegrationGaussLegendre((3, 6, 2))
        self._integ_rule = integ_rule

        if kernel is None:
            kernel = SpatialKernelUniform()
        self._kernel = kernel

    def get_spatial_dims(self) -> int:
        return 3

    def get_radius(self) -> float:
        return self._radius

    def get_height(self) -> float:
        return self._height

    def get_measure(self) -> float:
        return np.pi * (self._radius**2) * self._height

    def get_effective_measure(self) -> float:
        _, weights = self.get_local_points_and_weights(
            mode=EIntegrationMode.ACCUMULATE
        )
        return float(np.sum(weights))

    def get_local_points_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        nodes_can, weights_can = self._integ_rule.get_nodes_and_weights(dims=3)
        r_vals = 0.5 * self._radius * (nodes_can[:, 0] + 1.0)
        theta_vals = np.pi * (nodes_can[:, 1] + 1.0)
        z_vals = 0.5 * self._height * nodes_can[:, 2]

        jac = r_vals * (0.5 * self._radius) * np.pi * (0.5 * self._height)
        raw_weights = weights_can * jac

        local_pts = np.zeros((nodes_can.shape[0], 3), dtype=float)
        local_pts[:, 0] = r_vals * np.cos(theta_vals)
        local_pts[:, 1] = r_vals * np.sin(theta_vals)
        local_pts[:, 2] = z_vals

        kernel_weights = self._kernel.eval_weights(local_pts)
        composite_weights = raw_weights * kernel_weights

        if mode == EIntegrationMode.AVERAGE:
            tot = np.sum(composite_weights)
            norm_w = (
                composite_weights / tot if tot > 0.0 else composite_weights
            )
            return (local_pts, norm_w)

        return (local_pts, composite_weights)


class SpatialWindowSphere(ISpatialWindow):
    """3D spherical support window (r <= radius)."""

    __slots__ = ("_radius", "_integ_rule", "_kernel")

    def __init__(
        self,
        radius: float,
        integ_rule: IIntegrationRule | None = None,
        kernel: ISpatialKernel | None = None,
    ) -> None:
        """
        Parameters
        ----------
        radius : float
            Sphere radius.
        integ_rule : IIntegrationRule | None, optional
            Integration rule, by default IntegrationGaussLegendre((3, 6, 4)).
        kernel : ISpatialKernel | None, optional
            Spatial weighting kernel, by default SpatialKernelUniform().
        """
        self._radius = float(radius)

        if integ_rule is None:
            integ_rule = IntegrationGaussLegendre((3, 6, 4))
        self._integ_rule = integ_rule

        if kernel is None:
            kernel = SpatialKernelUniform()
        self._kernel = kernel

    def get_spatial_dims(self) -> int:
        return 3

    def get_radius(self) -> float:
        return self._radius

    def get_measure(self) -> float:
        return (4.0 / 3.0) * np.pi * (self._radius**3)

    def get_effective_measure(self) -> float:
        _, weights = self.get_local_points_and_weights(
            mode=EIntegrationMode.ACCUMULATE
        )
        return float(np.sum(weights))

    def get_local_points_and_weights(
        self, mode: EIntegrationMode = EIntegrationMode.AVERAGE
    ) -> tuple[np.ndarray, np.ndarray]:
        nodes_can, weights_can = self._integ_rule.get_nodes_and_weights(dims=3)
        r_vals = 0.5 * self._radius * (nodes_can[:, 0] + 1.0)
        theta_vals = np.pi * (nodes_can[:, 1] + 1.0)
        phi_vals = 0.5 * np.pi * (nodes_can[:, 2] + 1.0)

        # Jacobian r^2 * sin(phi) * (R/2) * pi * (pi/2)
        jac = (
            (r_vals**2)
            * np.sin(phi_vals)
            * (0.5 * self._radius)
            * np.pi
            * (0.5 * np.pi)
        )
        raw_weights = weights_can * jac

        local_pts = np.zeros((nodes_can.shape[0], 3), dtype=float)
        local_pts[:, 0] = r_vals * np.sin(phi_vals) * np.cos(theta_vals)
        local_pts[:, 1] = r_vals * np.sin(phi_vals) * np.sin(theta_vals)
        local_pts[:, 2] = r_vals * np.cos(phi_vals)

        kernel_weights = self._kernel.eval_weights(local_pts)
        composite_weights = raw_weights * kernel_weights

        if mode == EIntegrationMode.AVERAGE:
            tot = np.sum(composite_weights)
            norm_w = (
                composite_weights / tot if tot > 0.0 else composite_weights
            )
            return (local_pts, norm_w)

        return (local_pts, composite_weights)
