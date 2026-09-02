# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
import pyvista as pv
from pyvale.dataio.simdata import SimData
from pyvale.sensorsim.field import IField
from pyvale.sensorsim.fieldconverter import simdata_to_pyvista_vis
from pyvale.sensorsim.sensorarray import ISensorArray
from pyvale.sensorsim.sensordata import SensorData
from pyvale.sensorsim.sensordescriptor import SensorDescriptor
from pyvale.sensorsim.enums import EDim, ERayMode
from pyvale.sensorsim.errorintegrator import ErrIntegrator, ErrIntOpts
from pyvale.sensorsim.errorgraph import ErrGraph
from pyvale.sensorsim.errorsimulator import IErrSimulator


class SensorsRay(ISensorArray):
    """Ray-casting sensor array (LIDAR standoff distance, optical surface
    pyrometers, and line-of-sight sensors) utilizing PyVista/VTK ray tracing.

    Directly implements the `ISensorArray` interface.
    """

    __slots__ = (
        "_sim_data",
        "_field",
        "_disp_field",
        "_ray_origins",
        "_ray_directions",
        "_max_distance",
        "_sample_times",
        "_mode",
        "_descriptor",
        "_sensor_data",
        "_truth",
        "_measurements",
        "_error_integrator",
    )

    def __init__(
        self,
        sim_data: SimData,
        ray_origins: np.ndarray,
        ray_directions: np.ndarray,
        field: IField | None = None,
        disp_field: IField | None = None,
        max_distance: float = 1000.0,
        sample_times: np.ndarray | None = None,
        mode: ERayMode = ERayMode.DISTANCE,
        descriptor: SensorDescriptor | None = None,
    ) -> None:
        """
        Parameters
        ----------
        sim_data : SimData
            Simulation data containing mesh geometry and coordinates.
        ray_origins : np.ndarray
            3D ray origins with shape (n_rays, 3).
        ray_directions : np.ndarray
            3D ray direction vectors with shape (n_rays, 3).
        field : IField | None, optional
            Field to sample at surface strike point or integrate along ray,
            by default None.
        disp_field : IField | None, optional
            Displacement field to dynamically deform mesh during ray trace,
            by default None.
        max_distance : float, optional
            Maximum ray casting reach, by default 1000.0.
        sample_times : np.ndarray | None, optional
            Sample times for measurement, by default simulation time steps.
        mode : ERayMode, optional
            Ray measurement mode (DISTANCE, SURFACE_FIELD, LINE_OF_SIGHT),
            by default DISTANCE.
        descriptor : SensorDescriptor | None, optional
            Sensor metadata descriptor, by default None.
        """
        self._sim_data = sim_data
        self._field = field
        self._disp_field = disp_field

        origins = np.asarray(ray_origins, dtype=np.float64)
        if origins.ndim == 1:
            origins = origins[np.newaxis, :]
        self._ray_origins = origins

        dirs = np.asarray(ray_directions, dtype=np.float64)
        if dirs.ndim == 1:
            dirs = dirs[np.newaxis, :]
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        self._ray_directions = dirs / norms

        self._max_distance = max_distance

        if sample_times is None:
            if field is not None:
                sample_times = field.get_time_steps()
            elif sim_data.time is not None:
                sample_times = sim_data.time
            else:
                sample_times = np.array([0.0])
        self._sample_times = sample_times

        self._mode = mode

        if descriptor is None:
            tag = "LIDAR" if mode == ERayMode.DISTANCE else "RAY"
            descriptor = SensorDescriptor(name="Ray Sensor", tag=tag)
        self._descriptor = descriptor

        self._sensor_data = SensorData(
            positions=self._ray_origins,
            sample_times=self._sample_times,
        )

        self._error_integrator = None
        self._truth = None
        self._measurements = None

    def get_ray_origins(self) -> np.ndarray:
        return self._ray_origins

    def get_ray_directions(self) -> np.ndarray:
        return self._ray_directions

    def get_mode(self) -> ERayMode:
        return self._mode

    def get_descriptor(self) -> SensorDescriptor:
        return self._descriptor

    def get_field(self) -> IField | None:
        return self._field

    def get_sensor_data(self) -> SensorData:
        return self._sensor_data

    def get_all_components(self) -> tuple[str, ...]:
        if self._mode == ERayMode.DISTANCE:
            return ("standoff_distance",)
        elif self._mode == ERayMode.SURFACE_FIELD:
            if self._field is not None:
                return self._field.get_all_components()
            return ("surface_value",)
        else:
            return ("path_integral",)

    def get_component_index(self, comp_key: str) -> int:
        comps = self.get_all_components()
        return comps.index(comp_key)

    def get_measurement_shape(self) -> tuple[int, int, int]:
        n_rays = self._ray_origins.shape[0]
        n_comps = len(self.get_all_components())
        n_times = self._sample_times.shape[0]
        return (n_rays, n_comps, n_times)

    def get_sample_times(self) -> np.ndarray:
        return self._sample_times

    def get_time_steps(self) -> np.ndarray:
        return self._sample_times

    def _build_surface_grid(self) -> pv.PolyData:
        is_3d = self._sim_data.coords.shape[1] == 3 and np.any(
            np.abs(self._sim_data.coords[:, 2]) > 1e-12
        )
        s_dim = EDim.THREED if is_3d else EDim.TWOD
        grid = simdata_to_pyvista_vis(self._sim_data, spatial_dims=s_dim)
        if isinstance(grid, pv.UnstructuredGrid):
            return grid.extract_surface(algorithm="dataset_surface")
        elif isinstance(grid, pv.PolyData):
            return grid
        else:
            return pv.PolyData(self._sim_data.coords)

    def calc_truth(self) -> np.ndarray:
        n_rays = self._ray_origins.shape[0]
        n_times = self._sample_times.shape[0]
        n_comps = len(self.get_all_components())

        surface = self._build_surface_grid()
        base_points = np.array(surface.points, copy=True)

        truth = np.zeros((n_rays, n_comps, n_times), dtype=np.float64)

        for tt, t_val in enumerate(self._sample_times):
            if self._disp_field is not None:
                t_arr = np.array([t_val])
                disp_samples = self._disp_field.sample_field(
                    base_points, times=t_arr
                )
                n_disp = disp_samples.shape[1]
                deformed_points = np.array(base_points, copy=True)
                deformed_points[:, :n_disp] += disp_samples[:, :n_disp, 0]
                surface.points = deformed_points

            for rr in range(n_rays):
                p0 = self._ray_origins[rr]
                d = self._ray_directions[rr]
                p1 = p0 + self._max_distance * d

                intersection_pts, _ = surface.ray_trace(
                    p0, p1, first_point=True
                )

                if intersection_pts.size >= 3:
                    hit_pt = (
                        intersection_pts[:3]
                        if intersection_pts.ndim == 1
                        else intersection_pts[0]
                    )
                    dist = float(np.linalg.norm(hit_pt - p0))

                    if self._mode == ERayMode.DISTANCE:
                        truth[rr, 0, tt] = dist
                    elif self._mode == ERayMode.SURFACE_FIELD:
                        if self._field is not None:
                            val = self._field.sample_field(
                                hit_pt.reshape(1, 3), times=np.array([t_val])
                            )
                            truth[rr, :, tt] = val[0, :, 0]
                        else:
                            truth[rr, 0, tt] = dist
                    else:
                        if self._field is not None:
                            n_quad = 10
                            s_nodes = np.linspace(0.0, dist, n_quad)
                            ds = dist / (n_quad - 1) if n_quad > 1 else dist
                            seg_pts = (
                                p0.reshape(1, 3)
                                + s_nodes[:, np.newaxis]
                                * d.reshape(1, 3)
                            )
                            seg_vals = self._field.sample_field(
                                seg_pts, times=np.array([t_val])
                            )
                            truth[rr, 0, tt] = (
                                np.sum(seg_vals[:, 0, 0]) * ds
                            )
                        else:
                            truth[rr, 0, tt] = dist
                else:
                    truth[rr, :, tt] = (
                        self._max_distance
                        if self._mode == ERayMode.DISTANCE
                        else np.nan
                    )

        self._truth = truth
        return self._truth

    def calc_ray_intersections(
        self, time_step: int = -1
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates 3D intersection hit points, distances, and valid hit
        booleans for visual rendering and analysis.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            - hits: shape (n_rays, 3) 3D surface intersection points
            - dists: shape (n_rays,) distance from origin to hit point
            - valid: shape (n_rays,) boolean flag indicating if ray hit mesh
        """
        n_rays = self._ray_origins.shape[0]
        surface = self._build_surface_grid()
        base_points = np.array(surface.points, copy=True)

        tt = time_step if time_step >= 0 else len(self._sample_times) - 1
        t_val = self._sample_times[tt]

        if self._disp_field is not None:
            t_arr = np.array([t_val])
            disp_samples = self._disp_field.sample_field(
                base_points, times=t_arr
            )
            n_disp = disp_samples.shape[1]
            deformed_points = np.array(base_points, copy=True)
            deformed_points[:, :n_disp] += disp_samples[:, :n_disp, 0]
            surface.points = deformed_points

        hits = np.zeros((n_rays, 3), dtype=np.float64)
        dists = np.zeros(n_rays, dtype=np.float64)
        valid = np.zeros(n_rays, dtype=bool)

        for rr in range(n_rays):
            p0 = self._ray_origins[rr]
            d = self._ray_directions[rr]
            p1 = p0 + self._max_distance * d

            pts, _ = surface.ray_trace(p0, p1, first_point=True)
            if pts.size >= 3:
                hit_pt = pts[:3] if pts.ndim == 1 else pts[0]
                hits[rr] = hit_pt
                dists[rr] = float(np.linalg.norm(hit_pt - p0))
                valid[rr] = True
            else:
                hits[rr] = p1
                dists[rr] = self._max_distance
                valid[rr] = False

        return (hits, dists, valid)

    def get_truth(self) -> np.ndarray:
        if self._truth is None:
            self._truth = self.calc_truth()
        return self._truth

    def set_error_chain(
        self,
        err_chain: (
            IErrSimulator
            | list[IErrSimulator]
            | tuple[IErrSimulator, ...]
            | ErrGraph
            | None
        ),
        err_opts: ErrIntOpts | None = None,
    ) -> None:
        """Sets the error chain or graph for the ray sensor array."""
        if err_chain is None:
            self._error_integrator = None
            return None

        if isinstance(err_chain, ErrGraph):
            self._error_integrator = err_chain
            return None

        if not isinstance(err_chain, (list, tuple)):
            err_chain = (err_chain,)
        else:
            err_chain = tuple(err_chain)

        self._error_integrator = ErrIntegrator(
            err_chain=err_chain,
            sensor_data_initial=self.get_sensor_data(),
            meas_shape=self.get_measurement_shape(),
            err_int_opts=err_opts,
        )

    def set_error_model(
        self,
        err_model: (
            IErrSimulator
            | list[IErrSimulator]
            | tuple[IErrSimulator, ...]
            | ErrGraph
            | None
        ),
        err_opts: ErrIntOpts | None = None,
    ) -> None:
        """Convenience method to set an error model, chain, or graph."""
        self.set_error_chain(err_model, err_opts=err_opts)

    def set_error_graph(self, err_graph: ErrGraph | None) -> None:
        self._error_integrator = err_graph

    def get_errors_systematic(self) -> np.ndarray | None:
        if self._error_integrator is None:
            return None
        return self._error_integrator.get_errs_sys()

    def get_errors_random(self) -> np.ndarray | None:
        if self._error_integrator is None:
            return None
        return self._error_integrator.get_errs_rand()

    def get_errors_total(self) -> np.ndarray | None:
        if self._error_integrator is None:
            return None
        return self._error_integrator.get_errs_total()

    def get_errs_sys(self) -> np.ndarray | None:
        return self.get_errors_systematic()

    def get_errs_rand(self) -> np.ndarray | None:
        return self.get_errors_random()

    def get_errs_total(self) -> np.ndarray | None:
        return self.get_errors_total()

    def calc_errors(self) -> np.ndarray | None:
        if self._error_integrator is None:
            return None
        if isinstance(self._error_integrator, ErrGraph):
            return self._error_integrator.calc_errors_from_graph(
                self.get_truth()
            )
        return self._error_integrator.calc_errors_from_chain(
            self.get_truth()
        )

    def sim_measurements(self) -> np.ndarray:
        truth = self.get_truth()

        if self._error_integrator is None:
            self._measurements = np.array(truth, copy=True)
            return self._measurements

        tot_err = self.calc_errors()
        if tot_err is not None:
            self._measurements = truth + tot_err
        else:
            self._measurements = np.array(truth, copy=True)

        return self._measurements

    def get_measurements(self) -> np.ndarray:
        if self._measurements is None:
            self._measurements = self.sim_measurements()
        return self._measurements
