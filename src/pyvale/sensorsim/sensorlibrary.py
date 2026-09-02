# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import numpy as np
from scipy.spatial.transform import Rotation
from pyvale.dataio.simdata import SimData
from pyvale.sensorsim.field import IField
from pyvale.sensorsim.fieldscalar import FieldScalar
from pyvale.sensorsim.fieldvector import FieldVector
from pyvale.sensorsim.fieldtensor import FieldTensor
from pyvale.sensorsim.fieldreductions import (
    FieldTransformed,
    FieldTransformDirectional,
    FieldTransformFlux,
    FieldTransformTraction,
)
from pyvale.sensorsim.sensordata import SensorData
from pyvale.sensorsim.sensordescriptor import SensorDescriptor
from pyvale.sensorsim.enums import (
    EDim,
    EIntegrationMode,
    EDifferentialMode,
    ERayMode,
)
from pyvale.sensorsim.sensorsspatial import SensorsSpatial
from pyvale.sensorsim.sensorsdifferential import SensorsDifferential
from pyvale.sensorsim.sensorsray import SensorsRay
from pyvale.sensorsim.spatialwindows import (
    SpatialWindowPoint,
    SpatialWindowLine,
    SpatialWindowRectangle,
    SpatialWindowDisk,
    SpatialWindowBox,
)
from pyvale.sensorsim.temporalwindows import (
    TemporalWindowInstant,
    TemporalWindowCausal,
    TemporalKernelExponentialDecay,
)
from pyvale.sensorsim.sensortools import (
    orient_from_direction,
    orient_from_normal,
)
from pyvale.sensorsim.errorsysindep import ErrSysOffset
from pyvale.sensorsim.errorrand import ErrRandGen
from pyvale.sensorsim.generatorsrandom import GenNormal


class SensorLibrary:
    """Comprehensive catalog of named experimental thermo-mechanical
    transducers with configurable ideal or error-injected modes.
    """

    @staticmethod
    def thermocouple(
        sim_data: SimData,
        positions: np.ndarray,
        comp_key: str = "temperature",
        spatial_dims: EDim = EDim.TWOD,
        sample_times: np.ndarray | None = None,
        with_meas_errs: bool = False,
        time_constant: float = 0.05,
    ) -> SensorsSpatial:
        """Assembles a thermocouple temperature sensor array."""
        field = FieldScalar(sim_data, comp_key, spatial_dims)
        sens_data = SensorData(positions=positions, sample_times=sample_times)
        descriptor = SensorDescriptor(
            name="Thermocouple", tag="TC", symbol="T", units="°C"
        )

        if with_meas_errs:
            t_win = TemporalWindowCausal(
                duration=3.0 * time_constant,
                kernel=TemporalKernelExponentialDecay(time_constant),
            )
        else:
            t_win = TemporalWindowInstant()

        sensor = SensorsSpatial(
            sensor_data=sens_data,
            field=field,
            spatial_window=SpatialWindowPoint(),
            temporal_window=t_win,
            descriptor=descriptor,
            integration_mode=EIntegrationMode.AVERAGE,
        )

        if with_meas_errs:
            err_chain = (
                ErrSysOffset(offset=0.2),
                ErrRandGen(GenNormal(std=0.15, mean=0.0)),
            )
            sensor.set_error_chain(err_chain)

        return sensor

    @staticmethod
    def rtd(
        sim_data: SimData,
        positions: np.ndarray,
        bulb_size: tuple[float, float, float] = (4.0, 4.0, 4.0),
        comp_key: str = "temperature",
        spatial_dims: EDim = EDim.THREED,
        sample_times: np.ndarray | None = None,
        with_meas_errs: bool = False,
    ) -> SensorsSpatial:
        """Assembles a 3D platinum resistance temperature detector (RTD)."""
        field = FieldScalar(sim_data, comp_key, spatial_dims)
        sens_data = SensorData(positions=positions, sample_times=sample_times)
        descriptor = SensorDescriptor(
            name="RTD Bulb", tag="RTD", symbol="T", units="°C"
        )

        s_win = SpatialWindowBox(
            length_x=bulb_size[0],
            length_y=bulb_size[1],
            length_z=bulb_size[2],
        )

        sensor = SensorsSpatial(
            sensor_data=sens_data,
            field=field,
            spatial_window=s_win,
            temporal_window=TemporalWindowInstant(),
            descriptor=descriptor,
            integration_mode=EIntegrationMode.AVERAGE,
        )

        if with_meas_errs:
            err_chain = (
                ErrSysOffset(offset=0.1),
                ErrRandGen(GenNormal(std=0.05, mean=0.0)),
            )
            sensor.set_error_chain(err_chain)

        return sensor

    @staticmethod
    def strain_gauge(
        sim_data: SimData,
        positions: np.ndarray,
        grid_length_x: float = 3.0,
        grid_length_y: float = 2.0,
        angles: tuple[Rotation, ...] | None = None,
        spatial_dims: EDim = EDim.TWOD,
        sample_times: np.ndarray | None = None,
        with_meas_errs: bool = False,
    ) -> SensorsSpatial:
        """Assembles a rectangular foil strain gauge array."""
        if spatial_dims == EDim.TWOD:
            norm_keys = ("strain_xx", "strain_yy")
            dev_keys = ("strain_xy",)
        else:
            norm_keys = ("strain_xx", "strain_yy", "strain_zz")
            dev_keys = ("strain_xy", "strain_xz", "strain_yz")

        field = FieldTensor(sim_data, norm_keys, dev_keys, spatial_dims)
        sens_data = SensorData(
            positions=positions,
            angles=angles,
            sample_times=sample_times,
        )
        descriptor = SensorDescriptor(
            name="Strain Gauge", tag="SG", symbol="ε", units="με"
        )

        s_win = SpatialWindowRectangle(
            length_x=grid_length_x, length_y=grid_length_y
        )

        sensor = SensorsSpatial(
            sensor_data=sens_data,
            field=field,
            spatial_window=s_win,
            descriptor=descriptor,
            integration_mode=EIntegrationMode.AVERAGE,
        )

        if with_meas_errs:
            err_chain = (
                ErrSysOffset(offset=5e-6),
                ErrRandGen(GenNormal(std=2e-6, mean=0.0)),
            )
            sensor.set_error_chain(err_chain)

        return sensor

    @staticmethod
    def strain_rosette(
        sim_data: SimData,
        position: tuple[float, float, float] | np.ndarray,
        angles_deg: tuple[float, float, float] = (0.0, 45.0, 90.0),
        grid_size: tuple[float, float] = (3.0, 2.0),
        spatial_dims: EDim = EDim.TWOD,
        sample_times: np.ndarray | None = None,
        with_meas_errs: bool = False,
    ) -> SensorsSpatial:
        """Assembles a 3-branch strain gauge rosette."""
        pos = np.asarray(position, dtype=np.float64).reshape(1, 3)
        pos_3 = np.repeat(pos, len(angles_deg), axis=0)

        rots = tuple(
            Rotation.from_euler("z", ang, degrees=True) for ang in angles_deg
        )

        if spatial_dims == EDim.TWOD:
            norm_keys = ("strain_xx", "strain_yy")
            dev_keys = ("strain_xy",)
        else:
            norm_keys = ("strain_xx", "strain_yy", "strain_zz")
            dev_keys = ("strain_xy", "strain_xz", "strain_yz")

        field = FieldTensor(sim_data, norm_keys, dev_keys, spatial_dims)
        sens_data = SensorData(
            positions=pos_3,
            angles=rots,
            sample_times=sample_times,
        )
        descriptor = SensorDescriptor(
            name="Strain Rosette", tag="ROSETTE", symbol="ε", units="με"
        )

        s_win = SpatialWindowRectangle(
            length_x=grid_size[0], length_y=grid_size[1]
        )

        sensor = SensorsSpatial(
            sensor_data=sens_data,
            field=field,
            spatial_window=s_win,
            descriptor=descriptor,
            integration_mode=EIntegrationMode.AVERAGE,
        )

        if with_meas_errs:
            err_chain = (
                ErrSysOffset(offset=3e-6),
                ErrRandGen(GenNormal(std=1.5e-6, mean=0.0)),
            )
            sensor.set_error_chain(err_chain)

        return sensor

    @staticmethod
    def fbg_fiber(
        sim_data: SimData,
        point_start: tuple[float, float, float] | np.ndarray,
        point_end: tuple[float, float, float] | np.ndarray,
        comp_key: str = "temperature",
        spatial_dims: EDim = EDim.TWOD,
        sample_times: np.ndarray | None = None,
        with_meas_errs: bool = False,
    ) -> SensorsSpatial:
        """Assembles an optical fiber line sensor (e.g. FBG)."""
        p0 = np.asarray(point_start, dtype=np.float64)
        p1 = np.asarray(point_end, dtype=np.float64)
        center = (0.5 * (p0 + p1)).reshape(1, 3)
        diff = p1 - p0
        length = float(np.linalg.norm(diff))
        rot = orient_from_direction(diff)

        field = FieldScalar(sim_data, comp_key, spatial_dims)
        sens_data = SensorData(
            positions=center,
            angles=(rot,),
            sample_times=sample_times,
        )
        descriptor = SensorDescriptor(
            name="FBG Optical Fiber", tag="FBG", symbol="S", units="a.u."
        )

        sensor = SensorsSpatial(
            sensor_data=sens_data,
            field=field,
            spatial_window=SpatialWindowLine(length=length),
            descriptor=descriptor,
            integration_mode=EIntegrationMode.AVERAGE,
        )

        if with_meas_errs:
            err_chain = (
                ErrSysOffset(offset=0.05),
                ErrRandGen(GenNormal(std=0.02, mean=0.0)),
            )
            sensor.set_error_chain(err_chain)

        return sensor

    @staticmethod
    def extensometer(
        sim_data: SimData,
        anchor_a: tuple[float, float, float] | np.ndarray,
        anchor_b: tuple[float, float, float] | np.ndarray,
        disp_keys: tuple[str, ...] = ("disp_x", "disp_y"),
        knife_edge_length: float = 5.0,
        spatial_dims: EDim = EDim.TWOD,
        sample_times: np.ndarray | None = None,
        with_meas_errs: bool = False,
    ) -> SensorsDifferential:
        """Assembles a clip-on dual knife-edge tensile extensometer."""
        pa = np.asarray(anchor_a, dtype=np.float64).reshape(1, 3)
        pb = np.asarray(anchor_b, dtype=np.float64).reshape(1, 3)

        field_a = FieldVector(sim_data, disp_keys, spatial_dims)
        field_b = FieldVector(sim_data, disp_keys, spatial_dims)

        sens_data_a = SensorData(positions=pa, sample_times=sample_times)
        sens_data_b = SensorData(positions=pb, sample_times=sample_times)

        s_win = SpatialWindowLine(
            length=knife_edge_length, axis=(0.0, 1.0, 0.0)
        )

        sens_a = SensorsSpatial(
            sensor_data=sens_data_a,
            field=field_a,
            spatial_window=s_win,
            integration_mode=EIntegrationMode.AVERAGE,
        )
        sens_b = SensorsSpatial(
            sensor_data=sens_data_b,
            field=field_b,
            spatial_window=s_win,
            integration_mode=EIntegrationMode.AVERAGE,
        )

        descriptor = SensorDescriptor(
            name="Extensometer", tag="EXT", symbol="ε", units="mm/mm"
        )

        ext = SensorsDifferential(
            sensor_a=sens_a,
            sensor_b=sens_b,
            mode=EDifferentialMode.STRAIN,
            descriptor=descriptor,
        )

        if with_meas_errs:
            err_chain = (
                ErrSysOffset(offset=10e-6),
                ErrRandGen(GenNormal(std=5e-6, mean=0.0)),
            )
            ext.set_error_chain(err_chain)

        return ext

    @staticmethod
    def lvdt(
        sim_data: SimData,
        target_position: tuple[float, float, float] | np.ndarray,
        axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
        disp_keys: tuple[str, ...] = ("disp_x", "disp_y", "disp_z"),
        spatial_dims: EDim = EDim.THREED,
        sample_times: np.ndarray | None = None,
        with_meas_errs: bool = False,
    ) -> SensorsSpatial:
        """Assembles a Linear Variable Differential Transformer (LVDT)."""
        pos = np.asarray(target_position, dtype=np.float64).reshape(1, 3)
        raw_vec = FieldVector(sim_data, disp_keys, spatial_dims)
        derived_field = FieldTransformed(
            raw_vec,
            FieldTransformDirectional(
                direction=axis, component_name="displacement"
            ),
        )

        sens_data = SensorData(positions=pos, sample_times=sample_times)
        descriptor = SensorDescriptor(
            name="LVDT", tag="LVDT", symbol="u", units="mm"
        )

        sensor = SensorsSpatial(
            sensor_data=sens_data,
            field=derived_field,
            spatial_window=SpatialWindowPoint(),
            descriptor=descriptor,
            integration_mode=EIntegrationMode.AVERAGE,
        )

        if with_meas_errs:
            err_chain = (
                ErrSysOffset(offset=0.005),
                ErrRandGen(GenNormal(std=0.002, mean=0.0)),
            )
            sensor.set_error_chain(err_chain)

        return sensor

    @staticmethod
    def load_cell(
        sim_data: SimData,
        mount_position: tuple[float, float, float] | np.ndarray,
        contact_area_x: float = 20.0,
        contact_area_y: float = 20.0,
        normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
        spatial_dims: EDim = EDim.THREED,
        sample_times: np.ndarray | None = None,
        with_meas_errs: bool = False,
    ) -> SensorsSpatial:
        """Assembles a multi-axis load cell measuring resultant force."""
        pos = np.asarray(mount_position, dtype=np.float64).reshape(1, 3)
        rot = orient_from_normal(normal)

        prefix = (
            "stress"
            if sim_data.node_vars and "stress_xx" in sim_data.node_vars
            else "sigma"
        )
        if spatial_dims == EDim.TWOD:
            norm_keys = (f"{prefix}_xx", f"{prefix}_yy")
            dev_keys = (f"{prefix}_xy",)
        else:
            norm_keys = (f"{prefix}_xx", f"{prefix}_yy", f"{prefix}_zz")
            dev_keys = (f"{prefix}_xy", f"{prefix}_xz", f"{prefix}_yz")

        raw_tensor = FieldTensor(sim_data, norm_keys, dev_keys, spatial_dims)
        traction_field = FieldTransformed(
            raw_tensor,
            FieldTransformTraction(include_scalar_projections=True),
        )

        sens_data = SensorData(
            positions=pos,
            angles=(rot,),
            sample_times=sample_times,
        )
        descriptor = SensorDescriptor(
            name="Load Cell", tag="LC", symbol="F", units="N"
        )

        s_win = SpatialWindowRectangle(
            length_x=contact_area_x, length_y=contact_area_y
        )

        sensor = SensorsSpatial(
            sensor_data=sens_data,
            field=traction_field,
            spatial_window=s_win,
            descriptor=descriptor,
            integration_mode=EIntegrationMode.ACCUMULATE,
        )

        if with_meas_errs:
            err_chain = (
                ErrSysOffset(offset=0.5),
                ErrRandGen(GenNormal(std=0.2, mean=0.0)),
            )
            sensor.set_error_chain(err_chain)

        return sensor

    @staticmethod
    def heat_flux_meter(
        sim_data: SimData,
        position: tuple[float, float, float] | np.ndarray,
        foil_radius: float = 5.0,
        normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
        flux_keys: tuple[str, ...] = ("flux_x", "flux_y", "flux_z"),
        spatial_dims: EDim = EDim.THREED,
        sample_times: np.ndarray | None = None,
        with_meas_errs: bool = False,
    ) -> SensorsSpatial:
        """Assembles a circular normal heat flux sensor."""
        pos = np.asarray(position, dtype=np.float64).reshape(1, 3)
        rot = orient_from_normal(normal)

        raw_vec = FieldVector(sim_data, flux_keys, spatial_dims)
        flux_field = FieldTransformed(raw_vec, FieldTransformFlux())

        sens_data = SensorData(
            positions=pos,
            angles=(rot,),
            sample_times=sample_times,
        )
        descriptor = SensorDescriptor(
            name="Heat Flux Sensor", tag="FLUX", symbol="q", units="W/m²"
        )

        sensor = SensorsSpatial(
            sensor_data=sens_data,
            field=flux_field,
            spatial_window=SpatialWindowDisk(radius=foil_radius),
            descriptor=descriptor,
            integration_mode=EIntegrationMode.AVERAGE,
        )

        if with_meas_errs:
            err_chain = (
                ErrSysOffset(offset=10.0),
                ErrRandGen(GenNormal(std=5.0, mean=0.0)),
            )
            sensor.set_error_chain(err_chain)

        return sensor

    @staticmethod
    def flow_meter(
        sim_data: SimData,
        pipe_center: tuple[float, float, float] | np.ndarray,
        pipe_radius: float = 25.0,
        flow_direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
        velocity_keys: tuple[str, ...] = ("vel_x", "vel_y", "vel_z"),
        spatial_dims: EDim = EDim.THREED,
        sample_times: np.ndarray | None = None,
        with_meas_errs: bool = False,
    ) -> SensorsSpatial:
        """Assembles a fluid velocity volumetric flow meter."""
        pos = np.asarray(pipe_center, dtype=np.float64).reshape(1, 3)
        rot = orient_from_normal(flow_direction)

        raw_vec = FieldVector(sim_data, velocity_keys, spatial_dims)
        flux_field = FieldTransformed(raw_vec, FieldTransformFlux())

        sens_data = SensorData(
            positions=pos,
            angles=(rot,),
            sample_times=sample_times,
        )
        descriptor = SensorDescriptor(
            name="Flow Meter", tag="FLOW", symbol="Q", units="m³/s"
        )

        sensor = SensorsSpatial(
            sensor_data=sens_data,
            field=flux_field,
            spatial_window=SpatialWindowDisk(radius=pipe_radius),
            descriptor=descriptor,
            integration_mode=EIntegrationMode.ACCUMULATE,
        )

        if with_meas_errs:
            err_chain = (
                ErrSysOffset(offset=0.001),
                ErrRandGen(GenNormal(std=0.0005, mean=0.0)),
            )
            sensor.set_error_chain(err_chain)

        return sensor

    @staticmethod
    def lidar(
        sim_data: SimData,
        scanner_position: tuple[float, float, float] | np.ndarray,
        beam_direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
        disp_field: IField | None = None,
        max_range: float = 1000.0,
        sample_times: np.ndarray | None = None,
        with_meas_errs: bool = False,
    ) -> SensorsRay:
        """Assembles a LIDAR standoff distance tracking sensor."""
        pos = np.asarray(scanner_position, dtype=np.float64).reshape(1, 3)
        beam = np.asarray(beam_direction, dtype=np.float64).reshape(1, 3)

        descriptor = SensorDescriptor(
            name="LIDAR", tag="LIDAR", symbol="d", units="mm"
        )

        ray_sens = SensorsRay(
            sim_data=sim_data,
            ray_origins=pos,
            ray_directions=beam,
            disp_field=disp_field,
            max_distance=max_range,
            sample_times=sample_times,
            mode=ERayMode.DISTANCE,
            descriptor=descriptor,
        )

        if with_meas_errs:
            err_chain = (
                ErrSysOffset(offset=0.05),
                ErrRandGen(GenNormal(std=0.02, mean=0.0)),
            )
            ray_sens.set_error_chain(err_chain)

        return ray_sens

    @staticmethod
    def pyrometer(
        sim_data: SimData,
        sensor_position: tuple[float, float, float] | np.ndarray,
        aim_direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
        temp_key: str = "temperature",
        spatial_dims: EDim = EDim.TWOD,
        sample_times: np.ndarray | None = None,
        with_meas_errs: bool = False,
    ) -> SensorsRay:
        """Assembles an infrared optical pyrometer."""
        pos = np.asarray(sensor_position, dtype=np.float64).reshape(1, 3)
        aim = np.asarray(aim_direction, dtype=np.float64).reshape(1, 3)
        field = FieldScalar(sim_data, temp_key, spatial_dims)

        descriptor = SensorDescriptor(
            name="Optical Pyrometer", tag="PYRO", symbol="T", units="°C"
        )

        ray_sens = SensorsRay(
            sim_data=sim_data,
            ray_origins=pos,
            ray_directions=aim,
            field=field,
            sample_times=sample_times,
            mode=ERayMode.SURFACE_FIELD,
            descriptor=descriptor,
        )

        if with_meas_errs:
            err_chain = (
                ErrSysOffset(offset=1.5),
                ErrRandGen(GenNormal(std=0.8, mean=0.0)),
            )
            ray_sens.set_error_chain(err_chain)

        return ray_sens

    @staticmethod
    def pressure_gauge(
        sim_data: SimData,
        position: tuple[float, float, float] | np.ndarray,
        diaphragm_radius: float = 3.0,
        comp_key: str = "pressure",
        spatial_dims: EDim = EDim.TWOD,
        sample_times: np.ndarray | None = None,
        with_meas_errs: bool = False,
    ) -> SensorsSpatial:
        """Assembles a diaphragm pressure transducer."""
        pos = np.asarray(position, dtype=np.float64).reshape(1, 3)
        field = FieldScalar(sim_data, comp_key, spatial_dims)
        sens_data = SensorData(positions=pos, sample_times=sample_times)
        descriptor = SensorDescriptor(
            name="Pressure Gauge", tag="PG", symbol="P", units="MPa"
        )

        sensor = SensorsSpatial(
            sensor_data=sens_data,
            field=field,
            spatial_window=SpatialWindowDisk(radius=diaphragm_radius),
            descriptor=descriptor,
            integration_mode=EIntegrationMode.AVERAGE,
        )

        if with_meas_errs:
            err_chain = (
                ErrSysOffset(offset=0.01),
                ErrRandGen(GenNormal(std=0.005, mean=0.0)),
            )
            sensor.set_error_chain(err_chain)

        return sensor

    @staticmethod
    def accelerometer(
        sim_data: SimData,
        position: tuple[float, float, float] | np.ndarray,
        axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
        disp_keys: tuple[str, ...] = ("disp_x", "disp_y", "disp_z"),
        spatial_dims: EDim = EDim.THREED,
        sample_times: np.ndarray | None = None,
        with_meas_errs: bool = False,
    ) -> SensorsSpatial:
        """Assembles a directional accelerometer / dynamic displacement
        probe."""
        pos = np.asarray(position, dtype=np.float64).reshape(1, 3)
        raw_vec = FieldVector(sim_data, disp_keys, spatial_dims)
        derived = FieldTransformed(
            raw_vec,
            FieldTransformDirectional(
                direction=axis, component_name="accel_displacement"
            ),
        )

        sens_data = SensorData(positions=pos, sample_times=sample_times)
        descriptor = SensorDescriptor(
            name="Accelerometer", tag="ACCEL", symbol="a", units="mm"
        )

        sensor = SensorsSpatial(
            sensor_data=sens_data,
            field=derived,
            spatial_window=SpatialWindowPoint(),
            descriptor=descriptor,
            integration_mode=EIntegrationMode.AVERAGE,
        )

        if with_meas_errs:
            err_chain = (
                ErrSysOffset(offset=0.002),
                ErrRandGen(GenNormal(std=0.001, mean=0.0)),
            )
            sensor.set_error_chain(err_chain)

        return sensor
