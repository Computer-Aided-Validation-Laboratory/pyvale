# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Physical selection and independent PyVista scalar layers."""

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

import pyvale.dataio as io
import pyvale.sensorsim as sens
import pyvale.valid as val


@pytest.fixture
def scene() -> tuple[val.ValResult, io.SimData]:
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    domain = val.ValDomain(
        {
            "temperature": val.ValObservable(
                val.ValGeometry(
                    val.EValGeometry.POINT, coords, coord_units="mm", coord_frame="cube"
                ),
                val.EValOperator.SAMPLE,
                ("temperature",),
                np.array([0.0, 1.0]),
                "degC",
                time_units="s",
            )
        }
    )
    shape = (4, 1, 2, 2)
    plus = np.broadcast_to(
        np.array([2.0, 0.0, 1.0, np.nan])[:, None, None, None], shape
    ).copy()
    minus = np.broadcast_to(
        np.array([0.0, 3.0, 1.0, np.nan])[:, None, None, None], shape
    ).copy()
    field = val.ValMetricField(
        (
            val.EValResAxis.POINT,
            val.EValResAxis.COMPONENT,
            val.EValResAxis.TIME,
            val.EValResAxis.SIM_TRANS,
        ),
        plus + minus,
        plus,
        minus,
    )
    result = val.ValResult(domain, {"temperature": {val.EMetricType.MAVM: field}}, {})
    simulation = io.SimData(
        coords=coords.copy(),
        connect={"connect1": np.array([[1, 2, 3, 4]])},
        time=np.array([0.0, 1.0]),
        num_spat_dims=3,
    )
    return result, simulation


def test_overlay_preserves_xyz_ids_and_independent_scalar_bars(
    scene: tuple[val.ValResult, io.SimData],
    tmp_path: Path,
) -> None:
    result, simulation = scene
    original_coords = simulation.coords.copy()
    scalars = np.array([100.0, 200.0, 300.0, 400.0])
    original_theme = pv.global_theme.font.color
    opts = sens.VisOptsSimSensors(
        interactive=False,
        window_size_px=(640, 480),
        camera_position="iso",
        font_colour="navy",
    )
    plotter = val.plot_val_metric_on_sim(
        result,
        "temperature",
        val.EMetricType.MAVM,
        selectors={val.EValResAxis.TIME: 1, val.EValResAxis.SIM_TRANS: 0},
        part=val.EValMetricPart.BALANCE,
        point_idxs=(2, 0, 3),
        sim_data=simulation,
        sim_field=val.ValSimField(
            scalars, 1, sens.SensorDescriptor(name="Temperature", units="degC")
        ),
        sim_coord_units="mm",
        sim_coord_frame="cube",
        vis_opts=opts,
        metric_opts=val.ValVisOpts(point_labels=("P0", "P1", "P2", "P3")),
    )
    try:
        assert plotter.off_screen
        metric_actor = plotter.actors["validation-points"]
        background_actor = plotter.actors["validation-background"]
        point_mesh = metric_actor.mapper.dataset
        np.testing.assert_array_equal(point_mesh.points, original_coords[[2, 0, 3]])
        np.testing.assert_array_equal(point_mesh["validation_point_id"], [2, 0, 3])
        np.testing.assert_allclose(point_mesh["validation_metric"], [0.0, 2.0, np.nan])
        assert metric_actor.mapper.scalar_range == (-2.0, 2.0)
        assert background_actor.mapper.scalar_range == (100.0, 400.0)
        assert len(plotter.scalar_bars) == 2
        assert any("MAVM d+ − d−" in title for title in plotter.scalar_bars.keys())
        assert any(
            "Simulation:" in title and "degC" in title
            for title in plotter.scalar_bars.keys()
        )
        assert any(name.startswith("validation-labels") for name in plotter.actors)
        screenshot = plotter.screenshot(tmp_path / "overlay.png")
        assert screenshot.shape[:2] == (480, 640)
        assert np.ptp(screenshot) > 0
        np.testing.assert_array_equal(simulation.coords, original_coords)
        np.testing.assert_array_equal(scalars, [100.0, 200.0, 300.0, 400.0])
        assert pv.global_theme.font.color == original_theme
        assert opts.colour_bar_lims is None
    finally:
        plotter.close()


def test_magnitude_and_zero_balance_have_valid_colour_limits(
    scene: tuple[val.ValResult, io.SimData],
) -> None:
    result, _ = scene
    selectors = {val.EValResAxis.TIME: 0, val.EValResAxis.SIM_TRANS: 0}
    plotter = val.plot_val_metric_on_sim(
        result, "temperature", val.EMetricType.MAVM, selectors=selectors
    )
    assert plotter.actors["validation-points"].mapper.scalar_range == (0.0, 3.0)
    plotter.close()
    plotter = val.plot_val_metric_on_sim(
        result,
        "temperature",
        val.EMetricType.MAVM,
        selectors=selectors,
        point_idxs=(2,),
        part=val.EValMetricPart.BALANCE,
    )
    assert plotter.actors["validation-points"].mapper.scalar_range == (-1.0, 1.0)
    plotter.close()


def test_selection_requires_all_scenarios_and_balance_preserves_cancellation(
    scene: tuple[val.ValResult, io.SimData],
) -> None:
    result, _ = scene
    field = result.get("temperature", val.EMetricType.MAVM)
    with pytest.raises(ValueError, match="Select"):
        val.extract_metric_values(field)
    balance = val.extract_metric_values(
        field,
        selectors={val.EValResAxis.TIME: 0, val.EValResAxis.SIM_TRANS: 0},
        part=val.EValMetricPart.BALANCE,
    )
    np.testing.assert_allclose(balance, [2.0, -3.0, 0.0, np.nan])
    assert field.metric[2, 0, 0, 0] == 2.0
    with pytest.raises(ValueError, match="directional"):
        val.select_metric_part(
            val.ValMetricField(field.axes, field.metric), val.EValMetricPart.D_PLUS
        )


@pytest.mark.parametrize(
    "time_idx, units, coord_frame, values, message",
    [
        (0, "mm", "cube", np.ones(4), "times must match"),
        (1, "m", "cube", np.ones(4), "units/coord_frame"),
        (1, "mm", "other", np.ones(4), "units/coord_frame"),
        (1, "mm", "cube", np.ones(3), "n_nodes"),
    ],
)
def test_overlay_rejects_misaligned_background(
    scene: tuple[val.ValResult, io.SimData],
    time_idx: int,
    units: str,
    coord_frame: str,
    values: np.ndarray,
    message: str,
) -> None:
    result, simulation = scene
    with pytest.raises(ValueError, match=message):
        val.plot_val_metric_on_sim(
            result,
            "temperature",
            val.EMetricType.MAVM,
            selectors={val.EValResAxis.TIME: 1, val.EValResAxis.SIM_TRANS: 0},
            sim_data=simulation,
            sim_field=val.ValSimField(values, time_idx, sens.SensorDescriptor()),
            sim_coord_units=units,
            sim_coord_frame=coord_frame,
        )


def test_signed_limits_must_be_symmetric(
    scene: tuple[val.ValResult, io.SimData],
) -> None:
    result, _ = scene
    with pytest.raises(ValueError, match="symmetric"):
        val.plot_val_metric_on_sim(
            result,
            "temperature",
            val.EMetricType.MAVM,
            selectors={val.EValResAxis.TIME: 0, val.EValResAxis.SIM_TRANS: 0},
            part=val.EValMetricPart.BALANCE,
            metric_opts=val.ValVisOpts(metric_colour_lims=(-1.0, 2.0)),
        )


def test_composition_preserves_existing_plotter_camera(
    scene: tuple[val.ValResult, io.SimData],
) -> None:
    result, _ = scene
    plotter = pv.Plotter(off_screen=True)
    plotter.camera_position = [(4.0, 2.0, 3.0), (0.5, 0.5, 0.5), (0.0, 0.0, 1.0)]
    camera = np.array(
        [plotter.camera.position, plotter.camera.focal_point, plotter.camera.up]
    )
    try:
        returned = val.plot_val_metric_on_sim(
            result,
            "temperature",
            val.EMetricType.MAVM,
            selectors={val.EValResAxis.TIME: 0, val.EValResAxis.SIM_TRANS: 0},
            plotter=plotter,
        )
        assert returned is plotter
        np.testing.assert_allclose(
            [plotter.camera.position, plotter.camera.focal_point, plotter.camera.up],
            camera,
        )
    finally:
        plotter.close()
