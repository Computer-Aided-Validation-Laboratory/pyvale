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
from pyvale import valid
from pyvale.valid import validvis


@pytest.fixture
def scene() -> tuple[valid.ValidResult, io.SimData]:
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    observables = {
        "temperature": valid.ValidObservable(
            components=("temperature",),
            geometry=valid.EValidGeometry.POINT,
            coords=coords,
            times=np.array([0.0, 1.0]),
            phys_units="degC",
            coord_units="mm",
            coord_frame="cube",
            time_units="s",
        )
    }
    shape = (4, 1, 2, 2)
    plus = np.broadcast_to(
        np.array([2.0, 0.0, 1.0, np.nan])[:, None, None, None], shape
    ).copy()
    minus = np.broadcast_to(
        np.array([0.0, 3.0, 1.0, np.nan])[:, None, None, None], shape
    ).copy()
    field = valid.ValidMetricField(
        (
            valid.EValidResAxis.POINT,
            valid.EValidResAxis.COMPONENT,
            valid.EValidResAxis.TIME,
            valid.EValidResAxis.SIM_TRANS,
        ),
        {valid.EMetricOutput.D_PLUS: plus, valid.EMetricOutput.D_MINUS: minus},
    )
    result = valid.ValidResult(
        observables, {"temperature": {valid.EMetricType.MAVM: field}}, {}
    )
    sim_data = io.SimData(
        coords=coords.copy(),
        connect={"connect1": np.array([[1, 2, 3, 4]])},
        time=np.array([0.0, 1.0]),
        num_spat_dims=3,
        node_vars={
            "temperature": np.array(
                [[10.0, 100.0], [20.0, 200.0], [30.0, 300.0], [40.0, 400.0]]
            ),
        },
    )
    return result, sim_data


@pytest.mark.parametrize(
    "sim_field", ["temperature", np.array([100.0, 200.0, 300.0, 400.0])]
)
def test_overlay_preserves_xyz_ids_and_independent_scalar_bars(
    scene: tuple[valid.ValidResult, io.SimData],
    tmp_path: Path,
    sim_field: str | np.ndarray,
) -> None:
    result, sim_data = scene
    original_coords = sim_data.coords.copy()
    scalars = np.array([100.0, 200.0, 300.0, 400.0])
    original_theme = pv.global_theme.font.color
    opts = sens.VisOptsSimSensors(
        interactive=False,
        window_size_px=(640, 480),
        camera_position="iso",
        font_colour="navy",
    )
    plotter = validvis.plot_valid_metric_on_sim(
        result,
        "temperature",
        valid.EMetricType.MAVM,
        selectors={valid.EValidResAxis.TIME: 1, valid.EValidResAxis.SIM_TRANS: 0},
        output=valid.EMetricOutput.D_MINUS,
        point_idxs=(2, 0, 3),
        sim_data=sim_data,
        sim_field=sim_field,
        sim_time_idx=1,
        descriptor=sens.SensorDescriptor(name="Temperature", units="degC", tag="P"),
        sim_coord_units="mm",
        sim_coord_frame="cube",
        vis_opts=opts,
        metric_opts=validvis.ValidMetricVisOpts(labels_always_visible=True),
    )
    try:
        assert plotter.off_screen
        metric_actor = plotter.actors["validation-points"]
        background_actor = plotter.actors["validation-background"]
        point_mesh = metric_actor.mapper.dataset
        np.testing.assert_array_equal(point_mesh.points, original_coords[[2, 0, 3]])
        np.testing.assert_array_equal(point_mesh["validation_point_id"], [2, 0, 3])
        np.testing.assert_allclose(point_mesh["validation_metric"], [1.0, 0.0, np.nan])
        assert metric_actor.mapper.scalar_range == (0.0, 1.0)
        assert background_actor.mapper.scalar_range == (100.0, 400.0)
        assert len(plotter.scalar_bars) == 2
        assert any("MAVM d−" in title for title in plotter.scalar_bars.keys())  # noqa: SIM118
        assert any(
            "Sim.:" in title and "degC" in title
            for title in plotter.scalar_bars.keys()  # noqa: SIM118
        )
        assert any(name.startswith("validation-labels") for name in plotter.actors)
        screenshot = plotter.screenshot(tmp_path / "overlay.png")
        assert screenshot.shape[:2] == (480, 640)
        assert np.ptp(screenshot) > 0
        label_mapper = plotter.actors["validation-labels-labels"].GetMapper()
        point_tags = label_mapper.GetInputDataObject(0, 0).GetLabels()
        assert [point_tags.GetValue(index) for index in range(3)] == ["P3", "P1", "P4"]
        np.testing.assert_array_equal(sim_data.node_vars["temperature"][:, 1], scalars)
        if isinstance(sim_field, np.ndarray):
            np.testing.assert_array_equal(sim_field, scalars)
        np.testing.assert_array_equal(sim_data.coords, original_coords)
        np.testing.assert_array_equal(scalars, [100.0, 200.0, 300.0, 400.0])
        assert pv.global_theme.font.color == original_theme
        assert opts.colour_bar_lims is None
    finally:
        plotter.close()


def test_directional_and_zero_outputs_have_valid_colour_limits(
    scene: tuple[valid.ValidResult, io.SimData],
) -> None:
    result, _ = scene
    selectors = {valid.EValidResAxis.TIME: 0, valid.EValidResAxis.SIM_TRANS: 0}
    plotter = validvis.plot_valid_metric_on_sim(
        result,
        "temperature",
        valid.EMetricType.MAVM,
        selectors=selectors,
        output=valid.EMetricOutput.D_MINUS,
    )
    assert plotter.actors["validation-points"].mapper.scalar_range == (0.0, 3.0)
    plotter.close()
    plotter = validvis.plot_valid_metric_on_sim(
        result,
        "temperature",
        valid.EMetricType.MAVM,
        selectors=selectors,
        point_idxs=(0,),
        output=valid.EMetricOutput.D_MINUS,
    )
    assert plotter.actors["validation-points"].mapper.scalar_range == (0.0, 1.0)
    plotter.close()


def test_selection_requires_all_scenarios_and_preserves_output(
    scene: tuple[valid.ValidResult, io.SimData],
) -> None:
    result, _ = scene
    field = result.metrics["temperature"][valid.EMetricType.MAVM]
    with pytest.raises(ValueError, match="Select"):
        valid.extract_metric_values(field, output=valid.EMetricOutput.D_MINUS)
    selected = valid.extract_metric_values(
        field,
        selectors={valid.EValidResAxis.TIME: 0, valid.EValidResAxis.SIM_TRANS: 0},
        output=valid.EMetricOutput.D_MINUS,
    )
    np.testing.assert_allclose(selected, [0.0, 3.0, 1.0, np.nan])
    assert field.outputs[valid.EMetricOutput.D_MINUS][2, 0, 0, 0] == 1.0
    with pytest.raises(ValueError, match="output"):
        valid.select_metric_output(
            valid.ValidMetricField(
                field.axes,
                {
                    valid.EMetricOutput.D_MINUS: field.outputs[
                        valid.EMetricOutput.D_MINUS
                    ]
                },
            ),
            valid.EMetricOutput.D_PLUS,
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
    scene: tuple[valid.ValidResult, io.SimData],
    time_idx: int,
    units: str,
    coord_frame: str,
    values: np.ndarray,
    message: str,
) -> None:
    result, sim_data = scene
    with pytest.raises(ValueError, match=message):
        validvis.plot_valid_metric_on_sim(
            result,
            "temperature",
            valid.EMetricType.MAVM,
            selectors={valid.EValidResAxis.TIME: 1, valid.EValidResAxis.SIM_TRANS: 0},
            sim_data=sim_data,
            sim_field=values,
            sim_time_idx=time_idx,
            sim_coord_units=units,
            sim_coord_frame=coord_frame,
            output=valid.EMetricOutput.D_PLUS,
        )


def test_output_limits_must_increase(
    scene: tuple[valid.ValidResult, io.SimData],
) -> None:
    result, _ = scene
    with pytest.raises(ValueError, match="increasing"):
        validvis.plot_valid_metric_on_sim(
            result,
            "temperature",
            valid.EMetricType.MAVM,
            selectors={valid.EValidResAxis.TIME: 0, valid.EValidResAxis.SIM_TRANS: 0},
            output=valid.EMetricOutput.D_MINUS,
            metric_opts=validvis.ValidMetricVisOpts(metric_colour_lims=(2.0, 1.0)),
        )


def test_composition_preserves_existing_plotter_camera(
    scene: tuple[valid.ValidResult, io.SimData],
) -> None:
    result, _ = scene
    plotter = pv.Plotter(off_screen=True)
    plotter.camera_position = [(4.0, 2.0, 3.0), (0.5, 0.5, 0.5), (0.0, 0.0, 1.0)]
    camera = np.array(
        [plotter.camera.position, plotter.camera.focal_point, plotter.camera.up]
    )
    try:
        returned = validvis.plot_valid_metric_on_sim(
            result,
            "temperature",
            valid.EMetricType.MAVM,
            selectors={valid.EValidResAxis.TIME: 0, valid.EValidResAxis.SIM_TRANS: 0},
            plotter=plotter,
            output=valid.EMetricOutput.D_PLUS,
        )
        assert returned is plotter
        np.testing.assert_allclose(
            [plotter.camera.position, plotter.camera.focal_point, plotter.camera.up],
            camera,
        )
    finally:
        plotter.close()


@pytest.mark.parametrize(
    ("sim_field", "sim_time_idx", "message"),
    [
        ("missing", 1, "no nodal field"),
        ("temperature", None, "time index"),
        ("temperature", -1, "time index"),
        ("temperature", 2, "time index"),
        (np.ones((4, 2)), 1, "n_nodes"),
        (np.array([1.0, 2.0, 3.0, np.inf]), 1, "infinite"),
    ],
)
def test_background_rejects_invalid_fields_and_steps(
    scene: tuple[valid.ValidResult, io.SimData],
    sim_field: str | np.ndarray,
    sim_time_idx: int | None,
    message: str,
) -> None:
    result, sim_data = scene
    with pytest.raises(ValueError, match=message):
        validvis.plot_valid_metric_on_sim(
            result,
            "temperature",
            valid.EMetricType.MAVM,
            output=valid.EMetricOutput.D_PLUS,
            selectors={valid.EValidResAxis.TIME: 1, valid.EValidResAxis.SIM_TRANS: 0},
            sim_data=sim_data,
            sim_field=sim_field,
            sim_time_idx=sim_time_idx,
            sim_coord_units="mm",
            sim_coord_frame="cube",
        )


def test_background_defaults_and_direct_style_options(
    scene: tuple[valid.ValidResult, io.SimData],
) -> None:
    result, sim_data = scene
    plotter = validvis.plot_valid_metric_on_sim(
        result,
        "temperature",
        valid.EMetricType.MAVM,
        output=valid.EMetricOutput.D_PLUS,
        selectors={valid.EValidResAxis.TIME: 1, valid.EValidResAxis.SIM_TRANS: 0},
        sim_data=sim_data,
        sim_field="temperature",
        sim_time_idx=1,
        sim_coord_units="mm",
        sim_coord_frame="cube",
        sim_cmap="plasma",
        sim_colour_lims=(0.0, 500.0),
    )
    try:
        assert plotter.actors["validation-background"].mapper.scalar_range == (
            0.0,
            500.0,
        )
        assert "Sim.: temperature" in plotter.scalar_bars
        assert not any(name.startswith("validation-labels") for name in plotter.actors)
    finally:
        plotter.close()
