# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Contract tests for the Feebee renderer scaffold."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import pyvale.render as render


def make_camera() -> render.Camera:
    """Create a small valid perspective camera."""
    return render.Camera(
        pixels_num=np.array((16, 16)),
        pixels_size=np.array((0.1, 0.1)),
        pos_world=np.array((0.0, 0.0, 2.0)),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3),
        focal_length=1.0,
    )


def make_mesh(shader: object) -> render.Mesh3D:
    """Create a valid front-facing triangular render mesh."""
    return render.Mesh3D(
        render.EElementType.TRI3,
        np.array(((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0))),
        np.array(((0, 1, 2),)),
        shader,
    )


def test_feebee_validates_scene_before_backend_dispatch() -> None:
    """Valid input reaches the intentional unavailable-backend boundary."""
    shader = render.FeebeeColourShader(np.ones((1, 1, 3)))
    renderer = render.Feebee()

    scene = render.Scene3D([make_mesh(shader)], [make_camera()])
    renderer.verify_input(scene)

    with pytest.raises(NotImplementedError, match="compiled rendering backend"):
        renderer._render(scene)


def test_feebee_aggregates_unsupported_scene_features() -> None:
    """Feebee rejects invalid material and explicit lights before rendering."""
    material = render.FeebeeMaterial(
        material_type=render.EFeebeeMaterialType.REFRACTIVE,
    )
    shader = render.FeebeeColourShader(np.ones((1, 1, 3)), material)
    light = render.Light(
        render.ELightType.POINT,
        np.zeros(3),
        np.array((0.0, 0.0, -1.0)),
        1.0,
    )

    with pytest.raises(render.RenderInputError) as exception:
        render.Feebee().verify_input(
            render.Scene3D(
                (make_mesh(shader),),
                (make_camera(),),
                (light,),
            )
        )

    assert {issue.code for issue in exception.value.issues} == {
        "UNSUPPORTED",
        "VALUE",
    }


def test_feebee_checks_higher_order_connectivity() -> None:
    """Feebee detects topology data that disagrees with its element type."""
    mesh = make_mesh(render.FeebeeColourShader(np.ones((1, 1, 3))))
    mesh.element_type = render.EElementType.QUAD9

    with pytest.raises(render.RenderInputError, match="9 nodes"):
        render.Feebee().verify_input(render.Scene3D([mesh], [make_camera()]))


def test_feebee_texture_shader_accepts_nodal_uvs() -> None:
    """Feebee accepts its backend-owned texture shader representation."""
    shader = render.FeebeeTextureShader(
        np.array(((0.0, 0.0), (1.0, 0.0), (0.5, 1.0))),
        np.ones((4, 4)),
    )

    render.Feebee().verify_input(
        render.Scene3D(
            (make_mesh(shader),),
            (make_camera(),),
        )
    )


def test_feebee_uses_a_static_colour_field_with_an_animated_mesh() -> None:
    """A single colour frame remains valid for every displacement frame."""
    mesh = make_mesh(render.FeebeeColourShader(np.ones((1, 1, 3))))
    mesh.displacements = np.zeros((2, 3, 3))

    render.Feebee().verify_input(render.Scene3D([mesh], [make_camera()]))


def test_feebee_reports_non_numeric_material_data_cleanly() -> None:
    """A malformed material is a validation issue rather than a type error."""
    material = render.FeebeeMaterial(
        material_type=render.EFeebeeMaterialType.REFRACTIVE,
        refractive_index="invalid",  # type: ignore[arg-type]
    )
    shader = render.FeebeeColourShader(np.ones((1, 1, 3)), material)

    with pytest.raises(render.RenderInputError, match="refractive_index"):
        render.Feebee().verify_input(
            render.Scene3D(
                (make_mesh(shader),),
                (make_camera(),),
            )
        )
