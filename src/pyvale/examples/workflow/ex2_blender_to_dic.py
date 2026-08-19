# %%
"""Blender render to DIC workflow
==============================

This example connects the optional Blender renderer to :mod:`pyvale.dic`
through generic :class:`pyvale.workflow.FunctionStep` objects. The workflow
core does not know about Blender or DIC: each function below can be replaced by
an equivalent Riley, Feebee, or experimental-image implementation.

Blender is optional. The example exits cleanly when the ``blender`` extra is
not installed for Python 3.13.
"""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.dic as dic
import pyvale.render as render
import pyvale.workflow as workflow


def _verify_empty_input(
    case: workflow.WorkflowCase,
    context: workflow.WorkflowContext,
    input_data: None,
) -> None:
    """Verify that the first pipeline step receives no payload."""
    if input_data is not None:
        raise ValueError("The rendering step must be first in the pipeline.")


def _make_scene(offset: float) -> tuple[render.Mesh, render.Camera]:
    """Create a small, deterministically translated Blender triangle scene."""
    coords = np.array(((-0.8 + offset, -0.8, 0.0),
                       (0.8 + offset, -0.8, 0.0),
                       (0.0 + offset, 0.8, 0.0)))
    mesh = render.Mesh(
        render.EElementType.TRI3,
        coords,
        np.array(((0, 1, 2),)),
        object(),
    )
    camera = render.Camera(
        pixels_num=np.array((64, 64)),
        pixels_size=np.array((0.02, 0.02)),
        pos_world=np.array((0.0, 0.0, 2.0)),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3),
        focal_length=1.0,
    )
    return mesh, camera


def _render_images(
    case: workflow.WorkflowCase,
    context: workflow.WorkflowContext,
    input_data: None,
) -> dict[str, np.ndarray]:
    """Render reference and deformed images with independent scene calls."""
    reference_mesh, camera = _make_scene(0.0)
    deformed_mesh, _ = _make_scene(float(case.values["offset_world"]))
    renderer = render.Blender(
        render.BlenderConfig(context.case_dir / "blender", samples=1),
    )
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        reference_result = renderer.render([reference_mesh], [camera])
        deformed_result = renderer.render([deformed_mesh], [camera])
    if reference_result.images is None or deformed_result.images is None:
        raise RuntimeError("Blender did not return an in-memory image.")
    reference = reference_result.images[0, 0, :, :, 0]
    deformed = deformed_result.images[0, 0, :, :, 0]
    return {
        "reference": _to_uint8(reference),
        "deformed": _to_uint8(deformed),
    }


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalise a Blender image to the 8-bit DIC input dynamic range."""
    image = np.asarray(image, dtype=np.float64)
    minimum = float(image.min())
    maximum = float(image.max())
    if maximum == minimum:
        raise ValueError("Blender image has no intensity variation for DIC.")
    return np.rint(255.0 * (image - minimum) / (maximum - minimum)).astype(np.uint8)


def _verify_images(
    case: workflow.WorkflowCase,
    context: workflow.WorkflowContext,
    images: dict[str, np.ndarray],
) -> None:
    """Verify that the renderer supplied reference and deformed images."""
    if set(images) != {"reference", "deformed"}:
        raise ValueError("Expected reference and deformed images from Blender.")


def _run_dic(
    case: workflow.WorkflowCase,
    context: workflow.WorkflowContext,
    images: dict[str, np.ndarray],
) -> dict[str, float]:
    """Correlate the Blender images and retain the DIC result files."""
    output_dir = context.case_dir / "dic"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = images["reference"]
    dic.calculate_2d(
        reference=reference,
        deformed=images["deformed"],
        roi_mask=np.ones(reference.shape, dtype=np.uint8),
        seed=[reference.shape[1] // 2, reference.shape[0] // 2],
        subset_size=int(case.values["subset_size"]),
        subset_step=int(case.values["subset_step"]),
        shape_function="AFFINE",
        max_displacement=8,
        output_basepath=output_dir,
        output_prefix="blender_dic_",
        debug_level=0,
    )
    context.add_artifact(output_dir)
    return {"offset_world": float(case.values["offset_world"])}


def run_example(output_dir: Path = Path("workflow-blender-dic-output")) -> None:
    """Run one Blender-to-DIC case when the optional backend is available."""
    design = workflow.ExplicitCases(({
        "offset_world": 0.05,
        "subset_size": 21,
        "subset_step": 10,
    },))
    pipeline = workflow.PipelineWorkflow((
        workflow.FunctionStep(_verify_empty_input, _render_images),
        workflow.FunctionStep(_verify_images, _run_dic),
    ))
    results = workflow.WorkflowRunner(
        workflow.WorkflowConfig(output_dir, workers=1),
    ).run(pipeline, list(design.generate(2026)))
    if results[0].status is not workflow.ECaseStatus.COMPLETED:
        raise RuntimeError(results[0].error)


if __name__ == "__main__":
    if render.blender_available():
        run_example()
