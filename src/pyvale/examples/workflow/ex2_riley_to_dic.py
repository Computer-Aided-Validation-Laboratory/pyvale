# %%
"""Riley render to DIC workflow
============================

This compact parameter-study example uses Riley, pyvale's default 3D renderer,
to generate a reference and a translated image before correlating them with
pyvale DIC.  Riley and DIC remain ordinary functions in a generic linear
workflow; replacing either stage does not require a new workflow class.
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.dic as dic
import pyvale.render as render
import pyvale.workflow as workflow
import riley


def _verify_empty_input(
    case: workflow.WorkflowCase,
    context: workflow.WorkflowContext,
    input_data: None,
) -> None:
    """Verify that the rendering stage is the first pipeline stage."""
    if input_data is not None:
        raise ValueError("The rendering step expects no preceding payload.")


def _make_scene(offset: float) -> tuple[render.Mesh, render.Camera]:
    """Build one textured triangle scene for a Riley render request."""
    coords = np.array(((-0.8 + offset, -0.8, 0.0),
                       (0.8 + offset, -0.8, 0.0),
                       (0.0 + offset, 0.8, 0.0)))
    shader = render.FunctionShader(
        riley.FuncShaderBuiltin.checker,
        riley.FuncCoordMode.world_reference,
    )
    mesh = render.Mesh(
        render.EElementType.TRI3,
        coords,
        np.array(((0, 1, 2),)),
        shader,
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
    """Create fresh Riley renderers and return DIC-ready image arrays."""
    reference_mesh, camera = _make_scene(0.0)
    deformed_mesh, _ = _make_scene(float(case.values["offset_world"]))
    config = riley.create_raster_config(
        int(case.values["sub_sample"]),
        save_strategy=riley.SaveStrategy.memory,
    )
    reference = render.Riley(config).render([reference_mesh], [camera]).images
    deformed = render.Riley(config).render([deformed_mesh], [camera]).images
    if reference is None or deformed is None:
        raise RuntimeError("Riley did not return an in-memory image.")
    return {
        "reference": _to_uint8(reference[0, 0, :, :, 0]),
        "deformed": _to_uint8(deformed[0, 0, :, :, 0]),
    }


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Map a normalised Riley image onto the DIC 8-bit dynamic range."""
    return np.rint(255.0 * np.clip(image, 0.0, 1.0)).astype(np.uint8)


def _verify_images(
    case: workflow.WorkflowCase,
    context: workflow.WorkflowContext,
    images: dict[str, np.ndarray],
) -> None:
    """Verify that the rendering stage supplied both image arrays."""
    if set(images) != {"reference", "deformed"}:
        raise ValueError("Expected reference and deformed Riley images.")


def _run_dic(
    case: workflow.WorkflowCase,
    context: workflow.WorkflowContext,
    images: dict[str, np.ndarray],
) -> dict[str, float]:
    """Run DIC in a case-private directory and return compact parameters."""
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
        num_threads=context.config.threads_per_case,
        output_basepath=output_dir,
        output_prefix="riley_dic_",
        debug_level=0,
    )
    context.add_artifact(output_dir)
    return {"offset_world": float(case.values["offset_world"])}


def run_example(output_dir: Path = Path("workflow-riley-dic-output")) -> None:
    """Run a small numeric and categorical Riley-to-DIC parameter study."""
    design = workflow.FullFactorial((
        workflow.ParameterValues(
            "offset_world", workflow.EParameterKind.NUMERIC, (0.05,),
        ),
        workflow.ParameterValues(
            "sub_sample", workflow.EParameterKind.NUMERIC, (1,),
        ),
        workflow.ParameterValues(
            "shader_name", workflow.EParameterKind.CATEGORICAL, ("checker",),
        ),
        workflow.ParameterValues(
            "subset_size", workflow.EParameterKind.NUMERIC, (21,),
        ),
        workflow.ParameterValues(
            "subset_step", workflow.EParameterKind.NUMERIC, (10,),
        ),
    ))
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
    run_example()
