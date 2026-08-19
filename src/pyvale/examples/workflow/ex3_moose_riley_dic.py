# %%
"""MOOSE to Riley to DIC workflow
==============================

This is an end-to-end template for an explicit subset of MOOSE input cases.
It deliberately uses :class:`pyvale.workflow.ExplicitCases`: a workflow case
can vary the simulation input, renderer controls, and DIC controls together
without forcing a full factorial product. Set ``PYVALE_MOOSE_MAIN`` and
``PYVALE_MOOSE_APP`` before running it with a local MOOSE application.

The functions are ordinary workflow functions. MOOSE, Riley, and DIC are not
special workflow-core classes.
"""

import os
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

import pyvale.dic as dic
from pyvale.mooseherder import ExodusLoader, MooseConfig, MooseRunner
import pyvale.render as render
import pyvale.workflow as workflow
import riley


def _verify_case(case: workflow.WorkflowCase) -> None:
    """Check explicit simulation and image-processing choices before workers."""
    input_file = Path(str(case.values["input_file"]))
    if not input_file.is_file():
        raise FileNotFoundError(input_file)
    if int(case.values["subset_size"]) <= 0:
        raise ValueError("subset_size must be positive.")


def _verify_empty(
    case: workflow.WorkflowCase,
    context: workflow.WorkflowContext,
    payload: None,
) -> None:
    """Verify that MOOSE is the first stage in the pipeline."""
    if payload is not None:
        raise ValueError("The MOOSE stage expects no preceding payload.")


def _run_moose(
    case: workflow.WorkflowCase,
    context: workflow.WorkflowContext,
    payload: None,
) -> object:
    """Run one copied MOOSE input and load its case-private Exodus output."""
    input_path = Path(str(case.values["input_file"]))
    local_input = context.case_dir / input_path.name
    shutil.copy2(input_path, local_input)
    config = MooseConfig({
        "main_path": Path(os.environ["PYVALE_MOOSE_MAIN"]),
        "app_path": Path(os.environ["PYVALE_MOOSE_APP"]),
        "app_name": str(case.values["moose_app"]),
    })
    runner = MooseRunner(config)
    runner.set_run_opts(
        n_tasks=1,
        n_threads=context.config.threads_per_case,
        redirect_out=True,
    )
    runner.set_input_file(local_input)
    runner.run()
    output_path = local_input.with_name(f"{local_input.stem}_out.e")
    context.add_artifact(output_path)
    return ExodusLoader(output_path).load_all_sim_data()


def _verify_simdata(
    case: workflow.WorkflowCase,
    context: workflow.WorkflowContext,
    sim_data: object,
) -> None:
    """Confirm that MOOSE returned data before Riley scene construction."""
    if sim_data is None:
        raise ValueError("MOOSE did not produce simulation data.")


def _render_riley(
    case: workflow.WorkflowCase,
    context: workflow.WorkflowContext,
    sim_data: object,
) -> dict[str, np.ndarray]:
    """Convert MOOSE surface data and render reference/deformed Riley frames."""
    shader = render.FunctionShader(
        riley.FuncShaderBuiltin.checker,
        riley.FuncCoordMode.world_reference,
    )
    mesh = render.mesh_from_simdata(
        sim_data,
        shader,
        displacement_keys=("disp_x", "disp_y", "disp_z"),
        extract_surface=True,
    )
    camera = render.Camera(
        pixels_num=np.array((128, 128)),
        pixels_size=np.array((0.02, 0.02)),
        pos_world=np.array((0.0, 0.0, 2.0)),
        rot_world=Rotation.identity(),
        roi_cent_world=np.zeros(3),
        focal_length=1.0,
        sub_sample=int(case.values["sub_sample"]),
    )
    frames = 1 if mesh.displacements is None else mesh.displacements.shape[0]
    config = riley.create_raster_config(
        frames,
        total_threads=context.config.threads_per_case,
        save_strategy=riley.SaveStrategy.memory,
    )
    images = render.Riley(config).render([mesh], [camera]).images
    if images is None or images.shape[0] < 2:
        raise ValueError("MOOSE output needs a reference and deformed frame.")
    return {
        "reference": _to_uint8(images[0, 0, :, :, 0]),
        "deformed": _to_uint8(images[-1, 0, :, :, 0]),
    }


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Quantise normalised renderer output for the DIC engine."""
    return np.rint(255.0 * np.clip(image, 0.0, 1.0)).astype(np.uint8)


def _verify_images(
    case: workflow.WorkflowCase,
    context: workflow.WorkflowContext,
    images: dict[str, np.ndarray],
) -> None:
    """Verify that Riley produced reference and deformed image arrays."""
    if set(images) != {"reference", "deformed"}:
        raise ValueError("Expected reference and deformed Riley images.")


def _run_dic(
    case: workflow.WorkflowCase,
    context: workflow.WorkflowContext,
    images: dict[str, np.ndarray],
) -> dict[str, float]:
    """Correlate Riley images and retain the DIC output directory."""
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
        num_threads=context.config.threads_per_case,
        output_basepath=output_dir,
        output_prefix="moose_riley_dic_",
        debug_level=0,
    )
    context.add_artifact(output_dir)
    return {"subset_size": float(case.values["subset_size"])}


def run_example(input_file: Path) -> None:
    """Run two selected MOOSE-to-DIC parameter combinations."""
    design = workflow.ExplicitCases((
        {"input_file": str(input_file), "moose_app": "proteus-opt",
         "sub_sample": 1, "subset_size": 21, "subset_step": 10},
        {"input_file": str(input_file), "moose_app": "proteus-opt",
         "sub_sample": 2, "subset_size": 31, "subset_step": 10},
    ))
    pipeline = workflow.PipelineWorkflow((
        workflow.FunctionStep(_verify_empty, _run_moose),
        workflow.FunctionStep(_verify_simdata, _render_riley),
        workflow.FunctionStep(_verify_images, _run_dic),
    ), _verify_case)
    workflow.WorkflowRunner(
        workflow.WorkflowConfig("workflow-moose-riley-dic-output", workers=1),
    ).run(pipeline, list(design.generate(2026)))


if __name__ == "__main__":
    raise SystemExit(
        "Pass a local MOOSE input file to run_example() after setting the "
        "PYVALE_MOOSE_MAIN and PYVALE_MOOSE_APP environment variables.",
    )
