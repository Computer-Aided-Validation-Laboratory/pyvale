"""Plate-with-a-hole DIC convergence pipeline composition."""

from pathlib import Path

import numpy as np

import pyvale.workflow as workflow


def verify_case(case: workflow.WorkflowCase) -> None:
    """Validate the DIC/VSG relation before worker submission."""
    subset_size = int(case.values["subset_size"])
    subset_step = int(case.values["subset_step"])
    vsg_length = int(case.values["vsg_length_px"])
    window_size = 1 + (vsg_length - subset_size) / subset_step
    if window_size <= 0 or not window_size.is_integer() or int(window_size) % 2 == 0:
        raise ValueError("Invalid subset size, step, and VSG length.")


def verify_source(case, context, input_data) -> None:
    """Check the first pipeline payload."""
    if input_data is not None:
        raise ValueError("The source step expects no input.")


def verify_images(case, context, images) -> None:
    """Verify that a preceding step supplied the image mapping."""
    if not isinstance(images, dict):
        raise TypeError("Expected an image mapping from the previous step.")


def make_images(case, context, input_data) -> dict[str, np.ndarray]:
    """Create small images; replace with plate-hole image loading in a study."""
    pixels_x, pixels_y = np.meshgrid(np.arange(64), np.arange(64))
    reference = (100 + pixels_x + pixels_y).astype(np.uint8)
    deformed = np.roll(reference, int(case.values["deformation_level"]), axis=1)
    return {"reference": reference, "deformed": deformed}


def add_noise(case, context, images) -> dict[str, np.ndarray]:
    """Create independently noisy static and deformed image pairs."""
    return {
        "static_reference": workflow.add_grey_level_noise(
            images["reference"], context.rng,
        ),
        "static_deformed": workflow.add_grey_level_noise(
            images["reference"], context.rng,
        ),
        "reference": workflow.add_grey_level_noise(images["reference"], context.rng),
        "deformed": workflow.add_grey_level_noise(images["deformed"], context.rng),
    }


def calculate_metrics(case, context, images) -> dict[str, float]:
    """Calculate compact static-noise and deformed-signal metrics."""
    static_difference = images["static_deformed"].astype(float)
    static_difference -= images["static_reference"].astype(float)
    signal = images["deformed"].astype(float) - images["reference"]
    return {
        "noise_floor_mean": float(np.std(static_difference)),
        "signal_mean": float(np.mean(signal)),
    }


design = workflow.FullFactorial((
    workflow.ParameterValues(
        "subset_size", workflow.EParameterKind.NUMERIC, (21,),
    ),
    workflow.ParameterValues(
        "subset_step", workflow.EParameterKind.NUMERIC, (10,),
    ),
    workflow.ParameterValues(
        "vsg_length_px", workflow.EParameterKind.NUMERIC, (61,),
    ),
    workflow.ParameterValues(
        "deformation_level", workflow.EParameterKind.NUMERIC, (1,),
    ),
    workflow.ParameterValues("repeat", workflow.EParameterKind.NUMERIC, (0,)),
))
pipeline = workflow.PipelineWorkflow((
    workflow.FunctionStep(verify_source, make_images),
    workflow.FunctionStep(verify_images, add_noise),
    workflow.FunctionStep(verify_images, calculate_metrics),
), verify_case)
results = workflow.WorkflowRunner(
    workflow.WorkflowConfig(Path("workflow-platehole-output"), retain_artifacts=False),
).run(pipeline, list(design.generate(12)))
