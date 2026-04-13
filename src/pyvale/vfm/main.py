# Enable postponed evaluation of type annotations (helps with typing and imports)
from __future__ import annotations

# Library for building command-line interfaces
import argparse

# Modern path handling object for file paths
from pathlib import Path

import numpy as np

# Functions from the pyvale Virtual Fields Method toolkit
from pyvale.vfm.identification_manager import run_identification
from pyvale.vfm.project_definition import PhaseResult, create_default_project
from pyvale.vfm.project_io import load_project
from pyvale.vfm.ui import launch_gui


def _format_phase_parameter_results(result: PhaseResult) -> list[str]:
    """Build a simple readable summary of the identified parameter values."""

    summary_lines: list[str] = []

    for parameter_name in sorted(result.parameter_maps):
        parameter_map = result.parameter_maps[parameter_name]
        parameter_state = result.parameter_states.get(parameter_name)
        dofs = [] if parameter_state is None else parameter_state.collect_dofs()

        if len(dofs) == 1:
            summary_lines.append(
                f"  {parameter_name}: {dofs[0].value:.6g}"
            )
            continue

        finite_values = parameter_map[np.isfinite(parameter_map)]
        if finite_values.size == 0:
            summary_lines.append(f"  {parameter_name}: no finite values")
            continue

        mean_value = float(np.mean(finite_values))
        min_value = float(np.min(finite_values))
        max_value = float(np.max(finite_values))

        if np.allclose(finite_values, finite_values[0]):
            summary_lines.append(f"  {parameter_name}: {mean_value:.6g}")
        else:
            summary_lines.append(
                f"  {parameter_name}: mean={mean_value:.6g}, "
                f"min={min_value:.6g}, max={max_value:.6g}"
            )

    return summary_lines


def main() -> None:
    # Create a command-line argument parser
    parser = argparse.ArgumentParser(
        description="Run the pyvale virtual fields method toolkit."
    )

    # Optional argument specifying a YAML project configuration file
    parser.add_argument(
        "--project",
        type=Path,
        help="Path to a YAML project file.",
    )

    # Optional flag to open the Qt GUI interface
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the thin Qt toolkit shell before running.",
    )

    # Optional flag to run the identification algorithm
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run identification after loading the project.",
    )

    # Parse the command-line arguments into the 'args' object
    args = parser.parse_args()

    # Load an existing project if a YAML file is provided,
    # otherwise create a default project configuration
    if args.project is not None:
        project = load_project(args.project)
    else:
        project = create_default_project()

    # Launch the GUI if the --gui flag was passed
    if args.gui:
        project = launch_gui(project)

    # Run the identification routine if the --run flag was passed
    if args.run:
        results = run_identification(project)

        # Print results for each identified phase
        for result in results:
            print(
                f"{result.phase_name}: cost={result.cost:.6g}, "
                f"metrics={result.metric_values}"
            )
            print("Identified parameters:")
            for line in _format_phase_parameter_results(result):
                print(line)


# Run main() only when this file is executed directly
if __name__ == "__main__":
    main()


# example usage:
# python -m pyvale.vfm.main --gui --run
# python -m pyvale.vfm.main --project /home/robh/1_Projects/vfmap-numerical-paper/data/notchedButtWeld_bilin_lin360420S_hom3700H_imDef_1.5/5-testData/vfm_project.yaml --run
