from pathlib import Path

from pyvale.vfm import (
    AnsysConfig,
    Edge,
    EdgeConditions,
    EEdgeCondition,
    ExperimentData,
    InputDataConfig,
    MooseConfig,
    process_input_data,
)

ANSYS_DIR = Path(
    "/Users/chris/work/example_input_data/plate-with-hole-hom-lin-hard/fe-data"
)
EXODUS_FILE = "/Users/chris/work/vfmverif/data/out_hole2d_plas_32f.e"


def build_example_ansys_config() -> InputDataConfig:
    return AnsysConfig(
        x_file= ANSYS_DIR / "x_coordinates.txt",
        y_file= ANSYS_DIR / "y_coordinates.txt",
        strain_xx_file= ANSYS_DIR / "eps_xx.txt",
        strain_yy_file= ANSYS_DIR / "eps_yy.txt",
        strain_xy_file= ANSYS_DIR / "eps_xy.txt",
        force_file= ANSYS_DIR / "reaction_history.csv",
        time_file= ANSYS_DIR / "time_values.txt",
        thickness=0.001,
        edge_conditions=EdgeConditions(
            min_x_edge=Edge(
                x=EEdgeCondition.Fixed,
                y=EEdgeCondition.Fixed
            ),
            max_x_edge=Edge(
                x=EEdgeCondition.Traction,
                y=EEdgeCondition.Fixed
            ),
            min_y_edge=Edge(
                x=EEdgeCondition.Free,
                y=EEdgeCondition.Free
            ),
            max_y_edge=Edge(
                x=EEdgeCondition.Free,
                y=EEdgeCondition.Free
            ),
        ),
        mesh_file=ANSYS_DIR / "mesh2d_holeplate.msh",
    )


def build_example_moose_config() -> InputDataConfig:
    return MooseConfig(
        exodus_file_path=EXODUS_FILE,
        height=35e-3,
        width=25e-3,
        thickness=0.001,
        grid_divs=101,
        edge_conditions=EdgeConditions(
            min_x_edge=Edge(
                x=EEdgeCondition.Fixed,
                y=EEdgeCondition.Fixed
            ),
            max_x_edge=Edge(
                x=EEdgeCondition.Traction,
                y=EEdgeCondition.Fixed
            ),
            min_y_edge=Edge(
                x=EEdgeCondition.Free,
                y=EEdgeCondition.Free
            ),
            max_y_edge=Edge(
                x=EEdgeCondition.Free,
                y=EEdgeCondition.Free
            ),
        )
    )


def main():
    print("=== ANSYS scenario ===")
    x = process_input_data(build_example_ansys_config())
    y = ExperimentData.load_from_file(x)
    print(y)


    print("=== MOOSE scenario ===")
    process_input_data(build_example_moose_config())


if __name__ == "__main__":
    main()
