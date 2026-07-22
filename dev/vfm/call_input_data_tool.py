from pyvale.vfm.experimentdata import Edge, EdgeConditions, EEdgeCondition
from pyvale.vfm.inputdata import (
    AnsysConfig,
    CoordConfig,
    EFeDataSource,
    EForceUnits,
    ForceConfig,
    InputDataConfig,
    MooseConfig,
    TimeConfig,
)
from pyvale.vfm.inputdatafiles import (
    MultiFieldCsvFile,
    TxtFile,
)
from pyvale.vfm.inputdatapreprocessor import preprocess_input_data

ANSYS_DIR = "/Users/chris/work/example_input_data/plate-with-hole-hom-lin-hard"
FE_DATA_DIR = f"{ANSYS_DIR}/fe-data"
EXODUS_FILE = "/Users/chris/work/vfmverif/data/out_hole2d_plas_32f.e"


def _edge_conditions() -> EdgeConditions:
    return EdgeConditions(
        Edge(EEdgeCondition.Fixed, EEdgeCondition.Fixed),
        Edge(EEdgeCondition.Traction, EEdgeCondition.Fixed),
        Edge(EEdgeCondition.Free, EEdgeCondition.Free),
        Edge(EEdgeCondition.Free, EEdgeCondition.Free),
    )


def _common_configs() -> dict:
    """Config fields shared across both scenarios."""
    return dict(
        x=CoordConfig(TxtFile(f"{FE_DATA_DIR}/x_coordinates.txt")),
        y=CoordConfig(TxtFile(f"{FE_DATA_DIR}/y_coordinates.txt")),
        force=ForceConfig(
            MultiFieldCsvFile(
                f"{FE_DATA_DIR}/reaction_history.csv", "reaction_fy"
            ),
            units=EForceUnits.N,
        ),
        time=TimeConfig(
            MultiFieldCsvFile(f"{FE_DATA_DIR}/reaction_history.csv", "time"),
        ),
        thickness=0.001,
        edge_conditions=_edge_conditions(),
    )


def build_ansys_config() -> InputDataConfig:
    return InputDataConfig(
        **_common_configs(),
        data_source=EFeDataSource.ANSYS,
        ansys=AnsysConfig(
            fe_data_dir=FE_DATA_DIR,
            mesh_file="mesh2d_holeplate.msh",
        ),
    )


def build_moose_config() -> InputDataConfig:
    return InputDataConfig(
        **_common_configs(),
        data_source=EFeDataSource.MOOSE,
        moose=MooseConfig(
            exodus_file_path=EXODUS_FILE,
            grid_divs=101,
            plate_height=35e-3,
            plate_width=25e-3,
        ),
    )


def _print_output(x, y, strain, force, time) -> None:
    print(f"  x     : shape={x.shape}")
    print(f"  y     : shape={y.shape}")
    print(f"  strain: shape={strain.shape}")
    print(f"  force : shape={force.shape}")
    print(f"  time  : shape={time.shape}")


def main():
    print("=== ANSYS scenario ===")
    _print_output(*preprocess_input_data(build_ansys_config()))

    print("=== MOOSE scenario ===")
    _print_output(*preprocess_input_data(build_moose_config()))


if __name__ == "__main__":
    main()
