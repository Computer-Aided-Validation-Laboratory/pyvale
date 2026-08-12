"""Prepare sampled WDBN1 MatchID data for VFM identification."""

from pathlib import Path

from pyvale.vfm import Edge, EdgeConditions, EEdgeCondition, ExperimentData, process_input_data
from pyvale.vfm.inputdatamatchidassembled import MatchIDAssembledConfig


DATA_DIR = Path(
    "/media/data/3_Resources/gr91-weld-dic-results/wdbn1/"
    "01-wdbn1-assembled-sampled-260812-1223"
)
OUTPUT_ROOT = DATA_DIR.parent / "pyvale-input"
THICKNESS_MM = 0.8


def main() -> None:
    config = MatchIDAssembledConfig(
        assembled_file=DATA_DIR / "assembled.h5",
        force_history_file=DATA_DIR / "force_history.csv",
        thickness=THICKNESS_MM,
        edge_conditions=EdgeConditions(
            min_x_edge=Edge(EEdgeCondition.Free, EEdgeCondition.Free),
            max_x_edge=Edge(EEdgeCondition.Free, EEdgeCondition.Free),
            min_y_edge=Edge(EEdgeCondition.Fixed, EEdgeCondition.Fixed),
            max_y_edge=Edge(EEdgeCondition.Free, EEdgeCondition.Traction),
        ),
    )

    experiment_data_file = process_input_data(config, output_root=OUTPUT_ROOT)
    experiment_data = ExperimentData.load_from_file(experiment_data_file)
    print(f"Prepared: {experiment_data_file}")
    print(f"Strain shape: {experiment_data.strain.shape}")
    print(f"ROI points: {experiment_data.specimen_geometry.region_of_interest.sample_specimen_mask(experiment_data.specimen_geometry.x, experiment_data.specimen_geometry.y).sum()}")


if __name__ == "__main__":
    main()
