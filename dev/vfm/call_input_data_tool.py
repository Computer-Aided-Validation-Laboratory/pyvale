from pyvale.vfm.experimentdata import Edge, EdgeConditions, EEdgeCondition
from pyvale.vfm.inputdata import (
    CoordConfig,
    EForceUnits,
    ForceConfig,
    InputDataConfig,
    ROIConfig,
    StrainConfig,
    TimeConfig,
)
from pyvale.vfm.inputdatafiles import (
    MultiFieldCsvFile,
    NpyFile,
    TxtFile,
    YamlFile,
)
from pyvale.vfm.inputdatapreprocessor import preprocess_input_data


def main():
    config = InputDataConfig(
        CoordConfig(
            TxtFile("/Users/chris/work/example_input_data/plate-with-hole-hom-lin-hard/fe-data/x_coordinates.txt")
        ),
        CoordConfig(
            TxtFile("/Users/chris/work/example_input_data/plate-with-hole-hom-lin-hard/fe-data/y_coordinates.txt")
        ),
        StrainConfig(
            file=NpyFile(
                "/Users/chris/work/example_input_data/plate-with-hole-hom-lin-hard/vfm-input-data-260709-1739/strain.npy"
            ),
            timestep_dim_index=0,
            components_dim_index=1,
            y_dim_index=2,
            x_dim_index=3,
            xx_component_index=0,
            yy_component_index=1,
            xy_component_index=2
        ),
        ForceConfig(
            MultiFieldCsvFile(
                "/Users/chris/work/example_input_data/plate-with-hole-hom-lin-hard/fe-data/reaction_history.csv",
                "reaction_fy"
            ),
            units=EForceUnits.N
        ),
        TimeConfig(
            MultiFieldCsvFile(
                "/Users/chris/work/example_input_data/plate-with-hole-hom-lin-hard/fe-data/reaction_history.csv",
                "time"
            ),
        ),
        ROIConfig(
            YamlFile("/Users/chris/work/example_input_data/plate-with-hole-hom-lin-hard/vfm-input-data-260709-1739/region_of_interest.yaml")
        ),
        thickness=0.001,
        edge_conditions=EdgeConditions(
            Edge(
                EEdgeCondition.Fixed,
                EEdgeCondition.Fixed
            ),
            Edge(
                EEdgeCondition.Traction,
                EEdgeCondition.Fixed
            ),
            Edge(
                EEdgeCondition.Free,
                EEdgeCondition.Free
            ),
            Edge(
                EEdgeCondition.Free,
                EEdgeCondition.Free
            )
        )
    )

    preprocess_input_data(config)


if __name__ == "__main__":
    main()
