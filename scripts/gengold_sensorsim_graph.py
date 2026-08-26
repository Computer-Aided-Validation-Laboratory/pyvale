# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

import pyvale.verif.pointsens as pointsens
import pyvale.verif.pointsensconst as pointsensconst
import pyvale.verif.pointsensgraph as pointsensgraph


def main() -> None:
    tag = "graph"

    print(80 * "=")
    print(f"Gold Output Generator for pyvale {tag} Point Sensors")
    print(80 * "=")
    print(f"Saving gold output to: {pointsensconst.GOLD_PATH}\n")

    print(f"Generating gold output for {tag} error graph point sensors...")
    pointsens.gen_gold_measurements(
        pointsensgraph.sens_arrays_graph_dict()
    )

    print(80 * "-")
    print("Gold output generation complete.\n")


if __name__ == "__main__":
    main()
