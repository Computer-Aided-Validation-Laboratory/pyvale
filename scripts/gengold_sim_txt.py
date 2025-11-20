#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

from pathlib import Path
import pyvale.dataset as dataset
import pyvale.mooseherder as mh
from pyvale.mooseherder.simsaver import (SimDataSaveOpts,
                                            ESaveFieldOpt,
                                            ESaveArray,
                                            save_sim_data_to_arrays)


def main() -> None:
    print(80*"-")
    print("Gold Output Generator: mooseherder sim data text io")
    print(80*"-")
    data_path = dataset.element_case_output_path(dataset.EElemTest.HEX20)
    sim_data = mh.ExodusLoader(data_path).load_all_sim_data()

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "tests" / "mooseherder" / "txt_gold"

    if not output_path.is_dir():
        raise FileNotFoundError(f"Gold output directory '{output_path}'" \
            "does not exist, check base directory")

    print(f"Saving gold output to:\n    {output_path.resolve()}")

    save_opts = SimDataSaveOpts(fields_save_by=ESaveFieldOpt.BOTH,
                                array_format = ESaveArray.BOTH,
                                sim_tag="hex20")

    save_sim_data_to_arrays(output_path,sim_data,save_opts)

    print("Saving gold output complete.\n")


if __name__ == "__main__":
    main()