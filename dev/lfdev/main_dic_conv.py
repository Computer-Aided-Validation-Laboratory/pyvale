# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pyvale as pyv

# TODO
# - Allow user to specify: subset size, shape function, step, VSG, VS shape
# - Perform sweep of parameters saving in different directories

@dataclass(slots=True)
class DICParams:
    subset: int
    step: int
    shape_fun: str


class DIC2DConvergence:

    def __init__(self,
                 subsets: list[int] | np.ndarray,
                 steps: list[int] | np.ndarray,
                 shape_funs: list[str]) -> None:
        
        self._subsets = subsets
        self._steps = steps
        self._shape_funs = shape_funs

    def set_subsets(self) -> None:

    def run_sweep(self) -> None:
        pass


def main() -> None:

    ref_img = pyv.DataSet.dic_plate_with_hole_ref()
    def_img = pyv.DataSet.dic_plate_with_hole_def()

    output_path = Path.cwd() / "pyvale-output"
    if not output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)

    subset_size = 31

    roi = pyv.DICRegionOfInterest(ref_img)
    roi_file = output_path / "roi.dat"

    if not roi_file.is_file():
        roi.interactive_selection(subset_size)
        roi.save_array(filename=roi_file, binary=False)
    else:
        roi.read_array(roi_file)
        roi.seed = [515,355]

    #---------------------------------------------------------------------------
    # Parameter Sweep
    subsets = [21,31]
    steps = [5,10]
    shape_funs = ["RIGID","AFFINE"]

    cases = {}
    for ss in subsets:
        for tt in steps:
            for ff in shape_funs:
                case_str = f"ss{ss}_st{tt}_sf{ff}"
                cases[case_str] = DICParams(subset=ss,
                                            step=tt,
                                            shape_fun=ff)



    print(cases)
    print(len(cases))





    # pyv.dic_2d(reference=ref_img,
    #           deformed=def_img,
    #           roi_mask=roi.mask,
    #           seed=roi.seed,
    #           subset_size=subset_size,
    #           subset_step=10,
    #           shape_function="AFFINE",
    #           max_displacement=10,
    #           correlation_criteria="ZNSSD",
    #           output_basepath=output_path,
    #           output_delimiter=",",
    #           output_prefix="dic_results_")



if __name__ == "__main__":
    main()