from pathlib import Path
import numpy as np

'''
    prefix = "check_"
    save_path = Path.home()/"test"/"pyvale-check"
    np.save(save_path/(prefix+"coord_raster"),coords_raster)
    np.save(save_path/(prefix+"connect_in_frame"),connect_in_frame)
    np.save(save_path/(prefix+"elem_bound_box_inds"),elem_bound_box_inds)
    np.save(save_path/(prefix+"elem_areas"),elem_areas)
'''

def main() -> None:
    check_path = Path.home()/"test"/"pyvale-check"

    check_vars = ("image_buff_subpx",
                "depth_buff_subpx",
                "fields_div_z",)

    exp_pref = "exp_"
    check_pref = "check_"

    suffix = ".npy"

    for vv in check_vars:
        check_data = np.load(check_path/(check_pref+vv+suffix))
        exp_data = np.load(check_path/(exp_pref+vv+suffix))

        print(80*"-")
        print(f"Checking: {vv}")
        print(f"{check_data.shape=}")
        print(f"{exp_data.shape=}")
        print(f"{np.allclose(exp_data,check_data)=}")
        print(80*"-")


if __name__ == "__main__":
    main()