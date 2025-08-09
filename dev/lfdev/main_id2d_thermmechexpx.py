from pathlib import Path
import numpy as np
import pyvale.mooseherder as mh
import pyvale as pyv

def print_sim_data(sim_data: mh.SimData) -> None:
    print(80*"-")
    if sim_data.time is not None:
        print(f"{sim_data.time.shape=}")
    print()

    if sim_data.coords is not None:
        print(f"{sim_data.coords.shape=}")
    print()

    def print_dict(in_dict: dict | None) -> None:
        if in_dict is None:
            return

        print(f"{in_dict.keys()=}")
        for kk in in_dict:
            print(f"    {kk}.shape={in_dict[kk].shape}")

        print()

    print_dict(sim_data.connect)
    print_dict(sim_data.node_vars)
    print_dict(sim_data.elem_vars)
    print_dict(sim_data.glob_vars)

    print(80*"-")


def main() -> None:
    # Load simulation data:
    main_path = Path.home()/"test"/"pyvDIC2D"
    sim_path = main_path/"data"/"dobone3d_ODIN_plas_ad_out.e"
    sim_data = mh.ExodusLoader(sim_path).read_all_sim_data()

    print_sim_data(sim_data)


    # Skin the mesh
    #surf_data = pyv.extract_surf_mesh(sim_data)








if __name__ == "__main__":
    main()