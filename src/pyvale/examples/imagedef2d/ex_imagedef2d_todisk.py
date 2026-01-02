# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================


from pathlib import Path
import numpy as np

# Pyvale Imports
import pyvale.sensorsim as sens
import pyvale.mooseherder as mh
import pyvale.dataset as dataset


def main() -> None:
    sim_path = dataset.mechanical_2d_path()
    sim_data = mh.ExodusLoader(sim_path).load_all_sim_data()

    image_path = dataset.dic_pattern_5mpx_path()
    image_speckle = sens.ImageTools.load_image_greyscale(image_path)

    save_path = Path.cwd()/"pyvale-output"
    if not save_path.is_dir():
        save_path.mkdir(parents=True, exist_ok=True)

    coords = sim_data.coords
    connectivity = (sim_data.connect["connect1"]-1).T # Beware 0 indexing here
    disp_x = sim_data.node_vars["disp_x"][:,:]
    disp_y = sim_data.node_vars["disp_y"][:,:]

    print()
    print(80*"-")
    print(f"{coords.shape=}")
    print(f"{connectivity.shape=}")
    print(f"{disp_x.shape=}")
    print(f"{disp_y.shape=}")
    print(80*"-")


    cam_data = sens.CameraData2D(pixels_count=np.array((1040,1540)),
                                   leng_per_px=0.1e-3,
                                   bits=8,
                                   roi_cent_world=np.mean(coords,axis=0),
                                   subsample=3)
    id_opts = sens.ImageDefOpts(save_path=save_path,
                                  crop_on=True,
                                  add_static_ref=True)



    (upsampled_image,
     image_mask,
     image_input,
     disp_x,
     disp_y) = sens.ImageDef2D.preprocess(cam_data,
                                            image_speckle,
                                            coords,
                                            connectivity,
                                            disp_x,
                                            disp_y,
                                            id_opts,
                                            print_on=True)

    ff = -1
    disp = np.array((disp_x[:,ff],disp_y[:,ff])).T
    print(f"{disp.shape=}")


    sens.ImageDef2D.deform_images_to_disk(cam_data,
                                            upsampled_image,
                                            coords,
                                            connectivity,
                                            disp_x,
                                            disp_y,
                                            image_mask,
                                            id_opts,
                                            print_on=True)


if __name__ == "__main__":
    main()

