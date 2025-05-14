# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================


from pathlib import Path
import numpy as np
import mooseherder as mh
import pyvale as pyv


def main() -> None:
    sim_path = pyv.DataSet.mechanical_2d_path()
    sim_data = mh.ExodusReader(sim_path).read_all_sim_data()

    image_path = pyv.DataSet.dic_pattern_5mpx_path()
    image_speckle = pyv.CameraTools.load_image(image_path)

    save_path = Path.cwd()/"exampleoutput"

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


    cam_data = pyv.CameraData2D(pixels_count=np.array((1040,1540)),
                                   leng_per_px=0.1e-3,
                                   bits=8,
                                   roi_cent_world=np.mean(coords,axis=0),
                                   subsample=3)
    id_opts = pyv.ImageDefOpts(save_path=save_path,
                                  crop_on=True,
                                  add_static_ref=True)



    (upsampled_image,
     image_mask,
     image_input,
     disp_x,
     disp_y) = pyv.ImageDef2D.preprocess(cam_data,
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


    pyv.ImageDef2D.deform_images_to_disk(cam_data,
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

