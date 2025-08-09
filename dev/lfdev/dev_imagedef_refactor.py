"""
================================================================================
pyvale: the python computer aided validation engine

License: MIT
Copyright (C) 2025 The Computer Aided Validation Team
================================================================================
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pyvale.mooseherder as mh
import pyvale


def main() -> None:
    #---------------------------------------------------------------------------
    # LOAD FILES
    #sim_file = "platehole2d_out.e"
    sim_file = "mechplate2d_shear_out.e"
    sim_path = Path("dev/lfdev/imagedef_testsims") / sim_file
    sim_data = mh.ExodusLoader(sim_path).read_all_sim_data()

    image_path = pyvale.DataSet.dic_pattern_5mpx_path()
    image_speckle = pyvale.CameraTools.load_image(image_path)

    #pyvale.ImageDefDiags.plot_speckle_image(image_speckle)
    #plt.show()

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

    #---------------------------------------------------------------------------
    # INPUT DATA
    cam_data = pyvale.CameraData2D(pixels_count=np.array((1040,1540)),
                                   leng_per_px=0.1e-3,
                                   bits=8,
                                   roi_cent_world=np.mean(coords,axis=0),
                                   subsample=3)
    id_opts = pyvale.ImageDefOpts(crop_on=True,
                                  add_static_ref=True)


    #---------------------------------------------------------------------------
    # PRE-PROCESS IMAGES
    (upsampled_image,
     image_mask,
     image_input,
     disp_x,
     disp_y) = pyvale.ImageDef2D.preprocess(cam_data,
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

    # (def_image,
    #  def_image_subpx,
    #  subpx_disp_x,
    #  subpx_disp_y,
    #  def_mask) = pyvale.ImageDef2D.deform_one_image(upsampled_ref_image,
    #                                                 cam_data,
    #                                                 id_opts,
    #                                                 coords,
    #                                                 disp,
    #                                                 image_mask,
    #                                                 print_on=True)

    pyvale.ImageDef2D.deform_images_to_disk(cam_data,
                                            upsampled_image,
                                            coords,
                                            connectivity,
                                            disp_x,
                                            disp_y,
                                            image_mask,
                                            id_opts,
                                            print_on=True)

    #pyvale.ImageDefDiags.plot_speckle_image(image_input)
    #pyvale.ImageDefDiags.plot_speckle_image(image_mask)
    #pyvale.ImageDefDiags.plot_speckle_image(upsampled_image)
    # pyvale.ImageDefDiags.plot_speckle_image(def_image)
    # plt.show()


if __name__ == "__main__":
    main()

