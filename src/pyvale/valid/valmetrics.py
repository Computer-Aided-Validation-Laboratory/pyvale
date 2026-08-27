'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from typing import Any
from dataclasses import dataclass
from pathlib import Path
from multiprocessing.pool import Pool
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.interpolate import griddata
import pyvale

#-------------------------------------------------------------------------------
# pyvale generic exp data reader, need to merge into pyvale main

@dataclass(slots=True)
class ExpDataLoadOpts:
    file_ext: str = ".csv"
    delimiter: str = ","
    skip_header: int = 1
    threads_num: int | None = None


@dataclass(slots=True)
class ExpData:
    coords: dict[str,np.ndarray] | None = None
    time: dict[str,np.ndarray] | None = None
    fields: dict[str,np.ndarray] | None = None


def load_exp_data(data_path: Path,
                  field_slices: dict[str,slice],
                  frames: slice | None = None,
                  load_opts: ExpDataLoadOpts | None = None
                  ) -> dict[str,np.ndarray]:

    if not data_path.is_dir():
        raise FileNotFoundError("Data path does not exist.")

    if load_opts is None:
        load_opts = ExpDataLoadOpts()

    csv_files = list(data_path.glob("*" + load_opts.file_ext))
    csv_files = sorted(csv_files)

    # print(80*"-")
    # print("Debug load_exp_data:")
    # print(f"{csv_files[0]=}")
    # print(f"{csv_files[1]=}")
    # print(f"{csv_files[-1]=}")
    # print()
    # if frames is not None:
    #     slice_frames = csv_files[frames]
    #     print(f"{slice_frames[0]=}")
    #     print(f"{slice_frames[-1]=}")
    # print(80*"-")

    if frames is not None:
        csv_files = csv_files[frames]

    # We load the first csv to find out what shape of data we are expecting
    data = pd.read_csv(csv_files[0])
    data = data.to_numpy()

    # Using the first csv we initialise all our numpy arrays to the correct
    # shape to hold our data as shape=(num_frames,num_points,slice.len)
    field_data: dict[str,np.ndarray] = {}
    for ff in field_slices:
        # shape=(num_points,slice.len)
        field_temp = data[:,field_slices[ff]]
        # shape=(num_points,num_frames,slice.len)
        field_data[ff] = np.zeros((data.shape[0],
                                len(csv_files),
                                field_temp.shape[1]))
        field_data[ff][:,0,:] = field_temp

        #print(f"key={ff} , {field_data.shape=}")

    # if coord_slices is not None:
    #     coord_data: dict[str,np.ndarray] = {}
    #     for cc in coord_slices:
    #         # shape=(num_points,slice.len)
    #         coord_temp = data[:,field_slices[cc]]
    #         # shape=(num_points,num_frames,slice.len)
    #         coord_data[cc] = np.zeros((data.shape[0],
    #                             len(csv_files),
    #                             coord_temp.shape[1]))
    #         coord_data[cc][:,0,:] = coord_temp


    # We have loaded the first data frame so we can remove it now, then we will
    # loop over all the others and load them
    csv_files.pop(0)

    if load_opts.threads_num is not None:
        assert load_opts.threads_num > 0, (
            "Number of threads must be greater than 0.")

        with Pool(load_opts.threads_num) as pool:
            processes_with_id = []

            for ii,ff in enumerate(csv_files):
                args = (ff,
                        field_slices)

                process = pool.apply_async(_load_one_exp, args=args)
                processes_with_id.append({"process": process,
                                          "frame": ii+1,
                                          "file": ff})

            for pp in processes_with_id:
                frame_data = pp["process"].get()

                for kk in field_slices:
                    field_data[kk][:,pp["frame"],:] = frame_data[kk]

    else:
        for ii,ff in enumerate(csv_files):
            # print(f"Loading experiment data file: {ii+1}. From path:")
            # print(f"{ff}\n")

            data = pd.read_csv(ff)
            data = data.to_numpy()

            for kk in field_slices:
                # shape=(num_frames,num_points,slice.len)
                field_data[kk][:,ii+1,:] = data[:,field_slices[kk]]

    return field_data # dict[str,np.ndarray]

def _load_one_exp(path: Path,
                  field_slices: dict[str,slice],
                  ) -> tuple[dict[str,np.ndarray]]:

    data = pd.read_csv(path)
    data = data.to_numpy()

    exp_data: dict[str,np.ndarray] = {}
    for ff in field_slices:
        # shape=(num_points,slice.len)
        exp_data[ff] = data[:,field_slices[ff]]

    # if coord_slices is not None:
    #     coord_data = {}
    #     for cc in coord_slices:
    #         coord_data[cc] = data[:]
    # else:
    #     coord_data = None

    return exp_data

#-------------------------------------------------------------------------------

def load_sim_data_tup(data_path: Path, skip_header: int = 0
                  ) -> tuple[np.ndarray,np.ndarray]:
    csv_files = list(data_path.glob("*.csv"))
    csv_files = sorted(csv_files)

    data = np.genfromtxt(csv_files[0],
                         skip_header=skip_header,
                         dtype=np.float64,
                         delimiter=",")


    # Coords are the same for all FE files, store once:
    # fe_coords.shape = (num_nodes, coord[x,y]) = (num_nodes,2)
    sim_coords = data[:,0:2]

    # fe_data.shape = (num_files,num_nodes,disp[x,y,z])
    sim_disp = np.zeros((len(csv_files),sim_coords.shape[0],3))
    # The last three columns of the file are displacements
    sim_disp[0,:,:] = data[:,2:]

    # We have loaded the first data frame so we can remove it now
    csv_files.pop(0)

    for ii,ff in enumerate(csv_files):
        data = np.genfromtxt(ff,
                             skip_header=skip_header,
                             dtype=np.float64,
                             delimiter=",")
        sim_disp[ii+1,:,:] = data[:,2:]

    # fe_coords.shape = (num_nodes, coord[x,y]) = (num_nodes,2)
    # fe_data.shape = (num_files,num_nodes,disp[x,y,z])
    return (sim_coords,sim_disp)

def load_exp_data_tup(data_path: Path,
                  num_load: int | None = None,
                  run_para: int | None = None
                  ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:

    csv_files = list(data_path.glob("*.csv"))
    csv_files = sorted(csv_files)

    if num_load is not None:
        csv_files = csv_files[0:num_load]

    # Coords change with every DIC data file, need to account for this
    data = np.genfromtxt(csv_files[0],
                         dtype=np.float64,
                         delimiter=",",
                         skip_header=1)

    # print(f"{data.shape}")

    # exp_coords.shape=(num_files,num_points,coord[x,y,z])
    exp_coords = np.zeros((len(csv_files),data.shape[0],3))
    # exp_disp.shape=(num_files,num_points,disp[x,y,z])
    exp_disp = np.zeros((len(csv_files),data.shape[0],3))
    # exp_strain.shape=(num_files,num_points,strain[xx,yy,xy])
    exp_strain = np.zeros((len(csv_files),data.shape[0],3))

    exp_coords[0,:,:] = data[:,2:5]
    exp_disp[0,:,:] = data[:,5:8]
    exp_strain[0,:,:] = data[:,18:21]

    # print(f"{exp_coords[0,0,:]=}")
    # print(f"{exp_disp[0,0,:]=}")
    # print(f"{exp_strain[0,0,:]=}")

    # We have loaded the first data frame so we can remove it now
    csv_files.pop(0)

    if run_para is not None:

        with Pool(run_para) as pool:
            processes = []

            for ff in csv_files:
                processes.append(
                    pool.apply_async(_load_one_exp_tup, args=(ff,)))

            data_list = [pp.get() for pp in processes]

        exp_data = np.zeros((data.shape[0],9))
        exp_data[:,0:3] = data[:,2:5]
        exp_data[:,3:6] = data[:,5:8]
        exp_data[:,6:9] = data[:,18:21]
        data_list.insert(0,exp_data)

        # print(f"{len(data_list)=}")
        # print(f"{data_list[0].shape=}")
        # print(f"{data_list[1].shape=}")

        data_arr = np.stack(data_list)
        exp_coords = data_arr[:,:,0:3]
        exp_disp = data_arr[:,:,3:6]
        exp_strain = data_arr[:,:,6:9]

        # print()
        # print(f"{data_arr.shape=}")
        # print(f"{exp_coords.shape=}")
        # print(f"{exp_disp.shape=}")
        # print(f"{exp_strain.shape=}")#
        # print()

    else:
        for ii,ff in enumerate(csv_files):
            print(f"Loading experiment data file: {ii}.")
            data = np.genfromtxt(ff,
                            dtype=np.float64,
                            delimiter=",",
                            skip_header=1)
            exp_coords[ii,:,:] = data[:,2:5]
            exp_disp[ii,:,:] = data[:,5:8]
            exp_strain[ii,:,:] = data[:,18:21]


    # exp_coords.shape=(num_files,num_points,coord[x,y,z])
    # exp_disp.shape=(num_files,num_points,disp[x,y,z])
    # exp_strain.shape=(num_files,num_points,strain[xx,yy,xy])
    return (exp_coords,exp_disp,exp_strain)

def _load_one_exp_tup(path: Path) -> np.ndarray:
        data = np.genfromtxt(path,
                        dtype=np.float64,
                        delimiter=",",
                        skip_header=1)
        exp_data = np.zeros((data.shape[0],9))
        exp_data[:,0:3] = data[:,2:5]
        exp_data[:,3:6] = data[:,5:8]
        exp_data[:,6:9] = data[:,18:21]
        return exp_data

def fit_coord_matrix(coords: np.ndarray) -> np.ndarray:
    coord_mat = np.zeros((4,4))
    coord_mat[-1,-1] = 1

    # TODO: this should be an option, max-min/2 or mean
    origin = np.mean(coords,axis=0)
    coord_mat[:-1,-1] = origin

    # print(f"{origin=}")

    cov_mat = np.cov(coords.T)
    (_,_,v_mat) = np.linalg.svd(cov_mat)

    # print(f"{cov_mat.shape=}")
    # print(f"{v_mat.shape=}")

    # V matrix contains eigen vectors which are the axes
    axis_x = v_mat[0,:]
    axis_y = v_mat[1,:]
    axis_z = v_mat[2,:]

    # print(v_mat)
    # print()
    # print(axis_x)
    # print(axis_y)
    # print(axis_z)

    # Ensure that the coordinate system is right-handed.
    if np.dot(np.cross(axis_x, axis_y), axis_z) < 0:
        axis_y = -axis_y

    coord_mat[:-1,0] = axis_x
    coord_mat[:-1,1] = axis_y
    coord_mat[:-1,2] = axis_z

    return coord_mat

def find_nearest_points(coords: np.ndarray,
                        find_point: np.ndarray,
                        k: int = 5) -> np.ndarray:
    if coords.shape[1] >= 3:
        coords = coords[:,0:2]

    distances = np.sqrt(np.sum((coords - find_point)**2,axis=1))
    return np.argsort(distances)[:k]

def plot_disp_comp_maps(sim_coords: np.ndarray,
                       sim_disp_avg: np.ndarray,
                       exp_coords_avg: np.ndarray,
                       exp_disp_avg: np.ndarray,
                       ax_ind: int,
                       ax_str: str,
                       scale_cbar: bool = True,
                       save_tag: str = "",
                       save_path: Path | None = None) -> None:

    if save_path is None:
        save_path = Path.cwd() / "images"

    sim_x_min = np.min(sim_coords[:,0])
    sim_x_max = np.max(sim_coords[:,0])
    sim_y_min = np.min(sim_coords[:,1])
    sim_y_max = np.max(sim_coords[:,1])

    step = 0.5
    x_vec = np.arange(sim_x_min,sim_x_max,step)
    y_vec = np.arange(sim_y_min,sim_y_max,step)
    (x_grid,y_grid) = np.meshgrid(x_vec,y_vec)

    exp_disp_grid_avg = griddata(exp_coords_avg[:,0:2],
                             exp_disp_avg[:,ax_ind],
                             (x_grid,y_grid),
                             method="linear")

    # This will do minimal interpolation as the input points are the same as the sim
    sim_disp_grid_avg = griddata(sim_coords[:,0:2],
                             sim_disp_avg[:,ax_ind],
                             (x_grid,y_grid),
                             method="linear")

    disp_diff_avg = sim_disp_grid_avg - exp_disp_grid_avg

    color_max = np.nanmax((np.nanmax(sim_disp_grid_avg),np.nanmax(exp_disp_grid_avg)))
    color_min = np.nanmin((np.nanmin(sim_disp_grid_avg),np.nanmin(exp_disp_grid_avg)))

    cbar_font_size = 6.0

    plot_opts = pyvale.sensorsim.PlotOptsGeneral()
    fig_size = (plot_opts.a4_print_width,plot_opts.a4_print_width/(plot_opts.aspect_ratio*2.8))

    fig,ax = plt.subplots(1,3,figsize=fig_size,layout='constrained')
    fig.set_dpi(plot_opts.resolution)

    if scale_cbar:
        image = ax[0].imshow(exp_disp_grid_avg,
                            extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max),
                            vmin = color_min,
                            vmax = color_max)
    else:
        image = ax[0].imshow(exp_disp_grid_avg,
                            extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max))

    ax[0].set_title(f"Exp. Avg. \ndisp. {ax_str} [mm]",
                    fontsize=plot_opts.font_head_size, fontname=plot_opts.font_name)
    ax[0].set_xlabel("x [mm]",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax[0].set_ylabel("y [mm]",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    cbar = plt.colorbar(image)
    cbar.ax.tick_params(labelsize=cbar_font_size)

    if scale_cbar:
        image = ax[1].imshow(sim_disp_grid_avg,
                            extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max),
                            vmin = color_min,
                            vmax = color_max)
    else:
        image = ax[1].imshow(sim_disp_grid_avg,
                            extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max))

    ax[1].set_title(f"Sim. Avg.\ndisp. {ax_str} [mm]",
                    fontsize=plot_opts.font_head_size, fontname=plot_opts.font_name)
    ax[1].set_xlabel("x [mm]",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax[1].set_ylabel("y [mm]",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    cbar = plt.colorbar(image)
    cbar.ax.tick_params(labelsize=cbar_font_size)

    image = ax[2].imshow(disp_diff_avg,
                         extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max),
                         cmap="RdBu")
    ax[2].set_title(f"(Sim. - Exp.)\ndisp. {ax_str} [mm]",
                    fontsize=plot_opts.font_head_size, fontname=plot_opts.font_name)
    ax[2].set_xlabel("x [mm]",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax[2].set_ylabel("y [mm]",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    cbar = plt.colorbar(image)
    cbar.ax.tick_params(labelsize=cbar_font_size)

    if scale_cbar:
        if save_tag:
            save_fig_path = save_path/f"disp_comp_{ax_str}_{save_tag}.png"
        else:
            save_fig_path = save_path/f"disp_comp_{ax_str}.png"
    else:
        if save_tag:
            save_fig_path = save_path/f"disp_comp_{ax_str}_cbarfree_{save_tag}.png"
        else:
            save_fig_path = save_path/f"disp_comp_{ax_str}_cbarfree.png"

    fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")


def plot_avg_field_maps_nosave(sim_coords: np.ndarray,
                       sim_disp_avg: np.ndarray,
                       exp_coords_avg: np.ndarray,
                       exp_disp_avg: np.ndarray,
                       ax_ind: int,
                       field_str: str,
                       scale_cbar: bool = True,
                       ) -> tuple[Any,Any]:

    sim_x_min = np.min(sim_coords[:,0])
    sim_x_max = np.max(sim_coords[:,0])
    sim_y_min = np.min(sim_coords[:,1])
    sim_y_max = np.max(sim_coords[:,1])

    step = 0.5
    x_vec = np.arange(sim_x_min,sim_x_max,step)
    y_vec = np.arange(sim_y_min,sim_y_max,step)
    (x_grid,y_grid) = np.meshgrid(x_vec,y_vec)

    exp_disp_grid_avg = griddata(exp_coords_avg[:,0:2],
                             exp_disp_avg[:,ax_ind],
                             (x_grid,y_grid),
                             method="linear")

    # This will do minimal interpolation as the input points are the same as the sim
    sim_disp_grid_avg = griddata(sim_coords[:,0:2],
                             sim_disp_avg[:,ax_ind],
                             (x_grid,y_grid),
                             method="linear")

    disp_diff_avg = sim_disp_grid_avg - exp_disp_grid_avg

    color_max = np.nanmax((np.nanmax(sim_disp_grid_avg),np.nanmax(exp_disp_grid_avg)))
    color_min = np.nanmin((np.nanmin(sim_disp_grid_avg),np.nanmin(exp_disp_grid_avg)))

    cbar_font_size = 6.0

    plot_opts = pyvale.sensorsim.PlotOptsGeneral()
    fig_size = (plot_opts.a4_print_width,plot_opts.a4_print_width/(plot_opts.aspect_ratio*2.8))
    fig,ax = plt.subplots(1,3,figsize=fig_size,layout='constrained')
    fig.set_dpi(plot_opts.resolution)

    if scale_cbar:
        image = ax[0].imshow(exp_disp_grid_avg,
                            extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max),
                            vmin = color_min,
                            vmax = color_max)
    else:
        image = ax[0].imshow(exp_disp_grid_avg,
                            extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max))

    ax[0].set_title(f"Exp. Avg. \n{field_str}",
                    fontsize=plot_opts.font_head_size, fontname=plot_opts.font_name)
    ax[0].set_xlabel("x [mm]",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax[0].set_ylabel("y [mm]",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    cbar = plt.colorbar(image)
    cbar.ax.tick_params(labelsize=cbar_font_size)

    if scale_cbar:
        image = ax[1].imshow(sim_disp_grid_avg,
                            extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max),
                            vmin = color_min,
                            vmax = color_max)
    else:
        image = ax[1].imshow(sim_disp_grid_avg,
                            extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max))

    ax[1].set_title(f"Sim. Avg.\n{field_str}",
                    fontsize=plot_opts.font_head_size, fontname=plot_opts.font_name)
    ax[1].set_xlabel("x [mm]",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax[1].set_ylabel("y [mm]",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    cbar = plt.colorbar(image)
    cbar.ax.tick_params(labelsize=cbar_font_size)

    image = ax[2].imshow(disp_diff_avg,
                         extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max),
                         cmap="RdBu")
    ax[2].set_title(f"(Sim. - Exp.)\n{field_str}",
                    fontsize=plot_opts.font_head_size, fontname=plot_opts.font_name)
    ax[2].set_xlabel("x [mm]",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax[2].set_ylabel("y [mm]",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    cbar = plt.colorbar(image)
    cbar.ax.tick_params(labelsize=cbar_font_size)

    return (fig,ax)


def _interp_one_instance(coords: np.ndarray,
                         disp: np.ndarray,
                         x_grid: np.ndarray,
                         y_grid: np.ndarray,
                         status: str | None = None) -> np.ndarray:

    if status is not None:
        print(f"Interpolating field components: {status}")

    disp_common = np.zeros((x_grid.size,3))

    for aa in range(0,3):
        sim_disp_grid = griddata(coords,
                                disp[:,aa],
                                (x_grid,y_grid),
                                method="linear")
        disp_common[:,aa] = sim_disp_grid.flatten()

    return disp_common


def interp_sim_to_common_grid(coords: np.ndarray,
                              disp: np.ndarray,
                              x_grid: np.ndarray,
                              y_grid: np.ndarray,
                              run_para: int | None = None) -> np.ndarray:


    if run_para is not None:
        with Pool(run_para) as pool:
            processes = []

            for ss in range(0,disp.shape[0]):
                status = f"{ss}/{disp.shape[0]}"
                processes.append(pool.apply_async(_interp_one_instance,
                                                  args=(coords[:,0:2],
                                                        disp[ss,:,:],
                                                        x_grid,
                                                        y_grid,
                                                        status)))

            data_list = [pp.get() for pp in processes]

        disp_common = np.stack(data_list)

        # print(80*"-")
        # print(f"{len(data_list)=}")
        # print(f"{data_list[0].shape=}")
        # print(f"{disp_common.shape=}")
        # print(80*"-")

        return disp_common

    # Non-parallel run
    disp_common = np.zeros_like(disp)

    for ss in range(0,disp.shape[0]):
        print(f"Interpolating: {ss}")
        for aa in range(0,3):
            disp_grid = griddata(coords[:,0:2],
                                 disp[ss,:,aa],
                                 (x_grid,y_grid),
                                 method="linear")
            disp_common[ss,:,aa] = disp_grid.flatten()

    return disp_common



def interp_exp_to_common_grid(coords: np.ndarray,
                              disp: np.ndarray,
                              x_grid: np.ndarray,
                              y_grid: np.ndarray,
                              run_para: None | int = None) -> np.ndarray:

    if run_para is not None:
        with Pool(run_para) as pool:
            processes = []

            for ss in range(0,disp.shape[0]):
                status = f"{ss}/{disp.shape[0]}"
                processes.append(pool.apply_async(_interp_one_instance,
                                                  args=(coords[ss,:,0:2],
                                                        disp[ss,:,:],
                                                        x_grid,
                                                        y_grid,
                                                        status)))

            data_list = [pp.get() for pp in processes]

        disp_common = np.stack(data_list)

        # print(80*"-")
        # print(f"{len(data_list)=}")
        # print(f"{data_list[0].shape=}")
        # print(f"{disp_common.shape=}")
        # print(80*"-")

        return disp_common

    # Non-parallel run
    disp_common = np.zeros((disp.shape[0],x_grid.size,3))

    for ss in range(0,disp.shape[0]):
        print(f"Interpolating: {ss}")
        for aa in range(0,3):
            disp_grid = griddata(coords[ss,:,0:2],
                                 disp[ss,:,aa],
                                 (x_grid,y_grid),
                                 method="linear")
            disp_common[ss,:,aa] = disp_grid.flatten()

    return disp_common

#-------------------------------------------------------------------------------
# MAVM Calculation
def mavm(model_data: np.ndarray,
         exp_data: np.ndarray,
         test: str | None = None
         ) -> dict[str,Any]:
    """
    Calculates the Modified Area Validation Metric.
    Adapted from Whiting et al., 2023, "Assessment of Model Validation, Calibration, and Prediction Approaches in the Presence of Uncertainty", Journal of Verification, Validation and Uncertainty Quantification, Vol. 8.
    Downloaded from http://asmedigitalcollection.asme.org/verification/article-pdf/8/1/011001/6974199/vvuq_008_01_011001.pdf on 24 May 2024.
    """

    # find empirical cdf
    model_cdf = stats.ecdf(model_data).cdf
    exp_cdf = stats.ecdf(exp_data).cdf

    F_mod_vec = np.copy(model_cdf.quantiles)
    Sn_exp_vec = np.copy(exp_cdf.quantiles)

    # NOTE: the CDF can have a different number of bins to the number of sims/experiments
    S_num_mod = F_mod_vec.shape[0]
    N_num_exp = Sn_exp_vec.shape[0]

    df = len(Sn_exp_vec)-1
    t_alph = stats.t.ppf(0.95,df)

    Sn_conf_exp_list = [
        Sn_exp_vec - t_alph*(np.nanstd(Sn_exp_vec)/np.sqrt(N_num_exp)),
        Sn_exp_vec + t_alph*(np.nanstd(Sn_exp_vec)/np.sqrt(N_num_exp))
    ]

    Sn_Y_exp_vec = np.copy(exp_cdf.probabilities)
    F_Y_mod_vec = np.copy(model_cdf.probabilities)

    p_F_mod_scalar = 1/S_num_mod
    p_Sn_exp_scalar = 1/N_num_exp

    if test is not None:
        print(80*"=")
        print(f"{model_data.shape=}")
        print(f"{model_cdf.quantiles.shape=}")
        print(f"{model_cdf.probabilities.shape=}")
        print()
        print(f"{exp_data.shape=}")
        print(f"{exp_cdf.quantiles.shape=}")
        print(f"{exp_cdf.probabilities.shape=}")
        print()

        print("EXP QUANTS")
        print(exp_cdf.quantiles)
        print("EXP PROBS")
        print(exp_cdf.probabilities)
        print("EXP CONF UP")
        print(Sn_conf_exp_list[1])
        print("EXP CONF LOW")
        print(Sn_conf_exp_list[0])
        print()
        print("MODEL QUANTS")
        print(model_cdf.quantiles)
        print("MODEL PROBS")
        print(model_cdf.probabilities)
        print()
        print(80*"=")

        save_path = Path.cwd()/"tests"
        if not save_path.is_dir():
            save_path.mkdir()

        model_cdf_save = np.vstack((model_cdf.probabilities,model_cdf.quantiles)).T
        np.savetxt(save_path/f"model_cdf_{test}.txt",model_cdf_save,delimiter=",")
        exp_cdf_save = np.vstack((exp_cdf.probabilities,exp_cdf.quantiles)).T
        np.savetxt(save_path/f"exp_cdf_{test}.txt",exp_cdf_save,delimiter=",")

        save_data = np.atleast_2d(exp_data).T
        np.savetxt(save_path/f"exp_data_{test}.txt",save_data,delimiter=",")

    d_conf_plus: list = []
    d_conf_minus: list = []

    tol = 1e-12



    for kk in [0,1]:
        if test is not None:
            print(80*"=")
            print(f"CONFIDENCE BOUND, k: {kk}")
            print()

        # USed to step through the longer of experiment vs simulation
        ii: int = 0

        d_rem: float = 0.0
        d_plus: float = 0.0
        d_minus: float = 0.0

        # Grab the lower [0] or upper [1] bound of the confidence interval
        Sn_exp_vec = np.copy(Sn_conf_exp_list[kk])

        #If more experimental data points than model data points
        if N_num_exp > S_num_mod:
            if test is not None:
                print(80*"=")
                print("More experiments than simulations")
                print(f"{(N_num_exp > S_num_mod)=}")
                print()

            # We step through the simulation as we have less simulation points
            for jj in range(len(F_mod_vec)):

                if test is not None:
                    print(80*"-")
                    print(f"Model: {jj}")
                    print(f"{d_rem=}")
                    print(f"{((d_rem >= tol) or (d_rem <= -tol))=}")
                    print(80*"-")
                    print()

                if (d_rem >= tol) or (d_rem <= -tol): # Fixed floating point issues here

                    d_ii = ((Sn_exp_vec[ii] - F_mod_vec[jj])
                          * (p_Sn_exp_scalar*(ii+1) - p_F_mod_scalar*jj))

                    if d_ii > 0:
                        d_plus += d_ii
                    else:
                        d_minus += d_ii

                    if test is not None:
                        print("REMAINDER BLOCK")
                        print(f"Experiment: {ii}")
                        print(f"{Sn_exp_vec[ii]=}")
                        print(f"{F_mod_vec[jj]=}")
                        print(f"{(Sn_exp_vec[ii] - F_mod_vec[jj])=}")
                        print()
                        print(f"{(p_Sn_exp_scalar*(ii+1))=}")
                        print(f"{(p_F_mod_scalar*jj)=}")
                        print(f"{(p_Sn_exp_scalar*(ii+1) - p_F_mod_scalar*jj)=}")
                        print()
                        print(f"{d_ii=}")
                        print(f"{d_plus=}")
                        print(f"{d_minus=}")
                        print()

                    ii += 1

                # While model prob is greater than experimental prob do:
                # This steps through the experiment until we hit the next simulation
                while (jj+1)*p_F_mod_scalar > (ii+1)*p_Sn_exp_scalar:
                    # Difference between exp conf bound and simulation multiplied
                    # by the simulation probability = Area
                    # NOTE: ERROR HERE - should be experiment prob
                    # d_ii = (Sn_exp_vec[ii] - F_mod_vec[jj])*p_F_mod_scalar
                    d_ii = (Sn_exp_vec[ii] - F_mod_vec[jj])*p_Sn_exp_scalar

                    if d_ii > tol:
                        d_plus += d_ii
                    else:
                        d_minus += d_ii

                    if test is not None:
                        print("WHILE BLOCK:")
                        print(f"Experiment: {ii}")
                        print(f"{Sn_exp_vec[ii]=}")
                        print(f"{F_mod_vec[jj]=}")
                        print(f"{(Sn_exp_vec[ii] - F_mod_vec[jj])=}")
                        print(f"{p_F_mod_scalar=}")
                        print(f"{p_Sn_exp_scalar=}")
                        print()
                        print(f"{d_ii=}")
                        print(f"{d_plus=}")
                        print(f"{d_minus=}")
                        print()

                    ii += 1

                #TODO: bug here for coil data, this is a rough fix. Remove this
                # if statement to replicate the bug
                if ii < len(Sn_exp_vec) and jj < len(F_mod_vec):

                    d_rem = (Sn_exp_vec[ii]-F_mod_vec[jj])*(p_F_mod_scalar*(jj+1) - p_Sn_exp_scalar*ii)

                    if d_rem > 0.0:
                        d_plus += d_rem
                    else:
                        d_minus += d_rem

                    if test is not None:
                        print("FINAL BLOCK:")
                        print(f"{Sn_exp_vec[ii]=}")
                        print(f"{F_mod_vec[jj]=}")
                        print(f"{(Sn_exp_vec[ii] - F_mod_vec[jj])=}")
                        print()

                        print(f"{d_plus=}")
                        print(f"{d_minus=}")
                        print(f"{d_rem=}")

            if test is not None:
                print()
                print(80*"=")

        #=======================================================================
        #If more model data points than experimental data points (more typical)
        elif N_num_exp <= S_num_mod:

            if test is not None:
                print(80*"=")
                print("More simulations than experiments")
                print(f"{(N_num_exp <= S_num_mod)=}")
                print()

            for jj in range(0,len(Sn_exp_vec)):

                if test is not None:
                    print(80*"-")
                    print(f"Model: {jj}")
                    print(f"{d_rem=}")
                    print(f"{((d_rem >= tol) or (d_rem <= -tol))=}")
                    print(80*"-")
                    print()

                if (d_rem >= tol) or (d_rem <= -tol):
                    d_ii = (Sn_exp_vec[jj]-F_mod_vec[ii])*(p_F_mod_scalar*(ii+1) - p_Sn_exp_scalar*jj)

                    if d_ii > tol:
                        d_plus += d_ii
                    else:

                        d_minus += d_ii

                    if test is not None:
                        print(f"REMAINDER BLOCK: kk={kk},mod[{ii}],exp[{jj}]")
                        print(f"Simulation: {ii}")
                        print(f"{Sn_exp_vec[jj]=}")
                        print(f"{F_mod_vec[ii]=}")
                        print(f"{(Sn_exp_vec[jj] - F_mod_vec[ii])=}")
                        print()
                        print(f"{(p_Sn_exp_scalar*jj)=}")
                        print(f"{(p_F_mod_scalar*(ii+1))=}")
                        print(f"{(p_F_mod_scalar*(ii+1) - p_Sn_exp_scalar*jj)=}")
                        print()
                        print(f"{d_ii=}")
                        print(f"{d_plus=}")
                        print(f"{d_minus=}")
                        print()

                    ii += 1

                while (ii+1)*p_F_mod_scalar < (jj+1)*p_Sn_exp_scalar:
                    d_ii = (Sn_exp_vec[jj]-F_mod_vec[ii])*p_F_mod_scalar
                    if d_ii > tol:
                        d_plus += d_ii
                    else:
                        d_minus += d_ii

                    if test is not None:
                        print("WHILE BLOCK:")
                        print(f"Simulation: {ii}")
                        print(f"{Sn_exp_vec[jj]=}")
                        print(f"{F_mod_vec[ii]=}")
                        print(f"{(Sn_exp_vec[jj] - F_mod_vec[ii])=}")
                        print(f"{p_F_mod_scalar=}")
                        print(f"{p_Sn_exp_scalar=}")
                        print()
                        print(f"{d_ii=}")
                        print(f"{d_plus=}")
                        print(f"{d_minus=}")
                        print()

                    ii += 1

                d_rem = (Sn_exp_vec[jj]-F_mod_vec[ii])*(p_Sn_exp_scalar*(jj+1) - p_F_mod_scalar*ii)
                if d_rem > tol:
                    d_plus += d_rem
                else:
                    d_minus += d_rem

                if test is not None:
                    print("FINAL BLOCK:")
                    print(f"{Sn_exp_vec[jj]=}")
                    print(f"{F_mod_vec[ii]=}")
                    print(f"{(Sn_exp_vec[jj] - F_mod_vec[ii])=}")
                    print()

                    print(f"{d_plus=}")
                    print(f"{d_minus=}")
                    print(f"{d_rem=}")

            if test is not None:
                print()
                print(80*"=")

        # This is a two element list
        d_conf_plus.append(np.abs(d_plus))
        d_conf_minus.append(np.abs(d_minus))


    d_plus = np.nanmax(d_conf_plus)
    d_minus = np.nanmax(d_conf_minus)

    if test is not None:
        print()
        print("CALCULATION END")
        print(f"{d_conf_plus=}")
        print(f"{d_conf_minus=}")
        print(f"{d_plus=}")
        print(f"{d_minus=}")
        print()

    output_dict = {"model_cdf":model_cdf,
                   "exp_cdf":exp_cdf,
                   "d+":d_plus,
                   "d-":d_minus,
                   "Sn_conf":Sn_conf_exp_list,
                   "F_":F_mod_vec,
                   "F_Y":F_Y_mod_vec,}

    return output_dict


def mavm_figs(mavm_res: dict[str,Any],
              title_str: str,
              field_label: str,
              field_tag: str = "",
              save_tag: str = "",
              save_path: Path | None = None) -> None:

    if save_path is None:
        save_path = Path.cwd() / "images"


    model_cdf = mavm_res["model_cdf"]
    exp_cdf = mavm_res["exp_cdf"]
    Sn_conf = mavm_res["Sn_conf"]
    d_plus = mavm_res["d+"]
    d_minus = mavm_res["d-"]
    F_ = mavm_res["F_"]
    F_Y = mavm_res["F_Y"]

    # fig,axs=plt.subplots(1,1)
    # model_cdf.plot(axs,label="model")
    # exp_cdf.plot(axs,label="experiment")
    # axs.legend()
    # axs.set_xlabel(field_label)
    # axs.set_ylabel("Probability")
    plot_opts = pyvale.sensorsim.PlotOptsGeneral()

    # plot empirical cdf with conf. int. cdfs
    fig,axs=plt.subplots(1,1,
                         figsize=plot_opts.single_fig_size_landscape,
                         layout="constrained")
    fig.set_dpi(plot_opts.resolution)

    axs.ecdf(model_cdf.quantiles,label="sim.",linewidth=plot_opts.lw)
    axs.ecdf(exp_cdf.quantiles,label="exp.",linewidth=plot_opts.lw)
    axs.ecdf(Sn_conf[0],ls="dashed",color="k",linewidth=plot_opts.lw,label=r"95% C.I.")
    axs.ecdf(Sn_conf[1],ls="dashed",color="k",linewidth=plot_opts.lw)
    axs.legend()

    axs.set_title(title_str,fontsize=plot_opts.font_head_size)
    axs.set_xlabel(field_label,fontsize=plot_opts.font_ax_size)
    axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)
    if save_tag:
        save_fig_path = save_path / f"mavm_ci_{field_tag}_{title_str}_{save_tag}.png"
    else:
        save_fig_path = save_path / f"mavm_ci_{field_tag}_{title_str}.png"

    fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

    fig,axs=plt.subplots(1,1,
                         figsize=plot_opts.single_fig_size_landscape,
                         layout="constrained")
    fig.set_dpi(plot_opts.resolution)

    axs.plot(F_,F_Y,"k-",linewidth=plot_opts.lw)
    axs.plot(F_+d_plus,F_Y,"k--",linewidth=plot_opts.lw)
    axs.plot(F_-d_minus,F_Y,"k--",linewidth=plot_opts.lw)
    axs.fill_betweenx(F_Y,F_-d_minus,F_+d_plus,color="k",alpha=0.2)

    # axs.plot(F_,F_Y,"k-")
    # axs.plot(d_plus,F_Y,"k--")
    # axs.plot(d_minus,F_Y,"k--")
    # axs.fill_betweenx(F_Y,d_minus,d_plus,color="k",alpha=0.2)


    axs.set_title(title_str,fontsize=plot_opts.font_head_size)
    axs.set_xlabel(field_label,fontsize=plot_opts.font_ax_size)
    axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)

    if save_tag:
        save_fig_path = save_path / f"mavm_fill_{field_tag}_{title_str}_{save_tag}.png"
    else:
        save_fig_path = save_path / f"mavm_fill_{field_tag}_{title_str}.png"

    fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")


def plot_mavm_map(mavm_d_plus: np.ndarray,
                  mavm_d_minus:np.ndarray,
                  ax_ind: int,
                  ax_str: str,
                  grid_shape: tuple[int,int],
                  extent: tuple[float,float,float,float],
                  save_tag: str = "",
                  field_str: str = "disp.",
                  unit_str: str = "mm",
                  save_path: Path | None = None) -> None:

    if save_path is None:
        save_path = Path("images")

    plot_opts = pyvale.sensorsim.PlotOptsGeneral()
    fig_size = (plot_opts.a4_print_width,plot_opts.a4_print_width/(plot_opts.aspect_ratio*2))

    fig,ax = plt.subplots(1,2,figsize=fig_size,layout='constrained')
    fig.set_dpi(plot_opts.resolution)

    mavm_dp_grid = np.reshape(mavm_d_plus[:,ax_ind],grid_shape)
    mavm_dm_grid = np.reshape(mavm_d_minus[:,ax_ind],grid_shape)

    image = ax[0].imshow(mavm_dp_grid,
                      extent=extent)
    cbar = plt.colorbar(image)
    ax[0].set_title(f"MAVM d+\n{field_str} {ax_str} [{unit_str}]")
    ax[0].set_xlabel("x [mm]",fontsize=plot_opts.font_ax_size)
    ax[0].set_ylabel("y [mm]",fontsize=plot_opts.font_ax_size)

    image = ax[1].imshow(mavm_dm_grid,
                      extent=extent)
    cbar = plt.colorbar(image)
    ax[1].set_title(f"MAVM d-\n{field_str} {ax_str} [{unit_str}]")
    ax[1].set_xlabel("x [mm]",fontsize=plot_opts.font_ax_size)
    ax[1].set_ylabel("y [mm]",fontsize=plot_opts.font_ax_size)

    if save_tag:
        image_path = save_path/f"mavm_map_{field_str}{ax_str}_{save_tag}.png"
    else:
        image_path = save_path/f"mavm_map_{field_str}{ax_str}.png"
    fig.savefig(image_path,dpi=300,format="png",bbox_inches="tight")


