from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as np_test
import numpy.typing as npt
import pytest
import pyvista as pv

from pyvale import mooseherder, sensorsim
from pyvale.mooseherder.simdata import SimData
from pyvale.vfm.constitutive_laws.constitutive_parameter import (
    ConstitutiveParameter,
)
from pyvale.vfm.constitutive_laws.hardening_functions.linear import (
    LinearHardening,
)
from pyvale.vfm.constitutive_laws.isotropic_von_mises_elastoplasticity import (
    IsotropicVonMisesElastoplasticity,
)
from pyvale.vfm.experiment_data import (
    BoundaryConditions,
    Edge,
    EdgeConditions,
    EEdgeCondition,
    ExperimentData,
    SpecimenGeometry,
)
from pyvale.vfm.identification import Identification, IdentificationPhase
from pyvale.vfm.metrics.virtual_fields.sensitivity_based_virtual_fields import (
    SensitivityBasedVirtualFieldsMetric,
)
from pyvale.vfm.objective_functions.vector_first_result_passthrough import (
    VectorFirstResultPassthrough,
)
from pyvale.vfm.optimisers.least_squares import LeastSquares
from pyvale.vfm.spatial_parameterisations.homogeneous import (
    HomogeneousSpatialParameterisation,
)
from pyvale.vfm.spatial_parameterisations.known import (
    KnownSpatialParameterisation,
)
from pyvale.vfm.vfm import vfm
from pyvale.vfm.constitutive_laws.radial_return import EUnloading, radial_return

from interpolate_fe_elements_to_grid import (
    interpolate_fe_elements_to_grid,
    read_gmsh_element_centres,
)
 

PYVALE_ROOT = Path(__file__).resolve().parent.parent.parent
VFMVERIF_ROOT = PYVALE_ROOT.parent / "vfmverif"


def load_sim_data_to_grid(
    exodus_file_name: str,
    component_keys: tuple[str,...]
) -> tuple[
    npt.NDArray[np.float64], # x_grid
    npt.NDArray[np.float64], # y_grid
    npt.NDArray[np.float64], # grid_data
    npt.NDArray[np.float64], # force
    npt.NDArray[np.float64], # time
    npt.NDArray[np.float64], # yield_stress_out
]:
    exodus_file_path = VFMVERIF_ROOT / "data"/ exodus_file_name

    sim_data = mooseherder.ExodusLoader(exodus_file_path).load_all_sim_data()
    sensorsim.simtools.print_sim_data(sim_data)
    
    yield_stress_out = sim_data.elem_vars[('yield_stress_out', 1)]

    plate_height = 35e-3
    plate_width = 25e-3

    grid_divs = 101

    def grid_inner_vec(lower: float, upper: float, num_divs: int) -> np.ndarray:
        step = (upper - lower) / num_divs
        start = lower + (step / 2)
        stop = upper - (step / 2)
        return np.linspace(start, stop, num_divs)

    x_vec = grid_inner_vec(
        plate_width / 2,
        -plate_width / 2,
        grid_divs
    )

    y_vec = (
        grid_inner_vec(
            plate_height/ 2,
            -plate_height / 2,
            grid_divs
        ) + plate_height / 2
    )

    z_vec = np.full((1,), 0.0, dtype=np.float64)

    # x going from positive to negative down a col with 0 at row 50
    # y going from higher to lower along a row, always positive
    # TODO: does this need to be swapped around to fit our conventions?
    (x_grid, y_grid, z_grid) = np.meshgrid(x_vec, y_vec, z_vec, indexing='ij')

    # Stack them along a new first axis to create the (3, Nx, Ny, Nz) array
    interp_grid = np.stack([x_grid, y_grid, z_grid], axis=0)

    # interp_grid shape is (3, Nx, Ny, Nz) -> spatial_shape is (Nx, Ny, Nz)
    spatial_grid_shape = interp_grid.shape[1:]

    # Reshape to (N_total_points, 3)
    interp_points = interp_grid.reshape(3, -1).T

    pyvista_interp = sensorsim.simdata_to_pyvista_interp(
        sim_data,
        component_keys,
        sensorsim.EDim.THREED
    )
    pv_points = pv.PolyData(interp_points)
    sample_data = pv_points.sample(pyvista_interp)

    invalid = ~sample_data["vtkValidPointMask"].astype(bool)

    n_comps = len(component_keys)
    (n_sensors,n_time_steps) = np.array(sample_data[component_keys[0]]).shape
    sample_at_sim_time = np.empty((n_sensors,n_comps,n_time_steps))

    for ii,cc in enumerate(component_keys):
        data_mat = np.array(sample_data[cc])
        data_mat[invalid,:] = np.nan
        sample_at_sim_time[:,ii,:] = data_mat

    # Target: (Nx, Ny, Nz, n_comps, n_time_steps)
    final_shape = spatial_grid_shape + (n_comps, n_time_steps)
    grid_data = sample_at_sim_time.reshape(final_shape)

    return (
        x_grid,
        y_grid,
        grid_data, 
        sim_data.glob_vars["react_y_top"],
        sim_data.time,
        yield_stress_out
    )


# TODO: plot (for 1 timestep):
#   strain compnents
#   yield stress
#   all input stuff really
def test_end_to_end() -> None:
    print("Loading data...")
    exodus_file_name = "hole3d_plas_het_24f.e"

    (
        x_grid, # shape: (x, y, z)
        y_grid, # shape: (x, y, z)
        grid_data, # shape: (x, y, z, components, timesteps)
        force, # shape: (timesteps)
        time, # shape: (timesteps)
        yield_stress_out # shape: (timesteps)
    ) = load_sim_data_to_grid(
        exodus_file_name,
        ("strain_xx", "strain_yy", "strain_xy","vonmises_stress")
    )

    print("Shaping inputs...")
    # remove redundant z component
    x_grid = x_grid[:, :, 0] # shape: (x, y)
    y_grid = y_grid[:, :, 0] # shape: (x, y)
    grid_data = grid_data[:, :, 0, :, :] # shape: (x, y, components, timesteps)

    # reshape the grid and data to use our conventions
    x_grid = x_grid.transpose(1, 0) # shape: (y, x)
    y_grid = y_grid.transpose(1, 0) # shape: (y, x)
    grid_data = grid_data.transpose(3, 2, 1, 0) # shape: (timesteps, components, y, x)

    # update x_grid values to use our conventions:
    #   - x increases in value as column number increases
    #   - x is constant in each column
    #   - x is always positive
    x_grid = np.fliplr(x_grid)
    x_grid += np.nanmax(x_grid)

    # update y_grid values to use our conventions:
    #   - y increases as row number increases
    #   - y is constant in each row
    #   - y is always positive
    y_grid = np.flipud(y_grid)

    # flip grid data to keep it consistent with x_grid and y_grid
    grid_data = np.flip(grid_data, axis=2)
    grid_data = np.flip(grid_data, axis=3)

    # TODO: do we need to update shear strain?
    # need to plot shear stress vs shear strain, should have positive slope
    # lookup shear modulus

    # debug_plot(
    #     grid_data[-1, 3, :, :],
    #     x_grid,
    #     y_grid,
    # )

    specimen_mask = ~np.isnan(grid_data[0, 0, :, :])

    plate_thickness = 1e-3

    grid_element_area = (
        (x_grid[0, 1] - x_grid[0, 0])
        * (y_grid[1, 0] - y_grid[0, 0])
    )

    specimen_geometry = SpecimenGeometry(
        x_grid,
        y_grid,
        # TODO: get roi from sample data valid point mask
        specimen_mask,
        plate_thickness,
        np.full_like(x_grid, grid_element_area, dtype=np.float64)
    )

    # force = force * -1

    boundary_conditions = BoundaryConditions(
        EdgeConditions(
            min_x_edge=Edge(
                x=EEdgeCondition.Free,
                y=EEdgeCondition.Free
            ),
            max_x_edge=Edge(
                x=EEdgeCondition.Free,
                y=EEdgeCondition.Free
            ),
            min_y_edge=Edge(
                x=EEdgeCondition.Fixed,
                y=EEdgeCondition.Fixed
            ),
            max_y_edge=Edge(
                x=EEdgeCondition.Fixed,
                y=EEdgeCondition.Traction
            )
        ),
        # TODO: needs to be reversed to be y,x to be the right convention
        # and ensure the virtual displacement term also gets updated
        np.column_stack((np.zeros_like(force), force))
    )

    experiment_data = ExperimentData(
        grid_data,
        specimen_geometry,
        boundary_conditions,
        time
    )



    # Parameters
    YieldInf = 200      # MPa
    PeakYield = 240     # MPa

    plateWidth = 25e-3      # m, change as required
    plateHeight = 35e-3     # m, change as required

    centX = 0.0
    centY = plateHeight / 2

    stdX = plateWidth / 2
    stdY = plateWidth / 4

    # Create 101 x 101 coordinate grid
    nx = 101
    ny = 101

    x = np.linspace(-plateWidth / 2, plateWidth / 2, nx)
    y = np.linspace(0, plateHeight, ny)

    X, Y = np.meshgrid(x, y)

    # Yield stress field
    yield_stress_grid_analytical= YieldInf + (PeakYield - YieldInf) * np.exp(
        -0.5 * (
            ((X - centX) / stdX)**2 +
            ((Y - centY) / stdY)**2
        )
    )

    # plt.figure()
    # plt.imshow(
    #     Yield,
    #     extent=[x.min(), x.max(), y.min(), y.max()],
    #     origin="lower",
    #     aspect="auto"
    # )
    # plt.colorbar(label="Yield stress / Pa")
    # plt.xlabel("x / m")
    # plt.ylabel("y / m")
    # plt.axis("image")
    # plt.show()
   

    # # yield_stress_out is wrong shape (npts x timesteps)
    # plt.figure()
    # plt.imshow(
    #     yield_stress_out,
    #     origin="lower",
    #     aspect="auto"
    # )
    # plt.colorbar(label="Yield stress / Pa")
    # plt.xlabel("x / m")
    # plt.ylabel("y / m")
    # plt.axis("image")
    # plt.show()


    # Compare FE yield stess with analytical yield stress
    DATA_DIR = Path(__file__).resolve().parent.parent.parent / "dev" / "vfm" / "rob-data"
    MESH_PATH = DATA_DIR / "mesh3d_holeplate.msh"
    element_centres = read_gmsh_element_centres(MESH_PATH)

    yield_stress_fe_interpolated = interpolate_fe_elements_to_grid(
        element_centres,
        yield_stress_out,
        X,
        Y,
        value_scale=1e-6,
        specimen_mask=specimen_mask,
    )

    yield_stress_grid_analytical = yield_stress_grid_analytical.copy()
    yield_stress_grid_analytical[~specimen_mask] = np.nan

    diff_grid = yield_stress_fe_interpolated - yield_stress_grid_analytical

    print(f"{element_centres.shape[0]=}")
    print(f"{yield_stress_fe_interpolated.shape=}")
    print(f"yield_stress_fe_interpolated nanmean [MPa] = {np.nanmean(yield_stress_fe_interpolated):.6f}")
    print(f"analytical nanmean [MPa] = {np.nanmean(yield_stress_grid_analytical):.6f}")
    print(f"abs diff nanmax [MPa] = {np.nanmax(np.abs(diff_grid)):.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    plots = (
        (yield_stress_fe_interpolated, "Interpolated FE Yield [MPa]"),
        (yield_stress_grid_analytical, "Analytical Yield [MPa]"),
        (diff_grid, "Difference [MPa]"),
    )

    for ax, (field, title) in zip(axes, plots, strict=True):
        image = ax.imshow(
            field,
            extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
            origin="lower",
            aspect="equal",
            cmap="viridis",
        )
        fig.colorbar(image, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    plt.show()


    parameters = {
        "elastic_modulus": ConstitutiveParameter(
            200_000, 100_000, 250_000, np.array([101, 101])
        ),
        "poissons_ratio": ConstitutiveParameter(
            0.3, 0.2, 0.4, np.array([101, 101])
        ),
        # "yield_strength": ConstitutiveParameter(
        #     220, 100, 1000, np.array([101, 101])
        # ),
        "yield_strength": ConstitutiveParameter(
            yield_stress_grid_analytical, 100, 1000
        ),
        # TODO: what are the assumed units here, vfmverif value is
        # 1000 MPa
        "hardening_modulus": ConstitutiveParameter(
            1000, 500, 10_000, np.array([101, 101])
        ),
    }

    phases = [
        IdentificationPhase(
            {
                "elastic_modulus": HomogeneousSpatialParameterisation(),
                "poissons_ratio": HomogeneousSpatialParameterisation(),
                "yield_strength": HomogeneousSpatialParameterisation(),
                "hardening_modulus": HomogeneousSpatialParameterisation(),
            },
            [
                SensitivityBasedVirtualFieldsMetric(
                    experiment_data.specimen_geometry.x,
                    experiment_data.specimen_geometry.y,
                    experiment_data.specimen_geometry.region_of_interest,
                    experiment_data.boundary_conditions.edge_conditions,
                    np.array([15, 15]),
                )
            ],
            VectorFirstResultPassthrough(),
            LeastSquares(),
        )
    ]

    identification = Identification(
        IsotropicVonMisesElastoplasticity(
            LinearHardening()
        ),
        parameters,
        phases
    )


    constitutive_parameter_maps = {
        "elastic_modulus": parameters["elastic_modulus"].value,
        "poissons_ratio": parameters["poissons_ratio"].value,
        "yield_strength": parameters["yield_strength"].value,
        "hardening_modulus": parameters["hardening_modulus"].value,
    }

    strain = grid_data[:, 0:3, :, :]

    # checks confirmed shear strain is in tensorial convention
    # checks confirmed -1 * shear strain didnt help
    stress, equivalent_stress_rr, yield_map, equivalent_plastic_strain = radial_return(
            strain,
            constitutive_parameter_maps,
            constitutive_parameter_maps["elastic_modulus"],
            constitutive_parameter_maps["poissons_ratio"],
            LinearHardening(),
            unloading = EUnloading.NoCompensation,
        )

    # stress = identification.constitutive_law.calculate_stress(grid_data[:, 0:2, :, :], constitutive_parameter_maps)
    equivalent_stress_fe=grid_data[:, 3, :, :] * 1e-6



    step = 6
    data = equivalent_stress_fe[step, :, :]   
    data_name='equivalent_stress_fe'
    plt.figure()
    im1 = plt.imshow(data, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Stress')
    plt.xlabel('x')
    plt.ylabel('y')
    #include param name and dof index in title
    plt.title(f'{data_name}, step {step}')
    vmin = np.nanpercentile(data, 5)
    vmax = np.nanpercentile(data, 95)
    im1.set_clim(vmin, vmax)
    im1=plt.show()

    data = equivalent_stress_rr[step, :, :]   
    data_name='equivalent_stress_rr'
    plt.figure()
    im1 = plt.imshow(data, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Stress')
    plt.xlabel('x')
    plt.ylabel('y')
    #include param name and dof index in title
    plt.title(f'{data_name}, step {step}')
    vmin = np.nanpercentile(data, 5)
    vmax = np.nanpercentile(data, 95)
    im1.set_clim(vmin, vmax)
    im1=plt.show()

    data = equivalent_stress_rr[step, :, :] - equivalent_stress_fe[step, :, :]  
    data_name='equivalent_stress_rr-fe'
    plt.figure()
    im1 = plt.imshow(data, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Stress')
    plt.xlabel('x')
    plt.ylabel('y')
    #include param name and dof index in title
    plt.title(f'{data_name}, step {step}')
    vmin = np.nanpercentile(data, 5)
    vmax = np.nanpercentile(data, 95)
    im1.set_clim(vmin, vmax)
    im1=plt.show()


    data = (np.abs(equivalent_stress_rr[step, :, :] - equivalent_stress_fe[step, :, :]) / equivalent_stress_fe[step, :, :]) * 100
    data_name='equivalent_stress_rr-fe_abs_perc_diff'
    plt.figure()
    im1 = plt.imshow(data, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Stress')
    plt.xlabel('x')
    plt.ylabel('y')
    #include param name and dof index in title
    plt.title(f'{data_name}, step {step}')
    vmin = np.nanpercentile(data, 5)
    vmax = np.nanpercentile(data, 95)
    im1.set_clim(vmin, vmax)
    im1=plt.show()

    print("Running VFM...")
    vfm_result = vfm(experiment_data, identification)

    gold_parameters = {
        "elastic_modulus": np.full((101, 101), 200_000),
        "poissons_ratio": np.full((101, 101), 0.3),
        "yield_strength": np.full((101, 101), 200),
        # TODO: what are the assumed units here
        "hardening_modulus": np.full((101, 101), 1_000)
    }

    for param_name, param in vfm_result.items():
        print(f"{param_name}={param.value}")
        # np_test.assert_allclose(param_map, gold_parameters[param_name], rtol=, atol=)
        # np_test.assert_allclose(param.value, gold_parameters[param_name])


def debug_plot(data, x_grid, y_grid):
    fig, ax = plt.subplots()

    im = ax.pcolormesh(
        x_grid,
        y_grid,
        data,
        shading='auto',
        cmap='viridis'
    )

    vmin = np.nanpercentile(data, 5)
    vmax = np.nanpercentile(data, 95)
    im.set_clim(vmin, vmax)

    fig.colorbar(im, ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.invert_yaxis()

    plt.show()


test_end_to_end()
