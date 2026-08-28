#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

"""
Accessors for data that comes pre-packaged with pyvale for demonstrating its
functionality. This includes moose simulation outputs as exodus files, input
files for moose and gmsh for additional simulation cases, and images required
for testing the image deformation and digital image correlation modules.
"""

from enum import Enum
from importlib.resources import files
from pathlib import Path

import numpy as np
import riley

from pyvale.render.mesh import EElementType

SIM_CASE_COUNT = 26
"""Constant describing the number of simulation test case input files for moose
and gmsh that come packaged with pyvale.
"""

class EElemTest(Enum):
    """Enumeration used to specify different 3D element types for extracting
    specific test simulation datasets.
    """

    TET4 = "TET4"
    """Tetrahedral element, linear with 4 nodes.
    """

    TET10 = "TET10"
    """Tetrahedral element, quadratic with 10 nodes.
    """

    TET14 = "TET14"
    """Tetrahedral element, quadratic with 14 nodes.
    """

    HEX8 = "HEX8"
    """Hexahedral element, linear with 8 nodes.
    """

    HEX20 = "HEX20"
    """Hexahedral element, quadratic with 20 nodes.
    """

    HEX27 = "HEX27"
    """Hexahedral element, quadratic with 27 nodes.
    """

    def __str__(self):
        return self.value


class DataSetError(Exception):
    """Custom error class for file io errors associated with retrieving datasets
    and files packaged with pyvale.
    """


def _data_path(*parts: str) -> Path:
    """Return an installed path below the :mod:`pyvale.data` package."""
    return Path(files("pyvale.data").joinpath(*parts))


def sim_case_input_file_path(case_num: int) -> Path:
    """Gets the path to MOOSE input file (*.i) for a particular simulation
    case.

    Parameters
    ----------
    case_num : int
        Integer defining the case number to be retrieved. Must be greater
        than 0 and less than the number of simulation cases.

    Returns
    -------
    Path
        Path object to the MOOSE *.i file for the selected simulation case.

    Raises
    ------
    DataSetError
        Raised if an invalid simulation case number is specified.
    """
    if case_num <= 0:
        raise DataSetError("Simulation case number must be greater than 0")
    elif case_num > SIM_CASE_COUNT:
        raise DataSetError("Simulation case number must be less than " \
                            + f"{SIM_CASE_COUNT}")

    case_num_str = str(case_num).zfill(2)
    case_file = f"case{case_num_str}.i"
    return _data_path("simulation", "simcases", case_file)


def sim_case_gmsh_file_path(case_num: int) -> Path | None:
    """Gets the path to Gmsh input file (\*.geo) for a particular simulation
    case. Note that not all simulation cases use Gmsh for geometry and mesh
    generation. If the specified simulation case does not have an associated
    Gmsh \*.geo file. In this case 'None' is returned

    Parameters
    ----------
    case_num : int
        Integer defining the case number to be retrieved. Must be greater
        than 0 and less than the number of simulation cases.

    Returns
    -------
    Path | None
        Path object to the Gmsh *.geo file for the selected simulation case.
        Returns None if there is no *.geo for this simulation case.

    Raises
    ------
    DataSetError
        Raised if an invalid simulation case number is specified.
    """
    if case_num <= 0:
        raise DataSetError("Simulation case number must be greater than 0")
    elif case_num > SIM_CASE_COUNT:
        raise DataSetError("Simulation case number must be less than " \
                            + f"{SIM_CASE_COUNT}")

    case_num_str = str(case_num).zfill(2)
    case_file = f"case{case_num_str}.geo"
    case_path = _data_path("simulation", "simcases", case_file)

    if case_path.is_file():
        return case_path

    return None


def dic_pattern_5mpx_path() -> Path:
    """Path to a 5 mega-pixel speckle pattern image (2464 x 2056 pixels)
    with 8 bit resolution stored as a *.tiff. Speckles are sampled by
    5 pixels. A gaussian blur has been applied to the image to remove sharp
    transitions from black to white.

    Path
        Path to the *.tiff file containing the speckle pattern.
    """
    return _data_path(
        "render", "patterns", "optspeckle_2464x2056px_spec5px_8bit_gblur1px.tiff",
    )


def thermal_2d_path() -> Path:
    """Path to a MOOSE simulation output in exodus format. This case is a
    thermal problem solving for a scalar temperature field. The geometry is
    a 2D plate (in x,y) with a heat flux applied on one edge and a heat
    transfer coefficient applied on the opposite edge inducing a temperature
    gradient along the x axis of the plate.

    The simulation parameters can be found in the corresponding MOOSE input
    file: case18.i which can be retrieved using `sim_case_input_file_path`
    in this class.

    Returns
    -------
    Path
        Path to the exodus (``*.e``) output file for this simulation case.
    """
    return _data_path("simulation", "exodus", "case18_out.e")


def thermal_3d_path() -> Path:
    """Path to a MOOSE simulation output in exodus format. This case is a 3D
    thermal problem solving for a scalar temperature field. The model is a
    divertor armour monoblock composed of a tungsten block bonded to a
    copper-chromium-zirconium pipe with a pure copper interlayer. A heat
    flux is applied to the top surface of the block and a heat transfer
    coefficient for cooling water is applied to the inner surface of the
    pipe inducing a temperature gradient from the top of the block to the
    pipe.

    The simulation parameters can be found in the corresponding MOOSE input
    file: case16.i which can be retrieved using `sim_case_input_file_path`
    in this class. Note that this case uses a Gmsh *.geo file for geometry
    and mesh creation.

    Returns
    -------
    Path
        Path to the exodus (``*.e``) output file for this simulation case.
    """
    return _data_path("simulation", "exodus", "case16_out.e")


def mechanical_2d_path() -> Path:
    """Path to a MOOSE simulation output in exodus format. This case is a 2D
    plate with a hole in the center with the bottom edge fixed and a
    displacement applied to the top edge. This is a mechanical problem and
    solves for the displacement vector field and the tensorial strain field.

    The simulation parameters can be found in the corresponding MOOSE input
    file: case17.i which can be retrieved using `sim_case_input_file_path`
    in this class. Note that this case uses a Gmsh *.geo file for geometry
    and mesh creation.

    Returns
    -------
    Path
        Path to the exodus (``*.e``) output file for this simulation case.
    """
    return _data_path("simulation", "exodus", "case17_out.e")


def thermomechanical_2d_path() -> Path:
    """Path to a MOOSE simulation output in exodus format. This case is a
    thermo-mechanical analysis of a 2D plate with a heat flux applied on two
    edges and a heat transfer coefficient applied on the opposing edges. The
    mechanical deformation results from thermal expansion due to the imposed
    temperature gradient. This model is solved for the scalar temperature
    field, vector displacement and tensor strain field.

    The simulation parameters can be found in the corresponding MOOSE input
    file: case18.i which can be retrieved using `sim_case_input_file_path`
    in this class.

    Returns
    -------
    Path
        Path to the exodus (``*.e``) output file for this simulation case.
    """
    return _data_path("simulation", "exodus", "case18_out.e")


def thermomechanical_3d_path() -> Path:
    """Path to a MOOSE simulation output in exodus format. This case is a
    thermo-mechanical analysis of a 3D monoblock divertor armour with a heat
    flux applied on the top surface and a heat transfer coefficient applied
    on the inner surface of the pipe. The mechanical deformation results
    from thermal expansion due to the imposed temperature gradient.
    This model is solved for the scalar temperature field, vector
    displacement and tensor strain field.

    The simulation parameters can be found in the corresponding MOOSE input
    file: case16.i which can be retrieved using `sim_case_input_file_path`
    in this class.
    
    Returns
    -------
    Path
        Path to the exodus (``*.e``) output file for this simulation case.
    """
    return _data_path("simulation", "exodus", "case16_out.e")


def thermomechanical_2d_experiment_paths() -> list[Path]:
    """List of paths to MOOSE simulation output in exodus format. This case is a
    thermo-mechanical analysis of a 2D plate with a heat flux applied on one
    edge and a heat transfer coefficient applied on the opposing edge. The
    mechanical deformation results from thermal expansion due to the imposed
    temperature gradient. This model is solved for the scalar temperature
    field, vector temperature and tensor strain field.

    Here we analyse 2 separate experiments where the thermal conductivity of
    the material is perturbed from the nominal case by -10%.

    The simulation parameters can be found in the corresponding MOOSE input
    file: case18.i which can be retrieved using `sim_case_input_file_path`
    in this class.

    Returns
    -------
    list[Path]
        Paths to the exodus (``*.e``) output files for this simulated experiment.
    """
    return [
        _data_path("simulation", "exodus", "case18_out.e"),
        _data_path("simulation", "exodus", "case18_d_out.e"),
    ]

def thermomechanical_3d_experiment_paths() -> list[Path]:
    """List of paths to MOOSE simulation output in exodus format. This case is a
    thermo-mechanical analysis of a 3D monoblock divertor armour with a heat
    flux applied on the top surface and a heat transfer coefficient applied on 
    the inner surface of the pipe. The mechanical deformation results from 
    thermal expansion due to the imposed temperature gradient. This model is 
    solved for the scalar temperature field, vector displacement and tensor 
    strain field.

    Here we analyse 2 separate experiments where the thermal conductivity and
    thermal expansion coefficients of the material are perturbed from the 
    nominal case by -10%.

    The simulation parameters can be found in the corresponding MOOSE input
    file: case16.i which can be retrieved using `sim_case_input_file_path`
    in this class.

    Returns
    -------
    list[Path]
        Paths to the exodus (``*.e``) output files for this simulated experiment.
    """

    return [
        _data_path("simulation", "exodus", "case16_out.e"),
        _data_path("simulation", "exodus", "case16_d_out.e"),
    ]


def render_mechanical_3d_path() -> Path:
    """Path to a MOOSE simulation output in exodus format. This case is a
    purely mechanical test case in 3D meant for testing image rendering
    algorithms for digital image correlation simulation. The simulation
    consists of a linear elastic thin plate with a hole loaded in tension.
    The simulation uses linear tetrahedral elements for rendering tests.

    Returns
    -------
    Path
        Path to the exodus (``*.e``) output file for this simulation case.
    """
    return _data_path("simulation", "exodus", "case26_out.e")
    

def element_case_input_path(elem_type: EElemTest) -> Path:
    """Path to a MOOSE simulation input file (.i) for a simple test
    case. This case is a 10mm cube undergoing thermo-mechanical loading
    solved for the temperature, displacement and strain fields. This case is
    solved using a variety of tetrahedral and hexahedral elements with
    linear or quadratic shapes functions. These simulation cases are
    intended for testing purposes and contain a minimal number of elements.

    Parameters
    ----------
    elem_type : EElemTest
        Enumeration specifying the element type for this test case.

    Returns
    -------
    Path
        Path to the moose input file (.i) for this simulation case.
    """
    return _data_path("simulation", "simcases", f"case00_{elem_type.value}.i")



def element_case_output_path(elem_type: EElemTest) -> Path:
    """Path to a MOOSE simulation output in exodus format. This case is a
    10mm cube undergoing thermo-mechanical loading solved for the
    temperature, displacement and strain fields. This case is solved using a
    variety of tetrahedral and hexahedral elements with linear or quadratic
    shapes functions. These simulation cases are intended for testing
    purposes and contain a minimal number of elements.

    Parameters
    ----------
    elem_type : EElemTest
        Enumeration specifying the element type for this test case.

    Returns
    -------
    Path
        Path to the exodus (``*.e``) output file for this simulation case.
    """
    return _data_path(
        "simulation", "exodus", f"case00_{elem_type.value}_out.e",
    )


def dic_plate_with_hole_cam0_ref() -> Path:
    """
    Path to the reference image for the plate with hole example.
    1040x1540 image in .tiff format.

    Parameters
    ----------
    elem_type : EElemTest
        Enumeration specifying the element type for this test case.

    Returns
    -------
    Path
        Path to the reference image (``.tiff``).
    """
    return _data_path("dic", "plate_hole", "hole_cam0_frame00.tiff")

def dic_plate_with_hole_cam1_ref() -> Path:
    """
    Path to the reference image for the plate with hole example.
    1040x1540 image in .tiff format.

    Parameters
    ----------
    elem_type : EElemTest
        Enumeration specifying the element type for this test case.

    Returns
    -------
    Path
        Path to the reference image (``.tiff``).
    """
    return _data_path("dic", "plate_hole", "hole_cam1_frame00.tiff")


def dic_plate_with_hole_cam0_def() -> Path:
    """
    Path to the deformed images for the plate with hole example.
    1040x1540 image in .tiff format.

    Parameters
    ----------
    elem_type : EElemTest
        Enumeration specifying the element type for this test case.

    Returns
    -------
    Path
        Path to the reference image (``.tiff``).
    """
    return _data_path("dic", "plate_hole", "hole_cam0_frame*.tiff")

def dic_plate_with_hole_cam1_def() -> Path:
    """
    Path to the deformed images for the plate with hole example.
    1040x1540 image in .tiff format.

    Parameters
    ----------
    elem_type : EElemTest
        Enumeration specifying the element type for this test case.

    Returns
    -------
    Path
        Path to the reference image (``.tiff``).
    """
    return _data_path("dic", "plate_hole", "hole_cam1_frame*.tiff")



def dic_plate_with_hydro_cam0_ref() -> Path:
    """
    Path to the reference image for the plate with hydro example.
    1040x1540 image in .tiff format.

    Parameters
    ----------
    elem_type : EElemTest
        Enumeration specifying the element type for this test case.

    Returns
    -------
    Path
        Path to the reference image (``.tiff``).
    """
    return _data_path("dic", "plate_hydro", "hydro_cam0_frame00.tiff")

def dic_plate_with_hydro_cam1_ref() -> Path:
    """
    Path to the reference image for the plate with hydro example.
    1040x1540 image in .tiff format.

    Parameters
    ----------
    elem_type : EElemTest
        Enumeration specifying the element type for this test case.

    Returns
    -------
    Path
        Path to the reference image (``.tiff``).
    """
    return _data_path("dic", "plate_hydro", "hydro_cam1_frame00.tiff")


def dic_plate_with_hydro_cam0_def() -> Path:
    """
    Path to the deformed images for the plate with hydro example.
    1040x1540 image in .tiff format.

    Parameters
    ----------
    elem_type : EElemTest
        Enumeration specifying the element type for this test case.

    Returns
    -------
    Path
        Path to the reference image (``.tiff``).
    """
    return _data_path("dic", "plate_hydro", "hydro_cam0_frame*.tiff")

def dic_plate_with_hydro_cam1_def() -> Path:
    """
    Path to the deformed images for the plate with hydro example.
    1040x1540 image in .tiff format.

    Parameters
    ----------
    elem_type : EElemTest
        Enumeration specifying the element type for this test case.

    Returns
    -------
    Path
        Path to the reference image (``.tiff``).
    """
    return _data_path("dic", "plate_hydro", "hydro_cam1_frame*.tiff")



def dic_plate_rigid_cam0_ref() -> Path:
    """
    Path to the reference image for the rigid deformation example from camera 1 perspective.
    1040x1540 image in .tiff format.

    Parameters
    ----------
    elem_type : EElemTest
        Enumeration specifying the element type for this test case.

    Returns
    -------
    Path
        Path to the reference image (``.tiff``).
    """
    return _data_path("dic", "plate_rigid", "rigid_cam0_frame00.tiff")


def dic_plate_rigid_cam0_def() -> Path:
    """
    Path to the rigid deformation example images from camera 0 perspective.
    1040x1540 image in .tiff format.

    Returns
    -------
    Path
        Path to the deformation images (``.tiff``).
    """
    return _data_path("dic", "plate_rigid", "rigid_cam0_frame*.tiff")


def dic_plate_rigid_cam1_ref() -> Path:
    """
    Path to the reference image for the rigid deformation example from camera 1 perspective.
    1040x1540 image in .tiff format.

    Parameters
    ----------
    elem_type : EElemTest
        Enumeration specifying the element type for this test case.

    Returns
    -------
    Path
        Path to the reference image (``.tiff``).
    """
    return _data_path("dic", "plate_rigid", "rigid_cam1_frame00.tiff")


def dic_plate_rigid_cam1_def() -> Path:
    """
    Path to the rigid deformation example images from camera 1 perspective.
    1040x1540 image in .tiff format.

    Returns
    -------
    Path
        Path to the deformation images (``.tiff``).
    """
    return _data_path("dic", "plate_rigid", "rigid_cam1_frame*.tiff")

def dic_plate_rigid_cam0_def_small() -> list[Path]:
    """
    Returns rigid_cam0_frame0000.tiff to rigid_cam1_frame0010.tiff.
    """
    data_dir = files("pyvale.data").joinpath("dic", "plate_rigid")

    return [
        Path(data_dir.joinpath(f"rigid_cam0_frame{i:02d}.tiff"))
        for i in range(11)
    ]

def dic_plate_rigid_cam1_def_small() -> list[Path]:
    """
    Returns rigid_cam1_frame0000.tiff to rigid_cam1_frame0010.tiff.
    """
    data_dir = files("pyvale.data").joinpath("dic", "plate_rigid")

    return [
        Path(data_dir.joinpath(f"rigid_cam1_frame{i:02d}.tiff"))
        for i in range(11)
    ]

def dic_plate_rigid_cam0_def_10px() -> Path:
    """
    Path to the 25px rigid deformation image.
    1040x1540 image in .tiff format.

    Returns
    -------
    Path
        Path to the 25 px deformed image (``.tiff``).
    """
    return _data_path("dic", "plate_rigid", "rigid_cam0_frame11.tiff")


def dic_plate_rigid_cam0_def_25px() -> Path:
    """
    Path to the 25px rigid deformation image.
    1040x1540 image in .tiff format.

    Returns
    -------
    Path
        Path to the 25 px deformed image (``.tiff``).
    """
    return _data_path("dic", "plate_rigid", "rigid_cam0_frame12.tiff")


def dic_plate_rigid_cam0_def_50px() -> Path:
    """
    Path to the 50px rigid deformation image.
    1040x1540 image in .tiff format.

    Returns
    -------
    Path
        Path to the 50px deformed image (``.tiff``).
    """
    return _data_path("dic", "plate_rigid", "rigid_cam0_frame13.tiff")


def dic_chal_2d_ref() -> Path:
    """
    Path to the reference images for the 2D-DIC Challenge 2.0.
    Images are openly available at:

    https://idics.org/challenge/

    Returns
    -------
    Path
        Path to the reference image (``.tiff``).
    """
    return _data_path(
        "dic", "challenge", "DIC_Challenge_Star_Noise_Ref.tiff",
    )


def dic_chal_2d_def() -> Path:
    """
    Path to the deformed images for the 2D-DIC Challenge 2.0.
    Images are openly available at:

    https://idics.org/challenge/

    Returns
    -------
    Path
        Path to the deformed image (``.tiff``).
    """
    return _data_path(
        "dic", "challenge", "DIC_Challenge_Star_Noise_Def.tiff",
    )


def dic_chal_3d_cam0() -> Path:
    """
    Path to the reference images for cam0 from the Stereo DIC challenge 1.0.

    Figures reproduced from:

    Ahmad, W., Helm, J., Bossuyt, S., et al.
    "Stereo-DIC Challenge 1.0 – Rigid Body Motion of a Complex Shape",
    Experimental Mechanics, 2024.

    Licensed under CC BY 4.0:
    https://creativecommons.org/licenses/by/4.0/

    Returns
    -------
    Path
        Path to the reference image (``.tiff``).
    """
    return _data_path(
        "dic", "stereo_challenge", "Step01_00,00-sys1-0000_0.tif",
    )


def dic_chal_3d_cam1() -> Path:
    """
    Path to the reference images for cam1 from the Stereo DIC challenge 1.0.

    Figures reproduced from:

    Ahmad, W., Helm, J., Bossuyt, S., et al.
    "Stereo-DIC Challenge 1.0 – Rigid Body Motion of a Complex Shape",
    Experimental Mechanics, 2024.

    Licensed under CC BY 4.0:
    https://creativecommons.org/licenses/by/4.0/

    Returns
    -------
    Path
        Path to the deformed image (``.tiff``).
    """
    return _data_path(
        "dic", "stereo_challenge", "Step01_00,00-sys1-0000_1.tif",
    )

def cal_target() -> Path:
    """
    Path to example calibration target.

    Returns
    -------
    Path
        Path to the image (``.tiff``).
    """
    return _data_path("calibration", "cal_target.tiff")


def pxint2d_single_element_path(case_name: str) -> Path:
    """Return a packaged PixInt2D single-element fixture directory.

    Parameters
    ----------
    case_name : str
        Fixture directory name, for example ``"plate42_cam32_quad9_affine"``.

    Returns
    -------
    pathlib.Path
        Directory containing coordinates, connectivity, and displacement CSVs.

    Raises
    ------
    DataSetError
        If the requested fixture is not packaged with pyvale.
    """
    path = _data_path("render", "pxint2d", "single_elem", case_name)
    if not path.is_dir():
        raise DataSetError(f"Unknown PixInt2D fixture: {case_name}.")
    return path


def riley_speckle_texture_path() -> Path:
    """Return the texture packaged for the Riley parity examples."""
    return _data_path("render", "riley", "textures", "speckle.bmp")


def riley_cal_target_texture_path() -> Path:
    """Return the Riley stereo-calibration target texture."""
    return _data_path(
        "render",
        "riley",
        "textures",
        "cal_target-simple.tiff",
    )


def riley_sphere200_case_path() -> Path:
    """Return the Riley Tri6 sphere demonstration data directory."""
    return _data_path("render", "riley", "min", "tri6_sphere200")


def riley_platehole_csv_case_path() -> Path:
    """Return the Riley plate-with-hole CSV demonstration data directory."""
    return _data_path("render", "riley", "fe", "platehole3d_2mr_63f")


def riley_platehole_exodus_path() -> Path:
    """Return the Riley plate-with-hole Exodus demonstration file."""
    return _data_path(
        "render",
        "riley",
        "fe",
        "platehole3d_2mr_63f.e",
    )


def riley_stereocal_case_path() -> Path:
    """Return the Riley stereo-calibration mesh data directory."""
    return _data_path("render", "riley", "calplate", "tri3_calplate3d")


def riley_rabbit_case_path(
    rabbit_name: str,
    topology: EElementType,
) -> Path:
    """Return one Riley rabbit mesh case packaged with pyvale.

    Parameters
    ----------
    rabbit_name : str
        One of ``"riley"`` or ``"feebs"``.
    topology : EElementType
        One of the supported Riley surface topologies.
    """
    path = _data_path(
        "render",
        "riley",
        "rabbits",
        f"{rabbit_name}_{topology.value}",
    )
    if not path.is_dir():
        raise DataSetError(
            f"Unknown Riley rabbit case: {rabbit_name}_{topology.value}."
        )
    return path


def riley_rabbit_meshes() -> list[riley.Mesh]:
    """Load the packaged Riley TRI3 rabbit meshes with the speckle texture.

    Returns
    -------
    list[riley.Mesh]
        The ``"riley"`` and ``"feebs"`` TRI3 surface meshes normalised to
        the shared mesh convention and textured with the packaged speckle
        image.
    """
    from pyvale.dataio import SimData, enforce_mesh_convention

    texture = riley.load_texture_u8(str(riley_speckle_texture_path()))
    meshes: list[riley.Mesh] = []
    for rabbit_name in ("riley", "feebs"):
        data_path = riley_rabbit_case_path(rabbit_name, EElementType.TRI3)
        mesh_data = enforce_mesh_convention(SimData(
            coords=np.loadtxt(data_path / "coords.csv", delimiter=","),
            connect={
                "connect1": np.loadtxt(
                    data_path / "connectivity.csv",
                    delimiter=",",
                    dtype=np.uintp,
                ),
            },
        ))
        assert mesh_data.coords is not None and mesh_data.connect is not None
        uvs = np.loadtxt(data_path / "uvs.csv", delimiter=",")
        meshes.append(riley.Mesh(
            riley.MeshType.tri3,
            mesh_data.coords,
            mesh_data.connect["connect1"],
            shader_type=riley.ShaderType.tex, uvs=uvs, texture=texture,
        ))
    return meshes



#TODO
def valid_data_dir() -> Path:
    return _data_path("valid")

def valid_data_csvs() -> list[Path]:
    data_dir = files("pyvale.data").joinpath("valid")
    return [Path(str(ff)) for ff in data_dir.glob("*.csv")]
