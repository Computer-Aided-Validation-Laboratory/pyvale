#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

"""
Accesors for data that comes pre-packaged with pyvale for demonstrating its
functionality. This includes moose simulation outputs as exodus files, input
files for moose and gmsh for additional simulation cases, and images required
for testing the image deformation and digital image correlation modules.
"""

from enum import Enum
from pathlib import Path
from importlib.resources import files

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
    return Path(files("pyvale.simcases").joinpath(case_file))


def sim_case_gmsh_file_path(case_num: int) -> Path | None:
    """Gets the path to Gmsh input file (*.geo) for a particular simulation
    case. Note that not all simulation cases use Gmsh for geometry and mesh
    generation. If the specified simulation case does not have an associated
    Gmsh *.geo file. In this case 'None' is returned

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
    case_path = Path(files("pyvale.simcases").joinpath(case_file))

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
    return Path(files("pyvale.data")
                .joinpath("optspeckle_2464x2056px_spec5px_8bit_gblur1px.tiff"))


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
        Path to the exodus (*.e) output file for this simulation case.
    """
    return Path(files("pyvale.data").joinpath("case18_out.e"))


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
        Path to the exodus (*.e) output file for this simulation case.
    """
    return Path(files("pyvale.data").joinpath("case16_out.e"))


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
        Path to the exodus (*.e) output file for this simulation case.
    """
    return Path(files("pyvale.data").joinpath("case17_out.e"))


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
        Path to the exodus (*.e) output file for this simulation case.
    """
    return Path(files("pyvale.data").joinpath("case18_out.e"))


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
        Path to the exodus (*.e) output file for this simulation case.
    """
    return Path(files("pyvale.data").joinpath("case16_out.e"))


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
        Paths to the exodus (*.e) output files for this simulated experiment.
    """
    return [Path(files("pyvale.data").joinpath("case18_out.e")),
            Path(files("pyvale.data").joinpath("case18_d_out.e"))]

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
        Paths to the exodus (*.e) output files for this simulated experiment.
    """

    return [Path(files("pyvale.data").joinpath("case16_out.e")),
            Path(files("pyvale.data").joinpath("case16_d_out.e"))]


def render_mechanical_3d_path() -> Path:
    """Path to a MOOSE simulation output in exodus format. This case is a
    purely mechanical test case in 3D meant for testing image rendering
    algorithms for digital image correlation simulation. The simulation
    consists of a linear elastic thin plate with a hole loaded in tension.
    The simulation uses linear tetrahedral elements for rendering tests.

    Returns
    -------
    Path
        Path to the exodus (*.e) output file for this simulation case.
    """
    return Path(files("pyvale.data").joinpath("case26_out.e"))
    

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
    return Path(files("pyvale.simcases")
                .joinpath(f"case00_{elem_type.value}.i"))



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
        Path to the exodus (*.e) output file for this simulation case.
    """
    return Path(files("pyvale.data")
                .joinpath(f"case00_{elem_type.value}_out.e"))


def dic_plate_with_hole_ref() -> Path:
    """Path to the reference image for the plate with hole example.
    1040x1540 image in .tiff format.

    Parameters
    ----------
    elem_type : EElemTest
        Enumeration specifying the element type for this test case.

    Returns
    -------
    Path
        Path to the reference image (*.tiff).
    """
    return Path(files("pyvale.data")
                .joinpath("plate_hole_ref0000.tiff"))


def dic_plate_with_hole_def() -> Path:
    """Path to the deformed images for the plate with hole example.
    1040x1540 image in .tiff format.

    Parameters
    ----------
    elem_type : EElemTest
        Enumeration specifying the element type for this test case.

    Returns
    -------
    Path
        Path to the reference image (*.tiff).
    """
    return Path(files("pyvale.data")
                .joinpath("plate_hole_def*.tiff"))


def dic_plate_rigid_ref() -> Path:
    """Path to the reference image for the rigid deformation example.
    1040x1540 image in .tiff format.

    Parameters
    ----------
    elem_type : EElemTest
        Enumeration specifying the element type for this test case.

    Returns
    -------
    Path
        Path to the reference image (*.tiff).
    """
    return Path(files("pyvale.data")
                .joinpath("plate_rigid_ref0000.tiff"))


def dic_plate_rigid_def() -> Path:
    """Path to the rigid deformation example images.
    1040x1540 image in .tiff format.

    Returns
    -------
    Path
        Path to the deformation images (*.tiff).
    """
    return Path(files("pyvale.data")
                .joinpath("plate_rigid_def0*.tiff"))


def dic_plate_rigid_def_25px() -> Path:
    """Path to the 25px rigid deformation image.
    1040x1540 image in .tiff format.

    Returns
    -------
    Path
        Path to the 25 px deformed image (*.tiff).
    """
    return Path(files("pyvale.data").joinpath("plate_rigid_def_25px.tiff"))


def dic_plate_rigid_def_50px() -> Path:
    """Path to the 50px rigid deformation image.
    1040x1540 image in .tiff format.

    Returns
    -------
    Path
        Path to the 50px deformed image (*.tiff).
    """
    return Path(files("pyvale.data").joinpath("plate_rigid_def_50px.tiff"))


def dic_challenge_ref() -> Path:
    """Path to the reference images for the 2D DIC challenge.

    Returns
    -------
    Path
        Path to the reference image (*.tiff).
    """
    return Path(files("pyvale.data")
                .joinpath("DIC_Challenge_Star_Noise_Ref.tiff"))


def dic_challenge_def() -> Path:
    """Path to the reference images for the 2D DIC challenge.

    Returns
    -------
    Path
        Path to the deformed image (*.tiff).
    """
    return Path(files("pyvale.data")
                .joinpath("DIC_Challenge_Star_Noise_Def.tiff"))

def cal_target() -> Path:
    """Path to example calibration target.

    Returns
    -------
    Path
        Path to the image (*.tiff).
    """
    return Path(files("pyvale.data")
                .joinpath("cal_target.tiff"))



