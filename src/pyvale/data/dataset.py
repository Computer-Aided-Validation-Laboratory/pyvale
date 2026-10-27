#===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
#===============================================================================

"""
Accessors for PyVale's demonstration and regression data. Files are downloaded
on demand and cached locally, unless ``PYVALE_DATA`` points to a local data
directory. The collection includes MOOSE simulation outputs, MOOSE and Gmsh
inputs, and images used by the digital image correlation examples and tests.
"""

from enum import Enum
from importlib.resources import files
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pooch
import riley

from pyvale.render.mesh import EElemType

SIM_CASE_COUNT = 26
"""Number of simulation test case input files available through PyVale.
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


_LOCAL_DATA_ENV = "PYVALE_DATA"
"""Environment variable pointing to a local PyVale data directory."""


_DATA = pooch.create(
    path=pooch.os_cache("pyvale"),
    base_url=(
        "https://raw.githubusercontent.com/"
        "Computer-Aided-Validation-Laboratory/pyvale-data/dev/Data/"
    ),
    version=None,
    registry={},
)
_DATA.load_registry(files("pyvale.data").joinpath("registry.txt"))


def _local_data_path() -> Path | None:
    """Return the user-specified local data directory, if configured."""
    data_dir = os.environ.get(_LOCAL_DATA_ENV)
    if data_dir is None:
        return None
    path = Path(data_dir).expanduser()
    if not path.is_dir():
        raise DataSetError(
            f"{_LOCAL_DATA_ENV} does not point to a directory: {path}"
        )
    return path


def _data_path(*parts: str) -> Path:
    """Return a local or downloaded path below the PyVale data directory."""
    relative_path = Path(*parts).as_posix()
    local_data = _local_data_path()
    if local_data is not None:
        return local_data.joinpath(*parts)

    if any(char in relative_path for char in "*?["):
        parent, pattern = relative_path.rsplit("/", maxsplit=1)
        _data_directory(*Path(parent).parts)
        return _DATA.abspath.joinpath(parent, pattern)

    return Path(_DATA.fetch(relative_path))


def _data_directory(*parts: str) -> Path:
    """Return a local data directory, fetching all registered files below it."""
    local_data = _local_data_path()
    if local_data is not None:
        return local_data.joinpath(*parts)

    prefix = Path(*parts).as_posix().rstrip("/") + "/"
    for filename in _DATA.registry_files:
        if filename.startswith(prefix):
            _DATA.fetch(filename)
    return _DATA.abspath.joinpath(*parts)


def sim_case_input_file_path(case_num: int) -> Path:
    """Gets the path to MOOSE input file (``*.i``) for a particular simulation
    case.

    Parameters
    ----------
    case_num : int
        Integer defining the case number to be retrieved. Must be greater
        than 0 and less than the number of simulation cases.

    Returns
    -------
    Path
        Path object to the MOOSE ``*.i`` file for the selected simulation case.

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
    """Gets the path to Gmsh input file (``*.geo``) for a particular simulation
    case. Note that not all simulation cases use Gmsh for geometry and mesh
    generation. If the specified simulation case does not have an associated
    Gmsh ``*.geo`` file, 'None' is returned.

    Parameters
    ----------
    case_num : int
        Integer defining the case number to be retrieved. Must be greater
        than 0 and less than the number of simulation cases.

    Returns
    -------
    Path | None
        Path object to the Gmsh ``*.geo`` file for the selected simulation case.
        Returns None if there is no ``*.geo`` for this simulation case.

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
    relative_path = Path("simulation", "simcases", case_file).as_posix()
    local_data = _local_data_path()
    if local_data is not None:
        case_path = local_data.joinpath(relative_path)
    elif relative_path not in _DATA.registry:
        return None
    else:
        case_path = _data_path("simulation", "simcases", case_file)

    if case_path.is_file():
        return case_path

    return None


def dic_pattern_5mpx_path() -> Path:
    """Return Riley's 5 mega-pixel mono speckle pattern image.

    The image has shape ``(2056, 2464)`` with 8-bit resolution. It is the
    shared packaged Riley texture used by Pyvale's rendering and DIC examples.

    Returns
    -------
    Path
        Path to Riley's packaged speckle image.
    """
    return riley.data.speckle_texture_path()


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
    in this class. Note that this case uses a Gmsh ``*.geo`` file for geometry
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
    Geometry and displacements are expressed in millimetres, with material
    properties in N/mm² (MPa).

    The simulation parameters can be found in the corresponding MOOSE input
    file: case17.i which can be retrieved using `sim_case_input_file_path`
    in this class. Note that this case uses a Gmsh ``*.geo`` file for geometry
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



def dic_rigid_exodus_path(max_displacement_px: int) -> Path:
    """Path to a Riley rigid-plate DIC benchmark Exodus file.

    Supported maximum displacements are 1, 25, and 50 pixels.
    """
    if max_displacement_px not in {1, 25, 50}:
        raise DataSetError("Rigid DIC displacement must be 1, 25, or 50 pixels")
    return _data_path(
        "simulation",
        "exodus",
        f"mechplate2d_rigid_{max_displacement_px}px_out.e",
    )


def dic_platehole_exodus_path() -> Path:
    """Path to the Riley plate-with-hole DIC benchmark Exodus file."""
    return _data_path(
        "simulation", "exodus", "platehole2d_pstress_out.e",
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
    data_dir = _data_directory("dic", "plate_rigid")

    return [
        Path(data_dir.joinpath(f"rigid_cam0_frame{i:02d}.tiff"))
        for i in [0,4,8,10]
    ]

def dic_plate_rigid_cam1_def_small() -> list[Path]:
    """
    Returns rigid_cam1_frame0000.tiff to rigid_cam1_frame0010.tiff.
    """
    data_dir = _data_directory("dic", "plate_rigid")

    return [
        Path(data_dir.joinpath(f"rigid_cam1_frame{i:02d}.tiff"))
        for i in [0,4,8,10]
    ]

def dic_plate_rigid_cam0_def_25px() -> Path:
    """
    Path to the 25px rigid deformation image.
    1040x1540 image in .tiff format.

    Returns
    -------
    Path
        Path to the 25 px deformed image (``.tiff``).
    """
    return _data_path("dic", "plate_rigid", "rigid_cam0_frame25.tiff")


def dic_plate_rigid_cam0_def_50px() -> Path:
    """
    Path to the 50px rigid deformation image.
    1040x1540 image in .tiff format.

    Returns
    -------
    Path
        Path to the 50px deformed image (``.tiff``).
    """
    return _data_path("dic", "plate_rigid", "rigid_cam0_frame50.tiff")


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
    """Return Riley's packaged calibration-target image.

    Returns
    -------
    Path
        Path to Riley's packaged calibration-target image.
    """
    return riley.data.cal_target_texture_path()


def riley_rabbit_meshes() -> list[riley.Mesh]:
    """Load Riley's packaged TRI3 rabbit meshes with its speckle texture.

    Returns
    -------
    list[riley.Mesh]
        The ``"riley"`` and ``"feebs"`` TRI3 surface meshes normalised to
        the shared mesh convention and textured with Riley's speckle image.
    """
    texture = riley.load_texture_mono_u8(riley.data.speckle_texture_path())
    meshes: list[riley.Mesh] = []
    for rabbit_name in ("riley", "feebs"):
        data_path = riley.data.rabbit_case_path(
            rabbit_name,
            EElemType.TRI3.value,
        )
        coords = riley.load_csv(data_path / "coords.csv")
        connect = riley.load_csv(
            data_path / "connectivity.csv", dtype=np.int64
        )
        uvs = riley.load_csv(data_path / "uvs.csv")
        convention = riley.ConnectConvention(
            riley.EElemType.TRI3,
            riley.EConnectAxis.ROW,
            0,
            riley.ENodeOrder.RILEY,
        )
        meshes.append(riley.create_mesh(
            convention,
            riley.MeshType.tri3,
            coords,
            connect,
            riley.TextureShader(uvs=uvs, texture=texture),
        ))
    return meshes

def dic_ex08_stereo_calibration() -> Path:
    """
    Path to the stereo calibration parameters for the DIC example.

    Returns
    -------
    Path
        Path to the stereo calibration file (``.txt``).
    """
    return _data_path("dic", "dic_ex08_stereo_calibration.txt")


def dic_ex02_roi() -> Path:
    """
    Path to the region of interest definition for the DIC example.

    Returns
    -------
    Path
        Path to the ROI definition file (``.yaml``).
    """
    return _data_path("dic", "dic_ex02_roi.yaml")


def dic_ex10_dic_chal_calibration() -> Path:
    """
    Path to the calibration parameters for the DIC Challenge example.

    Returns
    -------
    Path
        Path to the DIC Challenge calibration file (``.txt``).
    """
    return _data_path("dic", "dic_ex10_dic_chal_calibration.txt")


def dic_ex10_dic_chal_roi() -> Path:
    """
    Path to the region of interest definition for the DIC Challenge example.

    Returns
    -------
    Path
        Path to the DIC Challenge ROI definition file (``.yaml``).
    """
    return _data_path("dic", "dic_ex10_dic_chal_roi.yaml")


def dic_calibration() -> Path:
    """Return the extracted DIC calibration image directory.

    The calibration images are distributed as a single Pooch-managed archive
    because the uncompressed image set is large. The returned directory has
    the same layout as the original data directory.
    """
    local_data = _local_data_path()
    if local_data is not None:
        data_dir = local_data / "dic" / "calibration"
        if data_dir.is_dir():
            return data_dir

        archive = local_data / "dic" / "calibration.zip"
        if archive.is_file():
            data_dir = _DATA.abspath / "local" / "dic" / "calibration"
            if not data_dir.is_dir():
                with ZipFile(archive) as zip_file:
                    data_dir.mkdir(parents=True, exist_ok=True)
                    zip_file.extractall(data_dir)
            return data_dir

        raise DataSetError(
            "Local PyVale data is missing dic/calibration or "
            "dic/calibration.zip"
        )

    _DATA.fetch(
        "dic/calibration.zip",
        processor=pooch.Unzip(extract_dir="calibration"),
    )
    return _DATA.abspath / "dic" / "calibration"



#TODO
def valid_data_dir() -> Path:
    return _data_directory("valid")

def valid_data_csvs() -> list[Path]:
    data_dir = _data_directory("valid")
    return sorted(data_dir.glob("*.csv"))
