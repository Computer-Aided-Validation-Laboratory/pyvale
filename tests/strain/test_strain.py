

import numpy as np
import pytest
import scipy.linalg
import os
import glob
from pathlib import Path

#pyvale stuff
import pyvale.dic as dic
import pyvale.calib as calib
import pyvale.strain as strain
import pyvale.dataset as dataset


TEST_DATA_DIR = Path(__file__).parent / "test"


def generate_affine_displacement_grid(F, nx=100, ny=100):
    """Generate displacement field for a uniform deformation gradient F."""
    x = np.linspace(0, 990, nx)
    y = np.linspace(0, 990, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    Ux = np.zeros((1, nx, ny))
    Uy = np.zeros((1, nx, ny))

    # create displacement field
    for i in range(nx):
        for j in range(ny):
            pos = np.array([X[i,j], Y[i,j]])
            disp = (F - np.eye(2)) @ pos
            Ux[0, i, j] = disp[0]
            Uy[0, i, j] = disp[1]

    return X, Y, Ux, Uy


def reference_strain(F, formulation):
    I = np.eye(2)
    C = F.T @ F
    B = F @ F.T
    U = scipy.linalg.sqrtm(C)
    V = scipy.linalg.sqrtm(B)

    if formulation == "GREEN":
        return 0.5 * (C - I)
    if formulation == "ALMANSI":
        return 0.5 * (I - np.linalg.inv(B))
    if formulation == "BIOT_LAGRANGE":
        return U - I
    if formulation == "BIOT_EULER":
        return V - I
    if formulation == "HENCKY":
        return scipy.linalg.logm(U)

    raise ValueError(formulation)


# Strain formulations I've got implemented currently
@pytest.mark.parametrize("strain_formulation", [
    "GREEN",
    "ALMANSI",
    "BIOT_LAGRANGE",
    "BIOT_EULER",
    "HENCKY",
])

# list of deformation types to check against.
@pytest.mark.parametrize("deformation_type,F", [
    ("uniform_x_stretch", np.array([[1.02, 0], [0, 1.0]])),
    ("uniform_y_stretch", np.array([[1.0, 0], [0, 1.03]])),
    ("shear_x", np.array([[1.0, 0.05], [0.0, 1.0]])),
    ("shear_y", np.array([[1.0, 0.0], [0.05, 1.0]])),
    ("stretch_and_shear", np.array([[1.02, 0.03], [0.0, 1.01]])),
    ("rotation", np.array([[np.cos(np.pi/12), -np.sin(np.pi/12)],
                           [np.sin(np.pi/12),  np.cos(np.pi/12)]])),
])

def test_strain_deformations(strain_formulation, deformation_type, F):
    # Generate displacement field
    X, Y, Ux, Uy = generate_affine_displacement_grid(F)

    TEST_DATA_DIR.mkdir(exist_ok=True)
    np.savetxt(TEST_DATA_DIR / f"u_{strain_formulation}_{deformation_type}.txt", Ux[0])
    np.savetxt(TEST_DATA_DIR / f"v_{strain_formulation}_{deformation_type}.txt", Uy[0])

    input_data = dic.Results(ss_x=X, ss_y=Y, u_px=Ux, v_px=Uy)

    # Compute strain
    strain_results = strain.calculate_2d(
        data=input_data,
        window_size=5,
        window_element=9,
        strain_formulation=strain_formulation,
        output_prefix=f"strain_{strain_formulation}_{deformation_type}_",
        print_level=2
    )

    # Analytic reference strain
    expected = reference_strain(F, strain_formulation)

    print(F.shape)
    print(expected.shape)

    strainresults = strain.import_2d(f"./strain_{strain_formulation}_{deformation_type}_*.csv")

    # Map of deformation gradient and strain components
    checks = {
        "def_00": F[0,0],
        "def_01": F[0,1],
        "def_10": F[1,0],
        "def_11": F[1,1],
        "eps_xx": expected[0,0],
        "eps_xy": expected[0,1],
        "eps_yx": expected[1,0],
        "eps_yy": expected[1,1],
    }

    # Loop through and assert all
    for attr, val in checks.items():
        np.testing.assert_allclose(
            getattr(strainresults, attr),
            val,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"{attr} incorrect for {strain_formulation} / {deformation_type}"
        )

    for file_path in glob.glob(f"./strain_{strain_formulation}_{deformation_type}_*.csv"):
        os.remove(file_path)



def run_strain_test(window_element):
    ref0 = dataset.dic_plate_with_hole_cam0_ref()
    ref1 = dataset.dic_plate_with_hole_cam1_ref()
    def0 = dataset.dic_plate_with_hole_cam0_def()
    def1 = dataset.dic_plate_with_hole_cam1_def()

    roi = dic.RegionOfInterest(ref0)
    roi.read_yaml(Path(__file__).parent / "roi.yaml")

    calibration = calib.loadtxt(Path(__file__).parent / "calib.txt")

    common = dict(
        roi_mask=roi.mask,
        seed=roi.seed,
        subset_size=21,
        subset_step=10,
        max_displacement=10,
        output_basepath="./",
        output_delimiter=",",
    )

    dic.calculate_2d(
        reference=ref0,
        deformed=def0,
        output_prefix="dic_2d_",
        **common,
    )

    strain.calculate_2d(
        data="dic_2d_*csv",
        window_size=5,
        window_element=window_element,
        output_basepath="./",
        output_prefix="strain_2d_",
        strain_formulation="ALMANSI",
    )

    dic.calculate_3d(
        reference=[ref0, ref1],
        deformed=[def0, def1],
        calibration=calibration,
        output_prefix="dic_3d_",
        **common,
    )

    strain.calculate_3d(
        data="dic_3d_*csv",
        window_size=5,
        window_element=window_element,
        output_basepath="./",
        output_prefix="strain_3d_",
        strain_formulation="ALMANSI",
    )

    return (
        strain.import_2d("strain_2d_*"),
        strain.import_3d("strain_3d_*"),
    )


@pytest.mark.parametrize("window_element", [4, 9])
def test_strain_3d(window_element):
    strain_2d, strain_3d = run_strain_test(window_element)

    for field, atol in [
        ("eps_xx", 4e-5),
        ("eps_xy", 8e-5),
        ("eps_yy", 2e-4),
    ]:
        np.testing.assert_allclose(
            getattr(strain_2d, field)[1],
            getattr(strain_3d, field)[1],
            rtol=0.0,
            atol=atol,
            err_msg=f"mismatch for {field}",
        )

        
        output_files = sorted(glob.glob("*dic*.csv"))
        for files in output_files:
            os.remove(files)
