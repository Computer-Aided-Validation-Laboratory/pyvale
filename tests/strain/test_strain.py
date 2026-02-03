

import numpy as np
import pytest
import scipy.linalg
import os
import glob

#pyvale stuff
import pyvale.dic as dic
import pyvale.strain as strain


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

    input_data = dic.Results(X, Y, Ux, Uy)

    # Compute strain
    strain_results = strain.calculate_2d(
        data=input_data,
        window_size=5,
        window_element=4,
        strain_formulation=strain_formulation,
        output_prefix=f"strain_{strain_formulation}_{deformation_type}_"
    )

    # Analytic reference strain
    expected = reference_strain(F, strain_formulation)

    strainresults = strain.import_2d(f"./strain_{strain_formulation}_{deformation_type}_*.csv")

    # Map of deformation gradient and strain components
    checks = {
        "def_xx": F[0,0],
        "def_xy": F[0,1],
        "def_yx": F[1,0],
        "def_yy": F[1,1],
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

