
import os
import numpy as np
import pytest
import scipy.linalg
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

#pyvale stuff
import pyvale.dic as dic
import pyvale.calib as calib
import pyvale.strain as strain
import pyvale.data as dataset
from pyvale.dic.dicresults import Results, StereoResults



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


def run_partial_window_test(
    tmp_path: Path,
    mask: np.ndarray,
    partial_window: float,
    q: int = 9,
    stereo: bool = False,
    quadratic: bool = False,
    threads: int = 1,
    missing_centre: bool = False,
):
    x, y = np.meshgrid(np.arange(11), np.arange(11))
    shape = mask.shape

    u = np.broadcast_to(
        0.02 * x + (0.001 * x * x if quadratic else 0), shape
    ).copy()
    v = np.broadcast_to(0.03 * y, shape).copy()
    u[~mask] = np.nan
    v[~mask] = np.nan

    data = Results(ss_x=x, ss_y=y, u_px=u, v_px=v)

    if stereo:
        world_x = np.broadcast_to(x, shape).astype(float).copy()
        if missing_centre:
            world_x[:, 5, 5] = np.nan

        data.stereo = StereoResults(
            disparity_u_px=u,
            disparity_v_px=v,
            x_mm=world_x,
            y_mm=np.broadcast_to(y, shape).astype(float).copy(),
            z_mm=np.zeros(shape),
            u_mm=u,
            v_mm=v,
            w_mm=np.zeros(shape),
        )

    calculate = strain.calculate_3d if stereo else strain.calculate_2d
    calculate(
        data,
        window_size=11,
        window_element=q,
        partial_window=partial_window,
        strain_formulation="GREEN",
        output_basepath=tmp_path,
        output_binary=True,
        num_threads=threads,
        print_level=0,
    )

    dtype = np.dtype([("x", "i4"), ("y", "i4"), ("values", "f8", (13,))])
    return np.stack(
        [
            np.fromfile(path, dtype=dtype)["values"].reshape(11, 11, 13)
            for path in sorted(tmp_path.glob("*.dic3d"))
        ]
    )


@pytest.mark.parametrize("stereo", [False, True])
@pytest.mark.parametrize("q", [4, 9])
def test_partial_window_threshold(tmp_path: Path, stereo: bool, q: int):
    mask = np.ones((1, 11, 11), dtype=bool)
    mask[:, :, 6:] = False

    result = run_partial_window_test(tmp_path, mask, 0.5, q=q, stereo=stereo)
    expected = [1.02, 0, 0, 1.03]
    if stereo:
        expected = [
            1.0 / (1.0 - 0.02),
            0,
            0,
            1.0 / (1.0 - 0.03),
        ]
    np.testing.assert_allclose(
        result[0, 5, 5, 3:7], expected, atol=1e-12
    )

    result = run_partial_window_test(tmp_path, mask, 0.55, q=q, stereo=stereo)
    assert np.isnan(result[0, 5, 5, 3:]).all()


def test_partial_window_original_centre_and_boundary(tmp_path: Path):
    mask = np.ones((1, 11, 11), dtype=bool)
    mask[:, :, 6:] = False

    result = run_partial_window_test(tmp_path, mask, 0.0, quadratic=True)
    assert result[0, 5, 5, 3] == pytest.approx(1.03)
    assert result[0, 0, 0, 3] == pytest.approx(1.02)

    result = run_partial_window_test(tmp_path, np.ones_like(mask), 1.0)
    assert np.isnan(result[0, 0, 0, 3:]).all()
    assert np.isfinite(result[0, 5, 5, 3:]).all()


@pytest.mark.parametrize("q", [4, 9])
def test_partial_window_rank_deficient_and_empty_frames(tmp_path: Path, q: int):
    mask = np.ones((3, 11, 11), dtype=bool)
    mask[1:] = False
    mask[1, 5, :] = True  # Many points, but all on one line.

    result = run_partial_window_test(tmp_path, mask, 0.0, q=q)
    assert np.isfinite(result[0, 5, 5, 3:]).all()
    assert np.isnan(result[1:, :, :, 3:]).all()


def test_partial_window_3d_missing_centre(tmp_path: Path):
    result = run_partial_window_test(
        tmp_path,
        np.ones((1, 11, 11), dtype=bool),
        0.0,
        stereo=True,
        missing_centre=True,
    )
    assert np.isnan(result[0, 5, 5, 3:]).all()


def test_partial_window_thread_equivalence(tmp_path: Path):
    mask = np.ones((1, 11, 11), dtype=bool)
    mask[:, 2:4, 3:5] = False

    single = run_partial_window_test(tmp_path, mask, 0.25, threads=1)
    multi = run_partial_window_test(tmp_path, mask, 0.25, threads=4)
    np.testing.assert_allclose(multi, single, equal_nan=True)


@pytest.mark.parametrize("value", [-0.1, 1.1, np.nan, np.inf])
@pytest.mark.parametrize("calculate", [strain.calculate_2d, strain.calculate_3d])
def test_invalid_partial_window(value, calculate):
    with pytest.raises(ValueError, match="partial_window"):
        calculate(None, partial_window=value)


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

def test_strain_deformations(
    strain_formulation,
    deformation_type,
    F,
    tmp_path: Path,
):
    # Generate displacement field
    X, Y, Ux, Uy = generate_affine_displacement_grid(F)

    input_data = dic.Results(ss_x=X, ss_y=Y, u_px=Ux, v_px=Uy)

    # Compute strain
    strain_results = strain.calculate_2d(
        data=input_data,
        window_size=5,
        window_element=9,
        strain_formulation=strain_formulation,
        output_prefix=f"strain_{strain_formulation}_{deformation_type}_",
        output_basepath=tmp_path,
        print_level=2,
    )

    # Analytic reference strain
    expected = reference_strain(F, strain_formulation)

    print(F.shape)
    print(expected.shape)

    strainresults = strain.import_2d(
        tmp_path / f"strain_{strain_formulation}_{deformation_type}_*.csv",
    )

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

def run_strain_test(window_element: int, output_path: Path):
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
        output_basepath=output_path,
        output_delimiter=",",
        image_filter_kernel=1,
    )

    dic.calculate_2d(
        reference=ref0,
        deformed=def0,
        output_prefix="dic_2d_",
        **common,
    )

    strain.calculate_2d(
        data=output_path / "dic_2d_*csv",
        window_size=5,
        window_element=window_element,
        output_basepath=output_path,
        output_prefix="strain_2d_",
        strain_formulation="ALMANSI",
        partial_window=1,
    )

    dic.calculate_3d(
        reference=[ref0, ref1],
        deformed=[def0, def1],
        calibration=calibration,
        output_prefix="dic_3d_",
        **common,
    )

    strain.calculate_3d(
        data=output_path / "dic_3d_*csv",
        window_size=5,
        window_element=window_element,
        output_basepath=output_path,
        output_prefix="strain_3d_",
        strain_formulation="ALMANSI",
        partial_window=1,
    )

    return (
        strain.import_2d(output_path / "strain_2d_*"),
        strain.import_3d(output_path / "strain_3d_*"),
    )


@pytest.mark.parametrize("window_element", [4, 9])
def test_strain_3d(window_element: int, tmp_path: Path):
    _, strain_3d = run_strain_test(window_element, tmp_path)

    gold_path = (
        Path(__file__).parent
        / "gold"
        / f"strain_3d_q{window_element}.npz"
    )
    assert gold_path.is_file(), f"Missing gold file: {gold_path}"

    gold = np.load(gold_path)
    for field in gold.files:
        np.testing.assert_allclose(
            getattr(strain_3d, field)[1],
            gold[field],
            rtol=1e-5,
            atol=1e-5,
            equal_nan=True,
            err_msg=f"Gold check failed for {field}, Q{window_element}",
        )
