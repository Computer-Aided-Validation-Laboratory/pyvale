from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pyvale.vfm.mechanical_properties import (
    EConstituitiveLaw,
    HomogeneousParameter,
    MechanicalProperties,
    ParameterBounds,
    EParameterName,
)
from pyvale.vfm.radial_return import radial_return


def _make_mechanical_properties(
    elastic_modulus: float,
    poissons_ratio: float,
    yield_strength: float,
    hardening_modulus: float,
) -> MechanicalProperties:
    return MechanicalProperties(
        ConstituitiveLaw.LinearHardening,
        {
            ParameterName.ElasticModulus: HomogeneousParameter(
                IdentificationType.Known,
                ParameterBounds(1.0, 1.0e9),
                ScalarValue(elastic_modulus),
            ),
            ParameterName.PoissonsRatio: HomogeneousParameter(
                IdentificationType.Known,
                ParameterBounds(0.0, 0.49),
                ScalarValue(poissons_ratio),
            ),
            ParameterName.YieldStrength: HomogeneousParameter(
                IdentificationType.Known,
                ParameterBounds(1.0, 1.0e9),
                ScalarValue(yield_strength),
            ),
            ParameterName.HardeningModulus: HomogeneousParameter(
                IdentificationType.Known,
                ParameterBounds(0.0, 1.0e9),
                ScalarValue(hardening_modulus),
            ),
        },
    )


def _load_strain_from_test_data(data_mat_path: Path) -> np.ndarray:
    data = loadmat(
        data_mat_path,
        struct_as_record=False,
        squeeze_me=True,
        simplify_cells=True,
    )
    test_data = data["testData"]
    strain = test_data["strain"]

    num_rows = 113
    num_cols = 316
    num_timesteps = strain["c11"].shape[-1] if strain["c11"].ndim > 1 else 23

    strain_c11 = strain["c11"].reshape((num_rows, num_cols, num_timesteps), order="F")
    strain_c22 = strain["c22"].reshape((num_rows, num_cols, num_timesteps), order="F")
    strain_c12 = strain["c12"].reshape((num_rows, num_cols, num_timesteps), order="F")

    strain_c11 = np.transpose(strain_c11, (2, 0, 1))
    strain_c22 = np.transpose(strain_c22, (2, 0, 1))
    strain_c12 = np.transpose(strain_c12, (2, 0, 1))

    strain_4d = np.stack((strain_c11, strain_c22, strain_c12), axis=1)
    strain_4d = np.flip(strain_4d, axis=2)

    return strain_4d.astype(np.float64)


def _build_downsampled_case(
    full_strain: np.ndarray,
    selected_timesteps: np.ndarray,
    selected_points_y: np.ndarray,
    selected_points_x: np.ndarray,
) -> np.ndarray:
    num_timesteps = len(selected_timesteps)
    num_points = len(selected_points_y)

    downsampled = np.zeros((num_timesteps, 3, 1, num_points), dtype=np.float64)

    for i_t, t in enumerate(selected_timesteps):
        for i_p, (y, x) in enumerate(zip(selected_points_y, selected_points_x, strict=True)):
            downsampled[i_t, :, 0, i_p] = full_strain[t, :, y, x]

    return downsampled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a compact radial-return regression fixture from verified real strain data."
    )
    parser.add_argument(
        "--data-mat",
        type=Path,
        required=True,
        help="Path to testData.mat",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/vfm/fixtures/radial_return_downsampled_case.npz"),
        help="Output .npz path",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=8,
        help="Number of timesteps to sample",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=32,
        help="Number of spatial points to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260325,
        help="Random seed for spatial downsampling",
    )
    parser.add_argument(
        "--elastic-modulus",
        type=float,
        default=190000.0,
    )
    parser.add_argument(
        "--poissons-ratio",
        type=float,
        default=0.28,
    )
    parser.add_argument(
        "--yield-strength",
        type=float,
        default=320.0,
    )
    parser.add_argument(
        "--hardening-modulus",
        type=float,
        default=3000.0,
    )
    parser.add_argument(
        "--error-tolerance",
        type=float,
        default=1.0e-8,
    )
    parser.add_argument(
        "--iteration-limit",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--unloading",
        choices=["no_compensation", "constant_strain", "linear_extrapolation"],
        default="constant_strain",
    )
    args = parser.parse_args()

    full_strain = _load_strain_from_test_data(args.data_mat)
    num_timesteps_total = full_strain.shape[0]

    selected_timesteps = np.linspace(
        0,
        num_timesteps_total - 1,
        num=min(args.num_timesteps, num_timesteps_total),
        dtype=int,
    )
    selected_timesteps = np.unique(selected_timesteps)

    finite_mask = np.all(np.isfinite(full_strain[selected_timesteps]), axis=(0, 1))
    candidate_points = np.argwhere(finite_mask)

    if candidate_points.shape[0] < args.num_points:
        raise ValueError(
            f"Requested {args.num_points} points, but only {candidate_points.shape[0]} valid points "
            "are finite across all selected timesteps/components."
        )

    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(candidate_points.shape[0], size=args.num_points, replace=False)
    sampled_points = candidate_points[chosen]

    selected_points_y = sampled_points[:, 0]
    selected_points_x = sampled_points[:, 1]

    strain_fixture = _build_downsampled_case(
        full_strain,
        selected_timesteps,
        selected_points_y,
        selected_points_x,
    )

    mechanical_properties = _make_mechanical_properties(
        elastic_modulus=args.elastic_modulus,
        poissons_ratio=args.poissons_ratio,
        yield_strength=args.yield_strength,
        hardening_modulus=args.hardening_modulus,
    )

    stress_expected, equivalent_stress_expected, yield_map_expected, peeq_expected = radial_return(
        strain_fixture,
        mechanical_properties,
        error_tolerance=args.error_tolerance,
        iteration_limit=args.iteration_limit,
        unloading=args.unloading,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        strain=strain_fixture,
        elastic_modulus=np.float64(args.elastic_modulus),
        poissons_ratio=np.float64(args.poissons_ratio),
        yield_strength=np.float64(args.yield_strength),
        hardening_modulus=np.float64(args.hardening_modulus),
        error_tolerance=np.float64(args.error_tolerance),
        iteration_limit=np.int64(args.iteration_limit),
        unloading_mode=np.array(args.unloading),
        selected_timesteps=selected_timesteps.astype(np.int64),
        selected_points_y=selected_points_y.astype(np.int64),
        selected_points_x=selected_points_x.astype(np.int64),
        source_data_mat=np.array(str(args.data_mat)),
        seed=np.int64(args.seed),
        stress_expected=stress_expected,
        equivalent_stress_expected=equivalent_stress_expected,
        yield_map_expected=yield_map_expected,
        peeq_expected=peeq_expected,
    )

    print(f"Wrote fixture to {args.output}")
    print(f"Selected timesteps: {selected_timesteps.tolist()}")
    print(f"Selected points: {args.num_points}")


if __name__ == "__main__":
    main()
