import numpy as np
import numpy.testing as npt

from pyvale.vfm.mechanical_properties import (
    ConstituitiveLaw,
    HomogeneousParameter,
    IdentificationType,
    MechanicalProperties,
    ParameterBounds,
    ParameterName,
    ScalarValue,
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


def _plane_stress_stiffness(elastic_modulus: float, poissons_ratio: float) -> np.ndarray:
    return elastic_modulus / (1.0 - poissons_ratio**2) * np.array(
        [
            [1.0, poissons_ratio, 0.0],
            [poissons_ratio, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - poissons_ratio)],
        ],
        dtype=np.float64,
    )


def _compute_elastic_expected_stress(
    strain: np.ndarray,
    elastic_modulus: float,
    poissons_ratio: float,
) -> np.ndarray:
    d = _plane_stress_stiffness(elastic_modulus, poissons_ratio)

    incremental_strain = np.zeros_like(strain)
    incremental_strain[0] = strain[0]
    incremental_strain[1:] = np.diff(strain, axis=0)
    incremental_strain[:, 2] *= 2.0

    num_timesteps, _, size_y, size_x = strain.shape
    stress = np.zeros_like(strain)

    for t in range(num_timesteps):
        prev = stress[t - 1] if t > 0 else np.zeros((3, size_y, size_x), dtype=np.float64)
        delta_flat = np.moveaxis(incremental_strain[t], 0, -1).reshape(-1, 3)
        prev_flat = np.moveaxis(prev, 0, -1).reshape(-1, 3)
        next_flat = prev_flat + delta_flat @ d
        stress[t] = np.moveaxis(next_flat.reshape(size_y, size_x, 3), -1, 0)

    return stress


def _von_mises(stress: np.ndarray) -> np.ndarray:
    sig_xx = stress[:, 0]
    sig_yy = stress[:, 1]
    sig_xy = stress[:, 2]
    return np.sqrt(sig_xx**2 + sig_yy**2 - sig_xx * sig_yy + 3.0 * sig_xy**2)


def test_radial_return_zero_strain_returns_zero_everything() -> None:
    mechanical_properties = _make_mechanical_properties(
        elastic_modulus=210000.0,
        poissons_ratio=0.3,
        yield_strength=320.0,
        hardening_modulus=3000.0,
    )

    strain = np.zeros((5, 3, 2, 4), dtype=np.float64)

    stress, equivalent_stress, yield_map, peeq = radial_return(strain, mechanical_properties)

    npt.assert_allclose(stress, 0.0, rtol=0.0, atol=0.0)
    npt.assert_allclose(equivalent_stress, 0.0, rtol=0.0, atol=0.0)
    npt.assert_array_equal(yield_map, np.zeros((5, 2, 4), dtype=bool))
    npt.assert_allclose(peeq, 0.0, rtol=0.0, atol=0.0)


def test_radial_return_elastic_single_point_matches_plane_stress_and_shape() -> None:
    elastic_modulus = 210000.0
    poissons_ratio = 0.3
    mechanical_properties = _make_mechanical_properties(
        elastic_modulus=elastic_modulus,
        poissons_ratio=poissons_ratio,
        yield_strength=1.0e9,
        hardening_modulus=0.0,
    )

    strain = np.zeros((2, 3, 1, 1), dtype=np.float64)
    strain[0, :, 0, 0] = [1.0e-4, 2.0e-4, 5.0e-5]
    strain[1, :, 0, 0] = [2.0e-4, 1.0e-4, 1.0e-4]

    stress, equivalent_stress, yield_map, peeq = radial_return(strain, mechanical_properties)

    assert stress.shape == (2, 3, 1, 1)
    assert equivalent_stress.shape == (2, 1, 1)
    assert yield_map.shape == (2, 1, 1)
    assert peeq.shape == (2, 1, 1)

    expected_stress = _compute_elastic_expected_stress(strain, elastic_modulus, poissons_ratio)
    expected_equivalent_stress = _von_mises(expected_stress)

    npt.assert_allclose(stress, expected_stress, rtol=1.0e-10, atol=1.0e-10)
    npt.assert_allclose(
        equivalent_stress, expected_equivalent_stress, rtol=1.0e-10, atol=1.0e-10
    )
    npt.assert_array_equal(yield_map, np.zeros((2, 1, 1), dtype=bool))
    npt.assert_allclose(peeq, 0.0, rtol=0.0, atol=0.0)


def test_radial_return_elastic_multiple_points_match_pointwise_hookes_law() -> None:
    elastic_modulus = 190000.0
    poissons_ratio = 0.28
    mechanical_properties = _make_mechanical_properties(
        elastic_modulus=elastic_modulus,
        poissons_ratio=poissons_ratio,
        yield_strength=1.0e9,
        hardening_modulus=0.0,
    )

    strain = np.zeros((3, 3, 2, 3), dtype=np.float64)

    strain[0, 0] = np.array([[1.0e-4, 2.0e-4, -1.0e-4], [5.0e-5, -3.0e-5, 8.0e-5]])
    strain[0, 1] = np.array([[2.0e-5, -1.5e-4, 3.0e-5], [9.0e-5, 1.1e-4, -2.0e-5]])
    strain[0, 2] = np.array([[4.0e-5, -2.0e-5, 1.0e-5], [3.0e-5, -1.0e-5, 2.0e-5]])

    strain[1, 0] = strain[0, 0] + np.array([[3.0e-5, -1.0e-5, 4.0e-5], [2.0e-5, 1.0e-5, -2.0e-5]])
    strain[1, 1] = strain[0, 1] + np.array([[2.0e-5, 1.0e-5, -1.0e-5], [-2.0e-5, 3.0e-5, 2.0e-5]])
    strain[1, 2] = strain[0, 2] + np.array([[1.0e-5, -1.0e-5, 2.0e-5], [0.0e0, 1.0e-5, -1.0e-5]])

    strain[2] = 1.5 * strain[1]

    stress, equivalent_stress, yield_map, peeq = radial_return(strain, mechanical_properties)
    expected_stress = _compute_elastic_expected_stress(strain, elastic_modulus, poissons_ratio)
    expected_equivalent_stress = _von_mises(expected_stress)

    npt.assert_allclose(stress, expected_stress, rtol=1.0e-10, atol=1.0e-10)
    npt.assert_allclose(
        equivalent_stress, expected_equivalent_stress, rtol=1.0e-10, atol=1.0e-10
    )
    npt.assert_array_equal(yield_map, np.zeros((3, 2, 3), dtype=bool))
    npt.assert_allclose(peeq, 0.0, rtol=0.0, atol=0.0)


def test_radial_return_plastic_single_point_monotonic_loading_invariants() -> None:
    yield_strength = 250.0
    hardening_modulus = 1500.0
    mechanical_properties = _make_mechanical_properties(
        elastic_modulus=210000.0,
        poissons_ratio=0.3,
        yield_strength=yield_strength,
        hardening_modulus=hardening_modulus,
    )

    strain = np.zeros((6, 3, 1, 1), dtype=np.float64)
    strain[:, 0, 0, 0] = np.array([0.0, 8.0e-4, 1.6e-3, 2.4e-3, 3.2e-3, 4.0e-3])

    stress, equivalent_stress, yield_map, peeq = radial_return(
        strain,
        mechanical_properties,
        unloading="no_compensation",
    )

    peeq_1d = peeq[:, 0, 0]
    eq_1d = equivalent_stress[:, 0, 0]
    yield_1d = yield_map[:, 0, 0]

    assert np.all(np.diff(peeq_1d) >= -1.0e-12)
    assert np.any(yield_1d)

    plastic_steps = np.where(yield_1d)[0]
    current_yield_stress = yield_strength + hardening_modulus * peeq_1d
    npt.assert_allclose(
        eq_1d[plastic_steps],
        current_yield_stress[plastic_steps],
        rtol=1.0e-5,
        atol=1.0e-5,
    )

    assert np.isfinite(stress).all()


def test_radial_return_unloading_modes_behave_as_expected() -> None:
    mechanical_properties = _make_mechanical_properties(
        elastic_modulus=210000.0,
        poissons_ratio=0.3,
        yield_strength=250.0,
        hardening_modulus=1500.0,
    )

    strain = np.zeros((4, 3, 1, 1), dtype=np.float64)
    strain[:, 0, 0, 0] = np.array([0.0, 1.8e-3, 3.2e-3, 1.2e-3])

    stress_none, _, yield_map, _ = radial_return(
        strain, mechanical_properties, unloading="no_compensation"
    )
    stress_const, _, _, _ = radial_return(
        strain, mechanical_properties, unloading="constant_strain"
    )
    stress_lin, _, _, _ = radial_return(
        strain, mechanical_properties, unloading="linear_extrapolation"
    )

    unload_steps = np.where(yield_map[:-1, 0, 0] & (~yield_map[1:, 0, 0]))[0] + 1
    assert unload_steps.size > 0, "Synthetic history did not trigger an unload step."
    t_unload = int(unload_steps[0])

    npt.assert_allclose(
        stress_const[t_unload, :, 0, 0],
        stress_const[t_unload - 1, :, 0, 0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    npt.assert_allclose(
        stress_lin[t_unload, :, 0, 0],
        2.0 * stress_lin[t_unload - 1, :, 0, 0] - stress_lin[t_unload - 2, :, 0, 0],
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    assert not np.allclose(
        stress_none[t_unload, :, 0, 0],
        stress_const[t_unload, :, 0, 0],
        rtol=1.0e-10,
        atol=1.0e-10,
    )


def test_radial_return_output_shapes_and_finiteness() -> None:
    mechanical_properties = _make_mechanical_properties(
        elastic_modulus=190000.0,
        poissons_ratio=0.28,
        yield_strength=320.0,
        hardening_modulus=3000.0,
    )

    rng = np.random.default_rng(1234)
    strain = rng.normal(loc=0.0, scale=8.0e-4, size=(5, 3, 3, 4)).astype(np.float64)

    stress, equivalent_stress, yield_map, peeq = radial_return(strain, mechanical_properties)

    assert stress.shape == (5, 3, 3, 4)
    assert equivalent_stress.shape == (5, 3, 4)
    assert yield_map.shape == (5, 3, 4)
    assert peeq.shape == (5, 3, 4)

    assert np.isfinite(stress).all()
    assert np.isfinite(equivalent_stress).all()
    assert np.isfinite(peeq).all()
