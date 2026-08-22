from types import SimpleNamespace

import numpy as np

from pyvale.vfm.constlaws import IsotropicVonMisesElastoplasticity
from pyvale.vfm.hardening import HardeningLinear
from pyvale.vfm.optimiserslicewiseindependent import (
    _prepare_slice_constitutive_law,
)


def test_phase_prepared_constitutive_law_keeps_final_tolerance_independent() -> None:
    strain = np.zeros((8, 3, 2, 3), dtype=np.float64)
    strain[:, 0] = np.linspace(0.0, 0.006, strain.shape[0])[:, None, None]
    maps = {
        "elastic_modulus": np.full((2, 3), 190_000.0),
        "poissons_ratio": np.full((2, 3), 0.28),
        "yield_strength": np.full((2, 3), 320.0),
        "hardening_modulus": np.full((2, 3), 3_000.0),
    }
    law = IsotropicVonMisesElastoplasticity(HardeningLinear())

    prepared = law.prepare_for_optimisation(
        strain,
        error_tolerance=1.0e-6,
        fixed_elastic_parameter_maps=maps,
    )

    assert law.error_tolerance == 1.0e-8
    assert law._prepared_inputs is None
    assert prepared.error_tolerance == 1.0e-6
    assert prepared._prepared_inputs is not None
    assert prepared._prepared_inputs.elastic_stress_increment is not None
    reference = law.calculate_stress(strain, maps)
    relaxed = prepared.calculate_stress(strain, maps)
    np.testing.assert_allclose(relaxed, reference, rtol=1.0e-5, atol=1.0e-3)


def test_custom_elastic_parameter_labels_are_used_by_radial_return() -> None:
    strain = np.zeros((3, 3, 1, 1), dtype=np.float64)
    strain[:, 0, 0, 0] = [0.0, 1.0e-4, 2.0e-4]
    maps = {
        "youngs": np.full((1, 1), 200_000.0),
        "nu": np.full((1, 1), 0.3),
        "yield_strength": np.full((1, 1), 1.0e9),
        "hardening_modulus": np.zeros((1, 1)),
    }
    law = IsotropicVonMisesElastoplasticity(
        HardeningLinear(),
        elastic_modulus_label="youngs",
        poissons_ratio_label="nu",
    )

    stress = law.calculate_stress(strain, maps)
    expected_xx = 200_000.0 / (1.0 - 0.3**2) * strain[:, 0, 0, 0]
    np.testing.assert_allclose(stress[:, 0, 0, 0], expected_xx)


def test_slice_law_prepares_local_strain_and_fixed_elastic_maps() -> None:
    strain = np.zeros((4, 3, 1, 2), dtype=np.float64)
    fixed_maps = {
        "elastic_modulus": np.full((1, 2), 190_000.0),
        "poissons_ratio": np.full((1, 2), 0.28),
    }
    law = IsotropicVonMisesElastoplasticity(HardeningLinear())
    full_strain = np.zeros((4, 3, 2, 3), dtype=np.float64)
    full_maps = {
        "elastic_modulus": np.full((2, 3), 190_000.0),
        "poissons_ratio": np.full((2, 3), 0.28),
    }
    phase_law = law.prepare_for_optimisation(
        full_strain,
        error_tolerance=1.0e-6,
        fixed_elastic_parameter_maps=full_maps,
    )

    slice_law = _prepare_slice_constitutive_law(
        phase_law,
        SimpleNamespace(local_strain=strain, fixed_parameter_maps=fixed_maps),
    )

    assert slice_law is not phase_law
    assert slice_law.hardening_function is phase_law.hardening_function
    assert slice_law.error_tolerance == 1.0e-6
    assert slice_law._prepared_inputs is not None
    assert slice_law._prepared_inputs.strain_shape == strain.shape
    assert slice_law._prepared_inputs.elastic_stress_increment is not None
    assert phase_law._prepared_inputs is not None
    assert phase_law._prepared_inputs.strain_shape == full_strain.shape


def test_prepared_law_can_disable_radial_return_caching() -> None:
    strain = np.zeros((3, 3, 1, 1), dtype=np.float64)
    maps = {
        "elastic_modulus": np.full((1, 1), 190_000.0),
        "poissons_ratio": np.full((1, 1), 0.28),
        "yield_strength": np.full((1, 1), 320.0),
        "hardening_modulus": np.full((1, 1), 3_000.0),
    }
    law = IsotropicVonMisesElastoplasticity(HardeningLinear())

    uncached = law.prepare_for_optimisation(
        strain,
        error_tolerance=1.0e-6,
        fixed_elastic_parameter_maps=maps,
        cache_radial_return=False,
    )

    assert not uncached.cache_radial_return
    assert uncached._prepared_inputs is None
    np.testing.assert_allclose(
        uncached.calculate_stress(strain, maps),
        law.calculate_stress(strain, maps),
    )
