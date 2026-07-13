# from __future__ import annotations

# from pathlib import Path

# import numpy as np
# import numpy.testing as npt
# import pytest

# from pyvale.vfm.mechanical_properties import (
#     EConstituitiveLaw,
#     EParameterName,
#     HomogeneousParameter,
#     MechanicalProperties,
#     ParameterBounds,
# )
# from pyvale.vfm.radial_return import radial_return


# FIXTURE_PATH = Path(__file__).parent / "fixtures" / "radial_return_downsampled_case.npz"


# def _make_mechanical_properties(
#     elastic_modulus: float,
#     poissons_ratio: float,
#     yield_strength: float,
#     hardening_modulus: float,
# ) -> MechanicalProperties:
#     """Build a simple linear-hardening material for the radial-return tests."""

#     return MechanicalProperties(
#         EConstituitiveLaw.LinearHardening,
#         {
#             EParameterName.ElasticModulus: HomogeneousParameter(
#                 bounds=ParameterBounds(1.0, 1.0e9),
#                 value=elastic_modulus,
#             ),
#             EParameterName.PoissonsRatio: HomogeneousParameter(
#                 bounds=ParameterBounds(0.0, 0.49),
#                 value=poissons_ratio,
#             ),
#             EParameterName.YieldStrength: HomogeneousParameter(
#                 bounds=ParameterBounds(1.0, 1.0e9),
#                 value=yield_strength,
#             ),
#             EParameterName.HardeningModulus: HomogeneousParameter(
#                 bounds=ParameterBounds(0.0, 1.0e9),
#                 value=hardening_modulus,
#             ),
#         },
#     )


# def _load_mechanical_properties_from_fixture(
#     fixture_data: np.lib.npyio.NpzFile,
# ) -> MechanicalProperties:
#     """Rebuild the material properties stored in the regression fixture."""

#     return _make_mechanical_properties(
#         elastic_modulus=float(fixture_data["elastic_modulus"]),
#         poissons_ratio=float(fixture_data["poissons_ratio"]),
#         yield_strength=float(fixture_data["yield_strength"]),
#         hardening_modulus=float(fixture_data["hardening_modulus"]),
#     )


# def _plane_stress_stiffness(
#     elastic_modulus: float,
#     poissons_ratio: float,
# ) -> np.ndarray:
#     """Return the plane-stress Hooke matrix using engineering shear strain."""

#     return elastic_modulus / (1.0 - poissons_ratio**2) * np.array(
#         [
#             [1.0, poissons_ratio, 0.0],
#             [poissons_ratio, 1.0, 0.0],
#             [0.0, 0.0, 0.5 * (1.0 - poissons_ratio)],
#         ],
#         dtype=np.float64,
#     )


# def _elastic_expected_stress(
#     strain: np.ndarray,
#     elastic_modulus: float,
#     poissons_ratio: float,
# ) -> np.ndarray:
#     """Compute the elastic stress history directly from Hooke's law."""

#     stiffness = _plane_stress_stiffness(elastic_modulus, poissons_ratio)

#     strain_increment = np.zeros_like(strain)
#     strain_increment[0] = strain[0]
#     strain_increment[1:] = np.diff(strain, axis=0)
#     strain_increment[:, 2] *= 2.0

#     stress = np.zeros_like(strain)

#     for timestep in range(strain.shape[0]):
#         previous_stress = (
#             stress[timestep - 1]
#             if timestep > 0
#             else np.zeros_like(stress[0])
#         )
#         strain_increment_flat = np.moveaxis(strain_increment[timestep], 0, -1).reshape(-1, 3)
#         previous_stress_flat = np.moveaxis(previous_stress, 0, -1).reshape(-1, 3)
#         stress_flat = previous_stress_flat + strain_increment_flat @ stiffness
#         stress[timestep] = np.moveaxis(
#             stress_flat.reshape(strain.shape[2], strain.shape[3], 3),
#             -1,
#             0,
#         )

#     return stress


# def test_radial_return_zero_strain_returns_zero_everything() -> None:
#     """Zero strain should leave every output identically zero."""

#     mechanical_properties = _make_mechanical_properties(
#         elastic_modulus=210000.0,
#         poissons_ratio=0.3,
#         yield_strength=320.0,
#         hardening_modulus=3000.0,
#     )
#     strain = np.zeros((5, 3, 2, 4), dtype=np.float64)

#     stress, equivalent_stress, yield_map, peeq = radial_return(strain, mechanical_properties)

#     npt.assert_allclose(stress, 0.0, rtol=0.0, atol=0.0)
#     npt.assert_allclose(equivalent_stress, 0.0, rtol=0.0, atol=0.0)
#     npt.assert_array_equal(yield_map, np.zeros((5, 2, 4), dtype=bool))
#     npt.assert_allclose(peeq, 0.0, rtol=0.0, atol=0.0)


# def test_radial_return_elastic_response_matches_hookes_law() -> None:
#     """With a very high yield strength the update should stay fully elastic."""

#     elastic_modulus = 190000.0
#     poissons_ratio = 0.28
#     mechanical_properties = _make_mechanical_properties(
#         elastic_modulus=elastic_modulus,
#         poissons_ratio=poissons_ratio,
#         yield_strength=1.0e9,
#         hardening_modulus=0.0,
#     )

#     # Use several points and timesteps so we exercise the vectorised path,
#     # while still keeping the expected answer easy to compute analytically.
#     strain = np.zeros((3, 3, 2, 3), dtype=np.float64)
#     strain[0, 0] = np.array([[1.0e-4, 2.0e-4, -1.0e-4], [5.0e-5, -3.0e-5, 8.0e-5]])
#     strain[0, 1] = np.array([[2.0e-5, -1.5e-4, 3.0e-5], [9.0e-5, 1.1e-4, -2.0e-5]])
#     strain[0, 2] = np.array([[4.0e-5, -2.0e-5, 1.0e-5], [3.0e-5, -1.0e-5, 2.0e-5]])
#     strain[1, 0] = strain[0, 0] + np.array([[3.0e-5, -1.0e-5, 4.0e-5], [2.0e-5, 1.0e-5, -2.0e-5]])
#     strain[1, 1] = strain[0, 1] + np.array([[2.0e-5, 1.0e-5, -1.0e-5], [-2.0e-5, 3.0e-5, 2.0e-5]])
#     strain[1, 2] = strain[0, 2] + np.array([[1.0e-5, -1.0e-5, 2.0e-5], [0.0, 1.0e-5, -1.0e-5]])
#     strain[2] = 1.5 * strain[1]

#     stress, equivalent_stress, yield_map, peeq = radial_return(strain, mechanical_properties)
#     expected_stress = _elastic_expected_stress(strain, elastic_modulus, poissons_ratio)

#     npt.assert_allclose(stress, expected_stress, rtol=1.0e-10, atol=1.0e-10)
#     npt.assert_array_equal(yield_map, np.zeros((3, 2, 3), dtype=bool))
#     npt.assert_allclose(peeq, 0.0, rtol=0.0, atol=0.0)
#     assert equivalent_stress.shape == (3, 2, 3)


# def test_radial_return_unloading_modes_behave_as_expected() -> None:
#     """Check the two unloading options against their simple defining rules."""

#     mechanical_properties = _make_mechanical_properties(
#         elastic_modulus=210000.0,
#         poissons_ratio=0.3,
#         yield_strength=250.0,
#         hardening_modulus=1500.0,
#     )

#     # Load into plasticity and then unload so the special unloading handling is used.
#     strain = np.zeros((4, 3, 1, 1), dtype=np.float64)
#     strain[:, 0, 0, 0] = np.array([0.0, 1.8e-3, 3.2e-3, 1.2e-3])

#     stress_none, _, yield_map, _ = radial_return(
#         strain,
#         mechanical_properties,
#         unloading="no_compensation",
#     )
#     stress_constant, _, _, _ = radial_return(
#         strain,
#         mechanical_properties,
#         unloading="constant_strain",
#     )
#     stress_linear, _, _, _ = radial_return(
#         strain,
#         mechanical_properties,
#         unloading="linear_extrapolation",
#     )

#     unload_steps = np.where(yield_map[:-1, 0, 0] & (~yield_map[1:, 0, 0]))[0] + 1
#     assert unload_steps.size > 0, "Synthetic history did not trigger an unload step."
#     timestep = int(unload_steps[0])

#     npt.assert_allclose(
#         stress_constant[timestep, :, 0, 0],
#         stress_constant[timestep - 1, :, 0, 0],
#         rtol=1.0e-12,
#         atol=1.0e-12,
#     )
#     npt.assert_allclose(
#         stress_linear[timestep, :, 0, 0],
#         2.0 * stress_linear[timestep - 1, :, 0, 0]
#         - stress_linear[timestep - 2, :, 0, 0],
#         rtol=1.0e-12,
#         atol=1.0e-12,
#     )
#     assert not np.allclose(
#         stress_none[timestep, :, 0, 0],
#         stress_constant[timestep, :, 0, 0],
#         rtol=1.0e-10,
#         atol=1.0e-10,
#     )


# def test_radial_return_downsampled_regression_fixture() -> None:
#     """Keep a compact regression check based on saved real-data strain history."""

#     if not FIXTURE_PATH.exists():
#         pytest.skip(
#             f"Regression fixture not found at {FIXTURE_PATH}. "
#             "Generate it with tests/vfm/generate_radial_return_fixture.py"
#         )

#     with np.load(FIXTURE_PATH, allow_pickle=False) as fixture_data:
#         mechanical_properties = _load_mechanical_properties_from_fixture(fixture_data)
#         strain = fixture_data["strain"]
#         unloading = str(fixture_data["unloading_mode"])
#         error_tolerance = float(fixture_data["error_tolerance"])
#         iteration_limit = int(fixture_data["iteration_limit"])

#         expected_stress = fixture_data["stress_expected"]
#         expected_equivalent_stress = fixture_data["equivalent_stress_expected"]
#         expected_yield_map = fixture_data["yield_map_expected"]
#         expected_peeq = fixture_data["peeq_expected"]

#     stress, equivalent_stress, yield_map, peeq = radial_return(
#         strain,
#         mechanical_properties,
#         error_tolerance=error_tolerance,
#         iteration_limit=iteration_limit,
#         unloading=unloading,
#     )

#     npt.assert_allclose(stress, expected_stress, rtol=1.0e-12, atol=1.0e-12)
#     npt.assert_allclose(
#         equivalent_stress,
#         expected_equivalent_stress,
#         rtol=1.0e-12,
#         atol=1.0e-12,
#     )
#     npt.assert_array_equal(yield_map, expected_yield_map)
#     npt.assert_allclose(peeq, expected_peeq, rtol=1.0e-12, atol=1.0e-12)
