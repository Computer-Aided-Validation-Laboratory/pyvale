from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from pyvale.vfm.mechanical_properties import (
    EConstituitiveLaw,
    HomogeneousParameter,
    MechanicalProperties,
    ParameterBounds,
    EParameterName,
)
from pyvale.vfm.radial_return import radial_return


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "radial_return_downsampled_case.npz"


def _make_mechanical_properties_from_fixture(data: np.lib.npyio.NpzFile) -> MechanicalProperties:
    return MechanicalProperties(
        ConstituitiveLaw.LinearHardening,
        {
            ParameterName.ElasticModulus: HomogeneousParameter(
                IdentificationType.Known,
                ParameterBounds(1.0, 1.0e9),
                ScalarValue(float(data["elastic_modulus"])),
            ),
            ParameterName.PoissonsRatio: HomogeneousParameter(
                IdentificationType.Known,
                ParameterBounds(0.0, 0.49),
                ScalarValue(float(data["poissons_ratio"])),
            ),
            ParameterName.YieldStrength: HomogeneousParameter(
                IdentificationType.Known,
                ParameterBounds(1.0, 1.0e9),
                ScalarValue(float(data["yield_strength"])),
            ),
            ParameterName.HardeningModulus: HomogeneousParameter(
                IdentificationType.Known,
                ParameterBounds(0.0, 1.0e9),
                ScalarValue(float(data["hardening_modulus"])),
            ),
        },
    )


def test_radial_return_downsampled_real_data_regression() -> None:
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"Regression fixture not found at {FIXTURE_PATH}. "
            "Generate it with tests/vfm/generate_radial_return_fixture.py"
        )

    with np.load(FIXTURE_PATH, allow_pickle=False) as data:
        strain = data["strain"]
        unloading = str(data["unloading_mode"])
        error_tolerance = float(data["error_tolerance"])
        iteration_limit = int(data["iteration_limit"])

        expected_stress = data["stress_expected"]
        expected_equivalent_stress = data["equivalent_stress_expected"]
        expected_yield_map = data["yield_map_expected"]
        expected_peeq = data["peeq_expected"]

        mechanical_properties = _make_mechanical_properties_from_fixture(data)

    stress, equivalent_stress, yield_map, peeq = radial_return(
        strain,
        mechanical_properties,
        error_tolerance=error_tolerance,
        iteration_limit=iteration_limit,
        unloading=unloading,
    )

    npt.assert_allclose(stress, expected_stress, rtol=1.0e-12, atol=1.0e-12)
    npt.assert_allclose(
        equivalent_stress,
        expected_equivalent_stress,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    npt.assert_array_equal(yield_map, expected_yield_map)
    npt.assert_allclose(peeq, expected_peeq, rtol=1.0e-12, atol=1.0e-12)
