"""
==============================================================================
TEST: InputModifier with MOOSE

Authors: Lloyd Fletcher
==============================================================================
"""
import os
from pathlib import Path
import pytest
from mooseherder.inputmodifier import InputModifier


@pytest.fixture
def moose_mod() -> InputModifier:
    input_file = Path("tests/moose/moose-test.i")
    return InputModifier(input_file, "#", "")


@pytest.fixture(autouse=True)
def setup_teardown_moose():
    # Setup here
    yield
    # Teardown here - remove output files
    test_dir = Path("tests/moose/")
    all_files = os.listdir(test_dir)
    for ff in all_files:
        if "-mod" in ff:
            os.remove(test_dir / ff)


def test_moose_find_vars(moose_mod: InputModifier) -> None:
    moose_mod.find_vars()
    assert moose_mod._var_start_ind == 3
    assert moose_mod._var_end_ind == 14


def test_moose_read_vars(moose_mod: InputModifier) -> None:
    moose_mod.read_vars()
    assert moose_mod._vars == {
        "n_elem_x": 20,
        "n_elem_y": 10,
        "e_modulus": 1000000000.0,
        "p_ratio": 0.3,
        "e_type": "QUAD4",
        "add_vars": "true",
        "y_max": 1.0,
        "x_max": "${fparse 2*y_max}",
    }


def test_moose_get_vars(moose_mod: InputModifier) -> None:
    assert moose_mod.get_vars() == {
        "n_elem_x": 20,
        "n_elem_y": 10,
        "e_modulus": 1000000000.0,
        "p_ratio": 0.3,
        "e_type": "QUAD4",
        "add_vars": "true",
        "y_max": 1.0,
        "x_max": "${fparse 2*y_max}",
    }


def test_moose_update_vars(moose_mod: InputModifier):
    # Changes number of y elements, elastic modulus and element type
    # Does not change add_vars or other variables
    new_vars = {"n_elem_y": 25, "e_modulus": 2e9, "e_type": "QUAD8", "add_vars": "true"}
    moose_mod.update_vars(new_vars)
    assert moose_mod._vars == {
        "n_elem_x": 20,
        "n_elem_y": 25,
        "e_modulus": 2e9,
        "p_ratio": 0.3,
        "e_type": "QUAD8",
        "add_vars": "true",
        "y_max": 1.0,
        "x_max": "${fparse 2*y_max}",
    }


def test_moose_update_vars_error(moose_mod: InputModifier):
    new_vars = {"n_elem_y": 25, "n_elem_z": 50}

    with pytest.raises(KeyError) as err_info:
        moose_mod.update_vars(new_vars)

    (msg,) = err_info.value.args
    assert (
        msg
        == "Key n_elem_z does not exist in the variables found in the input file."
        + " Check input file to make sure the variable exists."
    )


def test_moose_write_file(moose_mod: InputModifier) -> None:
    new_vars = {"n_elem_y": 25, "e_modulus": 2e9, "e_type": "QUAD8"}
    moose_mod.update_vars(new_vars)

    mod_file = Path("tests/moose/moose-test-mod.i")
    moose_mod.write_file(mod_file)
    assert os.path.isfile(mod_file)

    moose_mod_check = InputModifier(mod_file, "#", "")
    assert moose_mod_check._vars == {
        "n_elem_x": 20,
        "n_elem_y": 25,
        "e_modulus": 2e9,
        "p_ratio": 0.3,
        "e_type": "QUAD8",
        "add_vars": "true",
        "y_max": 1.0,
        "x_max": "${fparse 2*y_max}",
    }


def test_moose_get_var_keys(moose_mod: InputModifier):
    assert moose_mod.get_var_keys() == [
        "n_elem_x",
        "n_elem_y",
        "e_modulus",
        "p_ratio",
        "e_type",
        "add_vars",
        "y_max",
        "x_max",
    ]


def test_moose_get_input_file(moose_mod: InputModifier):
    assert moose_mod.get_input_file() == Path("tests/moose/moose-test.i")


@pytest.mark.parametrize(
    ("input_str", "expected"),
    (
        pytest.param("", ("", "", ""), id="Degenerate blank case"),
        pytest.param("\t    \n", ("", "", ""), id="Degenerate whitespace case"),
        pytest.param("x1 = 1", ("x1", 1.0, ""), id="Variable only, int case"),
        pytest.param("x2 = 100.0", ("x2", 100.0, ""), id="Variable only, float case"),
        pytest.param(
            "x3 = 1e3", ("x3", 1000.0, ""), id="Variable only, exponential case"
        ),
        pytest.param(
            "order = SECOND", ("order", "SECOND", ""), id="Variable only, string case"
        ),
        pytest.param(
            "x=10 # comment",
            ("x", 10.0, " comment"),
            id="Numeric variable and comment case",
        ),
        pytest.param(
            "fun = $fparse{2*x}",
            ("fun", "$fparse{2*x}", ""),
            id="String variable, no commentcase",
        ),
        pytest.param(
            "fun = $fparse{2*x} # comment",
            ("fun", "$fparse{2*x}", " comment"),
            id="String variable and comment case",
        ),
        pytest.param(
            "fun = ${fparse 2*x}",
            ("fun", "${fparse 2*x}", ""),
            id="String variable, no comment case, retain whitespace for fparse",
        ),
        pytest.param(
            "fun = ${fparse 2*x} # comment",
            ("fun", "${fparse 2*x}", " comment"),
            id="String variable and comment case, retain whitespace for fparse",
        ),
        pytest.param("# comment", ("", "", " comment"), id="Comment only case"),
    ),
)
def test_extract_var_str_moose(input_str, expected, moose_mod):
    ext_strs = moose_mod._extract_var_str(input_str)
    assert ext_strs == expected
