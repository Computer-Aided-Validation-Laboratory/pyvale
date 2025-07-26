'''
==============================================================================
TEST: InputModifier with Gmsh

Authors: Lloyd Fletcher
==============================================================================
'''
import os
from pathlib import Path
import pytest
from mooseherder.inputmodifier import InputModifier


@pytest.fixture
def gmsh_mod():
    input_file = Path('tests/gmsh/gmsh-test.geo')
    return InputModifier(input_file,'//',';')


@pytest.fixture(autouse=True)
def setup_teardown_gmsh():
    # Setup here
    yield
    # Teardown here - remove output files
    test_dir = 'tests/gmsh/'
    all_files = os.listdir(test_dir)
    for ff in all_files:
        if '-mod' in ff:
            os.remove(test_dir + ff)


def test_gmsh_find_vars(gmsh_mod):
    gmsh_mod.find_vars()
    assert gmsh_mod._var_start_ind == 11
    assert gmsh_mod._var_end_ind == 17


def test_gmsh_read_vars(gmsh_mod):
    gmsh_mod.read_vars()
    assert gmsh_mod._vars == {'p0': 0.0015,
                               'p1': 0.001,
                               'p2': 0.0012,
                               'filename': '"gmsh-test.msh"'}


def test_gmsh_get_vars(gmsh_mod):
    assert gmsh_mod.get_vars()  == {'p0': 0.0015,
                                    'p1': 0.001,
                                    'p2': 0.0012,
                                    'filename': '"gmsh-test.msh"'}

def test_gmsh_update_vars(gmsh_mod):
    new_vars = {'p1': 0.0009,
                'p2': 0.001}
    gmsh_mod.update_vars(new_vars)
    assert gmsh_mod._vars == {'p0': 0.0015,
                                'p1': 0.0009,
                                'p2': 0.001,
                                'filename': '"gmsh-test.msh"'}


def test_gmsh_update_vars_error(gmsh_mod):
    new_vars = {'p1': 0.0009,
                'p7': 0.001}
    with pytest.raises(KeyError) as errinfo:
        gmsh_mod.update_vars(new_vars)
    msg, = errinfo.value.args
    assert msg == "Key p7 does not exist in the variables found in the input file. Check input file to make sure the variable exists."


def test_gmsh_write_file(gmsh_mod):
    new_vars = {'p1': 0.0009,
                'p2': 0.001}
    gmsh_mod.update_vars(new_vars)

    mod_file = Path('tests/gmsh/gmsh_vartest-mod.geo')
    gmsh_mod.write_file(mod_file)
    assert os.path.isfile(mod_file)

    gmsh_mod_check = InputModifier(mod_file,'//',';')
    assert gmsh_mod_check._vars == {'p0': 0.0015,
                                    'p1': 0.0009,
                                    'p2': 0.001,
                                    'filename': '"gmsh-test.msh"'}


@pytest.mark.parametrize(
        ('input_str','expected'),
        (
            pytest.param('',('','',''),id='Degenerate blank case'),
            pytest.param('x1 = 1;',('x1',1.0,''),id='Variable only, int case'),
            pytest.param('x2 = 100.0;',('x2',100.0,''),id='Variable only, float case'),
            pytest.param('x3 = 1e3;',('x3',1000.0,''),id='Variable only, exponential case'),
            pytest.param('order = SECOND;',('order','SECOND',''),id='Variable only, string case'),
            pytest.param('x=10; // comment',('x',10.0,' comment'),id='Numeric variable and comment case'),
            pytest.param('// comment',('','',' comment'),id='Comment only case'),
        )
)
def test_extract_var_str_gmsh(input_str,expected,gmsh_mod):
    ext_strs = gmsh_mod._extract_var_str(input_str)
    assert ext_strs == expected
