'''
==============================================================================
TEST: ExodusReader

Authors: Lloyd Fletcher
==============================================================================
'''
from pathlib import Path
from dataclasses import fields
import pytest
import numpy as np
import numpy.typing as npt
from mooseherder.exodusreader import ExodusReader
from mooseherder.simdata import SimData
import tests.herdchecker as hc

NUM_TIME_STEPS = 4
NUM_NODES = 441
NUM_ELEMS_PER_BLOCK = 200
NODES_PER_ELEM = 4
NUM_BLOCKS = 2

CONNECT_NAMES = ('connect1','connect2')

NUM_SIDESETS = 4
SIDESET_NAMES = ('bottom',
                'right',
                'top',
                'left')

NUM_NODE_VARS = 10
NODE_VAR_NAMES = ('disp_x',
             'disp_y',
             'strain_xx',
             'strain_xy',
             'strain_yy',
             'strain_zz',
             'stress_xx',
             'stress_xy',
             'stress_yy',
             'vonmises_stress')

NUM_ELEM_VARS = 8
ELEM_VAR_NAMES = ('strain_xx',
                  'strain_xy',
                  'strain_yy',
                  'strain_zz',
                  'stress_xx',
                  'stress_xy',
                  'stress_yy',
                  'vonmises_stress')


NUM_GLO_VARS = 4
GLO_VAR_NAMES = ('avg_yy_stress',
                'max_y_disp',
                'max_yy_stress',
                'react_y')


@pytest.fixture
def exodus_path() -> Path:
    return hc.OUTPUT_PATH / 'moose-mech-outtest_out.e'

@pytest.fixture
def reader(exodus_path: Path) -> ExodusReader:
    return ExodusReader(exodus_path)


def test_init_reader(reader: ExodusReader,
                     exodus_path: Path) -> None:
    assert reader._exodus_path == exodus_path
    assert reader._data is not None


def tests_init_reader_path_err() -> None:
    err_path = Path().home() / 'no-exist/no_exodus_here.wrong'
    with pytest.raises(FileNotFoundError) as err_info:
        check_reader = ExodusReader(err_path)

    (msg,) = err_info.value.args
    assert msg == 'Exodus file not found at specified path'


@pytest.mark.parametrize(
    ('name_key','expected'),
    (
        ('ss_names', SIDESET_NAMES),
        ('name_nod_var',NODE_VAR_NAMES),
        ('name_elem_var',ELEM_VAR_NAMES),
        ('name_glo_var',GLO_VAR_NAMES)
    )
)
def test_get_names(name_key: str,
                   expected: tuple[str],
                   reader: ExodusReader) -> None:
    check_names = reader.get_names(name_key)
    assert (check_names == expected).all()


def test_get_names_none(reader: ExodusReader) -> None:
    check_names = reader.get_names(None)
    assert check_names is None


def test_get_names_no_key(reader: ExodusReader) -> None:
    check_names = reader.get_names('no-exist')
    assert check_names is None


def test_get_var(reader: ExodusReader) -> None:
    key = 'coordx'
    check_var = reader.get_var(key)
    assert check_var.shape[0] > 0
    assert check_var.shape == (NUM_NODES,)


@pytest.mark.parametrize(
    ('keys','expected'),
    (
        (('top','ss_names','node_ns'),'node_ns3'),
        (('top','ss_names','elem_ss'),'elem_ss3'),
        (('disp_x','name_nod_var','vals_nod_var'),'vals_nod_var1'),
        (('disp_y','name_nod_var','vals_nod_var'),'vals_nod_var2'),
        (('stress_xy','name_nod_var','vals_nod_var'),'vals_nod_var8'),
        (('strain_yy','name_elem_var','vals_elem_var'),'vals_elem_var3'),
        (('stress_xx','name_elem_var','vals_elem_var'),'vals_elem_var5'),
    )
)
def test_get_key(keys: tuple[str,str,str],
                 expected: str,
                 reader: ExodusReader) -> None:

    name = keys[0]
    all_names = reader.get_names(keys[1])
    key_tag = keys[2]

    key = reader.get_key(name,all_names,key_tag) # type: ignore
    assert key == expected


def test_get_var_no_key(reader: ExodusReader) -> None:
    no_key = 'no_exist'
    check_var = reader.get_var(no_key)
    assert check_var.shape[0] == 0


def test_get_connectivity_names(reader: ExodusReader) -> None:
    check_names = reader.get_connectivity_names()
    assert check_names.shape == (NUM_BLOCKS,)
    assert (check_names == np.array(['connect1','connect2'])).all()


def test_get_connectivity(reader: ExodusReader) -> None:
    check_connect = reader.get_connectivity()
    assert check_connect is not None
    assert tuple(check_connect.keys()) == CONNECT_NAMES
    assert check_connect['connect1'].shape == (
        NODES_PER_ELEM,NUM_ELEMS_PER_BLOCK)
    assert check_connect['connect2'].shape == (
        NODES_PER_ELEM,NUM_ELEMS_PER_BLOCK)


def test_get_sideset_names(reader: ExodusReader) -> None:
    check_names = reader.get_sideset_names()
    assert check_names is not None
    assert check_names.shape == (NUM_SIDESETS,)
    assert (check_names == SIDESET_NAMES).all()


def test_get_sidesets_none(reader: ExodusReader) -> None:
    check_sidesets  = reader.get_sidesets(None)
    assert check_sidesets is None


def test_get_all_sidesets(reader: ExodusReader) -> None:
    check_sidesets = reader.get_all_sidesets()
    assert check_sidesets is not None
    assert len(check_sidesets.keys()) == 2*NUM_SIDESETS # NOTE: node+elem = *2
    assert check_sidesets[('bottom','node')].shape[0] > 0
    assert check_sidesets[('bottom','elem')].shape[0] > 0


def test_get_node_var_names(reader: ExodusReader) -> None:
    node_var_names = reader.get_node_var_names()
    assert node_var_names is not None
    assert node_var_names.shape == (NUM_NODE_VARS,)
    assert (node_var_names == NODE_VAR_NAMES).all()


def test_get_node_vars_none(reader: ExodusReader) -> None:
    node_vars = reader.get_node_vars(None)
    assert node_vars is None


def test_get_all_node_vars(reader: ExodusReader) -> None:
    node_vars = reader.get_all_node_vars()
    assert node_vars is not None
    assert len(node_vars.keys()) == NUM_NODE_VARS
    for nn in node_vars:
        assert node_vars[nn].shape == (NUM_NODES,NUM_TIME_STEPS)


def test_get_elem_var_names(reader: ExodusReader) -> None:
    elem_var_names = reader.get_elem_var_names()
    assert elem_var_names is not None
    assert elem_var_names.shape == (NUM_ELEM_VARS,)
    assert (elem_var_names == ELEM_VAR_NAMES).all()


def test_get_num_elem_blocks(reader: ExodusReader) -> None:
    num_blocks = reader.get_num_elem_blocks()
    assert num_blocks == NUM_BLOCKS


def test_get_elem_var_names_and_blocks(reader: ExodusReader) -> None:
    var_names_and_blocks = reader.get_elem_var_names_and_blocks()
    assert var_names_and_blocks is not None
    assert len(var_names_and_blocks) == NUM_ELEM_VARS*NUM_BLOCKS


def test_get_all_elem_vars(reader: ExodusReader) -> None:
    elem_vars = reader.get_all_elem_vars()
    assert elem_vars is not None
    assert len(elem_vars.keys()) == NUM_ELEM_VARS*NUM_BLOCKS
    for ee in elem_vars:
        assert elem_vars[ee].shape == (NUM_ELEMS_PER_BLOCK,NUM_TIME_STEPS)


def test_get_glob_var_names(reader: ExodusReader) -> None:
    glob_var_names = reader.get_glob_var_names()
    assert glob_var_names is not None
    assert glob_var_names.shape == (NUM_GLO_VARS,)
    assert (glob_var_names == GLO_VAR_NAMES).all()


def test_get_glob_vars_none(reader: ExodusReader) -> None:
    glob_vars = reader.get_glob_vars(None)
    assert glob_vars is None


def test_get_all_glob_vars(reader: ExodusReader) -> None:
    glob_vars = reader.get_all_glob_vars()
    assert glob_vars is not None
    assert len(glob_vars.keys()) == (NUM_TIME_STEPS)
    assert tuple(glob_vars.keys()) == GLO_VAR_NAMES
    for gg in glob_vars:
        assert glob_vars[gg].shape == (NUM_TIME_STEPS,)

def test_get_coords(reader: ExodusReader) -> None:
    (coords,_) = reader.get_coords()
    assert coords.shape == (NUM_NODES,3)


@pytest.mark.parametrize(
    ('coord','dim','expected'),
    (
        (np.array([]),4,np.array([0,0,0,0])),
        (np.array([0,1,2,3]),4,np.array([0,1,2,3])),
    )
)
def test_expand_coords(coord: npt.NDArray,
                       dim: int,
                       expected: npt.NDArray,
                       reader: ExodusReader) -> None:
    check = reader._expand_coord(coord,dim)
    assert (check == expected).all()


def test_get_time(reader: ExodusReader) -> None:
    check_time = reader.get_time()
    assert check_time.shape == (NUM_TIME_STEPS,)


def test_get_read_config(reader: ExodusReader) -> None:
    config = reader.get_read_config()

    for ff in fields(config):
        if ff.name != 'time_inds':
            assert getattr(config,ff.name) is not None

    assert (config.sidesets == SIDESET_NAMES).all()
    assert (config.node_vars == NODE_VAR_NAMES).all()
    assert len(config.elem_vars) == NUM_ELEM_VARS*NUM_BLOCKS # type: ignore
    assert (config.glob_vars == GLO_VAR_NAMES).all()


def test_read_sim_data(reader: ExodusReader) -> None:
    config = reader.get_read_config()
    data = reader.read_sim_data(config)
    check_sim_data(data)


def test_read_all_sim_data(reader: ExodusReader) -> None:
    data = reader.read_all_sim_data()
    check_sim_data(data)


def check_sim_data(data: SimData) -> None:
    for ff in fields(data):
        assert getattr(data,ff.name) is not None

    assert data.time.shape == (NUM_TIME_STEPS,)
    assert data.coords.shape == (NUM_NODES,3)

    assert tuple(data.connect.keys()) == CONNECT_NAMES
    for cc in data.connect:
        assert data.connect[cc].shape == (NODES_PER_ELEM,NUM_ELEMS_PER_BLOCK)

    assert len(data.side_sets.keys()) == NUM_SIDESETS*2
    for ss in data.side_sets:
        assert ss[0] in SIDESET_NAMES
        assert ss[1] in ('node','elem')

    assert tuple(data.node_vars.keys()) == NODE_VAR_NAMES
    for nn in data.node_vars:
        assert data.node_vars[nn].shape == (NUM_NODES,NUM_TIME_STEPS)

    assert len(data.elem_vars.keys()) == NUM_ELEM_VARS*NUM_BLOCKS
    for ee in data.elem_vars:
        assert ee[0] in ELEM_VAR_NAMES
        assert ee[1] in (1,2)
        assert data.elem_vars[ee].shape == (NUM_ELEMS_PER_BLOCK,NUM_TIME_STEPS)

    assert tuple(data.glob_vars.keys()) == GLO_VAR_NAMES
    for gg in data.glob_vars:
        data.glob_vars[gg].shape == (NUM_TIME_STEPS,)

