# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Tests for the PyVale and Riley example-data accessors."""

import inspect
import os
from pathlib import Path

import pytest
import riley

import pyvale.data as dataset
from pyvale.data import dataset as dataset_impl


@pytest.fixture(scope="session")
def network_data_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Use an isolated Pooch cache and require the remote data source."""

    previous_data_path = dataset_impl._DATA.path
    previous_local_data = os.environ.pop("PYVALE_DATA", None)
    cache_path = tmp_path_factory.mktemp("pyvale-data")
    dataset_impl._DATA.path = cache_path
    try:
        yield cache_path
    finally:
        dataset_impl._DATA.path = previous_data_path
        if previous_local_data is not None:
            os.environ["PYVALE_DATA"] = previous_local_data


def _paths_from_result(result: object) -> list[Path]:
    """Collect paths returned directly or inside a list/tuple."""

    if isinstance(result, Path):
        return [result]
    if isinstance(result, (list, tuple)):
        return [path for item in result for path in _paths_from_result(item)]
    return []


@pytest.mark.data
@pytest.mark.parametrize(
    "filename",
    tuple(dataset_impl._DATA.registry_files),
    ids=tuple(dataset_impl._DATA.registry_files),
)

def test_registry_file_downloads(network_data_cache: Path,
                                 filename: str) -> None:
    """Fetch each registered file from the remote data repository."""

    del network_data_cache
    path = Path(dataset_impl._DATA.fetch(filename))
    assert path.is_file(), filename


_ACCESSOR_ARGUMENTS = {
    "dic_rigid_exodus_path": (25,),
    "element_case_input_path": (dataset_impl.EElemTest.HEX20,),
    "element_case_output_path": (dataset_impl.EElemTest.HEX20,),
    "sim_case_gmsh_file_path": (17,),
    "sim_case_input_file_path": (17,),
}


def _dataset_accessor_cases() -> list[tuple[str, object, tuple[object, ...]]]:
    """Return all public dataset accessors and their test arguments."""

    cases = []
    for name, function in inspect.getmembers(dataset_impl, inspect.isfunction):
        if function.__module__ != dataset_impl.__name__ or name.startswith("_"):
            continue

        required_parameters = tuple(
            parameter
            for parameter in inspect.signature(function).parameters.values()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        )
        if required_parameters and name not in _ACCESSOR_ARGUMENTS:
            raise AssertionError(f"Add test arguments for {name}")
        cases.append((name, function, _ACCESSOR_ARGUMENTS.get(name, ())))
    return cases


_DATASET_ACCESSOR_CASES = _dataset_accessor_cases()


@pytest.mark.data
@pytest.mark.parametrize(
    "name,function,args",
    _DATASET_ACCESSOR_CASES,
    ids=[case[0] for case in _DATASET_ACCESSOR_CASES],
)
def test_dataset_accessor_fetches_data(network_data_cache: Path,
                                       name: str,
                                       function: object,
                                       args: tuple[object, ...]) -> None:
    """Verify each dataset accessor resolves its packaged data."""

    del network_data_cache
    result = function(*args)

    if name == "riley_rabbit_meshes":
        assert len(result) == 2
        return

    paths = _paths_from_result(result)
    assert paths, name
    for path in paths:
        if any(character in path.name for character in "*?["):
            assert tuple(path.parent.glob(path.name)), name
        else:
            assert path.exists(), name


def test_missing_optional_gmsh_file_returns_none(monkeypatch: pytest.MonkeyPatch, 
                                                 tmp_path: Path) -> None:

    """Cases without a Gmsh input remain represented by ``None``."""

    monkeypatch.setenv("PYVALE_DATA", str(tmp_path))
    assert dataset.sim_case_gmsh_file_path(1) is None


def test_invalid_dataset_inputs_raise_dataset_error() -> None:
    """Public dataset validators reject unsupported inputs clearly."""

    with pytest.raises(dataset.DataSetError):
        dataset.sim_case_input_file_path(0)

    with pytest.raises(dataset.DataSetError):
        dataset.sim_case_input_file_path(dataset.SIM_CASE_COUNT + 1)

    with pytest.raises(dataset.DataSetError):
        dataset.dic_rigid_exodus_path(2)


def test_riley_data_accessors_resolve_packaged_resources() -> None:
    """Riley-owned example assets resolve from the Riley package."""

    paths = (
        riley.data.speckle_texture_path(),
        riley.data.cal_target_texture_path(),
        riley.data.sphere200_case_path(),
        riley.data.platehole_csv_case_path(),
        riley.data.platehole_exodus_path(),
        riley.data.stereocal_case_path(),
        riley.data.rabbit_case_path("riley", "tri3"),
    )
    assert all(path.exists() for path in paths)
