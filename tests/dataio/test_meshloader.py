# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Tests for loading coordinates and connectivity without fields."""

from pathlib import Path

import numpy as np
import pytest

import pyvale.dataio as io
from pyvale.dataio.exceptions import SimLoadErr


GOLD_DIR = Path(__file__).resolve().parent / "txt_gold"


@pytest.mark.parametrize("suffix", (".csv", ".npy"))
def test_mesh_loader_matches_field_loader(suffix: str) -> None:
    """The focused loader must match the existing simulation loader."""
    coords_file = f"hex20_coords{suffix}"
    connect_file = f"hex20_connect1{suffix}"
    load_opts = io.SimLoadOpts()

    mesh = io.MeshLoader(
        load_dir=GOLD_DIR,
        coords_file=coords_file,
        connect_files=connect_file,
        load_opts=load_opts,
    ).load_mesh()
    simulation = io.SimLoaderByField(
        load_dir=GOLD_DIR,
        coords_file=coords_file,
        time_step_file=None,
        node_field_files=None,
        connect_files=connect_file,
        load_opts=load_opts,
    ).load_all_sim_data()

    np.testing.assert_array_equal(mesh.coords, simulation.coords)
    assert mesh.connect is not None
    assert simulation.connect is not None
    assert mesh.connect.keys() == simulation.connect.keys()
    for connect_key in mesh.connect:
        np.testing.assert_array_equal(
            mesh.connect[connect_key],
            simulation.connect[connect_key],
        )

    assert mesh.mesh_type is io.EMeshType.VOL
    assert mesh.time is None
    assert mesh.node_vars is None
    assert mesh.elem_vars is None
    assert mesh.glob_vars is None
    assert not io.check_mesh_convention(mesh)


@pytest.mark.parametrize(
    "connect_files",
    (
        "hex20_connect1.npy",
        "hex20_connect*.npy",
        ["hex20_connect1.npy"],
    ),
)
def test_mesh_loader_accepts_connectivity_selectors(
    connect_files: str | list[str],
) -> None:
    """A file name, pattern, or explicit list can select connectivity."""
    mesh = io.MeshLoader(
        load_dir=GOLD_DIR,
        coords_file="hex20_coords.npy",
        connect_files=connect_files,
    ).load_mesh()

    assert mesh.connect is not None
    assert tuple(mesh.connect) == ("hex20_connect1",)


def test_mesh_loader_can_preserve_raw_connectivity() -> None:
    """Convention conversion can be disabled for diagnostic workflows."""
    mesh = io.MeshLoader(
        load_dir=GOLD_DIR,
        coords_file="hex20_coords.npy",
        connect_files="hex20_connect1.npy",
        enforce_convention=False,
    ).load_mesh()

    assert mesh.connect is not None
    assert int(np.min(mesh.connect["hex20_connect1"])) == 1
    assert io.check_mesh_convention(mesh)


def test_mesh_loader_uses_text_options_and_infers_surface(
    tmp_path: Path,
) -> None:
    """Text options are shared with DataIO and surface type is inferred."""
    coords_path = tmp_path / "coords.txt"
    connect_path = tmp_path / "connect.txt"
    np.savetxt(
        coords_path,
        np.array(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        ),
        delimiter=";",
        header="x;y;z",
        comments="",
    )
    np.savetxt(
        connect_path,
        np.array(((1, 2, 3, 4),), dtype=np.int64),
        delimiter=";",
        header="n0;n1;n2;n3",
        comments="",
        fmt="%d",
    )

    mesh = io.MeshLoader(
        load_dir=tmp_path,
        coords_file=coords_path,
        connect_files=[connect_path.name],
        load_opts=io.SimLoadOpts(
            delimiter=";",
            coord_header=0,
            connect_header=0,
        ),
    ).load_mesh()

    assert mesh.mesh_type is io.EMeshType.SURF
    assert mesh.connect is not None
    np.testing.assert_array_equal(
        mesh.connect["connect"],
        np.array(((0, 1, 2, 3),)),
    )


def test_mesh_loader_rejects_missing_directory(tmp_path: Path) -> None:
    """A missing load directory reports a focused error."""
    with pytest.raises(SimLoadErr, match="is not a directory"):
        io.MeshLoader(
            load_dir=tmp_path / "missing",
            coords_file="coords.csv",
            connect_files="connect.csv",
        )


def test_mesh_loader_rejects_missing_coordinates() -> None:
    """A missing coordinate file is reported when loading starts."""
    loader = io.MeshLoader(
        load_dir=GOLD_DIR,
        coords_file="missing.npy",
        connect_files="hex20_connect1.npy",
    )

    with pytest.raises(FileNotFoundError, match="missing.npy"):
        loader.load_mesh()


def test_mesh_loader_rejects_empty_connectivity_match() -> None:
    """A pattern that selects no connectivity must fail explicitly."""
    loader = io.MeshLoader(
        load_dir=GOLD_DIR,
        coords_file="hex20_coords.npy",
        connect_files="missing_connect*.npy",
    )

    with pytest.raises(FileNotFoundError, match="No connectivity files"):
        loader.load_mesh()
