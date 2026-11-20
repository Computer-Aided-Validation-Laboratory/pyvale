# ===============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ===============================================================================

"""Archive/reuse policy, stable identities and exclusive filesystem ownership."""

from pathlib import Path

import pytest

import pyvale.mooseherder as mh


def test_archive_allocations(tmp_path: Path) -> None:
    manager = mh.DirectoryManager(tmp_path / "runs")
    with manager.reserve(2) as (sweep, directories):
        assert sweep.name == "sweep_0001"
        assert [p.name for p in directories] == ["run_0001", "run_0002"]
    with manager.reserve(1) as (sweep, _):
        assert sweep.name == "sweep_0002"


def test_reuse_replaces_only_selected_id(tmp_path: Path) -> None:
    manager = mh.DirectoryManager(tmp_path / "particles", mode=mh.EDirMode.REUSE)
    with manager.reserve(2, ["particle_1", "particle_7"]) as (_, directories):
        for directory in directories:
            (directory / "previous.txt").write_text("old evaluation")
    with manager.reserve(1, ["particle_7"]) as (_, directories):
        assert directories[0].name == "particle_7"
        assert not (directories[0] / "previous.txt").exists()
        assert (manager.output_dir / "particle_1/previous.txt").is_file()
    assert not tuple((manager.output_dir / ".reuse_locks").iterdir())


def test_overlapping_reservations_and_cleanup(tmp_path: Path) -> None:
    first = mh.DirectoryManager(tmp_path / "particles", mode=mh.EDirMode.REUSE)
    second = mh.DirectoryManager(tmp_path / "particles", mode=mh.EDirMode.REUSE)
    with first.reserve(1, ["particle_1"]):
        with (
            pytest.raises(RuntimeError, match="already reserved"),
            second.reserve(2, ["particle_7", "particle_1"]),
        ):
            pytest.fail("An overlapping batch must not run.")
        assert not (second.output_dir / "particle_7").exists()
        # A distinct particle may be evaluated while the first is reserved.
        with second.reserve(1, ["particle_7"]):
            pass
    with second.reserve(1, ["particle_1"]):
        pass


@pytest.mark.parametrize(
    "ids",
    [None, ["a", "a"], ["../escape"], ["/absolute"], ["particle-1"], ["Particle_1"]],
)
def test_invalid_ids_rejected_before_allocation(tmp_path: Path, ids) -> None:
    manager = mh.DirectoryManager(tmp_path / "particles", mode=mh.EDirMode.REUSE)
    with (
        pytest.raises(ValueError),
        manager.reserve(2 if ids == ["a", "a"] else 1, ids),
    ):
        pass
    assert not manager.output_dir.exists()


def test_unowned_directory_never_cleared(tmp_path: Path) -> None:
    manager = mh.DirectoryManager(tmp_path / "particles", mode=mh.EDirMode.REUSE)
    directory = manager.output_dir / "particle_1"
    directory.mkdir(parents=True)
    (directory / "important.txt").write_text("keep")
    with (
        pytest.raises(ValueError, match="unowned"),
        manager.reserve(1, ["particle_1"]),
    ):
        pass
    assert (directory / "important.txt").read_text() == "keep"
    assert not tuple((manager.output_dir / ".reuse_locks").iterdir())


def test_invalid_mode(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="EDirMode"):
        mh.DirectoryManager(tmp_path, mode="reuse")


def test_archive_rejects_reuse_ids_before_allocation(tmp_path: Path) -> None:
    manager = mh.DirectoryManager(tmp_path / "archive")
    with pytest.raises(ValueError, match="REUSE mode"), manager.reserve(1, ["reuse_1"]):
        pass
    assert not manager.output_dir.exists()


def test_archive_collision_search_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = mh.DirectoryManager(tmp_path / "archive")
    manager.output_dir.mkdir()
    original_mkdir = Path.mkdir
    attempts = []

    def collide(path, *args, **kwargs):
        if path.name.startswith("sweep_"):
            attempts.append(path)
            raise FileExistsError(path)
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", collide)
    with pytest.raises(RuntimeError, match="10000 attempts"), manager.reserve(1):
        pass
    assert len(attempts) == 10000
