"""Archive/reuse policy, stable identities and exclusive filesystem ownership."""

from pathlib import Path

import pytest

import pyvale.mooseherder as mh


def test_archive_allocations(tmp_path: Path) -> None:
    manager = mh.DirectoryManager(tmp_path / "runs")
    with manager.reserve(2) as (sweep, directories):
        assert sweep.name == "sweep-0001"
        assert [p.name for p in directories] == ["run-0001", "run-0002"]
    with manager.reserve(1) as (sweep, _):
        assert sweep.name == "sweep-0002"


def test_reuse_replaces_only_selected_id(tmp_path: Path) -> None:
    manager = mh.DirectoryManager(tmp_path / "particles", mode=mh.EDirectoryMode.REUSE)
    with manager.reserve(2, ["particle-1", "particle-7"]) as (_, directories):
        for directory in directories:
            (directory / "previous.txt").write_text("old evaluation")
    with manager.reserve(1, ["particle-7"]) as (_, directories):
        assert directories[0].name == "particle-7"
        assert not (directories[0] / "previous.txt").exists()
        assert (manager.output_dir / "particle-1/previous.txt").is_file()
    assert not tuple((manager.output_dir / ".locks").iterdir())


def test_overlapping_reservations_and_cleanup(tmp_path: Path) -> None:
    first = mh.DirectoryManager(tmp_path / "particles", mode=mh.EDirectoryMode.REUSE)
    second = mh.DirectoryManager(tmp_path / "particles", mode=mh.EDirectoryMode.REUSE)
    with first.reserve(1, ["particle-1"]):
        with (
            pytest.raises(RuntimeError, match="already reserved"),
            second.reserve(2, ["particle-7", "particle-1"]),
        ):
            pytest.fail("An overlapping batch must not run.")
        assert not (second.output_dir / "particle-7").exists()
        # A distinct particle may be evaluated while the first is reserved.
        with second.reserve(1, ["particle-7"]):
            pass
    with second.reserve(1, ["particle-1"]):
        pass


@pytest.mark.parametrize("ids", [None, ["a", "a"], ["../escape"], ["/absolute"]])
def test_invalid_ids_rejected_before_allocation(tmp_path: Path, ids) -> None:
    manager = mh.DirectoryManager(tmp_path / "particles", mode=mh.EDirectoryMode.REUSE)
    with (
        pytest.raises(ValueError),
        manager.reserve(2 if ids == ["a", "a"] else 1, ids),
    ):
        pass
    assert not manager.output_dir.exists()


def test_unowned_directory_never_cleared(tmp_path: Path) -> None:
    manager = mh.DirectoryManager(tmp_path / "particles", mode=mh.EDirectoryMode.REUSE)
    directory = manager.output_dir / "particle-1"
    directory.mkdir(parents=True)
    (directory / "important.txt").write_text("keep")
    with (
        pytest.raises(ValueError, match="unowned"),
        manager.reserve(1, ["particle-1"]),
    ):
        pass
    assert (directory / "important.txt").read_text() == "keep"
    assert not tuple((manager.output_dir / ".locks").iterdir())


def test_invalid_mode(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="EDirectoryMode"):
        mh.DirectoryManager(tmp_path, mode="reuse")
