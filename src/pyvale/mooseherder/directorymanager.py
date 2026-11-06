"""Allocate archived evaluations or reusable directories with stable identities."""

import json
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path


class EDirectoryMode(Enum):
    """Whether evaluations accumulate or replace a particle's previous run."""

    ARCHIVE = "archive"
    REUSE = "reuse"


class DirectoryManager:
    """Own evaluation directories independently of the workers executing them.

    ARCHIVE creates sweep-NNNN/run-NNNN directories. REUSE requires explicit
    evaluation IDs, for example particle-0001, and replaces each ID's previous
    directory. Only directories marked as owned by this manager can be cleared.
    Filesystem reservations reject overlapping evaluations of the same ID even
    across different manager instances. Locks are held until results/manifests
    have been written; a crashed orchestrator may leave a lock for manual review.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        mode: EDirectoryMode = EDirectoryMode.ARCHIVE,
    ) -> None:
        if not isinstance(mode, EDirectoryMode):
            raise TypeError("mode must be an EDirectoryMode value.")
        self.output_dir = output_dir.resolve()
        self.mode = mode

    @contextmanager
    def reserve(
        self,
        count: int,
        evaluation_ids: list[str] | None = None,
    ) -> Iterator[tuple[Path | None, tuple[Path, ...]]]:
        """Reserve an entire batch before clearing or executing any evaluation.

        Return the archive sweep directory (None in reuse mode) and the ordered
        case directories. Reuse keeps no historical sweep manifest: each case
        instead receives evaluation.json only after its whole chain succeeds.
        """
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer.")
        if evaluation_ids is not None:
            if len(evaluation_ids) != count or len(set(evaluation_ids)) != count:
                raise ValueError("Provide one unique evaluation ID per case.")
            if any(
                not isinstance(name, str)
                or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name) is None
                for name in evaluation_ids
            ):
                raise ValueError("Evaluation IDs must be simple names without paths.")
        if self.mode is EDirectoryMode.REUSE and evaluation_ids is None:
            raise ValueError("REUSE mode requires explicit evaluation_ids.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.mode is EDirectoryMode.ARCHIVE:
            number = 1
            while True:
                sweep = self.output_dir / f"sweep-{number:04d}"
                try:
                    sweep.mkdir()
                    break
                except FileExistsError:
                    number += 1
            yield sweep, tuple(sweep / f"run-{i:04d}" for i in range(1, count + 1))
            return

        lock_root = self.output_dir / ".locks"
        lock_root.mkdir(exist_ok=True)
        locks: list[Path] = []
        directories = tuple(self.output_dir / name for name in evaluation_ids)
        try:
            for name in evaluation_ids:
                lock = lock_root / name
                try:
                    lock.mkdir()
                except FileExistsError as err:
                    raise RuntimeError(
                        f"Evaluation {name!r} is already reserved."
                    ) from err
                locks.append(lock)
            # Check ownership for all cases before removing any previous output.
            for name, directory in zip(evaluation_ids, directories):
                if directory.is_symlink():
                    raise ValueError(f"Refusing to reuse a symlink: {directory}")
                if directory.exists():
                    marker = directory / ".pyvale-evaluation.json"
                    expected = {"owner": "pyvale.mooseherder", "evaluation_id": name}
                    if (
                        not marker.is_file()
                        or json.loads(marker.read_text()) != expected
                    ):
                        raise ValueError(
                            f"Refusing to clear an unowned directory: {directory}"
                        )
            for name, directory in zip(evaluation_ids, directories):
                if directory.exists():
                    shutil.rmtree(directory)
                directory.mkdir()
                (directory / ".pyvale-evaluation.json").write_text(
                    json.dumps({"owner": "pyvale.mooseherder", "evaluation_id": name}),
                    encoding="utf-8",
                )
            yield None, directories
        finally:
            for lock in locks:
                lock.rmdir()
