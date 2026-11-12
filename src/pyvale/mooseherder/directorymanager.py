# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ==============================================================================

"""Allocate archived evaluations or reusable directories with stable identities."""

import json
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path


class EDirMode(Enum):
    """Select whether simulation evaluation directories accumulate or are reused.

    Attributes
    ----------
    ARCHIVE : EDirMode
        Preserve each batch under a new sweep_NNNN/run_NNNN directory hierarchy.
    REUSE : EDirMode
        Replace the last evaluation for each explicit, stable reuse ID.
    """

    ARCHIVE = "archive"
    REUSE = "reuse"


class DirectoryManager:
    """Allocate evaluation directories independently of execution worker identities.

    Notes
    -----
    ARCHIVE preserves successive batches. REUSE replaces directories associated
    with explicit IDs such as reuse_0001, and clears only manager-owned paths.
    Filesystem reservations reject concurrent use of the same ID across manager
    instances in REUSE mode. A crashed orchestrator can leave a reuse lock
    requiring manual review.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        mode: EDirMode = EDirMode.ARCHIVE,
    ) -> None:
        """Select an output root and directory allocation policy.

        Parameters
        ----------
        output_dir : Path
            Root for archived sweeps or reusable evaluation directories. Stored as an
            absolute path; allocation occurs when reserve is entered.
        mode : EDirMode, optional
            Archive or reuse policy. Defaults to EDirMode.ARCHIVE.

        Raises
        ------
        TypeError
            mode is not an EDirMode member.
        """
        if not isinstance(mode, EDirMode):
            raise TypeError("mode must be an EDirMode value.")

        self.output_dir = output_dir.resolve()
        self.mode = mode

    @contextmanager
    def reserve(
        self,
        count: int,
        reuse_ids: list[str] | None = None,
    ) -> Iterator[tuple[Path | None, tuple[Path, ...]]]:
        """Reserve a batch before clearing or executing any of its evaluations.

        Parameters
        ----------
        count : int
            Positive number of cases in the batch.
        reuse_ids : list[str] | None, optional
            One unique simple name per case, in case order. Required in REUSE mode and
            rejected in ARCHIVE mode; names must start with a lowercase letter and contain lowercase letters,
            digits and underscores only.

        Yields
        ------
        tuple[Path | None, tuple[Path, ...]]
            Archive sweep directory (None for REUSE) and the ordered case directories.
            The caller creates step directories below these reserved locations.

        Raises
        ------
        ValueError
            Count/IDs are invalid, required IDs are absent, or a reusable path is a
            symlink, unowned, or has an invalid ownership marker.
        RuntimeError
            A reuse ID is already reserved, or 10000 archive names are occupied.

        Notes
        -----
        ARCHIVE tries at most 10000 numbered directories before raising.
        All IDs are reserved and existing ownership markers checked before any previous
        output is removed. Reservations are released on normal exit and exceptions.
        REUSE keeps no historical batch manifest: successful chains publish only their
        latest evaluation.json after execution; failed chains have no completion marker.
        """
        if type(count) is not int or count < 1:
            raise ValueError("count must be a positive integer.")

        if self.mode is EDirMode.ARCHIVE and reuse_ids is not None:
            raise ValueError("reuse_ids are only supported in REUSE mode.")

        if reuse_ids is not None:
            if len(reuse_ids) != count or len(set(reuse_ids)) != count:
                raise ValueError("Provide one unique reuse ID per case.")
            if any(
                not isinstance(name, str)
                or re.fullmatch(r"[a-z][a-z0-9_]*", name) is None
                for name in reuse_ids
            ):
                raise ValueError("Reuse IDs must be snake_case names without paths.")
        if self.mode is EDirMode.REUSE and reuse_ids is None:
            raise ValueError("REUSE mode requires explicit reuse_ids.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.mode is EDirMode.ARCHIVE:
            for number in range(1, 10_001):
                sweep = self.output_dir / f"sweep_{number:04d}"
                try:
                    sweep.mkdir()
                    break
                except FileExistsError:
                    continue
            else:
                raise RuntimeError(
                    "No archive directory available after 10000 attempts."
                )

            yield sweep, tuple(sweep / f"run_{i:04d}" for i in range(1, count + 1))

            return

        reuse_locks_root = self.output_dir / ".reuse_locks"
        reuse_locks_root.mkdir(exist_ok=True)
        reuse_locks: list[Path] = []
        directories = tuple(self.output_dir / name for name in reuse_ids)

        try:
            for name in reuse_ids:
                reuse_lock = reuse_locks_root / name
                try:
                    reuse_lock.mkdir()
                except FileExistsError as err:
                    raise RuntimeError(
                        f"Evaluation {name!r} is already reserved."
                    ) from err

                reuse_locks.append(reuse_lock)

            # Check ownership for all cases before removing any previous output.
            for name, directory in zip(reuse_ids, directories):
                if directory.is_symlink():
                    raise ValueError(f"Refusing to reuse a symlink: {directory}")

                if directory.exists():
                    marker = directory / ".pyvale_evaluation.json"
                    expected = {"owner": "pyvale.mooseherder", "reuse_id": name}
                    if (
                        not marker.is_file()
                        or json.loads(marker.read_text()) != expected
                    ):
                        raise ValueError(
                            f"Refusing to clear an unowned directory: {directory}"
                        )

            for name, directory in zip(reuse_ids, directories):
                if directory.exists():
                    shutil.rmtree(directory)

                directory.mkdir()
                (directory / ".pyvale_evaluation.json").write_text(
                    json.dumps({"owner": "pyvale.mooseherder", "reuse_id": name}),
                    encoding="utf-8",
                )

            yield None, directories

        finally:
            for reuse_lock in reuse_locks:
                reuse_lock.rmdir()
