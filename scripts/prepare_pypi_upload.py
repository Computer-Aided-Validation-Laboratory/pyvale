"""Stage PyVale distributions that are missing from a PyPI release."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen


def load_published_hashes(version: str) -> dict[str, str]:
    url = f"https://pypi.org/pypi/pyvale/{quote(version, safe='')}/json"
    try:
        with urlopen(url, timeout=30) as response:
            release = json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return {}
        raise
    return {entry["filename"]: entry["digests"]["sha256"] for entry in release["urls"]}


def stage_missing_distributions(
    distributions: Path,
    upload_directory: Path,
    published_hashes: dict[str, str],
) -> bool:
    distribution_files = sorted(
        [*distributions.glob("*.whl"), *distributions.glob("*.tar.gz")]
    )
    if not distribution_files:
        raise ValueError(f"No distribution files found in {distributions}")

    missing_distributions: list[Path] = []
    for distribution in distribution_files:
        published_hash = published_hashes.get(distribution.name)
        if published_hash is None:
            missing_distributions.append(distribution)
            continue
        with distribution.open("rb") as distribution_file:
            local_hash = hashlib.file_digest(distribution_file, "sha256").hexdigest()
        if local_hash != published_hash:
            raise ValueError(
                f"PyPI file {distribution.name} has SHA-256 {published_hash}, "
                f"but the build artifact has {local_hash}"
            )

    if missing_distributions:
        upload_directory.mkdir()
        for distribution in missing_distributions:
            shutil.copy2(distribution, upload_directory)
    return bool(missing_distributions)


def prepare_upload() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_tag")
    parser.add_argument("distributions", type=Path)
    parser.add_argument("upload_directory", type=Path)
    arguments = parser.parse_args()

    version = arguments.release_tag.removeprefix("v")
    has_missing = stage_missing_distributions(
        arguments.distributions,
        arguments.upload_directory,
        load_published_hashes(version),
    )
    print("true" if has_missing else "false")


if __name__ == "__main__":
    prepare_upload()
