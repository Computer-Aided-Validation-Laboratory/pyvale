# ============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 Sceptical Rabbit (Lloyd Fletcher)
# ============================================================================
"""Generate committed Riley rabbit multi-mesh image gold data."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

import pyvale.render as render
import pyvale.verif.renderverif as renderverif


def main() -> None:
    """Print a hash or write the trusted rabbit image array."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()

    result = render.Riley(renderverif.riley_memory_config()).render(
        renderverif.riley_rabbit_scene(),
    )
    assert result.images is not None
    digest = hashlib.sha256(result.images.tobytes()).hexdigest()
    if arguments.write:
        repository_root = Path(__file__).resolve().parents[1]
        path = repository_root / "tests" / "render" / "gold_riley" / "rabbits.npy"
        np.save(path, result.images)
        print(path)
    else:
        print(digest)


if __name__ == "__main__":
    main()
