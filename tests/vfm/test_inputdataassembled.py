from __future__ import annotations

import numpy as np

from pyvale.vfm import AssembledDataConfig, Edge, EdgeConditions, EEdgeCondition, process_input_data
from pyvale.vfm.inputdataassembled import load_assembled_data


def _edge_conditions() -> EdgeConditions:
    free = Edge(EEdgeCondition.Free, EEdgeCondition.Free)
    return EdgeConditions(free, free, free, free)


def test_load_assembled_data_and_write_exact_output_directory(tmp_path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    x, y = np.meshgrid(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    strain = np.zeros((2, 3, 2, 2))
    force = np.array([[0.0, 0.0], [10.0, 0.0]])
    time = np.array([0.0, 1.0])
    for name, value in (("x", x), ("y", y), ("strain", strain), ("force", force), ("time", time), ("specimen_mask", np.ones((2, 2), dtype=bool))):
        np.save(raw / f"{name}.npy", value)
    config = AssembledDataConfig(raw, thickness=1.0, edge_conditions=_edge_conditions())
    loaded = load_assembled_data(config)
    assert loaded.strain.shape == (2, 3, 2, 2)
    output = tmp_path / "prepared"
    result = process_input_data(config, output_root=output, timestamped=False)
    assert result == output / "experiment_data.yaml"
    assert result.is_file()
    assert (output / "diagnostic_images").is_dir()
