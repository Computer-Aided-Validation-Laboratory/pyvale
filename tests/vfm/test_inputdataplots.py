import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyvale.vfm.inputdataplots import _plot_strain_components, _scatter_field


def test_strain_diagnostic_uses_percentile_limits_and_labels_clipping() -> None:
    x, y = np.meshgrid(np.arange(10, dtype=float), np.arange(10, dtype=float))
    specimen_mask = np.ones_like(x, dtype=bool)
    strain = np.zeros((2, 3, 10, 10), dtype=float)
    strain[:, :, :, :] = np.linspace(-0.01, 0.01, 100).reshape(10, 10)
    strain[-1, 1, -1, -1] = 10.0

    figure = _plot_strain_components(
        x,
        y,
        strain,
        specimen_mask,
        ("strain_xx", "strain_yy", "strain_xy"),
    )

    last_yy_axis = figure.axes[4]
    norm = last_yy_axis.collections[0].norm
    expected_min, expected_max = np.percentile(strain[-1, 1], (1.0, 99.0))
    assert norm.vmin == pytest.approx(expected_min)
    assert norm.vmax == pytest.approx(expected_max)
    assert any("1st–99th percentiles" in text.get_text() for text in figure.texts)
    plt.close(figure)


def test_scatter_field_rejects_invalid_percentile_pair() -> None:
    x, y = np.meshgrid(np.arange(2, dtype=float), np.arange(2, dtype=float))
    figure, axis = plt.subplots()
    with pytest.raises(ValueError, match="color_percentiles"):
        _scatter_field(
            axis,
            x,
            y,
            x,
            "invalid",
            color_percentiles=(99.0, 1.0),
        )
    plt.close(figure)
