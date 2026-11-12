# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""Regenerate the small 2D plate validation fixture from packaged Exodus files.

Run with ``.venv/bin/python scripts/generate_validation_fixture.py``. The
packaged nominal and biased cases use conductivity and thermal expansion factors
1.0 and 0.9, respectively. SensorSim supplies the virtual measurement arrays;
fixed NumPy noise makes the stored regression fixture reproducible despite the
ExperimentSimulator's automatic error-chain reseeding.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy

import pyvale.data as dataset
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens


def save_plate_arrays(
    output_path: Path,
    coords: np.ndarray,
    times: np.ndarray,
    sim_temp: np.ndarray,
    exp_temp: np.ndarray,
    sim_strain: np.ndarray,
    exp_strain: np.ndarray,
) -> dict[str, dict[str, list[str]]]:
    """Save validation-ready variants from SensorSim (repeat, point, component, time).

    Scenario 0 is the nominal model; scenario 1 is the perturbed model. The
    perturbed observations also supply the synthetic experiment for these
    controlled examples. All saved validation arrays retain repeats last.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    raw_arrays = {
        "sim_temp": sim_temp,
        "exp_temp": exp_temp,
        "sim_strain": sim_strain,
        "exp_strain": exp_strain,
    }
    np.savez_compressed(
        output_path / "plate_sensorsim_2d.npz",
        coords=coords,
        times=times,
        **raw_arrays,
    )
    fixture_axes = {}
    for transient, epistemic, filename in (
        (True, True, "plate_validation_2d.npz"),
        (True, False, "plate_validation_2d_nominal.npz"),
        (False, True, "plate_validation_2d_steady.npz"),
        (False, False, "plate_validation_2d_steady_nominal.npz"),
    ):
        arrays = {}
        for key, samples in raw_arrays.items():
            physical_samples = samples if transient else samples[..., -1]
            arrays[key] = np.moveaxis(physical_samples, 0, -1)
        if epistemic:
            for quantity in ("temp", "strain"):
                arrays[f"sim_{quantity}"] = np.stack(
                    (arrays[f"sim_{quantity}"], arrays[f"exp_{quantity}"]), axis=-2
                )
        np.savez_compressed(
            output_path / filename,
            coords=coords,
            **({"times": times} if transient else {}),
            **arrays,
        )
        physical_axes = (
            ["point", "component", "time"] if transient else ["point", "component"]
        )
        fixture_axes[filename] = {
            key: [
                *physical_axes,
                *(["epis_trans"] if epistemic and key.startswith("sim_") else []),
                "aleatory",
            ]
            for key in arrays
        }
    fixture_axes["plate_sensorsim_2d.npz"] = {
        key: ["repeat", "point", "component", "time"] for key in raw_arrays
    }
    return fixture_axes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "pyvale-data/Data/valid",
        help="Destination in the pyvale-data checkout.",
    )
    output_path = parser.parse_args().output_dir
    nominal_path, perturbed_path = dataset.thermomechanical_2d_experiment_paths()
    nominal = mh.ExodusLoader(nominal_path).load_all_sim_data()
    perturbed = mh.ExodusLoader(perturbed_path).load_all_sim_data()
    for simulation in (nominal, perturbed):
        sens.scale_length_units(1000.0, simulation, disp_keys=("disp_x", "disp_y"))

    x = np.linspace(10.0, 90.0, 4)
    y = np.linspace(10.0, 40.0, 3)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coords = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    times = np.linspace(0.0, float(nominal.time[-1]), 5)
    sensor_data = sens.SensorData(positions=coords, sample_times=times)

    temp = sens.SensorFactory.scalar_point(
        nominal,
        sensor_data,
        "temperature",
        sens.EDim.TWOD,
        descriptor=sens.DescriptorFactory.temperature(),
    )
    strain = sens.SensorFactory.tensor_point(
        nominal,
        sensor_data,
        norm_comp_keys=("strain_xx", "strain_yy"),
        dev_comp_keys=("strain_xy",),
        spatial_dims=sens.EDim.TWOD,
    )
    # Zero stochastic error keeps SensorSim's output deterministic. The
    # measurement noise below uses a fixed seed and known physical units.
    temp.set_error_chain([sens.ErrRandGen(sens.GenNormal(std=0.0))])
    strain.set_error_chain([sens.ErrRandGen(sens.GenNormal(std=0.0))])
    simulator = sens.ExperimentSimulator(
        {"nominal": nominal, "perturbed": perturbed},
        {"temperature": temp, "strain": strain},
        sens.ExpSimOpts(workers=None),
        sens.ExpSimSaveKeys(
            sys=None, rand=None, pert_sens_pos=None, pert_sens_times=None
        ),
    )
    output = simulator.run_experiments(num_exp_per_sim=6)
    rng = np.random.default_rng(2871)
    for sim_key in ("nominal", "perturbed"):
        output[(sim_key, "temperature", "meas")] += rng.normal(
            0.0, 0.3, size=output[(sim_key, "temperature", "meas")].shape
        )
        output[(sim_key, "strain", "meas")] += rng.normal(
            0.0, 2.0e-6, size=output[(sim_key, "strain", "meas")].shape
        )

    fixture_axes = save_plate_arrays(
        output_path,
        coords,
        times,
        output[("nominal", "temperature", "meas")],
        output[("perturbed", "temperature", "meas")],
        output[("nominal", "strain", "meas")],
        output[("perturbed", "strain", "meas")],
    )
    metadata = {
        "fixture_axes": fixture_axes,
        "scenario_order": ["nominal", "perturbed"],
        "synthetic_experiment": "perturbed",
        "steady_time": float(times[-1]),
        "source_files": [nominal_path.name, perturbed_path.name],
        "source_inputs": ["case18.i", "case18_d.i"],
        "conductivity_factors": [1.0, 0.9],
        "thermal_expansion_factors": [1.0, 0.9],
        "sensor_grid": [4, 3],
        "experiments_per_simulation": 6,
        "evaluation_times": times.tolist(),
        "measurement_noise_seed": 2871,
        "temperature_noise_std_degC": 0.3,
        "strain_noise_std": 2.0e-6,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "regression_absolute_tolerance": {
            "temperature": 1.0e-10,
            "strain": 1.0e-12,
        },
    }
    metadata_path = output_path / "plate_validation_2d_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
