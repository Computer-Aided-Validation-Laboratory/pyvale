"""Regenerate the small 2D plate validation fixture from packaged Exodus files.

Run with ``.venv/bin/python scripts/generate_validation_fixture.py``. The
packaged nominal and biased cases use conductivity and thermal expansion factors
1.0 and 0.9, respectively. SensorSim supplies the virtual measurement arrays;
fixed NumPy noise makes the stored regression fixture reproducible despite the
ExperimentSimulator's automatic error-chain reseeding.
"""

import json
from pathlib import Path

import numpy as np
import scipy

import pyvale.data as dataset
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens


def main() -> None:
    output_path = Path(__file__).resolve().parents[1] / "src/pyvale/data/valid"
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

    np.savez_compressed(
        output_path / "plate_validation_2d.npz",
        coords=coords,
        times=times,
        nominal_temperature=output[("nominal", "temperature", "meas")],
        perturbed_temperature=output[("perturbed", "temperature", "meas")],
        nominal_strain=output[("nominal", "strain", "meas")],
        perturbed_strain=output[("perturbed", "strain", "meas")],
    )
    metadata = {
        "fixture": "plate_validation_2d.npz",
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
