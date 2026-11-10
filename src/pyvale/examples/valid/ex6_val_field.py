"""Convert prepared SensorSim runs and sample real SimData on a 2D grid.

This example compares nominal plate fields with synthetic measurements from
the packaged plate whose conductivity and thermal expansion are both 10% lower.
It covers the direct SensorSim-to-ExpData bridge and IField sampling without
running MOOSE or regenerating the SensorSim experiment.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np

import pyvale.data as dataset
import pyvale.mooseherder as mh
import pyvale.sensorsim as sens
import pyvale.valid as val
from pyvale.examples.valid.tools import save_diagnostic

with np.load(dataset.valid_data_dir() / "plate_validation_2d.npz") as saved:
    coords = saved["coords"].copy()
    times = saved["times"].copy()
    output = {
        ("perturbed", "temperature", "meas"): saved["perturbed_temperature"].copy(),
        ("perturbed", "strain", "meas"): saved["perturbed_strain"].copy(),
        ("perturbed", "temperature", "sens_times"): times,
        ("perturbed", "strain", "sens_times"): times,
    }

selections = {
    "temperature": ("temperature", 0, coords),
    "strain_xx": ("strain", 0, coords),
    "strain_yy": ("strain", 1, coords),
    "strain_xy": ("strain", 2, coords),
}
acquisitions = val.conv_exp_sim_to_data(
    output,
    sim_key="perturbed",
    selections=selections,
)

geometry = val.ValGeometry(
    val.EValGeometry.POINT, coords, coordinate_units="mm", frame="plate"
)
domain = val.ValDomain(
    {
        name: val.ValObservable(
            geometry,
            val.EValOperator.SAMPLE,
            (name,),
            times,
            "degC" if name == "temperature" else "1",
            time_units="s",
        )
        for name in selections
    }
)

sim_path = dataset.thermomechanical_2d_experiment_paths()[0]
sim_data = mh.ExodusLoader(sim_path).load_all_sim_data()
sens.scale_length_units(1000.0, sim_data, disp_keys=("disp_x", "disp_y"))
temperature = sens.FieldScalar(sim_data, "temperature", sens.EDim.TWOD)
strain = sens.FieldTensor(
    sim_data,
    ("strain_xx", "strain_yy"),
    ("strain_xy",),
    sens.EDim.TWOD,
)
fields = {
    "temperature": temperature,
    "strain_xx": strain,
    "strain_yy": strain,
    "strain_xy": strain,
}
simulation = val.val_from_sim(fields, domain, [((), sim_data)])
cases = [
    val.ValCase(domain, simulation, val.val_from_exp(run, domain))
    for run in acquisitions
]
case = val.combine_val_cases(*cases)
result = val.ValidationEngine([val.MetricAVM(), val.MetricMAVM()]).run(case)

fig, _ = val.plot_val_metric_map(
    result,
    "strain_xx",
    val.EMetricType.AVM,
    selectors={val.EResultAxis.TIME: 4},
)
save_diagnostic(fig, "ex6_val_field_strain_map.png")

fig, _ = val.plot_val_metric_time(
    result,
    "temperature",
    val.EMetricType.AVM,
    selectors={val.EResultAxis.POINT: 5},
)
save_diagnostic(fig, "ex6_val_field_temperature_time.png")
