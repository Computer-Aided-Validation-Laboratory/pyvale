# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""FEMU: identify elastic modulus and Poisson's ratio
==================================================

Identify elastic modulus and Poisson's ratio using PyVale's embedded 2D
plate-with-hole case17. Reaction force comes from the Exodus global variable
react_y_top; the nodal fields are transverse strain_xx and axial strain_yy. A mesh is generated once with Python
Gmsh, and every optimiser candidate performs a real Rabbit solve on that mesh.

Case17 uses millimetres, seconds and MPa (N/mm^2), with weak plane stress and
unit out-of-plane thickness. Multiplying its line reaction (N/mm) by the known
measurement thickness in mm gives force in N. We fit modulus in GPa and pass it
to MOOSE in MPa; Poisson's ratio is dimensionless. A single time step reaches
the original final displacement of 1 mm; it is sufficient for this small-strain linear-elastic identification.

The original prescribed-displacement loading is preserved. At fixed Poisson's
ratio, strain is effectively independent of elastic modulus. Its transverse
contraction and spatial distribution do depend on Poisson's ratio, however.
The strain residuals therefore constrain Poisson's ratio, while reaction force
provides the stress scale needed to identify modulus jointly with it.

Synthetic measurements are generated once, with a fixed seed. Noise floors
are illustrative standard deviations (100 N and 50 microstrain), not calibrated
instrument specifications. Matching mesh nodes avoids interpolation in this
small tutorial; real FEMU requires mapping simulation fields to measurements.
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.result import Result
from pymoo.problems.static import StaticProblem

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh

# %%
# Use a fresh example session; particle directories are reused within it.
SHOW_FIGURES = False
if not SHOW_FIGURES:
    matplotlib.use("Agg")
output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)
example_path: Path = mh.get_next_run_path(output_path / "ex05b_femu")
example_path.mkdir()
input_path: Path = dataset.sim_case_input_file_path(17)
print(f"Packaged MOOSE input: {input_path}")
mesh: mh.RunResult = mh.GmshRunner().run(
    dataset.sim_case_gmsh_file_path(17),
    output_dir=example_path / "mesh",
)
base_input_params: dict[str, mh.InputValues] = {
    "Mesh/file": str(mesh.output_file),
    "endTime": 1,
    "timeStep": 1,
}
true_modulus_pa = 108e9
true_poisson_ratio = 0.33
measurement_depth_mm = 1.0
sigma_force_n = 100.0
sigma_strain = 50e-6


def load_measurements(result: mh.RunResult) -> tuple[float, np.ndarray, np.ndarray]:
    """Read the final tensile reaction, strain components and nodal positions.

    Parameters
    ----------
    result : mh.RunResult
        Successful case17 solve with an Exodus output file.

    Returns
    -------
    force_n : float
        Final positive top reaction in N, from react_y_top (N/mm) multiplied by
        the known measurement_depth_mm.
    strain : np.ndarray
        Dimensionless nodal strains with shape (N, 2), ordered strain_xx, strain_yy.
    positions_mm : np.ndarray
        Matching reference-mesh nodal coordinates with shape (N, 3), in mm.

    Notes
    -----
    The loader context closes the NetCDF resource before return. All returned
    arrays are independent in-memory data. Native benchmark reaction and coordinate
    conventions are retained.
    """
    with mh.ExodusLoader(result.output_file) as loader:
        simulation: io.SimData = loader.load_all_sim_data()
    # The top reaction is positive in tension. Keep the benchmark's original
    # reaction convention and integrate through the known measurement thickness.
    force_n = float(simulation.glob_vars["react_y_top"][-1]) * measurement_depth_mm
    strain: np.ndarray = np.column_stack(
        (
            simulation.node_vars["strain_xx"][:, -1],
            simulation.node_vars["strain_yy"][:, -1],
        )
    )
    return force_n, strain, simulation.coords


def calc_cost_components(
    force_n: float,
    strain: np.ndarray,
    observed_force_n: float,
    observed_strain: np.ndarray,
    force_noise_n: float,
    strain_noise: float,
) -> tuple[float, float]:
    """Compute noise-normalised force and full-field strain discrepancies.

    Parameters
    ----------
    force_n : float
        Simulated tensile reaction in N.
    strain : np.ndarray
        Simulated dimensionless strains, shape (N, 2), ordered xx/yy.
    observed_force_n : float
        Measured tensile reaction in N.
    observed_strain : np.ndarray
        Measured dimensionless strains on the same nodes and in the same order,
        shape (N, 2).
    force_noise_n : float
        Positive finite force noise standard deviation in N.
    strain_noise : float
        Positive finite dimensionless strain noise standard deviation, used for
        both components.

    Returns
    -------
    force_cost : float
        Squared force residual divided by its noise variance.
    strain_cost : float
        Mean squared noise-normalised residual over all nodes and both components.

    Raises
    ------
    ValueError
        A noise floor is not finite/positive, or strain shapes do not match the
        nonempty (N, 2) field contract.

    Notes
    -----
    The caller combines these terms with equal modality weights. The strain mean
    keeps its total weight independent of sample count; this is a balanced
    discrepancy, not an unaveraged independent-Gaussian likelihood. Real fields
    require coordinate mapping and appropriate treatment of correlated noise.
    """
    if not (np.isfinite(force_noise_n) and force_noise_n > 0):
        raise ValueError("The force noise floor must be finite and positive.")
    if not (np.isfinite(strain_noise) and strain_noise > 0):
        raise ValueError("The strain noise floor must be finite and positive.")
    if (
        strain.shape != observed_strain.shape
        or strain.ndim != 2
        or strain.shape[1] != 2
        or not strain.size
    ):
        raise ValueError(
            "Strain fields must have matching nonempty (node, 2) shapes ordered xx/yy."
        )
    force_cost = float(((force_n - observed_force_n) / force_noise_n) ** 2)
    strain_cost = float(np.mean(((strain - observed_strain) / strain_noise) ** 2))
    return force_cost, strain_cost


# %%
# Generate noisy synthetic observations once; do not redraw noise per candidate.
reference: mh.RunResult = mh.MooseRunner().run(
    input_path,
    input_params={
        **base_input_params,
        "EMod": true_modulus_pa / 1e6,
        "PRatio": true_poisson_ratio,
    },
    output_dir=example_path / "reference",
)
reference_force_n, reference_strain, positions_mm = load_measurements(reference)
rng: np.random.Generator = np.random.default_rng(123)
observed_force_n = reference_force_n + float(rng.normal(0, sigma_force_n))
observed_strain: np.ndarray = reference_strain + rng.normal(
    0,
    sigma_strain,
    reference_strain.shape,
)
np.savez(
    example_path / "measurements.npz",
    force_n=observed_force_n,
    strain=observed_strain,
    reference_force_n=reference_force_n,
    reference_strain=reference_strain,
    positions_mm=positions_mm,
)

# %%
# Use stable particle identities. Each ID has one current evaluation, regardless
# of which parallel worker executes it. Successful evaluations have their own
# evaluation.json; failures propagate rather than returning a stale objective.
manager: mh.DirectoryManager = mh.DirectoryManager(
    example_path / "particles",
    mode=mh.EDirMode.REUSE,
)
herd: mh.MooseHerd = mh.MooseHerd(
    [mh.SimStep("solve", mh.MooseRunner(), input_path)],
    dir_manager=manager,
    parallel_runners=2,
)
population_size = 12
n_generations = 24
particle_ids: list[str] = [f"particle_{i:04d}" for i in range(1, population_size + 1)]

# %%
# Search in GPa for convenient numerical scaling; pass EMod to MOOSE in MPa.
# Both bounds are inside the stable isotropic elastic domain, E > 0 and nu < 0.5.
problem: Problem = Problem(
    n_var=2,
    n_obj=1,
    xl=np.array([60.0, 0.15]),
    xu=np.array([150.0, 0.45]),
)
algorithm: PSO = PSO(pop_size=population_size, pertube_best=False)
algorithm.setup(problem, termination=("n_gen", n_generations), seed=4, verbose=False)
history: list[list[float]] = []
convergence: list[float] = []
parameter_history: list[list[float]] = []

# %%
# pymoo's ask/tell interface separates candidate generation from real FEM
# evaluation. This fixed-size PSO preserves particle row identities; algorithms
# that reorder or filter candidates need to carry their IDs explicitly.
while algorithm.has_next():
    population: Population = algorithm.ask()
    # Candidate columns are [E in GPa, dimensionless Poisson ratio].
    candidates: np.ndarray = population.get("X")
    if len(candidates) != len(particle_ids):
        raise RuntimeError("The optimiser changed the fixed particle population.")
    case_input_params: list[dict[str, dict[str, mh.InputValues]]] = [
        {
            "solve": {
                **base_input_params,
                "EMod": float(modulus) * 1e3,
                "PRatio": float(poisson),
            }
        }
        for modulus, poisson in candidates
    ]
    evaluations: list[list[mh.RunResult]] = herd.run(
        case_input_params, reuse_ids=particle_ids
    )
    objectives: np.ndarray = np.empty((len(candidates), 1))
    for i, row in enumerate(evaluations):
        predicted_force_n, predicted_strain, predicted_positions = load_measurements(
            row[0]
        )
        if not np.array_equal(predicted_positions, positions_mm):
            raise ValueError(
                "The simulation nodes no longer match the measurement positions."
            )
        force_cost, strain_cost = calc_cost_components(
            predicted_force_n,
            predicted_strain,
            observed_force_n,
            observed_strain,
            sigma_force_n,
            sigma_strain,
        )
        objectives[i, 0] = 0.5 * (force_cost + strain_cost)
        history.append(
            [
                float(candidates[i, 0]),
                float(candidates[i, 1]),
                predicted_force_n,
                float(predicted_strain[:, 0].mean()),
                float(predicted_strain[:, 1].mean()),
                force_cost,
                strain_cost,
                float(objectives[i, 0]),
            ]
        )
    # StaticProblem supplies externally computed values through pymoo's evaluator
    # so its evaluation count and termination bookkeeping remain correct.
    algorithm.evaluator.eval(StaticProblem(problem, F=objectives), population)
    algorithm.tell(infills=population)
    convergence.append(float(algorithm.opt.get("F").min()))
    best_parameters = algorithm.opt.get("X")[0]
    parameter_history.append([float(best_parameters[0]), float(best_parameters[1])])

# %%
# A particle's last position need not be its personal or global best. Re-run the
# returned best parameter pair explicitly to obtain the final fitted force and field.
optimisation: Result = algorithm.result()
identified_modulus_pa = float(optimisation.X[0]) * 1e9
identified_poisson_ratio = float(optimisation.X[1])
best_result: mh.RunResult = herd.run(
    [
        {
            "solve": {
                **base_input_params,
                "EMod": identified_modulus_pa / 1e6,
                "PRatio": identified_poisson_ratio,
            }
        }
    ],
    reuse_ids=["best_fit"],
)[0][0]
fitted_force_n, fitted_strain, fitted_positions = load_measurements(best_result)
np.savez(
    example_path / "best_fit.npz",
    force_n=fitted_force_n,
    strain=fitted_strain,
    positions_mm=fitted_positions,
)
np.savetxt(
    example_path / "evaluations.csv",
    np.asarray(history),
    delimiter=",",
    header="EMod_GPa,PRatio,force_N,mean_strain_xx,mean_strain_yy,force_cost,strain_cost,total_cost",
    comments="",
)
summary = {
    "true_modulus_pa": true_modulus_pa,
    "identified_modulus_pa": identified_modulus_pa,
    "modulus_relative_error": abs(identified_modulus_pa / true_modulus_pa - 1),
    "true_poisson_ratio": true_poisson_ratio,
    "identified_poisson_ratio": identified_poisson_ratio,
    "poisson_absolute_error": abs(identified_poisson_ratio - true_poisson_ratio),
    "cost": float(optimisation.F[0]),
    "initial_best_cost": convergence[0],
    "n_evaluations": int(algorithm.evaluator.n_eval),
    "population_size": population_size,
    "n_generations": n_generations,
    "strain_components": ["strain_xx", "strain_yy"],
    "sigma_force_n": sigma_force_n,
    "case": "case17",
    "reaction_global": "react_y_top",
    "measurement_depth_mm": measurement_depth_mm,
    "sigma_strain": sigma_strain,
    "convergence": convergence,
    "parameter_history": parameter_history,
}
(example_path / "identification.json").write_text(json.dumps(summary, indent=2))
print(f"True modulus: {true_modulus_pa / 1e9:.3f} GPa")
print(f"Identified modulus: {identified_modulus_pa / 1e9:.3f} GPa")
print(f"True Poisson ratio: {true_poisson_ratio:.5f}")
print(f"Identified Poisson ratio: {identified_poisson_ratio:.5f}")
print(f"Objective: {float(optimisation.F[0]):.3f}")

# %%
# Save a headless convergence/strain comparison. Strains are shown in microstrain.
fig: plt.Figure
axes: np.ndarray
fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
generations = np.arange(1, len(convergence) + 1)
axes[0, 0].semilogy(generations, convergence, "o-")
axes[0, 0].set(xlabel="Generation", ylabel="Best dimensionless cost")
estimates: np.ndarray = np.asarray(parameter_history)
axes[1, 0].plot(
    generations, estimates[:, 0] / (true_modulus_pa / 1e9), label="E / true E"
)
axes[1, 0].plot(generations, estimates[:, 1] / true_poisson_ratio, label="nu / true nu")
axes[1, 0].axhline(1, color="grey", linestyle=":")
axes[1, 0].set(xlabel="Generation", ylabel="Estimate / synthetic truth")
axes[1, 0].legend()
for component, name in enumerate(("Transverse strain xx", "Axial strain yy")):
    limits = {
        "vmin": float(
            min(observed_strain[:, component].min(), fitted_strain[:, component].min())
            * 1e6
        ),
        "vmax": float(
            max(observed_strain[:, component].max(), fitted_strain[:, component].max())
            * 1e6
        ),
    }
    for ax, strain, title in zip(
        axes[component, 1:],
        (observed_strain[:, component], fitted_strain[:, component]),
        ("Synthetic observations", "Fitted FEM"),
    ):
        colours = ax.scatter(
            positions_mm[:, 0], positions_mm[:, 1], c=strain * 1e6, s=8, **limits
        )
        ax.set(xlabel="x [mm]", ylabel="y [mm]", title=title, aspect="equal")
    fig.colorbar(colours, ax=list(axes[component, 1:]), label=f"{name} [microstrain]")
fig.savefig(example_path / "femu_fit.png", dpi=160)

# %%
# Interactive display is opt-in; both the PNG and numerical results are saved.
if SHOW_FIGURES:
    plt.show()
plt.close(fig)
