# ==============================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2026 The Computer Aided Validation Team
# ==============================================================================

"""FEMU: identify elastic modulus and Poisson's ratio
==================================================

Connect a normal cost function to MooseHerd and pymoo's minimize(). Fit the
packaged case17 plate with a hole using reaction force and the strain_xx/strain_yy
fields. Force constrains elastic modulus; the strain field constrains Poisson's
ratio. Synthetic observations use fixed seeded noise and the same mesh as the fit.

Case17 uses mm, MPa and seconds, with weak plane stress. Multiply its line reaction
in N/mm by the known thickness in mm to obtain force in N. Optimiser coordinates
are [E in GPa, dimensionless nu]; convert E to MPa for the input deck. Candidates
are evaluated as parallel batches, reusing one directory per candidate row.
"""
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

import pyvale.data as dataset
import pyvale.dataio as io
import pyvale.mooseherder as mh

# %%
# Set illustrative measurement noise floors and the known specimen thickness.
# These constants are shared by synthetic-data generation and the cost function.
FORCE_NOISE_N = 100.0
STRAIN_NOISE = 50e-6
THICKNESS_MM = 1.0
PARALLEL_RUNNERS = 4
N_GENERATIONS = 24


# %%
# Define the objective first: solve a batch of candidates and compare their force
# and strain with the observations. The fixed arguments will be bound below with partial(),
# leaving the optimiser to supply rows of [elastic modulus in GPa, Poisson's ratio].
# Shapes: n_particles is the current batch size (12 in PSO, 1 for the best-fit rerun).
# n_params=2 means [E, nu]; n_strain_comps=2 means [strain_xx, strain_yy].
# n_nodes is the mesh node count; n_times is the number of saved simulation times.
def calc_cost(
    material_params: np.ndarray,  # shape=[n_particles, n_params]
    *,
    herd: mh.MooseHerd,
    base_input_params: dict[str, mh.InputValues],
    observed_force_n: float,
    observed_strain: np.ndarray,  # shape=[n_nodes, n_strain_comps]
    observed_coords: np.ndarray,  # shape=[n_nodes, 3], columns [x, y, z]
) -> np.ndarray:  # shape=[n_particles], one cost per input row
    # The outer list contains one complete case per candidate row. Each case
    # maps "solve" to its parameter dictionary. Submit the WHOLE batch once so
    # MooseHerd can run cases concurrently rather than waiting after each case.
    case_input_params = [
        {
            "solve": {
                **base_input_params,
                "EMod": float(modulus_gpa) * 1000.0,  # GPa -> MPa
                "PRatio": float(poisson_ratio),
            }
        }
        for modulus_gpa, poisson_ratio in material_params
    ]

    reuse_ids = [f"fit_{index}" for index in range(len(material_params))]
    results: list[list[mh.RunResult]] = herd.run(case_input_params, reuse_ids=reuse_ids)
    costs = np.empty(len(material_params))  # shape=[n_particles]

    # Results retain input case order even when solves finish out of order.
    # Each row contains just one stage, so row[0] is that candidate's solve.
    for index, row in enumerate(results):
        with mh.ExodusLoader(row[0].output_file) as loader:
            simulation: io.SimData = loader.load_all_sim_data()

        # react_y_top: shape=[n_times]; [-1] selects one scalar final reaction.
        predicted_force_n = (
            float(simulation.glob_vars["react_y_top"][-1]) * THICKNESS_MM
        )

        # Each nodal field: [n_nodes, n_times]; [:, -1]: [n_nodes].
        # Stack the final xx/yy columns: shape=[n_nodes, n_strain_comps].
        predicted_strain = np.column_stack(
            (
                simulation.node_vars["strain_xx"][:, -1],
                simulation.node_vars["strain_yy"][:, -1],
            )
        )

        if not np.array_equal(simulation.coords, observed_coords):
            raise ValueError(
                "Prediction nodes do not match the observation coordinates."
            )

        if predicted_strain.shape != observed_strain.shape:
            raise ValueError("Prediction and observation strain shapes do not match.")

        force_cost = ((predicted_force_n - observed_force_n) / FORCE_NOISE_N) ** 2
        # Residuals: [n_nodes, n_strain_comps]; mean() reduces both axes to a scalar.
        strain_cost = np.mean(
            ((predicted_strain - observed_strain) / STRAIN_NOISE) ** 2
        )

        costs[index] = 0.5 * (force_cost + strain_cost)

    return costs


# %%
# This small pymoo adapter sends the entire population to the cost function.
# The cost vector [n_particles] becomes one objective column [n_particles, 1].
# Optimiser generation, evaluation and termination remain managed by minimize().
# pymoo supplies the current algorithm so progress uses its actual generation number;
# the separate final best-fit rerun is not counted as another generation.
class FemuProblem(Problem):
    def __init__(self, cost_function: Callable[[np.ndarray], np.ndarray]) -> None:
        super().__init__(
            n_var=2,
            n_obj=1,
            requires_kwargs=True,  # Forward pymoo's algorithm to _evaluate().
            xl=np.array([60.0, 0.15]),  # shape=[n_params], lower bounds for [E, nu]
            xu=np.array([150.0, 0.45]),  # shape=[n_params], upper bounds for [E, nu]
        )

        self.cost_function = cost_function

    def _evaluate(
        self,
        material_params: np.ndarray,  # shape=[n_particles, n_params]
        output: dict[str, np.ndarray],  # output["F"]: shape=[n_particles, 1]
        *args: object,
        algorithm: Algorithm | None = None,
        **kwargs: object,
    ) -> None:

        start_cost = time.perf_counter()
        output["F"] = self.cost_function(material_params)[:, None]  # Add objective axis.
        cost_time = time.perf_counter() - start_cost

        if algorithm is not None:
            print(
                f"Generation {algorithm.n_gen}/{N_GENERATIONS} with "
                f"{len(material_params)} particles took {cost_time:.2f} seconds."
            )


# %%
# Create one mesh for all solves. A single time step reaches the original final
# displacement of 1 mm for this linear-elastic case. Keep each notebook run in a
# fresh session directory; candidate rows reuse separate directories inside it.

SHOW_FIGURES = False
if not SHOW_FIGURES:
    matplotlib.use("Agg")

output_path: Path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

example_path: Path = mh.get_next_run_path(output_path / "mooseherder_ex05a_femu")
example_path.mkdir()

input_file: Path = dataset.sim_case_input_file_path(17)
print(f"Packaged MOOSE input: {input_file}")

mesh: mh.RunResult = mh.GmshRunner().run(
    dataset.sim_case_gmsh_file_path(17), output_dir=example_path / "mesh"
)

base_input_params: dict[str, mh.InputValues] = {
    "Mesh/file": str(mesh.output_file),
    "endTime": 1,
    "timeStep": 1,
}

# %%
# Generate observations using known properties, E=108 GPa and nu=0.33. Add noise
# once so every candidate is compared with the same measurements. The strain
# array has shape=[n_nodes, n_strain_comps], columns xx/yy; force is one scalar.
# reference_data.coords: shape=[n_nodes, 3], columns [x, y, z].
reference: mh.RunResult = mh.MooseRunner().run(
    input_file,
    input_params={**base_input_params, "EMod": 108e3, "PRatio": 0.33},
    output_dir=example_path / "reference",
)

with mh.ExodusLoader(reference.output_file) as loader:
    reference_data: io.SimData = loader.load_all_sim_data()

rng: np.random.Generator = np.random.default_rng(123)
observed_force_n = float(
    reference_data.glob_vars["react_y_top"][-1]
) * THICKNESS_MM + float(rng.normal(0.0, FORCE_NOISE_N))

observed_strain: np.ndarray = np.column_stack(  # shape=[n_nodes, n_strain_comps]
    (
        reference_data.node_vars["strain_xx"][:, -1],
        reference_data.node_vars["strain_yy"][:, -1],
    )
)
# Noise: [n_nodes, n_strain_comps].
observed_strain += rng.normal(0.0, STRAIN_NOISE, size=observed_strain.shape)

# %%
# Run up to PARALLEL_RUNNERS candidate solves concurrently. The batch cost
# function assigns distinct reuse IDs fit_0, fit_1, ... so active solves never
# overwrite each other. These directories are reused for each population batch.
herd: mh.MooseHerd = mh.MooseHerd(
    [mh.SimStep("solve", mh.MooseRunner(), input_file)],
    dir_manager=mh.DirectoryManager(
        example_path / "evaluations", mode=mh.EDirMode.REUSE
    ),
    parallel_runners=PARALLEL_RUNNERS,
)

# %%
# Bind the fixed objective inputs, then optimise just [E in GPa, nu]. The bounds
# lie inside the stable isotropic elastic domain. pymoo manages candidate
# generation, evaluation and termination; save_history records each generation.
objective = partial(
    calc_cost,
    herd=herd,
    base_input_params=base_input_params,
    observed_force_n=observed_force_n,
    observed_strain=observed_strain,
    observed_coords=reference_data.coords,
)

problem: FemuProblem = FemuProblem(objective)

start_minimise = time.perf_counter()
optimisation = minimize(
    problem,
    PSO(pop_size=12, pertube_best=False),
    termination=("n_gen", N_GENERATIONS),
    seed=4,
    save_history=True,
)
minimise_time = time.perf_counter() - start_minimise


# %%
# Rerun the best parameters as a one-row batch so evaluations/fit_0 holds the
# best fit, rather than the final candidate evaluated. Report properties and cost.
# optimisation.X: shape=[n_params]; [None, :] adds a batch axis -> [1, n_params].
# objective() returns shape=[1]; [0] extracts the scalar best-fit cost.
best_cost = float(objective(optimisation.X[None, :])[0])
print(80*"-")
print(f"Optimisation time: {minimise_time:.2f} seconds")
print(f"Elastic modulus: {optimisation.X[0]:.3f} GPa (true: 108 GPa)")
print(f"Poisson's ratio: {optimisation.X[1]:.5f} (true: 0.33)")
print(f"Cost: {best_cost:.3f}")
print(80*"-")

# %%
# Save one simple convergence plot, taking the best objective from each saved
# generation. Interactive display is opt-in; the PNG is always written.
# Each saved objective array has shape=[n_best, 1] (n_best=1 here); min() is scalar.
# convergence and generation numbers: shape=[n_generations] (24 generations here).
convergence: np.ndarray = np.array(  # shape=[n_generations]
    [float(generation.opt.get("F").min()) for generation in optimisation.history]
)
fig, ax = plt.subplots()
ax.semilogy(np.arange(1, len(convergence) + 1), convergence)
ax.set(xlabel="Generation", ylabel="Best dimensionless cost")
fig.savefig(example_path / "convergence.png", dpi=160)
if SHOW_FIGURES:
    plt.show()
plt.close(fig)

# %%
# Keep a compact numerical record for checking or replotting the fit. NPZ keys
# name scalar values, material_params=[E in GPa, nu], the per-generation objective
# vector, and the observed strain array with columns xx/yy. The final predicted
# fields remain available in evaluations/fit_0/solve/results.e.
np.savez(
    example_path / "identification.npz",
    material_params=optimisation.X,
    cost=best_cost,
    convergence=convergence,
    observed_force_n=observed_force_n,
    observed_strain=observed_strain,
    force_noise_n=FORCE_NOISE_N,
    strain_noise=STRAIN_NOISE,
    thickness_mm=THICKNESS_MM,
)
