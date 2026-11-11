Rabbit, Gmsh and reproducible runs
----------------------------------

``MooseRunner()`` uses the installed ``rabbit-fem`` serial/SMP executable.
``GmshRunner()`` uses the installed Python ``gmsh`` library in an isolated worker.
Neither requires a source checkout or a program on PATH. Both packages are core
PyVale dependencies. On Linux, Gmsh also needs the system OpenGL/GLU libraries
(for example ``libglu1-mesa`` on Ubuntu).

Run inputs and CLI parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pathlib import Path
   import pyvale.data as dataset
   import pyvale.mooseherder as mh

   input_path = dataset.element_case_input_path(dataset.EElemTest.HEX8)
   print(f"Downloaded MOOSE input: {input_path}")
   result = mh.MooseRunner().run(
       input_path,
       input_params={"EMod": 110e9, "nElemX": 4},
       output_dir=Path.cwd() / "pyvale-output" / "single_run",
   )

The tutorials download their MOOSE and Gmsh inputs on demand from ``pyvale-data`` using Pooch:

.. list-table:: Downloaded cases used by the tutorials
   :header-rows: 1
   :widths: 20 40 40

   * - Case
     - Accessor and scripts
     - Physics and native units
   * - case00_HEX8
     - ``element_case_input_path(EElemTest.HEX8)``; case00_HEX8.i
     - 3D thermo-mechanical cube: m, Pa, seconds, degrees Celsius
   * - case17
     - ``sim_case_input_file_path(17)``; case17.i, case17.geo, case17_mesh.py
     - 2D plate with a hole, weak plane stress: mm, MPa, seconds

Print the returned Path to locate an installed script, or inspect it with
``input_path.read_text()``. The inputs and shared solver includes are cached together so relative paths remain valid.
There is no separate tutorial model directory. The small analytic models used
by isolated runner tests live only under ``tests/mooseherder/fixtures``.

The runner copies the declared ``input_root`` tree, preserving relative include
and asset paths. For a single input, its parent is the default root. Supply
``input_root`` explicitly for an ordered list or tuple of entry decks. MOOSE
resolves ``!include`` itself. Later entry decks override earlier decks; CLI
parameters override the merged inputs. No source file is edited.

Keys are native root variable names or slash-separated MOOSE parameter paths.
Values are strings, booleans, integers, finite floats, or flat lists/tuples of
those values. Python callers supply values without shell quoting. An override
that leaves a previously referenced variable unused can cause a MOOSE input
error: prefer a model's exposed root variables over replacing downstream uses.

``InputScalar`` describes a string, boolean, integer or float. ``InputValues``
also accepts a flat list or tuple of these scalars. The shared
``conv_args_dict_to_list(input_params)`` helper converts a dictionary to native
``name=value`` arguments. It validates assignment boundaries and scalar values;
the backend validates parameter-name grammar. ``stage_input_files(input_files,
work_dir, input_root)`` snapshots entry files and their dependencies.

Each output directory must be new and outside the input tree. This prevents
recursive copying and stale output reuse. Use ``get_next_run_path(path)`` to select
the preferred path if unused, then suffixes ``_1``, ``_2``, and so on when
rerunning a notebook or tutorial. The primary output defaults to ``results.e``. Configure its basename with
``MooseRunner(output_file_base="my_results")`` to produce ``my_results.e``;
basenames cannot contain directories.
``Outputs/file_base`` and ``Outputs/exodus`` are managed by the runner. Inputs
with their own named output objects may also write additional files. Run
metadata comprises ``input_params.json``, ``command.json`` and ``run.log``.
For a deck exposing a root output variable such as Rabbit's ``simName``, use
``MooseRunner(output_parameter="simName")`` so that the variable remains used.
``RunResult`` exposes ``output_file``, ``output_dir``, ``command`` and
``log_file``. Failures raise an exception identifying the log. The runner has a configurable
timeout of 3600 seconds (one hour) by default. Relative external assets should live inside the declared input tree;
absolute external dependencies are the user's responsibility.

Output-loader contexts
^^^^^^^^^^^^^^^^^^^^^^

All ``IOutputLoader`` implementations must provide an idempotent ``close()``.
The interface provides ``__enter__`` and ``__exit__`` once for all subclasses;
``Self`` keeps the concrete loader type when using ``with``:

.. code-block:: python

   import pyvale.dataio as io

   with mh.ExodusLoader(result.output_file) as loader:
       sim_data: io.SimData = loader.load_all_sim_data()

For a field-by-field walkthrough, see
:doc:`../examples/mooseherder/ex01b_quickstart_simdata`. It prints all dataclass
fields and available dictionary keys/shapes, extracts boundary histories and
uses ``SimLoadConfig`` to load selected variables and time samples. Default
connectivity uses zero-based indices; side-set node numbers retain one-based
Exodus numbering and must be converted before NumPy indexing.

The NetCDF resource is closed when the block exits normally or raises an
exception. Exceptions are not suppressed. The loaded SimData arrays are owned
in memory and remain usable afterward. Direct callers can still close a loader
explicitly; repeated closure, including inside a with block, is safe. This
lifetime management is separate from directory reuse: load data before replacing
a reusable evaluation directory.

Mesh generation and chaining
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``GmshRunner`` accepts a ``geo_file=Path("model.geo")`` or
``py_script_file=Path("mesh.py")``. Geometry parameters follow Gmsh's ``-setnumber`` and
``-setstring`` convention; use ``DefineConstant`` for overridable geometry
variables. Geometry-only files are meshed by the worker. Existing ``Mesh`` and
``Save`` commands are supported, but remove ``Exit;`` so execution can complete.
The defaults ``output_file_name="mesh.msh"`` and ``msh_version=2.2`` provide
MSH 2.2 for MOOSE compatibility. Set ``msh_version=4.1`` for newer MSH output,
or choose another export format through the extension, for example
``GmshRunner(output_file_name="mesh.inp")``. ``msh_version`` only controls MSH
files. ``mesh_dims`` selects the meshing dimension and ``mesh_order`` sets the
optional element order.

Python scripts receive ``--config <path>``. The JSON object contains
``input_params``, ``mesh_dims``, ``mesh_order``, ``msh_version`` and an absolute
``output_path``; the script must write that mesh
and return successfully. A complete implementation is packaged as
``dataset.sim_case_input_file_path(17).with_name("case17_mesh.py")``. It meshes
the same plate geometry and boundary names as case17.geo, using quadratic
triangles instead of the .geo script's mapped quadrilaterals. User scripts own
their Gmsh initialization and meshing options. Neither path opens a graphical window.

``SimStep.name_subdir`` names both a step subdirectory and the key used in
each case's input-parameter dictionary. ``SimStep`` supplies a runner and its
input files. A solver step can specify
``mesh_from="mesh"`` to receive the earlier mesh step's output through
``Mesh/file``. For a deck using a named mesh variable, set ``mesh_parameter`` to
that variable instead, avoiding an unused variable error.

``MooseHerd.run(case_input_params)`` accepts a list of dictionaries keyed by
``SimStep.name_subdir``. ``SweepLoader.case_input_params`` exposes the same
submitted parameter dictionaries after loading a manifest. Independent chains can run
concurrently with ``parallel_runners``; this is available with serial Rabbit. Each
call creates a new ``sweep_NNNN`` directory containing isolated case/step
folders. Only a completed sweep receives ``sweep.json``. ``SweepLoader`` reads
this manifest and loads a named step's Exodus files in original case order.
Manifest output paths are relative and use forward slashes on every OS,
allowing completed sweeps to be relocated to another computer.

MPI and custom installations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All bundled cases provide a default SMP entry and an ``_mpi.i`` entry sharing
the same physics. ``sim_case_input_file_path(..., mpi=True)`` and
``element_case_input_path(..., mpi=True)`` select the MPI variants. Shared SMP
settings use ILU/GMRES for thermal and coupled cases, and LU/GMRES for small
elastic mechanics cases. Cube cases use Rabbit's tighter nonlinear tolerances;
other families use its standard tolerances. Coupled cases group displacement
variables during automatic scaling. MPI thermal/coupled inputs switch to
Hypre/BoomerAMG/GMRES; small elastic cases retain LU as in Rabbit. These follow
the corresponding Rabbit examples. Units, load definitions, output fields and time
sampling are preserved; changed tolerances can alter numerical results. Stored
Exodus regression datasets are not automatically regenerated.

MPI execution requires both a suitable deck and an MPI-capable executable:

.. code-block:: python

   runner = mh.MooseRunner(n_tasks=4, n_threads=1)
   mpi_input = dataset.element_case_input_path(dataset.EElemTest.HEX8, mpi=True)
   result = runner.run(mpi_input, output_dir=output_dir)

Omitting ``n_tasks`` or setting it to ``None`` or ``1`` runs without MPI.
Counts greater than 1 select MPI; a serial Rabbit installation is rejected
at runner construction and again before launch. ``launcher`` defaults
to ``("mpirun", "-np")``; configure the prefix for another MPI launcher. Install
an MPI-capable Rabbit distribution in the Python installation used for MPI;
simply
setting a rank count does not add MPI support to a serial executable.

``MooseRunner(moose_exe_file=...)`` and ``GmshRunner(gmsh_exe_file=...)`` select custom
executables explicitly. Both runners check their backends at construction and before every run.
``runner.get_backend()`` returns a ``RunnerBackend`` containing the resolved
``backend_file`` path: an executable for MOOSE/custom Gmsh, or the gmsh.py
module for Python Gmsh. Missing/non-executable files and unavailable packages
raise immediately. Invalid paths never fall back to packaged defaults.
OS environment variable overrides can be passed through ``env_vars`` (for
example, ``env_vars={"OMP_NUM_THREADS": "4"}``). These overrides are merged with
the calling Python process's OS environment variables for the solver process.
The calling Python process's OS environment variables are not modified. This
setting does not select or create a Python virtual environment.
A MOOSE checkout and the former ``MooseConfig``
configuration are no longer required. MPI capabilities of custom executables
are the user's responsibility.

The former modifier/runner positional sweep API is replaced by named steps and
native parameters. ``InputModifier`` remains available for deliberate editing
of one marked text block; it does not parse included dependencies. Example :doc:`../examples/mooseherder/ex02d_moose_modify_input_text`
demonstrates this workflow explicitly.

Historical geometry corrections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The input audit also fixes two existing geometry errors: case10 now derives
selection bounds from its existing rectangles (width plus armour thickness),
and case25 defines mesh_ref before using it. The former corrects which surfaces
are selected for extrusion and boundary conditions; the constructed rectangle
dimensions are unchanged. Neither case has a stored Exodus baseline to refresh.

Reusable evaluation directories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``DirectoryManager`` separates allocation policy from execution parallelism.
``EDirMode.ARCHIVE`` is the default: it preserves each sweep in numbered
folders. ``EDirMode.REUSE`` replaces one evaluation directory per explicit,
stable reuse ID:

.. code-block:: python

   manager = mh.DirectoryManager(
       output_path / "reused", mode=mh.EDirMode.REUSE,
   )
   herd = mh.MooseHerd(steps, dir_manager=manager, parallel_runners=4)
   results = herd.run(case_input_params, reuse_ids=["reuse_0001", "reuse_0007"])

Step subdirectory names and reuse IDs use snake_case (lowercase letters, digits and
underscores, starting with a letter). ``reuse_ids`` are required in REUSE mode and rejected in ARCHIVE mode.
IDs are independent of worker number and batch order. A subset can be submitted
on the next iteration. Every requested ID is reserved before any old output is
removed; overlapping batches using the same ID are rejected, while disjoint
batches can run independently. Only marked, manager-owned evaluation directories
can be cleared. Input trees must remain outside the managed output tree.

Each successful reusable chain publishes ``<ID>/evaluation.json`` atomically.
``SweepLoader`` can open that manifest just like an archived sweep manifest,
but it describes only the latest evaluation for that ID. There is no historical
sweep manifest in REUSE mode. Old results and completion markers are removed
before launching replacements, so a failure cannot return a stale result.
Read numerical results before submitting the same IDs again: returned paths
will be reused. A crashed orchestrator may leave a reservation in ``.reuse_locks``;
remove it manually only after verifying that its evaluation is no longer active.

FEMU with pymoo
^^^^^^^^^^^^^^^

Example :doc:`../examples/mooseherder/ex05a_femu` defines a normal cost function and connects it to pymoo's
a small vectorised ``Problem`` adapter and ``minimize()`` using
``functools.partial``. It uses the embedded
``case17`` 2D plate with a hole. It meshes once, then runs a Rabbit solve per
candidate. The reaction force is read from ``SimData.glob_vars["react_y_top"]``;
the transverse and axial strain fields are ``SimData.node_vars["strain_xx"]``
and ``SimData.node_vars["strain_yy"]``. It jointly identifies elastic modulus and
Poisson's ratio (the native ``EMod`` and ``PRatio`` input parameters). Case17 uses mm and MPa,
with weak plane stress; the line reaction in N/mm is multiplied by a known
thickness in mm. Optimiser coordinates in GPa are converted to MPa for the deck.
The original final displacement is reached in a single step for this elastic
demo. The benchmark's original boundary conditions and material model remain
unchanged, and observation/prediction nodes are checked for identical positions.

Synthetic observations have fixed seeded noise, with illustrative standard
deviations of 100 N and 50 microstrain. The dimensionless objective is

.. math::

   J = \tfrac12\left(\frac{F-F_{obs}}{\sigma_F}\right)^2
       + \tfrac{1}{4N}\sum_{i=1}^{N}\sum_{c\in\{xx,yy\}}
          \left(\frac{\epsilon_{c,i}-\epsilon_{c,i}^{obs}}
          {\sigma_\epsilon}\right)^2.

The field mean over both components and nodes prevents its weight increasing
with the number of sample nodes or strain components.
This balances two modalities; it is not the unaveraged likelihood of independent
Gaussian measurements. Real experimental fields require coordinate mapping and
appropriate treatment of correlated noise.

With prescribed displacement, modulus sets the reaction-force scale while
Poisson's ratio changes transverse contraction and the strain distribution
around the hole. The full-field strain residuals constrain Poisson's ratio,
allowing both parameters to be identified jointly. At fixed Poisson's ratio,
the strain field remains effectively invariant to modulus; force measurements
are essential to recover the modulus. The search bounds are 60--150 GPa and
0.15--0.45; the latter stay inside the stable isotropic elastic domain. A seeded
12-particle, 24-generation search performs 288 real candidate solves.

The cost function submits the whole candidate batch to MooseHerd, which runs
up to ``PARALLEL_RUNNERS`` (default 2) solves concurrently. Each candidate row
gets a distinct reusable directory, ``fit_0``, ``fit_1``, and so on, preventing
concurrent writes to the same directory. A small pymoo Problem adapter forwards
each population batch to the cost function; no manual ask/tell loop is needed.
pymoo records the optimisation history. The best parameter pair is rerun as a
one-row batch so ``fit_0`` holds the final best fit.
Outputs include a simple headless convergence figure, ``identification.npz``
with the fitted parameters, cost and observations, and the final Exodus output
at ``evaluations/fit_0/solve/results.e``. No example-specific helper files are used.
