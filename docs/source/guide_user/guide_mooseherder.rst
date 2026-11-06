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

   tree = dataset.mooseherder_data_dir()
   result = mh.MooseRunner().run(
       tree / "model.i",
       parameters={"EMod": 110e9, "nElemX": 4},
       output_dir=Path.cwd() / "pyvale-output" / "single-run",
   )

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

Each output directory must be new and outside the input tree. This prevents
recursive copying and stale output reuse. Use ``next_run_path(path)`` to select
an unused suffixed path when rerunning a notebook or tutorial. The primary output is ``results.e``;
``Outputs/file_base`` and ``Outputs/exodus`` are managed by the runner. Inputs
with their own named output objects may also write additional files. Run
metadata comprises ``parameters.json``, ``command.json`` and ``run.log``.
For a deck exposing a root output variable such as Rabbit's ``simName``, use
``MooseRunner(output_parameter="simName")`` so that the variable remains used.
Failures raise an exception identifying the log. The runner has a configurable
timeout. Relative external assets should live inside the declared input tree;
absolute external dependencies are the user's responsibility.

Mesh generation and chaining
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``GmshRunner`` accepts a ``geometry=Path("model.geo")`` or
``script=Path("mesh.py")``. Geometry parameters follow Gmsh's ``-setnumber`` and
``-setstring`` convention; use ``DefineConstant`` for overridable geometry
variables. Geometry-only files are meshed by the worker. Existing ``Mesh`` and
``Save`` commands are supported, but remove ``Exit;`` so execution can complete.
The default mesh format is MSH 2.2 for MOOSE compatibility.

Python scripts receive ``--config <path>``. The JSON object contains
``parameters`` and an absolute ``output_path``; the script must write that mesh
and return successfully. A complete implementation is packaged as
``dataset.mooseherder_data_dir() / "mesh_specimen.py"``. User scripts own their
Gmsh initialization and meshing options. Neither path opens a graphical window.

``SimulationStep`` names a runner and its inputs. A solver step can specify
``mesh_from="mesh"`` to receive the earlier mesh step's output through
``Mesh/file``. For a deck using a named mesh variable, set ``mesh_parameter`` to
that variable instead, avoiding an unused variable error.

``MooseHerd.run`` accepts cases keyed by step name. Independent chains can run
concurrently with ``num_parallel``; this is available with serial Rabbit. Each
call creates a new ``sweep-NNNN`` directory containing isolated case/step
folders. Only a completed sweep receives ``sweep.json``. ``SweepLoader`` reads
this manifest and loads a named step's Exodus files in original case order.
Manifest output paths are relative, allowing completed sweeps to be relocated.

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

   runner = mh.MooseRunner(mpi=True, n_tasks=4, n_threads=1)
   result = runner.run(tree / "model_mpi.i", output_dir=output_dir)

A serial Rabbit installation is rejected before launch. ``launcher`` defaults
to ``("mpirun", "-np")``; configure the prefix for another MPI launcher. Install
an MPI-capable Rabbit distribution in the environment used for MPI; simply
setting a flag does not add MPI support to a serial executable.

``MooseRunner(executable=...)`` and ``GmshRunner(executable=...)`` select custom
executables explicitly. Invalid paths never fall back to packaged defaults.
MOOSE runtime environment overrides can be passed through ``env`` without
changing the parent process. A MOOSE checkout and the former ``MooseConfig``
configuration are no longer required. MPI capabilities of custom executables
are the user's responsibility.

The former modifier/runner positional sweep API and DirectoryManager are
replaced by named steps, native parameters and isolated run directories. ``InputModifier`` remains available for deliberate editing
of one marked text block; it does not parse included dependencies. The final
gallery tutorial demonstrates this workflow explicitly.

Historical geometry corrections
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The input audit also fixes two existing geometry errors: case10 now derives
selection bounds from its existing rectangles (width plus armour thickness),
and case25 defines mesh_ref before using it. The former corrects which surfaces
are selected for extrusion and boundary conditions; the constructed rectangle
dimensions are unchanged. Neither case has a stored Exodus baseline to refresh.
