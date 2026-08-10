.. _guide_vfm:

VFM User Guide
==============

The Virtual Fields Method (VFM) engine in ``pyvale`` identifies the parameters
of a material constitutive model directly from full-field strain measurements
(for example from digital image correlation) together with the global reaction
force measured during a mechanical test. Rather than running a forward
simulation and iterating a full finite-element model, the VFM uses the
principle of virtual work to compare the *internal* virtual work predicted by a
candidate set of material parameters against the *external* virtual work done
by the measured boundary force, and drives an optimiser until the two agree.

The workflow has four steps:

#. **Process** your raw solver or experiment output into the
   ``ExperimentData`` format using the input data processor.
#. **Load** the processed ``ExperimentData`` back from file.
#. **Set up** the identification: the constitutive model, an initial guess and
   bounds for each parameter, and one or more identification phases.
#. **Run** the identification and inspect the identified parameters.

The complete runnable script for this guide is available in the
:ref:`VFM quickstart example <examples_vfm>`.

Everything below is available from the ``pyvale.vfm`` namespace::

    import numpy as np
    import pyvale.vfm as vfm


1. Process your data into the ExperimentData format
---------------------------------------------------

The VFM needs the measured strain field, the specimen geometry, the boundary
conditions and the reaction-force history, all sampled on a regular grid. The
input data processor takes raw output from a solver (MOOSE or ANSYS) or an
experiment, interpolates the fields onto a regular grid, validates the result,
writes a set of diagnostic images, and saves a portable ``ExperimentData``
bundle to disk (an ``experiment_data.yaml`` file alongside ``.npy`` field
arrays).

You describe the input with a solver-specific configuration
(``MooseConfig`` or
``AnsysConfig``). Both configs require the
specimen ``thickness`` and the ``edge_conditions`` describing the mechanical
boundary condition on each of the four specimen edges. Each edge has an
independent condition in the global ``x`` and ``y`` directions, chosen from
``EEdgeCondition.Free``, ``EEdgeCondition.Fixed`` or ``EEdgeCondition.Traction``
(an edge with a known applied force):

.. code-block:: python

    edge_conditions = vfm.EdgeConditions(
        min_x_edge=vfm.Edge(x=vfm.EEdgeCondition.Free,  y=vfm.EEdgeCondition.Free),
        max_x_edge=vfm.Edge(x=vfm.EEdgeCondition.Free,  y=vfm.EEdgeCondition.Free),
        min_y_edge=vfm.Edge(x=vfm.EEdgeCondition.Fixed, y=vfm.EEdgeCondition.Fixed),
        max_y_edge=vfm.Edge(x=vfm.EEdgeCondition.Free,  y=vfm.EEdgeCondition.Traction),
    )

Here the bottom (minimum ``y``) edge is fully fixed and the top (maximum ``y``)
edge is pulled in the ``y`` direction with a known traction, while the left and
right edges are free.

For a MOOSE exodus output, build a ``MooseConfig`` and run
``process_input_data``:

.. code-block:: python

    input_config = vfm.MooseConfig(
        exodus_file_path="path/to/your/moose_output.e",
        height=50.0,       # specimen height (mm)
        width=50.0,        # specimen width (mm)
        thickness=1.0,     # out-of-plane thickness (mm)
        grid_divs=101,     # interpolation grid divisions per axis
        edge_conditions=edge_conditions,
    )

    # Returns the path to the saved experiment_data.yaml, inside a new
    # timestamped run directory created under output_root.
    experiment_data_file = vfm.process_input_data(input_config, output_root=".")

By default the strain components are read from the exodus keys
``("strain_xx", "strain_yy", "strain_xy")`` and the reaction force from
``"react_y_top"``; override ``strain_component_keys`` and ``force_key`` on the
``MooseConfig`` if your model uses different names.

.. note::

   For ANSYS FE centroid data use an
   ``AnsysConfig`` instead, which points at
   the individual coordinate, strain-component, force and time text files. The
   rest of the workflow is identical.

After processing, inspect the ``diagnostic_images`` written into the run
directory to confirm the fields were loaded and interpolated as expected before
moving on.


2. Load the experiment data from file
--------------------------------------

Processing is decoupled from identification: once the data has been saved you
can reload it at any time with
``load_from_file``, without
repeating the (potentially slow) interpolation step.

.. code-block:: python

    experiment_data = vfm.ExperimentData.load_from_file(experiment_data_file)

An ``ExperimentData`` holds the full-field
strain history (shape ``(timesteps, components, y, x)`` with components ordered
``[xx, yy, xy]``), the ``SpecimenGeometry``
(grid coordinates, per-point area, thickness and region of interest), the
``BoundaryConditions`` (edge conditions and
the measured force history) and the timesteps.

The identified parameter maps span the same grid as the measured strain field,
so it is convenient to derive the map size directly from the loaded geometry:

.. code-block:: python

    map_size = np.array(
        experiment_data.specimen_geometry.x.shape, dtype=np.uint32
    )


3. Set up the identification
----------------------------

**Constitutive model.** Choose the model whose parameters you want to identify.
For example, isotropic von Mises elasto-plasticity with linear (bilinear)
hardening:

.. code-block:: python

    constitutive_law = vfm.IsotropicVonMisesElastoplasticity(vfm.HardeningLinear())

This model exposes four parameters: ``elastic_modulus``, ``poissons_ratio``,
``yield_strength`` and ``hardening_modulus``. Other hardening laws
(``HardeningSwift``,
``HardeningVoce``,
``HardeningLudwik``) are available and expose
their own parameter names.

**Initial parameters.** For each parameter provide an initial guess together
with lower and upper bounds; the optimiser searches within these bounds. A
``ConstitutiveParameter`` created from a scalar
value plus ``map_size`` represents a spatially uniform starting field:

.. code-block:: python

    parameters = {
        "elastic_modulus":   vfm.ConstitutiveParameter(200_000.0, 100_000.0, 300_000.0, map_size),
        "poissons_ratio":    vfm.ConstitutiveParameter(0.3,       0.1,       0.5,       map_size),
        "yield_strength":    vfm.ConstitutiveParameter(250.0,     100.0,     1000.0,    map_size),
        "hardening_modulus": vfm.ConstitutiveParameter(1000.0,    500.0,     10_000.0,  map_size),
    }

**Identification phases.** The search itself is described by one or more
``IdentificationPhase`` objects. A
phase pairs, for each parameter, a *spatial parameterisation* (how the parameter
is allowed to vary in space) with the *metric*, *objective function* and
*optimiser* used to solve it:

* **Spatial parameterisation** –
  ``SpatialParameterisationHomogeneous``
  treats a parameter as a single value across the whole specimen. Use
  ``SpatialParameterisationKnown`` to fix
  a parameter to its supplied map, or
  ``SpatialParameterisationBasisFunction``
  to let it vary smoothly in space.
* **Metric** – ``MetricSBVF`` implements the
  sensitivity-based virtual fields, evaluated on a virtual mesh whose size you
  provide (e.g. ``np.array([15, 15])``).
* **Objective function** –
  ``VectorFirstResultPassthrough``
  passes the metric residual vector straight to a least-squares optimiser.
* **Optimiser** –
  ``OptimiserLeastSquares`` drives the
  parameter search.

.. code-block:: python

    phases = [
        vfm.IdentificationPhase(
            spatial_parameterisations={
                "elastic_modulus":   [vfm.SpatialParameterisationHomogeneous()],
                "poissons_ratio":    [vfm.SpatialParameterisationHomogeneous()],
                "yield_strength":    [vfm.SpatialParameterisationHomogeneous()],
                "hardening_modulus": [vfm.SpatialParameterisationHomogeneous()],
            },
            metrics=[vfm.MetricSBVF(np.array([15, 15], dtype=np.uint32))],
            objective_function=vfm.VectorFirstResultPassthrough(),
            optimiser=vfm.OptimiserLeastSquares(),
        )
    ]

When several phases are supplied they run in sequence, with the output of one
phase becoming the initial guess for the next — useful, for example, to first
identify the elastic parameters homogeneously and then refine the plastic
parameters.

Finally, combine the model, initial parameters and phases into a single
``IdentificationConfig``:

.. code-block:: python

    identification_config = vfm.IdentificationConfig(
        constitutive_law=constitutive_law,
        parameters=parameters,
        phases=phases,
    )


4. Run the identification
--------------------------

``run_identification`` executes the configured
phases and returns the identified parameters as a mapping of parameter name to
``ConstitutiveParameter``:

.. code-block:: python

    identified_parameters = vfm.run_identification(
        experiment_data, identification_config
    )

    for name, parameter in identified_parameters.items():
        print(f"{name} = {np.nanmean(parameter.map):.4f}")

Each returned ``ConstitutiveParameter`` carries a ``map`` of the identified
values over the specimen grid. For a homogeneous parameterisation every entry
holds the same value, so ``np.nanmean(parameter.map)`` recovers the scalar
result; for a spatially varying parameterisation the ``map`` gives the full
identified field.
