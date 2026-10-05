.. _guide_vfm:

VFM User Guide
==============

``pyvale.vfm`` (VFMAP: Virtual Fields Method with Automated
Parameterisation) provides a toolkit for the inverse identification of
constitutive parameters from full-field strain measurements. The current
implementation is focused on the Virtual Fields Method (VFM)
[PierronGrediac2012]_.

The VFM combines measured strain fields with a postulated constitutive model
and initial parameter estimates to reconstruct stress fields. The static
admissibility of the reconstructed stress is assessed using the principle of
virtual work (PVW), which compares internal and external virtual work. The
identified parameters are those that minimise the residual of the PVW.

This toolkit was originally developed during a PhD project at the University
of Southampton [Hamill2024]_ and was developed further during a UKAEA research
fellowship [HamillEtAl2026]_. Full acknowledgements and references are given in
the :ref:`acknowledgements section <acknowledgements>`.

The code is modular and extensible, allowing users to implement new
constitutive models, identification methods and spatial parameterisations. The
current implementation supports plane-stress elasto-plasticity with an
isotropic von Mises yield surface and linear, Swift, Voce or Ludwik hardening.


Release scope
-------------

This initial release supports homogeneous and slicewise parameterisations,
sequential identification phases, adaptive slice refinement, sensitivity-based
virtual fields (SBVF), force reconstruction error (FRE), and the equilibrium
gap indicator (EGI). EGI is currently provided as a diagnostic metric. The
automated basis-function workflow and combined EGI--FRE cost function described
in the wider VFMAP methodology are not included in this release.

The workflow has four steps:

#. **Process** your raw solver or experiment output into the
   ``ExperimentData`` format using the input data processor.
#. **Load** the processed ``ExperimentData`` back from file.
#. **Set up** the identification: the constitutive model, an initial guess and
   bounds for each parameter, and one or more identification phases.
#. **Run** the identification and inspect the identified parameters.

The complete runnable script for this guide is available in the
:ref:`homogeneous VFM example <sphx_glr_examples_vfm_vfm_ex1_hom.py>`.

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

The current input data processor has some preliminary support for Moose, Ansys and MatchID data.
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

   If the regular-grid arrays have already been assembled, use
   ``AssembledDataConfig`` with ``x.npy``, ``y.npy``, ``strain.npy``,
   ``force.npy`` and ``time.npy``. The repository utility
   ``dev/vfm/prepare_assembled_input.py`` provides a minimal example.

.. important::

   The current VFM data contract uses millimetres for geometry and thickness,
   MPa for stress-like constitutive parameters, newtons for force, and
   dimensionless strain.

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
  a parameter to its supplied map. ``SliceWiseSpatialParameterisation`` assigns
  one value to each region in a ``SupportSlice`` partition.
* **Metric** – ``MetricSBVF`` implements sensitivity-based virtual fields
  [MarekEtAl2017]_. ``SliceWiseForceReconstructionMetric`` evaluates the
  reconstructed force in each slice [SuttonEtAl2008]_.
* **Objective function** –
  ``VectorFirstResultPassthrough``
  passes the metric residual vector straight to a least-squares optimiser.
* **Optimiser** –
  ``OptimiserLeastSquares`` drives the
  general parameter search. ``SliceWiseIndependentLeastSquares`` can solve
  fully independent slices in parallel.

``EquilibriumGapMetric`` evaluates local stress-equilibrium discrepancies
[DevivierEtAl2013]_. Its reported diagnostic is the dimensionless
``Normalised equilibrium gap [-]``.

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
phase becoming the initial guess for the next. For example, yield strength and
hardening modulus can first be identified homogeneously before yield strength
is refined slicewise while hardening remains homogeneous.

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

``run_identification`` executes the configured phases and returns an
``IdentificationResult``:

.. code-block:: python

    result = vfm.run_identification(experiment_data, identification_config)

    for name, parameter_map in result.parameter_maps.items():
        print(f"{name} = {np.nanmean(parameter_map):.4f}")

The result has two parts:

* ``result.parameter_maps`` is a mapping from parameter name to the final
  identified ``(y, x)`` map. For a homogeneous parameterisation every entry
  holds the same value, so ``np.nanmean(parameter_map)`` recovers the scalar
  result; for a spatially varying parameterisation the map gives the full
  identified field.
* ``result.history`` is an ``IdentificationHistory`` with one ``PhaseSnapshot``
  per phase, taken at the end of that phase. Each snapshot holds, per
  constitutive parameter, the spatial parameterisations and their
  degree-of-freedom values:

.. code-block:: python

    for phase_index, phase_snapshot in enumerate(result.history.phases):
        for name, snapshots in phase_snapshot.spatial_parameterisations.items():
            for snapshot in snapshots:
                print(phase_index, name, snapshot.dof_values)

The result can be saved to disk with ``save_to_yaml``. The run bundle contains
``identification_result.yaml`` and ``final_parameter_maps.npz``; when final
stress is available it also contains ``final_identified_stress.npz``. With no
argument a new ``vfm-identification-result_{timestamp}`` directory is created
in the current directory; pass a path to choose your own:

.. code-block:: python

    result.save_to_yaml()          # vfm-identification-result_<timestamp>/
    result.save_to_yaml("my_run")  # my_run/

The YAML manifest records the array file names, metadata and identification
history, including the spatial parameterisations and solve results for each
phase.


Examples
--------

The release includes four self-contained examples using synthetic tensile
data:

#. :ref:`Homogeneous SBVF identification
   <sphx_glr_examples_vfm_vfm_ex1_hom.py>`.
#. :ref:`Parallel slicewise FRE identification and result analysis
   <sphx_glr_examples_vfm_vfm_ex2_slicewise.py>`.
#. :ref:`Slicewise identification with adaptive refinement
   <sphx_glr_examples_vfm_vfm_ex3_slicewise_refinement.py>`.
#. :ref:`Homogeneous SBVF followed by slicewise FRE
   <sphx_glr_examples_vfm_vfm_ex4_hom_slicewise.py>`.


.. _acknowledgements:

Acknowledgements and references
-------------------------------

The VFMAP methodology and original toolkit were developed through the doctoral
research of Robert Hamill at the University of Southampton in collaboration
with the United Kingdom Atomic Energy Authority. Continued development and
integration into PyVale were undertaken during a UKAEA research fellowship
(supported by the Engineering and Physical Sciences Research Council under
grant EP/W006839/1).

.. [PierronGrediac2012] F. Pierron and M. Grédiac, *The Virtual Fields Method:
   Extracting Constitutive Mechanical Parameters from Full-field Deformation
   Measurements*, Springer, 2012.
   `doi:10.1007/978-1-4614-1824-5
   <https://doi.org/10.1007/978-1-4614-1824-5>`__.

.. [Hamill2024] R. J. Hamill, *Development of a Methodology for the Automated
   Spatial Mapping of Heterogeneous Elastoplastic Properties of Welded Joints*,
   PhD thesis, University of Southampton, 2024.

.. [HamillEtAl2026] R. Hamill, A. Harte, A. Marek and F. Pierron,
   "Development of a methodology for the automated spatial mapping of
   heterogeneous elastoplastic properties of welded joints", *Comptes Rendus
   Mécanique*, 354 (2026), pp. 561--592.
   `doi:10.5802/crmeca.371 <https://doi.org/10.5802/crmeca.371>`__.

.. [SuttonEtAl2008] M. A. Sutton, J. H. Yan, S. Avril, F. Pierron and
   S. M. Adeeb, "Identification of Heterogeneous Constitutive Parameters in a
   Welded Specimen: Uniform Stress and Virtual Fields Methods for Material
   Property Estimation", *Experimental Mechanics*, 48 (2008), no. 4,
   pp. 451--464.
   `doi:10.1007/s11340-008-9132-6
   <https://doi.org/10.1007/s11340-008-9132-6>`__.

.. [DevivierEtAl2013] C. Devivier, F. Pierron and M. R. Wisnom, "Impact Damage
   Detection in Composite Plates Using Deflectometry and the Virtual Fields
   Method", *Composites Part A: Applied Science and Manufacturing*, 48 (2013),
   pp. 201--218.
   `doi:10.1016/j.compositesa.2013.01.011
   <https://doi.org/10.1016/j.compositesa.2013.01.011>`__.

.. [MarekEtAl2017] A. Marek, F. M. Davis and F. Pierron, "Sensitivity-Based
   Virtual Fields for the Non-Linear Virtual Fields Method", *Computational
   Mechanics*, 60 (2017), no. 3, pp. 409--431.
   `doi:10.1007/s00466-017-1411-6
   <https://doi.org/10.1007/s00466-017-1411-6>`__.
