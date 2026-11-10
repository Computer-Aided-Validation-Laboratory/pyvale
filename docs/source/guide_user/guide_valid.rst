Validation
==========

``pyvale.valid`` compares simulation and experimental observables on a shared
physical domain. Import it as ``import pyvale.valid as val``. A ``ValCase`` has
one ``ValDomain`` and separate simulation and experimental ``ValData`` objects.
The domain records point coordinates, component order, evaluation times,
measurement operator, field units, optional time and coordinate units, and frame.
Arrays supplied directly must already use that physical support and those
units. Assembly validates shapes and names; it does not register coordinates,
convert units, interpolate, or pool runs.

Named axes
----------

``ValArray`` names each dimension with ``EValAxis``. Axes, when present, follow
``POINT, COMPONENT, TIME, EPISTEMIC_TRANS, EPISTEMIC_NUIS, ALEATORY``. An absent
physical axis has length one. Simulation and experiment may have different
uncertainty axis lengths. The final aleatory axis is contiguous for ordinary
C-order arrays, so one conditional sample is a cheap slice. ``val_from_arrays``
preserves the original array or memory map; the caller controls its lifetime.

``EpistemicOffsets`` gives nonnegative ``lower`` and ``upper`` magnitudes in
observable units. For each conditional aleatory sample, the lower scenario
subtracts ``lower`` from every sample value, and the upper scenario adds
``upper``. Magnitudes may be scalar or vary by named physical point, component,
and time axes. A scenario represents one coherent choice over the domain. These
scenarios can coexist with sampled epistemic axes. They do not imply that every
pointwise envelope combination is a jointly attainable physical field.

Basic workflow
--------------

.. code-block:: python

   import pyvale.valid as val

   case = val.ValCase(domain, sim_data, exp_data)
   engine = val.ValidationEngine([val.MetricAVM(), val.MetricMAVM()])
   result = engine.run(case)
   avm = result.get("temperature", val.EMetricType.AVM)
   maximum, indices = result.find_extreme("temperature", val.EMetricType.AVM)

Metrics also work independently on one-dimensional NumPy samples through
``calc_avm_1d``, ``calc_mavm_1d``, ``calc_ks_1d``, and ``calc_cvm_1d`` or their
``Metric*`` strategy objects. RMSE and relative error require paired samples of
equal length. The distribution metrics accept unequal sample lengths. Missing
values represented by NaN are omitted by the distribution kernels; a cell with
no finite observations remains NaN in ``ValResult``. The MAVM strategy defaults
to the robust tie-aware mode. ``calc_mavm_1d`` retains its original mode default
for direct callers, so select ``EMAVMMode.ROBUST`` explicitly for repeated data.

``ValResult`` keeps a labelled array for each observable and metric. It retains
sampled epistemic and offset scenarios rather than reducing them. The MAVM
field also stores directional ``d_plus`` and ``d_minus``. CDF diagnostics can
be reconstructed from the ``ValCase`` at a selected point and scenario with
``plot_val_cdfs``. ``plot_val_cdf_envelope`` computes a pointwise envelope over
the supplied *coherent* scenario CDFs; crossing CDFs need not correspond to a
single limiting scenario. Metric maps and time plots use domain metadata kept
in ``ValResult``.

Conversion and repeated experiments
-----------------------------------

``val_from_sim`` samples a stream of indexed ``SimData`` runs through existing
SensorSim ``IField`` objects. Each run index explicitly identifies its
rectangular uncertainty coordinates. ``SensorData`` enables SensorSim's
spatial averaging; precomputed MOOSE global variables may supply global mean or
integral observables. ``val_from_exp`` selects a single registered ``ExpData``
acquisition for each observable. A steady window becomes aleatory observations;
transient inputs require the domain's exact evaluation times. Separate
acquisitions normally produce separate ``ValCase`` objects.

``combine_val_cases`` joins compatible cases along the experimental aleatory
axis, keeping complete transient traces together and recording source case IDs.
``conv_exp_sim_to_data`` splits SensorSim's current result dictionary into
one ``ExpData`` per experiment and selected sensor component. This is an
interim bridge until ``ExperimentSimulator`` emits ``ExpData`` directly.

Examples and fixture
--------------------

The six scripts in ``src/pyvale/examples/valid`` each save diagnostic figures
under ``pyvale-output``. The prepared ``plate_validation_2d.npz`` fixture uses
the packaged 2D plate simulations ``case18_out.e`` and ``case18_d_out.e``. The
latter scales both thermal conductivity and thermal expansion by 0.9. Its
SensorSim measurement grids contain 12 thermocouples and 12 strain gauge
locations, with six virtual experiments and five evaluation times. The
regenerator is ``scripts/generate_validation_fixture.py`` and records its
fixed NumPy noise seed in source. Ordinary examples and tests use the prepared
fixture; they do not execute MOOSE.
