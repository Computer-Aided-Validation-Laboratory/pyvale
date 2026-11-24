Validation
==========

``pyvale.validation`` compares simulation predictions with experimental
observations. A ``ValidCase`` joins three dictionaries by observable tag.
``ValidObservable`` records geometry, component order, units, and any
coordinates and evaluation times. The simulation and experiment dictionaries
contain the corresponding ``ValidArray`` samples. Supply both arrays on the
observable's physical support and in its declared units.

Named axes
----------

``ValidArray`` names each dimension with ``EValidAxis``. Axes, when present, follow
``POINT, COMPONENT, TIME, EPISTEMIC_TRANS, EPISTEMIC_NUIS, ALEATORY``. Omit a
physical axis when its length is one. Simulation and experiment may have different
uncertainty axis lengths. The final aleatory axis is contiguous for ordinary
C-order arrays, so one conditional sample is a cheap slice. ``ValidArray`` preserves
the original array or memory map; the caller controls its lifetime.

``EpistemicOffsets`` gives nonnegative ``lower`` and ``upper`` magnitudes in
observable units. For each conditional aleatory sample, the lower scenario
subtracts ``lower`` from every sample value, and the upper scenario adds
``upper``. Magnitudes may be scalar or vary by named physical point, component,
and time axes. A scenario represents one coherent choice across the observable.
These scenarios can coexist with sampled epistemic axes.

Basic workflow
--------------

.. code-block:: python

   import pyvale.validation as vl

   case = vl.ValidCase(observables=observables, sim=sim_arrays, exp=exp_arrays)
   engine = vl.ValidEngine([vl.MetricAVM(), vl.MetricMAVM()])
   result = engine.run(case)
   avm = result.metrics["temperature"][vl.EMetricType.AVM]
   distances = avm.metric_vals[vl.EMetricOut.DISTANCE]
   maximum, indices = result.find_extreme("temperature", vl.EMetricType.AVM)

Metrics also work independently on one-dimensional NumPy samples through
``calc_avm_1d``, ``calc_mavm_1d``, ``calc_ks_1d``, and ``calc_cvm_1d`` or their
``Metric*`` strategy objects. RMSE and relative error require paired samples of
equal length. The distribution metrics accept unequal sample lengths. Missing
values represented by NaN are omitted by the distribution kernels; a cell with
no finite observations remains NaN in ``ValidResult``. The MAVM strategy defaults
to ``EMAVMDuplicates.INCLUDE``. Pass that option explicitly to
``calc_mavm_1d`` for repeated values.

``ValidResult`` stores a ``MetricField`` for each observable and metric. The field
records its axes and maps each output to a NumPy array. It retains sampled
epistemic and offset scenarios. MAVM exposes ``D_PLUS`` and ``D_MINUS`` outputs.
Use ``plot_sim_exp_cdfs`` to compare distributions from a ``ValidCase`` and
``plot_sim_ecdfs`` for a pointwise envelope across simulation cases. Metric maps
and time plots use observable metadata from the ``ValidResult``.

Conversion and repeated experiments
-----------------------------------

``sample_sim_sweep_for_valid`` samples ``SimData`` runs through SensorSim fields.
Supply a flat list for one uncertainty axis or indexed run pairs for multiple
axes. ``SensorData`` enables spatial averaging, and stored simulation global
variables can supply mean or integral observables. ``assemble_exp_arrays``
converts ``ExpData`` traces to validation arrays. Use a channel map for
multi-channel acquisitions and a steady window to treat selected samples as
aleatory observations. Transient inputs use the observable's evaluation times.

``concatenate_valid_cases`` joins compatible cases along the experimental aleatory
axis, keeping complete transient traces together and recording source case IDs.
``extract_exp_sim_data`` splits SensorSim's current result dictionary into
one ``ExpData`` per experiment and selected sensor component. This is an
interim bridge until ``ExperimentSimulator`` emits ``ExpData`` directly.

Examples and fixture
--------------------

The :doc:`validation examples </examples/examples_valid>` save diagnostic
figures under ``pyvale-output``. The prepared ``plate_validation_2d.npz`` fixture uses
the packaged 2D plate simulations ``case18_out.e`` and ``case18_d_out.e``. The
latter scales both thermal conductivity and thermal expansion by 0.9. Its
SensorSim measurement grids contain 12 thermocouples and 12 strain gauge
locations, with six virtual experiments and five evaluation times. The
regenerator is ``scripts/generate_validation_fixture.py`` and records its
fixed NumPy noise seed in source. The plate quickstart uses the prepared fixture.
The simulation sweep example runs MOOSE to generate its inputs.
