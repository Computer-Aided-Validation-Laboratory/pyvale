Workflow orchestration
======================

``pyvale.workflow`` executes reproducible parameter studies by composing
ordinary Python functions. It keeps domain work in the render, DIC,
Mooseherder, and SensorSim modules rather than creating a class per study.

``FullFactorial``, ``ExplicitCases``, and ``RandomSampling`` create stable
case assignments. ``WorkflowRunner`` executes independent cases in serial or
with process workers. ``threads_per_case`` is passed explicitly by workflow
functions to native backends, so users can avoid CPU oversubscription.

The default ``HYBRID`` storage writes compact case manifests and returns
in-memory results. Use ``WorkflowGatherer`` for a later disk pass and use
pixel-coordinate selectors with ``SignalExtraction`` for strain metrics.

The workflow examples include plate-hole mechanics, Riley-to-DIC,
Blender-to-DIC, and an explicit MOOSE-to-Riley-to-DIC case subset.

See :doc:`../examples/workflow/index` for executable compositions.
