.. _guide_meshing:

Geometry and meshing with Gmsh
==================================

Build a mesh with material regions and boundary names that a solver can use.
PyVale uses Gmsh through ``GmshRunner``; geometry and mesh choices remain
explicit in your geometry script.

Before you start
--------------------

A geometry defines shapes; a mesh discretises them. Gmsh entities are points,
curves, surfaces and volumes. Physical groups give entities meaningful names
for material regions and boundary conditions. A boundary name in a solver
input must match the exported mesh. Choose a consistent unit system: Gmsh
coordinates do not carry an automatic metres/mm conversion.

1. Start with a 2D mesh
---------------------------

The existing case17 plate-with-a-hole examples show both a ``.geo`` file and a
Python Gmsh script. They use millimetres. The ``.geo`` version uses mapped
quadrilaterals; the Python alternative uses triangles.

.. code-block:: python

   from pathlib import Path
   import pyvale.data as dataset
   import pyvale.mooseherder as mh

   geometry = dataset.sim_case_gmsh_file_path(17)
   if geometry is None:
       raise FileNotFoundError("Case17 geometry is unavailable")
   runner = mh.GmshRunner(geo_file=geometry, mesh_dims=2)
   result = runner.run(
       input_params={"hole_sect_nodes": 7, "plate_rad_nodes": 7},
       output_dir=mh.get_next_run_path(
           Path.cwd() / "pyvale-output/guide_mesh"
       ),
   )
   print(result.output_file)


For a ``.geo`` file, parameters intended for ``input_params`` overrides use
``DefineConstant``. Plain assignments in a legacy script can replace values
passed on the command line, so inspect the source before assuming a name is an
exposed control.

For Python geometry, supply ``py_script_file`` to ``GmshRunner``. The script
receives ``--config`` and a JSON file containing ``input_params`` and
``output_path``; it owns Gmsh initialisation, geometry, mesh generation and
writing the requested output. The existing case17 Python example demonstrates
this contract.

2. Move to a volume mesh
----------------------------

Run the new box example to see a 20 × 10 × 5 mm volume with linear tetrahedra,
a ``solid`` physical group and ``fixed`` / ``loaded`` end faces. Its companion
:download:`gmsh_box.geo <../../../src/pyvale/examples/mooseherder/gmsh_box.geo>`
defines the geometry and an overridable mesh size.

``mesh_dims=3`` requests a volume mesh when the geometry script has not already
generated one. Keep three choices separate:

* **Dimension:** a surface mesh or a volume discretisation.
* **Element order:** the interpolation used within each element.
* **Size/refinement:** the resolution of the discretisation.

Choose them for the physics and geometry you need to resolve. Raising element
order and reducing element size have different effects on accuracy and cost.

3. Mesh the three-material monoblock
----------------------------------------

Use the case16 example for the divertor monoblock. It resolves the authoritative
``case16.geo`` and matching ``case16.i`` from PyVale's simulation library.
Geometry is in metres. The regions are tungsten armour (``armour-w``), copper
interlayer (``interlayer-cu``) and CuCrZr tube (``pipe-cucrzr``); the coolant
bore is a void.

The script meshes a mapped cross-section and extrudes it through the depth,
using quadratic hexahedra. Its original refinement, element order, groups and
boundary names are retained. The example only meshes: it does not solve the
heat-transfer or mechanics model.

Analyse the result
------------------

Inspect dimensions, material-group contents, boundary locations and element
quality. For the monoblock, use a section view to relate the mesh to the three
material regions and coolant bore.

At material interfaces, inspect the connectivity or the model's explicit
interface treatment. Surfaces that look coincident need not share nodes. Check
that physical-group names and element conventions match the chosen input deck
before passing the mesh to a solver.

Assess refinement using a physical response relevant to the experiment, rather
than the mesh illustration alone. Preserve geometry and meshing parameters
alongside results. The default MSH export version is 2.2 for MOOSE
compatibility; a file extension selects the export format.

Continue with these examples and guides:

:doc:`2D meshing </examples/mooseherder/ex03b_gmsh_python>` · :doc:`3D box </examples/mooseherder/ex03d_gmsh_box3d>` · :doc:`Case16 monoblock </examples/mooseherder/ex03e_gmsh_monoblock>` · `Gmsh tutorials <https://gmsh.info/doc/texinfo/gmsh.html#Tutorial>`_

:doc:`Run the solver <guide_mooseherder>` · :doc:`Inspect fields <guide_simdata>`
