# 3D rendering examples

These examples use `pyvale.render` to render three-dimensional scenes.
Riley is the default 3D renderer, and the Blender examples use the same
public scene data. Install the Blender extra on Python 3.13 before running
the `ex2_*` examples.

Riley meshes use `riley.MeshShaderSet` materials. File format and scaling are
configured with camera `output_files`, while `RasterConfig.save_strategy`
selects memory, disk or both. `ex1i_riley_shader_pipeline.py` demonstrates
shader composition and mono, RGB and multifield output.

For this branch's unreleased Riley dependency, keep the Riley checkout beside
PyVale at `../riley-raster`; see `dev/RILEY.md`.
