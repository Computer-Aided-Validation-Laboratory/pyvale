# 3D rendering examples

These examples use `pyvale.render` to render three-dimensional scenes.
Riley is the default 3D renderer, and the Blender examples use the same
public scene data. Install the Blender extra on Python 3.13 before running
the `ex2*_*` examples.

Riley meshes use `riley.MeshShaderSet` materials. File format and scaling are
configured with camera `output_files`, while `RasterConfig.save_strategy`
selects memory, disk or both. The gallery progresses through basic rendering,
procedural speckle, shader composition, stereo DIC from Exodus, calibration,
combined stereo DIC and infrared, and camera models.

PyVale installs its pinned Riley release from PyPI. Run the integration checks with:

```bash
uv run --no-sync -- python scripts/test_agent.py tests/render tests/riley
```
