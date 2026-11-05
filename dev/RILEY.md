# Riley render integration

PyVale 2026.11.0 targets Riley 2026.11.0. This migration was verified against
Riley source commit `d9b88dbada87cd7a902a9c4b4fc701f76da5695b`.

The release is not yet available from the configured package index. For branch
development, `tool.uv.sources` and `uv.lock` use an editable sibling checkout at
`../riley-raster`. Clone Riley there before running `uv sync`. The package
requirement remains `riley-raster==2026.11.0`; published PyVale packages do not
use uv source overrides. Once Riley is published, remove the source override
and regenerate the lockfile against the index.

## Meshes and shaders

`Mesh3D.shader` holds the backend-owned material. For Riley it must be a
`riley.MeshShaderSet`, with `mono`, `rgb`, `ir` and/or `multifield` sources.
Each source can be a direct shader or a `riley.ShaderPipe`. Individual shaders
are no longer accepted as the mesh material. For example:

```python
mesh.shader = riley.MeshShaderSet(
    mono=riley.TextureShader(uvs=uvs, texture=texture),
)
```

`Mesh3D.normal_type` controls Riley's normal calculation, defaulting to exact
normals. This setting belongs to the mesh rather than its shaders.
Native `riley.Mesh` and `riley.Camera` instances pass through unchanged.
Explicit scene lights remain unsupported; shader modulation and Lambertian
normal shading do not implement PyVale's physical light model.

## Cameras and image output

`Camera.render_format` selects mono, RGB, IR or multifield output;
`render_channels` sets the multifield channel count. Configure file format,
file bits, ADC bits, scaling and packing through `Camera.output_files`, a tuple
of `riley.ImageSaveOpts`. Background values also belong to the camera.
`RasterConfig` controls rendering, threading, buffering and save strategy.

The common camera forwards thin-prism and tilt distortion coefficients,
polynomial distortion maps, coordinate-system selection, subpixel centre maps,
and separable PSF selection. Native Riley cameras expose the complete native
configuration. Existing length units and SciPy rotation conventions remain:
common rotations are converted from XYZ Euler storage to Riley's ZYX storage.

Riley memory output is `(camera, frame, channel, height, width)`. PyVale converts
it to contiguous `(frame, camera, height, width, channel)` storage. This requires
one layout copy; shader and geometry preparation remain owned by Riley. A disk
only render returns `images=None`.

## Validation

```bash
uv run --no-sync python -m pytest tests/render
uv run --no-sync python -m pytest tests/examples
uv run --no-sync python -m pytest tests/riley
```

The Riley parity tests compare complete output trees against the installed
native demos, including image and calibration file bytes. Gold conformance and
rabbit baselines are independent committed references; do not regenerate them
to conceal a migration mismatch. The shader pipeline example exercises mono,
RGB and multifield selection through the common API.

## Migration verification record

The render suite passed with 100 tests and five skips. The complete example
suite passed with 46 tests and five skips. All seven native Riley
demo output-tree comparisons passed byte for byte. Existing conformance and
rabbit gold data were retained unchanged. The HTML documentation build
succeeded with warnings, including missing gallery images and stale references
outside the render module. No performance speedup is claimed.
