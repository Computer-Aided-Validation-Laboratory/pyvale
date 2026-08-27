# Mesh Convention

Pyvale uses one finite-element mesh convention across DataIO, SensorSim, and
the render APIs. The common implementation is in
`pyvale.dataio.meshconv`.

## Required representation

- `coords` has one node per row. Three columns are `[x, y, z]`; planar meshes
  use `z = 0`.
- Connectivity has one element per row and contains zero-based node indices.
- Every connectivity index refers to a row in `coords`.
- Element connectivity is row-major. Legacy transposed and one-based tables
  are detected and normalised by `enforce_mesh_convention`.

## Mesh topology

`SimData.mesh_type` explicitly records the topology as `EMeshType.VOL` or
`EMeshType.SURF`. `SimData.__post_init__` infers it whenever both coordinates
and connectivity are supplied. The Exodus and CSV/array loaders call
`refresh_mesh_type()` after loading their fields, so their returned objects
carry the same value.

Use `is_volume_mesh(sim_data)` where code needs to distinguish a volume mesh
from a surface mesh. It resolves the ambiguous four- and eight-node tables by
their signed cell volume (TET4 versus QUAD4, and HEX8 versus QUAD8). A
`SimData` containing both surface and volume connectivity tables is rejected:
it cannot have one unambiguous `EMeshType`.

## Winding and handedness

For a planar face, list corner nodes counter-clockwise when viewing its
outward/visible side. The right-hand rule then gives the face normal: curl the
fingers from the first edge to the second, and the thumb points outward.

For a volume element, corner order must give a positive signed volume. Surface
extraction orients each exposed face outward from its parent element.

`check_mesh_convention(mesh)` reports zero-based indexing, row-major layout,
valid indices, counter-clockwise winding, and right-handed geometry.
`enforce_mesh_convention(mesh)` normalises supported input and should be
idempotent.

## Element node order

The first nodes are always the corner nodes. Higher-order edge nodes follow
the perimeter in the same direction as the corners.

| Element | Corner order | Extra-node order |
| --- | --- | --- |
| TRI3 | `0, 1, 2` | — |
| TRI6 | `0, 1, 2` | edges `01, 12, 20` |
| QUAD4 | `0, 1, 2, 3` | — |
| QUAD8 | `0, 1, 2, 3` | edges `01, 12, 23, 30` |
| QUAD9 | QUAD8 order | QUAD8 edges, then centre |
| TET4 | `0, 1, 2, 3` | — |
| TET10 | TET4 corners | canonical tetrahedral edge nodes |
| HEX8 | `0, 1, 2, 3, 4, 5, 6, 7` | — |
| HEX20 | HEX8 corners | canonical hexahedral edge nodes |
| HEX27 | HEX20 order | face centres, then volume centre |

TET14 is packaged as a cube fixture but is not yet supported by the common
convention checker or surface extractor.

## Verification

The cube fixtures provide regression coverage for TET4, TET10, HEX8, HEX20,
and HEX27:

```bash
python -m pytest tests/dataio/meshtools_test.py -k cube
```

The suite verifies raw legacy layout, successful enforcement, idempotence, and
the local outward orientation of extracted closed surfaces.
