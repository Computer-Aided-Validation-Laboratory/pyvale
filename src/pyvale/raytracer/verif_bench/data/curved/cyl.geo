// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================
//
// Benchmark case 2 - cylinder geometry for the quadratic-vs-linear study.
//
// A single source geometry so that every mesh used in the comparison (quadratic
// coarse/fine and linear) provably comes from the same shape. Verified to match
// the reference curved_cyl_3d_verycoarse.vtk:
//
//   * axis   = y
//   * radius = 0.5
//   * y in [0, 1]
//   * full circular cross-section in the x-z plane (curved skin satisfies
//     sqrt(x^2 + z^2) = 0.5).
//
// Only the boundary is meshed (ray tracing only ever hits the boundary): the
// curved lateral surface plus the two flat end caps.
//
// Density is controlled from the command line with -setnumber Mesh.MeshSizeMax,
// so this one file yields every mesh. Mesh.RecombineAll turns the
// quadrilateral-friendly lateral strip into QUAD9/QUAD4 cells (with TRI6/TRI3
// where the algorithm cannot recombine), matching the reference element mix.
// ================================================================================

SetFactory("OpenCASCADE");

Mesh.MshFileVersion = 2.2;
Mesh.MeshSizeMin = 0.02;

// Mesh.RecombineAll is set from the command line (-setnumber Mesh.RecombineAll
// 1) to produce an all-quadrilateral mesh. Left unset the surface mesher returns
// triangles instead, giving the triangular element family (TRI3 / TRI6). Both
// variants come from this same geometry.

r = 0.5;     // cylinder radius
h = 1.0;     // cylinder height in y

// Disk at y = 0: centre plus four points on the circle, built as four arcs so
// the full circle closes without the degenerate two-arc construction. Point
// sizes are 1.0 so the -setnumber Mesh.MeshSizeMax option governs the density.
Point(1) = { 0.0, 0.0,  0.0, 1.0};   // centre
Point(2) = { r,   0.0,  0.0, 1.0};   // on +x
Point(3) = { 0.0, 0.0,  r,   1.0};   // on +z
Point(4) = {-r,   0.0,  0.0, 1.0};   // on -x
Point(5) = { 0.0, 0.0, -r,   1.0};   // on -z

Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};        // bottom cap

// Extrude the disk along +y: creates the top cap and the lateral surface.
Extrude {0, h, 0} { Surface{1}; }

// OpenCASCADE tags the extruded volume as 1. The lateral surface is tag 3, the
// two end caps are 1 (bottom) and 2 (top).
Physical Volume("cylinder")     = {1};
Physical Surface("cap_in")      = {1};
Physical Surface("cap_out")     = {2};
Physical Surface("curved_skin") = {3};

// ================================================================================
// Build (from this directory). Each mesh comes from this one geometry, so the
// linear and quadratic meshes provably describe the same half-space boundary.
//
//   Quadratic, quadrilateral family (QUAD9):
//     gmsh -nopopup cyl.geo -2 -order 2 -setnumber Mesh.MeshSizeMax 0.28 \
//          -setnumber Mesh.RecombineAll 1 -o curved_cyl_quad_coarse.vtk -format vtk
//     gmsh -nopopup cyl.geo -2 -order 2 -setnumber Mesh.MeshSizeMax 0.16 \
//          -setnumber Mesh.RecombineAll 1 -o curved_cyl_quad_fine.vtk -format vtk
//   Quadratic, triangular family (TRI6):
//     gmsh -nopopup cyl.geo -2 -order 2 -setnumber Mesh.MeshSizeMax 0.16 \
//          -o curved_cyl_tri.vtk -format vtk
//   Linear, triangular family (TRI3):
//     gmsh -nopopup cyl.geo -2 -order 1 -setnumber Mesh.MeshSizeMax 0.16 \
//          -o curved_cyl_lin.vtk -format vtk
// ================================================================================
