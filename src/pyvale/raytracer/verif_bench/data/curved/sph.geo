// ================================================================================
// pyvale: the python validation engine
// License: MIT
// Copyright (C) 2025 The Computer Aided Validation Team
// ================================================================================
//
// Benchmark case 2 - sphere geometry for the quadratic-vs-linear study.
//
// A single source geometry so that every mesh used in the comparison (quadratic
// coarse/fine and linear) provably comes from the same shape. Verified to match
// the reference curved_sphere_3d_verycoarse.vtk curved shell:
//
//   * centre = origin
//   * radius = 0.5
//
// Only the surface is meshed (ray tracing only ever hits the boundary). Both
// element families are available: triangles (TRI3/TRI6) by default, or
// quadrilaterals (QUAD4/QUAD9) with -setnumber Mesh.RecombineAll 1.
//
// Density is controlled from the command line with -setnumber Mesh.MeshSizeMax,
// so this one file yields every mesh. Element order is set with -order.
// ================================================================================

SetFactory("OpenCASCADE");

Mesh.MshFileVersion = 2.2;
Mesh.MeshSizeMin = 0.01;

r = 0.5;     // sphere radius

// Sphere primitive: centre at the origin, radius r. Meshing with -2 meshes only
// its boundary (the sphere surface), which is all a ray ever hits.
Sphere(1) = {0, 0, 0, r};

Physical Surface("sphere_skin") = {1};

// ================================================================================
// Build (from this directory). Each mesh comes from this one geometry. For a
// primitive sphere the discretisation is driven by Mesh.MeshSizeFactor (the
// raw -setnumber Mesh.MeshSizeMax snaps to fixed subdivisions and is unreliable
// here).
//
//   Quadratic, triangular family (TRI6):
//     gmsh -nopopup sph.geo -2 -order 2 -setnumber Mesh.MeshSizeFactor 1.0 \
//          -o curved_sph_tri_coarse.vtk -format vtk
//     gmsh -nopopup sph.geo -2 -order 2 -setnumber Mesh.MeshSizeFactor 0.5 \
//          -o curved_sph_tri_fine.vtk -format vtk
//   Quadratic, quadrilateral family (QUAD9):
//     gmsh -nopopup sph.geo -2 -order 2 -setnumber Mesh.MeshSizeFactor 0.5 \
//          -setnumber Mesh.RecombineAll 1 -o curved_sph_quad_fine.vtk -format vtk
//   Linear, triangular family (TRI3):
//     gmsh -nopopup sph.geo -2 -order 1 -setnumber Mesh.MeshSizeFactor 0.5 \
//          -o curved_sph_lin.vtk -format vtk
// ================================================================================
