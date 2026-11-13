// A 20 x 10 x 5 mm solid for the introductory 3D meshing example.
SetFactory("OpenCASCADE");
DefineConstant[
  mesh_size_mm = {2.5, Min 0.1, Max 10, Name "Mesh/Size (mm)"}
];

Box(1) = {0, 0, 0, 20, 10, 5};
Physical Volume("solid") = {1};
tol = 1e-6;
fixed() = Surface In BoundingBox{-tol, -tol, -tol, tol, 10+tol, 5+tol};
loaded() = Surface In BoundingBox{20-tol, -tol, -tol, 20+tol, 10+tol, 5+tol};
Physical Surface("fixed") = {fixed()};
Physical Surface("loaded") = {loaded()};
Mesh.MeshSizeMin = mesh_size_mm;
Mesh.MeshSizeMax = mesh_size_mm;
