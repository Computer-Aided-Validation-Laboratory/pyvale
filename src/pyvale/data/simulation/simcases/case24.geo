//==============================================================================
// Gmsh 2D plate imaging test case
// author: Lloyd Fletcher (scepticalrabbit)
//==============================================================================
// Always set to OpenCASCADE - circles and boolean opts are much easier!
SetFactory("OpenCASCADE");

// Allows gmsh to print to terminal in vscode - easier debugging
General.Terminal = 1;

// View options - not required when
Geometry.PointLabels = 1;
Geometry.CurveLabels = 1;
Geometry.SurfaceLabels = 1;
Geometry.VolumeLabels = 0;

//-------------------------------------------------------------------------
//_* MOOSEHERDER VARIABLES - START
file_name = "case24.msh";

// Geometric variables
plate_height = 50e-3;
plate_width = 100e-3;

// Must be an integer
elem_order = 1;
mesh_ref = 1;
mesh_size = 5e-3/mesh_ref;
num_threads = 4;
//** MOOSEHERDER VARIABLES - END
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Calculated / Fixed Variables
tol = mesh_size/4; // Used for bounding box selection tolerance

//------------------------------------------------------------------------------
// Geometry Definition
s1 = news;
Rectangle(s1) =
    {0.0,0.0,0.0,
      plate_width,plate_height};

//------------------------------------------------------------------------------
// Mesh Sizing
MeshSize{ PointsOf{ Surface{:}; } } = mesh_size;
Transfinite Surface{Surface{:}};
//Recombine Surface{Surface{:}};

// Extrude{0.0,0.0,plate_thick}{
//     Surface{:}; Layers{plate_thick_layers}; Recombine;
// }

//------------------------------------------------------------------------------
// Physical Volumes and Surfaces
Physical Surface("plate-surf") = {Surface{:}};

Physical Curve("bc-top") = {3};
Physical Curve("bc-base") = {1};
Physical Curve("bc-left") = {4};
Physical Curve("bc-right") = {2};

//------------------------------------------------------------------------------
// Global meshing
Mesh.Algorithm = 6;
Mesh.Algorithm3D = 10;

General.NumThreads = num_threads;
Mesh.MaxNumThreads1D = num_threads;
Mesh.MaxNumThreads2D = num_threads;
Mesh.MaxNumThreads3D = num_threads;

Mesh.ElementOrder = elem_order;
Mesh 2;

//------------------------------------------------------------------------------
// Save and exit
Save Str(file_name);
Exit;
