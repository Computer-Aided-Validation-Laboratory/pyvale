//==============================================================================
// Gmsh 3D cylinder imaging test case
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
file_name = "case21.msh";

// Geometric variables
cyl_height = 25e-3;
cyl_diam = 25e-3;

// Must be an integer
elem_order = 1;
mesh_ref = 4;
mesh_size = 2.5e-3/mesh_ref;
num_threads = 4;
//** MOOSEHERDER VARIABLES - END
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Calculated / Fixed Variables
cyl_rad = cyl_diam/2;
tol = mesh_size/4; // Used for bounding box selection tolerance

//------------------------------------------------------------------------------
// Geometry Definition
v1 = newv;
Cylinder(v1) = {0.0,0.0,0.0, // center location of first face
                0.0,cyl_height,0.0, // vector defining direction
                cyl_rad,2*Pi};

//------------------------------------------------------------------------------
// Mesh Sizing
MeshSize{ PointsOf{ Volume{:}; } } = mesh_size;

//------------------------------------------------------------------------------
// Physical Volumes and Surfaces
Physical Volume("cyl-vol") = {Volume{:}};

// Physical surface for mechanical BC for disp_y
Physical Surface("cyl-surf-vis") = {1};
Physical Surface("bc-top-disp") = {2};
Physical Surface("bc-base-disp") = {3};

//------------------------------------------------------------------------------
// Global meshing
Mesh.Algorithm = 6;
Mesh.Algorithm3D = 10;

General.NumThreads = num_threads;
Mesh.MaxNumThreads1D = num_threads;
Mesh.MaxNumThreads2D = num_threads;
Mesh.MaxNumThreads3D = num_threads;

Mesh.ElementOrder = elem_order;
Mesh 3;

//------------------------------------------------------------------------------
// Save and exit
Save Str(file_name);
Exit;
