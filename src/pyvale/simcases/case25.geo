//==============================================================================
// Gmsh 3D simple block imaging test case
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
file_name = "case25.msh";

// Geometric variables
plate_height = 15e-3;
plate_width = 10e-3;
plate_thick = 5e-3;

mesh_size = 5e-3/mesh_ref;

// Must be an integer
plate_thick_layers = 1;
elem_order = 1;
mesh_ref = 1;
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

Extrude{0.0,0.0,plate_thick}{
    Surface{:}; Layers{plate_thick_layers}; // Recombine;
}

//------------------------------------------------------------------------------
// Physical Volumes and Surfaces
Physical Volume("plate-vol") = {Volume{:}};

Physical Surface("plate-surf-vis-front") = {6};
Physical Surface("plate-surf-vis-back") = {1};
Physical Surface("bc-top-disp") = {4};
Physical Surface("bc-base-disp") = {2};

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
