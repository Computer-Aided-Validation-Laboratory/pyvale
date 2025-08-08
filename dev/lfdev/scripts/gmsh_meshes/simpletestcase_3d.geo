//==============================================================================
// Gmsh 2D parametric plate mesh
// author: Lloyd Fletcher (scepticalrabbit)
//==============================================================================
// Always set to OpenCASCADE - circles and boolean opts are much easier!
SetFactory("OpenCASCADE");

// Allows gmsh to print to terminal in vscode - easier debugging
General.Terminal = 1;

// View options - not required when
Geometry.PointLabels = 1;
Geometry.CurveLabels = 1;
Geometry.SurfaceLabels = 0;
Geometry.VolumeLabels = 0;

//------------------------------------------------------------------------------
// Variables
file_name = "simpletestcase_3d.msh";

// Geometric variables
block_width = 25e-3;
block_height = block_width+8e-3; // Must be greater than plate width
block_depth= 50e-3;
block_diff = block_height-block_width;

hole_rad = 6e-3;
hole_loc_x = 0.0;
hole_loc_y = block_width/2;
hole_circ = 2*Pi*hole_rad;

// Mesh variables
mesh_ref = 1;
hole_sect_nodes = 7*mesh_ref; // Must be odd
block_rad_nodes = 7*mesh_ref;
block_diff_nodes = 7*mesh_ref; // numbers of nodes along the rectangular extension
block_depth_divs = 5*mesh_ref;

block_edge_nodes = Floor((hole_sect_nodes-1)/2)+1;
elem_size = hole_circ/(4*(hole_sect_nodes-1));
tol = elem_size/4; // Used for bounding box selection tolerance

//------------------------------------------------------------------------------
// Geometry Definition
// Split block into six pieces with a square around the hole to allow spider
// web meshing around the hole

// 4 squares around the hole center
s1 = news;
Rectangle(s1) = {-block_width/2,0.0,0.0,
                block_width/2,block_width/2};
s2 = news;
Rectangle(s2) = {0.0,0.0,0.0,
                block_width/2,block_width/2};

s3 = news;
Rectangle(s3) = {-block_width/2,block_width/2,0.0,
                block_width/2,block_width/2};
s4 = news;
Rectangle(s4) = {0.0,block_width/2,0.0,
                block_width/2,block_width/2};

// Two rectangles above the hole (armour)
s5 = news;
Rectangle(s5) = {-block_width/2,block_width,0.0,
                block_width/2,block_diff};
s6 = news;
Rectangle(s6) = {0.0,block_width,0.0,
                block_width/2,block_diff};

// Merge coincicent edges of the four overlapping squares
BooleanFragments{ Surface{s1}; Delete; }
                { Surface{s2,s3,s4,s5,s6}; Delete; }


// Create the hole surface
c2 = newc; Circle(c2) = {hole_loc_x,hole_loc_y,0.0,hole_rad};
cl2 = newcl; Curve Loop(cl2) = {c2};
s9 = news; Plane Surface(s9) = {cl2};
// Bore out the hole from the quarters of the plate
BooleanDifference{ Surface{s1,s2,s3,s4}; Delete; }{ Surface{s9}; Delete; }

//------------------------------------------------------------------------------
// Transfinite meshing (line element sizes and mapped meshing)
// Line sizing
Transfinite Curve{21,24,28,19} = block_rad_nodes;
Transfinite Curve{22,26,25,31,12,9,27,18,14,17} = block_edge_nodes;
Transfinite Curve{20,23,30,29} = hole_sect_nodes;
Transfinite Curve{15,13,16} = block_diff_nodes;

// Spider web mesh around the 4 quadrants of the hole
Transfinite Surface{s1} = {17,16,15,13};
Recombine Surface{s1};
Transfinite Surface{s2} = {17,16,18,19};
Recombine Surface{s2};
Transfinite Surface{s3} = {13,15,21,7};
Recombine Surface{s3};
Transfinite Surface{s4} = {21,18,19,7};
Recombine Surface{s4};

// Mesh the armour
Transfinite Surface{s5} = {8,7,10,11};
Recombine Surface{s5};
Transfinite Surface{s6} = {7,9,10,12};
Recombine Surface{s6};

//------------------------------------------------------------------------------
// Extrude the surface mesh to 3D
Extrude{0.0,0.0,block_depth}{
    Surface{:}; Layers{block_depth_divs}; Recombine;
}

//------------------------------------------------------------------------------
// Physical surfaces and volumes for export/BCs
//Physical Point("Embedded point") = {p};
//Physical Curve("Embdded curve") = {l};
//Physical Surface("Embedded surface") = {s};
//Physical Volume("Volume") = {1};

Physical Volume("stc-vol") = {Volume{:}};

ps1() = Surface In BoundingBox{
    -block_width/2-tol,0.0-tol,0.0-tol,
    block_width/2+tol,0.0+tol,block_depth+tol};
Physical Surface("bc-base-disp") = {ps1(0),ps1(1)};

ps2() = Surface In BoundingBox{
    -block_width/2-tol,block_height-tol,0.0-tol,
    block_width/2+tol,block_height+tol,block_depth+tol};
Physical Surface("bc-top-heatflux") = {ps2(0),ps2(1)};

ps3() = Surface In BoundingBox{
    -hole_rad-tol,block_width/2-hole_rad-tol,0.0-tol,
    hole_rad+tol,block_width/2+hole_rad+tol,block_depth+tol};
Physical Surface("bc-pipe-htc") = {ps3(0),ps3(1),ps3(2),ps3(3)};

pc1() = Curve In BoundingBox{
    -tol,-tol,-tol,
    +tol,+tol,block_depth+tol};
Physical Curve("bc-mech-axz-dispxy") = {pc1(0)};

pc2() = Curve In BoundingBox{
    -block_width/2-tol,-tol,-tol,
    block_width/2+tol,+tol,+tol};
Physical Curve("bc-mech-axx-dispyz") = {pc2(0),pc2(1)};

//------------------------------------------------------------------------------
// Global meshing
Mesh.Algorithm = 6;
Mesh.Algorithm3D = 10;

num_threads = 4;
General.NumThreads = num_threads;
Mesh.MaxNumThreads1D = num_threads;
Mesh.MaxNumThreads2D = num_threads;
Mesh.MaxNumThreads3D = num_threads;

Mesh.ElementOrder = 2;
Mesh 3;

//------------------------------------------------------------------------------
// Save and exit
//Save Str(file_name);
//Exit;