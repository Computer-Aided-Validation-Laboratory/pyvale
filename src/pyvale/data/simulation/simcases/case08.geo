//==============================================================================
// Gmsh 3D parametric rectangular plate with hole mesh
// author: Lloyd Fletcher (scepticalrabbit)
//==============================================================================
// Always set to OpenCASCADE - circles and boolean opts are much easier!
SetFactory("OpenCASCADE");

// Allows gmsh to print to terminal in vscode - easier debugging
General.Terminal = 0;

// View options - not required when
Geometry.PointLabels = 0;
Geometry.CurveLabels = 0;
Geometry.SurfaceLabels = 0;
Geometry.VolumeLabels = 0;

//------------------------------------------------------------------------------
// Variables
file_name = "case08.msh";

// Geometric variables
plate_width = 100e-3;
plate_height = plate_width+50e-3; // Must be greater than plate width
plate_diff = plate_height-plate_width;
plate_thick = 5e-3;

hole_rad = 25e-3/2;
hole_loc_x = plate_width/2;
hole_loc_y = plate_height/2;
hole_circ = 2*Pi*hole_rad;

// Mesh variables
hole_sect_nodes = 9; // Must be odd
plate_rad_nodes = 9;
plate_diff_nodes = 5; // numbers of nodes along the rectangular extension
plate_thick_divs = 2; // using quadratic elements so 2 through thickness should be ok

plate_edge_nodes = Floor((hole_sect_nodes-1)/2)+1;
elem_size = hole_circ/(4*(hole_sect_nodes-1));
tol = elem_size; // Used for bounding box selection tolerance

//------------------------------------------------------------------------------
// Geometry Definition

// Split plate into eight pieces with a square around the hole to allow spider
// web meshing around the hole
s1 = news;
Rectangle(s1) = {0.0,0.0,0.0,
                plate_width/2,plate_diff/2};
s2 = news;
Rectangle(s2) = {plate_width/2,0.0,0.0,
                plate_width/2,plate_diff/2};

s3 = news;
Rectangle(s3) = {0.0,plate_diff/2,0.0,
                plate_width/2,plate_width/2};
s4 = news;
Rectangle(s4) = {plate_width/2,plate_diff/2,0.0,
                plate_width/2,plate_width/2};

s5 = news;
Rectangle(s5) = {0.0,plate_width/2+plate_diff/2,0.0,
                plate_width/2,plate_width/2};
s6 = news;
Rectangle(s6) = {plate_width/2,plate_width/2+plate_diff/2,0.0,
                plate_width/2,plate_width/2};

s7 = news;
Rectangle(s7) = {0.0,plate_height-plate_diff/2,0.0,
                plate_width/2,plate_diff/2};
s8 = news;
Rectangle(s8) = {plate_width/2,plate_height-plate_diff/2,0.0,
                plate_width/2,plate_diff/2};

// Merge coincicent edges of the four overlapping squares
BooleanFragments{ Surface{s1}; Delete; }
                { Surface{s2,s3,s4,s5,s6,s7,s8}; Delete; }


// Create the hole surface
c2 = newc; Circle(c2) = {hole_loc_x,hole_loc_y,0.0,hole_rad};
cl2 = newcl; Curve Loop(cl2) = {c2};
s9 = news; Plane Surface(s9) = {cl2};
// Bore out the hole from the quarters of the plate
BooleanDifference{ Surface{s3,s4,s5,s6}; Delete; }{ Surface{s9}; Delete; }


//------------------------------------------------------------------------------
// Transfinite meshing (line element sizes and mapped meshing)
Transfinite Curve{31,24,26,28} = plate_rad_nodes;
Transfinite Curve{1,5,3,7,23,29,30,34,14,17,19,22} = plate_edge_nodes;
Transfinite Curve{32,33,25,27} = hole_sect_nodes;
Transfinite Curve{4,2,6,20,18,21} = plate_diff_nodes;

// NOTE: recombine surface turns default triangles into quads

Transfinite Surface{s1} = {1,2,3,4};
Recombine Surface{s1};
Transfinite Surface{s2} = {2,5,6,3};
Recombine Surface{s2};

Transfinite Surface{s3} = {17,18,3,16};
Recombine Surface{s3};
Transfinite Surface{s4} = {18,19,20,3};
Recombine Surface{s4};
Transfinite Surface{s5} = {17,21,10,16};
Recombine Surface{s5};
Transfinite Surface{s6} = {19,21,10,20};
Recombine Surface{s6};

Transfinite Surface{s7} = {11,10,13,14};
Recombine Surface{s7};
Transfinite Surface{s8} = {10,12,15,13};
Recombine Surface{s8};

//------------------------------------------------------------------------------
// Extrude the surface mesh to 3D
Extrude{0.0,0.0,plate_thick}{
    Surface{:}; Layers{plate_thick_divs}; Recombine;
}

//------------------------------------------------------------------------------
// Physical surfaces and volumes for export/BCs
Physical Volume("plate") = {Volume{:}};

ps1() = Surface In BoundingBox{
    0.0-tol,0.0-tol,0.0-tol,
    plate_width+tol,0.0+tol,plate_thick+tol};
Physical Surface("bc-base") = {ps1(0),ps1(1)};

ps2() = Surface In BoundingBox{
    0.0-tol,plate_height-tol,0.0-tol,
    plate_width+tol,plate_height+tol,plate_thick+tol};
Physical Surface("bc-top") = {ps2(0),ps2(1)};

//------------------------------------------------------------------------------
// Global meshing
Mesh.ElementOrder = 2;
Mesh 3;

//------------------------------------------------------------------------------
// Save and exit
Save Str(file_name);
Exit;