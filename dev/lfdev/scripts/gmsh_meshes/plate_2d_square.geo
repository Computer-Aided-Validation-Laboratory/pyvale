//==============================================================================
// Gmsh 2D parametric plate mesh
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
file_name = "plate_2d_square.msh";

// Geometric variables
plate_width = 100e-3;
plate_height = plate_width;
hole_rad = 25e-3/2;
hole_loc_x = plate_width/2;
hole_loc_y = plate_height/2;
hole_circ = 2*Pi*hole_rad;

// Mesh variables
hole_sect_nodes = 9; // Must be odd
plate_rad_nodes = 9;
plate_edge_nodes = Floor((hole_sect_nodes-1)/2)+1;
elem_size = hole_circ/(4*(hole_sect_nodes-1));
tol = elem_size; // Used for bounding box selection tolerance

//------------------------------------------------------------------------------
// Geometry Definition

// Split plate into four squares to allow spider web meshing around the hole
s1 = news;
Rectangle(s1) = {0.0,0.0,0.0,plate_width/2,plate_height/2};
s2 = news;
Rectangle(s2) = {plate_width/2,0.0,0.0,plate_width/2,plate_height/2};
s3 = news;
Rectangle(s3) = {plate_width/2,plate_height/2,0.0,plate_width/2,plate_height/2};
s4 = news;
Rectangle(s4) = {0.0,plate_height/2,0.0,plate_width/2,plate_height/2};
// Merge coincicent edges of the four overlapping squares
BooleanFragments{ Surface{s1}; Delete; }{ Surface{s2,s3,s4}; Delete; }

// Create the hole surface
c2 = newc; Circle(c2) = {hole_loc_x,hole_loc_y,0.0,hole_rad};
cl2 = newcl; Curve Loop(cl2) = {c2};
s5 = news; Plane Surface(s5) = {cl2};
// Bore out the hole from the quarters of the plate
BooleanDifference{ Surface{s1,s2,s3,s4}; Delete; }{ Surface{s5}; Delete; }

//------------------------------------------------------------------------------
// Transfinite meshing (line element sizes and mapped meshing)

Transfinite Curve{2,4,7,11} = plate_rad_nodes;
Transfinite Curve{1,5,14,15,12,13,8,9} = plate_edge_nodes;
Transfinite Curve{3,16,6,10} = hole_sect_nodes;

// NOTE: recombine surface turns default triangles into quads
Transfinite Surface{s1} = {1,3,4,5};
Recombine Surface{s1};
Transfinite Surface{s2} = {4,5,7,6};
Recombine Surface{s2};
Transfinite Surface{s3} = {6,7,9,10};
Recombine Surface{s3};
Transfinite Surface{s4} = {3,9,10,1};
Recombine Surface{s4};

//------------------------------------------------------------------------------
// Physical lines and surfaces for export/BCs
Physical Surface("plate") = {Surface{:}};

pc1() = Curve In BoundingBox{
    0.0-tol,0.0-tol,0.0-tol,
    plate_width+tol,0.0+tol,0.0+tol};
Physical Curve("bc-base") = {pc1(0),pc1(1)};

pc2() = Curve In BoundingBox{
    0.0-tol,plate_height-tol,0.0-tol,
    plate_width+tol,plate_height+tol,0.0+tol};
Physical Curve("bc-top") = {pc2(0),pc2(1)};

//------------------------------------------------------------------------------
// Global meshing
Mesh.ElementOrder = 2;
Mesh 2;

//------------------------------------------------------------------------------
// Save and exit
Save Str(file_name);
//Exit;
