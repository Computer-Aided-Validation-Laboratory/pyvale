//==============================================================================
// Gmsh 3D Monoblock
// author: Lloyd Fletcher (scepticalrabbit)
//==============================================================================
SetFactory("OpenCASCADE");
General.Terminal = 0;

// View options: turns on/off which pieces of geometry we show
Geometry.Points = 0;
Geometry.Curves = 0;
Geometry.Surfaces = 0;
Geometry.Volumes = 0;

// View options: turns labels on and off for different types of geometry
Geometry.PointLabels = 0;
Geometry.CurveLabels = 0;
Geometry.SurfaceLabels = 0;
Geometry.VolumeLabels = 0;

//------------------------------------------------------------------------------
// Variable Definitions

//------------------------------------------------------------------------------
//_* MOOSEHERDER VARIABLES - START
file_name = "case16.msh";
num_threads = 8;

// Specified Geometry variables
pipe_rad_int = 6e-3;
pipe_thick = 1.5e-3;

interlayer_thick = 2e-3;

monoblock_depth = 12e-3;
monoblock_side = 3e-3;
monoblock_arm_height = 8e-3;

// Specified Mesh variables
base_divs = 1;
mesh_ref = 1; //  Set to 1 for testing, Must be an integer greater than 0
elem_order = 2;

//_* MOOSEHERDER VARIABLES - END
//------------------------------------------------------------------------------

// Calculated Geometry Variables
pipe_rad_ext = pipe_rad_int+pipe_thick;

interlayer_rad_int = pipe_rad_ext;
interlayer_rad_ext = interlayer_rad_int+interlayer_thick;

monoblock_width = 2*interlayer_rad_ext + 2*monoblock_side;
monoblock_height = monoblock_width + monoblock_arm_height;

pipe_cent_x = 0.0;
pipe_cent_y = interlayer_rad_ext + monoblock_side;

// Calculated Mesh Variables
pipe_sect_nodes = Round(mesh_ref*5); // Must be odd
pipe_rad_nodes = Round(mesh_ref*4);
interlayer_rad_nodes = Round(mesh_ref*4);
monoblock_side_nodes = Round(mesh_ref*5);
monoblock_arm_nodes = Round(mesh_ref*5);
monoblock_depth_nodes = Round(mesh_ref*3); // NOTE: this is half the depth!
monoblock_width_nodes = Floor((pipe_sect_nodes-1)/2)+1;


// This is a more reasonable mesh refinement for the monoblock but solve time
// is much longer
// pipe_sect_nodes = Round(mesh_ref*11); // Must be odd
// pipe_rad_nodes = Round(mesh_ref*7);
// interlayer_rad_nodes = Round(mesh_ref*7);
// monoblock_side_nodes = Round(mesh_ref*9);
// monoblock_arm_nodes = Round(mesh_ref*11);
// monoblock_depth_nodes = Round(mesh_ref*5);
// monoblock_width_nodes = Floor((pipe_sect_nodes-1)/2)+1;


// Calculate approx element size by dividing the circumference
elem_size = 2*Pi*pipe_rad_int/(4*(pipe_sect_nodes-1));
tol = elem_size/4; // Used for selection tolerance of bounding boxes

//------------------------------------------------------------------------------
// Geometry Definition
s1 = news;
Rectangle(s1) =
    {-monoblock_width/2,0.0,0.0,
    monoblock_width/2,monoblock_width/2};

s2 = news;
Rectangle(s2) =
    {-monoblock_width/2,monoblock_width/2,0.0,
    monoblock_width/2,monoblock_width/2};

s3 = news;
Rectangle(s3) =
    {0.0,monoblock_width/2,0.0,
    monoblock_width/2,monoblock_width/2};

s4 = news;
Rectangle(s4) =
    {0.0,0.0,0.0,
    monoblock_width/2,monoblock_width/2};

sa1 = news;
Rectangle(sa1) =
    {-monoblock_width/2,monoblock_width,0.0,
    monoblock_width/2,monoblock_arm_height};

sa2 = news;
Rectangle(sa2) =
    {0.0,monoblock_width,0.0,
    monoblock_width/2,monoblock_arm_height};

BooleanFragments{ Surface{s1}; Delete; }{ Surface{s2,s3,s4,sa1,sa2}; Delete; }

pipe_block_surfs() = Surface In BoundingBox{
        -monoblock_width/2-tol,-tol,-tol,
        monoblock_width+tol,monoblock_width+tol,tol};

c2 = newc; Circle(c2) = {pipe_cent_x,pipe_cent_y,0.0,interlayer_rad_ext};
cl2 = newcl; Curve Loop(cl2) = {c2};
s6 = news; Plane Surface(s6) = {cl2};

c3 = newc; Circle(c3) = {pipe_cent_x,pipe_cent_y,0.0,pipe_rad_ext};
cl3 = newcl; Curve Loop(cl3) = {c3};
s7 = news; Plane Surface(s7) = {cl3};

c4 = newc; Circle(c4) = {pipe_cent_x,pipe_cent_y,0.0,pipe_rad_int};
cl4 = newcl; Curve Loop(cl4) = {c4};
s8 = news; Plane Surface(s8) = {cl4};

BooleanDifference{Surface{pipe_block_surfs(),s6,s7}; Delete;}
                 { Surface{s8}; Delete;}

//------------------------------------------------------------------------------
// Transfinite Line/Curve Meshing

cm1() = Curve In BoundingBox{
    pipe_cent_x-interlayer_rad_ext-tol,pipe_cent_y-interlayer_rad_ext-tol,-tol,
    pipe_cent_x+interlayer_rad_ext+tol,pipe_cent_y+interlayer_rad_ext+tol,+tol};

For ss In {0:#cm1()-1}
    Transfinite Curve(cm1(ss)) = pipe_sect_nodes;
EndFor

cm1() = Curve In BoundingBox{
    pipe_cent_x+pipe_rad_int-tol,pipe_cent_y-tol,-tol,
    pipe_cent_x+pipe_rad_ext+tol,pipe_cent_y+tol,+tol};
Transfinite Curve(cm1(0)) = pipe_rad_nodes;

cm2() = Curve In BoundingBox{
    pipe_cent_x-pipe_rad_ext-tol,pipe_cent_y-tol,-tol,
    pipe_cent_x-pipe_rad_int+tol,pipe_cent_y+tol,+tol};
Transfinite Curve(cm2(0)) = pipe_rad_nodes;

cm3() = Curve In BoundingBox{
    pipe_cent_x-tol,pipe_cent_y-pipe_rad_ext-tol,-tol,
    pipe_cent_x+tol,pipe_cent_y-pipe_rad_int+tol,+tol};
Transfinite Curve(cm3(0)) = pipe_rad_nodes;

cm4() = Curve In BoundingBox{
    pipe_cent_x-tol,pipe_cent_y+pipe_rad_int-tol,-tol,
    pipe_cent_x+tol,pipe_cent_y+pipe_rad_ext+tol,+tol};
Transfinite Curve(cm4(0)) = pipe_rad_nodes;


cm1() = Curve In BoundingBox{
    pipe_cent_x+interlayer_rad_int-tol,pipe_cent_y-tol,-tol,
    pipe_cent_x+interlayer_rad_ext+tol,pipe_cent_y+tol,+tol};
Transfinite Curve(cm1(0)) = interlayer_rad_nodes;

cm2() = Curve In BoundingBox{
    pipe_cent_x-interlayer_rad_ext-tol,pipe_cent_y-tol,-tol,
    pipe_cent_x-interlayer_rad_int+tol,pipe_cent_y+tol,+tol};
Transfinite Curve(cm2(0)) = interlayer_rad_nodes;

cm3() = Curve In BoundingBox{
    pipe_cent_x-tol,pipe_cent_y-interlayer_rad_ext-tol,-tol,
    pipe_cent_x+tol,pipe_cent_y-interlayer_rad_int+tol,+tol};
Transfinite Curve(cm3(0)) = interlayer_rad_nodes;

cm4() = Curve In BoundingBox{
    pipe_cent_x-tol,pipe_cent_y+interlayer_rad_int-tol,-tol,
    pipe_cent_x+tol,pipe_cent_y+interlayer_rad_ext+tol,+tol};
Transfinite Curve(cm4(0)) = interlayer_rad_nodes;


cm1() = Curve In BoundingBox{
    pipe_cent_x+interlayer_rad_ext-tol,pipe_cent_y-tol,-tol,
    pipe_cent_x+monoblock_width/2+tol,pipe_cent_y+tol,+tol};
Transfinite Curve(cm1(0)) = monoblock_side_nodes;

cm2() = Curve In BoundingBox{
    pipe_cent_x-monoblock_width/2-tol,pipe_cent_y-tol,-tol,
    pipe_cent_x-interlayer_rad_ext+tol,pipe_cent_y+tol,+tol};
Transfinite Curve(cm2(0)) = monoblock_side_nodes;

cm3() = Curve In BoundingBox{
    pipe_cent_x-tol,pipe_cent_y-monoblock_width/2-tol,-tol,
    pipe_cent_x+tol,pipe_cent_y-interlayer_rad_ext+tol,+tol};
Transfinite Curve(cm3(0)) = monoblock_side_nodes;

cm4() = Curve In BoundingBox{
    pipe_cent_x-tol,pipe_cent_y+interlayer_rad_ext-tol,-tol,
    pipe_cent_x+tol,pipe_cent_y+monoblock_width/2+tol,+tol};
Transfinite Curve(cm4(0)) = monoblock_side_nodes;


cm1() = Curve In BoundingBox{
    -monoblock_width/2-tol,monoblock_width-tol,-tol,
    monoblock_width/2+tol,monoblock_width+monoblock_arm_height+tol,+tol};

For ss In {0:#cm1()-1}

    Transfinite Curve(cm1(ss)) = monoblock_arm_nodes;
EndFor



cm1() = Curve In BoundingBox{
    -monoblock_width/2-tol,-tol,-tol,
    -monoblock_width/2+tol,monoblock_width+tol,+tol};

For ss In {0:#cm1()-1}
    Transfinite Curve(cm1(ss)) = monoblock_width_nodes;
EndFor


cm2() = Curve In BoundingBox{
    monoblock_width/2-tol,-tol,-tol,
    monoblock_width/2+tol,monoblock_width+tol,+tol};

For ss In {0:#cm2()-1}
    Transfinite Curve(cm2(ss)) = monoblock_width_nodes;
EndFor


// Mesh top, bottom and armour horizontal lines
cm3() = Curve In BoundingBox{
    -monoblock_width/2-tol,-tol,-tol,
    monoblock_width/2+tol,+tol,+tol};

For ss In {0:#cm3()-1}
    Transfinite Curve(cm3(ss)) = monoblock_width_nodes;
EndFor


cm3() = Curve In BoundingBox{
    -monoblock_width/2-tol,monoblock_width-tol,-tol,
    monoblock_width/2+tol,monoblock_width+tol,+tol};

For ss In {0:#cm3()-1}
    Transfinite Curve(cm3(ss)) = monoblock_width_nodes;
EndFor


cm4() = Curve In BoundingBox{
    -monoblock_width/2-tol,monoblock_width+monoblock_arm_height+-tol,-tol,
    monoblock_width/2+tol,monoblock_width+monoblock_arm_height+tol,+tol};

For ss In {0:#cm4()-1}
    Transfinite Curve(cm4(ss)) = monoblock_width_nodes;
EndFor


//------------------------------------------------------------------------------
// Transfinite Surface Meshing

// Mesh the pipe and interlayer
sm1() = Surface In BoundingBox{
    pipe_cent_x-interlayer_rad_ext-tol,pipe_cent_y-interlayer_rad_ext-tol,-tol,
    pipe_cent_x+interlayer_rad_ext+tol,pipe_cent_y+interlayer_rad_ext+tol,+tol};

For ss In {0:#sm1()-1}
    Transfinite Surface{sm1(ss)};
    Recombine Surface{sm1(ss)};
EndFor

// Mesh the armour on top of the block
sm1() = Surface In BoundingBox{
    -monoblock_width/2-tol,monoblock_width-tol,-tol,
    monoblock_width/2+tol,monoblock_width+monoblock_arm_height+tol,+tol};

For ss In {0:#sm1()-1}
    Transfinite Surface{sm1(ss)};
    Recombine Surface{sm1(ss)};
EndFor

// Mesh the block around the interlayer
Transfinite Surface{22} = {15,16,17,13};
Recombine Surface{22};

Transfinite Surface{27} = {15,24,5,13};
Recombine Surface{27};

Transfinite Surface{30} = {24,5,28,27};
Recombine Surface{30};

Transfinite Surface{31} = {27,28,17,16};
Recombine Surface{31};

//------------------------------------------------------------------------------
// Extrude the surface mesh to 3D
Extrude{0.0,0.0,monoblock_depth/2}{
    Surface{:}; Layers{monoblock_depth_nodes}; Recombine;
}

es1() = Surface In BoundingBox{
    -monoblock_width/2-tol,0.0-tol,monoblock_depth/2-tol,
    monoblock_width/2+tol,monoblock_height+tol,monoblock_depth/2+tol};

Extrude{0.0,0.0,monoblock_depth/2}{
    Surface{es1(0),es1(1),es1(2),es1(3),es1(4),es1(5),es1(6),es1(7),es1(8),es1(9),es1(10),es1(11),es1(12),es1(13)}; Layers{monoblock_depth_nodes}; Recombine;
}

//------------------------------------------------------------------------------
// Physical Surfaces for Loads and Boundary Condition

ps0() = Surface In BoundingBox{
    -monoblock_width/2-tol,0.0-tol,0.0-tol,
    monoblock_width/2+tol,0.0+tol,monoblock_depth+tol};
Physical Surface("bc-base-disp") = {ps0(0),ps0(1),ps0(2),ps0(3)};

ps1() = Surface In BoundingBox{
    -monoblock_width/2-tol,monoblock_height-tol,0.0-tol,
    monoblock_width/2+tol,monoblock_height+tol,monoblock_depth+tol};
Physical Surface("bc-top-heatflux") = {ps1(0),ps1(1),ps1(2),ps1(3)};

ps2() = Surface In BoundingBox{
    -pipe_rad_int-tol,monoblock_width/2-pipe_rad_int-tol,0.0-tol,
    pipe_rad_int+tol,monoblock_width/2+pipe_rad_int+tol,monoblock_depth+tol};
Physical Surface("bc-pipe-heattransf") = {ps2(0),ps2(1),ps2(2),ps2(3),ps2(4),ps2(5),ps2(6),ps2(7)};


// Physical points for applying mechanical BCs
// Center of the base of the block - lock all DOFs
pp0() = Point In BoundingBox{
    -tol,-tol,monoblock_depth/2-tol,
    +tol,+tol,monoblock_depth/2+tol};
Physical Point("bc-c-point-xyz-mech") = {pp0(0)};

// Left and right on the base center line
pp1() = Point In BoundingBox{
    -monoblock_width/2-tol,-tol,monoblock_depth/2-tol,
    -monoblock_width/2+tol,+tol,monoblock_depth/2+tol};
Physical Point("bc-l-point-yz-mech") = {pp1(0)};

pp2() = Point In BoundingBox{
    monoblock_width/2-tol,-tol,monoblock_depth/2-tol,
    monoblock_width/2+tol,+tol,monoblock_depth/2+tol};
Physical Point("bc-r-point-yz-mech") = {pp2(0)};

// Front and back on the base center line
pp3() = Point In BoundingBox{
    -tol,-tol,monoblock_depth-tol,
    +tol,+tol,monoblock_depth+tol};
Physical Point("bc-f-point-xy-mech") = {pp3(0)};

pp4() = Point In BoundingBox{
    -tol,-tol,-tol,
    +tol,+tol,+tol};
Physical Point("bc-b-point-xy-mech") = {pp4(0)};

//------------------------------------------------------------------------------
// Physical Volumes for Material Defs
Physical Volume("pipe-cucrzr") = {5,19,6,20,9,23,14,28};
Physical Volume("interlayer-cu") = {4,18,7,21,10,24,13,27};
Physical Volume("armour-w") = {3,17,8,22,25,11,12,26,1,15,2,16};

//------------------------------------------------------------------------------
// Global Mesh controls
Mesh.Algorithm = 6;
Mesh.Algorithm3D = 10;

General.NumThreads = num_threads;
Mesh.MaxNumThreads1D = num_threads;
Mesh.MaxNumThreads2D = num_threads;
Mesh.MaxNumThreads3D = num_threads;

Mesh.SecondOrderIncomplete = 1;
Mesh.ElementOrder = elem_order;

Mesh 3;

//------------------------------------------------------------------------------
// Save and exit
Save Str(file_name);
Exit;


