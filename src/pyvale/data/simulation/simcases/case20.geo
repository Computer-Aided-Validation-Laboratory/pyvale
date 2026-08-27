//==============================================================================
// Gmsh 3D simple test case divertor armour mock-up
// author: Lloyd Fletcher (scepticalrabbit)
//==============================================================================
// Always set to OpenCASCADE - circles and boolean opts are much easier!
SetFactory("OpenCASCADE");

// Allows gmsh to print to terminal in vscode - easier debugging
General.Terminal = 0;

// View options - not required when
Geometry.PointLabels = 1;
Geometry.CurveLabels = 0;
Geometry.SurfaceLabels = 1;
Geometry.VolumeLabels = 0;

//-------------------------------------------------------------------------
//_* MOOSEHERDER VARIABLES - START
file_name = "case20.msh";

// Geometric variables
block_width = 37e-3;
block_leng = 49.5e-3;
block_armour = 12e-3;
block_height_square = 11.5e-3;
block_height_above_pipe = 12.5e-3+block_armour;
block_height_tot = block_height_square+block_height_above_pipe;

// Block half width must be greater than the sum of:
// block_width/2 >= pipe_rad_in+pipe_thick_fillet_rad
fillet_rad = 2e-3;
pipe_rad_in = 6e-3;
pipe_thick = 1.5e-3;
pipe_leng = 100e-3;

// Must be an integer
elem_order = 2;
mesh_ref = 1;
mesh_size = 2e-3/mesh_ref;
num_threads = 4;
//** MOOSEHERDER VARIABLES - END
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Calculated / Fixed Variables
pipe_loc_x = 0.0;
pipe_loc_y = block_height_square;

pipe_rad_out = pipe_rad_in + pipe_thick;
pipe_circ_in = 2*Pi*pipe_rad_in;

tol = mesh_size/4; // Used for bounding box selection tolerance

//------------------------------------------------------------------------------
// Geometry Definition

// Create the block and the outer pipe diam as solid cylinders
v1 = newv;
Box(v1) = {-block_width/2,0,0,
            block_width,block_height_tot,block_leng/2};
v2 = newv;
Box(v2) = {-block_width/2,0.0,-block_leng/2,
            block_width,block_height_tot,block_leng/2};

v3 = newv;
Cylinder(v3) = {pipe_loc_x,pipe_loc_y,block_leng/2,
                0.0,0.0,(pipe_leng/2-block_leng/2),pipe_rad_out,2*Pi};
v4 = newv;
Cylinder(v4) = {pipe_loc_x,pipe_loc_y,-block_leng/2,
                0.0,0.0,-(pipe_leng/2-block_leng/2),pipe_rad_out,2*Pi};

// Need to join the cylinder to the block to create a fillet
BooleanUnion{ Volume{v1}; Delete; }{ Volume{v3}; Delete; }
BooleanUnion{ Volume{v2}; Delete; }{ Volume{v4}; Delete; }

// Grab the curves between the pipe outer edge and the block to fillet
cf1() = Curve In BoundingBox{
    pipe_loc_x-pipe_rad_out-tol,pipe_loc_y-pipe_rad_out-tol,block_leng/2-tol,
    pipe_loc_x+pipe_rad_out+tol,pipe_loc_y+pipe_rad_out+tol,block_leng/2+tol};

cf2() = Curve In BoundingBox{
    pipe_loc_x-pipe_rad_out-tol,pipe_loc_y-pipe_rad_out-tol,-block_leng/2-tol,
    pipe_loc_x+pipe_rad_out+tol,pipe_loc_y+pipe_rad_out+tol,-block_leng/2+tol};

all_vols = Volume{:};
Fillet{all_vols(0)}{cf1(0)}{fillet_rad}
Fillet{all_vols(1)}{cf2(0)}{fillet_rad}

// Join the two halves of the block but maintain the dividing line
all_vols = Volume{:};
BooleanFragments{Volume{all_vols(0)}; Delete;}{Volume{all_vols(1)}; Delete;}

// Create the pipe bore
all_vols = Volume{:};
v5 = newv;
Cylinder(v5) = {pipe_loc_x,pipe_loc_y,-pipe_leng/2,
                0.0,0.0,pipe_leng,pipe_rad_in,2*Pi};
BooleanDifference{Volume{all_vols(0),all_vols(1)}; Delete;}
                {Volume{v5}; Delete;}
all_vols = Volume{:};

// Actual geometry complete - remainder are points for mech BCs
// For mech BCs on the base of the block
pb1 = newp; Point(pb1) = {0,0,0};
pb2 = newp; Point(pb2) = {0,0,block_leng/2};
pb3 = newp; Point(pb3) = {0,0,-block_leng/2};
pb4 = newp; Point(pb4) = {-block_width/2,0,0};
pb5 = newp; Point(pb5) = {block_width/2,0,0};

// For mech BCs on the pipe
pm1 = newp; Point(pm1) = {pipe_loc_x+pipe_rad_out,pipe_loc_y+0.0,pipe_leng/2};
pm2 = newp; Point(pm2) = {pipe_loc_x+0.0,pipe_loc_y+pipe_rad_out,pipe_leng/2};
pm3 = newp; Point(pm3) = {pipe_loc_x-pipe_rad_out,pipe_loc_y+0.0,pipe_leng/2};
pm4 = newp; Point(pm4) = {pipe_loc_x-0.0,pipe_loc_y-pipe_rad_out,pipe_leng/2};

pm5 = newp; Point(pm5) = {pipe_loc_x+pipe_rad_out,pipe_loc_y+0.0,-pipe_leng/2};
pm6 = newp; Point(pm6) = {pipe_loc_x+0.0,pipe_loc_y+pipe_rad_out,-pipe_leng/2};
pm7 = newp; Point(pm7) = {pipe_loc_x-pipe_rad_out,pipe_loc_y+0.0,-pipe_leng/2};
pm8 = newp; Point(pm8) = {pipe_loc_x-0.0,pipe_loc_y-pipe_rad_out,-pipe_leng/2};

BooleanFragments{Volume{:}; Delete;}
{Point{pb1,pb2,pb3,pb4,pb5,pm1,pm2,pm3,pm4,pm5,pm6,pm7,pm8}; Delete;}

//------------------------------------------------------------------------------
// Physical surfaces and volumes for export/BCs

Physical Volume("stc-vol") = {Volume{:}};

// Physical surface for mechanical BC for dispy - like sitting on a flat surface
ps1() = Surface In BoundingBox{
    -block_width/2-tol,0.0-tol,-block_leng/2-tol,
    block_width/2+tol,0.0+tol,block_leng/2+tol};
Physical Surface("bc-base-disp") = {ps1(0),ps1(1)};

// thermal BCs for top surface heat flux and pipe htc
ps2() = Surface In BoundingBox{
    -block_width/2-tol,block_height_tot-tol,-block_leng/2-tol,
    block_width/2+tol,block_height_tot+tol,block_leng/2+tol};
Physical Surface("bc-top-heatflux") = {ps2(0),ps2(1)};

ps3() = Surface In BoundingBox{
    pipe_loc_x-pipe_rad_in-tol,pipe_loc_y-pipe_rad_in-tol,-pipe_leng/2-tol,
    pipe_loc_x+pipe_rad_in+tol,pipe_loc_y+pipe_rad_in+tol,pipe_leng/2+tol};
Physical Surface("bc-pipe-htc") = {ps3(0),ps3(1)};

// Physical points for applying mechanical BCs - Lines don't work in 3D
// Center of the base of the block - lock all DOFs
pp0() = Point In BoundingBox{
    -tol,-tol,-tol,
    +tol,+tol,+tol};
Physical Point("bc-base-c-loc-xyz") = {pp0(0)};

// Base points on the (p)ositive and (n)egative axes
pp1() = Point In BoundingBox{
    block_width/2-tol,-tol,-tol,
    block_width/2+tol,+tol,+tol};
Physical Point("bc-base-px-loc-z") = {pp1(0)};

pp2() = Point In BoundingBox{
    -block_width/2-tol,-tol,-tol,
    -block_width/2+tol,+tol,+tol};
Physical Point("bc-base-nx-loc-z") = {pp2(0)};

pp3() = Point In BoundingBox{
    -tol,-tol,block_leng/2-tol,
    +tol,+tol,block_leng/2+tol};
Physical Point("bc-base-pz-loc-x") = {pp3(0)};

pp4() = Point In BoundingBox{
    -tol,-tol,-block_leng/2-tol,
    +tol,+tol,-block_leng/2+tol};
Physical Point("bc-base-nz-loc-x") = {pp4(0)};

// Pipe end in the (p)ositive z direction, (n)orth,(s)outh,(e)ast,(w)est
loc_x = 0.0;
loc_y = pipe_rad_out;
pp5() = Point In BoundingBox{
    pipe_loc_x+loc_x-tol,pipe_loc_y+loc_y-tol,pipe_leng/2-tol,
    pipe_loc_x+loc_x+tol,pipe_loc_y+loc_y+tol,pipe_leng/2+tol};
Physical Point("bc-pipe-pzn-loc-x") = {pp5(0)};

loc_x = 0.0;
loc_y = -pipe_rad_out;
pp6() = Point In BoundingBox{
    pipe_loc_x+loc_x-tol,pipe_loc_y+loc_y-tol,pipe_leng/2-tol,
    pipe_loc_x+loc_x+tol,pipe_loc_y+loc_y+tol,pipe_leng/2+tol};
Physical Point("bc-pipe-pzs-loc-x") = {pp6(0)};

loc_x = pipe_rad_out;
loc_y = 0.0;
pp7() = Point In BoundingBox{
    pipe_loc_x+loc_x-tol,pipe_loc_y+loc_y-tol,pipe_leng/2-tol,
    pipe_loc_x+loc_x+tol,pipe_loc_y+loc_y+tol,pipe_leng/2+tol};
Physical Point("bc-pipe-pze-loc-y") = {pp7(0)};

loc_x = -pipe_rad_out;
loc_y = 0.0;
pp8() = Point In BoundingBox{
    pipe_loc_x+loc_x-tol,pipe_loc_y+loc_y-tol,pipe_leng/2-tol,
    pipe_loc_x+loc_x+tol,pipe_loc_y+loc_y+tol,pipe_leng/2+tol};
Physical Point("bc-pipe-pzw-loc-y") = {pp8(0)};

// Pipe end in the (n)egative z direction, (n)orth,(s)outh,(e)ast,(w)est
loc_x = 0.0;
loc_y = pipe_rad_out;
pp9() = Point In BoundingBox{
    pipe_loc_x+loc_x-tol,pipe_loc_y+loc_y-tol,-pipe_leng/2-tol,
    pipe_loc_x+loc_x+tol,pipe_loc_y+loc_y+tol,-pipe_leng/2+tol};
Physical Point("bc-pipe-nzn-loc-x") = {pp9(0)};

loc_x = 0.0;
loc_y = -pipe_rad_out;
pp10() = Point In BoundingBox{
    pipe_loc_x+loc_x-tol,pipe_loc_y+loc_y-tol,-pipe_leng/2-tol,
    pipe_loc_x+loc_x+tol,pipe_loc_y+loc_y+tol,-pipe_leng/2+tol};
Physical Point("bc-pipe-nzs-loc-x") = {pp10(0)};

loc_x = pipe_rad_out;
loc_y = 0.0;
pp11() = Point In BoundingBox{
    pipe_loc_x+loc_x-tol,pipe_loc_y+loc_y-tol,-pipe_leng/2-tol,
    pipe_loc_x+loc_x+tol,pipe_loc_y+loc_y+tol,-pipe_leng/2+tol};
Physical Point("bc-pipe-nze-loc-y") = {pp11(0)};

loc_x = -pipe_rad_out;
loc_y = 0.0;
pp12() = Point In BoundingBox{
    pipe_loc_x+loc_x-tol,pipe_loc_y+loc_y-tol,-pipe_leng/2-tol,
    pipe_loc_x+loc_x+tol,pipe_loc_y+loc_y+tol,-pipe_leng/2+tol};
Physical Point("bc-pipe-nzw-loc-y") = {pp12(0)};

//------------------------------------------------------------------------------
// Mesh Sizing
MeshSize{ PointsOf{ Volume{:}; } } = mesh_size;

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
