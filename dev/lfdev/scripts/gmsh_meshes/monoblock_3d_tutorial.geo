//==============================================================================
// Gmsh 3D Monoblock Tutorial
// author: Lloyd Fletcher (scepticalrabbit)
//==============================================================================

// Helpful hints: gmsh error messages can be cryptic and most of the time I
// found the error was a missing semi-colon - so check this first!

// Always use the OpenCASCADE geometry - it greatly simplifies boolean ops and
// allows more complex surfaces and volumes to be created directly.
SetFactory("OpenCASCADE");

// Prints gmsh output to the terminal in vscode - useful for debugging
// You can also see all messages in the gmsh console itself. Shortcut: ctrl+L
// or through the menu tools->message console
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

// NOTE: the visibility menu is very useful! shortcut keys: ctrl+shift+v or from
// the toolbar Tools->visibility. From here you can turn on and off entities
// that are visible including geometry/mesh/physical.

/*
You can also use blocks comments like this to block out parts of the script to
see what each part of the script does.
*/

//------------------------------------------------------------------------------
// Variable Definitions

// Variables in gmsh are easily defined using the assignment operator '='. Types
// do not need to specified.
file_name = "monoblock_3d_tutorial.msh";

// Geometry variables
pipe_rad_int = 6e-3;
pipe_thick = 1.5e-3;
pipe_rad_ext = pipe_rad_int+pipe_thick;

interlayer_thick = 2e-3;
interlayer_rad_int = pipe_rad_ext;
interlayer_rad_ext = interlayer_rad_int+interlayer_thick;

monoblock_depth = 12e-3;
monoblock_side = 3e-3;
monoblock_arm_height = 8e-3;
monoblock_width = 2*interlayer_rad_ext + 2*monoblock_side;
monoblock_height = monoblock_width + monoblock_arm_height;

pipe_cent_x = 0.0;
pipe_cent_y = interlayer_rad_ext + monoblock_side;

// Mesh variables
mesh_ref = 1;
// Mesh variables generally specify the number of nodes alond a given line/curve
pipe_sect_nodes = Round(mesh_ref*11); // Must be odd
pipe_rad_nodes = Round(mesh_ref*7);
interlayer_rad_nodes = Round(mesh_ref*7);
monoblock_side_nodes = Round(mesh_ref*9);
monoblock_arm_nodes = Round(mesh_ref*11);
monoblock_depth_nodes = Round(mesh_ref*9);
monoblock_width_nodes = Floor((pipe_sect_nodes-1)/2)+1;

// Calculate approx element size by dividing the circumference
elem_size = 2*Pi*pipe_rad_int/(4*(pipe_sect_nodes-1));
tol = elem_size/4; // Used for selection tolerance of bounding boxes

// Print statements can be useful for checking/debugging calculated variables.
// Use ctrl+L to open the message console where the print output is.
Printf("=====================================================================");
Printf("VARIABLES");
Printf("monoblock_width_nodes = %g",monoblock_width_nodes);
Printf("=====================================================================");

//------------------------------------------------------------------------------
// Geometry Definition

// Create a pattern of squares around the pipe hole to allow for mapped meshing
// with a spider web like mesh. Add rectangles to the top as the armour.

s1 = news; // Gets the next available surface tag and stores in s1
Rectangle(s1) = // Use the surface tag to create a new rectangle
    {-monoblock_width/2,0.0,0.0, // Location of the bottom corner [x,y,z]
    monoblock_width/2,monoblock_width/2}; // Dimesions of the rectangles [W,H]

// Repeat the process to create the remaining 3 squares
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

// Create two rectangles at the top as the armour.
sa1 = news;
Rectangle(sa1) =
    {-monoblock_width/2,monoblock_width,0.0,
    monoblock_width/2,monoblock_arm_height};

sa2 = news;
Rectangle(sa2) =
    {0.0,monoblock_width,0.0,
    monoblock_width/2,monoblock_arm_height};

// Add all the blocks together and merge coincident lines
// Note that 'Fragments' will keep all entities i.e. it doesn't create holes but
// it can create inclusions and merge coincident geometry.
BooleanFragments{ Surface{s1}; Delete; }{ Surface{s2,s3,s4,sa1,sa2}; Delete; }

// Get the surfaces that will need to be hollowed out later for the pipe.
// Here we use a bounding box to select all 'Surfaces' in the bounding box. Note
// that this also works for Points, Curves and Volumes and physical entities.
// We store the surfaces in the bounding box in a list called 'pipe_block_surfs'
// as denoted by the round brackets. Note that square brackets can also be used
// to indicate a list.
pipe_block_surfs() = Surface In BoundingBox{
        -monoblock_width/2-tol,-tol,-tol,               // Bottom left corner [x,y,z]
        monoblock_width+tol,monoblock_width+tol,tol};   // Top right corner [x,y,z]
// NOTE: It is very important that a tolerance is added to each dimension! This
// should be negative for the bottom left corner and positive for the top right
// corner. If the tolerance is omitted in any dimension then no entities are
// selected.

// Generate circles for the pipe and the interlayer
c2 = newc; Circle(c2) = {pipe_cent_x,pipe_cent_y,0.0,interlayer_rad_ext};
cl2 = newcl; Curve Loop(cl2) = {c2};
s6 = news; Plane Surface(s6) = {cl2};

c3 = newc; Circle(c3) = {pipe_cent_x,pipe_cent_y,0.0,pipe_rad_ext};
cl3 = newcl; Curve Loop(cl3) = {c3};
s7 = news; Plane Surface(s7) = {cl3};

c4 = newc; Circle(c4) = {pipe_cent_x,pipe_cent_y,0.0,pipe_rad_int};
cl4 = newcl; Curve Loop(cl4) = {c4};
s8 = news; Plane Surface(s8) = {cl4};

// Remove the pipe hole and interlayer from the block.
// Here we use boolean difference to make sure there is a hole in the pipe.
BooleanDifference{Surface{pipe_block_surfs(),s6,s7}; Delete;}
                 { Surface{s8}; Delete;}

//------------------------------------------------------------------------------
// Transfinite Line/Curve Meshing

// Transfinite meshing for lines/curves is called 'line element sizing' in other
// FE packages. Transfinite surface meshing is often called 'mapped' meshing in
// other FE packages.

// Here we select all circular lines for the pipe and interlayer using a
// bounding box. Note that 'cm1' is a list as indicated by the braces. Note that
// square braces can also be used []. Lists are zero indexed.
cm1() = Curve In BoundingBox{
    pipe_cent_x-interlayer_rad_ext-tol,pipe_cent_y-interlayer_rad_ext-tol,-tol,
    pipe_cent_x+interlayer_rad_ext+tol,pipe_cent_y+interlayer_rad_ext+tol,+tol};

// Print statements are useful for seeing what entities have been selected with
// a bounding box. Note that the '#' operator gets the length of a list. We also
// use a loop here to go through the list of selected curves.
Printf("=====================================================================");
For ss In {0:#cm1()-1}
    Printf("Meshing curve: %g",cm1(ss));
    Transfinite Curve(cm1(ss)) = pipe_sect_nodes;
EndFor
Printf("=====================================================================");

// Mesh the radial lines for the pipe
// Here we select one line at a time going around the pipe and getting the lines
// that extend radially in a cross pattern around the circular geometry.
cm1() = Curve In BoundingBox{
    pipe_cent_x+pipe_rad_int-tol,pipe_cent_y-tol,-tol,
    pipe_cent_x+pipe_rad_ext+tol,pipe_cent_y+tol,+tol};
// We only expect to select a single line so we use the cm1(0) notation to get
// the first selected curve.
Transfinite Curve(cm1(0)) = pipe_rad_nodes;

// Now we repeat the process for all the radial lines.
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

// Mesh the radial lines for the interlayer - as above but for the interlayer
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

// Mesh the lines of the tungsten block joining the circular radial lines
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

// Mesh the armour vertical lines first
cm1() = Curve In BoundingBox{
    -monoblock_width/2-tol,monoblock_width-tol,-tol,
    monoblock_width/2+tol,monoblock_width+monoblock_arm_height+tol,+tol};

Printf("=====================================================================");
Printf("Meshing the vertical lines of the armour block");
For ss In {0:#cm1()-1}
    Printf("Meshing curve: %g",cm1(ss));
    Transfinite Curve(cm1(ss)) = monoblock_arm_nodes;
EndFor
Printf("=====================================================================");

// Mesh the side lines of the monoblock
cm1() = Curve In BoundingBox{
    -monoblock_width/2-tol,-tol,-tol,
    -monoblock_width/2+tol,monoblock_width+tol,+tol};

Printf("=====================================================================");
Printf("Meshing left side of the monoblock lines");
For ss In {0:#cm1()-1}
    Printf("Meshing curve: %g",cm1(ss));
    Transfinite Curve(cm1(ss)) = monoblock_width_nodes;
EndFor
Printf("=====================================================================");


cm2() = Curve In BoundingBox{
    monoblock_width/2-tol,-tol,-tol,
    monoblock_width/2+tol,monoblock_width+tol,+tol};

Printf("=====================================================================");
Printf("Meshing right side of the monoblock lines");
For ss In {0:#cm2()-1}
    Printf("Meshing curve: %g",cm2(ss));
    Transfinite Curve(cm2(ss)) = monoblock_width_nodes;
EndFor
Printf("=====================================================================");

// Mesh top, bottom and armour horizontal lines
cm3() = Curve In BoundingBox{
    -monoblock_width/2-tol,-tol,-tol,
    monoblock_width/2+tol,+tol,+tol};

Printf("=====================================================================");
Printf("Meshing bottom monoblock lines");
For ss In {0:#cm3()-1}
    Printf("Meshing curve: %g",cm3(ss));
    Transfinite Curve(cm3(ss)) = monoblock_width_nodes;
EndFor
Printf("=====================================================================");


cm3() = Curve In BoundingBox{
    -monoblock_width/2-tol,monoblock_width-tol,-tol,
    monoblock_width/2+tol,monoblock_width+tol,+tol};

Printf("=====================================================================");
Printf("Meshing top monoblock lines");
For ss In {0:#cm3()-1}
    Printf("Meshing curve: %g",cm3+monoblock_arm_height+(ss));
    Transfinite Curve(cm3(ss)) = monoblock_width_nodes;
EndFor
Printf("=====================================================================");


cm4() = Curve In BoundingBox{
    -monoblock_width/2-tol,monoblock_width+monoblock_arm_height+-tol,-tol,
    monoblock_width/2+tol,monoblock_width+monoblock_arm_height+tol,+tol};

Printf("=====================================================================");
Printf("Meshing top armour lines");
For ss In {0:#cm4()-1}
    Printf("Meshing curve: %g",cm4(ss));
    Transfinite Curve(cm4(ss)) = monoblock_width_nodes;
EndFor
Printf("=====================================================================");

//------------------------------------------------------------------------------
// Transfinite Surface Meshing

// Transfinite surface meshing creates mapped meshes for various surfaces. We
// will create a nice surface mesh on the front face of the block and then
// extrude it to 3D later.

// Mesh the pipe and interlayer
sm1() = Surface In BoundingBox{
    pipe_cent_x-interlayer_rad_ext-tol,pipe_cent_y-interlayer_rad_ext-tol,-tol,
    pipe_cent_x+interlayer_rad_ext+tol,pipe_cent_y+interlayer_rad_ext+tol,+tol};

Printf("=====================================================================");
Printf("Surface meshing pipe and interlayer");
For ss In {0:#sm1()-1}
    Printf("Meshing surface: %g",sm1(ss));
    Transfinite Surface{sm1(ss)};
    Recombine Surface{sm1(ss)};
EndFor
Printf("=====================================================================");

// Mesh the armour on top of the block
sm1() = Surface In BoundingBox{
    -monoblock_width/2-tol,monoblock_width-tol,-tol,
    monoblock_width/2+tol,monoblock_width+monoblock_arm_height+tol,+tol};

Printf("=====================================================================");
Printf("Surface meshing the armour");
For ss In {0:#sm1()-1}
    Printf("Meshing surface: %g",sm1(ss));
    Transfinite Surface{sm1(ss)};
    Recombine Surface{sm1(ss)};
EndFor
Printf("=====================================================================");

// Mesh the block around the interlayer
// This is where the spider web mesh is created. We select a surface quadrant
// and set the four nodes along the radial lines as the map. Note that the
// corner node must be ignored. Try turning on surface and points numbers to
// see the pattern in gmsh.
Transfinite Surface{22} = {15,16,17,13};
Recombine Surface{22};

Transfinite Surface{27} = {15,24,5,13};
Recombine Surface{27};

Transfinite Surface{30} = {24,5,28,27};
Recombine Surface{30};

Transfinite Surface{31} = {27,28,17,16};
Recombine Surface{31};

// If you run this command you will be able to see the spider web mesh on the
// front face. Remember to comment this out once you want to create a 3d mesh.
// Mesh 2;

//------------------------------------------------------------------------------
// Extrude the surface mesh to 3D
// The first three parameters are the vector that defines the extrusion
// direction. The second set of brackets determines the surfaces to extrude - in
// this case 'all', how many layers to create, and the recombine turns the
// triangles into quads. Note that we already have a quad mesh on the front face
// so we must use combine here.
Extrude{0.0,0.0,monoblock_depth}{
    Surface{:}; Layers{monoblock_depth_nodes}; Recombine;
}

//------------------------------------------------------------------------------
// Physical Surfaces for Loads and Boundary Condition
Physical Surface("bc-top-heatflux") = {36,41};
Physical Surface("bc-pipe-heattransf") = {54,59,83,67};

//------------------------------------------------------------------------------
// Physical Volumes for Material Defs

// Now we define some physical volumes which will allow to easily assign
// material properties for the different pieces of the block. Try turning on
// numbering for volumes to see which numbers are which.
Physical Volume("pipe-cucrzr") = {6,9,5,14};
Physical Volume("interlayer-cu") = {4,7,10,13};
Physical Volume("armour-w") = {1,2,8,11,3,12};

//------------------------------------------------------------------------------
// Global Mesh controls

// Here we select the meshing algorithms - these are good defaults for the
// monoblock and option 10 for 3D is parallelisable.
Mesh.Algorithm = 6;
Mesh.Algorithm3D = 10;

// Here we can set the number of threads gmsh will use - note that it only does
// paralellisation with specific meshing algorithms. Note that parallelisation
// only happens within a particular entity so each volume is still meshed
// sequentially - there are ways to mesh volumes in parallel using the python
// API and the multi-processing library. I tested this with over 1 million nodes
// and it still mostly uses a single core no matter the setting below.
num_threads = 4;
General.NumThreads = num_threads;
Mesh.MaxNumThreads1D = num_threads;
Mesh.MaxNumThreads2D = num_threads;
Mesh.MaxNumThreads3D = num_threads;

// We specify higher order elements for mid side nodes which are a good standard
// for tensor mechanics in MOOSE.
Mesh.ElementOrder = 2;

// This actually starts the meshing procedure in 3D - Mesh 2; will mesh in 2D.
Mesh 3;

//------------------------------------------------------------------------------
// Save and exit
// This saves the mesh in the file format specified using the extension in the
// variable file_name. In this case we use gmsh's native *.msh format which can
// be read by moose.
Save Str(file_name);

// If we are running this in a large sweep we will want to exit after saving the
// the mesh - uncomment this line
//Exit;


