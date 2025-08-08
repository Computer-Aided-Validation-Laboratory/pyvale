// 
// Demo input file for parameterised geometries with Gmsh
// Author: Rory Spencer
// Date:  Nov-2023

// Geometry Variables
gaugeHeight = 10E-3;
gaugeWidth = 2.5E-3;
// gaugeThickness = 1E-3; //2D for now

// Parameterisation
//_*
p0 = 1.5E-3;
p1 = 1E-3;  // Testing comments in the variable block
// Another test comment
p2 = 1.2E-3;
filename = "gmsh-test.msh"; // String variable
//**
lc = 1E-4;

// Create some points defining the boundary
// Will have vertical symmetry
Point(1) = {0,-gaugeHeight,0,lc}; //Bottom of centreline on specimen
Point(2) = {p0,-5E-3,0,lc}; // Parameterised point 0
Point(3) = {p1,0,0,lc}; // Parameterised point 1
Point(4) = {p2,5E-3,0,lc}; // Parameterised point 2
Point(5) = {gaugeWidth,gaugeHeight,0,lc}; //Top of gauge side line
Point(6) = {0,gaugeHeight,0,lc}; //Top of gauge centreline
Point(7) = {gaugeWidth,-gaugeHeight,0,lc}; // Bottom Edge of gauge

// Connect things up with some lines
Line(1) = {1,7}; // Bottom Hor Line
Spline(2) = {7,2,3,4,5}; //Vertical up side
Line(3) = {5,6}; // Top Hor Line
Line(4) = {6,1}; // Centreline

Curve Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};

// Top & Bottom
Transfinite Curve{1} = 10;
Transfinite Curve{3} = 10;

// Sides
Transfinite Curve{4} = 50;
Transfinite Curve{2} = 50;

Transfinite Surface{1};

Recombine Surface{:};
Mesh 2;


Physical Surface("Specimen",1) = {1};

Physical Curve("Top-BC",2) = {3};
Physical Curve("Mid-BC",3) = {4};
Physical Curve("Btm-BC",4) = {1};

Save Str(filename);
Exit;
