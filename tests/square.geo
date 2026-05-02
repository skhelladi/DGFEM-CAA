// 2D unit square — test mesh for MPI partition validation
// 100m x 100m square, mesh size h=5m → ~800 triangles
// Physical groups:
//   10 : Absorbing boundary (left, right, top)
//   11 : Reflecting boundary (bottom)

SetFactory("OpenCASCADE");

L = 100.0;   // domain side [m]
h = 5.0;     // mesh element size [m]

Rectangle(1) = {0, 0, 0, L, L, 0};

// Boundary curves
//   Curve 1 = bottom (y=0)
//   Curve 2 = right  (x=L)
//   Curve 3 = top    (y=L)
//   Curve 4 = left   (x=0)

Physical Curve("abs", 10)  = {2, 3, 4};  // Absorbing
Physical Curve("ref", 11)  = {1};          // Reflecting
Physical Surface("domain") = {1};

MeshSize{PointsOf{Surface{1};}} = h;

Mesh 2;
Save "tests/square.msh";
