// 3D unit cube — structured hexahedral mesh for VTK display testing
// 100m x 100m x 100m cube, N divisions per edge → N³ hexahedra
// Physical groups:
//   "abs" : Absorbing boundary (all faces except bottom)
//   "ref" : Reflecting boundary (bottom z=0)

SetFactory("OpenCASCADE");

L = 100.0;   // domain side [m]
N = 50;       // number of divisions per edge

Box(1) = {0, 0, 0, L, L, L};

// Force purely structured transfinite meshing — disable QuadQuasiStructured
// to avoid the background-mesh subdivision that corrupts Transfinite Volume.
Mesh.Algorithm   = 10;   // Frontal-Delaunay (no QuadQuasiStructured)
Mesh.Algorithm3D = 4;   // Frontal 3D

// N divisions per edge
Transfinite Curve{:} = N + 1;

// 4-sided structured surfaces — Recombine after Transfinite gives quads
// without triggering the background-mesh step.
Transfinite Surface{:};
Recombine Surface{:};

// Structured hex volume
Transfinite Volume{1};
Recombine Volume{1};

// Physical groups
// Box(1) face tags with OpenCASCADE:
//   1: y=0 (front), 2: x=L (right), 3: y=L (back),
//   4: x=0 (left),  5: z=0 (bottom), 6: z=L (top)
Physical Surface("ref", 11) = {5};              // Reflecting: bottom z=0
Physical Surface("abs", 10) = {1, 2, 3, 4, 6}; // Absorbing: all other faces
Physical Volume("domain")   = {1};

Mesh 3;
Save "cube.msh";
