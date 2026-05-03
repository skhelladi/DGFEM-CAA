// 3D unstructured tetrahedral cube — partition + scaling test
// 100m x 100m x 100m cube. Mesh size h controls number of tets.
// No Transfinite/Recombine — gmsh's default Delaunay 3D produces tets.

SetFactory("OpenCASCADE");

L = 100.0;   // domain side [m]
h = 7.0;     // characteristic mesh size [m] — gives ~50-80k tets
p = 2;       // spatial order (1: linear, 2: quadratic, ...)

Box(1) = {0, 0, 0, L, L, L};

// Default 2D and 3D algorithms (Delaunay) — produces tetrahedra, not hexahedra
Mesh.Algorithm   = 6;   // Frontal-Delaunay (2D)
Mesh.Algorithm3D = 1;   // Delaunay (3D)
Mesh.ElementOrder = p;  // apply spatial order to mesh elements

MeshSize{ PointsOf{ Volume{1}; } } = h;

// Physical groups
//   Box(1) face tags with OpenCASCADE:
//     1: y=0 (front), 2: x=L (right), 3: y=L (back),
//     4: x=0 (left),  5: z=0 (bottom), 6: z=L (top)
Physical Surface("ref", 11) = {5};               // Reflecting: bottom z=0
Physical Surface("abs", 10) = {1, 2, 3, 4, 6};  // Absorbing: all other faces
Physical Volume("domain")   = {1};

Mesh 3;
Save "cube_unstr.msh";
