# DGFEM for Acoustic Wave Propagation 

[![Build Status](https://travis-ci.org/pvanberg/DGFEM-Acoustic.svg?branch=master)]()  [![Maintenance](https://img.shields.io/badge/version-1.2.1-red)](https://gitlab.ensam.eu/khelladi/DGFEM-Acoustic/-/tree/v1.2.1) [![Maintenance](https://img.shields.io/badge/c++-14%20|%2017%20|%2020-27ae60.svg)](https://gitlab.ensam.eu/khelladi/DGFEM-Acoustic/-/tree/v1.2.1) 

This repository implements a discontinuous Galerkin finite element method (DGFEM) applied to the linearized Euler equations and the acoustic perturbation equations. The solver is based on [GMSH](http://gmsh.info/) library and supports a wide range of features:

- 1D, 2D, 3D problems
- 4-th order Runge-Kutta
- High order elements
- Absorbing and reflecting boundaries
- Complex geometry and unstructured grid

For more information, a detailled report is available [here](https://github.com/pvanberg/DGFEM-Acoustic/blob/master/DGFEM_acoustic.pdf). 

| Auditorium     | Isosurfaces     | Bulk|
| ------------- |:-------------:| :-------------:| 
| <img src="https://gitlab.ensam.eu/khelladi/DGFEM-Acoustic/-/raw/b1026a1c6b9d312d02f6f70e776ed98e054ef00a/assets/auditorium_source2_2.png" width="400" height="200" />    | <img src="https://gitlab.ensam.eu/khelladi/DGFEM-Acoustic/-/raw/b1026a1c6b9d312d02f6f70e776ed98e054ef00a/assets/auditorium_source_iso1.png" width="400" height="200" />  | <img src="https://gitlab.ensam.eu/khelladi/DGFEM-Acoustic/-/raw/b1026a1c6b9d312d02f6f70e776ed98e054ef00a/assets/auditorium_source_bulk1.png" width="400" height="200" /> |


## Getting Started
 	
### Prerequisites

First, make sure the following libraries are installed. If you are running a linux distribution (ubuntu, debian, ...), an installation [script](https://github.com/pvanberg/MATH0471-DG/blob/master/build.sh) is provided. 

```
Gmsh
Eigen
Lapack
Blas
OpenMP
```

### Installing

```
git clone https://gitlab.ensam.eu/khelladi/DGFEM-Acoustic.git
cd DGFEM-Acoustic
mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release  -G "Unix Makefiles" -DGMSH_INCLUDE_DIRS="../gmsh-4.1.5-Linux64-sdk/include" -DGMSH_LIBRARIES="../gmsh-4.1.5-Linux64-sdk/lib/libgmsh.so" -DGMSH_EXECUTABLE="../gmsh-4.1.5-Linux64-sdk/bin/gmsh" -DEIGEN_INCLUDE_DIRS="/usr/include/eigen3"
make -j4
```

## Running the tests
Once the sources sucessfully build, you can start using with the solver. It required two arguments: a mesh file created with Gmsh and a config file containing the solver options. Examples of mesh files and config files are given [here](https://github.com/pvanberg/MATH0471-DG/tree/master/doc).

```
cd bin
./dgalerkin mymesh.msh myconfig.conf
```

### Minimal working example

2D propagation of an Gaussian initial condition over a square.

```
./dgalerkin ../../doc/2d/square.msh ../../doc/config/config.conf 
```

## Authors

* Pierre-Olivier Vanberg
* Martin Lacroix
* Tom Servais

## Contributors
* Sofiane Khelladi