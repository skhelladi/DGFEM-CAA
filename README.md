# DGFEM for Acoustic Wave Propagation 

[![version](https://img.shields.io/badge/version-1.3.6-red)](https://github.com/skhelladi/DGFEM-CAA/releases/tag/v1.3.6) 
[![compilers](https://img.shields.io/badge/c++-17%20|%2020-27ae60.svg)](https://github.com/skhelladi/DGFEM-CAA/releases/tag/v1.3.6) 

This repository implements a discontinuous Galerkin finite element method (DGFEM) applied to the linearized Euler equations and the acoustic 
perturbation equations. 
The solver is based on [GMSH](http://gmsh.info/) library and supports a wide range of features:

- 1D, 2D, 3D problems
- 4-th order Runge-Kutta
- High order elements
- Absorbing and reflecting boundaries
- Support 'json' format configartion file
- Multiple sources support: monopoles, dipoles, quadrupoles, user defined analytical formulation sources and external data (csv and sound 'wave' file supported) 
- Complex geometry and unstructured grid
- Optional MPI parallelization for distributed runs
- VTK post-processing, including `.pvtu`/`.pvd` outputs in MPI mode (use [Paraview](https://www.paraview.org/)) 
- User defined obervers position post-processing (text data time variables, Fast Fourier Transform, Pressure Power Spectral Density and sound 'wave' files)

Mesh support status:

- 2D: triangles and quadrilaterals are supported
- 3D: tetrahedra are supported and validated
- 3D hexahedra: work is in progress and not yet fully validated for production use

<!-- | Auditorium     | Isosurfaces     | Bulk|
| ------------- |:-------------:| :-------------:| 
| <img src="https://gitlab.ensam.eu/khelladi/DGFEM-Acoustic/-/raw/b1026a1c6b9d312d02f6f70e776ed98e054ef00a/assets/auditorium_source2_2.png" width="400" height="200" />    | <img src="https://gitlab.ensam.eu/khelladi/DGFEM-Acoustic/-/raw/b1026a1c6b9d312d02f6f70e776ed98e054ef00a/assets/auditorium_source_iso1.png" width="400" height="200" />  | <img src="https://gitlab.ensam.eu/khelladi/DGFEM-Acoustic/-/raw/b1026a1c6b9d312d02f6f70e776ed98e054ef00a/assets/auditorium_source_bulk1.png" width="400" height="200" /> | -->


## Getting Started
 	
### Prerequisites

First, make sure the following libraries are installed. If you are running a linux distribution (ubuntu, debian, ...), an installation [script](https://github.com/skhelladi/DGFEM-CAA/blob/main/build.sh) is provided. 

- Gmsh (v4.13.x)
- Eigen (v3.x)
- Lapack
- Blas
- OpenMP
- ~~Libtbb~~
- VTK (v9.x)
- FFTW
- MPI implementation when building with `-DDG_USE_MPI=ON` (`mpich` on macOS via `build.sh`, `openmpi` on Linux via `build.sh`)


### Installing
```
git clone https://github.com/skhelladi/DGFEM-CAA.git
cd DGFEM-CAA
sh build.sh
```

To enable MPI with the setup script:

```
git clone https://github.com/skhelladi/DGFEM-CAA.git
cd DGFEM-CAA
DG_USE_MPI=ON sh build.sh
```

or use
```
git clone https://github.com/skhelladi/DGFEM-CAA.git
cd DGFEM-CAA
mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release  -G "Unix Makefiles" -DGMSH_INCLUDE_DIRS="../gmsh-4.11.0-Linux64-sdk/include" -DGMSH_LIBRARIES="../gmsh-4.11.0-Linux64-sdk/lib/libgmsh.so" -DGMSH_EXECUTABLE="../gmsh-4.11.0-Linux64-sdk/bin/gmsh" -DEIGEN_INCLUDE_DIRS="/usr/include/eigen3"
make -j4
```

Minimal CMake configuration with MPI enabled:

```
git clone https://github.com/skhelladi/DGFEM-CAA.git
cd DGFEM-CAA
mkdir build-mpi && cd build-mpi
cmake .. -DCMAKE_BUILD_TYPE=Release -DDG_USE_MPI=ON
make -j4
```

## Running the tests
Once the sources sucessfully build, you can start using the solver. It requires a configuration file that references the mesh file and the solver options. Example configurations are provided in [tests](tests) and [doc/config](doc/config).

```
cd bin
./dgalerkin myconfig.conf
```
or

```
cd bin
./dgalerkin myconfig.json
```

### Minimal working example

2D propagation of an Gaussian initial condition over a square.

```
cd build/bin
./dgalerkin ../../tests/square.json
```

3D propagation on the tetrahedral cube regression case.

```
cd build/bin
./dgalerkin ../../tests/cube.json
```

MPI example on the same 2D regression case:

```
cd build-mpi/bin
mpirun -n 2 ./dgalerkin ../../tests/square.json
```

MPI writes one `.vtu` file per rank together with a `.pvtu` aggregator and a `results.pvd` time-series file.

Current regression cases in [tests](tests):

- [tests/square.json](tests/square.json): 2D quadrilateral mesh
- [tests/cube.json](tests/cube.json): 3D tetrahedral cube mesh
- [tests/cube_unstr.json](tests/cube_unstr.json): additional 3D cube case

or configure run_caa batch file with the right mesh and configurations files.

```
sh run_caa 
```
## Configuration file example
### Text format file
<!-- python style text highlight -->
```python 

meshFileName = doc/2d/square2.msh

# Initial, final time and time step(t>0)
timeStart=0
timeEnd=0.05
timeStep=0.00001

# Saving rate:
timeRate=0.001

# Element Type:
# ["Lagrange", "IsoParametric", ...]
elementType=Lagrange

# Time integration method:
# ["Euler1", "Euler2", "Runge-Kutta"...]
timeIntMethod=Runge-Kutta

# Boundary condition:
# /!\ The physical group name must match the Gmsh name (case sensitive)
# MyPhysicalName = Absorbing or Reflecting
Reflecting = Reflecting
Absorbing = Absorbing

# Number of thread
numThreads=12

# Mean Flow parameters
v0_x = -30
v0_y = 30
v0_z = 0
rho0 = 1.225
c0 = 100

# Source:
# name = fct,x,y,z,size,intensity,frequency,phase,duration
# - fct supported = [monopole, dipole, quadrupole, formula, file (csv, wav)]
# - if fct = formula => name = fct,"formula expr",x,y,z,size,duration (ex: formulat = 0.1 * sin(2 * pi * 50 * t))
# - if fct = file => name = fct,"filename",x,y,z,size
#	suported file formats are : csv, wav
# - (x,y,z) = source position
# - intensity = source intensity
# - frequency = source frequency
# NB: Extended source or Multiple sources are supported.
#     (source1 = ..., source2 = ...) indice must change.
source1 = formula, "0.1 * sin(2 * pi * 50 * t)", 0.0,0.0,0.0, 0.1, 0.1
source2 = monopole, 0.0,0.0,0.0, 0.1, 0.1,50,0,0.1
source3 = file,"data/data.csv", 0.0,0.0,0.0, 0.1
source4 = file,"data/data.wav", 0.0,0.0,0.0, 0.1
# source4 = udf, "-0.1 * sin(2 * pi * 50 * t)", -0.5,0.0,0.0, 0.1, 0.1

# Initial condition:
# name = gaussian,x,y,z, size, amplitude
# - fct supported = [gaussian]
# - (x,y,z) = position
# - amplitude = initial amplitude
# NB: Multiple CI are supported and recursively added.
#     (initial condition1 = ..., initial condition = ...)
# initialCondtition1 = gaussian, 0,0,0,1,1

# Observers position:
# name = x,y,z, size
# - (x,y,z) = position
# NB: Multiple observers are supported and recursively added.
#     (observer1 = ...; observer2 = ...)
observer1 = 2.11792,0.00340081,0.0,0.1
observer2 = -2.11792,0.00340081,0.0,0.1

```
### Json format file
<!-- python style text highlight -->
```json 
{
	"mesh": {
		"File": "doc/2d/square2.msh",
		"BC": {
			"number": 2,
			"boundary1": {
				"name": "Abs",
				"type": "Absorbing"
			},
			"boundary2": {
				"name": "Ref",
				"type": "Reflecting"
			}
		}
	},
	"solver": {
		"time": {
			"start": 0.0,
			"end": 0.05,
			"step": 5e-05,
			"rate": 0.001
		},
		"elementType": "Lagrange",
		"timeIntMethod": "Runge-Kutta",
		"numThreads": 12
	},
	"initialization": {
		"meanFlow": {
			"vx": 30.0,
			"vy": 0.0,
			"vz": 0.0,
			"rho": 1.225,
			"c": 100.0
		},
		"number": 2,
		"initialCondition1": {
			"type": "gaussian",
			"x": 0.2,
			"y": 0.0,
			"z": 0.0,
			"size": 1.0,
			"amplitude": 1.0
		},
		"initialCondition2": {
			"type": "gaussian",
			"x": -0.2,
			"y": 0.0,
			"z": 0.0,
			"size": 1.0,
			"amplitude": 1.0
		}
	},
	"observers": {
		"number": 2,
		"observer1": {
			"x": 2.11792,
			"y": 0.00340081,
			"z": 0.0,
			"size": 0.1
		},
		"observer2": {
			"x": -2.11792,
			"y": 0.00340081,
			"z": 0.0,
			"size": 0.1
		}
	},
	"sources": {
		"number": 3,
		"source1": {
			"type": "formula",
			"fct": "0.1 * sin(2 * pi * 50 * t)",
			"x": 0.0,
			"y": 0.0,
			"z": 0.0,
			"size": 0.1,
			"amplitude": 0.0,
			"frequency": 0.0,
			"phase": 0.0,
			"duration": 0.05
		},
		"source2": {
			"type": "file",
			"fct": "data/data2.wav",
			"x": 0.0,
			"y": 0.0,
			"z": 0.0,
			"size": 0.1,
			"amplitude": 0.0,
			"frequency": 0.0,
			"phase": 0.0,
			"duration": 0.05
		},
		"source3": {
			"type": "monopole",
			"fct": "",
			"x": 0.0,
			"y": 0.0,
			"z": 0.0,
			"size": 0.1,
			"amplitude": 0.1,
			"frequency": 50.0,
			"phase": 0.0,
			"duration": 0.05
		}
	}
}
```
## License

This project is licensed under the GPL-3 license.

## Author
* Sofiane Khelladi

#### Forked from code developed by
* Pierre-Olivier Vanberg
* Martin Lacroix
* Tom Servais

Link : https://github.com/pvanberg/DGFEM-Acoustic

