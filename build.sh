#!/bin/sh

echo "******************************************";
echo "*     Discontinuous galerkin setup.      *";
echo "******************************************";
echo

echo "[0] Create some project directories if don't exist.";

mkdir 3rdParty

mkdir results

echo "[1] Get depedencies and external libraries.";

# Detect OS
OS="$(uname -s)"

if [ "$OS" = "Darwin" ]; then
	# macOS: use Homebrew
	if ! command -v brew > /dev/null 2>&1; then
		echo "Homebrew not found. Please install it from https://brew.sh and re-run this script.";
		exit 1;
	fi

	brew_install() {
		if brew list "$1" > /dev/null 2>&1; then
			echo "$1 found.";
		else
			echo "$1 not found, installing...";
			brew install "$1";
			echo "$1 installed.";
		fi
	}

	brew_install cmake
	brew_install gcc
	brew_install gfortran
	brew_install openblas
	brew_install lapack
	brew_install glfw
	brew_install libxft
	brew_install vtk
	brew_install fftw
	brew_install nlohmann-json
	brew_install eigen
	brew_install mpich

else
	# Linux: use apt-get
	dpkg -s cmake > /dev/null 2>&1;
	if [ $? -eq 0 ]; then
		echo "Cmake found.";
	else
		echo "Cmake not found, installing...";
		sudo apt-get -y install cmake;
		echo "Cmake installed.";
	fi

	dpkg -s g++ > /dev/null 2>&1;
	if [ $? -eq 0 ]; then
		echo "G++ found.";
	else
		echo "G++ not found, installing...";
		sudo apt-get -y install g++;
		export CC=gcc;
		export CXX=g++;
		echo "G++ installed.";
	fi

	dpkg -s gfortran > /dev/null 2>&1;
	if [ $? -eq 0 ]; then
		echo "Gfortran found.";
	else
		echo "Gfortran not found, installing...";
		sudo apt-get -y install gfortran;
		echo "Gfortran installed.";
	fi

	dpkg -s libblas-dev liblapack-dev > /dev/null 2>&1;
	if [ $? -eq 0 ]; then
		echo "Lapack/Blas found.";
	else
		echo "Lapack/Blas not found, installing...";
		sudo apt-get -y install libblas-dev liblapack-dev;
		echo "Lapack/Blas installed.";
	fi

	dpkg -s libglu1-mesa > /dev/null 2>&1;
	if [ $? -eq 0 ]; then
		echo "libGLU found.";
	else
		echo "libGLU not found, installing...";
		sudo apt-get -y install libglu1-mesa;
		echo "libGLU installed.";
	fi

	dpkg -s libxft2 > /dev/null 2>&1;
	if [ $? -eq 0 ]; then
		echo "libxft found.";
	else
		echo "libxft not found, installing...";
		sudo apt-get -y install libxft2;
		echo "libxft2 installed.";
	fi

	dpkg -s libvtk9-dev > /dev/null 2>&1;
	if [ $? -eq 0 ]; then
		echo "libvtk9-dev found.";
	else
		echo "libvtk9-dev not found, installing...";
		sudo apt-get -y install libvtk9-dev;
		echo "libvtk9-dev installed.";
	fi

	dpkg -s libfftw3-dev > /dev/null 2>&1;
	if [ $? -eq 0 ]; then
		echo "libfftw3-dev found.";
	else
		echo "libfftw3-dev not found, installing...";
		sudo apt-get -y install libfftw3-dev;
		echo "libfftw3-dev installed.";
	fi

	dpkg -s nlohmann-json3-dev > /dev/null 2>&1;
	if [ $? -eq 0 ]; then
		echo "nlohmann-json3-dev found.";
	else
		echo "nlohmann-json3-dev not found, installing...";
		sudo apt-get -y install nlohmann-json3-dev;
		echo "nlohmann-json3-dev installed.";
	fi

	dpkg -s libeigen3-dev > /dev/null 2>&1;
	if [ $? -eq 0 ]; then
		echo "Eigen found.";
	else
		echo "libeigen3-dev not found, installing...";
		sudo apt-get -y install libeigen3-dev;
		echo "Eigen installed.";
	fi

	dpkg -s libopenmpi-dev > /dev/null 2>&1;
	if [ $? -eq 0 ]; then
		echo "OpenMPI found.";
	else
		echo "OpenMPI not found, installing...";
		sudo apt-get -y install libopenmpi-dev openmpi-bin;
		echo "OpenMPI installed.";
	fi
fi

gmsh_version=4.15.2

if [ "$OS" = "Darwin" ]; then
	if [ ! -d "3rdParty/gmsh" ]; then
		echo "Gmsh not found, installing...";
		ARCH="$(uname -m)"
		if [ "$ARCH" = "arm64" ]; then
			GMSH_PKG="gmsh-${gmsh_version}-MacOSARM-sdk"
		else
			GMSH_PKG="gmsh-${gmsh_version}-MacOSX-sdk"
		fi
		wget https://gmsh.info/bin/macOS/${GMSH_PKG}.tgz
		tar -xf ${GMSH_PKG}.tgz
		rm -rf ${GMSH_PKG}.tgz
		mv ${GMSH_PKG} 3rdParty/gmsh
		echo "Gmsh installed."
	else
		echo "Gmsh found.";
	fi
else
	if [ ! -d "3rdParty/gmsh" ]; then
		echo "Gmsh not found, installing...";
		wget http://gmsh.info/bin/Linux/gmsh-${gmsh_version}-Linux64-sdk.tgz
		tar -xf gmsh-${gmsh_version}-Linux64-sdk.tgz
		rm -rf gmsh-${gmsh_version}-Linux64-sdk.tgz
		mv gmsh-${gmsh_version}-Linux64-sdk 3rdParty/gmsh
		echo "Gmsh installed."
	else
		echo "Gmsh found.";
	fi
fi

cd 3rdParty/gmsh/
export FC=gfortran
export PATH=${PWD}/bin:${PWD}/lib:${PATH}
export INCLUDE=${PWD}/include:${INCLUDE}
export LIB=${PWD}/lib:${LIB}
export PYTHONPATH=${PWD}/lib:${PYTHONPATH}
export DYLD_LIBRARY_PATH=${PWD}/lib:${DYLD_LIBRARY_PATH}
cd ../../


echo "[2] Build sources.";

rm -rf build/
mkdir build

# Set DG_USE_MPI=ON to enable MPI parallelization (default: OFF)
DG_USE_MPI=${DG_USE_MPI:-OFF}

MPI_CMAKE_ARGS=""
if [ "$DG_USE_MPI" = "ON" ]; then
	if ! command -v mpicc > /dev/null 2>&1 || ! command -v mpicxx > /dev/null 2>&1 || ! command -v mpirun > /dev/null 2>&1; then
		echo "MPI toolchain not found in PATH (mpicc/mpicxx/mpirun required).";
		exit 1;
	fi
	MPI_CMAKE_ARGS="-DMPI_C_COMPILER=$(command -v mpicc) -DMPI_CXX_COMPILER=$(command -v mpicxx) -DMPIEXEC_EXECUTABLE=$(command -v mpirun)"
	echo "Using MPI toolchain:";
	echo "  mpicc  = $(command -v mpicc)";
	echo "  mpicxx = $(command -v mpicxx)";
	echo "  mpirun = $(command -v mpirun)";
fi

cd build/
cmake ../ -DCMAKE_BUILD_TYPE=Release -DDG_USE_MPI=${DG_USE_MPI} ${MPI_CMAKE_ARGS} -G "Unix Makefiles"
make -j
if [ $? -eq 0 ]; then
    	echo "[end] Everything went successfully.";
else
	echo "[end] Error!";
fi
