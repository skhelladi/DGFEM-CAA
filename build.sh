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

JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null)"
if [ -z "$JOBS" ]; then
	JOBS=4
fi

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

	brew_install_optional() {
		if brew info "$1" > /dev/null 2>&1; then
			brew_install "$1";
		else
			echo "$1 formula not available on this Homebrew setup, skipping.";
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

	echo "[1.a] Optional graph partitioners.";
	brew_install_optional metis
	brew_install_optional scotch
	brew_install_optional parmetis

else
	# Linux: use apt-get
	apt_package_available() {
		if command -v apt-cache > /dev/null 2>&1; then
			apt-cache show "$1" > /dev/null 2>&1;
		else
			apt list --all-versions "$1" 2>/dev/null | grep -q "/";
		fi
	}

	apt_install_first_available() {
		DESC="$1";
		shift;

		FOUND_PKG="";
		for PKG in "$@"; do
			if apt_package_available "$PKG"; then
				FOUND_PKG="$PKG";
				break;
			fi
		done

		if [ -z "$FOUND_PKG" ]; then
			echo "$DESC package not available, skipping.";
			return;
		fi

		if dpkg -s "$FOUND_PKG" > /dev/null 2>&1; then
			echo "$DESC found ($FOUND_PKG).";
		else
			echo "$DESC not found, installing $FOUND_PKG...";
			sudo apt-get -y install "$FOUND_PKG";
			echo "$DESC installed ($FOUND_PKG).";
		fi
	}

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

	echo "[1.a] Optional graph partitioners.";
	apt_install_first_available "METIS tools" metis
	apt_install_first_available "METIS development files" libmetis-dev
	apt_install_first_available "ParMETIS development files" libparmetis-dev parmetis
	apt_install_first_available "Scotch tools" scotch ptscotch
	apt_install_first_available "PT-Scotch development files" libptscotch-dev libscotch-dev
fi

echo "[1.b] Install and build GKlib, METIS and ParMETIS from source in 3rdParty.";

PROJECT_ROOT="${PWD}"

if ! command -v git > /dev/null 2>&1; then
	echo "git not found, installing...";
	if [ "$OS" = "Darwin" ]; then
		brew_install git
	else
		sudo apt-get -y install git
	fi
fi

clone_if_missing() {
	REPO_URL="$1";
	DEST_DIR="$2";
	LABEL="$3";
	if [ ! -d "$DEST_DIR" ]; then
		echo "$LABEL source not found, cloning...";
		git clone "$REPO_URL" "$DEST_DIR"
		if [ $? -ne 0 ]; then
			echo "Failed to clone $LABEL.";
			exit 1;
		fi
	else
		echo "$LABEL source found.";
	fi
}

clone_if_missing https://github.com/KarypisLab/GKlib.git 3rdParty/GKlib GKlib
clone_if_missing https://github.com/KarypisLab/METIS.git 3rdParty/METIS METIS
clone_if_missing https://github.com/KarypisLab/ParMETIS.git 3rdParty/ParMETIS ParMETIS

if ! command -v mpicc > /dev/null 2>&1 || ! command -v mpicxx > /dev/null 2>&1; then
	echo "MPI compilers (mpicc/mpicxx) are required to build ParMETIS.";
	exit 1;
fi

GKLIB_SRC_DIR="${PWD}/3rdParty/GKlib"
GKLIB_BUILD_DIR="${GKLIB_SRC_DIR}/build"
GKLIB_INSTALL_DIR="${GKLIB_SRC_DIR}/install"

METIS_SRC_DIR="${PWD}/3rdParty/METIS"
METIS_BUILD_DIR="${METIS_SRC_DIR}/build"
METIS_INSTALL_DIR="${METIS_SRC_DIR}/install"

PARMETIS_SRC_DIR="${PWD}/3rdParty/ParMETIS"
PARMETIS_BUILD_DIR="${PARMETIS_SRC_DIR}/build"
PARMETIS_INSTALL_DIR="${PARMETIS_SRC_DIR}/install"

echo "Build GKlib...";
cmake -S "${GKLIB_SRC_DIR}" -B "${GKLIB_BUILD_DIR}" \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX="${GKLIB_INSTALL_DIR}" \
	-DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
	-DCMAKE_C_COMPILER="$(command -v gcc)"
if [ $? -ne 0 ]; then
	echo "GKlib CMake configure failed.";
	exit 1;
fi

cmake --build "${GKLIB_BUILD_DIR}" --target GKlib -j
if [ $? -ne 0 ]; then
	echo "GKlib build failed.";
	exit 1;
fi

mkdir -p "${GKLIB_INSTALL_DIR}/include" "${GKLIB_INSTALL_DIR}/lib"
cp -f "${GKLIB_BUILD_DIR}/libGKlib.a" "${GKLIB_INSTALL_DIR}/lib/"
cp -f "${GKLIB_SRC_DIR}/include"/*.h "${GKLIB_INSTALL_DIR}/include/"

echo "Build METIS...";
cd "${METIS_SRC_DIR}"
make distclean > /dev/null 2>&1 || true
make config cc="$(command -v gcc)" gklib_path="${GKLIB_INSTALL_DIR}" prefix="${METIS_INSTALL_DIR}"
if [ $? -ne 0 ]; then
	echo "METIS configure failed.";
	exit 1;
fi

make -j"${JOBS}"
if [ $? -ne 0 ]; then
	echo "METIS build failed.";
	exit 1;
fi

make install
if [ $? -ne 0 ]; then
	echo "METIS install failed.";
	exit 1;
fi
cd "${PROJECT_ROOT}"

echo "Build ParMETIS...";

cmake -S "${PARMETIS_SRC_DIR}" -B "${PARMETIS_BUILD_DIR}" \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_INSTALL_PREFIX="${PARMETIS_INSTALL_DIR}" \
	-DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
	-DGKLIB_PATH="${GKLIB_INSTALL_DIR}" \
	-DMETIS_PATH="${METIS_INSTALL_DIR}" \
	-DCMAKE_C_COMPILER="$(command -v mpicc)" \
	-DCMAKE_CXX_COMPILER="$(command -v mpicxx)"

if [ $? -ne 0 ]; then
	echo "ParMETIS CMake configure failed.";
	exit 1;
fi

cmake --build "${PARMETIS_BUILD_DIR}" -j
if [ $? -ne 0 ]; then
	echo "ParMETIS build failed.";
	exit 1;
fi

cmake --install "${PARMETIS_BUILD_DIR}"
if [ $? -ne 0 ]; then
	echo "ParMETIS install failed.";
	exit 1;
fi

echo "ParMETIS installed in ${PARMETIS_INSTALL_DIR}.";

export INCLUDE=${PARMETIS_INSTALL_DIR}/include:${METIS_INSTALL_DIR}/include:${GKLIB_INSTALL_DIR}/include:${INCLUDE}
export LIB=${PARMETIS_INSTALL_DIR}/lib:${METIS_INSTALL_DIR}/lib:${GKLIB_INSTALL_DIR}/lib:${LIB}
export LD_LIBRARY_PATH=${PARMETIS_INSTALL_DIR}/lib:${METIS_INSTALL_DIR}/lib:${GKLIB_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
export DYLD_LIBRARY_PATH=${PARMETIS_INSTALL_DIR}/lib:${METIS_INSTALL_DIR}/lib:${GKLIB_INSTALL_DIR}/lib:${DYLD_LIBRARY_PATH}

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
make -j"${JOBS}"
if [ $? -eq 0 ]; then
    	echo "[end] Everything went successfully.";
else
	echo "[end] Error!";
fi
