SET(SRCS
	dgalerkin.cpp
	configParser.cpp
	Mesh.cpp
	MeshWorkflow.cpp
	MeshPartitioning.cpp
	utils.cpp
	solver.cpp
	eqEdit.cpp
	fft.cpp
	Parallel.cpp
	../include/configParser.h
	../include/Mesh.h
	../include/MeshWorkflow.h
	../include/MeshPartitioning.h
	../include/utils.h
	../include/solver.h
	../include/eqEdit.h
	../include/fft.h
	../include/Parallel.h
)

SET(PREPROCESS_SRCS
	mesh_preprocess.cpp
	configParser.cpp
	MeshWorkflow.cpp
	MeshPartitioning.cpp
	utils.cpp
	eqEdit.cpp
	fft.cpp
	Parallel.cpp
	../include/configParser.h
	../include/MeshWorkflow.h
	../include/MeshPartitioning.h
	../include/utils.h
	../include/eqEdit.h
	../include/fft.h
	../include/Parallel.h
)

SET(HALO_TEST_SRCS
	halo_invariants.cpp
	configParser.cpp
	Mesh.cpp
	MeshWorkflow.cpp
	MeshPartitioning.cpp
	utils.cpp
	eqEdit.cpp
	fft.cpp
	Parallel.cpp
	../include/configParser.h
	../include/Mesh.h
	../include/MeshWorkflow.h
	../include/MeshPartitioning.h
	../include/utils.h
	../include/eqEdit.h
	../include/fft.h
	../include/Parallel.h
)

ADD_EXECUTABLE(dgalerkin ${SRCS})
TARGET_LINK_LIBRARIES(dgalerkin ${GMSH_LIBRARIES} ${LAPACKBLAS_LIBRARIES} ${VTK_LIBRARIES} ${PARTITIONER_LIBRARIES} -O3 OpenMP::OpenMP_CXX -lpthread -lwave -lfftw3) #-ltbb

ADD_EXECUTABLE(dgmesh_preprocess ${PREPROCESS_SRCS})
TARGET_LINK_LIBRARIES(dgmesh_preprocess ${GMSH_LIBRARIES} ${LAPACKBLAS_LIBRARIES} ${PARTITIONER_LIBRARIES} OpenMP::OpenMP_CXX -lpthread -lwave -lfftw3)

IF(DG_USE_MPI)
    ADD_EXECUTABLE(dg_halo_invariants ${HALO_TEST_SRCS})
    TARGET_LINK_LIBRARIES(dg_halo_invariants ${GMSH_LIBRARIES} ${LAPACKBLAS_LIBRARIES} ${VTK_LIBRARIES} ${PARTITIONER_LIBRARIES} OpenMP::OpenMP_CXX -lpthread -lwave -lfftw3)
ENDIF()

IF(DG_USE_MPI)
    TARGET_LINK_LIBRARIES(dgalerkin MPI::MPI_CXX)
	TARGET_LINK_LIBRARIES(dgmesh_preprocess MPI::MPI_CXX)
	TARGET_LINK_LIBRARIES(dg_halo_invariants MPI::MPI_CXX)
ENDIF()

vtk_module_autoinit(
	TARGETS dgalerkin
	MODULES ${VTK_LIBRARIES}
)

IF(DG_USE_MPI)
vtk_module_autoinit(
	TARGETS dg_halo_invariants
	MODULES ${VTK_LIBRARIES}
)
ENDIF()

install(TARGETS dgalerkin
  RUNTIME DESTINATION bin
)

install(TARGETS dgmesh_preprocess
	RUNTIME DESTINATION bin
)

IF(DG_USE_MPI)
install(TARGETS dg_halo_invariants
	RUNTIME DESTINATION bin
)
ENDIF()

IF(DG_USE_MPI AND BUILD_TESTING)
add_test(
	NAME halo_invariants_square
	COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:dg_halo_invariants> ${PROJECT_SOURCE_DIR}/tests/square.json
)
set_tests_properties(halo_invariants_square PROPERTIES WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} TIMEOUT 120)

add_test(
	NAME halo_invariants_cube
	COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:dg_halo_invariants> ${PROJECT_SOURCE_DIR}/tests/cube.json
)
set_tests_properties(halo_invariants_cube PROPERTIES WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} TIMEOUT 300)

add_test(
	NAME mpi_regression_square
	COMMAND ${PROJECT_SOURCE_DIR}/scripts/check_mpi_regression.sh --case square --baseline-ranks 1 --mpi-ranks 2
)
set_tests_properties(mpi_regression_square PROPERTIES WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} TIMEOUT 240)

add_test(
	NAME mpi_regression_cube
	COMMAND ${PROJECT_SOURCE_DIR}/scripts/check_mpi_regression.sh --case cube --baseline-ranks 1 --mpi-ranks 4 --atol 1e-9 --rtol 1e-7
)
set_tests_properties(mpi_regression_cube PROPERTIES WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} TIMEOUT 600)
ENDIF()
