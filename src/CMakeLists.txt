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

ADD_EXECUTABLE(dgalerkin ${SRCS})
TARGET_LINK_LIBRARIES(dgalerkin ${GMSH_LIBRARIES} ${LAPACKBLAS_LIBRARIES} ${VTK_LIBRARIES} ${PARTITIONER_LIBRARIES} -O3 OpenMP::OpenMP_CXX -lpthread -lwave -lfftw3) #-ltbb

ADD_EXECUTABLE(dgmesh_preprocess ${PREPROCESS_SRCS})
TARGET_LINK_LIBRARIES(dgmesh_preprocess ${GMSH_LIBRARIES} ${LAPACKBLAS_LIBRARIES} ${PARTITIONER_LIBRARIES} OpenMP::OpenMP_CXX -lpthread -lwave -lfftw3)

IF(DG_USE_MPI)
    TARGET_LINK_LIBRARIES(dgalerkin MPI::MPI_CXX)
	TARGET_LINK_LIBRARIES(dgmesh_preprocess MPI::MPI_CXX)
ENDIF()

vtk_module_autoinit(
	TARGETS dgalerkin
	MODULES ${VTK_LIBRARIES}
)

install(TARGETS dgalerkin
  RUNTIME DESTINATION bin
)

install(TARGETS dgmesh_preprocess
	RUNTIME DESTINATION bin
)
