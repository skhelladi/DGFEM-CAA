SET(SRCS
	dgalerkin.cpp
	configParser.cpp
	Mesh.cpp
	utils.cpp
	solver.cpp
	eqEdit.cpp
	fft.cpp
	Parallel.cpp
	../include/configParser.h
	../include/Mesh.h
	../include/utils.h
	../include/solver.h
	../include/eqEdit.h
	../include/fft.h
	../include/Parallel.h
)

ADD_EXECUTABLE(dgalerkin ${SRCS})
TARGET_LINK_LIBRARIES(dgalerkin ${GMSH_LIBRARIES} ${LAPACKBLAS_LIBRARIES} ${VTK_LIBRARIES} -O3 OpenMP::OpenMP_CXX -lpthread -lwave -lfftw3) #-ltbb

IF(DG_USE_MPI)
    TARGET_LINK_LIBRARIES(dgalerkin MPI::MPI_CXX)
ENDIF()

vtk_module_autoinit(
	TARGETS dgalerkin
	MODULES ${VTK_LIBRARIES}
)

install(TARGETS dgalerkin
  RUNTIME DESTINATION bin
)
