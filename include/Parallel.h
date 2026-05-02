#pragma once

#ifdef DG_USE_MPI
// fft.h defines `#define REAL 0` which conflicts with MPI C++ bindings (extern Datatype REAL).
// Suppress it around the MPI include and restore it afterwards.
#pragma push_macro("REAL")
#undef REAL
#include <mpi.h>
#pragma pop_macro("REAL")
#endif

namespace Parallel {

void init(int* argc, char*** argv);
void finalize();
int  rank();
int  size();
bool isRoot();
void barrier();

#ifdef DG_USE_MPI

MPI_Comm comm();

template<typename T> MPI_Datatype mpiType();
template<> inline MPI_Datatype mpiType<double>() { return MPI_DOUBLE; }
template<> inline MPI_Datatype mpiType<int>()    { return MPI_INT; }
template<> inline MPI_Datatype mpiType<float>()  { return MPI_FLOAT; }

template<typename T>
inline void allReduce(T* sendbuf, T* recvbuf, int count, MPI_Op op = MPI_SUM) {
    MPI_Allreduce(sendbuf, recvbuf, count, mpiType<T>(), op, comm());
}

template<typename T>
inline void bcast(T* buf, int count, int root = 0) {
    MPI_Bcast(buf, count, mpiType<T>(), root, comm());
}

template<typename T>
inline T allReduceScalar(T val, MPI_Op op = MPI_SUM) {
    T result;
    MPI_Allreduce(&val, &result, 1, mpiType<T>(), op, comm());
    return result;
}

#else

// Stubs when MPI is disabled — single-rank semantics
template<typename T>
inline void allReduce(T* sendbuf, T* recvbuf, int count, int /*op*/ = 0) {
    for (int i = 0; i < count; ++i) recvbuf[i] = sendbuf[i];
}
template<typename T>
inline void bcast(T* /*buf*/, int /*count*/, int /*root*/ = 0) {}
template<typename T>
inline T allReduceScalar(T val, int /*op*/ = 0) { return val; }

#endif

} // namespace Parallel
