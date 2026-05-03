#include <cstdlib>

#include "Parallel.h"

#ifdef DG_USE_MPI

static int       _rank = 0;
static int       _size = 1;
static MPI_Comm  _comm = MPI_COMM_NULL;

namespace {

int getEnvInt(const char *name)
{
    const char *value = std::getenv(name);
    return value ? std::atoi(value) : -1;
}

int detectLauncherRank()
{
    const int rank = getEnvInt("OMPI_COMM_WORLD_RANK");
    if (rank >= 0)
        return rank;

    const int pmiRank = getEnvInt("PMI_RANK");
    if (pmiRank >= 0)
        return pmiRank;

    return getEnvInt("PMIX_RANK");
}

int detectLauncherSize()
{
    const int size = getEnvInt("OMPI_COMM_WORLD_SIZE");
    if (size > 0)
        return size;

    const int pmiSize = getEnvInt("PMI_SIZE");
    if (pmiSize > 0)
        return pmiSize;

    return getEnvInt("PMIX_SIZE");
}

} // namespace

namespace Parallel {

void init(int* argc, char*** argv) {
    int provided;
    MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
    _comm = MPI_COMM_WORLD;
    MPI_Comm_rank(_comm, &_rank);
    MPI_Comm_size(_comm, &_size);

    const int launcherSize = detectLauncherSize();
    const int launcherRank = detectLauncherRank();
    // Broader condition: trust launcher env vars even when MPI returns
    // corrupted values (gmsh SDK static initializers corrupt Comm_size).
    if (launcherSize > 0 && launcherRank >= 0 && launcherRank < launcherSize) {
        if (launcherSize != _size || launcherRank != _rank) {
            _size = launcherSize;
            _rank = launcherRank;
        }
    }
}

void finalize() { MPI_Finalize(); }
int  rank()
{
    const int launcherSize = detectLauncherSize();
    const int launcherRank = detectLauncherRank();
    if (launcherSize > 1 && launcherRank >= 0 && launcherRank < launcherSize)
        return launcherRank;
    return _rank;
}

int  size()
{
    const int launcherSize = detectLauncherSize();
    if (launcherSize > _size)
        return launcherSize;
    return _size;
}

bool isRoot()   { return rank() == 0; }
void barrier()  { MPI_Barrier(_comm); }
MPI_Comm comm() { return _comm; }

} // namespace Parallel

#else

namespace Parallel {

void init(int* /*argc*/, char*** /*argv*/) {}
void finalize() {}
int  rank()   { return 0; }
int  size()   { return 1; }
bool isRoot() { return true; }
void barrier() {}

} // namespace Parallel

#endif
