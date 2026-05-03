#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include <gmsh.h>

#include "Mesh.h"
#include "Parallel.h"
#include "configParser.h"

namespace
{
int checkListOrderingAndUniqueness(Mesh &mesh,
                                  const std::vector<size_t> &elements,
                                  const std::string &label,
                                  int neighbour,
                                  int rank)
{
    int failures = 0;
    std::unordered_set<size_t> seen;
    seen.reserve(elements.size());

    size_t previousTag = 0;
    bool first = true;

    for (size_t el : elements)
    {
        if (el >= static_cast<size_t>(mesh.getElNum()))
        {
            std::cerr << "[rank " << rank << "] invalid element index in " << label
                      << " list for neighbour " << neighbour << ": " << el << std::endl;
            ++failures;
            continue;
        }

        if (!seen.insert(el).second)
        {
            std::cerr << "[rank " << rank << "] duplicate element index in " << label
                      << " list for neighbour " << neighbour << ": " << el << std::endl;
            ++failures;
        }

        const size_t tag = mesh.elTag(el);
        if (!first && tag < previousTag)
        {
            std::cerr << "[rank " << rank << "] non-monotonic gmsh tag order in " << label
                      << " list for neighbour " << neighbour << " (" << previousTag << " -> " << tag << ")"
                      << std::endl;
            ++failures;
        }
        previousTag = tag;
        first = false;
    }

    return failures;
}
}

int main(int argc, char **argv)
{
#ifndef DG_USE_MPI
    (void)argc;
    (void)argv;
    std::cout << "dg_halo_invariants: MPI disabled, skipping." << std::endl;
    return EXIT_SUCCESS;
#else
    Parallel::init(&argc, &argv);

    if (argc != 2)
    {
        if (Parallel::isRoot())
            std::cerr << "Usage: dg_halo_invariants <config.(json|conf)>" << std::endl;
        Parallel::finalize();
        return E2BIG;
    }

    if (Parallel::size() <= 1)
    {
        if (Parallel::isRoot())
            std::cout << "dg_halo_invariants: requires at least 2 MPI ranks, skipping." << std::endl;
        Parallel::finalize();
        return EXIT_SUCCESS;
    }

    const bool verboseAllRanks = (std::getenv("DG_MPI_VERBOSE_ALL_RANKS") != nullptr);

    gmsh::initialize();
    gmsh::option::setNumber("General.Terminal", (Parallel::isRoot() || verboseAllRanks) ? 1.0 : 0.0);

    Config config;
    const std::string configName = argv[1];
    if (fileExtension(configName) == "json")
        config = config::parseJSON(configName);
    else if (fileExtension(configName) == "conf")
        config = config::parseConfig(configName);
    else
    {
        if (Parallel::isRoot())
            std::cerr << "Unsupported config extension for: " << configName << std::endl;
        gmsh::finalize();
        Parallel::finalize();
        return EINVAL;
    }

    Mesh mesh(config);

    int localFailures = 0;
    const int myRank = Parallel::rank();

    const auto &sendMap = mesh.getHaloSendElIds();
    const auto &recvMap = mesh.getHaloRecvElIds();

    std::set<int> neighbours;
    for (const auto &kv : sendMap)
        neighbours.insert(kv.first);
    for (const auto &kv : recvMap)
        neighbours.insert(kv.first);

    for (int neighbour : neighbours)
    {
        const auto sendIt = sendMap.find(neighbour);
        const auto recvIt = recvMap.find(neighbour);

        const std::vector<size_t> empty;
        const std::vector<size_t> &sendEls = (sendIt != sendMap.end()) ? sendIt->second : empty;
        const std::vector<size_t> &recvEls = (recvIt != recvMap.end()) ? recvIt->second : empty;

        localFailures += checkListOrderingAndUniqueness(mesh, sendEls, "send", neighbour, myRank);
        localFailures += checkListOrderingAndUniqueness(mesh, recvEls, "recv", neighbour, myRank);

        const int sendCount = static_cast<int>(sendEls.size());
        const int recvCount = static_cast<int>(recvEls.size());

        int remoteRecvCount = -1;
        MPI_Sendrecv(&sendCount, 1, MPI_INT, neighbour, 19001,
                     &remoteRecvCount, 1, MPI_INT, neighbour, 19001,
                     Parallel::comm(), MPI_STATUS_IGNORE);

        if (remoteRecvCount != recvCount)
        {
            std::cerr << "[rank " << myRank << "] mismatch with neighbour " << neighbour
                      << ": sendCount=" << sendCount
                      << " but neighbour recvCount=" << remoteRecvCount << std::endl;
            ++localFailures;
        }

        int remoteSendCount = -1;
        MPI_Sendrecv(&recvCount, 1, MPI_INT, neighbour, 19002,
                     &remoteSendCount, 1, MPI_INT, neighbour, 19002,
                     Parallel::comm(), MPI_STATUS_IGNORE);

        if (remoteSendCount != sendCount)
        {
            std::cerr << "[rank " << myRank << "] mismatch with neighbour " << neighbour
                      << ": recvCount=" << recvCount
                      << " but neighbour sendCount=" << remoteSendCount << std::endl;
            ++localFailures;
        }
    }

    const int globalFailures = Parallel::allReduceScalar<int>(localFailures);
    if (Parallel::isRoot())
    {
        if (globalFailures == 0)
            std::cout << "HALO_INVARIANTS: PASS" << std::endl;
        else
            std::cout << "HALO_INVARIANTS: FAIL (" << globalFailures << " issues)" << std::endl;
    }

    gmsh::finalize();
    Parallel::finalize();

    return (globalFailures == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
#endif
}
