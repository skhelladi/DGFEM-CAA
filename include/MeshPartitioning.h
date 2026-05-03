#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

struct PartitionLayout
{
    std::vector<int> ownedEntities3D;
    std::vector<int> haloEntities3D;
    std::vector<int> boundaryEntities2D;
    std::map<int, std::vector<int>> interfaceEntitiesByRank;
};

struct PartitionPackageData
{
    std::string manifestFile;
    std::string packageFile;
    std::string partitionedMeshFile;
    std::string partitionMode = "gmsh";
    int partitionIndex = 0;
    int partitionCount = 0;
    int meshDimension = 0;
    PartitionLayout layout;
    std::map<int, std::pair<std::string, double>> resolvedBoundaryConditions;
};

namespace meshpartitioning
{
    void ensurePartitionedModel(int partitionCount,
                                const std::string &partitionMode = "gmsh",
                                const std::string &partitionCommand = "");
    PartitionLayout buildLayoutForPartition(int partitionIndex,
                                            int partitionCount,
                                            bool emitDiagnostics = true,
                                            const std::string &partitionMode = "gmsh",
                                            const std::string &partitionCommand = "");
    PartitionPackageData loadPartitionPackageForRank(const std::string &manifestFile,
                                                     const std::string &packageFile,
                                                     int partitionIndex,
                                                     int partitionCount);
}

#ifdef DG_USE_MPI
PartitionLayout buildPartitionLayout();
#endif