#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gmsh.h>
#include <nlohmann/json.hpp>
#include <unistd.h>

#include "Parallel.h"

#ifdef DG_HAS_METIS
#include <metis.h>
#endif

#ifdef DG_HAS_SCOTCH
#include <scotch.h>
#endif

#if defined(DG_HAS_PARMETIS) && defined(DG_USE_MPI)
#include <parmetis.h>
#endif

#include "MeshPartitioning.h"

namespace fs = std::filesystem;
using ordered_json = nlohmann::ordered_json;

namespace
{
    struct ExplicitPartitionInput
    {
        int cellDim = 0;
        int elementType = -1;
        int cellNumNodes = 0;
        std::vector<std::size_t> elementTags;
        std::vector<std::set<int>> adjacency;
        std::unordered_map<std::string, std::vector<int>> ownerCellsBySubEntity;
        std::unordered_map<std::size_t, int> firstCellOwnerByNode;
    };

    bool modelAlreadyPartitioned()
    {
        const int cellDim = gmsh::model::getDimension();
        std::vector<std::pair<int, int>> entities;
        gmsh::model::getEntities(entities, cellDim);
        for (const auto &dimTag : entities)
        {
            std::vector<int> partitions;
            gmsh::model::getPartitions(dimTag.first, dimTag.second, partitions);
            if (!partitions.empty())
                return true;
        }
        return false;
    }

    size_t countEntityElements(int dim, int tag)
    {
        std::vector<int> elementTypes;
        std::vector<std::vector<std::size_t>> elementTags, nodeTags;
        gmsh::model::mesh::getElements(elementTypes, elementTags, nodeTags, dim, tag);
        size_t count = 0;
        for (const auto &tags : elementTags)
            count += tags.size();
        return count;
    }

    std::string normalizeMode(std::string mode)
    {
        std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
        return mode;
    }

    std::string canonicalNodeKey(const std::vector<std::size_t> &nodeTags, size_t offset, size_t count)
    {
        std::vector<std::size_t> key(nodeTags.begin() + offset, nodeTags.begin() + offset + count);
        std::sort(key.begin(), key.end());

        std::ostringstream stream;
        for (size_t tag : key)
            stream << tag << ':';
        return stream.str();
    }

    ExplicitPartitionInput buildElementAdjacencyGraph()
    {
        ExplicitPartitionInput input;
        input.cellDim = gmsh::model::getDimension();
        if (input.cellDim < 2)
            throw std::runtime_error("External graph partitioning currently supports 2D/3D meshes only.");

        std::vector<int> elementTypes;
        gmsh::model::mesh::getElementTypes(elementTypes, input.cellDim);
        if (elementTypes.size() != 1)
            throw std::runtime_error("External graph partitioning requires a single cell element type.");

        input.elementType = elementTypes.front();

        std::string elementName;
        int elementDim = 0;
        int elementOrder = 0;
        int numNodes = 0;
        int numPrimaryNodes = 0;
        std::vector<double> paramCoords;
        gmsh::model::mesh::getElementProperties(input.elementType,
                                                elementName,
                                                elementDim,
                                                elementOrder,
                                                numNodes,
                                                paramCoords,
                                                numPrimaryNodes);
        input.cellNumNodes = numNodes;

        const bool hasQuadrilateralFaces = elementName.find("Hexahedron") != std::string::npos;
        const size_t subEntityNodeCount = (input.cellDim == 2)
                                              ? static_cast<size_t>(elementOrder + 1)
                                              : static_cast<size_t>(hasQuadrilateralFaces ? 4 : 3);

        std::unordered_map<std::string, int> firstOwnerBySubEntity;
        std::vector<std::pair<int, int>> entities;
        gmsh::model::getEntities(entities, input.cellDim);

        for (const auto &dimTag : entities)
        {
            std::vector<std::size_t> entityElementTags;
            std::vector<std::size_t> entityNodeTags;
            gmsh::model::mesh::getElementsByType(input.elementType, entityElementTags, entityNodeTags, dimTag.second);
            if (entityElementTags.empty())
                continue;

            std::vector<std::size_t> entitySubEntityNodes;
            if (input.cellDim == 2)
                gmsh::model::mesh::getElementEdgeNodes(input.elementType, entitySubEntityNodes, dimTag.second);
            else
                gmsh::model::mesh::getElementFaceNodes(input.elementType,
                                                      hasQuadrilateralFaces ? 4 : 3,
                                                      entitySubEntityNodes,
                                                      dimTag.second);

            const size_t baseIndex = input.elementTags.size();
            input.elementTags.insert(input.elementTags.end(), entityElementTags.begin(), entityElementTags.end());
            input.adjacency.resize(input.elementTags.size());

            const size_t subEntitiesPerElement = entitySubEntityNodes.size() / (entityElementTags.size() * subEntityNodeCount);
            for (size_t localEl = 0; localEl < entityElementTags.size(); ++localEl)
            {
                const int globalEl = static_cast<int>(baseIndex + localEl);
                for (int node = 0; node < input.cellNumNodes; ++node)
                    input.firstCellOwnerByNode.emplace(entityNodeTags[localEl * input.cellNumNodes + node], globalEl);

                for (size_t subEntity = 0; subEntity < subEntitiesPerElement; ++subEntity)
                {
                    const size_t offset = (localEl * subEntitiesPerElement + subEntity) * subEntityNodeCount;
                    const std::string key = canonicalNodeKey(entitySubEntityNodes, offset, subEntityNodeCount);
                    std::vector<int> &owners = input.ownerCellsBySubEntity[key];
                    if (owners.empty() || owners.back() != globalEl)
                        owners.push_back(globalEl);

                    auto inserted = firstOwnerBySubEntity.emplace(key, globalEl);
                    if (!inserted.second)
                    {
                        const int otherEl = inserted.first->second;
                        if (otherEl != globalEl)
                        {
                            input.adjacency[globalEl].insert(otherEl);
                            input.adjacency[otherEl].insert(globalEl);
                        }
                    }
                }
            }
        }

        return input;
    }

    fs::path makeScratchDirectory()
    {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        fs::path path = fs::temp_directory_path() /
                        ("dgfem_partition_" + std::to_string(getpid()) + "_" + std::to_string(stamp));
        fs::create_directories(path);
        return path;
    }

    fs::path writeGraphFile(const ExplicitPartitionInput &input, const fs::path &directory)
    {
        const fs::path graphPath = directory / "mesh.graph";
        std::ofstream graph(graphPath);
        if (!graph.is_open())
            throw std::runtime_error("Unable to write graph file: " + graphPath.string());

        size_t edgeCount = 0;
        for (const auto &neighbours : input.adjacency)
            edgeCount += neighbours.size();
        edgeCount /= 2;

        graph << input.adjacency.size() << ' ' << edgeCount << '\n';
        for (const auto &neighbours : input.adjacency)
        {
            bool first = true;
            for (int neighbour : neighbours)
            {
                if (!first)
                    graph << ' ';
                graph << (neighbour + 1);
                first = false;
            }
            graph << '\n';
        }

        return graphPath;
    }

    template <typename IndexType>
    std::pair<std::vector<IndexType>, std::vector<IndexType>> buildCsrAdjacency(const ExplicitPartitionInput &input)
    {
        std::vector<IndexType> xadj(input.adjacency.size() + 1, 0);
        std::vector<IndexType> adjncy;

        size_t offset = 0;
        for (size_t vertex = 0; vertex < input.adjacency.size(); ++vertex)
        {
            xadj[vertex] = static_cast<IndexType>(offset);
            for (int neighbour : input.adjacency[vertex])
            {
                adjncy.push_back(static_cast<IndexType>(neighbour));
                ++offset;
            }
        }
        xadj[input.adjacency.size()] = static_cast<IndexType>(offset);

        return {xadj, adjncy};
    }

    std::string resolvePartitionCommand(const std::string &partitionMode, const std::string &partitionCommand)
    {
        if (!partitionCommand.empty())
            return partitionCommand;

        const std::string mode = normalizeMode(partitionMode);
        if (mode == "gpmetis-bin" || mode == "metis-bin" || mode == "pmetis-bin")
            return "gpmetis";
        if (mode == "scotch-bin" || mode == "dgpart-bin" || mode == "ptscotch-bin")
            return "dgpart";

        return std::string();
    }

    std::vector<int> readPartitionFile(const fs::path &partitionPath, size_t expectedCount)
    {
        std::ifstream input(partitionPath);
        if (!input.is_open())
            throw std::runtime_error("Unable to read partition file: " + partitionPath.string());

        std::vector<int> partitions;
        partitions.reserve(expectedCount);
        int partitionId = 0;
        while (input >> partitionId)
            partitions.push_back(partitionId + 1);

        if (partitions.size() != expectedCount)
            throw std::runtime_error("Unexpected partition count in: " + partitionPath.string());

        return partitions;
    }

    int inferElementPartition(const ExplicitPartitionInput &input,
                              const std::vector<int> &cellPartitions,
                              int dim,
                              int numPrimaryNodes,
                              int nodesPerElement,
                              const std::vector<std::size_t> &nodeTags,
                              size_t offset)
    {
        if (dim == input.cellDim - 1)
        {
            const size_t keyNodeCount = (input.cellDim == 2)
                                            ? static_cast<size_t>(nodesPerElement)
                                            : static_cast<size_t>(numPrimaryNodes);
            const std::string key = canonicalNodeKey(nodeTags, offset, keyNodeCount);
            auto ownerIter = input.ownerCellsBySubEntity.find(key);
            if (ownerIter != input.ownerCellsBySubEntity.end() && !ownerIter->second.empty())
                return cellPartitions[ownerIter->second.front()];
        }

        auto nodeOwner = input.firstCellOwnerByNode.find(nodeTags[offset]);
        if (nodeOwner == input.firstCellOwnerByNode.end())
            throw std::runtime_error("Unable to infer a partition for a lower-dimensional mesh element.");

        return cellPartitions[nodeOwner->second];
    }

    std::pair<std::vector<std::size_t>, std::vector<int>> buildExplicitPartitionVectors(const ExplicitPartitionInput &input,
                                                                                         const std::vector<int> &cellPartitions)
    {
        std::vector<std::size_t> elementTags = input.elementTags;
        std::vector<int> partitions = cellPartitions;

        for (int dim = input.cellDim - 1; dim >= 0; --dim)
        {
            std::vector<std::pair<int, int>> entities;
            gmsh::model::getEntities(entities, dim);
            for (const auto &dimTag : entities)
            {
                std::vector<int> elementTypes;
                std::vector<std::vector<std::size_t>> entityElementTags;
                std::vector<std::vector<std::size_t>> entityNodeTags;
                gmsh::model::mesh::getElements(elementTypes, entityElementTags, entityNodeTags, dim, dimTag.second);

                for (size_t typeIndex = 0; typeIndex < elementTypes.size(); ++typeIndex)
                {
                    const std::vector<std::size_t> &tags = entityElementTags[typeIndex];
                    const std::vector<std::size_t> &nodes = entityNodeTags[typeIndex];
                    if (tags.empty())
                        continue;

                    std::string elementName;
                    int elementDim = 0;
                    int elementOrder = 0;
                    int numNodes = 0;
                    int numPrimaryNodes = 0;
                    std::vector<double> paramCoords;
                    gmsh::model::mesh::getElementProperties(elementTypes[typeIndex],
                                                            elementName,
                                                            elementDim,
                                                            elementOrder,
                                                            numNodes,
                                                            paramCoords,
                                                            numPrimaryNodes);

                    for (size_t localEl = 0; localEl < tags.size(); ++localEl)
                    {
                        const size_t nodeOffset = localEl * static_cast<size_t>(numNodes);
                        elementTags.push_back(tags[localEl]);
                        partitions.push_back(inferElementPartition(input,
                                                                   cellPartitions,
                                                                   dim,
                                                                   numPrimaryNodes,
                                                                   numNodes,
                                                                   nodes,
                                                                   nodeOffset));
                    }
                }
            }
        }

        return {elementTags, partitions};
    }

    void applyExplicitPartitionToModel(const ExplicitPartitionInput &input,
                                       int partitionCount,
                                       const std::vector<int> &cellPartitions)
    {
        const auto explicitPartitions = buildExplicitPartitionVectors(input, cellPartitions);
        gmsh::option::setNumber("Mesh.PartitionCreateGhostCells", 1);
        gmsh::option::setNumber("Mesh.PartitionCreateTopology", 1);
        gmsh::model::mesh::partition(partitionCount, explicitPartitions.first, explicitPartitions.second);
    }

#ifdef DG_HAS_METIS
    void applyMetisApiPartition(const ExplicitPartitionInput &input,
                                int partitionCount,
                                const std::string &partitionMode)
    {
        auto csr = buildCsrAdjacency<idx_t>(input);
        idx_t nvtxs = static_cast<idx_t>(input.adjacency.size());
        idx_t ncon = 1;
        idx_t nparts = static_cast<idx_t>(partitionCount);
        idx_t options[METIS_NOPTIONS];
        idx_t edgecut = 0;
        std::vector<idx_t> part(nvtxs, 0);

        METIS_SetDefaultOptions(options);
        const std::string mode = normalizeMode(partitionMode);

        const int status = (mode == "pmetis")
                               ? METIS_PartGraphRecursive(&nvtxs,
                                                          &ncon,
                                                          csr.first.data(),
                                                          csr.second.data(),
                                                          nullptr,
                                                          nullptr,
                                                          nullptr,
                                                          &nparts,
                                                          nullptr,
                                                          nullptr,
                                                          options,
                                                          &edgecut,
                                                          part.data())
                               : METIS_PartGraphKway(&nvtxs,
                                                     &ncon,
                                                     csr.first.data(),
                                                     csr.second.data(),
                                                     nullptr,
                                                     nullptr,
                                                     nullptr,
                                                     &nparts,
                                                     nullptr,
                                                     nullptr,
                                                     options,
                                                     &edgecut,
                                                     part.data());

        if (status != METIS_OK)
            throw std::runtime_error("METIS direct API partitioning failed with status " + std::to_string(status));

        std::vector<int> cellPartitions(part.size(), 1);
        for (size_t i = 0; i < part.size(); ++i)
            cellPartitions[i] = static_cast<int>(part[i]) + 1;

        applyExplicitPartitionToModel(input, partitionCount, cellPartitions);
    }
#endif

#ifdef DG_HAS_SCOTCH
    void applyScotchApiPartition(const ExplicitPartitionInput &input,
                                 int partitionCount)
    {
        auto csr = buildCsrAdjacency<SCOTCH_Num>(input);
        std::vector<SCOTCH_Num> vend(input.adjacency.size(), 0);
        for (size_t vertex = 0; vertex < input.adjacency.size(); ++vertex)
            vend[vertex] = csr.first[vertex + 1];

        SCOTCH_Graph graph;
        SCOTCH_Strat strategy;
        if (SCOTCH_graphInit(&graph) != 0)
            throw std::runtime_error("SCOTCH_graphInit failed");
        if (SCOTCH_stratInit(&strategy) != 0)
        {
            SCOTCH_graphExit(&graph);
            throw std::runtime_error("SCOTCH_stratInit failed");
        }

        const int buildStatus = SCOTCH_graphBuild(&graph,
                                                  0,
                                                  static_cast<SCOTCH_Num>(input.adjacency.size()),
                                                  csr.first.data(),
                                                  vend.data(),
                                                  nullptr,
                                                  nullptr,
                                                  static_cast<SCOTCH_Num>(csr.second.size()),
                                                  csr.second.data(),
                                                  nullptr);
        if (buildStatus != 0)
        {
            SCOTCH_stratExit(&strategy);
            SCOTCH_graphExit(&graph);
            throw std::runtime_error("SCOTCH_graphBuild failed");
        }
        if (SCOTCH_graphCheck(&graph) != 0)
        {
            SCOTCH_stratExit(&strategy);
            SCOTCH_graphExit(&graph);
            throw std::runtime_error("SCOTCH_graphCheck failed");
        }

        std::vector<SCOTCH_Num> part(input.adjacency.size(), 0);
        const int partStatus = SCOTCH_graphPart(&graph,
                                                static_cast<SCOTCH_Num>(partitionCount),
                                                &strategy,
                                                part.data());

        SCOTCH_stratExit(&strategy);
        SCOTCH_graphExit(&graph);

        if (partStatus != 0)
            throw std::runtime_error("SCOTCH_graphPart failed");

        std::vector<int> cellPartitions(part.size(), 1);
        for (size_t i = 0; i < part.size(); ++i)
            cellPartitions[i] = static_cast<int>(part[i]) + 1;

        applyExplicitPartitionToModel(input, partitionCount, cellPartitions);
    }
#endif

#if defined(DG_HAS_PARMETIS) && defined(DG_USE_MPI)
    MPI_Datatype parmetisIdxMpiType()
    {
        if (sizeof(idx_t) == sizeof(int))
            return MPI_INT;
        if (sizeof(idx_t) == sizeof(long))
            return MPI_LONG;
        if (sizeof(idx_t) == sizeof(long long))
            return MPI_LONG_LONG;
        throw std::runtime_error("Unsupported idx_t size for MPI_Allgatherv.");
    }

    void applyParmetisApiPartition(const ExplicitPartitionInput &input,
                                   int partitionCount)
    {
        if (partitionCount <= 1)
        {
            std::vector<int> trivial(input.elementTags.size(), 1);
            applyExplicitPartitionToModel(input, partitionCount, trivial);
            return;
        }

        const int worldSize = Parallel::size();
        const int worldRank = Parallel::rank();

        if (worldSize <= 0)
            throw std::runtime_error("MPI world size is invalid for ParMETIS partitioning.");

        const idx_t globalVertexCount = static_cast<idx_t>(input.adjacency.size());
        if (globalVertexCount <= 0)
            throw std::runtime_error("ParMETIS partitioning requires a non-empty graph.");

        std::vector<idx_t> vtxdist(static_cast<size_t>(worldSize) + 1, 0);
        const idx_t baseChunk = globalVertexCount / static_cast<idx_t>(worldSize);
        const idx_t remainder = globalVertexCount % static_cast<idx_t>(worldSize);
        idx_t cursor = 0;
        for (int rank = 0; rank < worldSize; ++rank)
        {
            vtxdist[rank] = cursor;
            cursor += baseChunk + ((static_cast<idx_t>(rank) < remainder) ? 1 : 0);
        }
        vtxdist[worldSize] = globalVertexCount;

        const idx_t localStart = vtxdist[worldRank];
        const idx_t localEnd = vtxdist[worldRank + 1];
        const idx_t localVertexCount = localEnd - localStart;

        std::vector<idx_t> xadj(static_cast<size_t>(localVertexCount) + 1, 0);
        std::vector<idx_t> adjncy;
        for (idx_t localIndex = 0; localIndex < localVertexCount; ++localIndex)
        {
            const size_t globalIndex = static_cast<size_t>(localStart + localIndex);
            for (int neighbour : input.adjacency[globalIndex])
                adjncy.push_back(static_cast<idx_t>(neighbour));

            xadj[static_cast<size_t>(localIndex) + 1] = static_cast<idx_t>(adjncy.size());
        }

        idx_t wgtflag = 0;
        idx_t numflag = 0;
        idx_t ncon = 1;
        idx_t nparts = static_cast<idx_t>(partitionCount);
        idx_t edgecut = 0;
        std::vector<real_t> tpwgts(static_cast<size_t>(nparts) * static_cast<size_t>(ncon),
                                   1.0 / static_cast<real_t>(nparts));
        std::vector<real_t> ubvec(static_cast<size_t>(ncon), 1.05);
        idx_t options[3] = {0, 0, 0};
        std::vector<idx_t> localPart(static_cast<size_t>(localVertexCount), 0);

        MPI_Comm comm = Parallel::comm();
        const int status = ParMETIS_V3_PartKway(vtxdist.data(),
                                                xadj.data(),
                                                adjncy.data(),
                                                nullptr,
                                                nullptr,
                                                &wgtflag,
                                                &numflag,
                                                &ncon,
                                                &nparts,
                                                tpwgts.data(),
                                                ubvec.data(),
                                                options,
                                                &edgecut,
                                                localPart.data(),
                                                &comm);
        if (status != METIS_OK)
            throw std::runtime_error("ParMETIS direct API partitioning failed with status " + std::to_string(status));

        std::vector<int> recvCounts(static_cast<size_t>(worldSize), 0);
        std::vector<int> displs(static_cast<size_t>(worldSize), 0);
        for (int rank = 0; rank < worldSize; ++rank)
        {
            recvCounts[rank] = static_cast<int>(vtxdist[rank + 1] - vtxdist[rank]);
            displs[rank] = static_cast<int>(vtxdist[rank]);
        }

        std::vector<idx_t> gatheredPart(static_cast<size_t>(globalVertexCount), 0);
        const MPI_Datatype idxType = parmetisIdxMpiType();
        MPI_Allgatherv(localPart.data(),
                       static_cast<int>(localPart.size()),
                   idxType,
                       gatheredPart.data(),
                       recvCounts.data(),
                       displs.data(),
                   idxType,
                       Parallel::comm());

        std::vector<int> cellPartitions(gatheredPart.size(), 1);
        for (size_t i = 0; i < gatheredPart.size(); ++i)
            cellPartitions[i] = static_cast<int>(gatheredPart[i]) + 1;

        applyExplicitPartitionToModel(input, partitionCount, cellPartitions);
    }
#endif

    void applyGpmetisPartition(int partitionCount, const std::string &command)
    {
        const ExplicitPartitionInput input = buildElementAdjacencyGraph();
        const fs::path scratchDir = makeScratchDirectory();
        const fs::path graphPath = writeGraphFile(input, scratchDir);
        const fs::path logPath = scratchDir / "partition.log";

        std::ostringstream cmd;
        cmd << command << ' ' << graphPath.string() << ' ' << partitionCount
            << " >" << logPath.string() << " 2>&1";

        const int rc = std::system(cmd.str().c_str());
        if (rc != 0)
            throw std::runtime_error("Partition command failed: " + cmd.str() + ". See " + logPath.string());

        fs::path partitionPath = graphPath;
        partitionPath += ".part." + std::to_string(partitionCount);

        const std::vector<int> cellPartitions = readPartitionFile(partitionPath, input.elementTags.size());
        applyExplicitPartitionToModel(input, partitionCount, cellPartitions);
    }

    ordered_json readJsonFile(const fs::path &path)
    {
        std::ifstream input(path);
        if (!input.is_open())
            throw std::runtime_error("Unable to read JSON file: " + path.string());
        ordered_json json;
        input >> json;
        return json;
    }

    std::vector<int> readIntArrayCompat(const ordered_json &json,
                                        const std::string &preferredKey,
                                        const std::string &legacyKey)
    {
        if (json.contains(preferredKey))
            return json[preferredKey].get<std::vector<int>>();
        if (!legacyKey.empty() && json.contains(legacyKey))
            return json[legacyKey].get<std::vector<int>>();
        return {};
    }

    std::map<int, std::pair<std::string, double>> readResolvedBoundaryConditions(const ordered_json &json)
    {
        std::map<int, std::pair<std::string, double>> resolved;
        if (!json.is_array())
            return resolved;

        for (const auto &entry : json)
        {
            resolved[entry["physicalTag"].get<int>()] = {
                entry["type"].get<std::string>(),
                entry.value("value", 0.0)};
        }
        return resolved;
    }

    PartitionLayout readLayout(const ordered_json &json)
    {
        PartitionLayout layout;
        layout.ownedEntities3D = readIntArrayCompat(json, "ownedCellEntities", "ownedEntities3D");
        layout.haloEntities3D = readIntArrayCompat(json, "haloCellEntities", "haloEntities3D");
        layout.boundaryEntities2D = readIntArrayCompat(json, "boundaryEntities", "boundaryEntities2D");

        if (json.contains("interfaceEntitiesByRank"))
        {
            for (auto iter = json["interfaceEntitiesByRank"].begin(); iter != json["interfaceEntitiesByRank"].end(); ++iter)
                layout.interfaceEntitiesByRank[std::stoi(iter.key())] = iter.value().get<std::vector<int>>();
        }

        return layout;
    }

    fs::path resolveRelativeTo(const fs::path &baseFile, const std::string &value)
    {
        const fs::path path(value);
        if (path.is_absolute())
            return path;
        if (baseFile.empty())
            return path;
        return baseFile.parent_path() / path;
    }
}

namespace meshpartitioning
{
    void ensurePartitionedModel(int partitionCount,
                                const std::string &partitionMode,
                                const std::string &partitionCommand)
    {
        if (modelAlreadyPartitioned())
            return;

        const std::string mode = normalizeMode(partitionMode);
        if (mode == "gmsh")
        {
            gmsh::option::setNumber("Mesh.PartitionCreateGhostCells", 1);
            gmsh::option::setNumber("Mesh.PartitionCreateTopology", 1);
            gmsh::model::mesh::partition(partitionCount);
            return;
        }

            if (mode == "metis" || mode == "gpmetis" || mode == "pmetis")
            {
        #ifdef DG_HAS_METIS
                applyMetisApiPartition(buildElementAdjacencyGraph(), partitionCount, mode);
                return;
        #else
                throw std::runtime_error("METIS direct API backend requested but METIS was not found at build time.");
        #endif
            }

            if (mode == "parmetis")
            {
#if defined(DG_HAS_PARMETIS) && defined(DG_USE_MPI)
                applyParmetisApiPartition(buildElementAdjacencyGraph(), partitionCount);
                return;
#else
                throw std::runtime_error("ParMETIS direct API backend requested but ParMETIS/MPI support is not available at build time.");
#endif
            }

            if (mode == "scotch" || mode == "ptscotch" || mode == "dgpart")
            {
        #ifdef DG_HAS_SCOTCH
                applyScotchApiPartition(buildElementAdjacencyGraph(), partitionCount);
                return;
        #else
                throw std::runtime_error("SCOTCH direct API backend requested but SCOTCH was not found at build time.");
        #endif
            }

        const std::string command = resolvePartitionCommand(mode, partitionCommand);
        if (command.empty())
            throw std::runtime_error("Unsupported partition mode: " + partitionMode);

        applyGpmetisPartition(partitionCount, command);
    }

    PartitionLayout buildLayoutForPartition(int partitionIndex,
                                            int partitionCount,
                                            bool emitDiagnostics,
                                            const std::string &partitionMode,
                                            const std::string &partitionCommand)
    {
        const int myPart = partitionIndex + 1;
        const int cellDim = gmsh::model::getDimension();
        const int boundaryDim = cellDim - 1;
        ensurePartitionedModel(partitionCount, partitionMode, partitionCommand);

        PartitionLayout layout;

        {
            std::vector<std::pair<int, int>> entities;
            gmsh::model::getEntities(entities, cellDim);
            for (const auto &dimTag : entities)
            {
                std::vector<int> partitions;
                gmsh::model::getPartitions(dimTag.first, dimTag.second, partitions);
                if (std::find(partitions.begin(), partitions.end(), myPart) == partitions.end())
                    continue;

                std::vector<std::size_t> ghostTags;
                std::vector<int> ghostParts;
                gmsh::model::mesh::getGhostElements(dimTag.first, dimTag.second, ghostTags, ghostParts);
                if (ghostTags.empty())
                    layout.ownedEntities3D.push_back(dimTag.second);
                else
                    layout.haloEntities3D.push_back(dimTag.second);
            }
        }

        {
            std::vector<std::pair<int, int>> entities;
            gmsh::model::getEntities(entities, boundaryDim);
            for (const auto &dimTag : entities)
            {
                std::vector<int> partitions;
                gmsh::model::getPartitions(dimTag.first, dimTag.second, partitions);
                if (partitions.empty())
                    continue;
                if (std::find(partitions.begin(), partitions.end(), myPart) == partitions.end())
                    continue;

                if (partitions.size() == 1)
                {
                    layout.boundaryEntities2D.push_back(dimTag.second);
                }
                else if (partitions.size() == 2)
                {
                    const int otherPart = (partitions[0] == myPart) ? partitions[1] : partitions[0];
                    layout.interfaceEntitiesByRank[otherPart - 1].push_back(dimTag.second);
                }
            }
        }

        if (emitDiagnostics)
        {
            size_t nOwnedEls = 0, nHaloEls = 0, nBoundaryFaces = 0, nInterfaceFaces = 0;
            for (int tag : layout.ownedEntities3D)
                nOwnedEls += countEntityElements(cellDim, tag);
            for (int tag : layout.haloEntities3D)
                nHaloEls += countEntityElements(cellDim, tag);
            for (int tag : layout.boundaryEntities2D)
                nBoundaryFaces += countEntityElements(boundaryDim, tag);
            for (const auto &entry : layout.interfaceEntitiesByRank)
                for (int tag : entry.second)
                    nInterfaceFaces += countEntityElements(boundaryDim, tag);

            std::cout << "[partition " << partitionIndex << "] layout: "
                      << nOwnedEls << " owned 3D els, "
                      << nHaloEls << " halo 3D els, "
                      << nBoundaryFaces << " physical bdry faces, "
                      << nInterfaceFaces << " interface faces"
                      << " (" << layout.interfaceEntitiesByRank.size() << " neighbours)\n"
                      << std::flush;
        }

        return layout;
    }

    PartitionPackageData loadPartitionPackageForRank(const std::string &manifestFile,
                                                     const std::string &packageFile,
                                                     int partitionIndex,
                                                     int partitionCount)
    {
        if (manifestFile.empty() && packageFile.empty())
            throw std::runtime_error("A partition manifest or package file is required.");

        PartitionPackageData data;
        data.partitionIndex = partitionIndex;
        data.partitionCount = partitionCount;

        fs::path manifestPath;
        if (!manifestFile.empty())
        {
            manifestPath = fs::absolute(manifestFile);
            const ordered_json manifest = readJsonFile(manifestPath);
            data.manifestFile = manifestPath.string();
            data.partitionMode = manifest.value("partitionMode", data.partitionMode);
            if (manifest.contains("partitionCount") && manifest["partitionCount"].get<int>() != partitionCount)
                throw std::runtime_error("Partition manifest count does not match MPI world size.");

            if (manifest.contains("resolvedBoundaryConditions"))
                data.resolvedBoundaryConditions = readResolvedBoundaryConditions(manifest["resolvedBoundaryConditions"]);

            if (packageFile.empty())
            {
                bool found = false;
                for (const auto &entry : manifest["packages"])
                {
                    if (entry["partitionIndex"].get<int>() == partitionIndex)
                    {
                        data.packageFile = resolveRelativeTo(manifestPath, entry["file"].get<std::string>()).string();
                        found = true;
                        break;
                    }
                }
                if (!found)
                    throw std::runtime_error("Partition package entry missing for rank " + std::to_string(partitionIndex));
            }

            if (manifest.contains("partitionedMesh"))
                data.partitionedMeshFile = resolveRelativeTo(manifestPath, manifest["partitionedMesh"].get<std::string>()).string();
        }

        fs::path packagePath;
        if (!packageFile.empty())
            packagePath = fs::absolute(packageFile);
        else
            packagePath = fs::absolute(data.packageFile);

        const ordered_json package = readJsonFile(packagePath);
        data.packageFile = packagePath.string();
        data.partitionIndex = package.value("partitionIndex", partitionIndex);
        data.partitionCount = package.value("partitionCount", partitionCount);
        data.meshDimension = package.value("meshDimension", 0);
        data.partitionMode = package.value("partitionMode", data.partitionMode);
        if (data.partitionCount != partitionCount)
            throw std::runtime_error("Partition package count does not match MPI world size.");
        if (data.partitionIndex != partitionIndex)
            throw std::runtime_error("Partition package index does not match MPI rank.");

        if (package.contains("resolvedBoundaryConditions"))
            data.resolvedBoundaryConditions = readResolvedBoundaryConditions(package["resolvedBoundaryConditions"]);

        if (package.contains("partitionedMesh"))
            data.partitionedMeshFile = resolveRelativeTo(packagePath, package["partitionedMesh"].get<std::string>()).string();
        if (package.contains("layout"))
            data.layout = readLayout(package["layout"]);

        if (data.partitionedMeshFile.empty())
            throw std::runtime_error("Partition package does not provide a partitioned mesh path.");

        return data;
    }
}