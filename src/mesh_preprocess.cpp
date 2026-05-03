#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <gmsh.h>

#include "MeshPartitioning.h"
#include "Parallel.h"
#include "MeshWorkflow.h"
#include "configParser.h"
#include "utils.h"

namespace fs = std::filesystem;

namespace
{
    ordered_json layoutToJson(const PartitionLayout &layout)
    {
        ordered_json json;
        json["ownedCellEntities"] = layout.ownedEntities3D;
        json["haloCellEntities"] = layout.haloEntities3D;
        json["boundaryEntities"] = layout.boundaryEntities2D;

        ordered_json interfaces = ordered_json::object();
        for (const auto &entry : layout.interfaceEntitiesByRank)
            interfaces[std::to_string(entry.first)] = entry.second;
        json["interfaceEntitiesByRank"] = interfaces;
        return json;
    }

    ordered_json boundarySpecsToJson(const std::vector<BoundaryConditionSpec> &specs)
    {
        ordered_json array = ordered_json::array();
        for (const auto &spec : specs)
        {
            array.push_back({
                {"physicalName", spec.physicalName},
                {"type", spec.type},
                {"value", spec.value},
            });
        }
        return array;
    }

    ordered_json resolvedBoundaryConditionsToJson(const std::map<int, std::pair<std::string, double>> &resolved)
    {
        ordered_json array = ordered_json::array();
        for (const auto &entry : resolved)
        {
            array.push_back({
                {"physicalTag", entry.first},
                {"type", entry.second.first},
                {"value", entry.second.second},
            });
        }
        return array;
    }

    void writeJsonFile(const fs::path &path, const ordered_json &content)
    {
        std::ofstream output(path);
        if (!output.is_open())
            throw std::runtime_error("Unable to write file: " + path.string());
        output << content.dump(2) << std::endl;
    }
}

int main(int argc, char **argv)
{
    Parallel::init(&argc, &argv);

    if (argc < 4 || argc > 6)
    {
        Parallel::finalize();
        return E2BIG;
    }

    const std::string configName = argv[1];
    const fs::path outputDir = argv[2];
    const int partitionCount = std::atoi(argv[3]);

    if (partitionCount <= 0)
    {
        Parallel::finalize();
        return EINVAL;
    }

    gmsh::initialize();
    gmsh::option::setNumber("General.Terminal", 1.0);

    Config config;
    if (fileExtension(configName) == "conf")
        config = config::parseConfig(configName);
    else if (fileExtension(configName) == "json")
        config = config::parseJSON(configName);
    else
    {
        Parallel::finalize();
        return EINVAL;
    }

    std::string partitionMode = config.partitionMode.empty() ? "gmsh" : config.partitionMode;
    std::string partitionCommand = config.partitionCommand;
    if (argc >= 5)
        partitionMode = argv[4];
    if (argc == 6)
        partitionCommand = argv[5];

    meshworkflow::reloadModel(config.meshFileName);
    const int meshDimension = gmsh::model::getDimension();
    const auto resolvedBCs = meshworkflow::resolvePhysicalBoundaryConditions(config.pendingPhysBCs, meshDimension - 1);

    meshpartitioning::ensurePartitionedModel(partitionCount, partitionMode, partitionCommand);

    fs::create_directories(outputDir);
    const fs::path partitionedMeshPath = outputDir / "partitioned.msh";
    gmsh::write(partitionedMeshPath.string());

    ordered_json manifest;
    manifest["format"] = "dgfem-caa.partition-package.v1";
    manifest["sourceConfig"] = configName;
    manifest["sourceMesh"] = config.meshFileName;
    manifest["partitionedMesh"] = partitionedMeshPath.filename().string();
    manifest["partitionMode"] = partitionMode;
    if (!partitionCommand.empty())
        manifest["partitionCommand"] = partitionCommand;
    manifest["partitionCount"] = partitionCount;
    manifest["meshDimension"] = meshDimension;
    manifest["boundaryConditionSpecs"] = boundarySpecsToJson(config.pendingPhysBCs);
    manifest["resolvedBoundaryConditions"] = resolvedBoundaryConditionsToJson(resolvedBCs);
    manifest["packages"] = ordered_json::array();

    for (int partitionIndex = 0; partitionIndex < partitionCount; ++partitionIndex)
    {
        const PartitionLayout layout = meshpartitioning::buildLayoutForPartition(partitionIndex, partitionCount, false);
        const fs::path packagePath = outputDir / ("part_" + std::to_string(partitionIndex) + ".json");

        ordered_json package;
        package["format"] = "dgfem-caa.partition-package.v1";
        package["sourceMesh"] = config.meshFileName;
        package["partitionedMesh"] = partitionedMeshPath.filename().string();
        package["partitionIndex"] = partitionIndex;
        package["partitionCount"] = partitionCount;
        package["meshDimension"] = meshDimension;
        package["partitionMode"] = partitionMode;
        if (!partitionCommand.empty())
            package["partitionCommand"] = partitionCommand;
        package["boundaryConditionSpecs"] = boundarySpecsToJson(config.pendingPhysBCs);
        package["resolvedBoundaryConditions"] = resolvedBoundaryConditionsToJson(resolvedBCs);
        package["layout"] = layoutToJson(layout);

        writeJsonFile(packagePath, package);
        manifest["packages"].push_back({
            {"partitionIndex", partitionIndex},
            {"file", packagePath.filename().string()},
        });
    }

    writeJsonFile(outputDir / "manifest.json", manifest);

    gmsh::finalize();
    Parallel::finalize();
    return EXIT_SUCCESS;
}