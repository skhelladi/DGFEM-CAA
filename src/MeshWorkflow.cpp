#include <fstream>
#include <stdexcept>
#include <unordered_map>

#include <gmsh.h>

#include "MeshWorkflow.h"

namespace meshworkflow
{
    void assertMeshFileReadable(const std::string &meshFileName)
    {
        std::ifstream meshFile(meshFileName);
        if (!meshFile.is_open())
            throw std::runtime_error("Mesh file '" + meshFileName + "' not found or corrupted");
    }

    void reloadModel(const std::string &meshFileName)
    {
        assertMeshFileReadable(meshFileName);
        gmsh::clear();
        gmsh::open(meshFileName);
    }

    std::map<int, std::pair<std::string, double>> resolvePhysicalBoundaryConditions(
        const std::vector<BoundaryConditionSpec> &specs,
        int boundaryDim)
    {
        std::map<int, std::pair<std::string, double>> resolved;
        if (specs.empty())
            return resolved;

        gmsh::vectorpair physicalDimTags;
        gmsh::model::getPhysicalGroups(physicalDimTags, boundaryDim);

        std::unordered_map<std::string, int> nameToTag;
        nameToTag.reserve(physicalDimTags.size());
        for (const auto &dimTag : physicalDimTags)
        {
            std::string physName;
            gmsh::model::getPhysicalName(dimTag.first, dimTag.second, physName);
            if (!physName.empty())
                nameToTag[physName] = dimTag.second;
        }

        for (const auto &spec : specs)
        {
            auto it = nameToTag.find(spec.physicalName);
            if (it == nameToTag.end())
            {
                gmsh::logger::write("Boundary group not found in mesh: " + spec.physicalName);
                continue;
            }
            resolved[it->second] = std::make_pair(spec.type, spec.value);
        }

        return resolved;
    }
}