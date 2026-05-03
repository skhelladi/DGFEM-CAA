#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "configParser.h"

namespace meshworkflow
{
    void assertMeshFileReadable(const std::string &meshFileName);
    void reloadModel(const std::string &meshFileName);
    std::map<int, std::pair<std::string, double>> resolvePhysicalBoundaryConditions(
        const std::vector<BoundaryConditionSpec> &specs,
        int boundaryDim);
}