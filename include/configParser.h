#include "eqEdit.h"
#include <fstream>
#include <map>
#include <string>

#include <nlohmann/json.hpp>


#ifndef DGALERKIN_CONFIG_H
#define DGALERKIN_CONFIG_H

using json = nlohmann::json;
using ordered_json=nlohmann::ordered_json;


//! ////////////////////////////////////////////////////////////////
class Sources
{
public:
    Sources(std::string f, std::vector<double> s, std::vector<std::vector<double>> d = {})
    {
        formula = f;
        source = s;
        data = d;
    }
    // Sources(std::vector<std::vector<double>> d) {data=d;}
    Sources() {}

    std::string formula = "";
    std::vector<double> source;
    EQ_EDIT expression;
    double value(double t)
    {
        bool go = (formula != "");
        return expression.value(go, formula, {{"t", t}});
    }

    double interpolate_value(double t)
    {
        // linear interpolation using dechotomy method to find interpolation bounds in the data table

        size_t lo = 0, hi = (data.size() - 1), m(0);
        double val(0), tmin(data[lo][0]),tmax(data[hi][0]);
        while (lo <= hi && t >= tmin && t <= tmax)
        {
            m=(lo+hi)/2;

            if(t<data[m][0]) hi=m;
            else lo = m;

            if(hi==lo+1) 
            {
                val = data[lo][1] + ((data[hi][1] - data[lo][1]) / (data[hi][0] - data[lo][0])) * (t - data[lo][0]);
                return val;
            }
        }
        return 0.0;
    }

    std::vector<std::vector<double>> data;
};

// class Observers
// {
// public:
//     std::vector<double> observer;
// };

struct BoundaryConditionSpec
{
    std::string physicalName;
    std::string type;
    double value = 0.0;
};

struct Config
{
    ordered_json jsonData;
    
    std::string meshFileName;
    std::string partitionManifestFile;
    std::string partitionPackageFile;
    std::string partitionMode = "gmsh";
    std::string partitionCommand;
    
    // Initial, final time and time step(t>0)
    double timeStart = 0;
    double timeEnd = 1;
    double timeStep = 0.1;
    double timeRate = 0.1;

    // Element Type:
    std::string elementType = "Lagrange";

    // Time integration method
    std::string timeIntMethod = "Euler1";

    // Boundary condition
    // key : physical group Tag
    // value : tuple<BCType, BCValue>
    std::map<int, std::pair<std::string, double>> physBCs;
    std::vector<BoundaryConditionSpec> pendingPhysBCs;

    // Mean flow parameters
    std::vector<double> v0 = {0, 0, 0};
    double rho0 = 1;
    double c0 = 1;

    // Number of threads
    int numThreads = 1;

    // Sources
    // struct sources
    // {
    //     std::vector<std::vector<double>> source;
    //     std::string source_expression = "";
    // };
    std::vector<Sources> sources;
    // Obsertvers
    std::vector<std::vector<double>> observers;

    // Initial conditions
    std::vector<std::vector<double>> initConditions;

    // Save file
    // std::string saveFile = "results.msh";
};

namespace config
{
    Config parseConfig(std::string name);
    Config parseJSON(std::string name);
}

#endif
