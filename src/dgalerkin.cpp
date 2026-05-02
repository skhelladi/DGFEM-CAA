#include <cstdio>
#include <errno.h>
#include <gmsh.h>
#include <iostream>
#include <omp.h>

#include "Mesh.h"
#include "Parallel.h"
#include "configParser.h"
#include "solver.h"

int main(int argc, char **argv)
{
    Parallel::init(&argc, &argv);

    if (argc != 2)
    {
        Parallel::finalize();
        return E2BIG;
    }
    std::string config_name = argv[1];

    gmsh::initialize();
    // Suppress gmsh terminal output on non-root ranks to avoid log flooding
    gmsh::option::setNumber("General.Terminal", Parallel::isRoot() ? 1.0 : 0.0);

    Config config;

    if (fileExtension(config_name) == "conf")
        config = config::parseConfig(config_name);
    if (fileExtension(config_name) == "json")
        config = config::parseJSON(config_name);    

    gmsh::logger::write("Config loaded : " + config_name);

    Mesh mesh(config);

    /**
     * Initialize the solution:
     */
    std::vector<std::vector<double>> u(4, std::vector<double>(mesh.getNumNodes(), 0));
    for (int i = 0; i < config.initConditions.size(); ++i)
    {
        double x = config.initConditions[i][1];
        double y = config.initConditions[i][2];
        double z = config.initConditions[i][3];
        double size = config.initConditions[i][4];
        double amp = config.initConditions[i][5];

        for (int n = 0; n < mesh.getNumNodes(); n++)
        {
            std::vector<double> coord, paramCoord;
            int _dim, _tag;
            gmsh::model::mesh::getNode(mesh.getElNodeTags()[n], coord, paramCoord, _dim, _tag);
            u[0][n] += amp * exp(-((coord[0] - x) * (coord[0] - x) +
                                   (coord[1] - y) * (coord[1] - y) +
                                   (coord[2] - z) * (coord[2] - z)) /
                                 size);
        }
    }

    /**
     * Start solver
     */
    if (config.timeIntMethod == "Euler1")
        solver::forwardEuler(u, mesh, config);
    else if (config.timeIntMethod == "Runge-Kutta")
        solver::rungeKutta(u, mesh, config);
    else Fatal_Error("Time integration method error")    

    if (Parallel::isRoot())
        mesh.writePVD("results.pvd");

    gmsh::finalize();
    Parallel::finalize();

    return EXIT_SUCCESS;
}
