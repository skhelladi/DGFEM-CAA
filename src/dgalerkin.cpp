#include <cstdio>
#include <errno.h>
#include <cstdlib>
#include <fstream>
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

    std::ofstream mpiNullStream;
    std::streambuf *savedCoutBuf = nullptr;
    std::streambuf *savedCerrBuf = nullptr;
    std::streambuf *savedClogBuf = nullptr;
    const bool mpiVerboseAllRanks = (std::getenv("DG_MPI_VERBOSE_ALL_RANKS") != nullptr);
    if (Parallel::size() > 1 && !Parallel::isRoot() && !mpiVerboseAllRanks)
    {
        mpiNullStream.open("/dev/null");
        if (mpiNullStream.is_open())
        {
            savedCoutBuf = std::cout.rdbuf();
            savedCerrBuf = std::cerr.rdbuf();
            savedClogBuf = std::clog.rdbuf();
            std::cout.rdbuf(mpiNullStream.rdbuf());
            std::cerr.rdbuf(mpiNullStream.rdbuf());
            std::clog.rdbuf(mpiNullStream.rdbuf());
        }
    }
    else if (Parallel::isRoot())
    {
        // Keep root logs ordered and flushed line-by-line in MPI runs.
        std::cout.setf(std::ios::unitbuf);
        std::cerr.setf(std::ios::unitbuf);
    }

    if (argc != 2)
    {
        Parallel::finalize();
        return E2BIG;
    }
    std::string config_name = argv[1];

    gmsh::initialize();
    // Suppress gmsh terminal output on non-root ranks to avoid log flooding
    gmsh::option::setNumber("General.Terminal", (Parallel::isRoot() || mpiVerboseAllRanks) ? 1.0 : 0.0);

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
                        const double dx = mesh.nodeCoord(n, 0) - x;
                        const double dy = mesh.nodeCoord(n, 1) - y;
                        const double dz = mesh.nodeCoord(n, 2) - z;
                        u[0][n] += amp * exp(-(dx * dx +
                                                                     dy * dy +
                                                                     dz * dz) /
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

    if (savedCoutBuf)
        std::cout.rdbuf(savedCoutBuf);
    if (savedCerrBuf)
        std::cerr.rdbuf(savedCerrBuf);
    if (savedClogBuf)
        std::clog.rdbuf(savedClogBuf);

    gmsh::finalize();
    Parallel::finalize();

    return EXIT_SUCCESS;
}
