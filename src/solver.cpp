#include <chrono>
#include <gmsh.h>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <utils.h>
#include <vector>

#include "Mesh.h"
#include "Parallel.h"
#include "configParser.h"

namespace solver
{

    /**
     * Common variables to all solver
     */
    int elNumNodes;
    int numNodes;
    std::vector<std::string> g_names;
    std::vector<int> elTags;
    std::vector<double> elFlux;
    std::vector<double> elStiffvector;
    std::vector<std::vector<std::vector<double>>> Flux;

    std::vector<std::vector<float>> data4wave;

    int localElBegin(const Mesh &mesh)
    {
#ifdef DG_USE_MPI
        if (Parallel::size() > 1)
            return mesh.getLocalElStart();
#endif
        return 0;
    }

    int localElEnd(const Mesh &mesh)
    {
#ifdef DG_USE_MPI
        if (Parallel::size() > 1)
            return mesh.getLocalElEnd();
#endif
        return mesh.getElNum();
    }

    /**
     * Perform a numerical step: u[t+1] = dt*M^-1*(S[u[t]]-F[u[t]]) + beta*u[t]
     * for all elements in mesh object.
     *
     * @param mesh Mesh object
     * @param config Configuration file
     * @param u Nodal solution vector
     * @param Flux Nodal physical Flux
     * @param beta double coefficient
     */
    void numStep(Mesh &mesh, Config config, std::vector<std::vector<double>> &u,
                 std::vector<std::vector<std::vector<double>>> &Flux, double beta)
    {
        const int elBegin = localElBegin(mesh);
        const int elEnd = localElEnd(mesh);

        for (int eq = 0; eq < 4; ++eq)
        {
            mesh.precomputeFlux(u[eq], Flux[eq], eq);

#pragma omp parallel for schedule(static) firstprivate(elFlux, elStiffvector) num_threads(config.numThreads)
            for (int el = elBegin; el < elEnd; ++el)
            {

                mesh.getElFlux(el, elFlux.data());
                mesh.getElStiffVector(el, Flux[eq], u[eq], elStiffvector.data());
                eigen::minus(elStiffvector.data(), elFlux.data(), elNumNodes);
                eigen::linEq(&mesh.elMassMatrix(el), &elStiffvector[0], &u[eq][el * elNumNodes],
                             config.timeStep, beta, elNumNodes);
            }
        }
    }

    /**
     * Solve using forward explicit scheme. O(h)
     *
     * @param u initial nodal solution vector
     * @param mesh
     * @param config
     */
    void forwardEuler(std::vector<std::vector<double>> &u, Mesh &mesh, Config config)
    {
        const bool rootRank = Parallel::isRoot();
        const int elBegin = localElBegin(mesh);
        const int elEnd = localElEnd(mesh);

        /** Memory allocation */
        elNumNodes = mesh.getElNumNodes();
        numNodes = mesh.getNumNodes();
        elTags = std::vector<int>(&mesh.elTag(0), &mesh.elTag(0) + mesh.getElNum());
        elFlux.resize(elNumNodes);
        elStiffvector.resize(elNumNodes);
        Flux = std::vector<std::vector<std::vector<double>>>(4,
                                                             std::vector<std::vector<double>>(mesh.getNumNodes(),
                                                                                              std::vector<double>(3)));

        /** Gmsh save init */
        gmsh::model::list(g_names);
        int gp_viewTag = gmsh::view::add("Pressure");
        int gv_viewTag = gmsh::view::add("Velocity");
        int grho_viewTag = gmsh::view::add("Density");
        std::vector<std::vector<double>> g_p(mesh.getElNum(), std::vector<double>(elNumNodes));
        std::vector<std::vector<double>> g_rho(mesh.getElNum(), std::vector<double>(elNumNodes));
        std::vector<std::vector<double>> g_v(mesh.getElNum(), std::vector<double>(3 * elNumNodes));

        /** Precomputation */
        mesh.precomputeMassMatrix();

        /** Source */
        std::vector<std::vector<int>> srcIndices;
        for (int i = 0; i < config.sources.size(); ++i)
        {
            std::vector<int> indice;
            for (int n = 0; n < mesh.getNumNodes(); n++)
            {
                std::vector<double> coord, paramCoord;
                int _dim, _tag;
                gmsh::model::mesh::getNode(mesh.getElNodeTags()[n], coord, paramCoord, _dim, _tag);
                if (pow(coord[0] - config.sources[i].source[1], 2) +
                        pow(coord[1] - config.sources[i].source[2], 2) +
                        pow(coord[2] - config.sources[i].source[3], 2) <
                    pow(config.sources[i].source[4], 2))
                {
                    indice.push_back(n);
                }
            }
            srcIndices.push_back(indice);
        }

        /** Observer */
        std::vector<std::vector<int>> obsIndices;
        std::vector<std::vector<double>> obsPtDistance;
        for (int i = 0; i < config.observers.size(); ++i)
        {
            std::vector<int> indice;
            std::vector<double> dist;
            for (int n = 0; n < mesh.getNumNodes(); n++)
            {
                std::vector<double> coord, paramCoord;
                int _dim, _tag;
                gmsh::model::mesh::getNode(mesh.getElNodeTags()[n], coord, paramCoord, _dim, _tag);
                double distance = sqrt(pow(coord[0] - config.observers[i][0], 2) +
                                       pow(coord[1] - config.observers[i][1], 2) +
                                       pow(coord[2] - config.observers[i][2], 2));
                if (distance < config.observers[i][3])
                {
                    indice.push_back(n);
                    dist.push_back(distance);
                }
            }
            obsIndices.push_back(indice);
            obsPtDistance.push_back(dist);
        }

        /**
         * Main Loop : Time iteration
         */
        std::ofstream outfile;
        if (rootRank)
        {
            outfile.open("residuals.csv");
            outfile << "time;res_p;res_rho;res_vx;res_vy;res_vz;elapsed_time" << std::endl;
        }

        std::vector<std::ofstream> obs_outfile(config.observers.size());
        data4wave.clear();
        data4wave.resize(config.observers.size());
        for (int obs = 0; obs < config.observers.size(); ++obs)
        {
            if (rootRank)
            {
                std::string filename = "results/observers" + std::to_string(obs + 1) + ".txt";
                obs_outfile[obs].open(filename.c_str());
                obs_outfile[obs] << "time;pressure;density;velocity_x;velocity_y;velocity_z" << std::endl;
            }
        }

        mesh.haloExchange(u);

        auto start = std::chrono::system_clock::now();
        for (double t = config.timeStart, step = 0, tDisplay = 0; t <= config.timeEnd;
             t += config.timeStep, tDisplay += config.timeStep, ++step)
        {

            auto start_time = std::chrono::system_clock::now();
            std::vector<double> residual(5, 0.0);
            /**
             *  Savings and prints
             */

            if (tDisplay >= config.timeRate || step == 0)
            {
                tDisplay = 0;

/** [1] Copy solution to match GMSH format */
#pragma omp parallel for schedule(static) num_threads(config.numThreads)
                for (int el = 0; el < mesh.getElNum(); ++el)
                {
                    for (int n = 0; n < mesh.getElNumNodes(); ++n)
                    {
                        int elN = el * elNumNodes + n;
                        g_p[el][n] = u[0][elN];
                        g_rho[el][n] = u[0][elN] / (config.c0 * config.c0);
                        g_v[el][3 * n + 0] = u[1][elN];
                        g_v[el][3 * n + 1] = u[2][elN];
                        g_v[el][3 * n + 2] = u[3][elN];
                    }
                }

                /** [2] Print and compute iteration time */
                auto end = std::chrono::system_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);
                if (rootRank)
                {
                    gmsh::logger::write("[" + std::to_string(t) + "/" + std::to_string(config.timeEnd) + "s] Step number : " + std::to_string((int)step) + ", Elapsed time: " + std::to_string(elapsed.count()) + "s");
                    screen_display::write_string("time\t\tres_p\t\tres_rho\t\tres_vx\t\tres_vy\t\tres_vz\t\telapsed time", BOLDBLUE);
                }

                if (Parallel::size() > 1)
                {
                    std::string vtu_filename = "results/result" + std::to_string((int)step) + "_rank" + std::to_string(Parallel::rank()) + ".vtu";
                    mesh.writeVTUb(vtu_filename, u);
                    Parallel::barrier();
                    if (rootRank)
                    {
                        std::string pvtu_filename = "results/result" + std::to_string((int)step) + ".pvtu";
                        mesh.writePVTUb(pvtu_filename);
                    }
                }
                else if (rootRank)
                {
                    std::string vtu_filename = "results/result" + std::to_string((int)step) + ".vtu";
                    mesh.writeVTUb(vtu_filename, u);
                }
            }

            /**
             * Update Source
             */

            for (int src = 0; src < config.sources.size(); ++src)
            {
                if (config.sources[src].formula == "" && config.sources[src].data.empty())
                {
                    double amp = config.sources[src].source[5];
                    double freq = config.sources[src].source[6];
                    double phase = config.sources[src].source[7];
                    double duration = config.sources[src].source[8];
                    if (t < duration)
                        for (int n = 0; n < srcIndices[src].size(); ++n)
                            u[0][srcIndices[src][n]] = amp * sin(2 * M_PI * freq * t + phase);
                }
                else
                {
                    if (config.sources[src].data.empty())
                    {
                        double duration = config.sources[src].source[5];
                        if (t < duration)
                            for (int n = 0; n < srcIndices[src].size(); ++n)
                                u[0][srcIndices[src][n]] = config.sources[src].value(t);
                    }
                    else
                    {
                        for (int n = 0; n < srcIndices[src].size(); ++n)
                            u[0][srcIndices[src][n]] = config.sources[src].interpolate_value(t);
                    }
                }
            }

            /**
             * First Order Euler
             */
            mesh.updateFlux(u, Flux, config.v0, config.c0, config.rho0);
            numStep(mesh, config, u, Flux, 1);
            mesh.haloExchange(u);

            /**
             * Compute residuals
             */
#pragma omp parallel for schedule(static) num_threads(config.numThreads)
            for (int el = elBegin; el < elEnd; ++el)
            {
                for (int n = 0; n < mesh.getElNumNodes(); ++n)
                {
                    int elN = el * elNumNodes + n;
#pragma omp atomic update
                    residual[0] += pow(g_p[el][n] - u[0][elN], 2);
#pragma omp atomic update
                    residual[1] += pow(g_rho[el][n] - u[0][elN] / (config.c0 * config.c0), 2);
#pragma omp atomic update
                    residual[2] += pow(g_v[el][3 * n + 0] - u[1][elN], 2);
#pragma omp atomic update
                    residual[3] += pow(g_v[el][3 * n + 1] - u[2][elN], 2);
#pragma omp atomic update
                    residual[4] += pow(g_v[el][3 * n + 2] - u[3][elN], 2);
                }
            }

            std::vector<double> residualGlobal(residual.size(), 0.0);
            Parallel::allReduce(residual.data(), residualGlobal.data(), static_cast<int>(residual.size()));
            residual.swap(residualGlobal);

            // Global #DOFs = sum over ranks of (elEnd - elBegin) * elNumNodes.
            // We allreduce that count too so root has the correct divisor in
            // partitioned mode (where mesh.getElNum() is local+halo, not global).
            int localDof  = (elEnd - elBegin) * mesh.getElNumNodes();
            int globalDof = Parallel::allReduceScalar<int>(localDof);

            auto end_time = std::chrono::system_clock::now();
            auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            if (rootRank)
            {
                outfile << t << ";";
                std::cout << std::scientific << t << "\t";
                for (int eq = 0; eq < residual.size(); ++eq)
                {
                    residual[eq] /= globalDof;
                    std::cout << std::scientific << residual[eq] << "\t";
                    outfile << residual[eq] << ";";
                }
                std::cout << elapsed_time.count() * 1.0e-6 << " s" << std::endl;
                outfile << elapsed_time.count() * 1.0e-6 << std::endl;
            }

            /**
             * get observers value
             * Inverse-distance interpolation. Each rank accumulates local
             * contributions; the global sum is reduced to root for output.
             */
            for (int obs = 0; obs < config.observers.size(); ++obs)
            {
                double localPVW[5] = {0, 0, 0, 0, 0}; // p, vx, vy, vz, w_sum
                for (int n = 0; n < obsIndices[obs].size(); ++n)
                {
                    int nodeIdx = obsIndices[obs][n];
                    int elIdx   = nodeIdx / elNumNodes;
                    if (elIdx < elBegin || elIdx >= elEnd) continue;
                    double w = 1.0 / (pow(obsPtDistance[obs][n], 2) + 1.0e-12);
                    localPVW[0] += u[0][nodeIdx] * w;
                    localPVW[1] += u[1][nodeIdx] * w;
                    localPVW[2] += u[2][nodeIdx] * w;
                    localPVW[3] += u[3][nodeIdx] * w;
                    localPVW[4] += w;
                }
                double globalPVW[5] = {0, 0, 0, 0, 0};
                Parallel::allReduce(localPVW, globalPVW, 5);
                if (rootRank && globalPVW[4] > 0.0)
                {
                    double p   = globalPVW[0] / globalPVW[4];
                    double vx  = globalPVW[1] / globalPVW[4];
                    double vy  = globalPVW[2] / globalPVW[4];
                    double vz  = globalPVW[3] / globalPVW[4];
                    double rho = p / pow(config.c0, 2);
                    data4wave[obs].push_back(p);
                    obs_outfile[obs] << t << ";" << rho << ";" << p << ";" << vx << ";" << vy << ";" << vz << std::endl;
                }
            }
        }
        if (rootRank)
        {
            for (int obs = 0; obs < config.observers.size(); ++obs)
            {
                io::writeWave(data4wave[obs], "results/observer_" + std::to_string(obs + 1) + ".wav", 1.0 / config.timeStep, 16, 1, 1);
                io::writeFFT(data4wave[obs],config.timeStep,"results/observer_" + std::to_string(obs + 1));
            }

            outfile.close();
            for (int obs = 0; obs < config.observers.size(); ++obs)
                obs_outfile[obs].close();
        }
    }

    /**
     * Solve using explicit Runge-Kutta integration method. O(h^4)
     *
     * @param u initial nodal solution vector
     * @param mesh
     * @param config
     */
    void rungeKutta(std::vector<std::vector<double>> &u, Mesh &mesh, Config config)
    {
        const bool rootRank = Parallel::isRoot();
        const int elBegin = localElBegin(mesh);
        const int elEnd = localElEnd(mesh);

        /** Memory allocation */
        elNumNodes = mesh.getElNumNodes();
        numNodes = mesh.getNumNodes();
        elTags = std::vector<int>(&mesh.elTag(0), &mesh.elTag(0) + mesh.getElNum());
        elFlux.resize(elNumNodes);
        elStiffvector.resize(elNumNodes);
        std::vector<std::vector<double>> k1, k2, k3, k4;
        Flux = std::vector<std::vector<std::vector<double>>>(4,
                                                             std::vector<std::vector<double>>(mesh.getNumNodes(),
                                                                                              std::vector<double>(3)));

        /** Gmsh save init */
        gmsh::model::list(g_names);
        int gp_viewTag = gmsh::view::add("Pressure");
        int gv_viewTag = gmsh::view::add("Velocity");
        int grho_viewTag = gmsh::view::add("Density");
        std::vector<std::vector<double>> g_p(mesh.getElNum(), std::vector<double>(elNumNodes));
        std::vector<std::vector<double>> g_rho(mesh.getElNum(), std::vector<double>(elNumNodes));
        std::vector<std::vector<double>> g_v(mesh.getElNum(), std::vector<double>(3 * elNumNodes));

        /** Precomputation (constants over time) */
        if (rootRank)
            screen_display::write_string("\t>>> Precomputation", BLUE);
        mesh.precomputeMassMatrix();
        if (rootRank)
            screen_display::write_string("\t>>> precomputeMassMatrix", BLUE);

        /** Source */
        std::vector<std::vector<int>> srcIndices;
        for (int i = 0; i < config.sources.size(); ++i)
        {
            std::vector<int> indice;
            for (int n = 0; n < mesh.getNumNodes(); n++)
            {
                std::vector<double> coord, paramCoord;
                int _dim, _tag;
                gmsh::model::mesh::getNode(mesh.getElNodeTags()[n], coord, paramCoord, _dim, _tag);
                if (pow(coord[0] - config.sources[i].source[1], 2) +
                        pow(coord[1] - config.sources[i].source[2], 2) +
                        pow(coord[2] - config.sources[i].source[3], 2) <
                    pow(config.sources[i].source[4], 2))
                {
                    indice.push_back(n);
                }
            }
            srcIndices.push_back(indice);
        }

        /** Observer */
        std::vector<std::vector<int>> obsIndices;
        std::vector<std::vector<double>> obsPtDistance;
        for (int i = 0; i < config.observers.size(); ++i)
        {
            std::vector<int> indice;
            std::vector<double> dist;
            for (int n = 0; n < mesh.getNumNodes(); n++)
            {
                std::vector<double> coord, paramCoord;
                int _dim, _tag;
                gmsh::model::mesh::getNode(mesh.getElNodeTags()[n], coord, paramCoord, _dim, _tag);
                double distance = sqrt(pow(coord[0] - config.observers[i][0], 2) +
                                       pow(coord[1] - config.observers[i][1], 2) +
                                       pow(coord[2] - config.observers[i][2], 2));
                if (distance < config.observers[i][3])
                {
                    indice.push_back(n);
                    dist.push_back(distance);
                }
            }
            obsIndices.push_back(indice);
            obsPtDistance.push_back(dist);
        }

        // for (int i = 0; i < obsIndices.size(); i++)
        // {
        //     for (int j = 0; j < obsIndices[i].size(); j++)
        //     {
        //         std::cout << obsIndices[i][j] << "\t";
        //     }
        //     std::cout << std::endl;
        // }
        // getchar();

        /**
         * Main Loop : Time iteration
         */

        std::ofstream outfile;
        if (rootRank)
        {
            outfile.open("residuals.csv");
            outfile << "time;res_p;res_rho;res_vx;res_vy;res_vz;elapsed_time" << std::endl;
        }

        std::vector<std::ofstream> obs_outfile(config.observers.size());
        data4wave.clear();
        data4wave.resize(config.observers.size());
        for (int obs = 0; obs < config.observers.size(); ++obs)
        {
            if (rootRank)
            {
                std::string filename = "results/observers" + std::to_string(obs + 1) + ".txt";
                obs_outfile[obs].open(filename.c_str());
                obs_outfile[obs] << "time;density;pressure;velocity_x;velocity_y;velocity_z" << std::endl;
            }
        }

        mesh.haloExchange(u);

        // Cumulative profiling counters (microseconds), printed at the end.
        long long t_halo_us = 0;
        long long t_updFlux_us = 0;
        long long t_numStep_us = 0;
        long long t_residual_us = 0;
        long long t_obs_io_us = 0;

        auto start = std::chrono::system_clock::now();
        for (double t = config.timeStart, step = 0, tDisplay = 0; t <= config.timeEnd;
             t += config.timeStep, tDisplay += config.timeStep, ++step)
        {
            auto start_time = std::chrono::system_clock::now();
            std::vector<double> residual(5, 0.0);
            /**
             *  Savings and prints
             */
            if (tDisplay >= config.timeRate || step == 0)
            {
                tDisplay = 0;

/** [1] Copy solution to match GMSH format */
// #pragma omp parallel for
#pragma omp parallel for schedule(static) num_threads(config.numThreads)
                for (int el = 0; el < mesh.getElNum(); ++el)
                {
                    for (int n = 0; n < mesh.getElNumNodes(); ++n)
                    {
                        int elN = el * elNumNodes + n;
                        g_p[el][n] = u[0][elN];
                        g_rho[el][n] = u[0][elN] / (config.c0 * config.c0);
                        g_v[el][3 * n + 0] = u[1][elN];
                        g_v[el][3 * n + 1] = u[2][elN];
                        g_v[el][3 * n + 2] = u[3][elN];
                    }
                }
                // gmsh::view::addModelData(gp_viewTag, step, g_names[0], "ElementNodeData", elTags, g_p, t, 1);
                // gmsh::view::addModelData(grho_viewTag, step, g_names[0], "ElementNodeData", elTags, g_rho, t, 1);
                // gmsh::view::addModelData(gv_viewTag, step, g_names[0], "ElementNodeData", elTags, g_v, t, 3);

                /** [2] Print and compute iteration time */
                auto end = std::chrono::system_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start);
                if (rootRank)
                {
                    gmsh::logger::write("[" + std::to_string(t) + "/" + std::to_string(config.timeEnd) + "s] Step number : " + std::to_string((int)step) + ", Elapsed time: " + std::to_string(elapsed.count()) + "s");
                    screen_display::write_string("time\t\tres_p\t\tres_rho\t\tres_vx\t\tres_vy\t\tres_vz\t\telapsed time", BOLDBLUE);
                }

                if (Parallel::size() > 1)
                {
                    std::string vtu_filename = "results/result" + std::to_string((int)step) + "_rank" + std::to_string(Parallel::rank()) + ".vtu";
                    mesh.writeVTUb(vtu_filename, u);
                    Parallel::barrier();
                    if (rootRank)
                    {
                        std::string pvtu_filename = "results/result" + std::to_string((int)step) + ".pvtu";
                        mesh.writePVTUb(pvtu_filename);
                    }
                }
                else if (rootRank)
                {
                    std::string vtu_filename = "results/result" + std::to_string((int)step) + ".vtu";
                    mesh.writeVTUb(vtu_filename, u);
                }
            }

            /** Source */
            for (int src = 0; src < config.sources.size(); ++src)
            {
                if (config.sources[src].formula == "" && config.sources[src].data.empty())
                {
                    double amp = config.sources[src].source[5];
                    double freq = config.sources[src].source[6];
                    double phase = config.sources[src].source[7];
                    double duration = config.sources[src].source[8];
                    if (t < duration)
                        for (int n = 0; n < srcIndices[src].size(); ++n)
                            u[0][srcIndices[src][n]] = amp * sin(2 * M_PI * freq * t + phase);
                }
                else
                {
                    if (config.sources[src].data.empty())
                    {
                        double duration = config.sources[src].source[5];
                        if (t < duration)
                            for (int n = 0; n < srcIndices[src].size(); ++n)
                                u[0][srcIndices[src][n]] = config.sources[src].value(t);
                    }
                    else
                    {
                        for (int n = 0; n < srcIndices[src].size(); ++n)
                            u[0][srcIndices[src][n]] = config.sources[src].interpolate_value(t);
                    }
                }
            }

            /**
             * Fourth order Runge-Kutta algorithm
             */
            using clk = std::chrono::system_clock;
            auto tic = [](){ return clk::now(); };
            auto us  = [](clk::time_point t0){
                return std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t0).count();
            };

            k1 = k2 = k3 = k4 = u;
            auto tt = tic();
            /** [1] Step R-K */
            mesh.updateFlux(k1, Flux, config.v0, config.c0, config.rho0);
            t_updFlux_us += us(tt); tt = tic();
            numStep(mesh, config, k1, Flux, 0);
            t_numStep_us += us(tt); tt = tic();
            mesh.haloExchange(k1);
            t_halo_us += us(tt);
            for (int eq = 0; eq < u.size(); ++eq)
                eigen::plusTimes(k2[eq].data(), k1[eq].data(), 0.5, numNodes);
            tt = tic();
            /** [2] Step R-K */
            mesh.updateFlux(k2, Flux, config.v0, config.c0, config.rho0);
            t_updFlux_us += us(tt); tt = tic();
            numStep(mesh, config, k2, Flux, 0);
            t_numStep_us += us(tt); tt = tic();
            mesh.haloExchange(k2);
            t_halo_us += us(tt);
            for (int eq = 0; eq < u.size(); ++eq)
                eigen::plusTimes(k3[eq].data(), k2[eq].data(), 0.5, numNodes);
            tt = tic();
            /** [3] Step R-K */
            mesh.updateFlux(k3, Flux, config.v0, config.c0, config.rho0);
            t_updFlux_us += us(tt); tt = tic();
            numStep(mesh, config, k3, Flux, 0);
            t_numStep_us += us(tt); tt = tic();
            mesh.haloExchange(k3);
            t_halo_us += us(tt);
            for (int eq = 0; eq < u.size(); ++eq)
                eigen::plusTimes(k4[eq].data(), k3[eq].data(), 1, numNodes);
            tt = tic();
            /** [4] Step R-K */
            mesh.updateFlux(k4, Flux, config.v0, config.c0, config.rho0);
            t_updFlux_us += us(tt); tt = tic();
            numStep(mesh, config, k4, Flux, 0);
            t_numStep_us += us(tt); tt = tic();
            mesh.haloExchange(k4);
            t_halo_us += us(tt);
            /** Concat results of R-K iterations */
            // #pragma omp parallel for
            for (int eq = 0; eq < u.size(); ++eq)
            {
                for (int i = 0; i < numNodes; ++i)
                {
                    // #pragma omp atomic
                    u[eq][i] += (k1[eq][i] + 2 * k2[eq][i] + 2 * k3[eq][i] + k4[eq][i]) / 6.0;
                }
            }

            tt = tic();
            mesh.haloExchange(u);
            t_halo_us += us(tt);

#pragma omp parallel for schedule(static) num_threads(config.numThreads)
            for (int el = elBegin; el < elEnd; ++el)
            {
                for (int n = 0; n < mesh.getElNumNodes(); ++n)
                {
                    int elN = el * elNumNodes + n;
#pragma omp atomic update
                    residual[0] += pow(g_p[el][n] - u[0][elN], 2);
#pragma omp atomic update
                    residual[1] += pow(g_rho[el][n] - u[0][elN] / (config.c0 * config.c0), 2);
#pragma omp atomic update
                    residual[2] += pow(g_v[el][3 * n + 0] - u[1][elN], 2);
#pragma omp atomic update
                    residual[3] += pow(g_v[el][3 * n + 1] - u[2][elN], 2);
#pragma omp atomic update
                    residual[4] += pow(g_v[el][3 * n + 2] - u[3][elN], 2);
                }
            }

            std::vector<double> residualGlobal(residual.size(), 0.0);
            Parallel::allReduce(residual.data(), residualGlobal.data(), static_cast<int>(residual.size()));
            residual.swap(residualGlobal);

            int localDof  = (elEnd - elBegin) * mesh.getElNumNodes();
            int globalDof = Parallel::allReduceScalar<int>(localDof);

            auto end_time = std::chrono::system_clock::now();
            auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            if (rootRank)
            {
                outfile << t << ";";
                std::cout << std::scientific << t << "\t";
                for (int eq = 0; eq < residual.size(); ++eq)
                {
                    residual[eq] /= globalDof;
                    std::cout << std::scientific << residual[eq] << "\t";
                    outfile << residual[eq] << ";";
                }
                std::cout << elapsed_time.count() * 1.0e-6 << " s" << std::endl;
                outfile << elapsed_time.count() * 1.0e-6 << std::endl;
            }
            /**
             * get observers value
             * Inverse-distance interpolation. Each rank accumulates local
             * contributions; the global sum is reduced to root for output.
             */
            for (int obs = 0; obs < config.observers.size(); ++obs)
            {
                double localPVW[5] = {0, 0, 0, 0, 0}; // p, vx, vy, vz, w_sum
                for (int n = 0; n < obsIndices[obs].size(); ++n)
                {
                    int nodeIdx = obsIndices[obs][n];
                    int elIdx   = nodeIdx / elNumNodes;
                    if (elIdx < elBegin || elIdx >= elEnd) continue;
                    double w = 1.0 / (pow(obsPtDistance[obs][n], 2) + 1.0e-12);
                    localPVW[0] += u[0][nodeIdx] * w;
                    localPVW[1] += u[1][nodeIdx] * w;
                    localPVW[2] += u[2][nodeIdx] * w;
                    localPVW[3] += u[3][nodeIdx] * w;
                    localPVW[4] += w;
                }
                double globalPVW[5] = {0, 0, 0, 0, 0};
                Parallel::allReduce(localPVW, globalPVW, 5);
                if (rootRank && globalPVW[4] > 0.0)
                {
                    double p   = globalPVW[0] / globalPVW[4];
                    double vx  = globalPVW[1] / globalPVW[4];
                    double vy  = globalPVW[2] / globalPVW[4];
                    double vz  = globalPVW[3] / globalPVW[4];
                    double rho = p / pow(config.c0, 2);
                    data4wave[obs].push_back(p);
                    obs_outfile[obs] << t << ";" << rho << ";" << p << ";" << vx << ";" << vy << ";" << vz << std::endl;
                }
            }
        }
        if (rootRank)
        {
            for (int obs = 0; obs < config.observers.size(); ++obs)
            {
                io::writeWave(data4wave[obs], "results/observer_" + std::to_string(obs + 1) + ".wav", 1.0 / config.timeStep, 16, 1, 1);
                io::writeFFT(data4wave[obs],config.timeStep,"results/observer_" + std::to_string(obs + 1));
            }

            outfile.close();
            for (int obs = 0; obs < config.observers.size(); ++obs)
                obs_outfile[obs].close();
        }

        // Profiling summary — printed by every rank so we can compare ranks.
        long long t_total = t_halo_us + t_updFlux_us + t_numStep_us;
        std::cout << "[MPI rank " << Parallel::rank() << "] RK4 hot-loop totals (s): "
                  << "halo=" << t_halo_us / 1e6
                  << " updFlux=" << t_updFlux_us / 1e6
                  << " numStep=" << t_numStep_us / 1e6
                  << " total=" << t_total / 1e6
                  << " (halo share=" << (t_total > 0 ? 100.0 * t_halo_us / t_total : 0.0) << "%)"
                  << std::endl;
    }
}
