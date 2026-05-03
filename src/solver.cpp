#include <array>
#include <chrono>
#include <cstdlib>
#include <gmsh.h>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <utils.h>
#include <vector>

#include "Mesh.h"
#include "Parallel.h"
#include "Profiling.h"
#include "configParser.h"

namespace solver
{

    bool envEnabled(const char *name)
    {
        const char *value = std::getenv(name);
        if (!value)
            return false;
        return value[0] != '\0' && value[0] != '0';
    }

    bool useHaloOverlapMPI()
    {
        if (Parallel::size() <= 1)
            return false;

        if (envEnabled("DG_DISABLE_HALO_OVERLAP"))
            return false;

        // MPI default is overlap disabled unless explicitly enabled.
        return envEnabled("DG_ENABLE_HALO_OVERLAP");
    }

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

    using prof_clk = profiling::CsvProfiler::Clock;

    struct SolverStageTimers
    {
        long long precomputeMassMatrixUs = 0;
        long long sourceLocatorUs = 0;
        long long observerLocatorUs = 0;
        long long initialHaloUs = 0;
        long long outputUs = 0;
        long long sourceUpdateUs = 0;
        long long updateFluxUs = 0;
        long long precomputeFluxUs = 0;
        long long numStepKernelUs = 0;
        long long haloUs = 0;
        long long residualAssemblyUs = 0;
        long long reductionUs = 0;
        long long observerSamplingUs = 0;
        long long finalObserverExportUs = 0;
    };

    inline prof_clk::time_point tic()
    {
        return prof_clk::now();
    }

    inline long long us(prof_clk::time_point t0)
    {
        return std::chrono::duration_cast<std::chrono::microseconds>(prof_clk::now() - t0).count();
    }

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

    void emitSolverProfile(const std::string &name, const SolverStageTimers &timers,
                           const Mesh &mesh, long long stepsCompleted, long long totalUs)
    {
        profiling::CsvProfiler profiler(name, true);
        profiler.addUs("precomputeMassMatrix", timers.precomputeMassMatrixUs);
        profiler.addUs("sourceLocator", timers.sourceLocatorUs);
        profiler.addUs("observerLocator", timers.observerLocatorUs);
        profiler.addUs("initialHalo", timers.initialHaloUs);
        profiler.addUs("output", timers.outputUs);
        profiler.addUs("sourceUpdate", timers.sourceUpdateUs);
        profiler.addUs("updateFlux", timers.updateFluxUs);
        profiler.addUs("precomputeFlux", timers.precomputeFluxUs);
        profiler.addUs("numStepKernel", timers.numStepKernelUs);
        profiler.addUs("haloExchange", timers.haloUs);
        profiler.addUs("residualAssembly", timers.residualAssemblyUs);
        profiler.addUs("reductions", timers.reductionUs);
        profiler.addUs("observerSampling", timers.observerSamplingUs);
        profiler.addUs("finalObserverExport", timers.finalObserverExportUs);
        profiler.metric("steps", static_cast<double>(stepsCompleted));
        profiler.metric("stored_elements", static_cast<double>(mesh.getElNum()));
        profiler.metric("owned_elements", static_cast<double>(localElEnd(mesh) - localElBegin(mesh)));
        profiler.metric("stored_nodes", static_cast<double>(mesh.getNumNodes()));
        profiler.emit(totalUs);
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
                 std::vector<std::vector<std::vector<double>>> &Flux, double beta,
                 SolverStageTimers *timers = nullptr)
    {
        const int elBegin = localElBegin(mesh);
        const int elEnd = localElEnd(mesh);

        for (int eq = 0; eq < 4; ++eq)
        {
            if (timers)
            {
                auto precomputeStart = tic();
                mesh.precomputeFlux(u[eq], Flux[eq], eq);
                timers->precomputeFluxUs += us(precomputeStart);
            }
            else
            {
                mesh.precomputeFlux(u[eq], Flux[eq], eq);
            }

            if (timers)
            {
                auto kernelStart = tic();
#pragma omp parallel for schedule(static) firstprivate(elFlux, elStiffvector) num_threads(config.numThreads)
                for (int el = elBegin; el < elEnd; ++el)
                {
                    mesh.getElFlux(el, elFlux.data());
                    mesh.getElStiffVector(el, Flux[eq], u[eq], elStiffvector.data());
                    eigen::minus(elStiffvector.data(), elFlux.data(), elNumNodes);
                    eigen::linEq(&mesh.elMassMatrix(el), &elStiffvector[0], &u[eq][el * elNumNodes],
                                 config.timeStep, beta, elNumNodes);
                }
                timers->numStepKernelUs += us(kernelStart);
            }
            else
            {
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
        const bool profileSolver = profiling::phasesEnabled("DG_PROFILE_SOLVER");
        const bool overlapHalo = useHaloOverlapMPI();
        const int elBegin = localElBegin(mesh);
        const int elEnd = localElEnd(mesh);
        SolverStageTimers timers;
        long long stepsCompleted = 0;
        const auto solverProfileStart = tic();

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
        if (profileSolver)
        {
            auto tt = tic();
            mesh.precomputeMassMatrix();
            timers.precomputeMassMatrixUs += us(tt);
        }
        else
        {
            mesh.precomputeMassMatrix();
        }

        /** Source */
        auto tt = tic();
        std::vector<std::vector<int>> srcIndices;
        for (int i = 0; i < config.sources.size(); ++i)
        {
            std::vector<int> indice;
            for (int n = 0; n < mesh.getNumNodes(); n++)
            {
                if (pow(mesh.nodeCoord(n, 0) - config.sources[i].source[1], 2) +
                        pow(mesh.nodeCoord(n, 1) - config.sources[i].source[2], 2) +
                        pow(mesh.nodeCoord(n, 2) - config.sources[i].source[3], 2) <
                    pow(config.sources[i].source[4], 2))
                {
                    indice.push_back(n);
                }
            }
            srcIndices.push_back(indice);
        }
        if (profileSolver)
            timers.sourceLocatorUs += us(tt);

        /** Observer */
        tt = tic();
        std::vector<std::vector<int>> obsIndices;
        std::vector<std::vector<double>> obsPtDistance;
        for (int i = 0; i < config.observers.size(); ++i)
        {
            std::vector<int> indice;
            std::vector<double> dist;
            for (int n = 0; n < mesh.getNumNodes(); n++)
            {
                double distance = sqrt(pow(mesh.nodeCoord(n, 0) - config.observers[i][0], 2) +
                                       pow(mesh.nodeCoord(n, 1) - config.observers[i][1], 2) +
                                       pow(mesh.nodeCoord(n, 2) - config.observers[i][2], 2));
                if (distance < config.observers[i][3])
                {
                    indice.push_back(n);
                    dist.push_back(distance);
                }
            }
            obsIndices.push_back(indice);
            obsPtDistance.push_back(dist);
        }
        if (profileSolver)
            timers.observerLocatorUs += us(tt);

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

        if (profileSolver)
        {
            tt = tic();
            mesh.haloExchange(u);
            timers.initialHaloUs += us(tt);
        }
        else
        {
            mesh.haloExchange(u);
        }

        auto start = std::chrono::system_clock::now();
        for (double t = config.timeStart, step = 0, tDisplay = 0; t <= config.timeEnd;
             t += config.timeStep, tDisplay += config.timeStep, ++step)
        {
            ++stepsCompleted;

            auto start_time = std::chrono::system_clock::now();
            std::vector<double> residual(5, 0.0);
            /**
             *  Savings and prints
             */

            if (tDisplay >= config.timeRate || step == 0)
            {
                tDisplay = 0;
                if (profileSolver)
                    tt = tic();

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

                if (profileSolver)
                    timers.outputUs += us(tt);
            }

            /**
             * Update Source
             */

            if (profileSolver)
                tt = tic();
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
            if (profileSolver)
                timers.sourceUpdateUs += us(tt);

            /**
             * First Order Euler
             */
            if (profileSolver)
            {
                tt = tic();
                mesh.updateFlux(u, Flux, config.v0, config.c0, config.rho0);
                timers.updateFluxUs += us(tt);
            }
            else
            {
                mesh.updateFlux(u, Flux, config.v0, config.c0, config.rho0);
            }

            numStep(mesh, config, u, Flux, 1, profileSolver ? &timers : nullptr);
            if (overlapHalo)
            {
                mesh.haloExchangeBegin(u);
            }
            else
            {
                if (profileSolver)
                    tt = tic();
                mesh.haloExchange(u);
                if (profileSolver)
                    timers.haloUs += us(tt);
            }

            /**
             * Compute residuals
             */
            if (profileSolver)
                tt = tic();
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
            if (profileSolver)
                timers.residualAssemblyUs += us(tt);

            std::array<double, 6> reduceLocal = {residual[0], residual[1], residual[2], residual[3], residual[4],
                                                 static_cast<double>((elEnd - elBegin) * mesh.getElNumNodes())};
            std::array<double, 6> reduceGlobal = {0, 0, 0, 0, 0, 0};

#ifdef DG_USE_MPI
            MPI_Request reduceReq = MPI_REQUEST_NULL;
            bool asyncReduce = (Parallel::size() > 1);
            if (asyncReduce)
            {
                if (profileSolver)
                    tt = tic();
                MPI_Iallreduce(reduceLocal.data(), reduceGlobal.data(), static_cast<int>(reduceLocal.size()),
                               MPI_DOUBLE, MPI_SUM, Parallel::comm(), &reduceReq);
            }
            else
#endif
            {
                if (profileSolver)
                    tt = tic();
                Parallel::allReduce(reduceLocal.data(), reduceGlobal.data(), static_cast<int>(reduceLocal.size()));
                if (profileSolver)
                    timers.reductionUs += us(tt);
            }

            /**
             * get observers value
             * Inverse-distance interpolation. Each rank accumulates local
             * contributions; the global sum is reduced to root for output.
             */
            if (profileSolver)
                tt = tic();
            const int obsCount = static_cast<int>(config.observers.size());
            std::vector<double> localObsPvw(obsCount * 5, 0.0);
            for (int obs = 0; obs < obsCount; ++obs)
            {
                double *localPVW = &localObsPvw[obs * 5]; // p, vx, vy, vz, w_sum
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
            }
            if (profileSolver)
                timers.observerSamplingUs += us(tt);

#ifdef DG_USE_MPI
            if (asyncReduce)
            {
                MPI_Wait(&reduceReq, MPI_STATUS_IGNORE);
                if (profileSolver)
                    timers.reductionUs += us(tt);
            }
#endif
            for (int eq = 0; eq < 5; ++eq)
                residual[eq] = reduceGlobal[eq];
            const int globalDof = static_cast<int>(reduceGlobal[5] + 0.5);
            const double postStepTime = (t + config.timeStep <= config.timeEnd) ? (t + config.timeStep) : config.timeEnd;

            auto end_time = std::chrono::system_clock::now();
            auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            if (rootRank)
            {
                outfile << postStepTime << ";";
                std::cout << std::scientific << postStepTime << "\t";
                for (int eq = 0; eq < residual.size(); ++eq)
                {
                    residual[eq] /= globalDof;
                    std::cout << std::scientific << residual[eq] << "\t";
                    outfile << residual[eq] << ";";
                }
                std::cout << elapsed_time.count() * 1.0e-6 << " s" << std::endl;
                outfile << elapsed_time.count() * 1.0e-6 << std::endl;
            }

            std::vector<double> globalObsPvw(obsCount * 5, 0.0);
#ifdef DG_USE_MPI
            MPI_Request obsReq = MPI_REQUEST_NULL;
            bool asyncObsReduce = (Parallel::size() > 1 && obsCount > 0);
            if (asyncObsReduce)
            {
                if (profileSolver)
                    tt = tic();
                MPI_Iallreduce(localObsPvw.data(), globalObsPvw.data(), obsCount * 5,
                               MPI_DOUBLE, MPI_SUM, Parallel::comm(), &obsReq);
            }
            else
#endif
            if (obsCount > 0)
            {
                if (profileSolver)
                    tt = tic();
                Parallel::allReduce(localObsPvw.data(), globalObsPvw.data(), obsCount * 5);
                if (profileSolver)
                    timers.reductionUs += us(tt);
            }

            if (overlapHalo)
            {
                if (profileSolver)
                    tt = tic();
                mesh.haloExchangeEnd(u);
                if (profileSolver)
                    timers.haloUs += us(tt);
            }

#ifdef DG_USE_MPI
            if (asyncObsReduce)
            {
                MPI_Wait(&obsReq, MPI_STATUS_IGNORE);
                if (profileSolver)
                    timers.reductionUs += us(tt);
            }
#endif

            if (rootRank)
            {
                for (int obs = 0; obs < obsCount; ++obs)
                {
                    const double *globalPVW = &globalObsPvw[obs * 5];
                    if (globalPVW[4] <= 0.0)
                        continue;
                    double p   = globalPVW[0] / globalPVW[4];
                    double vx  = globalPVW[1] / globalPVW[4];
                    double vy  = globalPVW[2] / globalPVW[4];
                    double vz  = globalPVW[3] / globalPVW[4];
                    double rho = p / pow(config.c0, 2);
                    data4wave[obs].push_back(p);
                    obs_outfile[obs] << postStepTime << ";" << rho << ";" << p << ";" << vx << ";" << vy << ";" << vz << std::endl;
                }
            }
        }
        if (rootRank)
        {
            if (profileSolver)
                tt = tic();
            for (int obs = 0; obs < config.observers.size(); ++obs)
            {
                io::writeWave(data4wave[obs], "results/observer_" + std::to_string(obs + 1) + ".wav", 1.0 / config.timeStep, 16, 1, 1);
                io::writeFFT(data4wave[obs],config.timeStep,"results/observer_" + std::to_string(obs + 1));
            }

            outfile.close();
            for (int obs = 0; obs < config.observers.size(); ++obs)
                obs_outfile[obs].close();
            if (profileSolver)
                timers.finalObserverExportUs += us(tt);
        }

        if (profileSolver)
            emitSolverProfile("solver_euler", timers, mesh, stepsCompleted, us(solverProfileStart));
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
        const bool profileSolver = profiling::phasesEnabled("DG_PROFILE_SOLVER");
        const bool overlapHalo = useHaloOverlapMPI();
        const int elBegin = localElBegin(mesh);
        const int elEnd = localElEnd(mesh);
        SolverStageTimers timers;
        long long stepsCompleted = 0;
        const auto solverProfileStart = tic();

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
        if (profileSolver)
        {
            auto tt = tic();
            mesh.precomputeMassMatrix();
            timers.precomputeMassMatrixUs += us(tt);
        }
        else
        {
            mesh.precomputeMassMatrix();
        }
        if (rootRank)
            screen_display::write_string("\t>>> precomputeMassMatrix", BLUE);

        /** Source */
        auto tt = tic();
        std::vector<std::vector<int>> srcIndices;
        for (int i = 0; i < config.sources.size(); ++i)
        {
            std::vector<int> indice;
            for (int n = 0; n < mesh.getNumNodes(); n++)
            {
                if (pow(mesh.nodeCoord(n, 0) - config.sources[i].source[1], 2) +
                        pow(mesh.nodeCoord(n, 1) - config.sources[i].source[2], 2) +
                        pow(mesh.nodeCoord(n, 2) - config.sources[i].source[3], 2) <
                    pow(config.sources[i].source[4], 2))
                {
                    indice.push_back(n);
                }
            }
            srcIndices.push_back(indice);
        }
        if (profileSolver)
            timers.sourceLocatorUs += us(tt);

        /** Observer */
        tt = tic();
        std::vector<std::vector<int>> obsIndices;
        std::vector<std::vector<double>> obsPtDistance;
        for (int i = 0; i < config.observers.size(); ++i)
        {
            std::vector<int> indice;
            std::vector<double> dist;
            for (int n = 0; n < mesh.getNumNodes(); n++)
            {
                double distance = sqrt(pow(mesh.nodeCoord(n, 0) - config.observers[i][0], 2) +
                                       pow(mesh.nodeCoord(n, 1) - config.observers[i][1], 2) +
                                       pow(mesh.nodeCoord(n, 2) - config.observers[i][2], 2));
                if (distance < config.observers[i][3])
                {
                    indice.push_back(n);
                    dist.push_back(distance);
                }
            }
            obsIndices.push_back(indice);
            obsPtDistance.push_back(dist);
        }
        if (profileSolver)
            timers.observerLocatorUs += us(tt);

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

        if (profileSolver)
        {
            tt = tic();
            mesh.haloExchange(u);
            timers.initialHaloUs += us(tt);
        }
        else
        {
            mesh.haloExchange(u);
        }

        auto start = std::chrono::system_clock::now();
        for (double t = config.timeStart, step = 0, tDisplay = 0; t <= config.timeEnd;
             t += config.timeStep, tDisplay += config.timeStep, ++step)
        {
            ++stepsCompleted;
            auto start_time = std::chrono::system_clock::now();
            std::vector<double> residual(5, 0.0);
            /**
             *  Savings and prints
             */
            if (tDisplay >= config.timeRate || step == 0)
            {
                tDisplay = 0;
                if (profileSolver)
                    tt = tic();

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

                if (profileSolver)
                    timers.outputUs += us(tt);
            }

            /** Source */
            if (profileSolver)
                tt = tic();
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
            if (profileSolver)
                timers.sourceUpdateUs += us(tt);

            /**
             * Fourth order Runge-Kutta algorithm
             */
            k1 = k2 = k3 = k4 = u;
            tt = tic();
            /** [1] Step R-K */
            mesh.updateFlux(k1, Flux, config.v0, config.c0, config.rho0);
            if (profileSolver)
                timers.updateFluxUs += us(tt);
            tt = tic();
            numStep(mesh, config, k1, Flux, 0, profileSolver ? &timers : nullptr);
            tt = tic();
            mesh.haloExchange(k1);
            if (profileSolver)
                timers.haloUs += us(tt);
            for (int eq = 0; eq < u.size(); ++eq)
                eigen::plusTimes(k2[eq].data(), k1[eq].data(), 0.5, numNodes);
            tt = tic();
            /** [2] Step R-K */
            mesh.updateFlux(k2, Flux, config.v0, config.c0, config.rho0);
            if (profileSolver)
                timers.updateFluxUs += us(tt);
            tt = tic();
            numStep(mesh, config, k2, Flux, 0, profileSolver ? &timers : nullptr);
            tt = tic();
            mesh.haloExchange(k2);
            if (profileSolver)
                timers.haloUs += us(tt);
            for (int eq = 0; eq < u.size(); ++eq)
                eigen::plusTimes(k3[eq].data(), k2[eq].data(), 0.5, numNodes);
            tt = tic();
            /** [3] Step R-K */
            mesh.updateFlux(k3, Flux, config.v0, config.c0, config.rho0);
            if (profileSolver)
                timers.updateFluxUs += us(tt);
            tt = tic();
            numStep(mesh, config, k3, Flux, 0, profileSolver ? &timers : nullptr);
            tt = tic();
            mesh.haloExchange(k3);
            if (profileSolver)
                timers.haloUs += us(tt);
            for (int eq = 0; eq < u.size(); ++eq)
                eigen::plusTimes(k4[eq].data(), k3[eq].data(), 1, numNodes);
            tt = tic();
            /** [4] Step R-K */
            mesh.updateFlux(k4, Flux, config.v0, config.c0, config.rho0);
            if (profileSolver)
                timers.updateFluxUs += us(tt);
            tt = tic();
            numStep(mesh, config, k4, Flux, 0, profileSolver ? &timers : nullptr);
            tt = tic();
            mesh.haloExchange(k4);
            if (profileSolver)
                timers.haloUs += us(tt);
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

            if (overlapHalo)
            {
                mesh.haloExchangeBegin(u);
            }
            else
            {
                if (profileSolver)
                    tt = tic();
                mesh.haloExchange(u);
                if (profileSolver)
                    timers.haloUs += us(tt);
            }

            if (profileSolver)
                tt = tic();
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
            if (profileSolver)
                timers.residualAssemblyUs += us(tt);

            std::array<double, 6> reduceLocal = {residual[0], residual[1], residual[2], residual[3], residual[4],
                                                 static_cast<double>((elEnd - elBegin) * mesh.getElNumNodes())};
            std::array<double, 6> reduceGlobal = {0, 0, 0, 0, 0, 0};

#ifdef DG_USE_MPI
            MPI_Request reduceReq = MPI_REQUEST_NULL;
            bool asyncReduce = (Parallel::size() > 1);
            if (asyncReduce)
            {
                if (profileSolver)
                    tt = tic();
                MPI_Iallreduce(reduceLocal.data(), reduceGlobal.data(), static_cast<int>(reduceLocal.size()),
                               MPI_DOUBLE, MPI_SUM, Parallel::comm(), &reduceReq);
            }
            else
#endif
            {
                if (profileSolver)
                    tt = tic();
                Parallel::allReduce(reduceLocal.data(), reduceGlobal.data(), static_cast<int>(reduceLocal.size()));
                if (profileSolver)
                    timers.reductionUs += us(tt);
            }

            /**
             * get observers value
             * Inverse-distance interpolation. Each rank accumulates local
             * contributions; the global sum is reduced to root for output.
             */
            if (profileSolver)
                tt = tic();
            const int obsCount = static_cast<int>(config.observers.size());
            std::vector<double> localObsPvw(obsCount * 5, 0.0);
            for (int obs = 0; obs < obsCount; ++obs)
            {
                double *localPVW = &localObsPvw[obs * 5]; // p, vx, vy, vz, w_sum
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
            }
            if (profileSolver)
                timers.observerSamplingUs += us(tt);

#ifdef DG_USE_MPI
            if (asyncReduce)
            {
                MPI_Wait(&reduceReq, MPI_STATUS_IGNORE);
                if (profileSolver)
                    timers.reductionUs += us(tt);
            }
#endif
            for (int eq = 0; eq < 5; ++eq)
                residual[eq] = reduceGlobal[eq];
            const int globalDof = static_cast<int>(reduceGlobal[5] + 0.5);
            const double postStepTime = (t + config.timeStep <= config.timeEnd) ? (t + config.timeStep) : config.timeEnd;

            auto end_time = std::chrono::system_clock::now();
            auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            if (rootRank)
            {
                outfile << postStepTime << ";";
                std::cout << std::scientific << postStepTime << "\t";
                for (int eq = 0; eq < residual.size(); ++eq)
                {
                    residual[eq] /= globalDof;
                    std::cout << std::scientific << residual[eq] << "\t";
                    outfile << residual[eq] << ";";
                }
                std::cout << elapsed_time.count() * 1.0e-6 << " s" << std::endl;
                outfile << elapsed_time.count() * 1.0e-6 << std::endl;
            }

            std::vector<double> globalObsPvw(obsCount * 5, 0.0);
#ifdef DG_USE_MPI
            MPI_Request obsReq = MPI_REQUEST_NULL;
            bool asyncObsReduce = (Parallel::size() > 1 && obsCount > 0);
            if (asyncObsReduce)
            {
                if (profileSolver)
                    tt = tic();
                MPI_Iallreduce(localObsPvw.data(), globalObsPvw.data(), obsCount * 5,
                               MPI_DOUBLE, MPI_SUM, Parallel::comm(), &obsReq);
            }
            else
#endif
            if (obsCount > 0)
            {
                if (profileSolver)
                    tt = tic();
                Parallel::allReduce(localObsPvw.data(), globalObsPvw.data(), obsCount * 5);
                if (profileSolver)
                    timers.reductionUs += us(tt);
            }

            if (overlapHalo)
            {
                if (profileSolver)
                    tt = tic();
                mesh.haloExchangeEnd(u);
                if (profileSolver)
                    timers.haloUs += us(tt);
            }

#ifdef DG_USE_MPI
            if (asyncObsReduce)
            {
                MPI_Wait(&obsReq, MPI_STATUS_IGNORE);
                if (profileSolver)
                    timers.reductionUs += us(tt);
            }
#endif

            if (rootRank)
            {
                for (int obs = 0; obs < obsCount; ++obs)
                {
                    const double *globalPVW = &globalObsPvw[obs * 5];
                    if (globalPVW[4] <= 0.0)
                        continue;
                    double p   = globalPVW[0] / globalPVW[4];
                    double vx  = globalPVW[1] / globalPVW[4];
                    double vy  = globalPVW[2] / globalPVW[4];
                    double vz  = globalPVW[3] / globalPVW[4];
                    double rho = p / pow(config.c0, 2);
                    data4wave[obs].push_back(p);
                    obs_outfile[obs] << postStepTime << ";" << rho << ";" << p << ";" << vx << ";" << vy << ";" << vz << std::endl;
                }
            }
        }
        if (rootRank)
        {
            if (profileSolver)
                tt = tic();
            for (int obs = 0; obs < config.observers.size(); ++obs)
            {
                io::writeWave(data4wave[obs], "results/observer_" + std::to_string(obs + 1) + ".wav", 1.0 / config.timeStep, 16, 1, 1);
                io::writeFFT(data4wave[obs],config.timeStep,"results/observer_" + std::to_string(obs + 1));
            }

            outfile.close();
            for (int obs = 0; obs < config.observers.size(); ++obs)
                obs_outfile[obs].close();
            if (profileSolver)
                timers.finalObserverExportUs += us(tt);
        }

        if (profileSolver)
            emitSolverProfile("solver_rk4", timers, mesh, stepsCompleted, us(solverProfileStart));
    }
}
