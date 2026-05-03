#pragma once

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "Parallel.h"

namespace profiling
{
    inline bool envEnabled(const char *name)
    {
        return name != nullptr && std::getenv(name) != nullptr;
    }

    inline bool phasesEnabled(const char *legacyEnv = nullptr)
    {
        return envEnabled("DG_PROFILE_PHASES") || envEnabled(legacyEnv);
    }

    class CsvProfiler
    {
    public:
        using Clock = std::chrono::steady_clock;
        using TimePoint = Clock::time_point;

        CsvProfiler(std::string prefix, bool enabled)
            : m_prefix(std::move(prefix)), m_enabled(enabled), m_rank(Parallel::rank()), m_start(Clock::now())
        {
        }

        bool enabled() const
        {
            return m_enabled;
        }

        TimePoint tic() const
        {
            return Clock::now();
        }

        void add(const std::string &label, TimePoint t0)
        {
            if (!m_enabled)
                return;
            m_sections[label] += std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - t0).count();
        }

        void addUs(const std::string &label, long long us)
        {
            if (!m_enabled)
                return;
            m_sections[label] += us;
        }

        void metric(const std::string &name, double value)
        {
            if (!m_enabled)
                return;
            m_metrics[name] = value;
        }

        void emit(long long totalUsOverride = -1) const
        {
            if (!m_enabled)
                return;

            static bool headerPrinted = false;
            if (m_rank == 0 && !headerPrinted)
            {
                std::cerr << "PROF_CSV,rank,section,seconds,share_percent" << std::endl;
                headerPrinted = true;
            }

            long long totalUs = totalUsOverride >= 0
                ? totalUsOverride
                : std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - m_start).count();
            long long sumUs = 0;
            for (const auto &entry : m_sections)
                sumUs += entry.second;

            for (const auto &entry : m_sections)
            {
                const double seconds = entry.second / 1e6;
                const double share = sumUs > 0 ? 100.0 * static_cast<double>(entry.second) / static_cast<double>(sumUs) : 0.0;
                std::cerr << "PROF_CSV," << m_rank << "," << scoped(entry.first)
                          << "," << seconds << "," << share << std::endl;
            }

            std::cerr << "PROF_CSV," << m_rank << "," << scoped("TOTAL")
                      << "," << totalUs / 1e6 << ",100" << std::endl;

            for (const auto &entry : m_metrics)
            {
                std::cerr << "PROF_CSV," << m_rank << "," << scoped(entry.first)
                          << "," << entry.second << ",-1" << std::endl;
            }
        }

    private:
        std::string scoped(const std::string &label) const
        {
            return m_prefix + "::" + label;
        }

        std::string m_prefix;
        bool m_enabled;
        int m_rank;
        TimePoint m_start;
        std::map<std::string, long long> m_sections;
        std::map<std::string, double> m_metrics;
    };
}