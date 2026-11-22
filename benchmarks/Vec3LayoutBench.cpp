#define GFXMATH_VEC3
#define GFXMATH_VEC4
#include "GfxMath.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#if defined(_MSC_VER)
#define BENCH_NOINLINE __declspec(noinline)
#else
#define BENCH_NOINLINE __attribute__((noinline))
#endif

namespace
{
    using Clock = std::chrono::steady_clock;
    constexpr std::size_t sampleCount = 7;
    constexpr std::size_t hotCount = 512;
    constexpr std::size_t largeCount = 4 * 1024 * 1024;
    enum class Kernel { dot, transform, indexedDistance };

    template <typename V> struct Buffers
    {
        using T = typename V::Scalar;
        std::vector<V> points;
        std::vector<V> transformed;
        std::vector<T> scalars;

        explicit Buffers(std::size_t count) : points(count), transformed(count), scalars(count)
        {
            std::mt19937 random(177);
            const auto coordinate = [&] { return T(random() & 1023u) / T(1024); };
            for (auto& point : points)
            {
                const auto x = coordinate();
                const auto y = coordinate();
                point = V(x, y, coordinate());
            }
        }
    };

    template <Kernel Work, typename V>
    BENCH_NOINLINE void run(Buffers<V>& buffers, std::span<const std::uint32_t> indices)
    {
        using T = typename V::Scalar;
        using V4 = dex::vec4<T>;
        const auto count = buffers.points.size();
        if constexpr (Work == Kernel::dot)
        {
            const V normal(T(0.75), T(-0.25), T(0.5));
            for (std::size_t index = 0; index < count; ++index)
                buffers.scalars[index] = buffers.points[index].dot(normal);
        }
        else if constexpr (Work == Kernel::transform)
        {
            const dex::Matrix4<T> transform(
                V4(1, T(0.125), T(-0.25), 0), V4(T(-0.5), 2, T(0.25), 0),
                V4(T(0.25), T(-0.125), T(0.5), 0), V4(2, -3, 4, 1));
            for (std::size_t index = 0; index < count; ++index)
                buffers.transformed[index] = transform * buffers.points[index];
        }
        else
        {
            const V origin(T(-0.25), T(0.5), T(0.75));
            for (std::size_t index = 0; index < count; ++index)
                buffers.scalars[index] = buffers.points[indices[index]].distanceSquared(origin);
        }
    }

    template <Kernel Work, typename V>
    BENCH_NOINLINE double checksum(const Buffers<V>& buffers)
    {
        if constexpr (Work == Kernel::transform)
        {
            double result = 0;
            for (const auto& point : buffers.transformed)
                result += double(point.x) + 2 * double(point.y) + 3 * double(point.z);
            return result;
        }
        else
        {
            return std::accumulate(buffers.scalars.begin(), buffers.scalars.end(), 0.0);
        }
    }

    template <Kernel Work, typename V>
    double measure(Buffers<V>& buffers, std::span<const std::uint32_t> indices, std::size_t passes)
    {
        const auto start = Clock::now();
        for (std::size_t pass = 0; pass < passes; ++pass)
            run<Work>(buffers, indices);
        const auto elapsed = std::chrono::duration<double, std::nano>(Clock::now() - start).count();
        return elapsed / double(passes * buffers.points.size());
    }

    template <Kernel Work, typename T>
    bool compare(std::string_view name, bool checkOnly, std::size_t passes,
        std::span<const std::uint32_t> indices, Buffers<dex::vec3<T>>& legacy,
        Buffers<dex::vec3_packed<T>>& packed, Buffers<dex::vec3_padded<T>>& padded)
    {
        constexpr std::array layoutNames{"legacy", "packed", "padded"};
        run<Work>(legacy, indices);
        run<Work>(packed, indices);
        run<Work>(padded, indices);

        std::array<std::array<double, sampleCount>, layoutNames.size()> samples{};
        if (!checkOnly)
        {
            for (std::size_t sample = 0; sample < sampleCount; ++sample)
            {
                // Rotate the measurement order so each layout runs first and last.
                for (std::size_t offset = 0; offset < layoutNames.size(); ++offset)
                {
                    const auto layout = (sample + offset) % layoutNames.size();
                    switch (layout)
                    {
                    case 0: samples[layout][sample] = measure<Work>(legacy, indices, passes); break;
                    case 1: samples[layout][sample] = measure<Work>(packed, indices, passes); break;
                    case 2: samples[layout][sample] = measure<Work>(padded, indices, passes); break;
                    }
                }
            }
        }

        const std::array sums{checksum<Work>(legacy), checksum<Work>(packed), checksum<Work>(padded)};
        if (!std::all_of(sums.begin(), sums.end(), [&](double sum)
            {
                return std::isfinite(sum) && std::abs(sum - sums.front()) <= 1e-8 * std::abs(sums.front());
            }))
        {
            std::cerr << name << " checksums differ: " << sums[0] << ", " << sums[1] << ", " << sums[2] << '\n';
            return false;
        }

        if (!checkOnly)
        {
            for (std::size_t layout = 0; layout < layoutNames.size(); ++layout)
            {
                auto& timings = samples[layout];
                std::sort(timings.begin(), timings.end());
                std::cout << (std::is_same_v<T, float> ? "float" : "double") << ',' << name << ','
                          << legacy.points.size() << ',' << layoutNames[layout] << ',' << std::setprecision(3)
                          << timings[sampleCount / 2] << ',' << timings.front() << ',' << timings.back() << ','
                          << std::setprecision(9) << sums[layout] << '\n';
            }
        }
        return true;
    }

    template <typename T> bool benchmark(std::size_t count, bool checkOnly)
    {
        const std::size_t passes = count == hotCount ? largeCount / hotCount : 3;
        Buffers<dex::vec3<T>> legacy(count);
        Buffers<dex::vec3_packed<T>> packed(count);
        Buffers<dex::vec3_padded<T>> padded(count);
        std::vector<std::uint32_t> indices(count);
        std::iota(indices.begin(), indices.end(), std::uint32_t{0});
        std::mt19937 random(308);
        std::shuffle(indices.begin(), indices.end(), random);

        return compare<Kernel::dot>("dot", checkOnly, passes, indices, legacy, packed, padded) &&
               compare<Kernel::transform>("transform", checkOnly, passes, indices, legacy, packed, padded) &&
               (count != hotCount || checkOnly
                    ? compare<Kernel::indexedDistance>("indexed_distance", checkOnly, passes, indices, legacy, packed, padded)
                    : true);
    }
}

int main(int argc, char** argv)
{
    const bool checkOnly = argc == 2 && std::string_view(argv[1]) == "--check";
    if (argc != 1 && !checkOnly)
    {
        std::cerr << "Usage: vec3_layout_bench [--check]\n";
        return EXIT_FAILURE;
    }

    if (!checkOnly)
        std::cout << std::fixed
                  << "scalar,kernel,points,layout,median_ns_per_point,min_ns_per_point,max_ns_per_point,checksum\n";
    if (!benchmark<float>(hotCount, checkOnly) || !benchmark<double>(hotCount, checkOnly))
        return EXIT_FAILURE;
    if (checkOnly)
    {
        std::cout << "All vec3 layout checksums agree.\n";
        return EXIT_SUCCESS;
    }
    return benchmark<float>(largeCount, false) && benchmark<double>(largeCount, false) ? EXIT_SUCCESS : EXIT_FAILURE;
}
