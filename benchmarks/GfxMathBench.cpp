#define GFXMATH_ALL
#include "GfxMath.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <locale>
#include <string_view>
#include <vector>

#if defined(_MSC_VER)
#define BENCH_NOINLINE __declspec(noinline)
#else
#define BENCH_NOINLINE __attribute__((noinline))
#endif

namespace
{
    constexpr std::size_t count = 4096;
    constexpr std::size_t passes = 256;
    constexpr std::size_t samples = 7;
    #if VCL_ARCH_ARM64
    constexpr auto architecture = "arm64";
    constexpr auto simdProfile = VCL_BACKEND;
    #else
    constexpr auto architecture = "x86";
    constexpr auto simdProfile = INSTRSET;
    #endif

    template <typename Input, typename Output, typename Operation>
    BENCH_NOINLINE void run(const Input* input, Output* output, const Operation& operation)
    {
        for (std::size_t index = 0; index < count; ++index)
            output[index] = operation(input[index]);
    }

    template <typename Factory, typename Operation, typename Checksum>
    bool measure(std::string_view name, std::size_t lanes, bool checkOnly, Factory make, Operation operation, Checksum checksum)
    {
        using Input = decltype(make(0));
        using Output = decltype(operation(make(0)));
        std::vector<Input> input;
        input.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
            input.push_back(make(index));
        std::vector<Output> output(count);
        run(input.data(), output.data(), operation);

        double expected = 0;
        for (const auto& item : output)
            expected += checksum(item);
        if (!std::isfinite(expected) || expected == 0)
        {
            std::cerr << name << " produced an invalid checksum\n";
            return false;
        }
        if (checkOnly)
            return true;

        std::array<double, samples> elapsed{};
        for (double& value : elapsed)
        {
            const auto start = std::chrono::steady_clock::now();
            for (std::size_t pass = 0; pass < passes; ++pass)
                run(input.data(), output.data(), operation);
            value = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() /
                static_cast<double>(count * lanes * passes);
        }

        double observed = 0;
        for (const auto& item : output)
            observed += checksum(item);
        if (observed != expected)
        {
            std::cerr << name << " changed its checksum between warmup and timed passes\n";
            return false;
        }

        std::sort(elapsed.begin(), elapsed.end());
        std::cout << name << ',' << architecture << ',' << simdProfile << ',' << count << ',' << lanes << ','
                  << passes << ',' << samples << ','
                  << std::fixed << std::setprecision(3) << elapsed[samples / 2] << ',' << elapsed.front() << ',' << elapsed.back()
                  << ',' << std::setprecision(12) << observed << '\n';
        return true;
    }

    float coordinate(std::size_t index)
    {
        return (static_cast<float>(index % 257) - 128.0f) / 64.0f;
    }
}

int main(int argc, char** argv)
{
    const bool checkOnly = argc == 2 && std::string_view(argv[1]) == "--check";
    if (argc != 1 && !checkOnly)
    {
        std::cerr << "Usage: gfxmath_bench [--check]\n";
        return EXIT_FAILURE;
    }

    std::cout.imbue(std::locale::classic());
    if (!checkOnly)
        std::cout << "case,architecture,simd_profile,records,lanes,passes,samples,"
                     "median_ns_per_item,min_ns_per_item,max_ns_per_item,checksum\n";

    const bool scalar = measure("float_angles_and_root", 1, checkOnly,
        [](std::size_t index) { return dex::vec2f(coordinate(index), coordinate(index * 73) + 2.25f); },
        [](dex::vec2f p) { return dex::fast_atan2(p.y, p.x) + dex::fast_asin(p.x * 0.25f) + dex::approx_sqrt(p.x + 3.0f); },
        [](float value) { return static_cast<double>(value); });

    using Wide = dex::float16;
    const Wide offsets(0.0f, 0.125f, 0.25f, 0.375f, 0.5f, 0.625f, 0.75f, 0.875f,
        1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f);
    const bool batch = measure("float16_normalize_and_dot", 16, checkOnly,
        [&](std::size_t index)
        {
            return dex::vec3<Wide>(Wide(coordinate(index)) + offsets,
                Wide(coordinate(index * 73) + 2.25f), Wide(1.0f) - offsets);
        },
        [](const dex::vec3<Wide>& p) { return p.normalizedx().dot({Wide(0.5f), Wide(1.0f), Wide(-0.25f)}); },
        [](Wide values) { return static_cast<double>(horizontal_add(values)); });

    using Point = dex::vec4f;
    const bool inverse = measure("matrix4f_inverse", 1, checkOnly,
        [](std::size_t index)
        {
            return dex::Matrix4f(Point(-2.0f, 2.0f, 0.0f, 0.0f), Point(3.0f, 3.0f, 0.0f, 0.0f),
                Point(0.0f, 0.0f, 4.0f, 0.0f), Point(coordinate(index), coordinate(index * 73), 1.0f, 1.0f));
        },
        [](const dex::Matrix4f& matrix) { return matrix.inversed(); },
        [](const dex::Matrix4f& matrix)
        {
            return double(horizontal_add(matrix.col0)) + double(horizontal_add(matrix.col1)) +
                   double(horizontal_add(matrix.col2)) + double(horizontal_add(matrix.col3));
        });

    const auto rotation = dex::Quatf::fromAxisAngle(Point(0, 0, 1, 0), 0.625f);
    const bool quaternion = measure("quatf_rotate", 1, checkOnly,
        [](std::size_t index) { return Point(coordinate(index), coordinate(index * 73), 1.0f, 0.0f); },
        [&](const Point& point) { return rotation.rotate(point); },
        [](const Point& point) { return double(point.x) + 2.0 * double(point.y) + 3.0 * double(point.z); });

    if (!(scalar && batch && inverse && quaternion))
        return EXIT_FAILURE;
    if (checkOnly)
        std::cout << "All four benchmark kernels completed their checksum checks.\n";
    return EXIT_SUCCESS;
}
