#include "TestSupport.h"
#include "ArgbTests.h"
#include "ReferenceResults.h"
#include "VectorTests.h"
#include "VectorCoverageTests.h"
#include "Vec3LayoutTests.h"
#include "TransformTests.h"
#include "TransformCoverageTests.h"
#include "SwizzleTests.h"
#include "FunctionTests.h"
#include "UtilityTests.h"
#include <optional>

#if defined(_MSC_VER) && defined(_DEBUG)
#include <crtdbg.h>
#include <rtcapi.h>
#include <cstdarg>
#include <cstdio>

static int reportRuntimeFailure(int, const char* file, int line, const char*, const char* format, ...)
{
    std::fprintf(stderr, "Runtime check failure at %s:%d: ", file, line);
    va_list arguments;
    va_start(arguments, format);
    std::vfprintf(stderr, format, arguments);
    va_end(arguments);
    std::fputc('\n', stderr);
    std::_Exit(EXIT_FAILURE);
}
#endif

int main(int argc, char** argv)
{
#if defined(_MSC_VER)
    _set_error_mode(_OUT_TO_STDERR);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif

#if defined(_MSC_VER) && defined(_DEBUG)
    _RTC_SetErrorFunc(reportRuntimeFailure);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif

    try
    {
        using Reference = gfxmath_tests::ReferenceResults;
        std::optional<Reference> reference;
        gfxmath_tests::TestContext::NumericObserver observe;
        if (argc != 1)
        {
            const auto option = std::string_view(argv[1]);
            if (argc != 3 || (option != "--compare-reference" && option != "--write-reference" && option != "--write-results"))
            {
                std::cerr << "Usage: gfxmath_tests [--compare-reference PATH | --write-reference PATH | --write-results PATH]\n";
                return EXIT_FAILURE;
            }
            const auto mode = option == "--compare-reference" ? Reference::Mode::compare :
                option == "--write-reference" ? Reference::Mode::writeReference : Reference::Mode::writeResults;
            reference.emplace(mode, argv[2]);
            observe = [&](std::string_view group, std::string_view check, double actual, double expected, double tolerance)
            {
                reference->observe(group, check, actual, expected, tolerance);
            };
        }
        gfxmath_tests::TestContext context(std::move(observe));
        gfxmath_tests::runVectorTests(context);
        gfxmath_tests::runVectorCoverageTests(context);
        gfxmath_tests::runVec3LayoutTests(context);
        gfxmath_tests::forFloatingTypes(
            [&]<typename T>(std::string_view name)
            {
                context.group("swizzles<" + std::string(name) + ">");
                gfxmath_tests::testSwizzles<T>(context);
                if constexpr (std::is_floating_point_v<T>)
                {
                    context.group("packed vec3 swizzles<" + std::string(name) + ">");
                    gfxmath_tests::testVec3Swizzles<T, dex::vec3_layout::packed>(context);
                    context.group("padded vec3 swizzles<" + std::string(name) + ">");
                    gfxmath_tests::testVec3Swizzles<T, dex::vec3_layout::padded>(context);
                    context.group("ARGB helpers<" + std::string(name) + ">");
                    gfxmath_tests::testArgbHelpers<T>(context);
                }
            });
        gfxmath_tests::runTransformTests(context);
        gfxmath_tests::runTransformCoverageTests(context);
        gfxmath_tests::runFunctionTests(context);
        gfxmath_tests::runUtilityTests(context);
        const int result = context.finish();
        return reference ? (reference->finish(result == EXIT_SUCCESS) ? EXIT_SUCCESS : EXIT_FAILURE) : result;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
