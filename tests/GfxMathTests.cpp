#include "TestSupport.h"
#include "VectorTests.h"
#include "VectorCoverageTests.h"
#include "TransformTests.h"
#include "TransformCoverageTests.h"
#include "SwizzleTests.h"
#include "FunctionTests.h"
#include "UtilityTests.h"

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

int main()
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

    gfxmath_tests::TestContext context;
    gfxmath_tests::runVectorTests(context);
    gfxmath_tests::runVectorCoverageTests(context);
    gfxmath_tests::forFloatingTypes(
        [&]<typename T>(std::string_view name)
        {
            context.group("swizzles<" + std::string(name) + ">");
            gfxmath_tests::testSwizzles<T>(context);
        });
    gfxmath_tests::runTransformTests(context);
    gfxmath_tests::runTransformCoverageTests(context);
    gfxmath_tests::runFunctionTests(context);
    gfxmath_tests::runUtilityTests(context);
    return context.finish();
}
