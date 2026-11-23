cmake_minimum_required(VERSION 3.20)
project(DEXGfxMath LANGUAGES CXX)

include(FetchContent)

set(GFXMATH_VECTORCLASS_DIR "" CACHE PATH "Directory containing vcl2_xa64.h")
option(GFXMATH_FETCH_VECTORCLASS "Fetch the pinned VCL2 XA64 when no local copy is found" ON)

if(NOT GFXMATH_VECTORCLASS_DIR)
    find_path(GFXMATH_VCL2_XA64_INCLUDE_DIR vcl2_xa64.h
        HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../vcl2-xa64")

    if(GFXMATH_VCL2_XA64_INCLUDE_DIR)
        set(GFXMATH_VECTORCLASS_DIR "${GFXMATH_VCL2_XA64_INCLUDE_DIR}")
    endif()
endif()

if(NOT GFXMATH_VECTORCLASS_DIR)
    if(NOT GFXMATH_FETCH_VECTORCLASS)
        message(FATAL_ERROR "Set GFXMATH_VECTORCLASS_DIR to a directory containing vcl2_xa64.h")
    endif()

    FetchContent_Declare(vcl2_xa64
        GIT_REPOSITORY https://github.com/DigitalExpressions/vcl2-xa64.git
        GIT_TAG 64efd8d0a8efc5f2b79ffdc4ffc5a3e4251b2b19 # xa64-v1.0.0-rc.1
        SOURCE_SUBDIR _gfxmath_headers_only)
    FetchContent_MakeAvailable(vcl2_xa64)
    set(GFXMATH_VECTORCLASS_DIR "${vcl2_xa64_SOURCE_DIR}")
endif()
if(NOT EXISTS "${GFXMATH_VECTORCLASS_DIR}/vcl2_xa64.h")
    message(FATAL_ERROR "GFXMATH_VECTORCLASS_DIR must point to VCL2 XA64, with vcl2_xa64.h at its root")
endif()

add_library(DEXGfxMath INTERFACE)
target_compile_features(DEXGfxMath INTERFACE cxx_std_20)
target_include_directories(DEXGfxMath INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/Source")
target_include_directories(DEXGfxMath SYSTEM INTERFACE "${GFXMATH_VECTORCLASS_DIR}")
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND CMAKE_CXX_COMPILER_ARCHITECTURE_ID STREQUAL "ARM64")
    # The ARM64 compatibility backend requires the conforming preprocessor.
    target_compile_options(DEXGfxMath INTERFACE /Zc:preprocessor)
endif()

if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    return()
endif()
include(CTest)

set(gfxmath_architectures "${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_GENERATOR_PLATFORM)
    string(REGEX REPLACE ",.*" "" gfxmath_architectures "${CMAKE_GENERATOR_PLATFORM}")
elseif(APPLE AND CMAKE_OSX_ARCHITECTURES)
    set(gfxmath_architectures "${CMAKE_OSX_ARCHITECTURES}")
endif()
string(TOLOWER "${gfxmath_architectures}" gfxmath_architectures)
if(gfxmath_architectures MATCHES "^(x64|amd64|x86_64|win32|x86|i[3-6]86)$")
    set(gfxmath_x86 TRUE)
    if(MSVC)
        set(gfxmath_simd_options /arch:AVX2)
    else()
        set(gfxmath_simd_options -mavx2 -mfma)
    endif()
endif()
if(MSVC)
    set(gfxmath_warning_options /W4)
else()
    set(gfxmath_warning_options -Wall -Wextra)
endif()

add_executable(vec3_layout_bench EXCLUDE_FROM_ALL benchmarks/Vec3LayoutBench.cpp)
target_link_libraries(vec3_layout_bench PRIVATE DEXGfxMath)
target_compile_options(vec3_layout_bench PRIVATE ${gfxmath_simd_options} ${gfxmath_warning_options})

if(BUILD_TESTING)
    add_executable(gfxmath_tests tests/GfxMathTests.cpp)
    target_link_libraries(gfxmath_tests PRIVATE DEXGfxMath)
    target_compile_options(gfxmath_tests PRIVATE ${gfxmath_simd_options} ${gfxmath_warning_options})
    if(MSVC)
        target_compile_options(gfxmath_tests PRIVATE /bigobj)
    endif()

    add_test(NAME gfxmath.unit COMMAND gfxmath_tests)
    set_tests_properties(gfxmath.unit PROPERTIES LABELS "unit;simd" TIMEOUT 60)

    # Keep standalone coverage for narrowing conversions in the sign API.
    add_executable(gfxmath_sign_tests tests/SignConversionTests.cpp)
    target_link_libraries(gfxmath_sign_tests PRIVATE DEXGfxMath)
    target_compile_options(gfxmath_sign_tests PRIVATE ${gfxmath_simd_options} ${gfxmath_warning_options})
    if(MSVC)
        target_compile_options(gfxmath_sign_tests PRIVATE /we4244)
    endif()

    add_test(NAME gfxmath.sign COMMAND gfxmath_sign_tests)
    set_tests_properties(gfxmath.sign PROPERTIES LABELS "unit;simd;conversion" TIMEOUT 60)

    if(gfxmath_x86)
        add_executable(fastmath_tests tests/FastMathTests.cpp)
        target_compile_features(fastmath_tests PRIVATE cxx_std_20)
        target_include_directories(fastmath_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Source")
        target_compile_options(fastmath_tests PRIVATE ${gfxmath_simd_options} ${gfxmath_warning_options})
        add_test(NAME fastmath.unit COMMAND fastmath_tests)
        set_tests_properties(fastmath.unit PROPERTIES LABELS "unit;simd;approximation" TIMEOUT 60)
    endif()

    foreach(feature NONE VEC2 VEC3 VEC4 VECN RGBA ARGB MAP2D ALL)
        string(TOLOWER "${feature}" feature_name)
        add_library(gfxmath_feature_${feature_name} OBJECT tests/FeatureTests.cpp)
        target_link_libraries(gfxmath_feature_${feature_name} PRIVATE DEXGfxMath)

        if(NOT feature STREQUAL "NONE")
            target_compile_definitions(gfxmath_feature_${feature_name} PRIVATE GFXMATH_${feature})
        endif()

        target_compile_options(gfxmath_feature_${feature_name} PRIVATE ${gfxmath_simd_options})
    endforeach()
endif()
