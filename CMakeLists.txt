cmake_minimum_required(VERSION 3.20)
project(DEXGfxMath LANGUAGES CXX)

include(CTest)
include(FetchContent)

set(GFXMATH_VECTORCLASS_DIR "" CACHE PATH "Directory containing vectorclass.h")
option(GFXMATH_FETCH_VECTORCLASS "Fetch the pinned Vector Class Library when no local copy is found" ON)

if(NOT GFXMATH_VECTORCLASS_DIR)
    find_path(GFXMATH_VECTORCLASS_INCLUDE_DIR vectorclass.h
        HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../vectorclass"
              "${CMAKE_CURRENT_SOURCE_DIR}/../Expressionist/build/_deps/vectorclass-src")

    if(GFXMATH_VECTORCLASS_INCLUDE_DIR)
        set(GFXMATH_VECTORCLASS_DIR "${GFXMATH_VECTORCLASS_INCLUDE_DIR}")
    endif()
endif()

if(NOT EXISTS "${GFXMATH_VECTORCLASS_DIR}/vectorclass.h")
    if(NOT GFXMATH_FETCH_VECTORCLASS)
        message(FATAL_ERROR "Set GFXMATH_VECTORCLASS_DIR to a directory containing vectorclass.h")
    endif()

    FetchContent_Declare(vectorclass
        GIT_REPOSITORY https://github.com/vectorclass/version2.git
        GIT_TAG 08959ebe6ea5d8317330b242e28ba0d2938ac52f)
    FetchContent_MakeAvailable(vectorclass)
    set(GFXMATH_VECTORCLASS_DIR "${vectorclass_SOURCE_DIR}")
endif()

add_library(DEXGfxMath INTERFACE)
target_compile_features(DEXGfxMath INTERFACE cxx_std_20)
target_include_directories(DEXGfxMath INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/Source" "${GFXMATH_VECTORCLASS_DIR}")

if(BUILD_TESTING)
    add_executable(gfxmath_tests tests/GfxMathTests.cpp)
    target_link_libraries(gfxmath_tests PRIVATE DEXGfxMath)

    if(MSVC)
        target_compile_options(gfxmath_tests PRIVATE /arch:AVX2 /W4 /bigobj)
    else()
        target_compile_options(gfxmath_tests PRIVATE -mavx2 -mfma -Wall -Wextra)
    endif()

    add_test(NAME gfxmath.unit COMMAND gfxmath_tests)
    set_tests_properties(gfxmath.unit PROPERTIES LABELS "unit;simd" TIMEOUT 60)

    add_executable(fastmath_tests tests/FastMathTests.cpp)
    target_compile_features(fastmath_tests PRIVATE cxx_std_20)
    target_include_directories(fastmath_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/Source")

    if(MSVC)
        target_compile_options(fastmath_tests PRIVATE /arch:AVX2 /W4)
    else()
        target_compile_options(fastmath_tests PRIVATE -mavx2 -Wall -Wextra)
    endif()

    add_test(NAME fastmath.unit COMMAND fastmath_tests)
    set_tests_properties(fastmath.unit PROPERTIES LABELS "unit;simd;approximation" TIMEOUT 60)

    foreach(feature NONE VEC2 VEC3 VEC4 VECN RGBA ARGB MAP2D ALL)
        string(TOLOWER "${feature}" feature_name)
        add_library(gfxmath_feature_${feature_name} OBJECT tests/FeatureTests.cpp)
        target_link_libraries(gfxmath_feature_${feature_name} PRIVATE DEXGfxMath)

        if(NOT feature STREQUAL "NONE")
            target_compile_definitions(gfxmath_feature_${feature_name} PRIVATE GFXMATH_${feature})
        endif()

        if(MSVC)
            target_compile_options(gfxmath_feature_${feature_name} PRIVATE /arch:AVX2)
        else()
            target_compile_options(gfxmath_feature_${feature_name} PRIVATE -mavx2 -mfma)
        endif()
    endforeach()
endif()
