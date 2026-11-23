#include "GfxMath.h"

static_assert(dex::elements_v<float> == 1);
#ifdef GFXMATH_EXPECTED_INSTRSET
static_assert(INSTRSET == GFXMATH_EXPECTED_INSTRSET, "The compiler did not select the requested VCL ISA");
static_assert(MAX_VECTOR_SIZE == 512, "The test suite requires all logical vector widths");
#endif

#if defined(GFXMATH_VEC2) || defined(GFXMATH_ALL)
static_assert(dex::vec2f::size() == 2);
static_assert(dex::vec2d8::size() == 2);
#endif

#if defined(GFXMATH_VEC3) || defined(GFXMATH_ALL)
static_assert(dex::vec3f::size() == 3);
static_assert(dex::vec3d8::size() == 3);
static_assert(sizeof(dex::vec3fpc) == 3 * sizeof(float));
static_assert(sizeof(dex::vec3dpd) == 4 * sizeof(double));
#endif

#if defined(GFXMATH_VEC4) || defined(GFXMATH_ALL)
static_assert(dex::vec4f::size() == 4);
static_assert(dex::vec4d8::size() == 4);
static_assert(sizeof(dex::Quatd) == 4 * sizeof(double));
#endif

#if (defined(GFXMATH_VEC3) && defined(GFXMATH_VEC4)) || defined(GFXMATH_ALL)
static_assert(sizeof(dex::Matrix4f) == 16 * sizeof(float));
#endif

#if defined(GFXMATH_VECN) || defined(GFXMATH_ALL)
static_assert(dex::vecNf<19>::size() == 19);
#endif

#if defined(GFXMATH_MAP2D) || defined(GFXMATH_ALL)
static_assert(sizeof(dex::Map2D<float>) > 0);
#endif
