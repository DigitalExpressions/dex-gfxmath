#include "GfxMath.h"

static_assert(dex::elements_v<float> == 1);

#if defined(GFXMATH_VEC2) || defined(GFXMATH_ALL)
static_assert(dex::vec2f::size() == 2);
static_assert(dex::vec2d8::size() == 2);
#endif

#if defined(GFXMATH_VEC3) || defined(GFXMATH_ALL)
static_assert(dex::vec3f::size() == 3);
static_assert(dex::vec3d8::size() == 3);
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
