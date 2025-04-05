/*
  ==============================================================================

    FastMath.h
    Created: 29 Mar 2021 10:19:40am
    Author:  mathi

  ==============================================================================
*/

#pragma once
#include <immintrin.h>
#include <cmath>

#ifndef forcedinline
#   if _MSC_VER
#       define forcedinline       __forceinline
#   else
#       define forcedinline       inline __attribute__((always_inline))
#   endif
#endif

//Most ninja tricks used here:
//http://fastcpp.blogspot.fr/2011/03/changing-sign-of-float-values-using-sse.html
//http://www.songho.ca/misc/sse/sse.html
//http://markplusplus.wordpress.com/2007/03/14/fast-sse-select-operation/
//http://www.masmforum.com/board/index.php?PHPSESSID=786dd40408172108b65a5a36b09c88c0&topic=9515.0
//http://cbloomrants.blogspot.fr/2010/11/11-20-10-function-approximation-by_20.html
//http://assemblyrequired.crashworks.org/2009/10/16/timing-square-root/
//http://nghiaho.com/?p=997
//http://www.researchgate.net/publication/3321724_Efficient_approximations_for_the_arctangent_function
//http://www.ganssle.com/approx/approx.pdf
//http://forum.allaboutcircuits.com/newsgroups/viewtopic.php?t=68185


//FT Accuracy :
//
//FT::sqrt / sqrt_ps max error : 0.032 % (average error : 0.0094 %)
//FT::atan2 / atan2_ps max error : 0.024 % (0.0015 radians, 0.086 degrees)
//FT::cos / cos_ps max error : 0.06 %
//FT::sin / sin_ps max error : 0.06 %
//
//FT Speed up(MSVC2012 x64) :
//
//    FT::sqrt speed up : x2.5 (from standard sqrt)
//    FT::atan2 speed up : x2.3 (from standard atan2)
//    FT::sin / cos speed up : x1.9 (from standard sin / cos)
//    FT::sincos speed up : x2.3 (from standard sin + cos)
//    FT::sqrt_ps speed up : x8(from standard sqrt)
//    FT::atan2_ps speed up : x7.3 (from standard atan2)
//    FT::sin_ps / cos_ps speed up : x4.9 (from standard sin / cos)
//    FT::sincos_ps speed up : x6.2 (from standard sin + cos)
//
//    FTA Accuracy :
//
//FTA::sqrt / sqrt_ps max error : 0 %
//FTA::atan2 / atan2_ps max error : 0.0005 %
//FTA::cos / cos_ps max error : 0.0007 %
//FTA::sin / sin_ps max error : 0.0007 %
//
//FTA Speed up(MSVC2012 x64) :
//
//    FTA::sqrt speed up : x1.5 (from standard sqrt)
//    FTA::atan2 speed up : x1.7 (from standard atan2)
//    FTA::sin / cos speed up : x1.6 (from standard sin / cos)
//    FTA::sincos speed up : x1.8 (from standard sin + cos)
//    FTA::sqrt_ps speed up : x4.9 (from standard sqrt)
//    FTA::atan2_ps speed up : x5.2 (from standard atan2)
//    FTA::sin_ps / cos_ps speed up : x4.3 (from standard sin / cos)
//    FTA::sincos_ps speed up : x5.2 (from standard sin + cos)



///////////////////////////////////
//FT NAMESPACE (DEFAULT ACCURACY)//
///////////////////////////////////

//SCALAR
namespace FT
{
    const float invtwopi = 0.1591549f;
    const float twopi = 6.283185f;
    const float threehalfpi = 4.7123889f;
    const float pi = 3.141593f;
    const float halfpi = 1.570796f;
    const float quarterpi = 0.7853982f;
    static const __m128 SIGNMASK = _mm_castsi128_ps(_mm_set1_epi32(0x80000000));

    static forcedinline float atan(float x);
    static forcedinline float cos_32s(float x);
    static forcedinline __m128 atan_ps(__m128 x);
    static forcedinline __m128 cos_32s_ps(__m128 x);

    static forcedinline float sqrt(float squared);
    static forcedinline float length(float x, float y);
    static forcedinline float length(float x, float y, float z);
    static forcedinline float atan2(float y, float x);
    static forcedinline float cos(float angle);
    static forcedinline float sin(float angle);
    static forcedinline std::pair<float, float> sincos(float angle);

    static forcedinline __m128 sqrt_ps(__m128 squared);
    static forcedinline __m128 length_ps(__m128 x, __m128 y);
    static forcedinline __m128 length_ps(__m128 x, __m128 y, __m128 z);
    static forcedinline __m128 atan2_ps(__m128 y, __m128 x);
    static forcedinline __m128 cos_ps(__m128 angle);
    static forcedinline __m128 sin_ps(__m128 angle);
    static forcedinline std::pair<__m128, __m128> sincos_ps(__m128 angle);

    /*
     * See http://martin.ankerl.com/2007/10/04/optimized-pow-approximation-for-java-and-c-c/
     *
     * All of these rely on being on a little endian machine, such as an Intel box.
     *
     * These can be _quite_ inaccurate. ~20% in many cases, but being much faster (~7x) may
     * permit more loop iterations of tuning algorithms that only need approximate powers.
     *
     * This version of Ankerl's algorithm has been extended to provide optionally conservative (lower) bounds
     * and also to generate a full linear interpolation across the entire significand rather than 'stair-step'
     * at the expense of performing a 64 bit operation rather than a 32 bit one. This is cheap these days.
     *
     * 'exp' is further improved by using a suggestion by Nic Schraudolph:
     *
     * "You can get a much better approximation (piecewise rational instead of linear) at
     * the cost of a single floating-point division by using better_exp(x) = exp(x/2)/exp(-x/2),
     * where exp() is my published approximation but you don't need the additive constant anymore,
     * you can use c=0. On machines with hardware division this is very attractive." -- Nic Schraudolph
     *
     * --Edward Kmett
     *
     * TODO: Incorporate the techniques from https://code.google.com/p/fastapprox/ to enable us
     * to calculate more interesting approximate functions. They might need to be generalized to work on
     * Double values where appropriate I suppose.
     *
     * Magic numbers:
     * float /int      : round(1<<23/log(2)) = 12102203,          127<<23 = 1065353216
     * double/int      : round(1<<20/log(2)) = 1512775,          1023<<20 = 1072693248
     * double/long long: round(1<<52/log(2)) = 6497320848556798, 1023<<52 = 4607182418800017408
     *
     * The fudge factors such that exp y <= exp_fast y:
     * >>> ceiling (2^23 * (1 - (log (log 2) + 1)/log 2))
     * 722019
     * >>> ceiling (2^20 * (1 - (log (log 2) + 1)/log 2))
     * 90253
     * >>> ceiling (2^52 * (1 - (log (log 2) + 1)/log 2))
     * 387630818974388
     *
     * The fudge factor such that exp_fast y <= exp y is uniformly -1
     *
     * TODO: perform exponential doubling for pow based on better_exp_fast instead for better accuracy.
     */

     /* Schraudolph's published algorithm extended into the least significant bits to avoid the stair step.
      double long long approximation: round 1<<52/log(2) 6497320848556798,
       mask = 0x3ff0000000000000LL = 4607182418800017408LL
      double approximation: round(1<<20/log(2)) = 1512775, 1023<<20 = 1072693248
     */

     /* 4607182418800017408 - 387630818974388 = 4606794787981043020

     Exponent mask adapted to full 64 bit precision:
     >>> 1023 * 2^52
     4607182418800017408

     The fudge factor for conservative lower bound adapted to full 64 bit precision:
     >>> round (2^52 * (1 - (log (log 2) + 1)/log 2))
     387630818974388

     As a lower bound this is suitable for use when generating Mass and Precision estimates.
     */

    static forcedinline double exp_fast_lb(double a) {
        union { double d; long long x; } u;
        u.x = (long long)(6497320848556798LL * a + 4606794787981043020);
        return u.d;
    }

    /* 4607182418800017408 + 1 */
    static forcedinline double exp_fast_ub(double a) {
        union { double d; long long x; } u;
        u.x = (long long)(6497320848556798LL * a + 4607182418800017409);
        return u.d;
    }

    static forcedinline double exp_fast(double a) {
        union { double d; long long x; } u;
        u.x = (long long)(6497320848556798LL * a + 0x3fef127e83d16f12LL);
        return u.d;
    }

    static forcedinline double better_exp_fast(double a) {
        union { double d; long long x; } u, v;
        u.x = (long long)(3248660424278399LL * a + 0x3fdf127e83d16f12LL);
        v.x = (long long)(0x3fdf127e83d16f12LL - 3248660424278399LL * a);
        return u.d / v.d;
    }

    /* Schraudolph's published algorithm */
    static forcedinline double exp_fast_schraudolph(double a) {
        union { double d; int x[2]; } u;
        u.x[1] = (int)(1512775 * a + 1072632447);
        u.x[0] = 0;
        return u.d;
    }

    /* 1065353216 + 1 */
    static forcedinline float expf_fast_ub(float a) {
        union { float f; int x; } u;
        u.x = (int)(12102203 * a + 1065353217);
        return u.f;
    }

    /* Schraudolph's published algorithm with John's constants */
    /* 1065353216 - 486411 = 1064866805 */
    static forcedinline float expf_fast(float a) {
        union { float f; int x; } u;
        u.x = (int)(12102203 * a + 1064866805);
        return u.f;
    }

    //  1056478197 
    static forcedinline double better_expf_fast(float a) {
        union { float f; int x; } u, v;
        u.x = (long long)(6051102 * a + 1056478197);
        v.x = (long long)(1056478197 - 6051102 * a);
        return u.f / v.f;
    }

    /* 1065353216 - 722019 */
    static forcedinline float expf_fast_lb(float a) {
        union { float f; int x; } u;
        u.x = (int)(12102203 * a + 1064631197);
        return u.f;
    }

    /* Ankerl's inversion of Schraudolph's published algorithm, converted to explicit multiplication */
    static forcedinline double log_fast_ankerl(double a) {
        union { double d; int x[2]; } u = { a };
        return (u.x[1] - 1072632447) * 6.610368362777016e-7; /* 1 / 1512775.0; */
    }

    static forcedinline double log_fast_ub(double a) {
        union { double d; long long x; } u = { a };
        return (u.x - 4606794787981043020) * 1.539095918623324e-16; /* 1 / 6497320848556798.0; */
    }

    /* Ankerl's inversion of Schraudolph's published algorithm with my constants */
    static forcedinline double log_fast(double a) {
        union { double d; long long x; } u = { a };
        return (u.x - 4606921278410026770) * 1.539095918623324e-16; /* 1 / 6497320848556798.0; */
    }

    static forcedinline double log_fast_lb(double a) {
        union { double d; long long x; } u = { a };
        return (u.x - 4607182418800017409) * 1.539095918623324e-16; /* 1 / 6497320848556798.0; */
    }


    /* 1065353216 - 722019 */
    static forcedinline float logf_fast_ub(float a) {
        union { float f; int x; } u = { a };
        return (u.x - 1064631197) * 8.262958405176314e-8f; /* 1 / 12102203.0; */
    }

    /* Ankerl's adaptation of Schraudolph's published algorithm with John's constants */
    /* 1065353216 - 486411 = 1064866805 */
    static forcedinline float logf_fast(float a) {
        union { float f; int x; } u = { a };
        return (u.x - 1064866805) * 8.262958405176314e-8f; /* 1 / 12102203.0; */
    }

    /* 1065353216 + 1 */
    static forcedinline float logf_fast_lb(float a) {
        union { float f; int x; } u = { a };
        return (u.x - 1065353217) * 8.262958405176314e-8f; /* 1 / 12102203.0 */
    }

    /* Ankerl's version of Schraudolph's approximation. */
    static forcedinline double pow_fast_ankerl(double a, double b) {
        union { double d; int x[2]; } u = { a };
        u.x[1] = (int)(b * (u.x[1] - 1072632447) + 1072632447);
        u.x[0] = 0;
        return u.d;
    }

    /*
     These constants are based loosely on the following comment off of Ankerl's blog:

     "I have used the same trick for float, not double, with some slight modification to the constants to suite IEEE754 float format. The first constant for float is 1<<23/log(2) and the second is 127<<23 (for double they are 1<<20/log(2) and 1023<<20)." -- John
    */

    /* 1065353216 + 1      = 1065353217 ub */
    /* 1065353216 - 486411 = 1064866805 min RMSE */
    /* 1065353216 - 722019 = 1064631197 lb */
    static forcedinline float powf_fast(float a, float b) {
        union { float d; int x; } u = { a };
        u.x = (int)(b * (u.x - 1064866805) + 1064866805);
        return u.d;
    }

    static forcedinline float powf_fast_lb(float a, float b) {
        union { float d; int x; } u = { a };
        u.x = (int)(b * (u.x - 1065353217) + 1064631197);
        return u.d;
    }

    static forcedinline float powf_fast_ub(float a, float b) {
        union { float d; int x; } u = { a };
        u.x = (int)(b * (u.x - 1064631197) + 1065353217);
        return u.d;
    }

    /*
      Now that 64 bit arithmetic is cheap we can (try to) improve on Ankerl's algorithm.

     double long long approximation: round 1<<52/log(2) 6497320848556798,
      mask = 0x3ff0000000000000LL = 4607182418800017408LL

    >>> round (2**52 * log (3 / (8 * log 2) + 1/2) / log 2 - 1/2)
    261140389990638
    >>> 0x3ff0000000000000 - round (2**52 * log (3 / (8 * log 2) + 1/2) / log 2 - 1/2)
    4606921278410026770

    */

    static forcedinline double pow_fast_ub(double a, double b) {
        union { double d; long long x; } u = { a };
        u.x = (long long)(b * (u.x - 4606794787981043020LL) + 4607182418800017409LL);
        return u.d;
    }

    static forcedinline double pow_fast(double a, double b) {
        union { double d; long long x; } u = { a };
        u.x = (long long)(b * (u.x - 4606921278410026770LL) + 4606921278410026770LL);
        return u.d;
    }

    static forcedinline double pow_fast_lb(double a, double b) {
        union { double d; long long x; } u = { a };
        u.x = (long long)(b * (u.x - 4607182418800017409LL) + 4606794787981043020LL);
        return u.d;
    }

    /* should be much more precise with large b, still ~3.3x faster. */
    static forcedinline double pow_fast_precise_ankerl(double a, double b) {
        int flipped = 0;
        if (b < 0) {
            flipped = 1;
            b = -b;
        }

        /* calculate approximation with fraction of the exponent */
        int e = (int)b;
        union { double d; int x[2]; } u = { a };
        u.x[1] = (int)((b - e) * (u.x[1] - 1072632447) + 1072632447);
        u.x[0] = 0;

        double r = 1.0;
        while (e) {
            if (e & 1) {
                r *= a;
            }
            a *= a;
            e >>= 1;
        }

        r *= u.d;
        return flipped ? 1.0 / r : r;
    }

    /* should be much more precise with large b, still ~3.3x faster. */
    static forcedinline double pow_fast_precise(double a, double b) {
        int flipped = 0;
        if (b < 0) {
            flipped = 1;
            b = -b;
        }

        /* calculate approximation with fraction of the exponent */
        int e = (int)b;
        double d = exp_fast(b - e);

        double r = 1.0;
        while (e) {
            if (e & 1) r *= a;
            a *= a;
            e >>= 1;
        }

        r *= d;
        return flipped ? 1.0 / r : r;
    }

    static forcedinline double better_pow_fast_precise(double a, double b) {
        int flipped = 0;
        if (b < 0) {
            flipped = 1;
            b = -b;
        }

        /* calculate approximation with fraction of the exponent */
        int e = (int)b;
        double d = better_exp_fast(b - e);

        double r = 1.0;
        while (e) {
            if (e & 1) r *= a;
            a *= a;
            e >>= 1;
        }

        r *= d;
        return flipped ? 1.0 / r : r;
    }


    /* should be much more precise with large b */
    static forcedinline float powf_fast_precise(float a, float b) {
        int flipped = 0;
        if (b < 0) {
            flipped = 1;
            b = -b;
        }

        /* calculate approximation with fraction of the exponent */
        int e = (int)b;
        union { float f; int x; } u = { a };
        u.x = (int)((b - e) * (u.x - 1065353216) + 1065353216);

        float r = 1.0f;
        while (e) {
            if (e & 1) {
                r *= a;
            }
            a *= a;
            e >>= 1;
        }

        r *= u.f;
        return flipped ? 1.0f / r : r;
    }

    /* should be much more precise with large b */
    static forcedinline float better_powf_fast_precise(float a, float b) {
        int flipped = 0;
        if (b < 0) {
            flipped = 1;
            b = -b;
        }

        /* calculate approximation with fraction of the exponent */
        int e = (int)b;
        float f = (float)better_expf_fast(b - e);

        float r = 1.0f;
        while (e) {
            if (e & 1) {
                r *= a;
            }
            a *= a;
            e >>= 1;
        }

        r *= f;
        return flipped ? 1.0f / r : r;
    }
};

static forcedinline float FT::sqrt(float squared)
{
    //static int csr = 0;
    //if (!csr) csr = _mm_getcsr() | 0x8040; //DAZ,FTZ (divide by zero=0)
    //_mm_setcsr(csr);
    return squared > 0 ? _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(squared))) * squared : 0.0f;
}

static forcedinline float FT::length(float x, float y)
{
    return FT::sqrt(x * x + y * y);
}
static forcedinline float FT::length(float x, float y, float z)
{
    return FT::sqrt(x * x + y * y + z * z);
}

static forcedinline float FT::atan(float x)
{
    return quarterpi * x - x * (fabs(x) - 1) * (0.2447f + 0.0663f * fabs(x));
}

static forcedinline float FT::atan2(float y, float x)
{
    if (fabs(x) > fabs(y))
    {
        float atan = FT::atan(y / x);
        if (x > 0.0f)
            return atan;
        else
            return y > 0.0f ? atan + pi : atan - pi;
    }
    else
    {
        float atan = FT::atan(x / y);
        if (x > 0.0f)
            return y > 0.0f ? halfpi - atan : -halfpi - atan;
        else
            return y > 0.0f ? halfpi + atan : -halfpi + atan;
    }
}

static forcedinline float FT::cos_32s(float x)
{
    const float c1 = 0.99940307f;
    const float c2 = -0.49558072f;
    const float c3 = 0.03679168f;
    float x2;      // The input argument squared
    x2 = x * x;
    return (c1 + x2 * (c2 + c3 * x2));
}

static forcedinline float FT::cos(float angle) 
{
    //clamp to the range 0..2pi
    angle = angle - floorf(angle * invtwopi) * twopi;
    angle = angle > 0.0f ? angle : -angle;

    if (angle < halfpi) return FT::cos_32s(angle);
    if (angle < pi) return -FT::cos_32s(pi - angle);
    if (angle < threehalfpi) return -FT::cos_32s(angle - pi);
    return FT::cos_32s(twopi - angle);
}

static forcedinline float FT::sin(float angle) 
{
    return FT::cos(halfpi - angle);
}

static forcedinline std::pair<float, float> FT::sincos(float angle)
{
    //clamp to the range 0..2pi
    angle = angle - floorf(angle * invtwopi) * twopi;
    float sinmultiplier = angle > 0.0f && angle < pi ? 1.0f : -1.0f;
    angle = angle > 0.0f ? angle : -angle;
    if (angle < halfpi) 
    {
        float co = FT::cos_32s(angle);
        return std::make_pair(sinmultiplier * FT::sqrt(1.0f - co * co), co);
    }
    else if (angle < pi) 
    {
        float co = -FT::cos_32s(pi - angle);
        return std::make_pair(sinmultiplier * FT::sqrt(1.0f - co * co), co);
    }
    else if (angle < threehalfpi) 
    {
        float co = -FT::cos_32s(angle - pi);
        return std::make_pair(sinmultiplier * FT::sqrt(1.0f - co * co), co);
    }
    else
    {
        float co = FT::cos_32s(twopi - angle);
        return std::make_pair(sinmultiplier * FT::sqrt(1.0f - co * co), co);
    }
}

//PACKED SCALAR
static forcedinline __m128 FT::sqrt_ps(__m128 squared)
{
    //static int csr = 0;
    //if (!csr) csr = _mm_getcsr() | 0x8040; //DAZ,FTZ (divide by zero=0)
    //_mm_setcsr(csr);
    //return _mm_mul_ps(_mm_rsqrt_ps(squared), squared);
    return _mm_sqrt_ps(squared);
}

static forcedinline __m128 FT::length_ps(__m128 x, __m128 y)
{
    return FT::sqrt_ps(_mm_add_ps(_mm_mul_ps(x, x), _mm_mul_ps(y, y)));
}

static forcedinline __m128 FT::length_ps(__m128 x, __m128 y, __m128 z)
{
    return FT::sqrt_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(x, x), _mm_mul_ps(y, y)), _mm_mul_ps(z, z)));
}

static forcedinline __m128 FT::atan_ps(__m128 x)
{
    // quarterpi*x
    // - x*(fabs(x) - 1)
    // *(0.2447f+0.0663f*fabs(x));
    return _mm_sub_ps(_mm_mul_ps(_mm_set1_ps(quarterpi), x),
        _mm_mul_ps(_mm_mul_ps(x, _mm_sub_ps(_mm_andnot_ps(SIGNMASK, x), _mm_set1_ps(1.0f))),
            (_mm_add_ps(_mm_set1_ps(0.2447f), _mm_mul_ps(_mm_set1_ps(0.0663f), _mm_andnot_ps(SIGNMASK, x))))));
}

static forcedinline __m128 FT::atan2_ps(__m128 y, __m128 x)
{
    __m128 absxgreaterthanabsy = _mm_cmpgt_ps(_mm_andnot_ps(SIGNMASK, x), _mm_andnot_ps(SIGNMASK, y));
    __m128 ratio = _mm_div_ps(_mm_add_ps(_mm_and_ps(absxgreaterthanabsy, y), _mm_andnot_ps(absxgreaterthanabsy, x)),
        _mm_add_ps(_mm_and_ps(absxgreaterthanabsy, x), _mm_andnot_ps(absxgreaterthanabsy, y)));
    __m128 atan = FT::atan_ps(ratio);

    __m128 xgreaterthan0 = _mm_cmpgt_ps(x, _mm_set1_ps(0.0f));
    __m128 ygreaterthan0 = _mm_cmpgt_ps(y, _mm_set1_ps(0.0f));

    atan = _mm_xor_ps(atan, _mm_andnot_ps(absxgreaterthanabsy, _mm_and_ps(xgreaterthan0, SIGNMASK))); //negate atan if absx<=absy & x>0

    __m128 shift = _mm_set1_ps(pi);
    shift = _mm_sub_ps(shift, _mm_andnot_ps(absxgreaterthanabsy, _mm_set1_ps(halfpi))); //substract halfpi if absx<=absy
    shift = _mm_xor_ps(shift, _mm_andnot_ps(ygreaterthan0, SIGNMASK)); //negate shift if y<=0
    shift = _mm_andnot_ps(_mm_and_ps(absxgreaterthanabsy, xgreaterthan0), shift); //null if abs>absy & x>0

    return _mm_add_ps(atan, shift);
}

static forcedinline __m128 FT::cos_32s_ps(__m128 x)
{
    const __m128 c1 = _mm_set1_ps(0.99940307f);
    const __m128 c2 = _mm_set1_ps(-0.49558072f);
    const __m128 c3 = _mm_set1_ps(0.03679168f);
    __m128 x2;      // The input argument squared
    x2 = _mm_mul_ps(x, x);
    //               (c1+           x2*          (c2+           c3*x2));
    return _mm_add_ps(c1, _mm_mul_ps(x2, _mm_add_ps(c2, _mm_mul_ps(c3, x2))));
}

static forcedinline __m128 FT::cos_ps(__m128 angle) {
    //clamp to the range 0..2pi

    //take absolute value
    angle = _mm_andnot_ps(SIGNMASK, angle);
    //fmod(angle,twopi)
    angle = _mm_sub_ps(angle, _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvttps_epi32(_mm_mul_ps(angle, _mm_set1_ps(invtwopi)))), _mm_set1_ps(twopi))); //simplied SSE2 fmod, must always operate on absolute value
    //if SSE4.1 is always available, comment the line above and uncomment the line below
    //angle=_mm_sub_ps(angle,_mm_mul_ps(_mm_floor_ps(_mm_mul_ps(angle,_mm_set1_ps(invtwopi))),_mm_set1_ps(twopi))); //faster if SSE4.1 is always available

    __m128 cosangle = angle;
    cosangle = _mm_xor_ps(cosangle, _mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(halfpi)), _mm_xor_ps(cosangle, _mm_sub_ps(_mm_set1_ps(pi), angle))));
    cosangle = _mm_xor_ps(cosangle, _mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(pi)), SIGNMASK));
    cosangle = _mm_xor_ps(cosangle, _mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(threehalfpi)), _mm_xor_ps(cosangle, _mm_sub_ps(_mm_set1_ps(twopi), angle))));

    __m128 result = FT::cos_32s_ps(cosangle);

    result = _mm_xor_ps(result, _mm_and_ps(_mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(halfpi)), _mm_cmplt_ps(angle, _mm_set1_ps(threehalfpi))), SIGNMASK));
    return result;
}

static forcedinline __m128 FT::sin_ps(__m128 angle) 
{
    return FT::cos_ps(_mm_sub_ps(_mm_set1_ps(halfpi), angle));
}

static forcedinline std::pair<__m128, __m128> FT::sincos_ps(__m128 angle) 
{
    __m128 anglesign = _mm_or_ps(_mm_set1_ps(1.0f), _mm_and_ps(SIGNMASK, angle));

    //clamp to the range 0..2pi

    //take absolute value
    angle = _mm_andnot_ps(SIGNMASK, angle);
    //fmod(angle,twopi)

    //angle = _mm_sub_ps(angle, _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvttps_epi32(_mm_mul_ps(angle, _mm_set1_ps(invtwopi)))), _mm_set1_ps(twopi))); //simplied SSE2 fmod, must always operate on absolute value
    //if SSE4.1 is always available, comment the line above and uncomment the line below
    angle=_mm_sub_ps(angle,_mm_mul_ps(_mm_floor_ps(_mm_mul_ps(angle,_mm_set1_ps(invtwopi))),_mm_set1_ps(twopi))); //faster if SSE4.1 is always available

    __m128 cosangle = angle;
    cosangle = _mm_xor_ps(cosangle, _mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(halfpi)), _mm_xor_ps(cosangle, _mm_sub_ps(_mm_set1_ps(pi), angle))));
    cosangle = _mm_xor_ps(cosangle, _mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(pi)), SIGNMASK));
    cosangle = _mm_xor_ps(cosangle, _mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(threehalfpi)), _mm_xor_ps(cosangle, _mm_sub_ps(_mm_set1_ps(twopi), angle))));

    __m128 result = FT::cos_32s_ps(cosangle);

    auto co = _mm_xor_ps(result, _mm_and_ps(_mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(halfpi)), _mm_cmplt_ps(angle, _mm_set1_ps(threehalfpi))), SIGNMASK));

    __m128 sinmultiplier = _mm_mul_ps(anglesign, _mm_or_ps(_mm_set1_ps(1.0f), _mm_and_ps(_mm_cmpgt_ps(angle, _mm_set1_ps(pi)), SIGNMASK)));
    auto si = _mm_mul_ps(sinmultiplier, FT::sqrt_ps(_mm_sub_ps(_mm_set1_ps(1.0f), _mm_mul_ps(co, co))));

    return std::pair(si, co);
}


/////////////////////////////////
//FTA NAMESPACE (MORE ACCURATE)//
/////////////////////////////////

//SCALAR
namespace FTA
{
    static forcedinline float atan(float x);
    static forcedinline float cos_52s(float x);
    static forcedinline __m128 atan_ps(__m128 x);
    static forcedinline __m128 cos_52s_ps(__m128 x);

    static forcedinline float sqrt(float squared);
    static forcedinline float length(float x, float y);
    static forcedinline float length(float x, float y, float z);
    static forcedinline float atan2(float y, float x);
    static forcedinline float cos(float angle);
    static forcedinline float sin(float angle);
    static forcedinline std::pair<float, float> sincos(float angle);

    static forcedinline __m128 sqrt_ps(__m128 squared);
    static forcedinline __m128 length_ps(__m128 x, __m128 y);
    static forcedinline __m128 length_ps(__m128 x, __m128 y, __m128 z);
    static forcedinline __m128 atan2_ps(__m128 y, __m128 x);
    static forcedinline __m128 cos_ps(__m128 angle);
    static forcedinline __m128 sin_ps(__m128 angle);
    static forcedinline std::pair<__m128, __m128> sincos_ps(__m128 angle);
};

static forcedinline float FTA::sqrt(float squared)
{
    return _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(squared)));
}

static forcedinline float FTA::length(float x, float y)
{
    return FTA::sqrt(x * x + y * y);
}

static forcedinline float FTA::length(float x, float y, float z)
{
    return FTA::sqrt(x * x + y * y + z * z);
}

static forcedinline float FTA::atan(float x)
{
    float u = x * x;
    float u2 = u * u;
    float u3 = u2 * u;
    float u4 = u3 * u;
    float f = 1.0f + 0.33288950512027f * u - 0.08467922817644f * u2 + 0.03252232640125f * u3 - 0.00749305860992f * u4;
    return x / f;
}

static forcedinline float FTA::atan2(float y, float x)
{
    if (fabs(x) > fabs(y)) 
    {
        float atan = FTA::atan(y / x);
        if (x > 0.0f)
            return atan;
        else
            return y > 0.0f ? atan + FT::pi : atan - FT::pi;
    }
    else 
    {
        float atan = FTA::atan(x / y);
        if (x > 0.0f)
            return y > 0.0f ? FT::halfpi - atan : -FT::halfpi - atan;
        else
            return y > 0.0f ? FT::halfpi + atan : -FT::halfpi + atan;
    }
}

static forcedinline float FTA::cos_52s(float x)
{
    const float c1 = 0.9999932946f;
    const float c2 = -0.4999124376f;
    const float c3 = 0.0414877472f;
    const float c4 = -0.0012712095f;
    float x2;      // The input argument squared
    x2 = x * x;
    return (c1 + x2 * (c2 + x2 * (c3 + c4 * x2)));
}

static forcedinline float FTA::cos(float angle) 
{
    //clamp to the range 0..2pi
    angle = angle - floorf(angle * FT::invtwopi) * FT::twopi;
    angle = angle > 0.0f ? angle : -angle;

    if (angle < FT::halfpi) return FTA::cos_52s(angle);
    if (angle < FT::pi) return -FTA::cos_52s(FT::pi - angle);
    if (angle < FT::threehalfpi) return -FTA::cos_52s(angle - FT::pi);
    return FTA::cos_52s(FT::twopi - angle);
}

static forcedinline float FTA::sin(float angle) 
{
    return FTA::cos(FT::halfpi - angle);
}

static forcedinline std::pair<float, float> FTA::sincos(float angle)
{
    //clamp to the range 0..2pi
    angle = angle - floorf(angle * FT::invtwopi) * FT::twopi;
    float sinmultiplier = angle > 0.0f && angle < FT::pi ? 1.0f : -1.0f;
    angle = angle > 0.0f ? angle : -angle;

    if (angle < FT::halfpi)
    {
        float co = FTA::cos_52s(angle);
        return std::make_pair(sinmultiplier * FTA::sqrt(1.0f - co * co), co);
    }
    else if (angle < FT::pi)
    {
        float co = -FTA::cos_52s(FT::pi - angle);
        return std::make_pair(sinmultiplier * FTA::sqrt(1.0f - co * co), co);
    }
    else if (angle < FT::threehalfpi)
    {
        float co = -FTA::cos_52s(angle - FT::pi);
        return std::make_pair(sinmultiplier * FTA::sqrt(1.0f - co * co), co);
    }
    else
    {
        float co = FTA::cos_52s(FT::twopi - angle);
        return std::make_pair(sinmultiplier * FTA::sqrt(1.0f - co * co), co);
    }
}

//PACKED SCALAR
static forcedinline __m128 FTA::sqrt_ps(__m128 squared)
{
    return _mm_sqrt_ps(squared);
}

static forcedinline __m128 FTA::length_ps(__m128 x, __m128 y)
{
    return FTA::sqrt_ps(_mm_add_ps(_mm_mul_ps(x, x), _mm_mul_ps(y, y)));
}

static forcedinline __m128 FTA::length_ps(__m128 x, __m128 y, __m128 z)
{
    return FTA::sqrt_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(x, x), _mm_mul_ps(y, y)), _mm_mul_ps(z, z)));
}

static forcedinline __m128 FTA::atan_ps(__m128 x)
{
    __m128 u = _mm_mul_ps(x, x);
    __m128 u2 = _mm_mul_ps(u, u);
    __m128 u3 = _mm_mul_ps(u2, u);
    __m128 u4 = _mm_mul_ps(u3, u);
    //__m128 f=1.0f+0.33288950512027f*u-0.08467922817644f*u2+0.03252232640125f*u3-0.00749305860992f*u4;

    __m128 f = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_set1_ps(1.0f),
        _mm_mul_ps(_mm_set1_ps(0.33288950512027f), u)),
        _mm_mul_ps(_mm_set1_ps(-0.08467922817644f), u2)),
        _mm_mul_ps(_mm_set1_ps(0.03252232640125f), u3)),
        _mm_mul_ps(_mm_set1_ps(-0.00749305860992f), u4));
    return _mm_div_ps(x, f);
}

static forcedinline __m128 FTA::atan2_ps(__m128 y, __m128 x)
{
    __m128 absxgreaterthanabsy = _mm_cmpgt_ps(_mm_andnot_ps(FT::SIGNMASK, x), _mm_andnot_ps(FT::SIGNMASK, y));
    __m128 ratio = _mm_div_ps(_mm_add_ps(_mm_and_ps(absxgreaterthanabsy, y), _mm_andnot_ps(absxgreaterthanabsy, x)),
        _mm_add_ps(_mm_and_ps(absxgreaterthanabsy, x), _mm_andnot_ps(absxgreaterthanabsy, y)));
    __m128 atan = FTA::atan_ps(ratio);

    __m128 xgreaterthan0 = _mm_cmpgt_ps(x, _mm_set1_ps(0.0f));
    __m128 ygreaterthan0 = _mm_cmpgt_ps(y, _mm_set1_ps(0.0f));

    atan = _mm_xor_ps(atan, _mm_andnot_ps(absxgreaterthanabsy, _mm_and_ps(xgreaterthan0, FT::SIGNMASK))); //negate atan if absx<=absy & x>0

    __m128 shift = _mm_set1_ps(FT::pi);
    shift = _mm_sub_ps(shift, _mm_andnot_ps(absxgreaterthanabsy, _mm_set1_ps(FT::halfpi))); //substract halfpi if absx<=absy
    shift = _mm_xor_ps(shift, _mm_andnot_ps(ygreaterthan0, FT::SIGNMASK)); //negate shift if y<=0
    shift = _mm_andnot_ps(_mm_and_ps(absxgreaterthanabsy, xgreaterthan0), shift); //null if abs>absy & x>0

    return _mm_add_ps(atan, shift);
}

static forcedinline __m128 FTA::cos_52s_ps(__m128 x)
{
    const __m128 c1 = _mm_set1_ps(0.9999932946f);
    const __m128 c2 = _mm_set1_ps(-0.4999124376f);
    const __m128 c3 = _mm_set1_ps(0.0414877472f);
    const __m128 c4 = _mm_set1_ps(-0.0012712095f);
    __m128 x2;      // The input argument squared
    x2 = _mm_mul_ps(x, x);
    //               (c1+           x2*          (c2+           x2*          (c3+           c4*x2)));
    return _mm_add_ps(c1, _mm_mul_ps(x2, _mm_add_ps(c2, _mm_mul_ps(x2, _mm_add_ps(c3, _mm_mul_ps(c4, x2))))));
}

static forcedinline __m128 FTA::cos_ps(__m128 angle) 
{
    //clamp to the range 0..2pi

    //take absolute value
    angle = _mm_andnot_ps(FT::SIGNMASK, angle);
    //fmod(angle,twopi)
    angle = _mm_sub_ps(angle, _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvttps_epi32(_mm_mul_ps(angle, _mm_set1_ps(FT::invtwopi)))), _mm_set1_ps(FT::twopi))); //simplied SSE2 fmod, must always operate on absolute value
    //if SSE4.1 is always available, comment the line above and uncomment the line below
    //angle=_mm_sub_ps(angle,_mm_mul_ps(_mm_floor_ps(_mm_mul_ps(angle,_mm_set1_ps(invtwopi))),_mm_set1_ps(twopi))); //faster if SSE4.1 is always available

    __m128 cosangle = angle;
    cosangle = _mm_xor_ps(cosangle, _mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(FT::halfpi)), _mm_xor_ps(cosangle, _mm_sub_ps(_mm_set1_ps(FT::pi), angle))));
    cosangle = _mm_xor_ps(cosangle, _mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(FT::pi)), FT::SIGNMASK));
    cosangle = _mm_xor_ps(cosangle, _mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(FT::threehalfpi)), _mm_xor_ps(cosangle, _mm_sub_ps(_mm_set1_ps(FT::twopi), angle))));

    __m128 result = FTA::cos_52s_ps(cosangle);

    result = _mm_xor_ps(result, _mm_and_ps(_mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(FT::halfpi)), _mm_cmplt_ps(angle, _mm_set1_ps(FT::threehalfpi))), FT::SIGNMASK));
    return result;
}

static forcedinline __m128 FTA::sin_ps(__m128 angle) 
{
    return FTA::cos_ps(_mm_sub_ps(_mm_set1_ps(FT::halfpi), angle));
}

static forcedinline std::pair<__m128, __m128> FTA::sincos_ps(__m128 angle)
{
    __m128 anglesign = _mm_or_ps(_mm_set1_ps(1.0f), _mm_and_ps(FT::SIGNMASK, angle));

    //clamp to the range 0..2pi

    //take absolute value
    angle = _mm_andnot_ps(FT::SIGNMASK, angle);
    //fmod(angle,twopi)
    angle = _mm_sub_ps(angle, _mm_mul_ps(_mm_cvtepi32_ps(_mm_cvttps_epi32(_mm_mul_ps(angle, _mm_set1_ps(FT::invtwopi)))), _mm_set1_ps(FT::twopi))); //simplied SSE2 fmod, must always operate on absolute value
    //if SSE4.1 is always available, comment the line above and uncomment the line below
    //angle=_mm_sub_ps(angle,_mm_mul_ps(_mm_floor_ps(_mm_mul_ps(angle,_mm_set1_ps(invtwopi))),_mm_set1_ps(twopi))); //faster if SSE4.1 is always available

    __m128 cosangle = angle;
    cosangle = _mm_xor_ps(cosangle, _mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(FT::halfpi)), _mm_xor_ps(cosangle, _mm_sub_ps(_mm_set1_ps(FT::pi), angle))));
    cosangle = _mm_xor_ps(cosangle, _mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(FT::pi)), FT::SIGNMASK));
    cosangle = _mm_xor_ps(cosangle, _mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(FT::threehalfpi)), _mm_xor_ps(cosangle, _mm_sub_ps(_mm_set1_ps(FT::twopi), angle))));

    __m128 result = FTA::cos_52s_ps(cosangle);

    result = _mm_xor_ps(result, _mm_and_ps(_mm_and_ps(_mm_cmpge_ps(angle, _mm_set1_ps(FT::halfpi)), _mm_cmplt_ps(angle, _mm_set1_ps(FT::threehalfpi))), FT::SIGNMASK));
    auto co = result;

    __m128 sinmultiplier = _mm_mul_ps(anglesign, _mm_or_ps(_mm_set1_ps(1.0f), _mm_and_ps(_mm_cmpgt_ps(angle, _mm_set1_ps(FT::pi)), FT::SIGNMASK)));
    auto si = _mm_mul_ps(sinmultiplier, FT::sqrt_ps(_mm_sub_ps(_mm_set1_ps(1.0f), _mm_mul_ps(co, co))));

    return std::pair(si, co);
}
