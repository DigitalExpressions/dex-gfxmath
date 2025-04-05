/*
  ==============================================================================

    ScriptGfxMath.h
    Created: 16 Feb 2025 1:24:04am
    Author:  mathi

  ==============================================================================
*/

#pragma once
#include "ScriptEngine.h"
#include "../GfxMath.h"

ASE_GLOBAL_DECL_NAMESPACE("Float");

static inline float e = E<float>;
static inline float halfPi = HalfPi<float>;
static inline float pi = Pi<float>;
static inline float twoPi = TwoPi<float>;
static inline float fourPi = FourPi<float>;
static inline float invPi = InvPi<float>;
static inline float invTwoPi = InvTwoPi<float>;
static inline float invFourPi = InvFourPi<float>;
static inline float sqrtPi = SqrtPi<float>;
static inline float invSqrtPi = InvSqrtPi<float>;
static inline float sqrtTwo = SqrtTwo<float>;
static inline float invSqrtTwo = InvSqrtTwo<float>;
static inline float sqrtTwoPi = SqrtTwoPi<float>;
static inline float invSqrtTwoPi = InvSqrtTwoPi<float>;
static inline float infinity = Infinity<float>;
static inline float min = Min<float>;
static inline float max = Max<float>;
static inline float epsilon = Epsilon<float>;

ASE_PROPERTY_GLOBAL("const float e", e);
ASE_PROPERTY_GLOBAL("const float halfPi", halfPi);
ASE_PROPERTY_GLOBAL("const float pi", pi);
ASE_PROPERTY_GLOBAL("const float twoPi", twoPi);
ASE_PROPERTY_GLOBAL("const float fourPi", fourPi);
ASE_PROPERTY_GLOBAL("const float invPi", invPi);
ASE_PROPERTY_GLOBAL("const float invTwoPi", invTwoPi);
ASE_PROPERTY_GLOBAL("const float invFourPi", invFourPi);
ASE_PROPERTY_GLOBAL("const float sqrtPi", sqrtPi);
ASE_PROPERTY_GLOBAL("const float invSqrtPi", invSqrtPi);
ASE_PROPERTY_GLOBAL("const float sqrtTwo", sqrtTwo);
ASE_PROPERTY_GLOBAL("const float invSqrtTwo", invSqrtTwo);
ASE_PROPERTY_GLOBAL("const float sqrtTwoPi", sqrtTwoPi);
ASE_PROPERTY_GLOBAL("const float invSqrtTwoPi", invSqrtTwoPi);
ASE_PROPERTY_GLOBAL("const float infinity", infinity);
ASE_PROPERTY_GLOBAL("const float min", min);
ASE_PROPERTY_GLOBAL("const float max", max);
ASE_PROPERTY_GLOBAL("const float epsilon", epsilon);

ASE_END_GLOBAL_DECL_NAMESPACE;

class ScriptVec4;

class ScriptVec3 :
    public ASE_ValueClass<ScriptVec3, asOBJ_POD | asOBJ_APP_CLASS_ALLFLOATS>
{
    ASE_CLASS(ScriptVec3, "vec3");
public:
    ScriptVec3(const vec3f& v) : value(v) {}

    ScriptVec3() = default;

    ASE_CONSTRUCTOR("(const vec3& v)", const ScriptVec3&);
    ScriptVec3(const ScriptVec3& v) : value(v.value) {}
    ASE_CONSTRUCTOR("(const vec4& v)", const ScriptVec4&);
    ScriptVec3(const ScriptVec4& v);
    ASE_CONSTRUCTOR("(float v)", float);
    ScriptVec3(float v) : value(v) {}
    ASE_CONSTRUCTOR("(float x, float y, float z)", float, float, float);
    ScriptVec3(float x, float y, float z) : value(x, y, z) {}

    ASE_PROPERTY_MEMBER("float x", value.x);
    ASE_PROPERTY_MEMBER("float y", value.y);
    ASE_PROPERTY_MEMBER("float z", value.z);

    ASE_METHOD_MEMBER("vec3& opAddAssign(const vec3&)", value, vec3f&, operator+=, (const vec3f&));
    ASE_METHOD_MEMBER("vec3& opSubAssign(const vec3&)", value, vec3f&, operator-=, (const vec3f&));
    ASE_METHOD_MEMBER("vec3& opMulAssign(const vec3&)", value, vec3f&, operator*=, (const vec3f&));
    ASE_METHOD_MEMBER("vec3& opDivAssign(const vec3&)", value, vec3f&, operator/=, (const vec3f&));

    ASE_METHOD_MEMBER("vec3 opAdd(const vec3&) const", value, vec3f, operator+, (const vec3f&) const);
    ASE_METHOD_MEMBER("vec3 opSub(const vec3&) const", value, vec3f, operator-, (const vec3f&) const);
    ASE_METHOD_MEMBER("vec3 opMul(const vec3&) const", value, vec3f, operator*, (const vec3f&) const);
    ASE_METHOD_MEMBER("vec3 opDiv(const vec3&) const", value, vec3f, operator/, (const vec3f&) const);
    ASE_METHOD_MEMBER("vec3 opNeg() const", value, vec3f, operator-, () const);

    ASE_METHOD_MEMBER("vec3 equals(const vec3&) const", value, vec3f, operator==, (const vec3f&) const);
    ASE_METHOD_MEMBER("vec3 greaterThan(const vec3&) const", value, vec3f, operator>, (const vec3f&) const);
    ASE_METHOD_MEMBER("vec3 greaterThanOrEquals(const vec3&) const", value, vec3f, operator>=, (const vec3f&) const);
    ASE_METHOD_MEMBER("vec3 lessThan(const vec3&) const", value, vec3f, operator<, (const vec3f&) const);
    ASE_METHOD_MEMBER("vec3 lessThanOrEquals(const vec3&) const", value, vec3f, operator<=, (const vec3f&) const);

    ASE_METHOD_MEMBER("float dot(const vec3&) const", value, float, dot, (const vec3f&) const);
    ASE_METHOD_MEMBER("vec3 cross(const vec3&) const", value, vec3f, cross, (const vec3f&) const);

    ASE_METHOD_MEMBER("float sum() const", value, float, sum, () const);
    ASE_METHOD_MEMBER("float prod() const", value, float, prod, () const);
    ASE_METHOD_MEMBER("float length() const", value, float, length, () const);
    ASE_METHOD_MEMBER("float length3() const", value, float, length, () const);
    ASE_METHOD_MEMBER("float lengthSquared() const", value, float, lengthSquared, () const);

    ASE_METHOD_MEMBER("vec3& normalize()", value, vec3f&, normalize, ());
    ASE_METHOD_MEMBER("vec3 normalized() const", value, vec3f, normalized, () const);
    ASE_METHOD_MEMBER("vec3 limit(float lim = 1.0f) const", value, vec3f, limit, (float) const);

    ASE_FUNCTION_STATIC("vec3 fastUnitRandom()", vec3f, vec3f::fast_unit_random, ());
    ASE_FUNCTION_STATIC_GLOBAL("vec3 min(const vec3&, const vec3&)", const vec3f, min, (const vec3f&, const vec3f&));
    ASE_FUNCTION_STATIC_GLOBAL("vec3 max(const vec3&, const vec3&)", const vec3f, max, (const vec3f&, const vec3f&));

    ASE_METHOD("float min() const", float, horizontalMin, () const);
    float horizontalMin() const { return horizontal_min(value); }
    ASE_METHOD("float max() const", float, horizontalMax, () const);
    float horizontalMax() const { return horizontal_min(value); }
    ASE_METHOD("float add() const", float, horizontalAdd, () const);
    float horizontalAdd() const { return add(value); }

    ASE_FUNCTION_STATIC_GLOBAL("vec3 abs(const vec3&)", const vec3f, abs, (const vec3f&));
    ASE_FUNCTION_STATIC_GLOBAL("vec3 exp(const vec3&)", const vec3f, exp, (const vec3f&));
    ASE_FUNCTION_STATIC_GLOBAL("vec3 sqrt(const vec3&)", const vec3f, sqrt, (const vec3f&));
    ASE_FUNCTION_STATIC_GLOBAL("vec3 sin(const vec3&)", const vec3f, sin, (const vec3f&));
    ASE_FUNCTION_STATIC_GLOBAL("vec3 cos(const vec3&)", const vec3f, cos, (const vec3f&));
    ASE_FUNCTION_STATIC_GLOBAL("vec3 approx_sqrt(const vec3&)", const vec3f, approx_sqrt, (const vec3f&));
    ASE_FUNCTION_STATIC_GLOBAL("vec3 sign(const vec3&)", const vec3f, sign, (const vec3f&));
    ASE_FUNCTION_STATIC_GLOBAL("vec3 step(const vec3&, const vec3&)", const vec3f, step, (const vec3f&, const vec3f&));
    ASE_FUNCTION_STATIC_GLOBAL("vec3 pow(const vec3&, float e)", const vec3f, pow, (const vec3f&, float));
    ASE_FUNCTION_STATIC_GLOBAL("vec3 trunc(const vec3&)", const vec3f, trunc, (const vec3f&));
    ASE_FUNCTION_STATIC_GLOBAL("vec3 floor(const vec3&)", const vec3f, floor, (const vec3f&));
    ASE_FUNCTION_STATIC_GLOBAL("vec3 ceil(const vec3&)", const vec3f, ceil, (const vec3f&));
    ASE_FUNCTION_STATIC_GLOBAL("vec3 modulo(const vec3&, const vec3&)", const vec3f, modulo, (const vec3f&, const vec3f&));

    vec3f value;
};

ASE_TYPE_DECLARATION(ScriptVec3);

class ScriptVec3Array : 
    public ASE_RefClass<>
{
    ASE_CLASS(ScriptVec3Array, "vec3array");
public:
    ASE_FACTORY_STATIC("()", ScriptVec3Array*, factory, ());
    static ScriptVec3Array* factory() { return new ScriptVec3Array(); };
    ScriptVec3Array() = default;
    ASE_FACTORY_STATIC("(const vec3array& other)", ScriptVec3Array*, factory, (const ScriptVec3Array&));
    static ScriptVec3Array* factory(const ScriptVec3Array& other) { return new ScriptVec3Array(other); };
    ScriptVec3Array(const ScriptVec3Array& other) = default;
    ASE_FACTORY_STATIC("(uint size)", ScriptVec3Array*, factory, (uint32_t));
    static ScriptVec3Array* factory(uint32_t size) { return new ScriptVec3Array(size); };
    ScriptVec3Array(uint32_t size) : values(size) {}
    ASE_FACTORY_STATIC("(const vec3array& other, uint pos, uint n)", ScriptVec3Array*, factory, (const ScriptVec3Array&, uint32_t, uint32_t));
    static ScriptVec3Array* factory(const ScriptVec3Array& other, uint32_t pos, uint32_t n) { return new ScriptVec3Array(other, pos, n); };
    ScriptVec3Array(const ScriptVec3Array& other, uint32_t pos, uint32_t n)
    { 
        if (pos < other.values.size())
        {
            uint32_t e = min(pos + n, (uint32_t)other.values.size());
            values.reserve(e - pos);
            for (uint32_t i = pos; i < e; ++i)
                values.push_back(other.values[i]);
        }
    }

    ASE_ADDREF(addRef);
    void addRef() { asAtomicInc(refCount); }
    ASE_RELEASE(release);
    void release() { if ( asAtomicDec(refCount) == 0) delete this; }

    ASE_METHOD("vec3array& opAssign(const vec3array& a)", opAssign);
    ScriptVec3Array& opAssign(const ScriptVec3Array& other) { values = other.values; return *this; }
    ASE_METHOD("vec3array@ createCopy(uint pos = 0, uint n = 0xffffffff)", ScriptVec3Array*, createCopy, (uint32_t, uint32_t));
    ScriptVec3Array* createCopy(uint32_t pos = 0, uint32_t n = -1) { return factory(*this, pos, n); }

    // Index operations
    ASE_METHOD("vec3& opIndex(uint i)", ScriptVec3&, operator[], (uint32_t));
    ScriptVec3& operator[](uint32_t i) { return values.at(i); }
    ASE_METHOD("const vec3& opIndex(uint i) const", const ScriptVec3&, operator[], (uint32_t) const);
    const ScriptVec3& operator[](uint32_t i) const { return values.at(i); }

    // Foreach support
    ASE_METHOD("uint opForBegin() const", opForBegin);
    uint32_t opForBegin() { return 0; }
    ASE_METHOD("bool opForEnd(uint i) const", opForEnd);
    bool opForEnd(uint32_t i) { return values.size() <= i; }
    ASE_METHOD("uint opForNext(uint) const", opForNext);
    uint32_t opForNext(uint32_t i) { return i + 1; }
    ASE_METHOD("vec3& opForValue0(uint i)", ScriptVec3&, opForValue0, (uint32_t));
    ScriptVec3& opForValue0(uint32_t i) { return values.at(i); }
    ASE_METHOD("const vec3& opForValue0(uint i) const", const ScriptVec3&, opForValue0, (uint32_t) const);
    const ScriptVec3& opForValue0(uint32_t i) const { return values.at(i); }
    ASE_METHOD("uint opForValue1(uint i) const", opForValue1);
    uint32_t opForValue1(uint32_t i) const { return i; }

    // std::vector operations
    ASE_METHOD("uint size() const", size);
    uint32_t size() const { return (uint32_t)values.size(); }
    ASE_METHOD_MEMBER("bool empty() const", values, empty);
    ASE_METHOD("void reserve(uint capacity)", reserve);
    void reserve(uint32_t capacity) { values.reserve(capacity); }
    ASE_METHOD("uint capacity() const", capacity);
    uint32_t capacity() const { return (uint32_t)values.capacity(); }
    ASE_METHOD_MEMBER("void clear()", values, clear);
    ASE_METHOD("void insert(uint i, const vec3& value)", void, insert, (uint32_t, const ScriptVec3&));
    void insert(uint32_t i, const ScriptVec3& v) { values.insert(values.begin() + i, v); }
    ASE_METHOD("void insert(uint i, uint nCopies, const vec3& value)", void, insert, (uint32_t, uint32_t, const ScriptVec3&));
    void insert(uint32_t i, uint32_t nCopies, const ScriptVec3& v) { values.insert(values.begin() + i, nCopies, v); }
    ASE_METHOD("void erase(uint i)", void, erase, (uint32_t));
    void erase(uint32_t i) { values.erase(values.begin() + i); }
    ASE_METHOD("void erase(uint iFirst, uint iLast)", void, erase, (uint32_t, uint32_t));
    void erase(uint32_t iFirst, uint32_t iLast) { values.erase(values.begin() + iFirst, values.begin() + iLast); }
    ASE_METHOD("void push_back(const vec3&)", push_back);
    void push_back(const ScriptVec3& v) { values.push_back(v); }
    ASE_METHOD_MEMBER("void pop_back()", values, pop_back);
    ASE_METHOD_MEMBER("vec3& front()", values, ScriptVec3&, front, ());
    ASE_METHOD_MEMBER("const vec3& front() const", values, const ScriptVec3&, front, () const);
    ASE_METHOD_MEMBER("vec3& back()", values, ScriptVec3&, back, ());
    ASE_METHOD_MEMBER("const vec3& back() const", values, const ScriptVec3&, back, () const);

    ASE_METHOD("vec3array& opAddAssign(const vec3& v)", ScriptVec3Array&, operator+=, (const ScriptVec3&));
    ScriptVec3Array& operator+=(const ScriptVec3& v) { for (auto& vv : values) vv.value += v.value; return *this; }
    ASE_METHOD("vec3array& opSubAssign(const vec3& v)", ScriptVec3Array&, operator-=, (const ScriptVec3&));
    ScriptVec3Array& operator-=(const ScriptVec3& v) { for (auto& vv : values) vv.value -= v.value; return *this; }
    ASE_METHOD("vec3array& opMulAssign(const vec3& v)", ScriptVec3Array&, operator*=, (const ScriptVec3&));
    ScriptVec3Array& operator*=(const ScriptVec3& v) { for (auto& vv : values) vv.value *= v.value; return *this;}
    ASE_METHOD("vec3array& opDivAssign(const vec3& v)", ScriptVec3Array&, operator/=, (const ScriptVec3&));
    ScriptVec3Array& operator/=(const ScriptVec3& v) { for (auto& vv : values) vv.value /= v.value; return *this;}

    ASE_METHOD("array<float>@ dot(const vec3& v) const", CScriptArray*, dot, (const ScriptVec3&) const);
    CScriptArray* dot(const ScriptVec3& v) const
    {
        CScriptArray* arr = CScriptArray::Create(getFloatArrayTypeInfo(), (uint32_t)values.size());
        for (asUINT i = 0; i < arr->GetSize(); ++i)
        {
            float val = values[i].value.dot(v.value);
            arr->SetValue(i, &val);
        }
        return arr;
    }

    ASE_METHOD("vec3array@ cross(const vec3&) const", ScriptVec3Array*, cross, (const ScriptVec3&) const);
    ScriptVec3Array* cross(const ScriptVec3& v) const
    {
        auto* newArray = factory((uint32_t)values.size());
        for (auto& vv : values)
            vv.value.cross(v.value);
        return newArray;
    }

    ASE_METHOD("array<float>@ sum() const", CScriptArray*, sum, () const);
    CScriptArray* sum() const
    {
        CScriptArray* arr = CScriptArray::Create(getFloatArrayTypeInfo(), (uint32_t)values.size());
        for (asUINT i = 0; i < arr->GetSize(); ++i)
        {
            float val = values[i].value.sum();
            arr->SetValue(i, &val);
        }
        return arr;
    }

    ASE_METHOD("array<float>@ prod() const", CScriptArray*, prod, () const);
    CScriptArray* prod() const
    {
        CScriptArray* arr = CScriptArray::Create(getFloatArrayTypeInfo(), (uint32_t)values.size());
        for (asUINT i = 0; i < arr->GetSize(); ++i)
        {
            float val = values[i].value.prod();
            arr->SetValue(i, &val);
        }
        return arr;
    }

    ASE_METHOD("array<float>@ length() const", CScriptArray*, length, () const);
    CScriptArray* length() const
    {
        CScriptArray* arr = CScriptArray::Create(getFloatArrayTypeInfo(), (uint32_t)values.size());
        for (asUINT i = 0; i < arr->GetSize(); ++i)
        {
            float val = values[i].value.length();
            arr->SetValue(i, &val);
        }
        return arr;
    }

    ASE_METHOD("array<float>@ lengthSquared() const", CScriptArray*, lengthSquared, () const);
    CScriptArray* lengthSquared() const
    {
        CScriptArray* arr = CScriptArray::Create(getFloatArrayTypeInfo(), (uint32_t)values.size());
        for (asUINT i = 0; i < arr->GetSize(); ++i)
        {
            float val = values[i].value.lengthSquared();
            arr->SetValue(i, &val);
        }
        return arr;
    }

    ASE_METHOD("vec3array@ normalized() const", ScriptVec3Array*, normalized, () const);
    ScriptVec3Array* normalized() const
    {
        auto* newArray = factory();
        newArray->reserve((uint32_t)values.size());
        for (auto& v : values)
            newArray->push_back(v.value.normalized());
        return newArray;
    }

    ASE_METHOD("vec3array& normalize()", ScriptVec3Array&, normalize, ());
    ScriptVec3Array& normalize() { for (auto& vv : values) vv.value.normalize(); return *this; }
    ASE_METHOD("vec3array& limit(float lim = 1.0f) const", ScriptVec3Array&, limit, (float));
    ScriptVec3Array& limit(float lim = 1.0f) { for (auto& vv : values) vv.value = vv.value.limit(lim); return *this; }
    ASE_FUNCTION_STATIC("vec3array@ fastUnitRandom(uint n)", ScriptVec3Array*, fastUnitRandom, (uint32_t));
    static ScriptVec3Array* fastUnitRandom(uint32_t n) 
    { 
        auto* newArray = factory();
        newArray->reserve(n);
        for (uint32_t i = 0; i < n; ++i)
            newArray->push_back(vec3f::fast_unit_random());
        return newArray;
    }

    ASE_METHOD("vec3array& abs()", ScriptVec3Array&, abs, ());
    ScriptVec3Array& abs() { for (auto& v : values) v.value = ::abs(v.value); return *this; }
    ASE_METHOD("vec3array& exp()", ScriptVec3Array&, exp, ());
    ScriptVec3Array& exp() { for (auto& v : values) v.value = ::exp(v.value); return *this; }
    ASE_METHOD("vec3array& sqrt()", ScriptVec3Array&, sqrt, ());
    ScriptVec3Array& sqrt() { for (auto& v : values) v.value = ::sqrt(v.value); return *this; }
    ASE_METHOD("vec3array& sin()", ScriptVec3Array&, sin, ());
    ScriptVec3Array& sin() { for (auto& v : values) v.value = ::sin(v.value); return *this; }
    ASE_METHOD("vec3array& cos()", ScriptVec3Array&, cos, ());
    ScriptVec3Array& cos() { for (auto& v : values) v.value = ::cos(v.value); return *this; }
    ASE_METHOD("vec3array& approx_sqrt()", ScriptVec3Array&, approx_sqrt, ());
    ScriptVec3Array& approx_sqrt() { for (auto& v : values) v.value = ::approx_sqrt(v.value); return *this; }
    ASE_METHOD("vec3array& sign()", ScriptVec3Array&, sign, ());
    ScriptVec3Array& sign() { for (auto& v : values) v.value = ::sign(v.value); return *this; }
    ASE_METHOD("vec3array& step(const vec3& v)", ScriptVec3Array&, step, (const ScriptVec3&));
    ScriptVec3Array& step(const ScriptVec3& s) { for (auto& v : values) v.value = ::step(v.value, s.value); return *this; }
    ASE_METHOD("vec3array& pow(float e)", ScriptVec3Array&, pow, (float e));
    ScriptVec3Array& pow(float e) { for (auto& v : values) v.value = ::pow(v.value, e); return *this; }
    ASE_METHOD("vec3array& trunc()", ScriptVec3Array&, trunc, ());
    ScriptVec3Array& trunc() { for (auto& v : values) v.value = ::trunc(v.value); return *this; }
    ASE_METHOD("vec3array& floor()", ScriptVec3Array&, floor, ());
    ScriptVec3Array& floor() { for (auto& v : values) v.value = ::floor(v.value); return *this; }
    ASE_METHOD("vec3array& ceil()", ScriptVec3Array&, ceil, ());
    ScriptVec3Array& ceil() { for (auto& v : values) v.value = ::ceil(v.value); return *this; }
    ASE_METHOD("vec3array& modulo(const vec3& v)", ScriptVec3Array&, modulo, (const ScriptVec3&));
    ScriptVec3Array& modulo(const ScriptVec3& m) { for (auto& v : values) v.value = ::modulo(v.value, m.value); return *this; }

    std::vector<ScriptVec3> values;

protected:
    static asITypeInfo* getFloatArrayTypeInfo()
    {
        static asITypeInfo* tiFloatArray;
        if (!tiFloatArray)
        {
            if (asIScriptContext* ctx = asGetActiveContext())
            {
                asIScriptEngine* engine = ctx->GetEngine();
                tiFloatArray = engine->GetTypeInfoByDecl("array<float>");
            }
        }
        return tiFloatArray;
    }

    mutable int refCount = 1;
};

ASE_TYPE_DECLARATION(ScriptVec3Array);


class ScriptVec4 : 
    public ASE_ValueClass<ScriptVec4>
{
    ASE_CLASS(ScriptVec4, "vec4");
public:
    ScriptVec4(const vec4f& v) { *vec() = v; }
    ScriptVec4(const Vec4f& v) { *vec() = v; }

    ASE_DESTRUCTOR;
    ~ScriptVec4() = default;
    ASE_CONSTRUCTOR("()");
    ScriptVec4() = default;
    ASE_CONSTRUCTOR("(const vec3& v)", const ScriptVec3&);
    ScriptVec4(const ScriptVec3& v) { *vec() = v.value; }
    ASE_CONSTRUCTOR("(const vec4& v)", const ScriptVec4&);
    ScriptVec4(const ScriptVec4& v) { *vec() = *v.vec(); }
    ASE_CONSTRUCTOR("(float v)", float);
    ScriptVec4(float f) { *vec() = f; }
    ASE_CONSTRUCTOR("(float x, float y, float z)", float, float, float);
    ScriptVec4(float x, float y, float z) { *vec() = vec4f(x, y, z); }
    ASE_CONSTRUCTOR("(float x, float y, float z, float w)", float, float, float, float);
    ScriptVec4(float x, float y, float z, float w) { *vec() = vec4f(x, y, z, w); }

    ASE_METHOD("float get_x() const property", get_x);
    float get_x() { return floats[offset]; }
    ASE_METHOD("void set_x(float x) property", set_x);
    void set_x(float x) { floats[offset] = x; }
    ASE_METHOD("float get_y() const property", get_y);
    float get_y() { return floats[offset + 1]; }
    ASE_METHOD("void set_y(float y) property", set_y);
    void set_y(float y) { floats[offset + 1] = y; }
    ASE_METHOD("float get_z() const property", get_z);
    float get_z() { return floats[offset + 2]; }
    ASE_METHOD("void set_z(float z) property", set_z);
    void set_z(float z) { floats[offset + 2] = z; }
    ASE_METHOD("float get_w() const property", get_w);
    float get_w() { return floats[offset + 3]; }
    ASE_METHOD("void set_w(float w) property", set_w);
    void set_w(float w) { floats[offset + 3] = w; }

    ASE_METHOD("vec4& opAssign(const vec4&)", ScriptVec4&, operator=, (const ScriptVec4&));
    ScriptVec4& operator=(const ScriptVec4& v) { *vec() = *v.vec(); return *this; }
    ASE_METHOD("vec4& opAddAssign(const vec4&)", ScriptVec4&, operator+=, (const ScriptVec4&));
    ScriptVec4& operator+=(const ScriptVec4& v) { *vec() += *v.vec(); return *this; }
    ASE_METHOD("vec4& opSubAssign(const vec4&)", ScriptVec4&, operator+=, (const ScriptVec4&));
    ScriptVec4& operator-=(const ScriptVec4& v) { *vec() -= *v.vec(); return *this; }
    ASE_METHOD("vec4& opMulAssign(const vec4&)", ScriptVec4&, operator+=, (const ScriptVec4&));
    ScriptVec4& operator*=(const ScriptVec4& v) { *vec() *= *v.vec(); return *this; }
    ASE_METHOD("vec4& opDivAssign(const vec4&)", ScriptVec4&, operator+=, (const ScriptVec4&));
    ScriptVec4& operator/=(const ScriptVec4& v) { *vec() /= *v.vec(); return *this; }

    ASE_METHOD("vec4 opAdd(const vec4&) const", ScriptVec4, operator+, (const ScriptVec4&) const);
    ScriptVec4 operator+(const ScriptVec4& v) const { return *vec() + *v.vec(); }
    ASE_METHOD("vec4 opSub(const vec4&) const", ScriptVec4, operator-, (const ScriptVec4&) const);
    ScriptVec4 operator-(const ScriptVec4& v) const { return *vec() - *v.vec(); }
    ASE_METHOD("vec4 opMul(const vec4&) const", ScriptVec4, operator*, (const ScriptVec4&) const);
    ScriptVec4 operator*(const ScriptVec4& v) const { return *vec() * *v.vec(); }
    ASE_METHOD("vec4 opDiv(const vec4&) const", ScriptVec4, operator/, (const ScriptVec4&) const);
    ScriptVec4 operator/(const ScriptVec4& v) const { return *vec() / *v.vec(); }
    ASE_METHOD("vec4 opNeg() const", ScriptVec4, operator-, () const);
    ScriptVec4 operator-() const { return - *vec(); }

    ASE_METHOD("vec4 equals(const vec4&) const", ScriptVec4, operator==, (const ScriptVec4&) const);
    ScriptVec4 operator==(const ScriptVec4& v) const { return *vec() == *v.vec(); }
    ASE_METHOD("vec4 greaterThan(const vec4&) const", ScriptVec4, operator>, (const ScriptVec4&) const);
    ScriptVec4 operator>(const ScriptVec4& v) const { return *vec() > *v.vec(); }
    ASE_METHOD("vec4 greaterThanOrEquals(const vec4&) const", ScriptVec4, operator>=, (const ScriptVec4&) const);
    ScriptVec4 operator>=(const ScriptVec4& v) const { return *vec() >= *v.vec(); }
    ASE_METHOD("vec4 lessThan(const vec4&) const", ScriptVec4, operator<, (const ScriptVec4&) const);
    ScriptVec4 operator<(const ScriptVec4& v) const { return *vec() < *v.vec(); }
    ASE_METHOD("vec4 lessThanOrEquals(const vec4&) const", ScriptVec4, operator<=, (const ScriptVec4&) const);
    ScriptVec4 operator<=(const ScriptVec4& v) const { return *vec() >= *v.vec(); }

    ASE_METHOD("float dot(const vec4&) const", float, dot, (const ScriptVec4&) const);
    float dot(const ScriptVec4& v) const { return vec()->dot(*v.vec()); }
    ASE_METHOD("vec4 cross(const vec4&) const", ScriptVec4, cross, (const ScriptVec4&) const);
    ScriptVec4 cross(const ScriptVec4& v) const { return vec()->dot(*v.vec()); }

    ASE_METHOD("float sum() const", float, sum, () const);
    float sum() const { return vec()->sum(); }
    ASE_METHOD("float prod() const", float, prod, () const);
    float prod() const { return vec()->prod(); }
    ASE_METHOD("float length() const", float, length, () const);
    float length() const { return vec()->length(); }
    ASE_METHOD("float lengthSquared() const", float, lengthSquared, () const);
    float lengthSquared() const { return vec()->lengthSquared(); }

    ASE_METHOD("vec4& normalize()", ScriptVec4&, normalize, ());
    ScriptVec4& normalize() { vec()->normalize(); return *this; }
    ASE_METHOD("vec4 normalized() const", ScriptVec4, normalized, () const);
    ScriptVec4 normalized() const { return vec()->normalized(); }
    ASE_METHOD("vec4 limit(float lim = 1.0f) const", ScriptVec4, limit, (float) const);
    ScriptVec4 limit(float lim = 1.0f) const { return vec()->limit(lim); }

    ASE_FUNCTION_STATIC("vec4 fastUnitRandom()", ScriptVec4, fastUnitRandom, ());
    static ScriptVec4 fastUnitRandom() { return vec4f::fast_unit_random(); }

    ASE_FUNCTION_STATIC_GLOBAL("vec4 min(const vec4&, const vec4&)", ScriptVec4, min, (const ScriptVec4&, const ScriptVec4&));
    static ScriptVec4 min(const ScriptVec4& v1, const ScriptVec4& v2) { return ::min(*v1.vec(), *v2.vec()); }
    ASE_FUNCTION_STATIC_GLOBAL("vec4 max(const vec4&, const vec4&)", ScriptVec4, max, (const ScriptVec4&, const ScriptVec4&));
    static ScriptVec4 max(const ScriptVec4& v1, const ScriptVec4& v2) { return ::max(*v1.vec(), *v2.vec()); }
    
    ASE_METHOD("float min() const", float, horizontalMin, () const);
    float horizontalMin() const { return horizontal_min(*vec()); }
    ASE_METHOD("float max() const", float, horizontalMax, () const);
    float horizontalMax() const { return horizontal_min(*vec()); }
    ASE_METHOD("float add() const", float, horizontalAdd, () const);
    float horizontalAdd() const { return horizontal_add(*vec()); }

    ASE_FUNCTION_STATIC_GLOBAL("vec4 abs(const vec4&)", ScriptVec4, abs, (const ScriptVec4&));
    static ScriptVec4 abs(const ScriptVec4& v) { return ::abs(*v.vec()); }
    ASE_FUNCTION_STATIC_GLOBAL("vec4 exp(const vec4&)", ScriptVec4, exp, (const ScriptVec4&));
    static ScriptVec4 exp(const ScriptVec4& v) { return ::exp(*v.vec()); }
    ASE_FUNCTION_STATIC_GLOBAL("vec4 sqrt(const vec4&)", ScriptVec4, sqrt, (const ScriptVec4&));
    static ScriptVec4 sqrt(const ScriptVec4& v) { return ::sqrt(*v.vec()); }
    ASE_FUNCTION_STATIC_GLOBAL("vec4 sin(const vec4&)", ScriptVec4, sin, (const ScriptVec4&));
    static ScriptVec4 sin(const ScriptVec4& v) { return ::sin(*v.vec()); }
    ASE_FUNCTION_STATIC_GLOBAL("vec4 cos(const vec4&)", ScriptVec4, cos, (const ScriptVec4&));
    static ScriptVec4 cos(const ScriptVec4& v) { return ::cos(*v.vec()); }
    ASE_FUNCTION_STATIC_GLOBAL("vec4 approx_sqrt(const vec4&)", ScriptVec4, approx_sqrt, (const ScriptVec4&));
    static ScriptVec4 approx_sqrt(const ScriptVec4& v) { return ::approx_sqrt(*v.vec()); }
    ASE_FUNCTION_STATIC_GLOBAL("vec4 sign(const vec4&)", ScriptVec4, sign, (const ScriptVec4&));
    static ScriptVec4 sign(const ScriptVec4& v) { return ::sign(*v.vec()); }
    ASE_FUNCTION_STATIC_GLOBAL("vec4 step(const vec4&, const vec4&)", ScriptVec4, step, (const ScriptVec4&, const ScriptVec4&));
    static ScriptVec4 step(const ScriptVec4& v1, const ScriptVec4& v2) { return ::step(*v1.vec(), *v2.vec()); }
    ASE_FUNCTION_STATIC_GLOBAL("vec4 pow(const vec4&, float e)", ScriptVec4, pow, (const ScriptVec4&, float));
    static ScriptVec4 pow(const ScriptVec4& v, float e) { return ::pow(*v.vec(), e); }
    ASE_FUNCTION_STATIC_GLOBAL("vec4 trunc(const vec4&)", ScriptVec4, trunc, (const ScriptVec4&));
    static ScriptVec4 trunc(const ScriptVec4& v) { return ::trunc(*v.vec()); }
    ASE_FUNCTION_STATIC_GLOBAL("vec4 floor(const vec4&)", ScriptVec4, floor, (const ScriptVec4&));
    static ScriptVec4 floor(const ScriptVec4& v) { return ::floor(*v.vec()); }
    ASE_FUNCTION_STATIC_GLOBAL("vec4 ceil(const vec4&)", ScriptVec4, ceil, (const ScriptVec4&));
    static ScriptVec4 ceil(const ScriptVec4& v) { return ::ceil(*v.vec()); }
    ASE_FUNCTION_STATIC_GLOBAL("vec4 modulo(const vec4&, const vec4&)", ScriptVec4, modulo, (const ScriptVec4&, const ScriptVec4&));
    static ScriptVec4 modulo(const ScriptVec4& v1, const ScriptVec4& v2) { return ::modulo(*v1.vec(), *v2.vec()); }

    forcedinline vec4f* vec() { return reinterpret_cast<vec4f*>(floats + offset); }
    forcedinline const vec4f* vec() const { return reinterpret_cast<const vec4f*>(floats + offset); }

protected:
    const uint32_t offset = (uint32_t)(((((size_t)floats + 15) & ~15) - (size_t)floats) >> 2);
    float floats[4 + 3];
};

ASE_TYPE_DECLARATION(ScriptVec4);

class ScriptMatrix4 : 
    public ASE_ValueClass<ScriptMatrix4>
{
    ASE_CLASS(ScriptMatrix4, "matrix4");
public:
    ScriptMatrix4(const Matrix4f& m) { *matrix() = m; }

    ASE_DESTRUCTOR;
    ~ScriptMatrix4() = default;
    ASE_CONSTRUCTOR("()");
    ScriptMatrix4() = default;
    ASE_CONSTRUCTOR("(const matrix4& other)", const ScriptMatrix4&);
    ScriptMatrix4(const ScriptMatrix4& other) { *matrix() = *other.matrix(); }
    ASE_CONSTRUCTOR("(int)", int);
    ScriptMatrix4(int) { matrix()->setIdentity(); }
    ASE_CONSTRUCTOR("(vec4 col0, vec4 col1, vec4 col2)", const ScriptVec4&, const ScriptVec4&, const ScriptVec4&);
    ScriptMatrix4(const ScriptVec4& col0, const ScriptVec4& col1, const ScriptVec4& col2) { *matrix() = Matrix4f(*col0.vec(), *col1.vec(), *col2.vec()); }
    ASE_CONSTRUCTOR("(vec4 col0, vec4 col1, vec4 col2, vec4 col3)", const ScriptVec4&, const ScriptVec4&, const ScriptVec4&, const ScriptVec4&);
    ScriptMatrix4(const ScriptVec4& col0, const ScriptVec4& col1, const ScriptVec4& col2, const ScriptVec4& col3) { *matrix() = Matrix4f(*col0.vec(), *col1.vec(), *col2.vec(), *col3.vec()); }

    ASE_METHOD("matrix4& opAssign(const matrix4&)", ScriptMatrix4&, operator=, (const ScriptMatrix4&));
    ScriptMatrix4& operator=(const ScriptMatrix4& m) { *matrix() = *m.matrix(); return *this; }
    ASE_METHOD("vec3 transformPoint(const vec3& p) const", transformPoint3);
    ScriptVec3 transformPoint3(const ScriptVec3& p) const { return matrix()->transformPoint(p.value); }
    ASE_METHOD("vec3array& transformPoints(vec3array& a, uint index = 0, uint items = 0xffffffff) const", transformPoints3);
    ScriptVec3Array& transformPoints3(ScriptVec3Array& a, uint32_t index = 0, uint32_t items = -1) const 
    { 
        for (uint32_t i = index, s = min(index + items, a.size()); i < s; ++i) 
            a[i].value = matrix()->transformPoint(a[i].value); 
        return a; 
    }
    ASE_METHOD("vec4 transformPoint(const vec4& p) const", transformPoint4);
    ScriptVec4 transformPoint4(const ScriptVec4& p) const { return matrix()->transformPoint(*p.vec()); }
    ASE_METHOD("vec3 transformPointCartesian(const vec3& p) const", transformPointCartesian3);
    ScriptVec3 transformPointCartesian3(const ScriptVec3& p) const { return matrix()->transformPointCartesian(p.value); }
    ASE_METHOD("vec3array& transformPointsCartesian(vec3array& a, uint index = 0, uint items = 0xffffffff) const", transformPointsCartesian3);
    ScriptVec3Array& transformPointsCartesian3(ScriptVec3Array& a, uint32_t index = 0, uint32_t items = -1) const
    { 
        for (uint32_t i = index, s = min(index + items, a.size()); i < s; ++i)
            a[i].value = matrix()->transformPointCartesian(a[i].value);
        return a; 
    }
    ASE_METHOD("vec4 transformPointCartesian(const vec4& p) const", transformPointCartesian4);
    ScriptVec4 transformPointCartesian4(const ScriptVec4& p) const { return matrix()->transformPointCartesian(*p.vec()); }
    ASE_METHOD("vec3 transformDir(const vec3& p) const", transformDir3);
    ScriptVec3 transformDir3(const ScriptVec3& p) const { return matrix()->transformDir(p.value); }
    ASE_METHOD("vec3array& transformDirs(vec3array& a, uint index = 0, uint items = 0xffffffff) const", transformDirs3);
    ScriptVec3Array& transformDirs3(ScriptVec3Array& a, uint32_t index = 0, uint32_t items = -1) const
    { 
        for (uint32_t i = index, s = min(index + items, a.size()); i < s; ++i)
            a[i].value = matrix()->transformDir(a[i].value);
        return a; 
    }
    ASE_METHOD("vec4 transformDir(const vec4& p) const", transformDir4);
    ScriptVec4 transformDir4(const ScriptVec4& p) const { return matrix()->transformDir(*p.vec()); }

    ASE_METHOD("matrix4& setIdentity()", setIdentity);
    ScriptMatrix4& setIdentity() { matrix()->setIdentity(); return *this; }
    ASE_METHOD("matrix4& setInvIdentity()", setInvIdentity);
    ScriptMatrix4& setInvIdentity() { matrix()->setInvIdentity(); return *this; }

    ASE_FUNCTION_STATIC("matrix4 identity()", identity);
    static ScriptMatrix4 identity() { return Matrix4f::identity(); }
    ASE_FUNCTION_STATIC("matrix4 translation(const vec3& p)", translation3);
    static ScriptMatrix4 translation3(const ScriptVec3& p) { return Matrix4f::translation(p.value); }
    ASE_FUNCTION_STATIC("matrix4 translation(const vec4& p)", translation4);
    static ScriptMatrix4 translation4(const ScriptVec4& p) { return Matrix4f::translation(*p.vec()); }
    ASE_FUNCTION_STATIC("matrix4 rotation(float radians, const vec3& axis)", rotation3);
    static ScriptMatrix4 rotation3(float radians, const ScriptVec3& axis) { return Matrix4f::rotation(radians, axis.value); }
    ASE_FUNCTION_STATIC("matrix4 rotation(float radians, const vec4& axis)", rotation4);
    static ScriptMatrix4 rotation4(float radians, const ScriptVec4& axis) { return Matrix4f::rotation(radians, *axis.vec()); }
    ASE_FUNCTION_STATIC("matrix4 scale(const vec3& s)", scale3);
    static ScriptMatrix4 scale3(const ScriptVec3& s) { return Matrix4f::scale(s.value); }
    ASE_FUNCTION_STATIC("matrix4 scale(const vec4& s)", scale4);
    static ScriptMatrix4 scale4(const ScriptVec4& s) { return Matrix4f::scale(*s.vec()); }
    ASE_FUNCTION_STATIC("matrix4 scale(float s)", scale);
    static ScriptMatrix4 scale(float s) { return Matrix4f::scale(s); }

    ASE_METHOD("vec4 getTranslation() const", getTranslation);
    ScriptVec4 getTranslation() const { return matrix()->getTranslation4(); }
    ASE_METHOD("matrix4& translateAbsolute(const vec3& p)", translateAbsolute3);
    ScriptMatrix4& translateAbsolute3(const ScriptVec3& p) { matrix()->translateAbsolute(p.value); return *this; }
    ASE_METHOD("matrix4& translateAbsolute(const vec4& p)", translateAbsolute4);
    ScriptMatrix4& translateAbsolute4(const ScriptVec4& p) { matrix()->translateAbsolute(*p.vec()); return *this; }
    ASE_METHOD("matrix4& thenTranslate(const vec3& p)", thenTranslate3);
    ScriptMatrix4& thenTranslate3(const ScriptVec3& p) { matrix()->thenTranslate(p.value); return *this; }
    ASE_METHOD("matrix4& thenTranslate(const vec4& p)", thenTranslate4);
    ScriptMatrix4& thenTranslate4(const ScriptVec4& p) { matrix()->thenTranslate(*p.vec()); return *this; }

    ASE_METHOD("vec4 getScale() const", getScale);
    ScriptVec4 getScale() const { return matrix()->getScale4(); }
    ASE_METHOD("matrix4& scaleAbsolute(const vec3& s)", scaleAbsolute3);
    ScriptMatrix4& scaleAbsolute3(const ScriptVec3& p) { matrix()->scaleAbsolute(p.value); return *this; }
    ASE_METHOD("matrix4& scaleAbsolute(const vec4& s)", scaleAbsolute4);
    ScriptMatrix4& scaleAbsolute4(const ScriptVec4& p) { matrix()->scaleAbsolute(*p.vec()); return *this; }
    ASE_METHOD("matrix4& thenScale(const vec3& s)", thenScale3);
    ScriptMatrix4& thenScale3(const ScriptVec3& p) { matrix()->thenScale(p.value); return *this; }
    ASE_METHOD("matrix4& thenScale(const vec4& s)", thenScale4);
    ScriptMatrix4& thenScale4(const ScriptVec4& p) { matrix()->thenScale(*p.vec()); return *this; }

    ASE_METHOD("vec4 getEulerRotation() const", getEulerRotation);
    ScriptVec4 getEulerRotation() const { return matrix()->getEulerRotation4(); }
    ASE_METHOD("matrix4& rotateAbsolute(float radians, const vec3& axis)", rotateAbsolute3);
    ScriptMatrix4& rotateAbsolute3(float radians, const ScriptVec3& axis) { matrix()->rotateAbsolute(radians, axis.value); return *this; }
    ASE_METHOD("matrix4& rotateAbsolute(float radians, const vec4& axis)", rotateAbsolute4);
    ScriptMatrix4& rotateAbsolute4(float radians, const ScriptVec4& axis) { matrix()->rotateAbsolute(radians, *axis.vec()); return *this; }
    ASE_METHOD("matrix4& thenRotate(float radians, const vec3& axis)", thenRotate3);
    ScriptMatrix4& thenRotate3(float radians, const ScriptVec3& axis) { matrix()->thenRotate(radians, axis.value); return *this; }
    ASE_METHOD("matrix4& thenRotate(float radians, const vec4& axis)", thenRotate4);
    ScriptMatrix4& thenRotate4(float radians, const ScriptVec4& axis) { matrix()->thenRotate(radians, *axis.vec()); return *this; }

    ASE_METHOD("matrix4& transpose()", transpose);
    ScriptMatrix4& transpose() { matrix()->transpose(); return *this; }
    ASE_METHOD("matrix4 transposed() const", transposed);
    ScriptMatrix4 transposed() const { return matrix()->transposed(); }
    ASE_METHOD("vec4 multiply(const vec4& v) const", ScriptVec4, multiply, (const ScriptVec4&) const);
    ScriptVec4 multiply(const ScriptVec4& v) const { return matrix()->multiply(*v.vec()); }
    ASE_METHOD("matrix4& multiply(const matrix4& m)", ScriptMatrix4&, multiply, (const ScriptMatrix4&));
    ScriptMatrix4& multiply(const ScriptMatrix4& m) { matrix()->multiply(*m.matrix()); return *this; }
    ASE_METHOD("matrix4 multipliedWith(const matrix4& m) const", multipliedWith);
    ScriptMatrix4 multipliedWith(const ScriptMatrix4& m) const { return matrix()->multipliedWith(*m.matrix()); }

    ASE_METHOD("matrix4 inversedNoScale() const", inversedNoScale);
    ScriptMatrix4 inversedNoScale() const { return matrix()->inversedNoScale(); }
    ASE_METHOD("matrix4& inverseNoScale()", inverseNoScale);
    ScriptMatrix4& inverseNoScale() { matrix()->inverseNoScale(); return *this; }
    ASE_METHOD("matrix4 inversed() const", inversed);
    ScriptMatrix4 inversed() const { return matrix()->inversed(); }
    ASE_METHOD("matrix4& inverse()", inverse);
    ScriptMatrix4& inverse() { matrix()->inverse(); return *this; }

    ASE_METHOD("matrix4& rotationAlign(const vec3& d, const vec3& z)", rotationAlign3);
    ScriptMatrix4& rotationAlign3(const ScriptVec3& d, const ScriptVec3& z) { matrix()->rotationAlign(d.value, z.value); return *this; }
    ASE_METHOD("matrix4& rotationAlign(const vec4& d, const vec4& z)", rotationAlign4);
    ScriptMatrix4& rotationAlign4(const ScriptVec4& d, const ScriptVec4& z) { matrix()->rotationAlign(*d.vec(), *z.vec()); return *this; }

    ASE_METHOD("vec4 get_col(int i) const property", get_col);
    ScriptVec4 get_col(int i) const { return const_cast<Matrix4f*>(matrix())->column(i); }
    ASE_METHOD("void set_col(int i, const vec4& c) property", set_col);
    void set_col(int i, const ScriptVec4& c) { matrix()->column(i) = (*c.vec()).vcl; }

    forcedinline Matrix4f* matrix() { return reinterpret_cast<Matrix4f*>(floats + offset); }
    forcedinline const Matrix4f* matrix() const { return reinterpret_cast<const Matrix4f*>(floats + offset); }

private:
    const uint32_t offset = (uint32_t)(((((size_t)floats + 15) & ~15) - (size_t)floats) >> 2);
    float floats[16 + 3];
};

ASE_TYPE_DECLARATION(ScriptMatrix4);

// Reverse dependent 
inline ScriptVec3::ScriptVec3(const ScriptVec4& v) : value(v.vec()->xyz()) {}
