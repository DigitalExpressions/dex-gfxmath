#pragma once

#include "TransformTests.h"

namespace gfxmath_tests
{
    inline ReferenceMatrix referenceAxisRotation(double angle, const std::array<double, 3>& axis)
    {
        ReferenceMatrix result{};
        const double length = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
        const std::array<double, 3> unit{axis[0] / length, axis[1] / length, axis[2] / length};
        const std::array<std::array<double, 3>, 3> cross{{{0, -unit[2], unit[1]}, {unit[2], 0, -unit[0]}, {-unit[1], unit[0], 0}}};
        const double cosine = std::cos(-angle);
        const double sine = std::sin(-angle);

        for (size_t row = 0; row < 3; ++row)
        {
            for (size_t column = 0; column < 3; ++column)
            {
                result[row][column] = (row == column ? cosine : 0) + (1 - cosine) * unit[row] * unit[column] + sine * cross[row][column];
            }
        }

        result[3][3] = 1;
        return result;
    }

    template <typename T> void testMatrixOverloads(TestContext& context)
    {
        using Matrix = dex::Matrix4<T>;
        using NativeColumn = dex::Vector_t<T, 4>;
        using V3 = dex::vec3<T>;
        using V4 = dex::vec4<T>;
        context.group(std::is_same_v<T, float> ? "Matrix4 float overload coverage" : "Matrix4 double overload coverage");
        const double error = tolerance<T>() * 8;
        const ReferenceMatrix identity{{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}};
        const ReferenceMatrix reference{{{1, 5, 9, 13}, {2, 6, 10, 14}, {3, 7, 11, 15}, {4, 8, 12, 16}}};
        const NativeColumn column0(1, 2, 3, 4);
        const NativeColumn column1(5, 6, 7, 8);
        const NativeColumn column2(9, 10, 11, 12);
        const NativeColumn column3(13, 14, 15, 16);
        const Matrix original(column0, column1, column2, column3);
        expectMatrix(context, original, reference, 0, "native four-column construction");
        ReferenceMatrix defaultColumnReference = reference;

        for (int row = 0; row < 4; ++row)
        {
            defaultColumnReference[row][3] = row == 3 ? 1 : 0;
        }

        expectMatrix(context, Matrix(column0, column1, column2), defaultColumnReference, 0,
            "native three-column construction supplies affine fourth column");
        expectMatrix(context, Matrix(V4(1, 2, 3, 4), V4(5, 6, 7, 8), V4(9, 10, 11, 12)), defaultColumnReference, 0,
            "vec4 three-column construction supplies affine fourth column");
        ReferenceMatrix threeComponentReference = defaultColumnReference;
        threeComponentReference[3] = {0, 0, 0, 1};
        expectMatrix(context, Matrix(V3(1, 2, 3), V3(5, 6, 7), V3(9, 10, 11)), threeComponentReference, 0,
            "vec3 basis construction supplies homogeneous entries");
        expectMatrix(context, Matrix(original), reference, 0, "matrix copy construction");
        Matrix mutableMatrix;
        context.expect(&(mutableMatrix = original) == &mutableMatrix, "matrix assignment returns self");
        expectMatrix(context, mutableMatrix, reference, 0, "matrix assignment writes every column");
        mutableMatrix = mutableMatrix;
        expectMatrix(context, mutableMatrix, reference, 0, "matrix self-assignment preserves columns");
        context.expect(&mutableMatrix.setIdentity() == &mutableMatrix, "setIdentity returns self");
        expectMatrix(context, mutableMatrix, identity, 0, "default construction followed by explicit identity initialization");

        mutableMatrix = original;
        ReferenceMatrix modified = reference;

        for (int index = -7; index <= 8; ++index)
        {
            const int column = index & 3;

            for (int row = 0; row < 4; ++row)
            {
                context.near(mutableMatrix.column(index).extract(row), modified[row][column], 0,
                    "column access wraps positive and negative indices");
            }

            const int row = (index + 7) % 4;
            const T replacement = T(100 + index);
            mutableMatrix.column(index).insert(row, replacement);
            modified[row][column] = replacement;
            expectMatrix(context, mutableMatrix, modified, 0, "column access returns mutable native column");
        }

        ReferenceMatrix transposed{};

        for (int row = 0; row < 4; ++row)
        {
            for (int column = 0; column < 4; ++column)
            {
                transposed[row][column] = reference[column][row];
            }
        }

        mutableMatrix = original;
        context.expect(&mutableMatrix.transpose() == &mutableMatrix, "in-place transpose returns self");
        expectMatrix(context, mutableMatrix, transposed, 0, "in-place transpose scalar array oracle");
        mutableMatrix.transpose();
        expectMatrix(context, mutableMatrix, reference, 0, "double transpose restores original matrix");
        const std::array<double, 4> input{1.25, -2.5, 3.75, 0.5};
        const auto product = referenceMultiply(reference, input);
        const auto nativeResult = original.multiply(NativeColumn(T(input[0]), T(input[1]), T(input[2]), T(input[3])));

        for (int row = 0; row < 4; ++row)
        {
            context.near(nativeResult.extract(row), product[row], error, "native-column multiply scalar dot-product oracle");
        }

        T suppliedW = T(-7.5);
        const auto withSuppliedW = original.multiply(V4(T(input[0]), T(input[1]), T(input[2]), T(input[3])), suppliedW);
        expectVector(
            context, withSuppliedW,
            [&](int componentIndex, int)
            {
                return componentIndex == 3 ? double(suppliedW) : product[componentIndex];
            },
            error, "multiply overload replaces output w with supplied value");
        context.near(suppliedW, T(-7.5), 0, "multiply supplied-w argument is preserved");

        const V3 position(2, -3, 4);
        const V4 position4(2, -3, 4, 99);
        const ReferenceMatrix translation{{{1, 0, 0, 2}, {0, 1, 0, -3}, {0, 0, 1, 4}, {0, 0, 0, 1}}};
        expectMatrix(context, Matrix::translation(position4), translation, 0, "vec4 translation factory ignores input w");
        modified = reference;

        for (int row = 0; row < 4; ++row)
        {
            modified[row][3] = translation[row][3];
        }

        mutableMatrix = original;
        context.expect(&mutableMatrix.translateAbsolute(position) == &mutableMatrix, "vec3 translateAbsolute returns self");
        expectMatrix(context, mutableMatrix, modified, 0, "vec3 translateAbsolute preserves basis columns");
        mutableMatrix = original;
        context.expect(&mutableMatrix.translateAbsolute(position4) == &mutableMatrix, "vec4 translateAbsolute returns self");
        expectMatrix(context, mutableMatrix, modified, 0, "vec4 translateAbsolute sets affine w");
        expectVector(
            context, mutableMatrix.getTranslation4(),
            [](int componentIndex, int)
            {
                constexpr double values[]{2, -3, 4, 1};
                return values[componentIndex];
            },
            0, "getTranslation4 returns entire translation column");

        for (const bool useVec4 : {false, true})
        {
            mutableMatrix = original;
            Matrix* returned = useVec4 ? &mutableMatrix.thenTranslate(position4) : &mutableMatrix.thenTranslate(position);
            context.expect(returned == &mutableMatrix, "thenTranslate overload returns self");
            expectMatrix(context, mutableMatrix, referenceProduct(reference, translation), error,
                "thenTranslate overload composes on right including homogeneous row");
        }

        const V3 scales(2, 3, 4);
        const V4 scales4(2, 3, 4, 77);
        const ReferenceMatrix scaling{{{2, 0, 0, 0}, {0, 3, 0, 0}, {0, 0, 4, 0}, {0, 0, 0, 1}}};
        const ReferenceMatrix uniformScaling{{{2.5, 0, 0, 0}, {0, 2.5, 0, 0}, {0, 0, 2.5, 0}, {0, 0, 0, 1}}};
        expectMatrix(context, Matrix::scale(scales), scaling, 0, "vec3 scale factory");
        expectMatrix(context, Matrix::scale(scales4), scaling, 0, "vec4 scale factory ignores w");
        expectMatrix(context, Matrix::scale(T(2.5)), uniformScaling, 0, "scalar scale factory preserves homogeneous one");
        modified = reference;
        modified[0][0] = 2;
        modified[1][1] = 3;
        modified[2][2] = 4;

        for (const bool useVec4 : {false, true})
        {
            mutableMatrix = original;
            Matrix* returned = useVec4 ? &mutableMatrix.scaleAbsolute(scales4) : &mutableMatrix.scaleAbsolute(scales);
            context.expect(returned == &mutableMatrix, "scaleAbsolute overload returns self");
            expectMatrix(context, mutableMatrix, modified, 0, "scaleAbsolute replaces diagonal and preserves other entries");
            expectVector(
                context, mutableMatrix.getScale(),
                [](int componentIndex, int)
                {
                    return componentIndex + 2;
                },
                0, "getScale reads three diagonal entries");
            expectVector(
                context, mutableMatrix.getScale4(),
                [](int componentIndex, int)
                {
                    return componentIndex == 3 ? 1 : componentIndex + 2;
                },
                0, "getScale4 appends homogeneous one");
            mutableMatrix = original;
            returned = useVec4 ? &mutableMatrix.thenScale(scales4) : &mutableMatrix.thenScale(scales);
            context.expect(returned == &mutableMatrix, "thenScale vector overload returns self");
            expectMatrix(
                context, mutableMatrix, referenceProduct(reference, scaling), error, "thenScale vector overload composes on right");
        }

        mutableMatrix = original;
        context.expect(&mutableMatrix.thenScale(T(2.5)) == &mutableMatrix, "thenScale scalar returns self");
        expectMatrix(context, mutableMatrix, referenceProduct(reference, uniformScaling), error, "thenScale scalar composes on right");
        const T angle = T(0.625);
        const V3 axis(1, -2, 3);
        const V4 axis4(1, -2, 3, 91);
        const auto rotation = referenceAxisRotation(double(angle), {1, -2, 3});
        expectMatrix(context, Matrix::rotation(angle, axis), rotation, error, "vec3 rotation factory normalizes axis");
        expectMatrix(context, Matrix::rotation(angle, axis4), rotation, error, "vec4 rotation factory ignores w");
        auto absoluteRotation = rotation;

        for (int row = 0; row < 4; ++row)
        {
            absoluteRotation[row][3] = reference[row][3];
        }

        for (const bool useVec4 : {false, true})
        {
            mutableMatrix = original;
            Matrix* returned = useVec4 ? &mutableMatrix.rotateAbsolute(angle, axis4) : &mutableMatrix.rotateAbsolute(angle, axis);
            context.expect(returned == &mutableMatrix, "rotateAbsolute overload returns self");
            expectMatrix(context, mutableMatrix, absoluteRotation, error, "rotateAbsolute replaces basis and preserves translation");
            mutableMatrix = original;
            returned = useVec4 ? &mutableMatrix.thenRotate(angle, axis4) : &mutableMatrix.thenRotate(angle, axis);
            context.expect(returned == &mutableMatrix, "thenRotate overload returns self");
            expectMatrix(
                context, mutableMatrix, referenceProduct(reference, rotation), error, "thenRotate composes normalized arbitrary axis");
        }

        const Matrix other = Matrix::translation(position4);
        expectMatrix(context, original.multipliedWith(other), referenceProduct(reference, translation), error,
            "multipliedWith returns independent product");
        expectMatrix(context, original, reference, 0, "multipliedWith preserves source");
        mutableMatrix = original;
        context.expect(&mutableMatrix.multiply(other) == &mutableMatrix, "matrix multiplication returns self");
        expectMatrix(context, mutableMatrix, referenceProduct(reference, translation), error, "matrix in-place multiplication");
        mutableMatrix = original;
        mutableMatrix.multiply(mutableMatrix);
        expectMatrix(
            context, mutableMatrix, referenceProduct(reference, reference), error, "matrix self-multiplication reads original columns");
    }

    template <typename T> void testQuaternionOverloads(TestContext& context, std::string_view typeName)
    {
        using Q = dex::Quaternion<T>;
        using V = dex::vec4<T>;
        using S = Scalar<T>;
        context.group(std::string("Quaternion overloads / ") + std::string(typeName));
        const V input = makeVector<V>(
            [](int componentIndex, int laneIndex)
            {
                return (componentIndex + 1) * (laneIndex + 2) * 0.25 - 1.5;
            });
        const Q original(input);
        const auto expectedOriginal = [&](int componentIndex, int laneIndex)
        {
            return (componentIndex + 1) * (laneIndex + 2) * 0.25 - 1.5;
        };
        const double error = tolerance<T>() * 4;
        expectVector(context, original.q, expectedOriginal, 0, "quaternion vec4 construction");
        expectVector(context, Q(original).q, expectedOriginal, 0, "quaternion copy construction");
        Q mutableQuaternion;
        context.expect(&(mutableQuaternion = original) == &mutableQuaternion, "quaternion assignment returns self");
        expectVector(context, mutableQuaternion.q, expectedOriginal, 0, "quaternion assignment copies every lane");
        mutableQuaternion = mutableQuaternion;
        expectVector(context, mutableQuaternion.q, expectedOriginal, 0, "quaternion self-assignment");
        const auto exactUnit = [&](int componentIndex, int laneIndex)
        {
            double squaredLength = 0;

            for (int coordinate = 0; coordinate < 4; ++coordinate)
            {
                const double value = expectedOriginal(coordinate, laneIndex);
                squaredLength += value * value;
            }

            return expectedOriginal(componentIndex, laneIndex) / std::sqrt(squaredLength);
        };
        expectVector(context, original.normalizedx().q, exactUnit, error, "quaternion normalizedx exact-length oracle");
        context.expect(&mutableQuaternion.normalizex() == &mutableQuaternion, "quaternion normalizex returns self");
        expectVector(context, mutableQuaternion.q, exactUnit, error, "quaternion normalizex mutates every lane");
        mutableQuaternion = original;
        context.expect(&mutableQuaternion.normalize() == &mutableQuaternion, "quaternion normalize returns self");
        expectVector(
            context, mutableQuaternion.q,
            [&](int componentIndex, int laneIndex)
            {
                double squaredLength = 0;

                for (int coordinate = 0; coordinate < 4; ++coordinate)
                {
                    const double value = expectedOriginal(coordinate, laneIndex);
                    squaredLength += value * value;
                }

                return expectedOriginal(componentIndex, laneIndex) / (std::sqrt(squaredLength) + dex::Epsilon<S>);
            },
            error, "quaternion normalize epsilon convention");
        const Q zero(T(0), T(0), T(0), T(0));
        const auto expectedZero = [](int, int)
        {
            return 0;
        };
        expectVector(context, zero.normalizedx().q, expectedZero, 0, "zero quaternion normalizedx remains finite");
        mutableQuaternion = zero;
        mutableQuaternion.normalizex();
        expectVector(context, mutableQuaternion.q, expectedZero, 0, "zero quaternion normalizex remains finite");
        expectVector(context, zero.normalized().q, expectedZero, 0, "zero quaternion normalized remains finite");
        mutableQuaternion = zero;
        mutableQuaternion.normalize();
        expectVector(context, mutableQuaternion.q, expectedZero, 0, "zero quaternion normalize remains finite");

        for (int index = 0; index < laneCount<T>(); ++index)
        {
            context.near(lane(original.x, index), expectedOriginal(0, index), 0, "quaternion x field alias");
            context.near(lane(original.y, index), expectedOriginal(1, index), 0, "quaternion y field alias");
            context.near(lane(original.z, index), expectedOriginal(2, index), 0, "quaternion z field alias");
            context.near(lane(original.w, index), expectedOriginal(3, index), 0, "quaternion w field alias");
        }
    }

    inline void runTransformCoverageTests(TestContext& context)
    {
        testMatrixOverloads<float>(context);
        testMatrixOverloads<double>(context);
        forFloatingTypes(
            [&]<typename T>(std::string_view name)
            {
                testQuaternionOverloads<T>(context, name);
            });
    }
}
