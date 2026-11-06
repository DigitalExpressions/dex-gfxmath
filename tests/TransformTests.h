#pragma once

#include "TestSupport.h"
#include "VectorTests.h"

namespace gfxmath_tests
{
    using ReferenceMatrix = std::array<std::array<double, 4>, 4>;

    inline std::array<double, 4> referenceMultiply(const ReferenceMatrix& matrix, const std::array<double, 4>& vector)
    {
        std::array<double, 4> result{};

        for (int row = 0; row < 4; ++row)
        {
            for (int column = 0; column < 4; ++column)
            {
                result[row] += matrix[row][column] * vector[column];
            }
        }

        return result;
    }

    inline ReferenceMatrix referenceProduct(const ReferenceMatrix& first, const ReferenceMatrix& second)
    {
        ReferenceMatrix result{};

        for (int row = 0; row < 4; ++row)
        {
            for (int column = 0; column < 4; ++column)
            {
                for (int inner = 0; inner < 4; ++inner)
                {
                    result[row][column] += first[row][inner] * second[inner][column];
                }
            }
        }

        return result;
    }

    // Independent pivoted Gauss-Jordan oracle; test inputs are nonsingular.
    inline ReferenceMatrix referenceInverse(ReferenceMatrix input)
    {
        ReferenceMatrix inverse{};

        for (int row = 0; row < 4; ++row)
        {
            inverse[row][row] = 1;
        }

        for (int column = 0; column < 4; ++column)
        {
            int pivot = column;

            for (int row = column + 1; row < 4; ++row)
            {
                if (std::abs(input[row][column]) > std::abs(input[pivot][column]))
                {
                    pivot = row;
                }
            }

            std::swap(input[column], input[pivot]);
            std::swap(inverse[column], inverse[pivot]);
            const double divisor = input[column][column];

            for (int entry = 0; entry < 4; ++entry)
            {
                input[column][entry] /= divisor;
                inverse[column][entry] /= divisor;
            }

            for (int row = 0; row < 4; ++row)
            {
                if (row == column)
                {
                    continue;
                }

                const double factor = input[row][column];

                for (int entry = 0; entry < 4; ++entry)
                {
                    input[row][entry] -= factor * input[column][entry];
                    inverse[row][entry] -= factor * inverse[column][entry];
                }
            }
        }

        return inverse;
    }

    template <typename T> ReferenceMatrix referenceMatrix(const dex::Matrix4<T>& matrix)
    {
        ReferenceMatrix result{};

        for (int row = 0; row < 4; ++row)
        {
            result[row][0] = matrix.col0.extract(row);
            result[row][1] = matrix.col1.extract(row);
            result[row][2] = matrix.col2.extract(row);
            result[row][3] = matrix.col3.extract(row);
        }

        return result;
    }

    template <typename T>
    void expectMatrix(
        TestContext& context, const dex::Matrix4<T>& actual, const ReferenceMatrix& expected, double error, std::string_view message)
    {
        const auto observed = referenceMatrix(actual);

        for (int row = 0; row < 4; ++row)
        {
            for (int column = 0; column < 4; ++column)
            {
                context.near(observed[row][column], expected[row][column], error, message);
            }
        }
    }

    template <typename T> void testMatrices(TestContext& context, std::string_view typeName)
    {
        context.group("Matrix4<" + std::string(typeName) + ">");
        using Matrix = dex::Matrix4<T>;
        using V3 = dex::vec3<T>;
        using V4 = dex::vec4<T>;
        const double error = tolerance<T>() * 4;
        const ReferenceMatrix identity{{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}};
        expectMatrix(context, Matrix::identity(), identity, 0, "identity");
        Matrix negative(1);
        negative.setInvIdentity();
        ReferenceMatrix negativeIdentity = identity;

        for (int i = 0; i < 4; ++i)
        {
            negativeIdentity[i][i] = -1;
        }

        expectMatrix(context, negative, negativeIdentity, 0, "negative identity");
        const Matrix arbitrary(V4(1, 2, 3, 4), V4(5, 6, 7, 8), V4(9, 10, 11, 12), V4(13, 14, 15, 16));
        const auto arbitraryReference = referenceMatrix(arbitrary);
        ReferenceMatrix transpose{};

        for (int row = 0; row < 4; ++row)
        {
            for (int column = 0; column < 4; ++column)
            {
                transpose[row][column] = arbitraryReference[column][row];
            }
        }

        expectMatrix(context, arbitrary.transposed(), transpose, 0, "transpose independent oracle");
        const auto homogeneous = arbitrary.multiply(V4(2, -3, 4, 0.5));
        const auto homogeneousReference = referenceMultiply(arbitraryReference, {2, -3, 4, 0.5});
        expectVector(
            context, homogeneous,
            [&](int d, int)
            {
                return homogeneousReference[d];
            },
            error, "general homogeneous multiply");

        std::mt19937 generator(0x75143u);
        std::uniform_real_distribution<double> position(-12, 12);
        std::uniform_real_distribution<double> angles(-3, 3);

        for (int iteration = 0; iteration < 80; ++iteration)
        {
            const T tx = static_cast<T>(position(generator));
            const T ty = static_cast<T>(position(generator));
            const T tz = static_cast<T>(position(generator));
            const T angle = static_cast<T>(angles(generator));
            const T sx = T(0.5 + std::abs(position(generator)));
            const T sy = T(0.5 + std::abs(position(generator)));
            const T sz = T(0.5 + std::abs(position(generator)));
            const T sine = std::sin(-angle);
            const T cosine = std::cos(angle);
            const ReferenceMatrix rotation{{{cosine, -sine, 0, 0}, {sine, cosine, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}};
            const ReferenceMatrix translation{{{1, 0, 0, tx}, {0, 1, 0, ty}, {0, 0, 1, tz}, {0, 0, 0, 1}}};
            const ReferenceMatrix scale{{{sx, 0, 0, 0}, {0, sy, 0, 0}, {0, 0, sz, 0}, {0, 0, 0, 1}}};
            Matrix rigid = Matrix::translation(V3(tx, ty, tz));
            rigid.thenRotate(angle, V3(0, 0, 1));
            const auto rigidReference = referenceProduct(translation, rotation);
            expectMatrix(context, rigid, rigidReference, error, "translation rotation composition");
            Matrix transform = rigid;
            transform.thenScale(V3(sx, sy, sz));
            const auto transformReference = referenceProduct(rigidReference, scale);
            expectMatrix(context, transform, transformReference, error, "TRS independent oracle");
            expectMatrix(context, transform.inversed(), referenceInverse(transformReference), error,
                "orthogonal affine inverse against Gauss-Jordan");
            expectMatrix(context, rigid.inversedNoScale(), referenceInverse(rigidReference), error, "rigid inverse against Gauss-Jordan");
            Matrix inPlace = transform;
            inPlace.inverse();
            expectMatrix(context, inPlace, referenceInverse(transformReference), error, "in-place inverse");
            inPlace = rigid;
            inPlace.inverseNoScale();
            expectMatrix(context, inPlace, referenceInverse(rigidReference), error, "in-place rigid inverse");

            const V3 point(T(1.25), T(-2.5), T(3.75));
            const auto pointReference = referenceMultiply(transformReference, {1.25, -2.5, 3.75, 1});
            const auto directionReference = referenceMultiply(transformReference, {1.25, -2.5, 3.75, 0});
            expectVector(
                context, transform.transformPoint(point),
                [&](int d, int)
                {
                    return pointReference[d];
                },
                error, "3D point includes translation");
            expectVector(
                context, transform.transformPoint(V4(point, T(9))),
                [&](int d, int)
                {
                    return pointReference[d];
                },
                error, "4D point forces w=1");
            expectVector(
                context, transform.transformDir(point),
                [&](int d, int)
                {
                    return directionReference[d];
                },
                error, "direction excludes translation");
            expectVector(
                context, transform.transformDir(V4(point, T(9))),
                [&](int d, int)
                {
                    return directionReference[d];
                },
                error, "4D direction forces w=0");
            expectVector(
                context, transform.transformPointCartesian(point),
                [&](int d, int)
                {
                    return pointReference[d];
                },
                error, "Cartesian affine point");
            expectVector(
                context, transform.getTranslation(),
                [&](int d, int)
                {
                    return d == 0 ? tx : d == 1 ? ty : tz;
                },
                error, "getTranslation");

            using Packed = dex::Vector_t<T, 4>;
            const auto packedPoints = makeVector<dex::vec3<Packed>>(
                [](int d, int i)
                {
                    return 0.75 + d * 2.0 - i * 0.5;
                });
            expectVector(
                context, transform.template transformDir<T>(packedPoints),
                [&](int d, int i)
                {
                    const auto expected = referenceMultiply(transformReference, {0.75 - i * 0.5, 2.75 - i * 0.5, 4.75 - i * 0.5, 0});
                    return expected[d];
                },
                error, "packed four-lane direction");
            expectVector(
                context, transform.template transformPointCartesian<T>(packedPoints),
                [&](int d, int i)
                {
                    const auto expected = referenceMultiply(transformReference, {0.75 - i * 0.5, 2.75 - i * 0.5, 4.75 - i * 0.5, 1});
                    return expected[d] / expected[3];
                },
                error, "packed four-lane Cartesian point");
        }

        const Matrix perspective(V4(2, 0, 0, 0.25), V4(0, 3, 0, 0.5), V4(0, 0, 4, 0.75), V4(5, 6, 7, 2));
        const auto projected = referenceMultiply(referenceMatrix(perspective), {1, 2, 3, 1});
        expectVector(
            context, perspective.transformPointCartesian(V3(1, 2, 3)),
            [&](int d, int)
            {
                return projected[d] / projected[3];
            },
            error, "projective homogeneous divide");
        expectVector(
            context, perspective.transformPointCartesian(V4(1, 2, 3, 9)),
            [&](int d, int)
            {
                return projected[d] / projected[3];
            },
            error, "projective vec4 homogeneous divide");
    }

    template <typename T> void testMatrixRotationUtilities(TestContext& context)
    {
        using Matrix = dex::Matrix4<T>;
        using V3 = dex::vec3<T>;
        using V4 = dex::vec4<T>;
        const double error = tolerance<T>() * 8;
        context.group(std::is_same_v<T, float> ? "Matrix4 float rotation utilities" : "Matrix4 double rotation utilities");

        for (int iteration = 0; iteration < 32; ++iteration)
        {
            const T x = T(-0.8 + iteration * 0.05);
            const T y = T(0.7 - iteration * 0.04);
            const T z = T(-0.6 + iteration * 0.035);
            Matrix rotation = Matrix::identity();
            rotation.thenRotate(z, V3(0, 0, 1));
            rotation.thenRotate(y, V3(0, 1, 0));
            rotation.thenRotate(x, V3(1, 0, 0));
            const double expected[]{double(x), double(y), double(z), 0};
            expectVector(
                context, rotation.getEulerRotation(),
                [&](int d, int)
                {
                    return expected[d];
                },
                error, "Euler extraction for nonsingular ZYX composed rotations");
            expectVector(
                context, rotation.getEulerRotation4(),
                [&](int d, int)
                {
                    return expected[d];
                },
                error, "Euler vec4 extraction has zero w");
        }

        const auto unit = [](double x, double y, double z)
        {
            const double divisor = std::sqrt(x * x + y * y + z * z);
            return V3(T(x / divisor), T(y / divisor), T(z / divisor));
        };
        const std::array<V3, 4> sources{V3(0, 0, 1), V3(1, 0, 0), unit(0.2, -0.4, 0.8), unit(0.8, 0.4, 0.2)};
        const std::array<V3, 4> targets{V3(1, 0, 0), V3(0, 1, 0), unit(-0.6, 0.3, 0.7), unit(0.8, 0.4, 0.2)};
        const ReferenceMatrix identity{{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}};

        for (std::size_t index = 0; index < sources.size(); ++index)
        {
            Matrix alignment = Matrix::translation(V3(9, 8, 7));
            alignment.rotationAlign(targets[index], sources[index]);
            expectVector(
                context, alignment.transformDir(sources[index]),
                [&](int d, int)
                {
                    return component(targets[index], d);
                },
                error, "alignment maps source unit vector to target unit vector");
            expectVector(
                context, alignment.getTranslation(),
                [](int, int)
                {
                    return 0;
                },
                0, "alignment resets translation");
            const auto observed = referenceMatrix(alignment);
            ReferenceMatrix transpose{};

            for (int row = 0; row < 4; ++row)
            {
                for (int column = 0; column < 4; ++column)
                {
                    transpose[row][column] = observed[column][row];
                }
            }

            const auto orthogonality = referenceProduct(transpose, observed);

            for (int row = 0; row < 4; ++row)
            {
                for (int column = 0; column < 4; ++column)
                {
                    context.near(orthogonality[row][column], identity[row][column], error, "alignment matrix is orthogonal");
                }
            }

            const double determinant = observed[0][0] * (observed[1][1] * observed[2][2] - observed[1][2] * observed[2][1]) -
                                       observed[0][1] * (observed[1][0] * observed[2][2] - observed[1][2] * observed[2][0]) +
                                       observed[0][2] * (observed[1][0] * observed[2][1] - observed[1][1] * observed[2][0]);
            context.near(determinant, 1, error, "alignment is a proper rotation");
            Matrix homogeneousAlignment = Matrix::identity();
            homogeneousAlignment.rotationAlign4(V4(targets[index], 12), V4(sources[index], -20));
            expectMatrix(context, homogeneousAlignment, observed, 0, "vec4 alignment ignores both w components");
        }
    }

    template <typename MatrixScalar, typename InputScalar> void testMixedMatrixInputs(TestContext& context)
    {
        using Matrix = dex::Matrix4<MatrixScalar>;
        using V3 = dex::vec3<InputScalar>;
        using V4 = dex::vec4<InputScalar>;
        const auto matrix = Matrix::translation(dex::vec3<MatrixScalar>(2, -3, 4)).thenScale(dex::vec3<MatrixScalar>(3, 4, 5));
        const auto reference = referenceMatrix(matrix);
        const auto point = referenceMultiply(reference, {1.25, -2.5, 3.75, 1});
        const auto direction = referenceMultiply(reference, {1.25, -2.5, 3.75, 0});
        const double error = std::max(tolerance<MatrixScalar>(), tolerance<InputScalar>()) * 4;
        expectVector(
            context, matrix.transformPointCartesian(V3(1.25, -2.5, 3.75)),
            [&](int d, int)
            {
                return point[d];
            },
            error, "mixed-precision scalar Cartesian point");
        expectVector(
            context, matrix.transformPointCartesian(V4(1.25, -2.5, 3.75, 1)),
            [&](int d, int)
            {
                return point[d];
            },
            error, "mixed-precision scalar vec4 Cartesian point");
        expectVector(
            context, matrix.transformDir(V3(1.25, -2.5, 3.75)),
            [&](int d, int)
            {
                return direction[d];
            },
            error, "mixed-precision scalar direction");
        expectVector(
            context, matrix.transformDir(V4(1.25, -2.5, 3.75, 0)),
            [&](int d, int)
            {
                return direction[d];
            },
            error, "mixed-precision scalar vec4 direction");
        using Packed = dex::Vector_t<InputScalar, 4>;
        const auto packedInput = makeVector<dex::vec3<Packed>>(
            [](int d, int i)
            {
                return (d + 1) * 1.25 - i * 0.5;
            });
        expectVector(
            context, matrix.template transformPointCartesian<InputScalar>(packedInput),
            [&](int d, int i)
            {
                const auto expected = referenceMultiply(reference, {1.25 - i * 0.5, 2.5 - i * 0.5, 3.75 - i * 0.5, 1});
                return expected[d];
            },
            error, "mixed-precision packed Cartesian points");
        expectVector(
            context, matrix.template transformDir<InputScalar>(packedInput),
            [&](int d, int i)
            {
                const auto expected = referenceMultiply(reference, {1.25 - i * 0.5, 2.5 - i * 0.5, 3.75 - i * 0.5, 0});
                return expected[d];
            },
            error, "mixed-precision packed directions");
    }

    inline std::array<double, 4> referenceQuaternionProduct(const std::array<double, 4>& a, const std::array<double, 4>& b)
    {
        return {a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]};
    }

    template <typename T> void testQuaternions(TestContext& context, std::string_view typeName)
    {
        context.group("Quaternion<" + std::string(typeName) + ">");
        using Q = dex::Quaternion<T>;
        using V = dex::vec4<T>;
        const double error = tolerance<T>() * 6;
        const Q identity;
        expectVector(
            context, identity.q,
            [](int d, int)
            {
                return d == 3 ? 1 : 0;
            },
            0, "default identity quaternion");
        const T angle = packed<T>(
            [](int i)
            {
                return 0.25 + i * 0.07;
            });
        const Q rotation = Q::fromAxisAngle(V(T(0), T(0), T(1), T(0)), angle);
        expectVector(
            context, rotation.q,
            [&](int d, int i)
            {
                const double halfAngle = lane(angle, i) * 0.5;
                return d == 2 ? std::sin(halfAngle) : d == 3 ? std::cos(halfAngle) : 0;
            },
            error, "axis angle");
        const V vector(T(2), T(-3), T(4), T(0));
        const auto expectedRotation = [&](int d, int i)
        {
            const double theta = lane(angle, i);

            if (d == 0)
            {
                return 2 * std::cos(theta) + 3 * std::sin(theta);
            }

            if (d == 1)
            {
                return 2 * std::sin(theta) - 3 * std::cos(theta);
            }

            return d == 2 ? 4.0 : 0.0;
        };
        expectVector(context, rotation * vector, expectedRotation, error, "Hamilton vector rotation");
        expectVector(context, rotation.rotate(vector), expectedRotation, error, "rotate independent trigonometric oracle");
        expectVector(context, rotation.rotateSSE2(vector), expectedRotation, error, "rotateSSE2 independent trigonometric oracle");
        const V axis = makeVector<V>(
            [](int d, int i)
            {
                const double x = 0.2 + i * 0.01;
                const double y = -0.4 + i * 0.015;
                const double z = 0.8 - i * 0.005;
                const double divisor = std::sqrt(x * x + y * y + z * z);
                return d == 0 ? x / divisor : d == 1 ? y / divisor : d == 2 ? z / divisor : 0;
            });
        const Q arbitraryRotation = Q::fromAxisAngle(axis, angle);
        const auto arbitraryExpected = [&](int d, int i)
        {
            if (d == 3)
            {
                return 0.0;
            }

            const double theta = lane(angle, i);
            const double sine = std::sin(theta);
            const double cosine = std::cos(theta);
            const double x = lane(axis.x, i);
            const double y = lane(axis.y, i);
            const double z = lane(axis.z, i);
            const double dot = 2 * x - 3 * y + 4 * z;
            const double input[]{2, -3, 4};
            const double components[]{x, y, z};
            const double cross[]{4 * y + 3 * z, 2 * z - 4 * x, -3 * x - 2 * y};
            return input[d] * cosine + cross[d] * sine + components[d] * dot * (1 - cosine);
        };
        expectVector(context, arbitraryRotation.rotate(vector), arbitraryExpected, error, "arbitrary-axis Rodrigues rotation oracle");
        expectVector(
            context, arbitraryRotation.rotateSSE2(vector), arbitraryExpected, error, "arbitrary-axis cross-product rotation oracle");
        expectVector(context, arbitraryRotation * vector, arbitraryExpected, error, "arbitrary-axis Hamilton rotation oracle");
        expectVector(
            context, (rotation * rotation.inverseUnitQ()).q,
            [](int d, int)
            {
                return d == 3 ? 1 : 0;
            },
            error, "unit quaternion inverse");

        const Q first(T(1), T(-2), T(3), T(4));
        const Q second(T(-5), T(6), T(7), T(-8));
        const auto productReference = referenceQuaternionProduct({1, -2, 3, 4}, {-5, 6, 7, -8});
        expectVector(
            context, (first * second).q,
            [&](int d, int)
            {
                return productReference[d];
            },
            error, "Hamilton product independent oracle");
        expectVector(
            context, (first * first.inverse()).q,
            [](int d, int)
            {
                return d == 3 ? 1 : 0;
            },
            error, "non-unit quaternion inverse");
        expectVector(
            context, first.normalized().q,
            [](int d, int)
            {
                constexpr double values[]{1, -2, 3, 4};
                return values[d] / std::sqrt(30.0);
            },
            error, "quaternion normalization");

        for (const double fraction : {0.0, 0.25, 0.5, 0.75, 1.0})
        {
            const auto expected = [&](int d, int i)
            {
                const double halfAngle = lane(angle, i) * fraction * 0.5;
                return d == 2 ? std::sin(halfAngle) : d == 3 ? std::cos(halfAngle) : 0;
            };
            expectVector(context, Q::slerp(identity, rotation, T(fraction)).q, expected, error, "static spherical interpolation");
            Q mutableIdentity;
            mutableIdentity.slerp(rotation, T(fraction));
            expectVector(context, mutableIdentity.q, expected, error, "mutating spherical interpolation");
            const Q negativeTarget(-rotation.q);
            expectVector(context, Q::slerp(identity, negativeTarget, T(fraction)).q, expected, error,
                "shortest-path interpolation handles negative quaternion");
        }

        const Q tiny = Q::fromAxisAngle(V(T(0), T(0), T(1), T(0)), T(0.001));
        const Q midpoint = Q::slerp(identity, tiny, T(0.5));
        expectVector(
            context, midpoint.q,
            [](int d, int)
            {
                return d == 2 ? std::sin(0.00025) : d == 3 ? std::cos(0.00025) : 0;
            },
            error, "near-equal slerp linear fallback");
    }

    inline void runTransformTests(TestContext& context)
    {
        testMatrices<float>(context, "float");
        testMatrices<double>(context, "double");
        testMatrixRotationUtilities<float>(context);
        testMatrixRotationUtilities<double>(context);
        context.group("Matrix4 mixed precision");
        testMixedMatrixInputs<float, double>(context);
        testMixedMatrixInputs<double, float>(context);
        forFloatingTypes(
            [&]<typename T>(std::string_view name)
            {
                testQuaternions<T>(context, name);
            });
    }
}
