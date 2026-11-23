#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <utility>

namespace math_tests
{
    inline bool withinTolerance(long double actual, long double expected, long double tolerance)
    {
        return (std::isnan(expected) && std::isnan(actual)) || actual == expected ||
               (std::isfinite(actual) && std::isfinite(expected) &&
                   std::abs(actual - expected) <= tolerance * std::max(1.0L, std::abs(expected)));
    }

    class TestContext
    {
    public:
        using NumericObserver = std::function<void(std::string_view, std::string_view, double, double, double)>;
        explicit TestContext(NumericObserver observer = {}) : observe(std::move(observer)) {}

        void group(std::string_view name) { currentGroup = name; }

        void expect(bool condition, std::string_view message)
        {
            ++checks;

            if (!condition)
            {
                ++failures;

                const std::string failure = currentGroup + ": " + std::string(message);
                auto& count = failureKinds[failure];
                ++count;

                if (count == 1)
                {
                    std::cerr << failure << '\n';
                }
            }
        }

        template <typename Actual, typename Expected, typename Tolerance>
        void near(Actual actual, Expected expected, Tolerance tolerance, std::string_view message)
        {
            const long double observed = static_cast<long double>(actual);
            const long double reference = static_cast<long double>(expected);
            const long double limit = static_cast<long double>(tolerance);
            expect(withinTolerance(observed, reference, limit), message);
            if (observe)
                observe(currentGroup, message, static_cast<double>(observed), static_cast<double>(reference),
                    static_cast<double>(limit));
        }

        int finish() const
        {
            std::cout << checks << " checks, " << failures << " failures\n";
            return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
        }

    private:
        NumericObserver observe;
        std::string currentGroup;
        std::size_t checks = 0;
        std::size_t failures = 0;
        std::map<std::string, std::size_t> failureKinds;
    };
}
