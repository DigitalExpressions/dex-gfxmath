#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <string_view>

namespace math_tests
{
    class TestContext
    {
    public:
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
            const bool equal = (std::isnan(reference) && std::isnan(observed)) || observed == reference ||
                               (std::isfinite(observed) && std::isfinite(reference) &&
                                   std::abs(observed - reference) <= limit * std::max(1.0L, std::abs(reference)));

            expect(equal, message);
        }

        int finish() const
        {
            std::cout << checks << " checks, " << failures << " failures\n";
            return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
        }

    private:
        std::string currentGroup;
        std::size_t checks = 0;
        std::size_t failures = 0;
        std::map<std::string, std::size_t> failureKinds;
    };
}
