#pragma once

#include "TestContext.h"

#include <array>
#include <bit>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace gfxmath_tests
{
    class ReferenceResults
    {
    public:
        enum class Mode { compare, writeReference, writeResults };

        ReferenceResults(Mode mode, std::filesystem::path path) : mode(mode), path(std::move(path))
        {
            if (mode == Mode::compare)
            {
                input.open(this->path, std::ios::binary);
                std::string line;
                if (!readLine(line) || line != referenceHeader)
                    throw std::runtime_error("Missing or invalid GfxMath reference header: " + this->path.string());
            }
            else
            {
                output.imbue(std::locale::classic());
                output << (mode == Mode::writeReference ? referenceHeader : resultsHeader) << '\n';
            }
        }

        void observe(std::string_view group, std::string_view check, double actual, double expected, double tolerance)
        {
            auto& entries = counts[std::string(group)];
            const std::size_t index = entries[std::string(check)]++;
            // Sample the first three results and later powers of two per named check.
            if (index > 2 && !std::has_single_bit(index))
                return;

            if (group.find_first_of("\t\r\n") != group.npos || check.find_first_of("\t\r\n") != check.npos)
                throw std::runtime_error("A GfxMath check name cannot be represented in the reference file");

            ++rows;
            if (mode != Mode::compare)
            {
                output << group << '\t' << check << '\t' << index << '\t'
                       << format(mode == Mode::writeReference ? expected : actual) << '\t' << format(tolerance) << '\n';
                return;
            }

            std::string line;
            if (!readLine(line))
            {
                discrepancy(group, check, index, "reference ends before this check");
                return;
            }
            const auto fields = split(line);
            if (fields[0] != group || fields[1] != check || parseIndex(fields[2]) != index)
            {
                discrepancy(group, check, index, "reference check or sample index differs");
                return;
            }
            const double frozen = parseValue(fields[3]);
            const double bound = parseValue(fields[4]);
            if (!std::isfinite(bound) || bound < 0 || bound != tolerance)
                discrepancy(group, check, index, "reference tolerance differs");
            else if (!math_tests::withinTolerance(expected, frozen, bound))
                discrepancy(group, check, index, "oracle differs from reference: " + format(expected) + " vs " + format(frozen));
            else if (!math_tests::withinTolerance(actual, frozen, bound))
                discrepancy(group, check, index, "result differs from reference: " + format(actual) + " vs " + format(frozen));
        }

        bool finish(bool unitTestsPassed)
        {
            if (mode == Mode::compare)
            {
                std::string extra;
                if (readLine(extra))
                    discrepancy({}, {}, 0, "reference contains extra rows");
                if (input.bad())
                    throw std::runtime_error("Could not read the complete GfxMath reference file");
                std::cout << rows << " reference results checked, " << failures << " differences\n";
                return unitTestsPassed && failures == 0;
            }

            if (!unitTestsPassed)
                return false;
            std::ofstream file(path, std::ios::binary | std::ios::trunc);
            file << output.str();
            file.close();
            if (!file)
                throw std::runtime_error("Could not write GfxMath results: " + path.string());
            std::cout << rows << " results written to " << path.string() << '\n';
            return true;
        }

    private:
        static constexpr std::string_view referenceHeader = "gfxmath-reference-v1\tgroup\tcheck\tsample\tvalue\ttolerance";
        static constexpr std::string_view resultsHeader = "gfxmath-observations-v1\tgroup\tcheck\tsample\tvalue\ttolerance";

        bool readLine(std::string& line)
        {
            if (!std::getline(input, line))
                return false;
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            return true;
        }

        static std::array<std::string_view, 5> split(std::string_view line)
        {
            std::array<std::string_view, 5> fields;
            for (std::size_t index = 0; index < fields.size(); ++index)
            {
                const auto delimiter = line.find('\t');
                if ((delimiter == line.npos) != (index == fields.size() - 1))
                    throw std::runtime_error("Invalid column count in GfxMath reference row");
                fields[index] = line.substr(0, delimiter);
                if (delimiter != line.npos)
                    line.remove_prefix(delimiter + 1);
            }
            return fields;
        }

        static std::size_t parseIndex(std::string_view token)
        {
            std::size_t value{};
            const auto end = token.data() + token.size();
            const auto result = std::from_chars(token.data(), end, value);
            if (result.ec != std::errc{} || result.ptr != end)
                throw std::runtime_error("Invalid sample index in GfxMath reference row");
            return value;
        }

        static double parseValue(std::string_view token)
        {
            if (token == "nan")
                return std::numeric_limits<double>::quiet_NaN();
            if (token == "inf")
                return std::numeric_limits<double>::infinity();
            if (token == "-inf")
                return -std::numeric_limits<double>::infinity();
            double value{};
            const auto end = token.data() + token.size();
            // Stream extraction can reject representable subnormals as underflow on libc++.
            const auto result = std::from_chars(token.data(), end, value);
            if (result.ec != std::errc{} || result.ptr != end)
                throw std::runtime_error("Invalid numeric result in GfxMath reference row: " + std::string(token));
            return value;
        }

        static std::string format(double value)
        {
            if (std::isnan(value))
                return "nan";
            if (std::isinf(value))
                return std::signbit(value) ? "-inf" : "inf";
            std::ostringstream number;
            number.imbue(std::locale::classic());
            number << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
            return number.str();
        }

        void discrepancy(std::string_view group, std::string_view check, std::size_t index, std::string_view reason)
        {
            if (++failures <= 12)
                std::cerr << "Reference row " << rows << " [" << group << " / " << check << " #" << index << "]: "
                          << reason << '\n';
        }

        Mode mode;
        std::filesystem::path path;
        std::ifstream input;
        std::ostringstream output;
        std::unordered_map<std::string, std::unordered_map<std::string, std::size_t>> counts;
        std::size_t rows{};
        std::size_t failures{};
    };
}
