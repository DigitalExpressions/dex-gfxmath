#!/usr/bin/env python3
"""Compare GfxMath timing CSV files from the same machine and ISA."""

import argparse
import csv
import math
from pathlib import Path


def load(path: Path) -> dict:
    rows = {}
    # Windows PowerShell 5 encodes redirected native output as UTF-16.
    with path.open("rb") as stream:
        encoding = "utf-16" if stream.read(2) in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    with path.open(encoding=encoding, newline="") as stream:
        for row in csv.DictReader(stream):
            key = tuple(row[column] for column in (
                "case", "architecture", "simd_profile", "records", "lanes", "passes", "samples"
            ))
            timing = float(row["median_ns_per_item"])
            checksum = float(row["checksum"])
            if key in rows or not math.isfinite(timing) or timing <= 0 or not math.isfinite(checksum):
                raise ValueError(f"{path}: invalid or duplicate row for {key[0]}")
            rows[key] = timing, checksum
    if not rows:
        raise ValueError(f"{path}: no benchmark rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    try:
        baseline = load(args.baseline)
        candidate = load(args.candidate)
        if baseline.keys() != candidate.keys():
            raise ValueError("Benchmark workloads or SIMD profiles differ")
        print("case,baseline_ns_per_item,candidate_ns_per_item,time_change_percent")
        for key, (before, expected) in baseline.items():
            after, actual = candidate[key]
            if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-8):
                raise ValueError(f"{key[0]}: checksum changed ({expected} vs {actual})")
            print(f"{key[0]},{before:.3f},{after:.3f},{(after / before - 1) * 100:+.1f}")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
