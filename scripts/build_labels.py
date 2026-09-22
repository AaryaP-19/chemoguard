"""Label NHANES with WHO 2024 anaemia severity, build the matrix, audit it.

Usage, from the repository root::

    python scripts/build_labels.py
    python scripts/build_labels.py --allow-over-65
    python scripts/build_labels.py --profile name-excluded

Prints the label distribution, every exclusion and its cost, and the leakage
audit for both matrix profiles.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chemoguard import leakage  # noqa: E402
from chemoguard.data.nhanes import DEFAULT_CACHE, load_nhanes  # noqa: E402
from chemoguard.features import (  # noqa: E402
    FEATURES,
    PROFILES,
    build_anaemia_matrix,
)
from chemoguard.labels import WHO_2024_BANDS, label_dataset  # noqa: E402


def show_bands():
    print("WHO 2024 severity bands (Table 3), as declared in g/L")
    print("-" * 78)
    print("{0:<34}{1:>10}{2:>10}{3:>10}".format(
        "band", "anaemia<", "mild>=", "moderate>="))
    for band in WHO_2024_BANDS.values():
        print("{0:<34}{1:>10}{2:>10}{3:>10}".format(
            band.label[:33],
            "{0:g}".format(band.anaemia.magnitude_in("g/dL")),
            "{0:g}".format(band.mild.magnitude_in("g/dL")),
            "{0:g}".format(band.moderate.magnitude_in("g/dL")),
        ))
    print("(printed in g/dL, converted from the declared g/L by Quantity.to)")
    print()


def show_labels(report):
    print("Label distribution")
    print("-" * 78)
    import pandas as pd

    counts = report.severity.value_counts(dropna=False)
    total = int(report.severity.notna().sum())
    for value, n in counts.items():
        missing = not isinstance(value, str) and pd.isna(value)
        label = "unlabelled" if missing else value
        share = float("nan") if (missing or not total) else n / total * 100
        print("  {0:<16}{1:>8}{2:>9}".format(
            str(label), n,
            "" if share != share else "{0:.2f}%".format(share)))
    print()
    print("  Row accounting")
    for key, value in report.counts.items():
        print("    {0:<34}{1:>8}".format(key, value))
    print()
    print("  Morphology")
    for value, n in report.morphology.value_counts(dropna=False).items():
        print("    {0:<34}{1:>8}".format(str(value), n))
    print()
    print("  Stated assumptions")
    for note in report.assumptions:
        print("    - {0}".format(note))
    print()


def show_matrix(matrix):
    print("Matrix: {0}".format(matrix.name))
    print("-" * 78)
    print("  columns ({0}): {1}".format(
        len(matrix.columns), ", ".join(matrix.columns)))
    print("  held out: {0}".format(", ".join(matrix.held_out)))
    print("  excluded:")
    for name, reason in sorted(matrix.excluded.items()):
        print("    {0:<22}{1}".format(name, reason))
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--allow-over-65", action="store_true")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None,
                        help="audit one profile only; default audits both")
    args = parser.parse_args(argv)

    show_bands()

    dataset = load_nhanes(cache_dir=args.cache_dir)
    print("Loaded {0}: {1} rows".format(dataset.name, len(dataset)))
    print()

    report = label_dataset(dataset, allow_over_65=args.allow_over_65)
    show_labels(report)

    print("Derived features")
    print("-" * 78)
    for feature in FEATURES.values():
        print("  {0:<18}{1:<42}{2}".format(
            feature.name, feature.expression,
            "reads Hb" if feature.reads_haemoglobin else ""))
    print()

    profiles = [args.profile] if args.profile else list(PROFILES)
    for profile in profiles:
        matrix = build_anaemia_matrix(dataset, profile=profile)
        show_matrix(matrix)
        print(leakage.format_report(
            leakage.audit(matrix, dataset, allow_over_65=args.allow_over_65)))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
