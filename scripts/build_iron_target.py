"""Build the iron-deficiency target and audit it before anything trains.

Usage, from the repository root::

    python scripts/build_iron_target.py
    python scripts/build_iron_target.py --definition ferritin-or-stfr
    python scripts/build_iron_target.py --cycle 2017-2020

Prints the WHO 2020 ferritin bands, the cohort accounting (including the
NHANES ferritin eligibility subsample, which differs between releases), the
class balance and baseline, and the Finding 8 leak audit run against the new
target.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chemoguard import leakage  # noqa: E402
from chemoguard.data.nhanes import (  # noqa: E402
    CYCLES,
    DEFAULT_CACHE,
    load_nhanes,
    with_iron_markers,
)
from chemoguard.features import build_iron_matrix  # noqa: E402
from chemoguard.iron import (  # noqa: E402
    DEFINITIONS,
    INFLAMMATION_CRP_THRESHOLD,
    WHO_2020_FERRITIN_BANDS,
    label_iron_status,
)


def show_bands():
    print("WHO 2020 ferritin thresholds (technical brief, Table 1)")
    print("-" * 78)
    print("{0:<42}{1:>12}{2:>14}".format(
        "band", "healthy <", "inflamed <"))
    for band in WHO_2020_FERRITIN_BANDS.values():
        print("{0:<42}{1:>12}{2:>14}".format(
            band.label[:41],
            "{0:g} ug/L".format(band.healthy.magnitude_in("ug/L")),
            "{0:g} ug/L".format(band.inflamed.magnitude_in("ug/L")),
        ))
    print()
    print("  Inflammation: CRP > {0}".format(INFLAMMATION_CRP_THRESHOLD))
    print("  Note the '60+ years' row: unlike the anaemia guideline, there is")
    print("  no upper age gap, so over-65s are labelled rather than excluded.")
    print()


def show_cohort(dataset):
    values, meta = dataset.values, dataset.meta
    print("Cohort")
    print("-" * 78)
    for note in dataset.notes:
        print("  note: {0}".format(note))
    print()
    print("  {0:<26}{1:>9}{2:>9}{3:>11}".format(
        "cycle", "CBC", "ferritin", "CRP"))
    for cycle in values.index.get_level_values("cycle").unique():
        block = values.xs(cycle, level="cycle")
        print("  {0:<26}{1:>9}{2:>9}{3:>11}".format(
            str(cycle), len(block),
            int(block["ferritin"].notna().sum()),
            int(block["crp"].notna().sum())))
        sub = meta.xs(cycle, level="cycle")[block["ferritin"].notna().values]
        if len(sub):
            print("     ferritin subsample: age {0:.0f}-{1:.0f}, {2}".format(
                sub["age_years"].min(), sub["age_years"].max(),
                sub["sex"].value_counts().to_dict()))
    print()


def show_labels(report):
    print("Iron-status labels: {0}".format(report.prevalence["definition"]))
    print("-" * 78)
    print("  Row accounting")
    for key, value in report.counts.items():
        print("    {0:<40}{1:>8}".format(key, value))
    print()
    p = report.prevalence
    print("  Class balance")
    print("    {0:<40}{1:>8}  {2:.2f}%".format(
        "deficient", p["deficient"], p["rate"] * 100))
    print("    {0:<40}{1:>8}  {2:.2f}%".format(
        "replete", p["replete"], (1 - p["rate"]) * 100))
    print("    {0:<40}{1:>8.2f}%".format(
        "MAJORITY-CLASS BASELINE", report.baseline * 100))
    print()
    print("    prevalence without the inflammation adjustment: {0:.2f}%".format(
        p["rate_without_inflammation_adjustment"] * 100))
    print("      (the adjustment's whole effect, in one number)")
    print()
    print("  Stated assumptions")
    for note in report.assumptions:
        print("    - {0}".format(note))
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--cycle", action="append", choices=sorted(CYCLES))
    parser.add_argument("--definition", choices=DEFINITIONS, default="ferritin")
    args = parser.parse_args(argv)

    show_bands()

    dataset = with_iron_markers(
        load_nhanes(cycles=args.cycle, cache_dir=args.cache_dir),
        cache_dir=args.cache_dir,
    )
    show_cohort(dataset)

    report = label_iron_status(dataset, definition=args.definition)
    show_labels(report)

    matrix = build_iron_matrix(dataset)
    print("Matrix: {0}".format(matrix.name))
    print("-" * 78)
    print("  columns ({0}): {1}".format(
        len(matrix.columns), ", ".join(matrix.columns)))
    print("  excluded:")
    for name, reason in sorted(matrix.excluded.items()):
        print("    {0:<22}{1}".format(name, reason))
    print()

    print(leakage.format_report(leakage.audit_ferritin(matrix, dataset)))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
