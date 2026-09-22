"""Run the staged coherence checks over NHANES and the quarantined copies.

Usage, from the repository root::

    python scripts/coherence_report.py
    python scripts/coherence_report.py --nhanes-only
    python scripts/coherence_report.py --sweep      # tolerance sensitivity

The two sources go through one identical set of checks, which is the point:
the contrast between them is only evidence if nothing about the checks
differs. Where NHANES cannot be checked -- it does not report plateletcrit --
the report says "not reported" rather than counting the row as a pass.

The quarantined files are read for audit only, from outside the repo. See
``chemoguard/data/legacy_audit.py``; nothing here trains on anything.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chemoguard.data import coherence  # noqa: E402
from chemoguard.data.coherence import (  # noqa: E402
    EXACT_IDENTITY_TOLERANCE,
    JOINT_IDENTITIES,
    evaluate,
    format_comparison,
    format_report,
)
from chemoguard.data.legacy_audit import (  # noqa: E402
    LEGACY_SOURCE_UNITS,
    QUARANTINE,
    available,
    load_all,
)
from chemoguard.data.nhanes import DEFAULT_CACHE, load_nhanes  # noqa: E402


def sweep(datasets, tolerances=(0.01, 0.02, 0.03, 0.05, 0.10)):
    """Joint survivor count against tolerance, for each dataset.

    The sensitivity is itself a result. On real analyser data the count rises
    to a ceiling and stops; on data whose rows were drawn per-column it is
    flat, because the failures are not near-misses. A flat curve means the
    survivor count was not chosen by the tolerance -- which is the objection
    CLAUDE.md Finding 2 raises against the old "3,000-3,800" figure.
    """
    width = max(22, max(len(d.name) for d in datasets) + 2)
    lines = ["Tolerance sweep: rows passing the joint exact identities",
             "=" * (14 + width * len(datasets))]
    lines.append("{0:<14}{1}".format(
        "tolerance", "".join("{0:>{1}}".format(d.name, width) for d in datasets)))

    for tolerance in tolerances:
        cells = []
        for dataset in datasets:
            scale = tolerance / EXACT_IDENTITY_TOLERANCE
            mask, evaluable = coherence._joint_mask(
                dataset.values, JOINT_IDENTITIES, tolerance_scale=scale)
            n, total = int(mask.sum()), int(evaluable.sum())
            rate = (n / total * 100) if total else float("nan")
            cells.append("{0:>{1}}".format(
                "{0} ({1:.2f}%)".format(n, rate), width))
        marker = "  <- constant" if tolerance == EXACT_IDENTITY_TOLERANCE else ""
        lines.append("{0:<14}{1}{2}".format(
            "{0:.0%}".format(tolerance), "".join(cells), marker))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--quarantine", default=QUARANTINE)
    parser.add_argument("--nhanes-only", action="store_true")
    parser.add_argument("--legacy-only", action="store_true")
    parser.add_argument("--sweep", action="store_true",
                        help="also print the tolerance sensitivity table")
    args = parser.parse_args(argv)

    print("Staged coherence report")
    print("Exact-identity tolerance: {0:.0%} (coherence.EXACT_IDENTITY_TOLERANCE)"
          .format(EXACT_IDENTITY_TOLERANCE))
    print()

    datasets = []
    if not args.legacy_only:
        datasets.append(load_nhanes(cache_dir=args.cache_dir))
    if not args.nhanes_only:
        found = available(args.quarantine)
        missing = [n for n in LEGACY_SOURCE_UNITS if n not in found]
        if missing:
            # Say which are absent. available() filters silently, and a
            # comparison quietly run over three files instead of four is
            # the kind of result that looks fine and is not.
            print("Quarantine path: {0}".format(args.quarantine))
            print("  not present: {0}".format(", ".join(missing)))
            print()
        datasets.extend(load_all(args.quarantine))

    if not datasets:
        print("No datasets loaded -- nothing to report.")
        print("Checked the quarantine path {0!r}, which holds none of the "
              "four declared files.".format(args.quarantine))
        return 1

    reports = []
    for dataset in datasets:
        report = evaluate(dataset)
        reports.append(report)
        print(format_report(report))
        print()

    if len(reports) > 1:
        print(format_comparison(reports))
        print()

    if args.sweep and datasets:
        print(sweep(datasets))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
