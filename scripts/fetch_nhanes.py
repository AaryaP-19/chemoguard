"""Download the NHANES CBC and demographics files and join them on SEQN.

Usage, from the repository root::

    python scripts/fetch_nhanes.py                 # both cycles
    python scripts/fetch_nhanes.py --cycle 2021-2023
    python scripts/fetch_nhanes.py --refresh       # re-download
    python scripts/fetch_nhanes.py --show-units    # print the column map

XPT files are cached under ``data/cache/``, which is gitignored. Reruns hit
the cache and touch the network only with ``--refresh``.

The column map, the unit of every column, and the cycle definitions live in
``chemoguard/data/nhanes.py``, not here. This script is the entry point; it
does not decide what anything means.
"""

from __future__ import annotations

import argparse
import os
import sys

# Run as a script from the repository root: put the root on the path so the
# flat application modules (units, cbc) and the chemoguard package both
# import the same way they do everywhere else.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chemoguard.data.nhanes import (  # noqa: E402
    ABSENT_FROM_NHANES,
    CYCLES,
    DEFAULT_CACHE,
    NHANES_CBC_COLUMNS,
    NHANES_DEMOGRAPHIC_COLUMNS,
    load_cycle,
    load_nhanes,
)
from chemoguard.data.analytes import DATASET_ANALYTES  # noqa: E402


def show_units():
    print("NHANES column map -- declared, not inferred")
    print("=" * 78)
    print("{0:<12} {1:<22} {2:<14} {3}".format(
        "NHANES", "analyte", "canonical", "documented as"))
    print("-" * 78)
    for column in NHANES_CBC_COLUMNS:
        canonical = DATASET_ANALYTES[column.analyte].canonical.symbol
        note = "" if column.unit == canonical else "  (converted)"
        print("{0:<12} {1:<22} {2:<14} {3}{4}".format(
            column.source, column.analyte, canonical,
            column.documented_as, note))
    print()
    print("Demographics, joined on SEQN:")
    for source, name in NHANES_DEMOGRAPHIC_COLUMNS.items():
        print("  {0:<10} -> {1}".format(source, name))
    print()
    print("Not reported by NHANES at all: {0}".format(
        ", ".join(ABSENT_FROM_NHANES)))
    print("  Plateletcrit is absent from the CBC file, so the PCT identity")
    print("  cannot be evaluated on NHANES. Absent is not the same as passing.")


def summarise(dataset):
    values = dataset.values
    print(dataset.name)
    print("-" * len(dataset.name))
    for note in dataset.notes:
        print("  note: {0}".format(note))
    print("  rows: {0}".format(len(values)))
    print()
    print("  {0:<22}{1:>8}{2:>10}{3:>10}{4:>10}  {5}".format(
        "analyte", "n", "min", "median", "max", "unit"))
    for key in values.columns:
        column = values[key].dropna()
        if column.empty:
            print("  {0:<22}{1:>8}".format(key, 0))
            continue
        print("  {0:<22}{1:>8}{2:>10.4g}{3:>10.4g}{4:>10.4g}  {5}".format(
            key, len(column), column.min(), column.median(), column.max(),
            dataset.units.get(key, "?")))
    if dataset.meta is not None and "sex" in dataset.meta.columns:
        counts = dataset.meta["sex"].value_counts(dropna=False).to_dict()
        print()
        print("  demographics joined: sex {0}, age {1:.0f}-{2:.0f} years".format(
            counts,
            dataset.meta["age_years"].min(),
            dataset.meta["age_years"].max(),
        ))
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cycle", action="append", choices=sorted(CYCLES),
                        help="cycle to fetch; repeatable. Default: all.")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--refresh", action="store_true",
                        help="re-download even if cached")
    parser.add_argument("--show-units", action="store_true",
                        help="print the column map and exit")
    args = parser.parse_args(argv)

    if args.show_units:
        show_units()
        return 0

    cycles = args.cycle or list(CYCLES)
    print("Fetching NHANES: {0}".format(", ".join(cycles)))
    print("Cache: {0}".format(os.path.abspath(args.cache_dir)))
    print()

    for key in cycles:
        dataset = load_cycle(key, args.cache_dir, args.refresh)
        summarise(dataset)

    if len(cycles) > 1:
        combined = load_nhanes(cycles, args.cache_dir, refresh=False)
        print("Combined: {0} rows across {1} cycles".format(
            len(combined), len(cycles)))

    print("Cached files:")
    for name in sorted(os.listdir(args.cache_dir)):
        path = os.path.join(args.cache_dir, name)
        print("  {0:<16} {1:>10,} bytes".format(name, os.path.getsize(path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
