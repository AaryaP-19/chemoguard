"""Generate the extraction-benchmark corpus and its ground truth.

Usage, from the repository root::

    python scripts/build_extraction_corpus.py                    # all 814 images
    python scripts/build_extraction_corpus.py --no-degradation   # strata A and B only
    python scripts/build_extraction_corpus.py --out data/extraction_corpus

Writes images, a per-image page-text sidecar, and ``manifest.json`` holding
ground truth for every field. Prints the strata sizes and the axis-balance
contingency counts, which are the numbers to check before scoring anything:
an unbalanced corpus makes every per-level comparison downstream
uninterpretable, and the only way to know it is balanced is to count.

The corpus is regenerable and is not committed -- see ``.gitignore``. It is
deterministic: the same code produces the same 814 images, so the manifest is
the artefact worth reviewing rather than the pixels.

Needs Pillow and numpy. Needs no Tesseract, no Poppler and no network.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chemoguard.extraction_bench.corpus import build, composition  # noqa: E402

DEFAULT_OUT = os.path.join("data", "extraction_corpus")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="corpus directory (default: %(default)s)",
    )
    parser.add_argument(
        "--no-degradation",
        action="store_true",
        help="strata A and B only; skip the 702 degraded images",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="do not print per-image progress"
    )
    args = parser.parse_args(argv)

    written = {"n": 0}
    started = time.time()

    def progress(stratum, image_id):
        written["n"] += 1
        if not args.quiet and written["n"] % 50 == 0:
            print("  {0:5d} images written ({1:.0f}s)".format(
                written["n"], time.time() - started
            ))

    print("Building corpus in {0}".format(args.out))
    manifest = build(
        args.out,
        include_degradation=not args.no_degradation,
        progress=progress,
    )

    comp = composition(manifest)
    print("\nWrote {0} images / {1} fields in {2:.0f}s\n".format(
        comp["total_images"], comp["total_fields"], time.time() - started
    ))

    print("Images and fields by stratum")
    for stratum in ("A", "B", "C"):
        if stratum not in comp["images_by_stratum"]:
            continue
        status = comp["status_by_stratum"][stratum]
        print(
            "  {0}  {1:4d} images  {2:5d} fields   "
            "extractable {3:5d} | no_unit {4:4d} | absent {5:4d}".format(
                stratum,
                comp["images_by_stratum"][stratum],
                comp["fields_by_stratum"][stratum],
                status.get("extractable", 0),
                status.get("no_unit", 0),
                status.get("absent", 0),
            )
        )

    print("\nStratum B axis balance (images per level)")
    for axis, counts in comp["stratum_b_axis_balance"].items():
        levels = "  ".join(
            "{0}={1}".format(k, v) for k, v in sorted(counts.items())
        )
        equal = "balanced" if len(set(counts.values())) == 1 else "UNBALANCED"
        print("  {0:<18} {1:<52} {2}".format(axis, levels, equal))

    if comp["stratum_c_family_counts"]:
        print("\nStratum C degradation families (images)")
        for family, count in sorted(comp["stratum_c_family_counts"].items()):
            print("  {0:<12} {1:5d}".format(family, count))

    summary_path = os.path.join(args.out, "composition.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(comp, handle, indent=2, sort_keys=True)
    print("\nComposition written to {0}".format(summary_path))
    print("Manifest written to {0}".format(os.path.join(args.out, "manifest.json")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
