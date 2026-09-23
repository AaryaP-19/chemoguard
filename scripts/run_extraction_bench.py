"""Score the extraction backends over the corpus and write the report.

Usage, from the repository root::

    python scripts/run_extraction_bench.py                     # pattern + tesseract
    python scripts/run_extraction_bench.py --arms pattern      # no binaries needed
    python scripts/run_extraction_bench.py --report docs/extraction-benchmark.md
    python scripts/run_extraction_bench.py --figures
    python scripts/run_extraction_bench.py --from-scores --report docs/x.md

Arms:

``pattern``
    The production regex set over the exact text the generator drew. No OCR.
    Runs anywhere, in seconds, and establishes the ceiling for the others: a
    field lost here was never recoverable by better glyph reading.

``tesseract``
    The full production path. Needs the Tesseract binary; the arm is skipped
    with a message rather than failing the run when it is absent.

``gpt4o_mini``
    Deferred. Requires ``--allow-openai`` *and*
    ``CHEMOGUARD_BENCH_ALLOW_OPENAI=1``, because every uncached image is a
    billable request against the key in ``.env``. Without both, the arm
    raises rather than spending money.

Writes per-field scores to ``scores.json`` beside the corpus so the report can
be regenerated, and any claim in it re-checked, without re-running OCR.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chemoguard.extraction_bench import alias_probe, report as R, score as S  # noqa: E402
from chemoguard.extraction_bench.backends import (  # noqa: E402
    GPT4O_MINI,
    PATTERN,
    TESSERACT,
    TESSERACT_AS_SHIPPED,
    GPTBackend,
    PatternLayerBackend,
    TesseractBackend,
)
from chemoguard.extraction_bench.corpus import composition  # noqa: E402
from chemoguard.extraction_bench.truth import load_manifest  # noqa: E402

DEFAULT_CORPUS = os.path.join("data", "extraction_corpus")

#: Report order for arms recovered from scores.json, ceiling first.
_ARM_ORDER = [PATTERN, TESSERACT, TESSERACT_AS_SHIPPED, GPT4O_MINI]


def _build_arm(name, corpus, allow_openai):
    if name == PATTERN:
        return PatternLayerBackend()
    if name == TESSERACT:
        return TesseractBackend()
    if name == TESSERACT_AS_SHIPPED:
        return TesseractBackend(psm=6, name=TESSERACT_AS_SHIPPED)
    if name == GPT4O_MINI:
        return GPTBackend(
            os.path.join(corpus, "gpt_cache"), allow_network=allow_openai
        )
    raise ValueError("unknown arm {0!r}".format(name))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[PATTERN, TESSERACT],
        choices=[PATTERN, TESSERACT, TESSERACT_AS_SHIPPED, GPT4O_MINI],
    )
    parser.add_argument(
        "--allow-openai",
        action="store_true",
        help="permit billable GPT-4o-mini calls; also needs "
             "CHEMOGUARD_BENCH_ALLOW_OPENAI=1",
    )
    parser.add_argument("--report", default=None, help="write markdown here")
    parser.add_argument(
        "--figures", action="store_true", help="write degradation curves to reports/figures"
    )
    parser.add_argument(
        "--from-scores",
        action="store_true",
        help="rebuild the report and figures from a previous run's "
             "scores.json instead of re-running any arm",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    manifest_path = os.path.join(args.corpus, "manifest.json")
    if not os.path.exists(manifest_path):
        print(
            "No manifest at {0}. Run scripts/build_extraction_corpus.py "
            "first.".format(manifest_path)
        )
        return 1

    manifest = load_manifest(manifest_path)
    comp = composition(manifest)
    print("Corpus: {0} images / {1} fields".format(
        comp["total_images"], comp["total_fields"]
    ))

    scores = []
    ran, skipped_arms = [], []

    if args.from_scores:
        # The OCR arms take about a quarter of an hour each. Every claim in
        # the report is computed from scores.json, so re-deriving it needs no
        # OCR at all -- and the numbers stay pinned to the exact run they
        # came from rather than drifting with the binary's version.
        path = os.path.join(args.corpus, "scores.json")
        if not os.path.exists(path):
            print("No scores at {0}. Run the arms first.".format(path))
            return 1
        with open(path, encoding="utf-8") as handle:
            scores = [S.FieldScore(**row) for row in json.load(handle)]
        ran = sorted({s.backend for s in scores}, key=_ARM_ORDER.index)
        print("Reusing {0} field scores from {1} (arms: {2})".format(
            len(scores), path, ", ".join(ran)
        ))

    for name in ([] if args.from_scores else args.arms):
        arm = _build_arm(name, args.corpus, args.allow_openai)
        if not arm.available():
            reason = (
                "needs --allow-openai and CHEMOGUARD_BENCH_ALLOW_OPENAI=1"
                if name == GPT4O_MINI
                else "binary not installed"
            )
            print("  {0:<12} SKIPPED ({1})".format(name, reason))
            skipped_arms.append((name, reason))
            continue

        started = time.time()
        count = 0
        for image in manifest:
            if name == PATTERN and not image.text_path:
                continue
            reading = arm.read(image, args.corpus)
            scores.extend(S.score_reading(reading, image))
            count += 1
            if not args.quiet and count % 100 == 0:
                print("    {0} {1}/{2} images ({3:.0f}s)".format(
                    name, count, len(manifest), time.time() - started
                ))

        print("  {0:<12} {1} images in {2:.0f}s".format(
            name, count, time.time() - started
        ))
        ran.append(name)

    if not scores:
        print("\nNo arm produced any score. Nothing to report.")
        return 1

    scores_path = os.path.join(args.corpus, "scores.json")
    with open(scores_path, "w", encoding="utf-8") as handle:
        json.dump([asdict(s) for s in scores], handle, indent=1)
    print("\nPer-field scores -> {0}".format(scores_path))

    probe = alias_probe.run()
    probe_summary = alias_probe.summary(probe)

    for name in ran:
        subset = [s for s in scores if s.backend == name]
        print("\n=== {0} ===".format(name))
        for stratum in ("A", "B", "C"):
            block = [s for s in subset if s.stratum == stratum]
            if not block:
                continue
            pooled = S.aggregate(block, "backend")[(name,)]
            print(
                "  stratum {0}  precision {1}  recall {2}   "
                "tp={3} fp={4} fn={5} tn={6}".format(
                    stratum,
                    R._rate(pooled["precision"]),
                    R._rate(pooled["recall"]),
                    pooled["tp"], pooled["fp"], pooled["fn"], pooled["tn"],
                )
            )
        silent = [s for s in S.silent_errors(subset)]
        understated = [s for s in S.severity_understated(subset)]
        print("  silent wrong values: {0}   of which understate CTCAE "
              "severity: {1}".format(len(silent), len(understated)))
        stages = Counter(s.stage for s in subset if s.stage)
        if stages:
            print("  stage attribution: {0}".format(dict(stages.most_common())))

    print("\nAlias coverage: {0} accepted, {1} need an alias, {2} need a new "
          "dimension (of {3} spellings probed)".format(
              probe_summary["counts"]["accepted"],
              probe_summary["counts"]["needs_alias"],
              probe_summary["counts"]["needs_dimension"],
              probe_summary["total"],
          ))

    if args.figures:
        from chemoguard.extraction_bench.figures import write_figures

        paths = write_figures(scores, manifest, ran)
        for path in paths:
            print("  figure -> {0}".format(path))

    if args.report:
        from chemoguard.extraction_bench.writeup import write_report

        write_report(
            args.report,
            scores=scores,
            manifest=manifest,
            composition=comp,
            arms_run=ran,
            arms_skipped=skipped_arms,
            probe=probe,
            probe_summary=probe_summary,
        )
        print("\nReport -> {0}".format(args.report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
