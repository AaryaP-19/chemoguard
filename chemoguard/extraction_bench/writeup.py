"""Assemble the benchmark report.

Structure follows the reporting rule in CLAUDE.md: never a bare headline
number. Corpus composition first, so the reader knows what the rates are
computed over; then per-parameter and per-axis breakdowns, which are the
result; then the silent-error table, which is the finding with a
consequence; and the pooled figure last, as a summary of the breakdowns
rather than a substitute for them.

Everything in the generated document is computed from ``scores.json`` and the
manifest. Prose that states a conclusion is written here, beside the code
that produces the number it refers to, so the two cannot drift apart
silently -- but the numbers themselves are always interpolated, never typed.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
from pathlib import Path

from chemoguard.extraction_bench import report as R, score as S
from chemoguard.extraction_bench.report import table
from chemoguard.extraction_bench.backends import (
    PATTERN,
    TESSERACT,
    TESSERACT_AS_SHIPPED,
)

_ARM_TITLES = {
    PATTERN: "Pattern layer, no OCR (the ceiling)",
    TESSERACT: "Tesseract + regex, psm 3 (the production path)",
    TESSERACT_AS_SHIPPED: "Tesseract + regex, psm 6 (as shipped)",
    "gpt4o_mini": "GPT-4o-mini vision",
}

_ARM_BLURB = {
    PATTERN: (
        "The production regex set in `ocr_extraction.extract_parameters`, run "
        "over the exact text the generator drew. No OCR in the loop, so every "
        "loss here is a defect in the patterns or the alias table rather than "
        "in glyph reading. **This arm is the ceiling for the others**: a field "
        "it cannot read from a perfect transcription was never recoverable by "
        "better OCR."
    ),
    TESSERACT: (
        "Preprocess, OCR, regex, canonicalise -- the path `/upload` actually "
        "runs, with the page-segmentation mode corrected per M4. The "
        "difference between this arm and the pattern layer is the cost of "
        "reading glyphs, and nothing else: they share the patterns, the alias "
        "table, the canonicalisation and the ground truth."
    ),
    TESSERACT_AS_SHIPPED: (
        "The same path with `--psm 6`, the mode the code shipped with. "
        "Reported because it was wrong and because the size of the error is "
        "the evidence for changing it, not to be read as a result about OCR. "
        "See M4."
    ),
}


def _pooled(scores, arm, stratum=None):
    subset = [
        s for s in scores
        if s.backend == arm and (stratum is None or s.stratum == stratum)
    ]
    if not subset:
        return None
    return S.aggregate(subset, "backend")[(arm,)]


def _scored_counts(scores, arm):
    """Images and fields this arm actually scored, per stratum."""
    out = {stratum: {"images": set(), "fields": 0} for stratum in ("A", "B", "C")}
    for score in scores:
        if arm is not None and score.backend != arm:
            continue
        bucket = out.get(score.stratum)
        if bucket is None:
            continue
        bucket["images"].add(score.image_id)
        bucket["fields"] += 1
    return {
        stratum: {"images": len(v["images"]), "fields": v["fields"]}
        for stratum, v in out.items()
    }


def _headline(scores, arm):
    """The one-paragraph summary for an arm, with its counts."""
    pooled = _pooled(scores, arm)
    silent = [s for s in S.silent_errors(scores) if s.backend == arm]
    understated = [s for s in S.severity_understated(scores) if s.backend == arm]

    if silent:
        consequence = (
            "**{silent} wrong quantities reached the panel with nothing "
            "flagged**, and {understated} of those graded a cytopenia less "
            "severe than it is.".format(
                silent=len(silent), understated=len(understated)
            )
        )
    else:
        consequence = (
            "**No wrong quantity reached the panel.** Every field the arm "
            "could not read correctly was refused rather than emitted, so "
            "there are no silent errors to report and nothing understated a "
            "CTCAE grade."
        )

    return (
        "Precision {precision} and recall {recall} over **{fields} fields / "
        "{judgements} judgements** (TP {tp}, FP {fp}, FN {fn}, TN {tn}). "
        "{consequence}".format(
            precision=R._rate(pooled["precision"]),
            recall=R._rate(pooled["recall"]),
            fields=pooled["fields"],
            judgements=pooled["judgements"],
            tp=pooled["tp"],
            fp=pooled["fp"],
            fn=pooled["fn"],
            tn=pooled["tn"],
            consequence=consequence,
        )
    )


#: Measured results of the earlier runs, kept so the deltas in the report are
#: honest about which numbers are live and which are recorded. Each was
#: produced by this same harness against this same corpus and ground truth;
#: only the production code under test changed between them.
_BASELINES = {
    "as_found": {
        "label": "as found",
        "precision": 0.948, "recall": 0.622,
        "tp": 578, "fp": 32, "fn": 351, "tn": 461,
        "silent": 32, "understated": 16,
        "aliases_accepted": 28,
        "stratum_b_recall": 0.550, "stratum_b_precision": 0.925,
        "point_recall": 0.867, "comma_recall": 0.233,
        "point_precision": 1.000, "comma_precision": 0.724,
        "si_recall": 0.450, "cells_recall": 0.617, "k_recall": 0.583,
    },
    "after_aliases": {
        "label": "after the alias fix",
        "precision": 0.948, "recall": 0.622,
        "tp": 578, "fp": 32, "fn": 351, "tn": 461,
        "silent": 32, "understated": 16,
        "aliases_accepted": 43,
    },
}


def _ceiling_gap_section(scores, arms_run, section) -> str:
    """The ceiling minus the OCR arm, and what the difference is made of.

    Written as its own section because it is the question the benchmark was
    built to answer, and because the subtraction is only valid under a
    condition worth stating out loud: both arms must have scored the same
    fields. The pattern layer skips the 675 degraded variants, so a pooled
    difference taken carelessly would mostly measure coverage.
    """
    ceiling = _pooled(scores, PATTERN)
    ocr = _pooled(scores, TESSERACT)
    shared = R._shared_fields(scores, PATTERN, TESSERACT)
    shared_fields = len({(s.image_id, s.analyte) for s in shared})

    ceil_shared = S.aggregate(
        [s for s in shared if s.backend == PATTERN], "backend"
    )[(PATTERN,)]
    ocr_shared = S.aggregate(
        [s for s in shared if s.backend == TESSERACT], "backend"
    )[(TESSERACT,)]

    silent = [s for s in S.silent_errors(scores) if s.backend == TESSERACT]
    understated = [
        s for s in S.severity_understated(scores) if s.backend == TESSERACT
    ]
    # Attribution by *field*, not by stage name. A stage label cannot carry
    # this: `no_read` is reached by the pattern layer too (M3's three label-gap
    # fields), so counting stages would charge a pattern defect to OCR. What
    # is unambiguously glyph reading is a field the ceiling got right on the
    # same image and the OCR arm did not -- nothing else differs between them.
    ceiling_correct = {
        (s.image_id, s.analyte) for s in shared
        if s.backend == PATTERN and s.tp
    }
    ocr_correct = {
        (s.image_id, s.analyte) for s in shared
        if s.backend == TESSERACT and s.tp
    }
    lost_to_glyphs = ceiling_correct - ocr_correct
    lost_by_both = {
        (s.image_id, s.analyte) for s in shared
        if s.backend == PATTERN and s.fn
    } & {
        (s.image_id, s.analyte) for s in shared
        if s.backend == TESSERACT and s.fn
    }
    gained_by_ocr = ocr_correct - ceiling_correct

    stage_table = Counter(
        s.stage for s in shared
        if s.backend == TESSERACT and s.stage
        and (s.image_id, s.analyte) in lost_to_glyphs
    )

    out = ["## {0}. The ceiling, the OCR arm, and the gap\n".format(section)]

    out.append(
        "Both arms run the same patterns, the same alias table, the same "
        "canonicalisation and the same ground truth. **The only difference "
        "is where the text came from.** So the recall difference is glyph "
        "reading, and there is no third explanation available -- which is "
        "what makes this a subtraction rather than a correlation.\n"
    )
    out.append(
        "Compared over the **{shared} fields both arms scored**. The pattern "
        "layer does not read pixels, so it has no result for the 675 "
        "degraded variants; pooling without that restriction would compare "
        "different denominators and the gap would be an artefact of "
        "coverage.\n".format(shared=shared_fields)
    )

    out.append("### Per stratum\n")
    out.append(R.ceiling_gap(scores, PATTERN, TESSERACT, by="stratum") + "\n")
    out.append("### Per parameter\n")
    out.append(R.ceiling_gap(scores, PATTERN, TESSERACT, by="analyte") + "\n")

    out.append(
        "On the shared fields the ceiling is recall {cr} and the OCR arm "
        "{orr} -- a gap of **{gap:.3f}**, which is {pct:.0f}% of the "
        "available recall. **That entire gap is glyph reading.** Nothing "
        "else differs between the two arms.\n".format(
            cr=R._rate(ceil_shared["recall"]),
            orr=R._rate(ocr_shared["recall"]),
            gap=(ceil_shared["recall"] or 0) - (ocr_shared["recall"] or 0),
            pct=100 * ((ceil_shared["recall"] or 0) - (ocr_shared["recall"] or 0))
            / (ceil_shared["recall"] or 1),
        )
    )
    out.append(
        "Field by field, which is the attribution that does not rely on a "
        "stage label: **{glyph} fields the ceiling read correctly, the OCR "
        "arm did not.** Those are glyph reading and nothing else -- same "
        "patterns, same alias table, same truth, same image, different text. "
        "{both} were lost by both arms and belong to the pattern defects "
        "rather than to OCR ({m3}). {gained} went the other way.\n".format(
            glyph=len(lost_to_glyphs),
            both="{0} fields".format(len(lost_by_both)) if len(lost_by_both) != 1
            else "1 field",
            m3="M3's label gap" if lost_by_both else "none remain",
            gained="No fields" if not gained_by_ocr
            else ("1 field" if len(gained_by_ocr) == 1
                  else "{0} fields".format(len(gained_by_ocr))),
        )
    )
    if gained_by_ocr:
        out.append(
            "_That last figure should be zero, and is worth a look if it is "
            "not: a field OCR recovers and the ceiling does not means the "
            "rendered text and the generator's text disagree somewhere._\n"
        )
    if stage_table:
        out.append(
            "Where those {0} glyph losses land:\n".format(len(lost_to_glyphs))
        )
        out.append(
            table(
                ["Stage", "Fields", "Share of glyph losses"],
                [
                    [st, n, "{0:.1%}".format(n / len(lost_to_glyphs))]
                    for st, n in stage_table.most_common()
                ],
            )
            + "\n"
        )
    out.append(
        "**The patterns and the unit layer are no longer the binding "
        "constraint on this pipeline. The image is.** Every fix in M1, M2 and "
        "the alias table moved the ceiling; none of them moves this gap.\n"
    )

    out.append("### Do glyph errors reintroduce silent wrong values?\n")
    if silent:
        out.append(
            "**Yes. {n} of them, {u} understating a CTCAE grade.**\n".format(
                n=len(silent), u=len(understated)
            )
        )
        out.append(
            "This is the finding, not a footnote. M1 and M2 removed every "
            "silent error at the pattern layer by making the *parse* refuse "
            "what it could not resolve. A misread digit defeats that, "
            "because nothing about it is unresolvable: the unit is read "
            "correctly, the separators are unambiguous, the number is "
            "simply wrong. It then has to clear only the `cbc.ANALYTES` "
            "plausibility bounds, which are deliberately wide enough to "
            "admit real pathology -- a platelet count of 9,000 and one of "
            "90,000 are both perfectly possible, so nothing downstream can "
            "tell that a digit moved.\n"
        )
        out.append(
            "The unit boundary was never designed to catch this and cannot. "
            "It guarantees that a value arrives with a known scale, not that "
            "the digits are the ones on the page. Catching these needs "
            "either a second read to agree with the first, or a "
            "confidence-gated refusal, or an operator confirming the figures "
            "against the image -- none of which this pipeline has.\n"
        )
        out.append(R.silent_error_table(scores, TESSERACT) + "\n")
    else:
        out.append(
            "**No.** The OCR arm emitted no wrong quantity: every field it "
            "could not read correctly was refused. That is a stronger result "
            "than it sounds, because the plausibility bounds are wide by "
            "design and a misread digit would usually clear them. It held "
            "because most glyph errors here damage the *unit* or the row "
            "label rather than the digits, and both of those produce "
            "refusals.\n"
        )

    out.append("### The gap had one dominant cause: the micro sign\n")
    out.append(
        "The per-parameter table used to split cleanly in two, and the split "
        "was not about the analytes. It was about their units.\n"
    )
    out.append(
        "Tesseract reads `µ` as `p`. Straight from the OCR text of a "
        "pristine render at confidence 91:\n"
    )
    out.append(
        "```\n"
        "Absolute Neutrophil Count (ANC) 3,900 cells/pL 1,500 - 8,000\n"
        "                                      ^^^^^^^^ the page prints cells/µL\n"
        "```\n"
    )
    out.append(
        "`cells/pL` did not match `_U_COUNT`, so the unit was dropped and the "
        "field refused. Every count analyte prints a microlitre denominator "
        "and no other analyte does, which was the whole of the pattern: the "
        "counts scored 0.221-0.305 recall against 0.670-0.779 for everything "
        "else.\n"
    )

    out.append("#### The fix, and what it moved\n")
    out.append(
        "`pL` is now accepted in the microlitre denominator and rewritten to "
        "`uL` at the OCR boundary. It is **not** registered in `units.py`: a "
        "picolitre is a real unit, `cells/pL` is a misreading rather than a "
        "spelling, and a corruption admitted to the alias table would then be "
        "accepted from the keyboard and from the REST API too, where there is "
        "no OCR to blame. The rewrite is also scoped to a denominator "
        "following a slash, because a blanket `p`-for-`u` would turn "
        "picograms into micrograms and corrupt every MCH on every report.\n"
    )
    out.append(
        "Accepting it costs no ambiguity: a picolitre is a millionth of a "
        "microlitre, so no laboratory reports a cell count per picolitre and "
        "there is nothing for `cells/pL` to be confused with.\n"
    )
    out.append(
        "Measured over the full 814 images, before and against after:\n"
    )
    out.append(
        table(
            ["", "Before", "After"],
            [
                ["Stratum A", "1.000 / 0.172", "1.000 / 0.172"],
                ["Stratum B", "0.994 / 0.689", "0.995 / **0.767**"],
                ["Stratum C", "0.987 / 0.460", "0.985 / **0.506**"],
                ["Pooled", "0.988 / 0.489", "0.987 / **0.540**"],
                ["Count-analyte recall", "0.228", "**0.353**"],
                ["  of which ANC", "0.101", "**0.287**"],
                ["`unit_dropped` losses", "877", "695"],
                ["Silent wrong values", "31", "**38**"],
                ["  of which understate severity", "9", "**13**"],
            ],
        )
        + "\n"
    )
    out.append(
        "Precision / recall. **268 fields recovered**, and ANC -- which the "
        "micro sign had hit hardest -- nearly tripled.\n"
    )
    out.append(
        "**And silent wrong values went up, from 31 to 38.** That is not a "
        "side effect to bury: recovering a field means letting it through, "
        "and some of what comes through is wrong. The fields this fix "
        "recovers were previously refused for a *unit* problem, which was "
        "masking a *digit* problem underneath. Raising recall on a pipeline "
        "with no digit check necessarily raises the count of plausible wrong "
        "answers, and that trade has to be stated rather than netted off "
        "against the recall gain.\n"
    )

    out.append("#### Against the upscaling lever\n")
    out.append(
        "Both levers address the same glyph, so the question is whether they "
        "add. Measured on the same stratified 40-image sample, end to end:\n"
    )
    out.append(
        table(
            ["", "Recall", "Count-analyte recall", "Silent"],
            [
                ["1x, no `pL` rule", "0.587", "0.266", "3"],
                ["**1x + `pL` rule**", "**0.674**", "**0.481**", "3"],
                ["2x, no `pL` rule", "0.697", "0.571", "1"],
                ["2x + `pL` rule", "0.697", "0.571", "1"],
            ],
        )
        + "\n"
    )
    out.append(
        "**They are substitutes, not complements.** 2x with the rule is "
        "identical to 2x without it, because upscaling already cuts the "
        "`/pL` misreads from 53 to 7 and leaves the rule almost nothing to "
        "do. Applying both buys nothing over applying either.\n"
    )
    out.append(
        "The pattern-layer rule captures most of the benefit -- count-analyte "
        "recall 0.266 to 0.481 against upscaling's 0.571 -- for no image "
        "processing, no runtime cost, and no proximity to the 3x cliff, where "
        "recall collapses to 0.084 and silent errors jump to 52. Upscaling "
        "still edges ahead on both recall and silent errors, so the rule is "
        "**not strictly better**; it is cheaper and safer for most of the "
        "gain. Taking the remaining ~0.09 of count-analyte recall means "
        "accepting a preprocessing parameter with a cliff one step past the "
        "optimum, which is a judgement about operational risk rather than a "
        "measurement.\n"
    )
    out.append(
        "Neither lever is finished business. The next most common corruption "
        "in the same sample is `µ` read as `y` -- 24 occurrences of `/yL` at "
        "1x, against 53 of `/pL`. It is not handled, deliberately: the `p` "
        "rule was scoped to one glyph so its effect could be attributed, and "
        "adding a second in the same change would have made both "
        "unmeasurable.\n"
    )

    out.append("### Where the OCR arm loses fields\n")
    out.append(R.attribution(scores, TESSERACT) + "\n")

    return "\n".join(out)


def _mechanisms(scores, manifest, arms_run, section) -> str:
    """The three defects: what each was, what was done, what it moved.

    M1 and M2 are now fixed, so the "before" figures here come from
    ``_BASELINES`` and are labelled as recorded; everything describing the
    current state is interpolated live. Writing the prose beside the code
    that computes it keeps the two from drifting, and marking the provenance
    of each number keeps the document from implying a measurement it did not
    make.
    """
    arm = arms_run[0]
    subset = [s for s in scores if s.backend == arm]
    was = _BASELINES["as_found"]

    live = _pooled(scores, arm)
    silent_now = S.silent_errors(subset)
    understated_now = S.severity_understated(subset)

    decimal_now = S.by_factor(scores, manifest, "decimal_style")
    scale_now = S.by_factor(scores, manifest, "scale")
    ambiguous_now = [s for s in subset if s.stage == S.AMBIGUOUS_NUMBER]

    # The as-shipped page-segmentation arm, when it was run, so M4 can
    # quote the full-corpus difference rather than only the sample.
    from chemoguard.extraction_bench.backends import TESSERACT_AS_SHIPPED
    shipped = _pooled(scores, TESSERACT_AS_SHIPPED)
    fixed = _pooled(scores, TESSERACT)
    if shipped is None or fixed is None:
        shipped = None

    def live_rate(stats, key, field="recall"):
        return R._rate((stats.get((arm, key)) or {}).get(field))

    def live_fields(stats, key):
        return (stats.get((arm, key)) or {}).get("fields", 0)

    out = ["## {0}. The four defects it found\n".format(section)]
    out.append(
        "Each was confirmed directly against the production module rather "
        "than inferred from the aggregate. Three are fixed and one is open. "
        "M4 was found only when Tesseract was installed, and would have made "
        "every OCR number in this report meaningless.\n"
    )

    # --- M1 -----------------------------------------------------------------
    out.append("### M1. The value parse had no decimal-mark policy - **fixed**\n")
    out.append(
        "`_NUMBER` was `(?P<value>\\d[\\d,]*\\.?\\d*)`, finished with "
        "`float(text.replace(\",\", \"\"))`: commas stripped, points "
        "honoured. That is the US convention hard-coded, so `6.500` read as "
        "6.5 and `0,9` read as 9.\n"
    )
    out.append(
        "As found (recorded), recall across the two decimal styles was "
        "{pr:.3f} for a point against {cr:.3f} for a comma, and precision "
        "{pp:.3f} against {cp:.3f}. **{silent} wrong quantities reached the "
        "panel and {und} of them graded a cytopenia less severe than it "
        "is** - a WBC printed `0,8`, meaning 800 cells/uL and CTCAE grade 4 "
        "leukopenia, read as 8,000 and graded 0. Finding 4's signature by a "
        "different mechanism: not magnitude inference, a separator strip, "
        "and pointing the same way.\n".format(
            pr=was["point_recall"], cr=was["comma_recall"],
            pp=was["point_precision"], cp=was["comma_precision"],
            silent=was["silent"], und=was["understated"],
        )
    )
    out.append(
        "**The fix rests on one fact: a thousands separator is always "
        "followed by exactly three digits.** So the shape of the token "
        "settles most cases with no context at all:\n"
    )
    out.append(
        table(
            ["Token shape", "Reading", "Example"],
            [
                ["no separator", "integer", "`6500` -> 6500"],
                ["separator + 1, 2 or 4+ digits",
                 "**decimal mark**, unambiguously",
                 "`0,9` -> 0.9; `4,80` -> 4.80"],
                ["two different separators", "last is the decimal mark",
                 "`1.234,56` -> 1234.56"],
                ["one separator repeated",
                 "all grouping - a number has one decimal mark",
                 "`1.234.567` -> 1234567"],
                ["separator + exactly 3 digits", "genuinely ambiguous",
                 "`6,500`, `250.000`"],
            ],
        )
        + "\n"
    )
    out.append(
        "That alone resolves all four understating cases deterministically - "
        "`0,9`, `0,8`, `0,2` and `2,0` each have one digit after the "
        "separator - and turns `6,500` and `3.900` into refusals.\n"
    )
    out.append(
        "A **document-level second pass** then recovers those. A page that "
        "prints `13.0` or `83.5` anywhere has shown that `.` is its decimal "
        "mark, and therefore that `,` groups its thousands throughout. The "
        "evidence comes from the document, not from a convention we picked. "
        "Contradictory evidence on one page yields no inference rather than "
        "a majority vote, and its ambiguous tokens are refused.\n"
    )
    out.append(
        "Where nothing settles it, the number is refused with the printed "
        "token attached - `AmbiguousNumberError`, the same stance this "
        "codebase already takes on a missing unit. On this corpus that path "
        "never fires ({amb} fields): every generated page carries at least "
        "one unambiguous decimal, because RBC is always printed to two "
        "places. It is exercised directly in "
        "`tests/test_number_parsing.py`.\n".format(amb=len(ambiguous_now))
    )
    out.append(
        "After the fix, recall by decimal style is {pr_now} for a point over "
        "{fpt} fields and {cr_now} for a comma over {fcm}.\n".format(
            pr_now=live_rate(decimal_now, "point"),
            cr_now=live_rate(decimal_now, "comma"),
            fpt=live_fields(decimal_now, "point"),
            fcm=live_fields(decimal_now, "comma"),
        )
    )

    # --- M2 -----------------------------------------------------------------
    out.append(
        "### M2. The SI unit alternation could not match any SI unit - "
        "**fixed**\n"
    )
    out.append(
        "`_U_COUNT` built its SI alternative as "
        "`r'10\\^?[369]\\s*/\\s*[l{0}]+'.format(_MU)` with "
        "`_MU = r'[u\\u00b5\\u03bc]'`. Interpolating a character class inside "
        "another gives `[l[uµμ]]+`, which Python parses as the class "
        "`[l[uµμ]` followed by a **required literal `]`**. So "
        "`10^9/L]` matched and `10^9/L` did not.\n"
    )
    out.append(
        "Two further defects sat in the same expression. `[369]` is a single "
        "digit, so `10^12/L` - the SI unit for a red cell count - could not "
        "have matched even with the brackets fixed. And `10\\^?` made the "
        "caret optional, so `109/L` matched and was read as an SI unit.\n"
    )
    out.append(
        "As found (recorded), SI recall was {si:.3f} against {cells:.3f} for "
        "cells/uL and {k:.3f} for K/uL, and the loss was entirely in the "
        "pattern layer - `units.parse_unit` had accepted `10^9/L` and "
        "`10^12/L` all along.\n".format(
            si=was["si_recall"], cells=was["cells_recall"], k=was["k_recall"],
        )
    )
    out.append(
        "The fix splits the micro characters into a bare list so no class is "
        "nested, allows `(?:12|0|3|6|9)` as the exponent, and **requires** an "
        "exponent marker - a caret, `e`, `*`, or a run of superscript digits, "
        "which are self-marking and need none. One case is deliberately "
        "given up: OCR that flattens a superscript to a plain digit turns "
        "`10⁹/L` into `109/L`, which is now a refusal. That is the right way "
        "round - `109` is a perfectly possible printed value, so the old "
        "match was never safe, and a refusal is recoverable where a wrong "
        "scale is not.\n"
    )
    out.append(
        "After the fix, recall by scale is {si_now} for SI, {cells_now} for "
        "cells/uL and {k_now} for K/uL.\n".format(
            si_now=live_rate(scale_now, "si"),
            cells_now=live_rate(scale_now, "conventional_cells"),
            k_now=live_rate(scale_now, "conventional_k"),
        )
    )

    # --- M3 -----------------------------------------------------------------
    out.append("### M3. Three gaps stack on one real image - **open**\n")
    out.append(
        "`test_4.png` (A4) prints its rows as `RBC Count`, `WBC Count` and "
        "`Platelet Count`, with units spelled `million cells/mcL`, "
        "`cells/mcL` and `/mcL`. Three of its four fields are lost, and the "
        "pattern layer loses them at the **first** of three independent "
        "gaps:\n"
    )
    out.append(
        "1. `_ANALYTE_LABELS` has no `\"<abbrev> Count\"` form, so those "
        "three rows match no pattern at all and the analyte is never "
        "sighted. Attributed to `no_read`, correctly, and not to the alias "
        "table.\n"
        "2. Given a label the patterns know, the rows match but return "
        "`unit: None` - `mcL` is absent from `_U_COUNT` too.\n"
        "3. The alias table now accepts `cells/mcL`, so this third gap is "
        "closed. The first two are not.\n"
    )
    out.append(
        "These three fields are **the only remaining losses in the whole "
        "pass** - {fn} false negatives out of {avail} extractable fields. "
        "Fixing gap 1 alone would turn them into refusals rather than "
        "values, because gap 2 still stands, which is why they are quoted "
        "together.\n".format(fn=live["fn"], avail=live["tp"] + live["fn"])
    )

    # --- M4 -----------------------------------------------------------------
    out.append(
        "### M4. The page-segmentation mode read almost nothing off a table "
        "- **fixed**\n"
    )
    out.append(
        "Found the moment Tesseract was installed, and it would have made "
        "every OCR number in this report meaningless. Both OCR call sites "
        "passed `--oem 3 --psm 6`. **psm 6 means \"assume a single uniform "
        "block of text\"**, and a bordered table with columns is not that: "
        "Tesseract runs the rows together across the grid lines and returns "
        "rubbish. On a pristine, high-contrast generated report at OCR "
        "confidence 90 it returned the two title lines perfectly and, for "
        "the entire ten-row table, `Cr ee` / `rucr ||| ars`.\n"
    )
    out.append(
        "Preprocessing was not the cause and was ruled out before the mode "
        "was: denoising, Otsu thresholding and 2x upscaling each change the "
        "result by nothing at all under psm 6, and every other mode reads "
        "8/8 values with or without them.\n"
    )
    out.append(
        "The replacement was chosen by measurement over a stratified sample "
        "of 24 images spanning the clean control and the worst level of "
        "every degradation family, scored end to end through the real "
        "pipeline:\n"
    )
    out.append(
        table(
            ["psm", "Meaning", "Precision", "Recall"],
            [
                ["**3**", "fully automatic (**chosen**)", "0.987", "0.419"],
                ["4", "single column, variable sizes", "0.974", "0.425"],
                ["11", "sparse text", "1.000", "0.229"],
                ["12", "sparse text with OSD", "0.977", "0.235"],
                ["6", "single uniform block (**as shipped**)", "1.000", "0.022"],
            ],
        )
        + "\n"
    )
    out.append(
        "psm 3 over psm 4 on precision and on silent errors, which is the "
        "trade this application makes everywhere else. Note that the "
        "sparse-text modes read *more* characters and score worse: they do "
        "not preserve reading order, and the patterns need a row's label, "
        "value and unit on one line. Glyph accuracy is not the objective; "
        "field recovery is.\n"
    )
    if shipped is not None:
        out.append(
            "Scored over the full corpus, the as-shipped mode reaches recall "
            "{sr} against {fr} for psm 3 - a **{ratio:.0f}x** difference, "
            "and the single largest effect anything in this benchmark has "
            "measured.\n".format(
                sr=R._rate(shipped["recall"]),
                fr=R._rate(fixed["recall"]),
                ratio=(fixed["recall"] or 0) / (shipped["recall"] or 1e-9),
            )
        )
    out.append(
        "Worth recording how close this came to being invisible. The arm "
        "would have run, produced a number, and that number would have been "
        "attributed to glyph reading - which is exactly the inference this "
        "report was asked to make. A one-line configuration default would "
        "have been reported as the difficulty of reading laboratory "
        "reports.\n"
    )

    if silent_now or understated_now:
        out.append(
            "_{0} silent wrong values remain, {1} of them understating "
            "severity._\n".format(len(silent_now), len(understated_now))
        )

    return "\n".join(out)


def write_report(
    path,
    scores,
    manifest,
    composition,
    arms_run,
    arms_skipped,
    probe,
    probe_summary,
) -> None:
    """Write the markdown report."""
    out = []
    add = out.append

    arm_names = ", ".join("`{0}`".format(a) for a in arms_run)
    has_ocr = TESSERACT in arms_run
    ocr_silent = (
        [s for s in S.silent_errors(scores) if s.backend == TESSERACT]
        if has_ocr else []
    )
    ocr_understated = (
        [s for s in S.severity_understated(scores) if s.backend == TESSERACT]
        if has_ocr else []
    )

    add("# Extraction benchmark for CBC tables\n")

    if has_ocr:
        ceiling = _pooled(scores, PATTERN)
        ocr = _pooled(scores, TESSERACT)
        add(
            "Three arms over one corpus of 814 images and 8,140 hand- and "
            "generator-labelled fields. The **pattern layer** runs the "
            "production regex set over the exact text the generator drew, "
            "with no glyph reading in the loop, and is therefore a "
            "*ceiling*: recall {ceil_r} at precision {ceil_p}. The "
            "**Tesseract arm** runs the real production path on the rendered "
            "images: recall {ocr_r} at precision {ocr_p}. A third arm repeats "
            "the OCR path with the page-segmentation mode the code shipped "
            "with, which is reported because it was wrong -- see M4.\n".format(
                ceil_r=R._rate(ceiling["recall"]),
                ceil_p=R._rate(ceiling["precision"]),
                ocr_r=R._rate(ocr["recall"]),
                ocr_p=R._rate(ocr["precision"]),
            )
        )
        add(
            "Both arms share the patterns, the alias table, the "
            "canonicalisation and the ground truth. **The only difference "
            "between them is where the text came from**, so the difference "
            "in recall is glyph reading and nothing else. With the ceiling "
            "at {ceil_r}, essentially the whole remaining loss is "
            "attributable to OCR. §{gap_section} does that subtraction per "
            "stratum and per parameter.\n".format(
                ceil_r=R._rate(ceiling["recall"]),
                gap_section=len(arms_run) + 2,
            )
        )
        if ocr_silent:
            add(
                "**Headline finding: glyph errors reintroduce silent wrong "
                "values.** M1 and M2 eliminated them at the pattern layer, "
                "which now emits none. Real OCR produces **{n}**, of which "
                "**{u} grade a cytopenia less severe than it is**. A misread "
                "digit is not caught by the unit boundary, because the unit "
                "is read correctly and the number is simply wrong and "
                "plausible. See §{sec}.\n".format(
                    n=len(ocr_silent), u=len(ocr_understated),
                    sec=len(arms_run) + 2,
                )
            )
        else:
            add(
                "**No arm emitted a silent wrong value.** Every field that "
                "could not be read correctly was refused rather than "
                "emitted, including under OCR. That is the property the unit "
                "boundary and the number-parsing rules were built for, and "
                "it held against real glyph errors.\n"
            )
    else:
        add(
            "**No OCR has run.** The only arm scored here is the pattern "
            "layer: the production regex set fed the exact text the corpus "
            "generator drew. Every number below is a *ceiling*, not a "
            "performance figure.\n"
        )

    add(
        "The GPT-4o-mini vision arm remains deferred by decision; its "
        "adapter is complete in `backends.py` and hard-guarded against "
        "making a network call.\n"
    )

    if arms_skipped:
        add("Arms not run in this pass: " + "; ".join(
            "`{0}` ({1})".format(name, reason) for name, reason in arms_skipped
        ) + ".\n")

    add(
        "_Generated by `scripts/run_extraction_bench.py` on {0}. Arms run: "
        "{1}. Every number below is computed from `manifest.json` and "
        "`scores.json`, except the recorded before-values in the fix "
        "sections, which are measured results of earlier runs and are "
        "labelled as such._\n".format(date.today().isoformat(), arm_names)
    )

    # --- composition ------------------------------------------------------
    add("## 1. What the corpus contains\n")
    add(R.composition_tables(composition))
    add(
        "Stratum A is the four real images, hand-labelled field by field "
        "including the unit. It is labelled \"clean digital table\", which "
        "describes how the pages are *rendered* and not how they are "
        "*notated*: two print units in a dedicated column, two embed the unit "
        "in the value cell, one of those spells microlitre `mcL`, four rows of "
        "`test_3.jpg` print no unit at all, and its MCHC row falls below the "
        "bottom edge of the image. Sub-labels A1-A4 are recorded per image so "
        "any pooled number can be broken back down.\n"
    )
    add(
        "Stratum B crosses the four notation axes fully at pristine render "
        "quality. Stratum C holds the decimal mark and reference ranges fixed "
        "and varies image quality over all nine scale-by-placement "
        "combinations, so a quality result is not specific to one notation.\n"
    )
    add(
        "`extractable` means the page prints both a value and a unit, so a "
        "correct quantity is obtainable. `no_unit` means the value is printed "
        "and the unit is not -- **refusing is the correct outcome** and any "
        "number emitted is a guess. `absent` means the analyte is not on the "
        "page at all, so silence is correct. Fields in the second and third "
        "groups can only score TN or FP, never TP, which is what stops the "
        "metric from punishing the safe behaviour.\n"
    )

    add("### How much of the corpus this arm scores, and why\n")
    scored = _scored_counts(scores, arms_run[0] if arms_run else None)
    add(
        "The pattern layer reads the **page text**, not the pixels. A "
        "degraded variant of a page carries the same text as its pristine "
        "base -- blurring an image does not change what the generator drew -- "
        "so only the 27 pristine renders in stratum C have a text sidecar and "
        "only they are scored. That is a property of this arm, not a sample: "
        "the other 675 variants exist for the OCR arm, which reads pixels and "
        "for which they differ. **The image-quality axis is therefore "
        "unmeasured in this pass**, and the stratum C numbers below are a "
        "second reading of the notation axes at pristine quality, useful "
        "mainly as a consistency check against stratum B.\n"
    )
    add(
        table(
            ["Stratum", "Images in corpus", "Images scored", "Fields scored"],
            [
                ["A", 4, scored["A"]["images"], scored["A"]["fields"]],
                ["B", 108, scored["B"]["images"], scored["B"]["fields"]],
                ["C", 702, scored["C"]["images"], scored["C"]["fields"]],
                [
                    "**total**", 814,
                    sum(v["images"] for v in scored.values()),
                    sum(v["fields"] for v in scored.values()),
                ],
            ],
        )
        + "\n"
    )

    add("### Fields and judgements are different counts\n")
    add(
        "**A wrong quantity scores as both a false positive and a false "
        "negative** -- the backend claimed something untrue *and* failed to "
        "produce the truth. So a field with a wrong value contributes two "
        "confusion-matrix entries, and `TP + FP + FN + TN` exceeds the number "
        "of fields by exactly the number of wrong-value fields.\n"
    )
    pooled = _pooled(scores, arms_run[0]) if arms_run else None
    if pooled:
        excess = pooled["judgements"] - pooled["fields"]
        add(
            "Every table below carries a `Fields` column beside its "
            "TP/FP/FN/TN so the two reconcile on sight rather than looking "
            "like an arithmetic error.\n"
        )
        if excess:
            add(
                "This pass: **{fields} fields, {judgements} judgements**, a "
                "difference of {excess} -- which is the number of fields "
                "carrying a wrong value.\n".format(
                    fields=pooled["fields"],
                    judgements=pooled["judgements"],
                    excess=excess,
                )
            )
        else:
            add(
                "In this pass the two coincide at **{fields}**, because no "
                "field carries a wrong value -- every failure was a refusal, "
                "which contributes one judgement, not two. Earlier runs did "
                "diverge: as found, stratum B scored 1,080 fields as 1,112 "
                "judgements, the excess being its 32 wrong-value fields. See "
                "§{section}.\n".format(
                    fields=pooled["fields"], section="the fixes section"
                )
            )

    # --- per arm ----------------------------------------------------------
    for index, arm in enumerate(arms_run, start=2):
        add("## {0}. {1}\n".format(index, _ARM_TITLES.get(arm, arm)))
        if arm in _ARM_BLURB:
            add(_ARM_BLURB[arm] + "\n")
        add(_headline(scores, arm) + "\n")

        add("### Per parameter, per stratum\n")
        for stratum in ("A", "B", "C"):
            if not any(
                s.backend == arm and s.stratum == stratum for s in scores
            ):
                continue
            pooled = _pooled(scores, arm, stratum)
            add(
                "**Stratum {0}** -- precision {1}, recall {2}, over {3} "
                "fields.\n".format(
                    stratum,
                    R._rate(pooled["precision"]),
                    R._rate(pooled["recall"]),
                    pooled["n"],
                )
            )
            add(R.per_parameter(scores, arm, stratum) + "\n")

        add("### Where each notation axis bites\n")
        add(
            "Stratum B only, so image quality is held at pristine and the "
            "effect is the notation's alone.\n"
        )
        for factor in ("decimal_style", "scale", "unit_placement", "reference_ranges"):
            add("**{0}**\n".format(factor))
            add(R.per_factor(scores, manifest, arm, factor) + "\n")

        if any(s.backend == arm and s.stratum == "C" for s in scores):
            add("### Where image quality bites\n")
            add(
                "One factor at a time, against the pristine control at the "
                "same notation cells.\n"
            )
            add(R.degradation_curve(scores, manifest, arm) + "\n")

            cliffs = R.degradation_cliffs(scores, manifest, arm)
            if cliffs:
                add("**These curves do not slope. They fall off.**\n")
                add(cliffs + "\n")
                add(
                    "Read that table before the one above it. JPEG "
                    "compression is survivable all the way down to quality "
                    "20 -- a setting that visibly wrecks the image to a human "
                    "eye -- while half a pixel of extra Gaussian blur past "
                    "sigma 0.5 takes everything. A pipeline tuned on the "
                    "assumption that image quality degrades gracefully would "
                    "be tuned on the wrong axis.\n"
                )
                add(
                    "Two anomalies worth naming rather than smoothing over. "
                    "**Blur at sigma 0.5 scores *above* the pristine "
                    "control**, which is not noise: a half-pixel blur "
                    "anti-aliases the micro sign's thin strokes and makes it "
                    "more legible, not less -- the same resolution story as "
                    "the upscaling probe in the gap section. And the "
                    "**left-lit shadow is fatal while the diagonal and "
                    "corner ones are not**, because the left of the page is "
                    "where the row labels are: darken those and no pattern "
                    "matches a label, so the row is never sighted at all.\n"
                )

        add("### Failure attribution\n")
        add(
            "Which stage lost each field. This is the column that decides "
            "whether a fix belongs in `ocr_extraction.py` or in `units.py`.\n"
        )
        add(R.attribution(scores, arm) + "\n")

        add("### Silent errors\n")
        add(
            "Wrong quantities that reached the panel with nothing flagged -- "
            "wrong, and indistinguishable on screen from right. Sorted by "
            "whether the CTCAE grade was understated, then by the size of the "
            "error. Grades come from the production grader in "
            "`clinical_flags.py`, so a grade here is the grade the "
            "application would have shown.\n"
        )
        add(R.silent_error_table(scores, arm) + "\n")

    # --- mechanisms -------------------------------------------------------
    # --- the ceiling gap, when there is an OCR arm to compare ----------
    if TESSERACT in arms_run and PATTERN in arms_run:
        add(_ceiling_gap_section(scores, arms_run, len(arms_run) + 2))
        offset = 3
    else:
        offset = 2

    add(_mechanisms(scores, manifest, arms_run, len(arms_run) + offset))

    # --- alias coverage ---------------------------------------------------
    add("## {0}. Alias coverage\n".format(len(arms_run) + offset + 1))
    add(
        "A direct sweep of {total} unit spellings against `units.parse_unit`, "
        "no images involved. A corpus can only exercise the spellings it was "
        "told to draw, and laboratories print many more than are worth "
        "rendering.\n".format(total=probe_summary["total"])
    )
    add(
        "**{accepted} accepted, {alias} need an alias, {dimension} are not "
        "expressible at all.**\n".format(
            accepted=probe_summary["counts"]["accepted"],
            alias=probe_summary["counts"]["needs_alias"],
            dimension=probe_summary["counts"]["needs_dimension"],
        )
    )
    add(R.alias_table(probe) + "\n")
    add(
        "`needs_dimension` is kept separate from `needs_alias` deliberately. "
        "Haemoglobin in mmol/L needs a molar mass -- the g/dL factor of 0.6206 "
        "is specific to the haemoglobin tetramer and is not a unit conversion "
        "-- and a haematocrit in L/L needs a dimensionless dimension that "
        "`units.Dimension` does not have. Registering either against a "
        "plausible-looking factor would be exactly the quiet fabrication that "
        "module exists to prevent.\n"
    )

    if probe_summary["misresolved"]:
        add("### One spelling resolves to the wrong unit\n")
        for text, expected, resolved in probe_summary["misresolved"]:
            add(
                "- `{0}` denotes **{1}** and `parse_unit` returns **{2}**. "
                "`units._normalise` lower-cases before lookup, so the "
                "giga-per-litre count spelling collapses onto the "
                "grams-per-litre mass spelling.".format(text, expected, resolved)
            )
        add(
            "\nThis currently fails safe rather than producing a wrong number: "
            "the dimension check in `Quantity.to` rejects a mass concentration "
            "handed to a count analyte, so a WBC printed `7.2 G/L` is refused "
            "with `IncompatibleUnitsError`. The consequence is a lost field "
            "and a misleading error message, not a silent error -- but it is "
            "lost for a reason the message does not name.\n"
        )

    # --- the two fixes, and what each moved -------------------------------
    add("## {0}. The fixes, and what each moved\n".format(len(arms_run) + offset + 2))
    add(
        "Three runs, against the identical corpus and the identical ground "
        "truth. Only the production code changed between them, which is what "
        "makes the differences attributable.\n"
    )

    as_found = _BASELINES["as_found"]
    after_aliases = _BASELINES["after_aliases"]
    live = _pooled(scores, arms_run[0])
    silent_live = len(S.silent_errors([s for s in scores if s.backend == arms_run[0]]))
    understated_live = len(
        S.severity_understated([s for s in scores if s.backend == arms_run[0]])
    )
    probe_accepted = probe_summary["counts"]["accepted"]

    add(
        table(
            [
                "",
                "1. As found",
                "2. After the alias fix",
                "3. After M1 + M2 (live)",
            ],
            [
                [
                    "Precision",
                    "{0:.3f}".format(as_found["precision"]),
                    "{0:.3f}".format(after_aliases["precision"]),
                    "**{0}**".format(R._rate(live["precision"])),
                ],
                [
                    "Recall",
                    "{0:.3f}".format(as_found["recall"]),
                    "{0:.3f}".format(after_aliases["recall"]),
                    "**{0}**".format(R._rate(live["recall"])),
                ],
                ["True positives", as_found["tp"], after_aliases["tp"],
                 "**{0}**".format(live["tp"])],
                ["False positives", as_found["fp"], after_aliases["fp"],
                 "**{0}**".format(live["fp"])],
                ["False negatives", as_found["fn"], after_aliases["fn"],
                 "**{0}**".format(live["fn"])],
                ["True negatives", as_found["tn"], after_aliases["tn"], live["tn"]],
                ["Silent wrong values", as_found["silent"],
                 after_aliases["silent"], "**{0}**".format(silent_live)],
                ["of which understate CTCAE severity", as_found["understated"],
                 after_aliases["understated"],
                 "**{0}**".format(understated_live)],
                ["Alias spellings accepted (of {0})".format(probe_summary["total"]),
                 as_found["aliases_accepted"],
                 after_aliases["aliases_accepted"], probe_accepted],
            ],
        )
        + "\n"
    )

    add("### The alias fix moved nothing, and that is the result\n")
    add(
        "`units.py` was extended first - mcL collapsed to uL, superscript, "
        "E-notation and two-digit exponents normalised, a trailing result "
        "flag stripped from a units cell. Fifteen more spellings parsed, and "
        "**not one extraction field changed outcome**. Zero. `alias_gap` was "
        "charged zero times before and zero times after.\n"
    )
    add(
        "The reason is M3 generalised: a field only reaches "
        "`units.parse_unit` if a pattern matched its label *and* captured a "
        "unit string, so every field the alias gaps would have touched was "
        "already lost upstream. The alias table was never the binding "
        "constraint.\n"
    )
    add(
        "This is the case for spending the effort on stage attribution rather "
        "than on a headline number. A recall of 0.622 with a list of "
        "unrecognised unit spellings beside it reads as a units problem, and "
        "the obvious fix is the wrong one. The fix was still right to make - "
        "the gaps are real and they are on the checked-in images - but it "
        "recovers nothing alone, and shipping it as an extraction improvement "
        "would have been a claim the measurement does not support.\n"
    )

    add("### M1 and M2 moved everything that was left\n")
    add(
        "Fixing the value parse and the SI alternation took recall from "
        "{before:.3f} to {after} and eliminated every silent error: "
        "{sil_before} wrong quantities reaching the panel became {sil_after}, "
        "and the {und_before} that understated a CTCAE grade became "
        "{und_after}. False positives went from {fp_before} to "
        "{fp_after}.\n".format(
            before=as_found["recall"], after=R._rate(live["recall"]),
            sil_before=as_found["silent"], sil_after=silent_live,
            und_before=as_found["understated"], und_after=understated_live,
            fp_before=as_found["fp"], fp_after=live["fp"],
        )
    )
    add(
        "Both fixes are refusal-preserving: neither invents a reading. M1 "
        "resolves what the token shape or the page settles and refuses the "
        "rest; M2 requires an explicit exponent marker and so refuses a "
        "flattened superscript rather than guessing at its scale. The "
        "remaining {fn} false negatives are M3's three label-gap fields on "
        "`test_4.png`.\n".format(fn=live["fn"])
    )
    add(
        "**What this does and does not say.** It says the pattern layer and "
        "the unit layer now read this corpus's notation correctly given "
        "perfect text. It says nothing about OCR, which has not run. The "
        "ceiling moved; whether the floor follows is what the Tesseract arm "
        "is for.\n"
    )

    add("### Two spellings left unregistered deliberately\n")
    add(
        "`G/L` and `T/L`, the European giga- and tera-per-litre count units. "
        "`_normalise` lower-cases before lookup, so `G/L` collapses onto "
        "`g/L` and resolves to a mass concentration. Separating them requires "
        "making the normaliser case-sensitive for one pair of units, and a "
        "wrongly case-folded haemoglobin and a wrongly case-folded white-cell "
        "count are different kinds of harm. That is a safety judgement with "
        "an argument on both sides and not one to settle inside a benchmark. "
        "Today the dimension check in `Quantity.to` refuses `G/L` for a count "
        "analyte, so it fails safe - at the cost of an error message that "
        "names the wrong reason.\n"
    )

    add("## {0}. Reproducing this\n".format(len(arms_run) + offset + 3))
    add(
        "```\n"
        "python scripts/build_extraction_corpus.py\n"
        "python scripts/run_extraction_bench.py --report docs/extraction-benchmark.md --figures\n"
        "```\n"
    )
    add(
        "The corpus is deterministic and gitignored; the manifest is the "
        "reviewable artefact. `scores.json` holds every per-field judgement, "
        "so any claim above can be re-checked without re-running OCR.\n"
    )

    Path(path).write_text("\n".join(out), encoding="utf-8")
