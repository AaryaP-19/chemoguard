"""Turn scores into the report tables.

Every table reports counts beside every rate. A precision of 1.000 over four
claims and a precision of 1.000 over four hundred are not the same finding,
and a table that prints only the rate hides which one it is. Undefined rates
print as ``n/a`` rather than 0.000: a backend that claimed nothing has
undefined precision, and writing a number there would put a fiction in the
table.

Follows the reporting rule in CLAUDE.md: never a bare headline number. The
per-parameter and per-stratum breakdowns are the result; the pooled figure is
a summary of them and is reported last, not first.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

from chemoguard.extraction_bench import degrade, score as S
from chemoguard.extraction_bench.panels import ANALYTE_ORDER
from chemoguard.extraction_bench.truth import ABSENT, EXTRACTABLE, NO_UNIT


def _rate(value) -> str:
    return "n/a" if value is None else "{0:.3f}".format(value)


def table(headers: List[str], rows: List[List[str]]) -> str:
    """A GitHub-flavoured markdown table."""
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def composition_tables(comp: Dict[str, object]) -> str:
    """Strata sizes and the contingency counts behind the balance claim."""
    parts = ["### Corpus composition\n"]

    status_rows = []
    for stratum in ("A", "B", "C"):
        if stratum not in comp["images_by_stratum"]:
            continue
        status = comp["status_by_stratum"][stratum]
        status_rows.append(
            [
                stratum,
                comp["images_by_stratum"][stratum],
                comp["fields_by_stratum"][stratum],
                status.get(EXTRACTABLE, 0),
                status.get(NO_UNIT, 0),
                status.get(ABSENT, 0),
            ]
        )
    status_rows.append(
        [
            "**total**",
            comp["total_images"],
            comp["total_fields"],
            sum(v.get(EXTRACTABLE, 0) for v in comp["status_by_stratum"].values()),
            sum(v.get(NO_UNIT, 0) for v in comp["status_by_stratum"].values()),
            sum(v.get(ABSENT, 0) for v in comp["status_by_stratum"].values()),
        ]
    )
    parts.append(
        table(
            ["Stratum", "Images", "Fields", "extractable", "no_unit", "absent"],
            status_rows,
        )
    )

    parts.append("\n### Stratum B axis balance\n")
    parts.append(
        "Images per level. Equal counts within a row mean a per-level "
        "comparison is not confounded by another axis.\n"
    )
    balance_rows = []
    for axis, counts in comp["stratum_b_axis_balance"].items():
        balance_rows.append(
            [axis, ", ".join("{0} = {1}".format(k, v) for k, v in sorted(counts.items()))]
        )
    parts.append(table(["Axis", "Images per level"], balance_rows))

    if comp["stratum_c_family_counts"]:
        parts.append("\n### Stratum C degradation families\n")
        parts.append(
            table(
                ["Family", "Images"],
                [
                    [family, comp["stratum_c_family_counts"].get(family, 0)]
                    for family in degrade.FAMILY_ORDER
                    if family in comp["stratum_c_family_counts"]
                ],
            )
        )

    return "\n".join(parts) + "\n"


def per_parameter(scores, backend: str, stratum: str) -> str:
    """Precision and recall per analyte for one backend in one stratum."""
    subset = [s for s in scores if s.backend == backend and s.stratum == stratum]
    agg = S.aggregate(subset, "analyte")

    rows = []
    for analyte in ANALYTE_ORDER:
        stats = agg.get((analyte,))
        if stats is None:
            continue
        rows.append(
            [
                analyte,
                stats["fields"],
                _rate(stats["precision"]),
                _rate(stats["recall"]),
                stats["tp"],
                stats["fp"],
                stats["fn"],
                stats["tn"],
            ]
        )

    pooled = S.aggregate(subset, "backend").get((backend,))
    if pooled:
        rows.append(
            [
                "**pooled**",
                pooled["fields"],
                _rate(pooled["precision"]),
                _rate(pooled["recall"]),
                pooled["tp"],
                pooled["fp"],
                pooled["fn"],
                pooled["tn"],
            ]
        )

    return table(
        ["Parameter", "Fields", "Precision", "Recall", "TP", "FP", "FN", "TN"],
        rows,
    )


def ceiling_gap(scores, ceiling_arm: str, ocr_arm: str, by: str = "stratum") -> str:
    """The gap between the pattern-layer ceiling and an OCR arm.

    Both arms run the same patterns, the same alias table and the same
    canonicalisation over the same ground truth. The *only* difference
    between them is where the text came from: the generator's own output
    versus Tesseract's reading of a rendered image. So the difference in
    recall is glyph reading, and nothing else -- there is no third
    explanation available, which is what makes the subtraction meaningful
    rather than suggestive.

    Grouped by ``stratum`` or ``analyte``. Only fields both arms scored are
    compared; the pattern layer does not score degraded variants, so pooling
    without that restriction would compare different denominators and the
    gap would be an artefact of coverage.
    """
    shared = _shared_fields(scores, ceiling_arm, ocr_arm)
    key = "stratum" if by == "stratum" else "analyte"

    ceiling = S.aggregate(
        [s for s in shared if s.backend == ceiling_arm], key
    )
    ocr = S.aggregate([s for s in shared if s.backend == ocr_arm], key)

    order = ("A", "B", "C") if key == "stratum" else ANALYTE_ORDER
    rows = []
    for name in order:
        top, bottom = ceiling.get((name,)), ocr.get((name,))
        if not top or not bottom:
            continue
        rows.append(_gap_row(name, top, bottom))

    pooled_top = S.aggregate(
        [s for s in shared if s.backend == ceiling_arm], "backend"
    ).get((ceiling_arm,))
    pooled_bottom = S.aggregate(
        [s for s in shared if s.backend == ocr_arm], "backend"
    ).get((ocr_arm,))
    if pooled_top and pooled_bottom:
        rows.append(_gap_row("**pooled**", pooled_top, pooled_bottom))

    return table(
        [
            key,
            "Fields",
            "Ceiling recall",
            "OCR recall",
            "OCR - ceiling",
            "Ceiling prec.",
            "OCR prec.",
        ],
        rows,
    )


def _gap_row(name, top, bottom):
    gap = (
        None
        if top["recall"] is None or bottom["recall"] is None
        else top["recall"] - bottom["recall"]
    )
    return [
        name,
        top["fields"],
        _rate(top["recall"]),
        _rate(bottom["recall"]),
        "n/a" if gap is None else "**{0:+.3f}**".format(-gap),
        _rate(top["precision"]),
        _rate(bottom["precision"]),
    ]


def _shared_fields(scores, *arms):
    """Only the ``(image, analyte)`` fields every named arm scored.

    Without this the comparison is meaningless: the pattern layer skips the
    675 degraded variants, so a pooled difference would mostly measure which
    images each arm looked at.
    """
    per_arm = {
        arm: {(s.image_id, s.analyte) for s in scores if s.backend == arm}
        for arm in arms
    }
    common = set.intersection(*per_arm.values()) if per_arm else set()
    return [
        s for s in scores
        if s.backend in arms and (s.image_id, s.analyte) in common
    ]


def per_factor(scores, manifest, backend: str, factor: str, stratum="B") -> str:
    """Precision and recall per level of one notation axis."""
    stats = S.by_factor(scores, manifest, factor, stratum=stratum)
    rows = [
        [
            level,
            v["fields"],
            _rate(v["precision"]),
            _rate(v["recall"]),
            v["tp"],
            v["fp"],
            v["fn"],
            v["tn"],
        ]
        for (arm, level), v in stats.items()
        if arm == backend
    ]
    return table(
        [factor, "Fields", "Precision", "Recall", "TP", "FP", "FN", "TN"], rows
    )


def degradation_curve(scores, manifest, backend: str) -> str:
    """Recall against severity, one row per degradation level."""
    stats = S.by_degradation(scores, manifest)
    rows = []
    for (arm, family, severity, label), v in stats.items():
        if arm != backend:
            continue
        rows.append(
            [
                family,
                label,
                v["fields"],
                _rate(v["precision"]),
                _rate(v["recall"]),
                v["tp"],
                v["fp"],
                v["fn"],
                v["tn"],
            ]
        )
    rows.sort(key=lambda row: (degrade.FAMILY_ORDER.index(row[0]), str(row[1])))
    return table(
        ["Family", "Severity", "Fields", "Precision", "Recall",
         "TP", "FP", "FN", "TN"],
        rows,
    )


def degradation_cliffs(scores, manifest, backend: str, drop: float = 0.5) -> str:
    """Where each degradation family stops being survivable.

    A 26-row table shows the numbers but not the shape, and the shape is the
    result: these curves do not slope, they fall off. Reported as the last
    level that still works and the first that does not, per family, computed
    rather than eyeballed.
    """
    stats = S.by_degradation(scores, manifest)
    control = next(
        (
            v["recall"]
            for (arm, fam, _s, _l), v in stats.items()
            if arm == backend and fam == degrade.PRISTINE
        ),
        None,
    )
    # No control, or a control of zero, and the comparison says nothing: every
    # level is trivially "above half of nothing". That is the psm 6 arm, whose
    # pristine recall is 0.000, and the pattern arm, which has no degraded
    # results at all. Both get no table rather than a misleading one.
    if not control:
        return ""

    rows = []
    for family in degrade.FAMILY_ORDER:
        if family in (degrade.PRISTINE, degrade.COMBINED):
            continue
        ladder = sorted(
            (
                (severity, label, v["recall"])
                for (arm, fam, severity, label), v in stats.items()
                if arm == backend and fam == family and v["recall"] is not None
            ),
            key=lambda row: row[0],
        )
        if not ladder:
            continue

        survived, broke = None, None
        for _severity, label, recall in ladder:
            if recall >= control * drop:
                survived = (label, recall)
            elif broke is None:
                broke = (label, recall)

        rows.append(
            [
                family,
                "{0} ({1:.3f})".format(*survived) if survived else "none",
                "{0} ({1:.3f})".format(*broke) if broke else "never",
                "{0:.3f}".format(min(r for _s, _l, r in ladder)),
            ]
        )

    if not rows:
        return ""

    return table(
        [
            "Family",
            "Worst level still above half the control",
            "First level below it",
            "Floor",
        ],
        rows,
    )


def attribution(scores, backend: str) -> str:
    """Which stage lost each field.

    The column that decides whether a fix belongs in ``ocr_extraction.py`` or
    in ``units.py``.
    """
    subset = [s for s in scores if s.backend == backend and s.stage]
    counts = Counter(s.stage for s in subset)
    total = sum(counts.values())

    rows = [
        [
            stage,
            n,
            "{0:.1%}".format(n / total) if total else "n/a",
            _STAGE_OWNER.get(stage, "-"),
        ]
        for stage, n in counts.most_common()
    ]
    return table(["Stage", "Fields lost", "Share", "Owner"], rows)


_STAGE_OWNER = {
    S.VALUE_WRONG: "`ocr_extraction._NUMBER` / the value parse",
    S.UNIT_WRONG: "the unit string was read as something else",
    S.ALIAS_GAP: "`units.py` alias table",
    S.PLAUSIBILITY: "`cbc.py` bounds (a refusal, so it failed safe)",
    S.NO_READ: "`ocr_extraction.py` patterns",
    S.UNIT_DROPPED: "`ocr_extraction.py` patterns",
    S.AMBIGUOUS_NUMBER: "refused: separators undecidable, no page evidence",
    "unit_invented": "a unit was inferred where none was printed",
    "invented": "a value was returned for an analyte not on the page",
    "unexplained": "unattributed -- the scorer's model is incomplete here",
}


def silent_error_table(scores, backend: str, limit: int = 40) -> str:
    """Wrong quantities that reached the panel, worst first.

    **One row per distinct error, not per affected field.** Rows are
    deduplicated on ``(parameter, true value, read value, truth status)`` and
    the ``Fields`` column gives how many fields each row stands for, so the
    row count and the instance count in the surrounding prose reconcile. The
    same misparse recurs across notation cells and panels, and listing it
    once per occurrence would bury eight distinct mechanisms under dozens of
    identical lines.

    Sorted by whether the CTCAE grade was understated, then by the size of
    the error. Understated severity leads because that is the direction with
    the clinical consequence.
    """
    subset = [s for s in S.silent_errors(scores) if s.backend == backend]
    subset.sort(
        key=lambda s: (
            not s.understated_severity,
            -abs(s.log10_ratio or 0),
        )
    )

    instances = Counter(
        (s.analyte, s.true_canonical, s.read_canonical, s.truth_status)
        for s in subset
    )

    rows = []
    seen = set()
    for s in subset:
        key = (s.analyte, s.true_canonical, s.read_canonical, s.truth_status)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            [
                s.analyte,
                instances[key],
                "{0:g}".format(s.true_canonical) if s.true_canonical else "-",
                "{0:g}".format(s.read_canonical) if s.read_canonical else "-",
                "{0:+.2f}".format(s.log10_ratio) if s.log10_ratio is not None else "-",
                "{0} -> {1}".format(s.grade_true, s.grade_read)
                if s.grade_changed
                else ("unchanged" if s.grade_true is not None else "n/a"),
                "**yes**" if s.understated_severity else "no",
                (s.detail or "")[:70],
            ]
        )
        if len(rows) >= limit:
            break

    if not rows:
        return (
            "_None. Every wrong reading was refused rather than emitted, so "
            "no wrong quantity reached the panel._"
        )

    return table(
        [
            "Parameter",
            "Fields",
            "True (canonical)",
            "Read",
            "log10 ratio",
            "CTCAE grade",
            "Understates severity",
            "Cause",
        ],
        rows,
    )


def alias_table(results) -> str:
    """The alias-coverage probe, gaps first."""
    from chemoguard.extraction_bench.alias_probe import (
        ACCEPTED,
        NEEDS_ALIAS,
        NEEDS_DIMENSION,
    )

    order = {NEEDS_ALIAS: 0, NEEDS_DIMENSION: 1, ACCEPTED: 2}
    rows = []
    for result in sorted(results, key=lambda r: (order[r.status], r.probe.text)):
        if result.status == ACCEPTED:
            continue
        rows.append(
            [
                "`{0}`".format(result.probe.text),
                result.probe.expected or "-",
                result.resolved or "rejected",
                result.status,
                result.probe.note or "",
            ]
        )
    return table(
        ["Printed spelling", "Denotes", "parse_unit gives", "Status", "Note"], rows
    )
