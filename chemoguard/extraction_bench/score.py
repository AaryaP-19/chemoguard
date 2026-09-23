"""Score a backend's reading of one field, and attribute every failure.

Precision and recall are computed over *emitted quantities*. A backend that
refuses a field claims nothing, so a refusal cannot be a false positive; a
backend that emits a number for a row printing no unit has claimed something
it could not know, so that is a false positive whatever the number happens to
be. Getting this the right way round matters more here than in an ordinary
classification problem, because in this application refusing is a supported
and correct outcome -- ``cbc.panel_from_extraction`` returns a ``skipped``
map precisely so that it can be -- and a metric that penalised it would score
the safe behaviour as failure.

    truth ``extractable``   emit correct -> TP
                            emit wrong   -> FP and FN
                            refuse       -> FN
                            silent       -> FN
    truth ``no_unit``       emit anything-> FP   (a unit was inferred)
                            refuse       -> TN   (correct, and legible)
                            silent       -> TN   (correct, and not legible)
    truth ``absent``        emit anything-> FP   (invented)
                            refuse/silent-> TN

The refusal/silence split under ``no_unit`` is kept because the two are not
equally useful even though both are correct. An operator cannot tell "this
report has no platelet count" from "the platelet count is there and I could
not read its unit", and only the second is something they can fix --
``ocr_extraction.UNITLESS_PATTERNS`` exists to make the second legible.

**Failure attribution is the point of this module.** Each loss is charged to
the stage that caused it:

``value_misread`` the number did not survive the value parse
``unit_misread``  the unit string was read as something else
``alias_gap``   the unit was read correctly and ``units.parse_unit`` does not
                know the spelling -- a defect in ``units.py``, not in OCR
``plausibility``the value canonicalised and failed its ``cbc`` bounds
``no_read``     the analyte was not matched at all
``unit_dropped``a value was read with no unit although the page printed one
``ambiguous_number`` the number was sighted and its separators left its
                value undecidable, so it was refused

``alias_gap`` is separated from ``unit_misread`` by trying ``parse_unit`` a second
time on a repaired spelling. The repairs are listed in ``_REPAIRS`` and are
all cases where the meaning is unambiguous -- mcL is a microlitre; a trailing
``(Low)`` is a flag, not part of a unit. A field that parses only after
repair was transcribed correctly, and the loss belongs to the alias table.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from chemoguard.extraction_bench.panels import CANONICAL
from chemoguard.extraction_bench.truth import EXTRACTABLE, NO_UNIT

#: Relative tolerance on a canonical value. Ground-truth values are exact
#: decimals, so this only has to absorb float round-trips -- it is nowhere
#: near loose enough to let a factor-of-ten error pass as correct.
VALUE_TOLERANCE = 1e-4

CORRECT = "correct"
WRONG_VALUE = "wrong_value"
REFUSED = "refused"
MISSED = "missed"
CORRECT_REFUSAL = "correct_refusal"
CORRECT_SILENCE = "correct_silence"
INFERRED_UNIT = "inferred_unit"
HALLUCINATED = "hallucinated"
FALSE_SIGHTING = "false_sighting"

VALUE_WRONG = "value_misread"
UNIT_WRONG = "unit_misread"
ALIAS_GAP = "alias_gap"
PLAUSIBILITY = "plausibility"
NO_READ = "no_read"
UNIT_DROPPED = "unit_dropped"
AMBIGUOUS_NUMBER = "ambiguous_number"

#: Spelling repairs that change no meaning. Applied only to decide *why* a
#: field was lost, never to rescue one: the scored run is the production code
#: as it stands.
_REPAIRS = (
    # A flag appended to a units cell. test_2.jpg prints "cells/uL (Low)".
    (re.compile(r"\s*\((?:low|high|h|l|abnormal|critical|panic)\)\s*$", re.I), ""),
    # mcL is microlitre. The screenshot report prints "cells/mcL" and "/mcL".
    (re.compile(r"mc\s*l\b", re.I), "uL"),
    # "million cells/uL" is "million/uL"; the word "cells" is redundant once
    # the multiplier is stated.
    (re.compile(r"\b(million|thousand)s?\s+cells\s*/", re.I), r"\1/"),
)


def repair_unit(text: str) -> str:
    """Normalise a unit spelling without changing what it denotes."""
    out = text
    for pattern, replacement in _REPAIRS:
        out = pattern.sub(replacement, out)
    return out.strip()


def denote(text) -> tuple:
    """What a unit string means, and whether ``units.py`` could tell.

    Returns ``(canonical_symbol, how)`` where ``how`` is ``"direct"`` if
    ``parse_unit`` accepted the spelling, ``"repaired"`` if it accepted a
    documented repair of it, and None if the string denotes nothing we
    recognise. ``"repaired"`` is the alias gap.
    """
    from units import UnitError, parse_unit

    if text is None:
        return None, None

    try:
        return parse_unit(text).symbol, "direct"
    except UnitError:
        pass

    repaired = repair_unit(str(text))
    if repaired != str(text):
        try:
            return parse_unit(repaired).symbol, "repaired"
        except UnitError:
            pass

    return None, None


def _ctcae_grade(analyte: str, canonical_value: float):
    """CTCAE grade for the four analytes that have published criteria.

    Uses the production grader, so a grade reported here is the grade the
    application would have shown.
    """
    from clinical_flags import flag_for
    from units import Quantity, parse_unit

    try:
        return flag_for(
            analyte, Quantity(canonical_value, parse_unit(CANONICAL[analyte]))
        ).grade
    except (KeyError, ValueError):
        return None


@dataclass(frozen=True)
class FieldScore:
    """One backend's outcome on one field, with the failure attributed."""

    backend: str
    image_id: str
    stratum: str
    analyte: str
    truth_status: str
    outcome: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    #: Which stage lost the field. None when nothing was lost.
    stage: Optional[str] = None
    detail: Optional[str] = None
    true_canonical: Optional[float] = None
    read_canonical: Optional[float] = None
    #: log10 of read/true. 1.0 is a ten-fold overstatement, -3.0 a
    #: thousand-fold understatement.
    log10_ratio: Optional[float] = None
    #: True when a wrong quantity reached the panel -- wrong, and nothing
    #: downstream complained. The failure mode with no symptom.
    silent: bool = False
    grade_true: Optional[int] = None
    grade_read: Optional[int] = None

    @property
    def grade_changed(self) -> bool:
        return (
            self.grade_true is not None
            and self.grade_read is not None
            and self.grade_true != self.grade_read
        )

    @property
    def understated_severity(self) -> bool:
        """The read value is graded *less* severe than the truth.

        The direction that matters. An overstated grade causes an
        unnecessary look; an understated one causes a missed cytopenia, and
        every error the old magnitude inference made pointed this way
        (CLAUDE.md Finding 4).
        """
        return (
            self.grade_true is not None
            and self.grade_read is not None
            and self.grade_read < self.grade_true
        )


def _emitted(reading, analyte):
    """The canonical value a reading put in the panel, or None."""
    panel = reading.panel
    if panel is None:
        return None
    quantity = panel.get(analyte)
    if quantity is None:
        return None
    return quantity.magnitude_in(CANONICAL[analyte])


def _attribute(truth, raw, skipped_entry) -> tuple:
    """Why an extractable field did not come back correct.

    Ordered so the earliest stage that failed is the one charged: a unit
    misread on top of a value misread is reported as the value misread,
    because fixing the alias table would not have saved it.
    """
    if raw is None:
        return NO_READ, "analyte not matched"

    read_value = raw.get("value") if isinstance(raw, dict) else raw
    read_unit = raw.get("unit") if isinstance(raw, dict) else None

    # Checked before the None-value case, which it would otherwise look like.
    # An ambiguous number is a sighting the extractor refused to resolve, not
    # a row it failed to find, and the two want different fixes.
    if isinstance(raw, dict) and raw.get("ambiguous_number") is not None:
        return AMBIGUOUS_NUMBER, "printed {0!r}; separators undecidable".format(
            raw["ambiguous_number"]
        )

    if read_value is None:
        return NO_READ, "no value in reading"

    value_ok = math.isclose(
        float(read_value), float(truth.value), rel_tol=VALUE_TOLERANCE
    )

    if read_unit is None:
        # A value read with no unit, from a page that printed one. Either OCR
        # lost the unit glyphs or no pattern reached them.
        return UNIT_DROPPED, "value {0} read without a unit".format(read_value)

    symbol, how = denote(read_unit)

    if not value_ok:
        return VALUE_WRONG, "read {0!r}, printed {1!r}".format(
            read_value, truth.printed_value
        )
    if symbol is None:
        return UNIT_WRONG, "unit {0!r} denotes nothing recognised".format(read_unit)
    if symbol != truth.unit:
        return UNIT_WRONG, "unit {0!r} denotes {1}, printed {2}".format(
            read_unit, symbol, truth.unit
        )
    if how == "repaired":
        return ALIAS_GAP, "units.parse_unit does not accept {0!r}".format(read_unit)
    if skipped_entry and skipped_entry.get("reason") == "ImplausibleValueError":
        return PLAUSIBILITY, skipped_entry.get("detail")

    return None, None


def score_field(reading, image, truth) -> FieldScore:
    """Score one field."""
    analyte = truth.analyte
    raw = (reading.raw or {}).get(analyte)
    skipped_entry = (reading.skipped or {}).get(analyte)
    read_canonical = _emitted(reading, analyte)
    common = dict(
        backend=reading.backend,
        image_id=image.image_id,
        stratum=image.stratum,
        analyte=analyte,
        truth_status=truth.status,
    )

    # --- the page printed a value and a unit ------------------------------
    if truth.status == EXTRACTABLE:
        true_canonical = truth.canonical_value

        if read_canonical is not None:
            if math.isclose(read_canonical, true_canonical, rel_tol=VALUE_TOLERANCE):
                return FieldScore(
                    outcome=CORRECT,
                    tp=1,
                    true_canonical=true_canonical,
                    read_canonical=read_canonical,
                    **common,
                )

            stage, detail = _attribute(truth, raw, skipped_entry)
            return FieldScore(
                outcome=WRONG_VALUE,
                fp=1,
                fn=1,
                stage=stage or VALUE_WRONG,
                detail=detail,
                true_canonical=true_canonical,
                read_canonical=read_canonical,
                log10_ratio=_log10_ratio(read_canonical, true_canonical),
                silent=True,
                grade_true=_ctcae_grade(analyte, true_canonical),
                grade_read=_ctcae_grade(analyte, read_canonical),
                **common,
            )

        stage, detail = _attribute(truth, raw, skipped_entry)
        if stage is None:
            # Nothing emitted and no stage owns the loss: the reading was
            # correct in every part and still did not arrive. Recorded
            # rather than smoothed over, because it means this scorer's
            # model of the pipeline is incomplete.
            stage, detail = "unexplained", "correct reading absent from panel"

        return FieldScore(
            outcome=REFUSED if skipped_entry else (MISSED if raw is None else REFUSED),
            fn=1,
            stage=stage,
            detail=detail,
            true_canonical=true_canonical,
            **common,
        )

    # --- the page printed a value and no unit -----------------------------
    if truth.status == NO_UNIT:
        if read_canonical is not None:
            return FieldScore(
                outcome=INFERRED_UNIT,
                fp=1,
                stage="unit_invented",
                detail="page prints no unit; backend emitted {0} {1}".format(
                    read_canonical, CANONICAL[analyte]
                ),
                read_canonical=read_canonical,
                silent=True,
                **common,
            )
        return FieldScore(
            outcome=CORRECT_REFUSAL if skipped_entry else CORRECT_SILENCE,
            tn=1,
            detail=(skipped_entry or {}).get("reason"),
            **common,
        )

    # --- the analyte is not on the page -----------------------------------
    if read_canonical is not None:
        return FieldScore(
            outcome=HALLUCINATED,
            fp=1,
            stage="invented",
            detail="analyte absent from the page; backend emitted {0}".format(
                read_canonical
            ),
            read_canonical=read_canonical,
            silent=True,
            **common,
        )

    return FieldScore(
        outcome=FALSE_SIGHTING if (raw or skipped_entry) else CORRECT_SILENCE,
        tn=1,
        detail=(skipped_entry or {}).get("reason"),
        **common,
    )


def _log10_ratio(read: float, true: float):
    if not read or not true or read <= 0 or true <= 0:
        return None
    return round(math.log10(read / true), 4)


def score_reading(reading, image) -> List[FieldScore]:
    """Score every field of one image for one backend."""
    return [
        score_field(reading, image, truth) for truth in image.fields.values()
    ]


# --- aggregation -------------------------------------------------------------


def _rates(counts) -> Dict[str, object]:
    """Confusion counts and the two rates, with fields and judgements apart.

    ``fields`` counts ``(image, analyte)`` pairs; ``judgements`` counts
    confusion-matrix entries. They differ because **a wrong quantity scores
    as both a false positive and a false negative** -- the backend claimed
    something untrue *and* failed to produce the truth -- so each such field
    contributes two judgements. ``judgements - fields`` is therefore exactly
    the number of wrong-quantity fields, and reporting only one of the two
    numbers makes the tables look as though they do not add up.
    """
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    claimed = tp + fp
    available = tp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fields": counts["fields"],
        "judgements": tp + fp + fn + tn,
        #: Retained under its old name; equals ``judgements``.
        "n": tp + fp + fn + tn,
        "precision": (tp / claimed) if claimed else None,
        "recall": (tp / available) if available else None,
    }


def aggregate(scores, *keys) -> Dict[tuple, Dict[str, object]]:
    """Precision and recall grouped by any combination of score attributes.

    Precision is None where nothing was claimed and recall is None where
    nothing was available, rather than 0.0. A backend that emitted nothing
    has undefined precision, not perfect or terrible precision, and
    substituting a number there would put a fiction into a table.
    """
    buckets = defaultdict(
        lambda: Counter({"tp": 0, "fp": 0, "fn": 0, "tn": 0, "fields": 0})
    )

    for score in scores:
        key = tuple(getattr(score, attribute) for attribute in keys)
        buckets[key].update(
            {
                "tp": score.tp, "fp": score.fp, "fn": score.fn, "tn": score.tn,
                "fields": 1,
            }
        )

    return {key: _rates(counts) for key, counts in sorted(buckets.items(), key=str)}


def outcome_counts(scores, *keys) -> Dict[tuple, Counter]:
    """How often each outcome occurred, grouped."""
    buckets = defaultdict(Counter)
    for score in scores:
        buckets[tuple(getattr(score, a) for a in keys)][score.outcome] += 1
    return dict(sorted(buckets.items(), key=str))


def stage_counts(scores, *keys) -> Dict[tuple, Counter]:
    """Which stage lost each field, grouped. The attribution table."""
    buckets = defaultdict(Counter)
    for score in scores:
        if score.stage:
            buckets[tuple(getattr(score, a) for a in keys)][score.stage] += 1
    return dict(sorted(buckets.items(), key=str))


def silent_errors(scores) -> List[FieldScore]:
    """Wrong quantities that reached the panel with nothing flagged.

    The result this benchmark exists to produce. A refused field is a
    nuisance; one of these is a number on a screen that is wrong and looks
    exactly like a number that is right.
    """
    return [score for score in scores if score.silent]


def severity_understated(scores) -> List[FieldScore]:
    """Silent errors that graded a cytopenia as less severe than it is.

    CLAUDE.md Finding 4: every error the old magnitude heuristic made pointed
    this way, because a dangerously low count in cells/uL looks like a
    comfortable count in thousands/uL. This is the direct test of whether any
    route to that outcome survives.
    """
    return [score for score in scores if score.understated_severity]


def by_degradation(scores, manifest) -> Dict[tuple, Dict[str, object]]:
    """Recall per backend per degradation family and severity.

    Keyed ``(backend, family, severity, severity_label)``, so a family's
    points sort into a curve against a common pristine control.
    """
    factors = {image.image_id: image.factors for image in manifest}
    buckets = defaultdict(
        lambda: Counter({"tp": 0, "fp": 0, "fn": 0, "tn": 0, "fields": 0})
    )

    for score in scores:
        if score.stratum != "C":
            continue
        meta = factors.get(score.image_id, {})
        key = (
            score.backend,
            meta.get("degradation_family"),
            meta.get("severity"),
            meta.get("severity_label"),
        )
        buckets[key].update(
            {
                "tp": score.tp, "fp": score.fp, "fn": score.fn, "tn": score.tn,
                "fields": 1,
            }
        )

    return {key: _rates(counts) for key, counts in sorted(buckets.items(), key=str)}


def by_factor(scores, manifest, factor: str, stratum: str = "B"):
    """Precision and recall per level of one notation axis."""
    factors = {image.image_id: image.factors for image in manifest}
    buckets = defaultdict(
        lambda: Counter({"tp": 0, "fp": 0, "fn": 0, "tn": 0, "fields": 0})
    )

    for score in scores:
        if score.stratum != stratum:
            continue
        level = factors.get(score.image_id, {}).get(factor)
        buckets[(score.backend, str(level))].update(
            {
                "tp": score.tp, "fp": score.fp, "fn": score.fn, "tn": score.tn,
                "fields": 1,
            }
        )

    return {key: _rates(counts) for key, counts in sorted(buckets.items(), key=str)}
