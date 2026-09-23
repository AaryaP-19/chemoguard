"""Tests for the extraction benchmark itself.

A benchmark is only worth its scorer. These tests exist because an error in
``score.py`` would not look like a failure -- it would look like a finding,
and it would be reported with a table and a figure behind it. So the scorer
is tested on hand-built cases whose right answer is obvious by inspection,
the generator is tested for self-consistency between what it draws and what
it claims to have drawn, and the independent conversion table in
``panels.py`` is tested against ``units.py`` so a divergence fails loudly
instead of the two quietly agreeing on the wrong number.

The Pillow-dependent tests skip when Pillow is absent. Nothing here needs
Tesseract, Poppler or a network.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from chemoguard.extraction_bench import alias_probe, corpus, degrade, panels, score
from chemoguard.extraction_bench.backends import GPTBackend, NetworkDisabledError
from chemoguard.extraction_bench.render import (
    ABSENT_UNIT,
    COLUMN,
    COMMA,
    LABEL,
    POINT,
    PageSpec,
    style_number,
    to_display,
)
from chemoguard.extraction_bench.stratum_a import stratum_a
from chemoguard.extraction_bench.truth import (
    ABSENT,
    EXTRACTABLE,
    NO_UNIT,
    FieldTruth,
    ImageTruth,
)


def _pillow():
    try:
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False


needs_pillow = pytest.mark.skipif(not _pillow(), reason="Pillow not installed")


# --- the independent conversion table ---------------------------------------


def test_panel_factors_agree_with_units_module():
    """``panels._FACTORS`` is a deliberate reimplementation of ``Unit.to_base``.

    Ground truth is computed from it rather than from the module under test,
    so that a conversion bug in ``units.py`` cannot agree with itself and
    become invisible. That only works while the two agree on the cases they
    share, which is what this asserts.
    """
    from units import ALL_UNITS

    for unit in ALL_UNITS:
        if unit.symbol not in panels._FACTORS:
            continue
        assert Decimal(str(unit.to_base)) == panels._FACTORS[unit.symbol], unit.symbol


def test_canonical_units_match_cbc():
    from cbc import ANALYTES

    assert {k: v.canonical.symbol for k, v in ANALYTES.items()} == panels.CANONICAL


@pytest.mark.parametrize("name", ("normal", "cytopenic", "severe"))
def test_panels_are_internally_coherent(name):
    """ANC = WBC x neutrophils%, exactly.

    An incoherent panel would make a correctly derived ANC look like an
    extraction error, which is a defect in the benchmark rather than a
    finding about a backend.
    """
    panels.PANELS[name].check_anc_identity()


@pytest.mark.parametrize("name", ("normal", "cytopenic", "severe"))
def test_panels_are_physiologically_plausible(name):
    """Every panel value must survive ``cbc.quantity_for``.

    A panel outside the plausibility bounds would be refused by the
    application for a reason that has nothing to do with extraction, and
    every field built from it would score as a loss.
    """
    from cbc import quantity_for

    for analyte, value in panels.PANELS[name].values.items():
        quantity_for(analyte, float(value), panels.CANONICAL[analyte])


@pytest.mark.parametrize("scale", tuple(panels.SCALES))
@pytest.mark.parametrize("name", ("normal", "cytopenic", "severe"))
def test_printed_number_round_trips_to_its_canonical_value(name, scale):
    """The printed glyphs are authoritative, and they must convert back.

    A report printing ANC as "0.9 K/uL" states 900 cells/uL. Ground truth is
    re-derived from the printed text for exactly that reason, and this
    asserts the derivation is consistent: parsing the printed number in the
    printed unit has to land on the recorded canonical value.
    """
    from units import Quantity

    panel = panels.PANELS[name]
    for analyte in panels.ANALYTE_ORDER:
        text, _printed_unit, canonical, true_value = panels.row_for(
            panel, analyte, scale
        )
        assert Decimal(text) == true_value

        expected = panels.convert(true_value, canonical, panels.CANONICAL[analyte])
        got = Quantity.of(float(true_value), canonical).magnitude_in(
            panels.CANONICAL[analyte]
        )
        assert got == pytest.approx(float(expected))


# --- notation styling -------------------------------------------------------


@pytest.mark.parametrize(
    "plain,style,expected",
    [
        ("4.80", POINT, "4.80"),
        ("4.80", COMMA, "4,80"),
        ("250000", POINT, "250,000"),
        ("250000", COMMA, "250.000"),
        ("6500", POINT, "6,500"),
        ("900", POINT, "900"),         # no grouping below four digits
        ("900", COMMA, "900"),
        ("0.9", COMMA, "0,9"),
        ("142", POINT, "142"),
    ],
)
def test_style_number(plain, style, expected):
    assert style_number(plain, style) == expected


def test_display_uses_the_micro_sign():
    """The page must print what a real report prints.

    ``panels.py`` declares units in ASCII for the reason ``units.py`` gives,
    but a page showing "uL" would not be a realistic report and whether
    Tesseract reads the micro sign is part of what is being measured.
    """
    assert to_display("cells/uL") == "cells/µL"
    assert to_display("g/dL") == "g/dL"


# --- the generator's self-consistency ---------------------------------------


@needs_pillow
@pytest.mark.parametrize("placement", (COLUMN, LABEL, ABSENT_UNIT))
def test_rendered_truth_matches_the_drawn_text(placement):
    """Ground truth must quote glyphs that are actually on the page.

    The single most dangerous failure a generator can have is claiming a
    value it did not draw, because then every backend scores a loss on it and
    the benchmark reports a finding about nothing.
    """
    from chemoguard.extraction_bench.render import render

    rendered = render(
        PageSpec(
            panel=panels.PANELS["cytopenic"],
            scale="conventional_k",
            placement=placement,
            decimal_style=POINT,
            reference_ranges=True,
        )
    )
    fields = corpus.truth_from_render(rendered)

    for row in rendered.rows:
        truth = fields[row.analyte]
        assert truth.printed_value in rendered.text
        assert truth.printed_value == row.value.text.split(" ")[0]

        if truth.status == EXTRACTABLE:
            assert truth.printed_unit is not None
            assert truth.printed_unit in rendered.text
        else:
            assert truth.status == NO_UNIT
            assert truth.printed_unit is None


@needs_pillow
def test_absent_placement_prints_no_unit_anywhere():
    """"Absent" has to mean absent from every column, including the label.

    "Neutrophils %" states its own unit, so that row's label has to change
    too or the axis level is a fiction for one analyte in ten.
    """
    from chemoguard.extraction_bench.render import render

    rendered = render(
        PageSpec(
            panel=panels.PANELS["normal"],
            scale="conventional_cells",
            placement=ABSENT_UNIT,
            decimal_style=POINT,
            reference_ranges=True,
        )
    )

    for unit_text in ("g/dL", "cells/µL", "millions/µL", "fL", "pg", "%"):
        assert unit_text not in rendered.text, unit_text

    assert all(field.status == NO_UNIT for field in corpus.truth_from_render(rendered).values())


@needs_pillow
def test_crop_that_takes_the_unit_column_turns_fields_into_refusals():
    """The crop case the stratum exists for.

    With the unit in its own column, a cut that removes it leaves a readable
    number of unknown scale -- so the correct outcome flips from emitting a
    quantity to refusing one. With the unit in the label, the same cut
    changes nothing. Both halves are asserted, because the contrast is the
    measurement.
    """
    from chemoguard.extraction_bench.render import render

    to_value = next(
        v for v in degrade.variants() if v.name == "crop_to_value"
    )

    column = render(
        PageSpec(
            panel=panels.PANELS["normal"],
            scale="conventional_cells",
            placement=COLUMN,
            decimal_style=POINT,
            reference_ranges=True,
        )
    )
    _image, keep = degrade.apply_to(column, to_value)
    after = corpus.truth_from_render(column, keep=keep)

    # Every analyte except one: "Neutrophils %" repeats its unit in the row
    # label, so that crop does not take it. See
    # test_a_crop_does_not_orphan_a_unit_that_is_also_in_the_label.
    flipped = {k: v.status for k, v in after.items() if k != "neutrophils_percent"}
    assert set(flipped.values()) == {NO_UNIT}
    assert all(f.value is not None for f in after.values())
    assert after["neutrophils_percent"].status == EXTRACTABLE

    labelled = render(
        PageSpec(
            panel=panels.PANELS["normal"],
            scale="conventional_cells",
            placement=LABEL,
            decimal_style=POINT,
            reference_ranges=True,
        )
    )
    _image, keep = degrade.apply_to(labelled, to_value)
    assert all(
        f.status == EXTRACTABLE
        for f in corpus.truth_from_render(labelled, keep=keep).values()
    )


@needs_pillow
def test_crop_that_takes_rows_marks_them_absent_not_refused():
    """A row that left the page is ``absent``: silence is correct, not refusal."""
    from chemoguard.extraction_bench.render import render

    rows_variant = next(v for v in degrade.variants() if v.name == "crop_last_rows")
    rendered = render(
        PageSpec(
            panel=panels.PANELS["normal"],
            scale="conventional_cells",
            placement=COLUMN,
            decimal_style=POINT,
            reference_ranges=True,
        )
    )
    _image, keep = degrade.apply_to(rendered, rows_variant)
    after = corpus.truth_from_render(rendered, keep=keep)

    absent = [k for k, v in after.items() if v.status == ABSENT]
    assert absent == [panels.ANALYTE_ORDER[-2], panels.ANALYTE_ORDER[-1]]


def test_variant_count_and_families():
    variants = degrade.variants()
    assert len(variants) == 26
    assert sum(1 for v in variants if v.family == degrade.PRISTINE) == 1
    assert sum(1 for v in variants if v.family == degrade.CROP) == 3


def test_notation_cross_is_fully_crossed_and_balanced():
    cells = corpus.notation_cells()
    assert len(cells) == 36
    assert len(set(cells)) == 36

    for index, levels in enumerate(
        (corpus.SCALE_LEVELS, corpus.PLACEMENTS, corpus.DECIMAL_STYLES, corpus.RANGE_LEVELS)
    ):
        counts = {level: 0 for level in levels}
        for cell in cells:
            counts[cell[index]] += 1
        assert len(set(counts.values())) == 1, counts


# --- ground-truth invariants -------------------------------------------------


def test_field_truth_rejects_an_extractable_field_with_no_unit():
    with pytest.raises(ValueError, match="extractable"):
        FieldTruth(
            analyte="wbc",
            status=EXTRACTABLE,
            printed_value="5800",
            value=5800.0,
        )


def test_field_truth_rejects_a_no_unit_field_that_records_a_unit():
    with pytest.raises(ValueError, match="no_unit"):
        FieldTruth(
            analyte="wbc",
            status=NO_UNIT,
            printed_value="5800",
            value=5800.0,
            printed_unit="cells/uL",
        )


def test_field_truth_rejects_an_absent_field_with_a_value():
    with pytest.raises(ValueError, match="absent"):
        FieldTruth(analyte="mchc", status=ABSENT, printed_value="33.3")


# --- stratum A hand labels ---------------------------------------------------


def test_stratum_a_canonical_values_agree_with_units_module():
    """Every hand-labelled canonical value, recomputed through ``units.py``.

    The labels were written by reading the images; this checks the arithmetic
    that turned "250 K/uL" into 250,000 cells/uL, using the production
    conversion rather than the one in ``panels.py``.
    """
    from units import Quantity

    for image in stratum_a():
        for truth in image.fields.values():
            if truth.status != EXTRACTABLE:
                continue
            expected = Quantity.of(truth.value, truth.unit).magnitude_in(
                panels.CANONICAL[truth.analyte]
            )
            assert truth.canonical_value == pytest.approx(expected), (
                image.image_id,
                truth.analyte,
            )


def test_stratum_a_values_are_physiologically_plausible():
    """A hand-label outside the bounds would score as a loss for every arm."""
    from cbc import quantity_for

    for image in stratum_a():
        for truth in image.fields.values():
            if truth.status != EXTRACTABLE:
                continue
            quantity_for(truth.analyte, truth.value, truth.unit)


def test_stratum_a_has_the_expected_shape():
    images = stratum_a()
    assert [i.sub for i in images] == ["A1", "A2", "A3", "A4"]

    counts = {"extractable": 0, "no_unit": 0, "absent": 0}
    for image in images:
        for status, count in image.counts().items():
            counts[status] += count

    # 29 readable fields, the 4 unit-less rows of test_3.jpg, and the 7 rows
    # those two short pages do not carry.
    assert counts == {"extractable": 29, "no_unit": 4, "absent": 7}


def test_stratum_a_records_the_mcl_spellings_by_what_they_denote():
    """``cells/mcL`` denotes ``cells/uL`` whether or not the alias table knows.

    Ground truth records meaning, so the same labels score a run before and
    after the alias table is extended. If these recorded "unparseable", the
    before/after comparison would be impossible.
    """
    a4 = next(image for image in stratum_a() if image.sub == "A4")

    assert a4.fields["wbc"].printed_unit == "cells/mcL"
    assert a4.fields["wbc"].unit == "cells/uL"
    assert a4.fields["platelets"].printed_unit == "/mcL"
    assert a4.fields["platelets"].unit == "cells/uL"
    assert a4.fields["rbc"].printed_unit == "million cells/mcL"
    assert a4.fields["rbc"].unit == "million/uL"


# --- the scorer -------------------------------------------------------------


class _FakeReading:
    """A backend reading, hand-built so the right score is obvious."""

    def __init__(self, raw=None, measured=None, skipped=None):
        self.backend = "test"
        self.raw = raw or {}
        self.skipped = skipped or {}
        self.panel = _FakePanel(measured or {})
        self.ocr_text = None
        self.confidence = None
        self.error = None


class _FakePanel:
    def __init__(self, measured):
        from units import Quantity

        self.measured = {
            key: Quantity.of(value, unit) for key, (value, unit) in measured.items()
        }

    def get(self, key):
        return self.measured.get(key)


def _image_with(truth):
    return ImageTruth(
        image_id="img",
        path="x.png",
        stratum="B",
        sub="cell",
        fields={truth.analyte: truth},
    )


_WBC_5800 = FieldTruth(
    analyte="wbc",
    status=EXTRACTABLE,
    printed_value="5800",
    printed_unit="cells/uL",
    unit_placement="column",
    value=5800.0,
    unit="cells/uL",
    canonical_value=5800.0,
)


def test_correct_reading_is_a_true_positive():
    reading = _FakeReading(
        raw={"wbc": {"value": 5800, "unit": "cells/uL"}},
        measured={"wbc": (5800, "cells/uL")},
    )
    result = score.score_field(reading, _image_with(_WBC_5800), _WBC_5800)

    assert (result.outcome, result.tp, result.fp, result.fn, result.tn) == (
        score.CORRECT, 1, 0, 0, 0,
    )
    assert result.silent is False


def test_a_correct_reading_in_a_different_but_equivalent_unit_is_still_correct():
    """5.8 thousand/uL is 5800 cells/uL. Scoring is on the quantity, not the string."""
    reading = _FakeReading(
        raw={"wbc": {"value": 5.8, "unit": "K/uL"}},
        measured={"wbc": (5.8, "K/uL")},
    )
    result = score.score_field(reading, _image_with(_WBC_5800), _WBC_5800)
    assert result.outcome == score.CORRECT


def test_a_wrong_value_that_reaches_the_panel_is_both_fp_and_fn_and_silent():
    """A tenfold error inside the plausibility bounds. The dangerous case.

    Both FP and FN, because the backend claimed something wrong *and* failed
    to produce the right answer, and ``silent`` because nothing refused it.
    """
    reading = _FakeReading(
        raw={"wbc": {"value": 58000, "unit": "cells/uL"}},
        measured={"wbc": (58000, "cells/uL")},
    )
    result = score.score_field(reading, _image_with(_WBC_5800), _WBC_5800)

    assert result.outcome == score.WRONG_VALUE
    assert (result.tp, result.fp, result.fn, result.tn) == (0, 1, 1, 0)
    assert result.silent is True
    assert result.log10_ratio == pytest.approx(1.0)
    assert result.stage == score.VALUE_WRONG


def test_a_refusal_of_a_readable_field_is_a_false_negative_not_a_false_positive():
    """Refusing claims nothing, so it cannot be a false positive.

    This is the asymmetry the whole metric turns on: in this application a
    refusal is a supported outcome, and a metric that charged it as a wrong
    answer would score the safe behaviour as failure.
    """
    reading = _FakeReading(
        raw={"wbc": {"value": 5800, "unit": None}},
        skipped={"wbc": {"reason": "MissingUnitError", "value": 5800, "unit": None}},
    )
    result = score.score_field(reading, _image_with(_WBC_5800), _WBC_5800)

    assert (result.tp, result.fp, result.fn, result.tn) == (0, 0, 1, 0)
    assert result.stage == score.UNIT_DROPPED


def test_a_field_lost_to_the_alias_table_is_charged_to_units_not_to_ocr(monkeypatch):
    """Read perfectly, lost at ``parse_unit``. The attribution that matters.

    A spelling the alias table does not know, transcribed correctly, is a
    defect in ``units.py`` -- charging it to OCR would point a fix at the
    wrong module.

    The spelling is injected through ``_REPAIRS`` rather than taken from the
    live alias table on purpose. Real gaps get fixed: ``cells/mcL`` was one
    when this benchmark first ran and is accepted now, so a test written
    against a real gap would go green for the wrong reason the moment the
    gap closed, and stop guarding the attribution logic at all.
    """
    import re

    monkeypatch.setattr(
        score,
        "_REPAIRS",
        ((re.compile(r"\bgigacells\s*/\s*L\b", re.I), "10^9/L"),),
    )

    truth = FieldTruth(
        analyte="wbc",
        status=EXTRACTABLE,
        printed_value="5.1",
        printed_unit="gigacells/L",
        unit_placement="value_cell",
        value=5.1,
        unit="thousand/uL",
        canonical_value=5100.0,
    )
    reading = _FakeReading(
        raw={"wbc": {"value": 5.1, "unit": "gigacells/L"}},
        skipped={"wbc": {"reason": "UnknownUnitError"}},
    )
    result = score.score_field(reading, _image_with(truth), truth)

    assert result.stage == score.ALIAS_GAP
    assert result.fn == 1 and result.fp == 0


def test_the_alias_gaps_this_benchmark_found_are_now_closed():
    """Regression guard on the alias-table change the first run motivated.

    These four spellings all appear on the checked-in report images and all
    failed ``parse_unit`` when the benchmark first ran. They are asserted
    here, not only in the probe, so that a future change to ``_normalise``
    cannot quietly reopen them.
    """
    from units import parse_unit

    assert parse_unit("cells/mcL").symbol == "cells/uL"
    assert parse_unit("/mcL").symbol == "cells/uL"
    assert parse_unit("million cells/mcL").symbol == "million/uL"
    assert parse_unit("cells/µL (Low)").symbol == "cells/uL"
    # And the SI exponent forms a report actually prints.
    assert parse_unit("10⁹/L").symbol == "thousand/uL"
    assert parse_unit("10¹²/L").symbol == "million/uL"
    assert parse_unit("×10⁹/L").symbol == "thousand/uL"
    assert parse_unit("10E9/L").symbol == "thousand/uL"


def test_giga_per_litre_is_still_refused_for_a_count_and_that_is_deliberate():
    """``G/L`` is left unregistered, and the reason is recorded in units.py.

    ``_normalise`` lower-cases, so "G/L" (a count) and "g/L" (a mass
    concentration) collapse to one key. Separating them means making the
    normaliser case-sensitive for one pair, which is a safety call with an
    argument each way and not one to settle inside a benchmark. What matters
    is that it fails safe today, which is what this pins.
    """
    from cbc import quantity_for
    from units import IncompatibleUnitsError

    with pytest.raises(IncompatibleUnitsError):
        quantity_for("wbc", 7.2, "G/L")


def test_a_misread_unit_is_charged_to_ocr_not_to_the_alias_table():
    """``K/uL`` read where ``cells/uL`` was printed is a transcription error."""
    reading = _FakeReading(
        raw={"wbc": {"value": 5800, "unit": "K/uL"}},
        measured={"wbc": (5800, "K/uL")},
    )
    result = score.score_field(reading, _image_with(_WBC_5800), _WBC_5800)

    assert result.stage == score.UNIT_WRONG
    assert result.outcome == score.WRONG_VALUE
    assert result.log10_ratio == pytest.approx(3.0)


_PLT_NO_UNIT = FieldTruth(
    analyte="platelets",
    status=NO_UNIT,
    printed_value="250,000",
    value=250000.0,
)


def test_refusing_a_unitless_row_is_a_true_negative():
    reading = _FakeReading(
        raw={"platelets": {"value": 250000, "unit": None}},
        skipped={"platelets": {"reason": "MissingUnitError"}},
    )
    result = score.score_field(reading, _image_with(_PLT_NO_UNIT), _PLT_NO_UNIT)

    assert (result.tp, result.fp, result.fn, result.tn) == (0, 0, 0, 1)
    assert result.outcome == score.CORRECT_REFUSAL


def test_silence_on_a_unitless_row_is_correct_but_recorded_separately():
    """Both are true negatives; only one tells the operator anything.

    They are not equally useful. An operator cannot tell "this report has no
    platelet count" from "the platelet count is there and I could not read
    its unit", and only the second is something they can act on.
    """
    result = score.score_field(
        _FakeReading(), _image_with(_PLT_NO_UNIT), _PLT_NO_UNIT
    )

    assert result.tn == 1
    assert result.outcome == score.CORRECT_SILENCE


def test_supplying_a_unit_the_page_never_printed_is_a_false_positive():
    """250,000 happens to be cells/uL here, and guessing it is still wrong.

    The guess would be equally confident, and equally wrong, on a platelet
    count of 900 -- which is the whole argument of CLAUDE.md Finding 4. So
    this scores as a false positive even though the number is right.
    """
    reading = _FakeReading(
        raw={"platelets": {"value": 250000, "unit": "cells/uL"}},
        measured={"platelets": (250000, "cells/uL")},
    )
    result = score.score_field(reading, _image_with(_PLT_NO_UNIT), _PLT_NO_UNIT)

    assert (result.tp, result.fp, result.fn, result.tn) == (0, 1, 0, 0)
    assert result.outcome == score.INFERRED_UNIT
    assert result.silent is True


def test_inventing_a_value_for_an_analyte_not_on_the_page_is_a_false_positive():
    truth = FieldTruth(analyte="mchc", status=ABSENT)
    reading = _FakeReading(
        raw={"mchc": {"value": 33.3, "unit": "g/dL"}},
        measured={"mchc": (33.3, "g/dL")},
    )
    result = score.score_field(reading, _image_with(truth), truth)

    assert result.outcome == score.HALLUCINATED
    assert (result.fp, result.tn) == (1, 0)


def test_silence_on_an_absent_analyte_is_a_true_negative():
    truth = FieldTruth(analyte="mchc", status=ABSENT)
    result = score.score_field(_FakeReading(), _image_with(truth), truth)

    assert result.outcome == score.CORRECT_SILENCE
    assert result.tn == 1


# --- severity direction ------------------------------------------------------


def test_understated_severity_is_detected_and_directional():
    """A tenfold overstatement of an ANC of 200 reads as grade 0.

    CLAUDE.md Finding 4: every error the old magnitude heuristic made pointed
    this way, because a dangerously low count in cells/uL looks like a
    comfortable count in thousands/uL. This is the assertion that the
    detector for that direction works.
    """
    truth = FieldTruth(
        analyte="anc",
        status=EXTRACTABLE,
        printed_value="0,2",
        printed_unit="K/uL",
        unit_placement="column",
        value=0.2,
        unit="thousand/uL",
        canonical_value=200.0,
    )
    reading = _FakeReading(
        raw={"anc": {"value": 2.0, "unit": "K/uL"}},
        measured={"anc": (2.0, "K/uL")},
    )
    result = score.score_field(reading, _image_with(truth), truth)

    assert result.grade_true == 4
    assert result.grade_read == 0
    assert result.understated_severity is True
    assert result.grade_changed is True


def test_overstated_severity_is_not_counted_as_understated():
    """The other direction causes an unnecessary look, not a missed cytopenia."""
    truth = FieldTruth(
        analyte="anc",
        status=EXTRACTABLE,
        printed_value="3.900",
        printed_unit="cells/uL",
        unit_placement="column",
        value=3900.0,
        unit="cells/uL",
        canonical_value=3900.0,
    )
    reading = _FakeReading(
        raw={"anc": {"value": 3.9, "unit": "cells/uL"}},
        measured={"anc": (3.9, "cells/uL")},
    )
    result = score.score_field(reading, _image_with(truth), truth)

    assert result.grade_true == 0
    assert result.grade_read == 4
    assert result.understated_severity is False
    assert result.grade_changed is True


# --- aggregation -------------------------------------------------------------


def test_precision_and_recall_are_none_when_undefined_not_zero():
    """A backend that claimed nothing has undefined precision, not 0.0.

    Substituting a number there would put a fiction into a table that reads
    as a measurement.
    """
    truth = FieldTruth(analyte="mchc", status=ABSENT)
    scores = [score.score_field(_FakeReading(), _image_with(truth), truth)]
    stats = score.aggregate(scores, "analyte")[("mchc",)]

    assert stats["precision"] is None
    assert stats["recall"] is None
    assert stats["tn"] == 1


def test_aggregate_sums_the_confusion_counts():
    correct = _FakeReading(
        raw={"wbc": {"value": 5800, "unit": "cells/uL"}},
        measured={"wbc": (5800, "cells/uL")},
    )
    wrong = _FakeReading(
        raw={"wbc": {"value": 58000, "unit": "cells/uL"}},
        measured={"wbc": (58000, "cells/uL")},
    )
    image = _image_with(_WBC_5800)
    scores = [
        score.score_field(correct, image, _WBC_5800),
        score.score_field(correct, image, _WBC_5800),
        score.score_field(wrong, image, _WBC_5800),
    ]
    stats = score.aggregate(scores, "analyte")[("wbc",)]

    assert (stats["tp"], stats["fp"], stats["fn"]) == (2, 1, 1)
    assert stats["precision"] == pytest.approx(2 / 3)
    assert stats["recall"] == pytest.approx(2 / 3)


# --- unit repair and the alias probe ----------------------------------------


@pytest.mark.parametrize(
    "printed,expected",
    [
        ("cells/µL (Low)", "cells/uL"),
        ("cells/mcL", "cells/uL"),
        ("/mcL", "cells/uL"),
        ("million cells/mcL", "million/uL"),
        ("cells/mc L", "cells/uL"),
    ],
)
def test_these_spellings_resolve_to_the_unit_they_denote(printed, expected):
    """Every spelling the benchmark found must land on the right unit.

    Deliberately agnostic about the route. Before the alias-table change
    these resolved only via ``score.repair_unit``; now ``parse_unit`` accepts
    them directly. Asserting the route would make this test fail every time
    the alias table improved, which is backwards -- what has to hold is the
    meaning.
    """
    symbol, how = score.denote(printed)
    assert symbol == expected
    assert how in ("direct", "repaired")


@pytest.mark.parametrize(
    "printed,expected",
    [
        ("cells/µL (Low)", "cells/µL"),
        ("cells/mcL", "cells/uL"),
        ("/mcL", "/uL"),
        ("million cells/mcL", "million/uL"),
    ],
)
def test_repair_unit_rewrites_the_spelling_without_changing_meaning(printed, expected):
    """The repair rewriter itself, independent of the alias table.

    ``repair_unit`` is what lets an alias gap be told apart from a
    transcription error, so it has to keep working even once every spelling
    it knows about is accepted natively -- otherwise the *next* gap gets
    misattributed to OCR.
    """
    assert score.repair_unit(printed) == expected


def test_repairs_do_not_rescue_a_genuinely_unknown_unit():
    assert score.denote("furlongs per fortnight") == (None, None)
    assert score.denote(None) == (None, None)


def test_alias_probe_expectations_are_internally_consistent():
    """Anything the probe marks ``accepted`` must resolve to what it denotes."""
    for result in alias_probe.run():
        if result.status == alias_probe.ACCEPTED and result.probe.expected:
            assert result.resolved == result.probe.expected, result.probe.text
            assert not result.via_repair, result.probe.text


# --- the deferred GPT arm ----------------------------------------------------


def test_gpt_arm_refuses_to_call_the_api_without_both_switches(tmp_path, monkeypatch):
    """Two independent switches, because a stray run spends real money.

    The arm is deferred by decision. The adapter is complete so that enabling
    it is a flag rather than a rewrite, and hard-guarded so that enabling it
    cannot happen by accident.
    """
    monkeypatch.delenv(GPTBackend.ENV_FLAG, raising=False)

    image = ImageTruth(image_id="i", path="A/test1.jpg", stratum="A", sub="A1")

    assert GPTBackend(tmp_path, allow_network=False).available() is False
    assert GPTBackend(tmp_path, allow_network=True).available() is False

    monkeypatch.setenv(GPTBackend.ENV_FLAG, "1")
    assert GPTBackend(tmp_path, allow_network=False).available() is False
    assert GPTBackend(tmp_path, allow_network=True).available() is True

    monkeypatch.delenv(GPTBackend.ENV_FLAG, raising=False)
    with pytest.raises(NetworkDisabledError, match="deferred"):
        GPTBackend(tmp_path, allow_network=True).read(image, "data/extraction_corpus")


# --- the Tesseract adapter tracks production ---------------------------------


def test_tesseract_adapter_calls_the_same_steps_as_process_file():
    """The adapter reimplements ``process_file`` to keep the raw reading.

    That is a duplication, and duplications drift. This pins the three calls
    the adapter makes to the three ``process_file`` makes, so a change to the
    production path fails here instead of quietly making the benchmark
    measure something the application no longer does.
    """
    import inspect

    from ocr_extraction import HematologyExtractor

    source = inspect.getsource(HematologyExtractor.process_file)
    for call in (
        "extract_text_from_image",
        "extract_parameters",
        "panel_from_extraction",
    ):
        assert call in source, call


@needs_pillow
def test_a_crop_does_not_orphan_a_unit_that_is_also_in_the_label():
    """"Neutrophils %" beside a units column reading "%" prints it twice.

    A crop that takes the column therefore does not take the unit, and the
    field stays extractable. ``test1.jpg`` prints exactly this duplication,
    so it is a real layout rather than a contrivance.

    Regression test for a ground-truth defect, not a backend one. The first
    OCR run reported nine fields where the backend appeared to invent a unit
    under crop -- which reads as a dangerous hallucination. The backend had
    read the unit out of the label, correctly, and the truth derivation was
    what was wrong.
    """
    from chemoguard.extraction_bench.render import render

    to_value = next(v for v in degrade.variants() if v.name == "crop_to_value")
    rendered = render(
        PageSpec(
            panel=panels.PANELS["normal"],
            scale="conventional_cells",
            placement=COLUMN,
            decimal_style=POINT,
            reference_ranges=True,
        )
    )
    row = next(r for r in rendered.rows if r.analyte == "neutrophils_percent")
    assert row.printed_unit in row.label.text, "precondition: the label repeats it"

    _image, keep = degrade.apply_to(rendered, to_value)
    after = corpus.truth_from_render(rendered, keep=keep)

    assert after["neutrophils_percent"].status == EXTRACTABLE
    assert after["neutrophils_percent"].printed_unit == "%"
    # Every other analyte on this layout does lose its unit to that crop.
    assert after["wbc"].status == NO_UNIT
