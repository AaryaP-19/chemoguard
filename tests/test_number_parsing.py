"""Separator roles in a printed number, and the page-level pass that settles them.

``1,234`` is one thousand two hundred and thirty-four on a US report and one
point two three four on a European one. The code this tests replaces resolved
that by stripping commas and honouring points -- the US convention, hard-coded
-- which read ``6.500`` as 6.5 and ``0,9`` as 9. CLAUDE.md Finding 9 / M1
measured the consequence: 32 silent wrong values across the extraction
benchmark, 16 of which graded a cytopenia less severe than it was, including a
grade 4 leukopenia printed ``0,8`` that was read as 8,000 cells/uL and graded
normal.

The rule rests on one fact: **a thousands separator is always followed by
exactly three digits.** So a separator followed by one, two, or four or more
digits can only be a decimal mark, and only the exactly-three case is
genuinely undecidable. That case is then settled by the page rather than by a
convention: a report that prints ``13.0`` anywhere has shown that ``.`` is its
decimal mark, and therefore that ``,`` groups its thousands.

Where the page settles nothing, the number is refused -- the same stance this
codebase already takes on a missing unit, and for the same reason.
"""

from __future__ import annotations

import pytest

from cbc import panel_from_extraction
from ocr_extraction import (
    AmbiguousNumber,
    HematologyExtractor,
    infer_decimal_mark,
    parse_number,
)


# --- separator followed by something other than three digits ----------------


@pytest.mark.parametrize(
    "token,expected",
    [
        ("14.5", 14.5),
        ("14,5", 14.5),
        ("4.80", 4.80),
        ("4,80", 4.80),
        ("0.9", 0.9),
        ("0,9", 0.9),
        ("2,0", 2.0),
        ("1234,5", 1234.5),
        ("1234.5", 1234.5),
        ("33,25", 33.25),
    ],
)
def test_one_two_or_four_plus_digits_after_a_separator_is_a_decimal_mark(
    token, expected
):
    """No grouping convention puts a non-three-digit run after a separator.

    So these resolve identically on a US and a European page, and need no
    page context at all. The four understating cases in Finding 9 / M1 --
    0,9 / 0,8 / 0,2 / 2,0 -- all live here, which is why the rule fixes them
    deterministically rather than probabilistically.
    """
    assert parse_number(token) == pytest.approx(expected)


@pytest.mark.parametrize("token,expected", [("6500", 6500.0), ("85000", 85000.0)])
def test_a_number_with_no_separator_needs_no_rule(token, expected):
    assert parse_number(token) == pytest.approx(expected)


# --- separator followed by exactly three digits ------------------------------


@pytest.mark.parametrize("token", ["6,500", "6.500", "250,000", "250.000", "3.900"])
def test_exactly_three_digits_after_a_separator_is_undecidable_alone(token):
    """The only genuinely ambiguous shape, and it raises rather than guesses."""
    with pytest.raises(AmbiguousNumber, match="undecidable"):
        parse_number(token)


@pytest.mark.parametrize(
    "token,mark,expected",
    [
        ("6,500", ".", 6500.0),      # ',' groups thousands
        ("250,000", ".", 250000.0),
        ("6.500", ".", 6.5),         # '.' is the decimal mark
        ("3.900", ".", 3.9),
        ("6,500", ",", 6.5),         # mirror image on a European page
        ("6.500", ",", 6500.0),
        ("3.900", ",", 3900.0),
    ],
)
def test_a_known_page_convention_resolves_the_ambiguous_shape(token, mark, expected):
    assert parse_number(token, mark) == pytest.approx(expected)


# --- more than one separator -------------------------------------------------


@pytest.mark.parametrize(
    "token,expected",
    [
        ("1.234,56", 1234.56),   # '.' groups, ',' is the decimal mark
        ("1,234.56", 1234.56),   # and the mirror image
    ],
)
def test_two_different_separators_settle_each_other(token, expected):
    """The last is the decimal mark, because the other is doing the grouping."""
    assert parse_number(token) == pytest.approx(expected)


@pytest.mark.parametrize("token,expected", [("1.234.567", 1234567.0),
                                            ("1,234,567", 1234567.0)])
def test_one_separator_repeated_can_only_be_grouping(token, expected):
    """A number has one decimal mark, so a repeated separator must group.

    Unambiguous with no page context, even though every run after it is
    three digits long -- the exactly-three rule does not apply here, because
    the repetition already rules the decimal reading out.
    """
    assert parse_number(token) == pytest.approx(expected)


@pytest.mark.parametrize("token", ["1.234.56", "12.34.567"])
def test_a_shape_no_convention_produces_is_refused(token):
    with pytest.raises(AmbiguousNumber):
        parse_number(token)


@pytest.mark.parametrize("token", ["", ".", ",", "..", "abc"])
def test_a_token_that_is_not_a_number_is_refused(token):
    with pytest.raises(AmbiguousNumber):
        parse_number(token)


# --- the document-level pass -------------------------------------------------


def test_a_page_that_prints_one_unambiguous_decimal_settles_the_rest():
    """This is the second pass, and it is what recovers 6,500 without guessing.

    The evidence comes from the document. Read row by row, "6,500" is
    undecidable even on a page that prints "13.0" three lines above it, which
    is why the inference has to run over the whole page before any row is
    parsed.
    """
    assert infer_decimal_mark("Hb 13.0 g/dL   WBC 6,500 cells/uL") == "."
    assert infer_decimal_mark("Hb 13,0 g/dL   WBC 6.500 cells/uL") == ","


def test_a_page_with_no_unambiguous_number_settles_nothing():
    assert infer_decimal_mark("WBC 6,500 cells/uL  PLT 250,000 cells/uL") is None


def test_a_page_with_contradictory_evidence_settles_nothing():
    """Contradiction yields None, not a majority vote.

    A page using both marks as decimals is a page we cannot read
    consistently, and the right answer there is to refuse its ambiguous
    numbers rather than to pick the commoner convention and be silently
    wrong on the rest.
    """
    assert infer_decimal_mark("Hb 13.0 g/dL   HCT 39,5 %") is None


def test_a_page_with_no_numbers_at_all_settles_nothing():
    assert infer_decimal_mark("HEMATOLOGY REPORT") is None


# --- end to end through the extractor ---------------------------------------


def _panel(text):
    return panel_from_extraction(HematologyExtractor().extract_parameters(text))


_EUROPEAN_PAGE = """LABORATORIUM HEMATOLOGIE
Hemoglobin (Hb)             7,2    g/dL
White Blood Cells (WBC)     0,8    K/uL
Absolute Neutrophil Count   0,2    K/uL
Platelets                   0,9    K/uL
"""


def test_the_european_page_that_used_to_grade_a_grade_4_leukopenia_as_normal():
    """Finding 9 / M1's worst case, asserted end to end.

    WBC printed ``0,8`` is 800 cells/uL -- CTCAE grade 4 leukopenia. The old
    parse stripped the comma and read 8,000 cells/uL, which grades 0. Every
    error of that shape pointed the same way, toward normal, which is the
    signature of CLAUDE.md Finding 4 by a different mechanism.
    """
    from clinical_flags import flag_for

    panel, skipped = _panel(_EUROPEAN_PAGE)

    assert skipped == {}
    assert panel.get("wbc").magnitude_in("cells/uL") == pytest.approx(800)
    assert panel.get("anc").magnitude_in("cells/uL") == pytest.approx(200)
    assert panel.get("platelets").magnitude_in("cells/uL") == pytest.approx(900)

    assert flag_for("wbc", panel.get("wbc")).grade == 4
    assert flag_for("anc", panel.get("anc")).grade == 4


def test_a_european_page_with_grouped_thousands_resolves_from_its_own_decimals():
    """``6.500`` is six thousand five hundred here, and the page proves it."""
    panel, skipped = _panel(
        "Hemoglobin (Hb)  13,0  g/dL\n"
        "White Blood Cells (WBC)  6.500  cells/uL\n"
        "Platelets  250.000  cells/uL\n"
    )

    assert skipped == {}
    assert panel.get("wbc").magnitude_in("cells/uL") == pytest.approx(6500)
    assert panel.get("platelets").magnitude_in("cells/uL") == pytest.approx(250000)


def test_a_us_page_with_grouped_thousands_resolves_the_same_way():
    panel, skipped = _panel(
        "Hemoglobin (Hb)  13.0  g/dL\n"
        "White Blood Cells (WBC)  6,500  cells/uL\n"
        "Platelets  250,000  cells/uL\n"
    )

    assert skipped == {}
    assert panel.get("wbc").magnitude_in("cells/uL") == pytest.approx(6500)
    assert panel.get("platelets").magnitude_in("cells/uL") == pytest.approx(250000)


def test_an_unresolvable_number_is_refused_legibly_not_dropped():
    """A refusal the operator can act on, not silence.

    The distinction matters: silence leaves them unable to tell "this report
    has no platelet count" from "the count is printed and I could not tell
    250,000 from 250.000", and only the second is something they can fix.
    The printed token travels with the refusal so they can see what it was.
    """
    panel, skipped = _panel(
        "White Blood Cells (WBC)  6,500  cells/uL\n"
        "Platelets  250,000  cells/uL\n"
    )

    assert panel.get("wbc") is None
    assert panel.get("platelets") is None
    assert skipped["wbc"]["reason"] == "AmbiguousNumberError"
    assert skipped["wbc"]["printed"] == "6,500"
    assert skipped["platelets"]["printed"] == "250,000"


def test_the_checked_in_fixtures_are_unaffected():
    """The three US-convention fixtures must read exactly as they did.

    Each carries an unambiguous decimal -- 14.5, 4.40, 14.5 -- so the page
    pass settles the comma as a thousands separator and nothing changes.
    """
    from pathlib import Path

    fixtures = Path(__file__).parent / "fixtures"

    panel, skipped = _panel((fixtures / "test1.txt").read_text(encoding="utf-8"))
    assert skipped == {}
    assert panel.get("platelets").magnitude_in("cells/uL") == pytest.approx(250_000)

    panel, skipped = _panel((fixtures / "test_3.txt").read_text(encoding="utf-8"))
    assert panel.get("wbc").magnitude_in("cells/uL") == pytest.approx(6_000)
    assert panel.get("anc").magnitude_in("cells/uL") == pytest.approx(2_500)
    # Still refused, and still for the right reason: no unit on the row.
    assert skipped["platelets"]["reason"] == "MissingUnitError"


# --- M2: the SI unit alternation ---------------------------------------------


@pytest.mark.parametrize(
    "line,analyte,value,unit",
    [
        ("White Blood Cells (WBC)  6.5  10^9/L", "wbc", 6500, "cells/uL"),
        ("Red Blood Cells (RBC)  4.80  10^12/L", "rbc", 4.80, "million/uL"),
        ("Platelets  250  10⁹/L", "platelets", 250_000, "cells/uL"),
        ("White Blood Cells (WBC)  6.5  10e9/L", "wbc", 6500, "cells/uL"),
        ("Red Blood Cells (RBC)  4.80  10¹²/L", "rbc", 4.80, "million/uL"),
    ],
)
def test_si_count_units_are_read(line, analyte, value, unit):
    """CLAUDE.md Finding 9 / M2.

    ``_U_COUNT`` interpolated one character class inside another, giving
    ``[l[uµμ]]+`` -- which Python parses as a class plus a *required literal*
    ``]``. So the alternative matched ``10^9/L]`` and never ``10^9/L``, and
    every count reported in SI lost its unit. A second defect in the same
    expression made the exponent a single digit, so ``10^12/L`` could not
    have matched even with the brackets fixed.
    """
    panel, skipped = _panel(line)

    assert skipped == {}
    assert panel.get(analyte).magnitude_in(unit) == pytest.approx(value)


def test_the_bracket_bug_artefact_no_longer_matches():
    """``10^9/L]`` was the only SI form the old pattern accepted."""
    import re

    from ocr_extraction import _SI_COUNT

    assert re.fullmatch(_SI_COUNT, "10^9/L", re.I)
    assert not re.fullmatch(_SI_COUNT, "10^9/L]", re.I)


@pytest.mark.parametrize("spelling", ["109/L", "1012/L"])
def test_an_exponent_marker_is_required(spelling):
    """``10\\^?`` made the caret optional, so ``109/L`` matched as an SI unit.

    Requiring a marker costs one case: OCR that flattens a superscript to a
    plain digit turns "10⁹/L" into "109/L", which is now a refusal. That is
    the right way round -- 109 is a perfectly possible printed value, so the
    match was never safe, and a refusal is recoverable where a wrong scale is
    not.
    """
    import re

    from ocr_extraction import _SI_COUNT

    assert not re.fullmatch(_SI_COUNT, spelling, re.I)


# --- the micro sign as Tesseract reads it (CLAUDE.md Finding 9) -------------


@pytest.mark.parametrize(
    "line,analyte,value,unit",
    [
        ("Absolute Neutrophil Count (ANC) 3900 cells/pL", "anc", 3900, "cells/uL"),
        ("White Blood Cells (WBC) 6.5 K/pL", "wbc", 6500, "cells/uL"),
        ("Red Blood Cells (RBC) 4.80 millions/pL", "rbc", 4.80, "million/uL"),
        ("Platelets 250 K/pL", "platelets", 250_000, "cells/uL"),
    ],
)
def test_a_microlitre_denominator_misread_as_pl_is_recovered(
    line, analyte, value, unit
):
    """Tesseract reads the micro sign as ``p``, reliably enough to name.

    A pristine render at OCR confidence 91 returns ``cells/pL`` where the
    page prints ``cells/µL``. That one glyph dominated the lost yield in the
    OCR arm: every count analyte prints a microlitre denominator and no other
    analyte does, which is why the counts scored 0.221-0.305 recall while
    everything else scored 0.670-0.779.

    Accepting it costs no ambiguity. A picolitre is a millionth of a
    microlitre, so no laboratory reports a cell count per picolitre and there
    is nothing for ``cells/pL`` to be confused with.
    """
    panel, skipped = _panel(line)

    assert skipped == {}
    assert panel.get(analyte).magnitude_in(unit) == pytest.approx(value)


@pytest.mark.parametrize(
    "line,analyte,value,unit",
    [
        ("MCH 29.5 pg", "mch", 29.5, "pg"),
        ("MCV 88.0 fL", "mcv", 88.0, "fL"),
    ],
)
def test_the_pl_repair_does_not_touch_picograms(line, analyte, value, unit):
    """``pg`` is MCH's real unit, and this is a live collision not a theory.

    A blanket ``p``-for-``u`` rewrite would turn picograms into micrograms
    and corrupt every MCH on every report. The repair is scoped to a
    denominator following a slash for exactly that reason.
    """
    panel, skipped = _panel(line)

    assert skipped == {}
    assert panel.get(analyte).magnitude_in(unit) == pytest.approx(value)


def test_the_repair_happens_at_the_ocr_boundary_not_in_the_alias_table():
    """``cells/pL`` must not become a unit the rest of the app accepts.

    ``units.py`` holds spellings a laboratory would print. A corruption
    admitted there would be accepted from the keyboard and from the REST API
    too, where there is no OCR to blame for it -- so the rewrite lives in
    ``ocr_extraction`` and ``parse_unit`` still refuses the raw form.
    """
    from units import UnknownUnitError, parse_unit

    from ocr_extraction import repair_ocr_unit

    with pytest.raises(UnknownUnitError):
        parse_unit("cells/pL")

    assert parse_unit(repair_ocr_unit("cells/pL")).symbol == "cells/uL"
