"""The three checked-in report images, through the real extraction path.

test_2.jpg is the case that matters: one page carrying RBC in millions/uL
alongside WBC and platelets in cells/uL, with platelets at 85,000 -- a real
thrombocytopenia that reads as a comfortable count to anything inferring
unit from magnitude. Extracting it correctly means reading three different
units off one table and keeping each attached to its own number.

The parser runs against transcriptions in tests/fixtures/ so the suite does
not require the Tesseract binary; the last test in this file runs the real
binary when it is present and skips when it is not.
"""

from pathlib import Path

import pytest

from cbc import panel_from_extraction
from clinical_flags import flag_for
from model_contract import features_for

FIXTURES = Path(__file__).parent / "fixtures"
IMAGES = Path(__file__).parent.parent


def read_fixture(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def parse(name):
    """Run the production OCR pattern set over a transcription."""
    from ocr_extraction import HematologyExtractor

    raw = HematologyExtractor().extract_parameters(read_fixture(name))
    return panel_from_extraction(raw)


# --- test_2.jpg: the mixed-unit thrombocytopenia case -----------------------

def test_test_2_reads_three_different_units_off_one_page():
    panel, skipped = parse("test_2.txt")

    assert skipped == {}
    assert panel.get("rbc").unit.symbol == "million/uL"
    assert panel.get("wbc").unit.symbol == "cells/uL"
    assert panel.get("platelets").unit.symbol == "cells/uL"


@pytest.mark.parametrize(
    "analyte,expected,unit",
    [
        ("rbc", 4.40, "million/uL"),
        ("wbc", 5800, "cells/uL"),
        ("platelets", 85_000, "cells/uL"),
        ("anc", 3500, "cells/uL"),
        ("neutrophils_percent", 60, "%"),
        ("hb", 13.0, "g/dL"),
        ("hematocrit", 39.0, "%"),
        ("mcv", 83.5, "fL"),
        ("mch", 28.0, "pg"),
        ("mchc", 33.2, "g/dL"),
    ],
)
def test_test_2_values_survive_extraction(analyte, expected, unit):
    panel, _ = parse("test_2.txt")

    assert panel.get(analyte).magnitude_in(unit) == pytest.approx(expected)


def test_test_2_platelets_are_thrombocytopenic_not_normal():
    """85,000 cells/uL is 85 thousand/uL: CTCAE grade 1, and abnormal.

    Read as 85,000 thousand/uL -- the magnitude inference -- it is eleven
    times the upper limit of normal and flagged as nothing at all.
    """
    panel, _ = parse("test_2.txt")
    flag = flag_for("platelets", panel.get("platelets"))

    assert flag.compared_value == 85.0
    assert flag.compared_unit == "thousand/uL"
    assert flag.grade == 1
    assert flag.condition == "thrombocytopenia"


def test_test_2_reaches_the_model_in_the_declared_units():
    panel, _ = parse("test_2.txt")
    row, record = features_for("thrombocytopenia", panel)

    # RBC in million/uL, WBC and platelets in thousand/uL, per the contract.
    assert row == pytest.approx([4.40, 5.8, 85.0])
    assert [f["fed_to_model"]["unit"] for f in record] == [
        "million/uL", "thousand/uL", "thousand/uL",
    ]


def test_test_2_anc_is_read_not_derived():
    """The page prints ANC 3500; WBC x 60% would give 3480."""
    panel, _ = parse("test_2.txt")
    filled = panel.with_derived_anc()

    assert filled.get("anc").magnitude_in("cells/uL") == 3500
    assert filled.unit_provenance["anc"] == "report"


# --- test1.jpg: uniform K/uL and M/uL ---------------------------------------

@pytest.mark.parametrize(
    "analyte,expected,unit",
    [
        ("hb", 14.5, "g/dL"),
        ("rbc", 5.1, "million/uL"),
        ("wbc", 7.2, "thousand/uL"),
        ("anc", 4.1, "thousand/uL"),
        ("neutrophils_percent", 57.0, "%"),
        ("platelets", 250, "thousand/uL"),
        ("hematocrit", 43.5, "%"),
        ("mcv", 85.3, "fL"),
        ("mch", 28.4, "pg"),
        ("mchc", 33.3, "g/dL"),
    ],
)
def test_test1_values_survive_extraction(analyte, expected, unit):
    panel, skipped = parse("test1.txt")

    assert skipped == {}
    assert panel.get(analyte).magnitude_in(unit) == pytest.approx(expected)


def test_test1_k_per_ul_becomes_cells_per_ul_canonically():
    """K/uL is thousand/uL. 7.2 K/uL is 7,200 cells, not 7.2."""
    panel, _ = parse("test1.txt")

    assert panel.get("wbc").magnitude_in("cells/uL") == pytest.approx(7200)
    assert panel.get("platelets").magnitude_in("cells/uL") == pytest.approx(250_000)


def test_test1_is_a_normal_panel():
    from clinical_flags import severe_flags

    panel, _ = parse("test1.txt")

    assert severe_flags(panel) == []


# --- test_3.jpg: a unit that is only in the reference-range column ----------

def test_test_3_reads_the_units_that_are_printed():
    panel, skipped = parse("test_3.txt")

    assert panel.get("rbc").magnitude_in("million/uL") == pytest.approx(4.5)
    assert panel.get("wbc").magnitude_in("cells/uL") == pytest.approx(6000)
    assert panel.get("anc").magnitude_in("cells/uL") == pytest.approx(2500)


def test_test_3_platelets_have_no_unit_on_their_row_and_are_refused():
    """The row reads "Platelets  250,000" with the unit only in the range.

    Refusing is correct. 250,000 happens to be cells/uL here, and guessing
    that from the magnitude is the behaviour being removed -- it would be
    equally confident, and equally wrong, on a platelet count of 900.
    """
    panel, skipped = parse("test_3.txt")

    assert panel.get("platelets") is None
    assert skipped["platelets"]["reason"] == "MissingUnitError"
    assert skipped["platelets"]["value"] == 250_000


def test_test_3_reports_every_unitless_row_rather_than_dropping_it():
    """Four rows on this page print a number and no unit.

    Each is reported as unusable. Silence would be worse than a refusal: the
    operator could not tell "the report has no platelet count" from "the
    platelet count is there and I could not read its unit", and only the
    second is something they can fix.
    """
    _panel, skipped = parse("test_3.txt")

    assert set(skipped) == {"platelets", "hematocrit", "mcv", "mch"}
    assert all(entry["unit"] is None for entry in skipped.values())


def test_test_3_reads_a_unit_printed_in_the_row_label():
    """"Neutrophils %  55" states its unit to the left of the number."""
    panel, _ = parse("test_3.txt")

    assert panel.get("neutrophils_percent").magnitude_in("%") == 55


def test_a_panel_missing_platelets_cannot_be_predicted_from():
    panel, _ = parse("test_3.txt")

    assert "platelets" in panel.missing_required()


# --- the real Tesseract path ------------------------------------------------

def _tesseract_available():
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _tesseract_available(), reason="Tesseract binary not installed")
@pytest.mark.parametrize("image", ["test1.jpg", "test_2.jpg", "test_3.jpg"])
def test_real_ocr_never_returns_a_value_without_a_unit(image):
    """Whatever Tesseract reads, nothing unitless escapes the extractor.

    Deliberately not an assertion about OCR accuracy -- that is the
    extraction benchmark in CLAUDE.md under Direction, and a different job.
    This asserts only the boundary property: every Quantity in the returned
    panel has a unit, and everything else was skipped.
    """
    from ocr_extraction import HematologyExtractor

    panel, _text, _confidence, skipped = HematologyExtractor().process_file(
        str(IMAGES / image), image.rsplit(".", 1)[1]
    )

    for key, quantity in panel.measured.items():
        assert quantity.unit is not None, key
    for key, entry in skipped.items():
        assert entry["reason"] in {
            "MissingUnitError", "UnknownUnitError",
            "IncompatibleUnitsError", "ImplausibleValueError",
        }
