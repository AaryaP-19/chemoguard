"""Stratum A: the four real report images, hand-labelled field by field.

Labelled by reading the images, and cross-checked against the transcriptions
in ``tests/fixtures/``. Every field records the unit as printed, not just the
number -- the unit is half of what an extraction has to get right, and a
benchmark that only scored numbers would call a 250,000 read as
"250,000 thousand/uL" a success.

**The stratum is not homogeneous, and the label "clean digital table"
describes its rendering, not its notation.** Two of the four print units in a
dedicated column; the other two embed the unit in the value cell, one of them
using ``mcL`` for microlitre. Four rows of ``test_3.jpg`` print no unit at
all, and its MCHC row is below the bottom edge of the image. Sub-labels
A1..A4 are recorded per image so any pooled number can be broken back down,
because pooling these four under a claim of "units in a dedicated third
column" would describe something that is not there.

Three of the spellings here are not in the ``units.py`` alias table:
``million cells/mcL``, ``cells/mcL`` and ``/mcL``. They are labelled with the
unit they *denote* -- mcL is microlitre, so they denote ``cells/uL`` -- which
is what lets the scorer separate "read the page wrong" from "read the page
right and the alias table did not know the spelling". See ``score.py``.

No patient identifiers, consistent with ``tests/fixtures/README.md``: no
name, MRN, date of birth or accession number on any of the four. The
screenshot carries ``Patient ID: DEMO-001`` and prints "Synthetic report
generated for academic use only" on its face.
"""

from __future__ import annotations

from decimal import Decimal

from chemoguard.extraction_bench.panels import CANONICAL, convert
from chemoguard.extraction_bench.truth import (
    ABSENT,
    EXTRACTABLE,
    NO_UNIT,
    FieldTruth,
    ImageTruth,
)

MU = "µ"   # MICRO SIGN, as printed on all four pages

#: The ten analytes, so a missing row is recorded as ABSENT rather than
#: omitted. An omitted row cannot be distinguished from one the scorer forgot,
#: and it is the ABSENT entries that catch a backend inventing a value for an
#: analyte the page does not carry.
ALL_ANALYTES = tuple(CANONICAL)


def _extractable(analyte, printed_value, value, printed_unit, unit, placement):
    return FieldTruth(
        analyte=analyte,
        status=EXTRACTABLE,
        printed_value=printed_value,
        printed_unit=printed_unit,
        unit_placement=placement,
        value=float(value),
        unit=unit,
        canonical_value=float(
            convert(Decimal(str(value)), unit, CANONICAL[analyte])
        ),
    )


def _no_unit(analyte, printed_value, value):
    return FieldTruth(
        analyte=analyte,
        status=NO_UNIT,
        printed_value=printed_value,
        value=float(value),
    )


def _absent(analyte):
    return FieldTruth(analyte=analyte, status=ABSENT)


def _image(image_id, path, sub, factors, fields):
    """Fill every analyte the page does not carry with an ABSENT entry."""
    complete = dict(fields)
    for analyte in ALL_ANALYTES:
        complete.setdefault(analyte, _absent(analyte))
    return ImageTruth(
        image_id=image_id,
        path=path,
        stratum="A",
        sub=sub,
        factors=dict(factors, source="real"),
        fields=complete,
    )


# --- A1: test1.jpg ----------------------------------------------------------
# "HEMATOLOGY REPORT". Four columns: TEST / RESULT / UNITS / REFERENCE RANGE.
# Uniform K/uL and M/uL, all ten analytes present, every unit in its own
# column. The reference-range column repeats the result ("14.5 - 14.5"),
# which is one reason this is clearly a synthetic layout.

_A1 = _image(
    "A1_test1",
    "test1.jpg",
    "A1",
    {
        "layout": "clean digital table",
        "unit_placement": "column",
        "decimal_style": "point",
        "reference_ranges": True,
        "notation": "conventional_k",
    },
    {
        "hb": _extractable("hb", "14.5", "14.5", "g/dL", "g/dL", "column"),
        "rbc": _extractable("rbc", "5.1", "5.1", "M/" + MU + "L", "million/uL", "column"),
        "wbc": _extractable("wbc", "7.2", "7.2", "K/" + MU + "L", "thousand/uL", "column"),
        "anc": _extractable("anc", "4.1", "4.1", "K/" + MU + "L", "thousand/uL", "column"),
        "neutrophils_percent": _extractable(
            "neutrophils_percent", "57.0", "57.0", "%", "%", "column"
        ),
        "platelets": _extractable(
            "platelets", "250", "250", "K/" + MU + "L", "thousand/uL", "column"
        ),
        "hematocrit": _extractable("hematocrit", "43.5", "43.5", "%", "%", "column"),
        "mcv": _extractable("mcv", "85.3", "85.3", "fL", "fL", "column"),
        "mch": _extractable("mch", "28.4", "28.4", "pg", "pg", "column"),
        "mchc": _extractable("mchc", "33.3", "33.3", "g/dL", "g/dL", "column"),
    },
)


# --- A2: test_2.jpg ---------------------------------------------------------
# "LABORATORY REPORT / HEMATOLOGY - CBC". Three columns, no reference ranges,
# and genuinely mixed scales on one page: RBC in millions/uL beside WBC,
# platelets and ANC in cells/uL. Platelets 85,000 cells/uL is CTCAE grade 1
# thrombocytopenia.
#
# The platelet unit cell reads "cells/uL (Low)". The annotation is part of the
# printed glyphs and is recorded as such; the unit those glyphs denote is
# still cells/uL. Whether a backend hands "cells/uL (Low)" or "cells/uL" to
# parse_unit decides whether the field survives, and the two backends differ
# on exactly this -- see docs/extraction-benchmark.md.

_A2 = _image(
    "A2_test_2",
    "test_2.jpg",
    "A2",
    {
        "layout": "clean digital table",
        "unit_placement": "column",
        "decimal_style": "point",
        "reference_ranges": False,
        "notation": "mixed",
        "annotated_unit_cell": True,
    },
    {
        "rbc": _extractable(
            "rbc", "4.40", "4.40", "millions/" + MU + "L", "million/uL", "column"
        ),
        "wbc": _extractable(
            "wbc", "5800", "5800", "cells/" + MU + "L", "cells/uL", "column"
        ),
        "platelets": _extractable(
            "platelets",
            "85000",
            "85000",
            "cells/" + MU + "L (Low)",
            "cells/uL",
            "column",
        ),
        "anc": _extractable(
            "anc", "3500", "3500", "cells/" + MU + "L", "cells/uL", "column"
        ),
        "neutrophils_percent": _extractable(
            "neutrophils_percent", "60", "60", "%", "%", "column"
        ),
        "hb": _extractable("hb", "13.0", "13.0", "g/dL", "g/dL", "column"),
        "hematocrit": _extractable("hematocrit", "39.0", "39.0", "%", "%", "column"),
        "mcv": _extractable("mcv", "83.5", "83.5", "fL", "fL", "column"),
        "mch": _extractable("mch", "28.0", "28.0", "pg", "pg", "column"),
        "mchc": _extractable("mchc", "33.2", "33.2", "g/dL", "g/dL", "column"),
    },
)


# --- A3: test_3.jpg ---------------------------------------------------------
# "Hospital / Complete Blood Count (CBC)". The unit is inside the value cell
# for five rows, absent entirely for four, and the MCHC row is below the
# bottom edge of the image.
#
# The four unit-less rows are the stratum's most useful fields: the correct
# behaviour is to refuse them, and 250,000 happens to be cells/uL here, so a
# backend that guesses from magnitude scores a hit for the wrong reason.
# Note also that no unit appears in the reference-range column for those rows
# either -- tests/fixtures/README.md says the platelet unit "appears only in
# the reference-range column", which is not what the page shows.

_A3 = _image(
    "A3_test_3",
    "test_3.jpg",
    "A3",
    {
        "layout": "clean digital table",
        "unit_placement": "value_cell",
        "decimal_style": "point",
        "reference_ranges": True,
        "notation": "mixed",
        "partial_crop": "MCHC row below the image edge",
    },
    {
        "hb": _extractable("hb", "14.5", "14.5", "g/dL", "g/dL", "value_cell"),
        "rbc": _extractable(
            "rbc", "4.5", "4.5", "millions/" + MU + "L", "million/uL", "value_cell"
        ),
        "wbc": _extractable(
            "wbc", "6,000", "6000", "cells/" + MU + "L", "cells/uL", "value_cell"
        ),
        "anc": _extractable(
            "anc", "2,500", "2500", "cells/" + MU + "L", "cells/uL", "value_cell"
        ),
        # "Neutrophils %  55" -- the unit is in the row label, to the left of
        # the number. ocr_extraction._unit_then_value exists for this case.
        "neutrophils_percent": _extractable(
            "neutrophils_percent", "55", "55", "%", "%", "label"
        ),
        "platelets": _no_unit("platelets", "250,000", "250000"),
        "hematocrit": _no_unit("hematocrit", "43", "43"),
        "mcv": _no_unit("mcv", "90", "90"),
        "mch": _no_unit("mch", "30", "30"),
        "mchc": _absent("mchc"),
    },
)


# --- A4: test_4.png ---------------------------------------------------------
# "Anemia Case CBC Report", a two-column screenshot carrying four analytes.
# Units are inside the value cell and spelled with mcL rather than the micro
# sign: "3.2 million cells/mcL", "5,100 cells/mcL", "210,000 /mcL".
#
# mcL is microlitre, so all three denote a cells-per-microlitre scale and are
# labelled accordingly. None of the three is in the units.py alias table, so
# this image is the stratum's clearest alias-coverage case: a backend can
# transcribe every glyph correctly and still lose three of four fields.

_A4 = _image(
    "A4_test_4",
    "test_4.png",
    "A4",
    {
        "layout": "clean digital table",
        "unit_placement": "value_cell",
        "decimal_style": "point",
        "reference_ranges": False,
        "notation": "conventional_cells",
        "unit_spelling": "mcL",
    },
    {
        "hb": _extractable("hb", "9.4", "9.4", "g/dL", "g/dL", "value_cell"),
        "rbc": _extractable(
            "rbc", "3.2", "3.2", "million cells/mcL", "million/uL", "value_cell"
        ),
        "wbc": _extractable(
            "wbc", "5,100", "5100", "cells/mcL", "cells/uL", "value_cell"
        ),
        "platelets": _extractable(
            "platelets", "210,000", "210000", "/mcL", "cells/uL", "value_cell"
        ),
    },
)


def stratum_a():
    """The four hand-labelled real images."""
    return [_A1, _A2, _A3, _A4]
