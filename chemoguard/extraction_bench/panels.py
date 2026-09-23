"""The panel bank, the unit scales, and how a printed number is derived.

Three panels, each a coherent CBC rather than a set of independent numbers.
Coherence matters here for one specific reason: ``ANC = WBC x neutrophils%``
is an identity the application computes (``cbc.derive_anc``), so a panel whose
three values disagree would make a derived-ANC reading look like an extraction
error. Each panel below satisfies it exactly.

The panels span the clinical range the silent-error analysis needs:

``normal``
    Nothing flagged. The control.

``cytopenic``
    Platelets 85,000 cells/uL and ANC 900 cells/uL -- CTCAE grade 1
    thrombocytopenia with grade 3 neutropenia. This is the ``test_2.jpg``
    case: read as 85,000 *thousand*/uL it is eleven times the upper limit of
    normal and flagged as nothing at all.

``severe``
    Platelets 900 cells/uL. This is the exact number in the ``units.py``
    docstring -- profound thrombocytopenia that reads as a comfortable 900
    thousand/uL to anything inferring scale from magnitude, and the reason
    that module exists. If a notation or quality variant reintroduces
    magnitude inference by another route, this panel is where it shows.

Printed numbers are computed with the ``_FACTORS`` table in this module and
**not** with ``units.Quantity``. That is a deliberate independent
reimplementation: if ground truth were derived from the module under test, a
conversion bug in ``units.py`` would agree with itself and be invisible.
``tests/test_extraction_bench.py`` asserts the two agree, so a divergence
fails loudly instead.

Numbers are carried as ``Decimal`` throughout. The printed glyphs are the
ground truth, and ``5800 / 1000`` in binary floating point is not exactly
``5.8``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Tuple

#: Canonical unit per analyte, mirroring ``cbc.ANALYTES``. Repeated here so
#: the expected value can be computed without importing the code under test.
CANONICAL = {
    "rbc": "million/uL",
    "wbc": "cells/uL",
    "anc": "cells/uL",
    "platelets": "cells/uL",
    "neutrophils_percent": "%",
    "hb": "g/dL",
    "hematocrit": "%",
    "mcv": "fL",
    "mch": "pg",
    "mchc": "g/dL",
}

#: Independent conversion factors to each dimension's base unit, as exact
#: Decimals. Compare with ``units.Unit.to_base``; the test suite asserts
#: agreement.
_FACTORS = {
    "cells/uL": Decimal(1),
    "thousand/uL": Decimal(1000),
    "million/uL": Decimal(1000000),
    "g/dL": Decimal(1),
    "g/L": Decimal("0.1"),
    "fL": Decimal(1),
    "pg": Decimal(1),
    "%": Decimal(1),
}

#: Analyte order as printed on a report. Not alphabetical: this is roughly
#: the order the checked-in images use, so a generated page reads like one.
ANALYTE_ORDER = (
    "hb",
    "rbc",
    "wbc",
    "anc",
    "neutrophils_percent",
    "platelets",
    "hematocrit",
    "mcv",
    "mch",
    "mchc",
)

#: Row labels, in the style of the checked-in images.
LABELS = {
    "hb": "Hemoglobin (Hb)",
    "rbc": "Red Blood Cells (RBC)",
    "wbc": "White Blood Cells (WBC)",
    "anc": "Absolute Neutrophil Count (ANC)",
    "neutrophils_percent": "Neutrophils %",
    "platelets": "Platelets",
    "hematocrit": "Hematocrit (HCT)",
    "mcv": "MCV",
    "mch": "MCH",
    "mchc": "MCHC",
}

#: Labels used when the unit is printed inside the parameter name, so that
#: "Hematocrit (HCT)" plus "%" reads "Hematocrit (%)" rather than
#: "Hematocrit (HCT) (%)". The abbreviation is dropped because the
#: parenthesis is now carrying the unit.
LABELS_FOR_EMBEDDED_UNIT = dict(
    LABELS,
    hb="Hemoglobin",
    hematocrit="Hematocrit",
    rbc="Red Blood Cells",
    wbc="White Blood Cells",
    anc="Absolute Neutrophil Count",
)

#: Analytes whose label already states the unit. Appending it again would
#: print "Neutrophils % (%)"; ``test_3.jpg`` shows the real form, which is
#: the label alone. The unit is still printed -- it is in the label -- so
#: these fields stay extractable under every placement except ``absent``.
UNIT_ALREADY_IN_LABEL = ("neutrophils_percent",)


@dataclass(frozen=True)
class Panel:
    """A coherent CBC, stated in canonical units as exact Decimals."""

    name: str
    values: Dict[str, Decimal]

    def check_anc_identity(self) -> None:
        """ANC must equal WBC x neutrophils%, exactly.

        Called by the test suite. An incoherent panel is a defect in the
        benchmark, not a finding about a backend.
        """
        expected = self.values["wbc"] * self.values["neutrophils_percent"] / 100
        if expected != self.values["anc"]:
            raise ValueError(
                "{0}: ANC {1} != WBC {2} x {3}% = {4}".format(
                    self.name,
                    self.values["anc"],
                    self.values["wbc"],
                    self.values["neutrophils_percent"],
                    expected,
                )
            )


def _d(text: str) -> Decimal:
    return Decimal(text)


PANELS = {
    panel.name: panel
    for panel in (
        Panel(
            "normal",
            {
                "hb": _d("14.2"),
                "rbc": _d("4.80"),
                "wbc": _d("6500"),
                "anc": _d("3900"),
                "neutrophils_percent": _d("60"),
                "platelets": _d("250000"),
                "hematocrit": _d("42.0"),
                "mcv": _d("88.0"),
                "mch": _d("29.5"),
                "mchc": _d("33.5"),
            },
        ),
        Panel(
            "cytopenic",
            {
                "hb": _d("9.8"),
                "rbc": _d("3.40"),
                "wbc": _d("2000"),
                "anc": _d("900"),
                "neutrophils_percent": _d("45"),
                "platelets": _d("85000"),
                "hematocrit": _d("29.0"),
                "mcv": _d("85.0"),
                "mch": _d("28.0"),
                "mchc": _d("32.5"),
            },
        ),
        Panel(
            "severe",
            {
                "hb": _d("7.2"),
                "rbc": _d("2.60"),
                "wbc": _d("800"),
                "anc": _d("200"),
                "neutrophils_percent": _d("25"),
                "platelets": _d("900"),
                "hematocrit": _d("22.0"),
                "mcv": _d("78.0"),
                "mch": _d("24.0"),
                "mchc": _d("30.5"),
            },
        ),
    )
}


# --- unit scales -------------------------------------------------------------
#
# Three ways the same panel is reported. Note what differs between
# ``conventional_k`` and ``si``: for the four count analytes, nothing but the
# unit string. 10^9/L *is* thousand/uL and 10^12/L *is* million/uL, so the
# digits are identical and only the spelling changes. That makes the pair an
# isolation: a backend that scores differently across them failed to read a
# unit string, not a number. Hb and MCHC are the only analytes whose digits
# move under SI, because g/L is a genuinely different scale from g/dL.

#: Scale -> analyte -> (printed unit text, canonical symbol it denotes).
#: An analyte absent from a scale's map keeps the ``_INVARIANT`` entry.
_INVARIANT = {
    "neutrophils_percent": ("%", "%"),
    "hematocrit": ("%", "%"),
    "mcv": ("fL", "fL"),
    "mch": ("pg", "pg"),
}

SCALES = {
    "conventional_cells": dict(
        _INVARIANT,
        hb=("g/dL", "g/dL"),
        mchc=("g/dL", "g/dL"),
        rbc=("millions/uL", "million/uL"),
        wbc=("cells/uL", "cells/uL"),
        anc=("cells/uL", "cells/uL"),
        platelets=("cells/uL", "cells/uL"),
    ),
    "conventional_k": dict(
        _INVARIANT,
        hb=("g/dL", "g/dL"),
        mchc=("g/dL", "g/dL"),
        rbc=("M/uL", "million/uL"),
        wbc=("K/uL", "thousand/uL"),
        anc=("K/uL", "thousand/uL"),
        platelets=("K/uL", "thousand/uL"),
    ),
    "si": dict(
        _INVARIANT,
        hb=("g/L", "g/L"),
        mchc=("g/L", "g/L"),
        rbc=("10^12/L", "million/uL"),
        wbc=("10^9/L", "thousand/uL"),
        anc=("10^9/L", "thousand/uL"),
        platelets=("10^9/L", "thousand/uL"),
    ),
}

#: Hematocrit under SI is properly a dimensionless fraction in L/L (0.42, not
#: 42%), and ``units.py`` has no L/L unit and no dimensionless dimension to
#: put one in. Rendering it would make every SI hematocrit field unreadable
#: for a reason that has nothing to do with OCR, so the SI scale keeps % here
#: and the gap is reported as an alias-coverage finding instead. See
#: ``alias_probe.py``.
SI_HEMATOCRIT_IS_UNREPRESENTABLE = True


def convert(value: Decimal, from_unit: str, to_unit: str) -> Decimal:
    """Exact Decimal conversion within a dimension, independent of units.py."""
    if from_unit == to_unit:
        return value
    return value * _FACTORS[from_unit] / _FACTORS[to_unit]


def _format_count(value: Decimal) -> str:
    """Print a count the way a laboratory prints it.

    Whole numbers at or above 10 get no decimal places -- a report says
    "250 K/uL", never "250.0 K/uL". Below 10 a single decimal place is kept,
    because that is where it carries information: 0.9 K/uL is grade 4
    thrombocytopenia and 0.2 K/uL is an ANC that stops treatment.
    """
    if value == value.to_integral_value() and value >= 10:
        return str(value.to_integral_value())
    return str(value.quantize(Decimal("0.1")))


#: Decimal places for the analytes whose formatting does not depend on scale.
_PLACES = {
    "rbc": 2,
    "neutrophils_percent": 0,
    "hematocrit": 1,
    "mcv": 1,
    "mch": 1,
}


def printed_number(analyte: str, value: Decimal, unit: str) -> str:
    """The digits a report would print, before decimal-mark styling.

    Returns a plain ASCII number using ``.`` as the decimal mark and no
    thousands separator; ``render.style_number`` applies the notation.
    """
    if analyte in _PLACES:
        return str(value.quantize(Decimal(1).scaleb(-_PLACES[analyte])))
    if analyte in ("wbc", "anc", "platelets"):
        if unit == "cells/uL":
            return str(value.to_integral_value())
        return _format_count(value)
    if analyte in ("hb", "mchc"):
        # g/dL keeps one decimal (14.2); g/L is reported whole (142).
        return str(value.quantize(Decimal("0.1") if unit == "g/dL" else Decimal(1)))
    raise KeyError("no print rule for {0!r}".format(analyte))


def row_for(panel: Panel, analyte: str, scale: str) -> Tuple[str, str, str, Decimal]:
    """One row's printed content under a scale.

    Returns ``(printed_number, printed_unit, canonical_unit, true_value)``
    where ``true_value`` is the number in the printed unit **after** rounding
    to the printed precision.

    The re-derivation from the printed text is the important part. A report
    that prints ANC as "0.9 K/uL" is stating 900 cells/uL, not the 903 the
    underlying panel might have held, and the benchmark scores whether a
    backend read the page. So the printed glyphs are authoritative and the
    canonical value is computed back from them.
    """
    printed_unit, canonical = SCALES[scale][analyte]
    canonical_for_analyte = CANONICAL[analyte]

    in_printed_unit = convert(panel.values[analyte], canonical_for_analyte, canonical)
    text = printed_number(analyte, in_printed_unit, canonical)
    return text, printed_unit, canonical, Decimal(text)
