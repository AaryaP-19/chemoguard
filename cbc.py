"""The CBC analytes, and a panel that refuses to hold a bare number.

Every analyte declares one canonical unit. Values are converted to it once, on
the way in, and stay there. Nothing downstream re-scales: a consumer that
needs a float asks for it in a named unit via ``Quantity.magnitude_in``.

Canonical units are declared here and nowhere else:

    RBC                  million/uL
    WBC, ANC, Platelets  cells/uL
    Hb, MCHC             g/dL
    HCT, Neutrophils %   %
    MCV                  fL
    MCH                  pg

cells/uL is canonical for the three counts because the clinical thresholds are
stated that way -- ANC 1500, platelets 150000 -- and because it keeps 900
cells/uL and 900 thousand/uL distinguishable after conversion (900 against
900000) rather than collapsing them onto the same float, which is what
produced CLAUDE.md Finding 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from units import (
    CELLS_PER_UL,
    FEMTOLITRE,
    G_PER_DL,
    G_PER_L,
    MILLION_PER_UL,
    PERCENT,
    PICOGRAM,
    THOUSAND_PER_UL,
    Dimension,
    MissingUnitError,
    Quantity,
    Unit,
)


class AmbiguousNumberError(ValueError):
    """A printed number's separators do not determine its value.

    Distinct from a unit error and from an implausible one: the unit may be
    perfectly readable and the number still undecidable. ``1,234`` is 1234 on
    a US page and 1.234 on a European one, and no amount of context inside
    the row settles it.

    Refused rather than resolved by convention, for the same reason a missing
    unit is refused. Picking one reading gets it right on half the world's
    reports and silently wrong on the other half, and the wrong half points
    the same way every time -- see CLAUDE.md Finding 9 / M1, where an ANC
    printed 0,2 read as 2,000 and graded a life-threatening neutropenia as
    normal.
    """


class ImplausibleValueError(ValueError):
    """A canonicalised value lies outside physiological possibility.

    Distinct from a unit error: the unit was supplied and understood, and the
    number still cannot be a measurement of this analyte. Usually an OCR
    misread or a unit selected wrongly at the keyboard.

    This check rejects. It never rescales. Rescaling a number until it looks
    plausible is exactly the magnitude inference this module exists to remove
    -- see ocr_extraction._validate_value as it was before this change, whose
    WBC bound of (0.5, 100.0) silently discarded a genuine 5800 cells/uL.
    """


@dataclass(frozen=True)
class Analyte:
    """One measurable quantity on a CBC report."""

    key: str
    label: str
    canonical: Unit
    #: Bounds in the canonical unit, outside which no real measurement lies.
    #: Deliberately far wider than a reference range -- hyperleukocytosis
    #: reaches 500,000 cells/uL in acute leukaemia and that is a real reading.
    plausible_min: float
    plausible_max: float
    #: Units this analyte is commonly reported in, for the input form.
    reported_units: tuple

    @property
    def dimension(self) -> Dimension:
        return self.canonical.dimension


ANALYTES = {
    a.key: a
    for a in (
        Analyte("rbc", "Red Blood Cells (RBC)", MILLION_PER_UL,
                0.3, 9.0, (MILLION_PER_UL,)),
        Analyte("wbc", "White Blood Cells (WBC)", CELLS_PER_UL,
                10.0, 600_000.0, (CELLS_PER_UL, THOUSAND_PER_UL)),
        Analyte("anc", "Absolute Neutrophil Count (ANC)", CELLS_PER_UL,
                0.0, 500_000.0, (CELLS_PER_UL, THOUSAND_PER_UL)),
        Analyte("platelets", "Platelets", CELLS_PER_UL,
                500.0, 5_000_000.0, (CELLS_PER_UL, THOUSAND_PER_UL)),
        Analyte("neutrophils_percent", "Neutrophils", PERCENT,
                0.0, 100.0, (PERCENT,)),
        Analyte("hb", "Haemoglobin (Hb)", G_PER_DL,
                1.0, 30.0, (G_PER_DL, G_PER_L)),
        Analyte("hematocrit", "Haematocrit (HCT)", PERCENT,
                3.0, 80.0, (PERCENT,)),
        Analyte("mcv", "MCV", FEMTOLITRE, 30.0, 200.0, (FEMTOLITRE,)),
        Analyte("mch", "MCH", PICOGRAM, 5.0, 80.0, (PICOGRAM,)),
        Analyte("mchc", "MCHC", G_PER_DL, 10.0, 60.0, (G_PER_DL,)),
    )
}

#: The prediction path will not run without these three.
REQUIRED_FOR_PREDICTION = ("rbc", "wbc", "platelets")


def canonical_unit(key: str) -> Unit:
    """The declared canonical unit for an analyte."""
    return ANALYTES[key].canonical


def as_reported(key: str, value, unit) -> Quantity:
    """Parse a value and its unit exactly as the source gave them.

    No conversion. Raises MissingUnitError naming the analyte -- the generic
    message alone leaves the caller guessing which of five fields was at
    fault.
    """
    try:
        analyte = ANALYTES[key]
    except KeyError:
        raise KeyError("{0!r} is not a CBC analyte".format(key)) from None

    if isinstance(value, Quantity):
        return value
    try:
        return Quantity.of(value, unit)
    except MissingUnitError:
        raise MissingUnitError(
            "{0} was given as {1!r} with no unit. This application will not "
            "infer a unit from the magnitude of a laboratory value. Supply "
            "one of: {2}.".format(
                analyte.label, value,
                ", ".join(u.symbol for u in analyte.reported_units),
            )
        ) from None


def quantity_for(key: str, value, unit) -> Quantity:
    """Build a canonicalised Quantity for a named analyte.

    Raises MissingUnitError if ``unit`` is absent, IncompatibleUnitsError if
    it measures the wrong dimension, and ImplausibleValueError if the result
    cannot be a real measurement.
    """
    analyte = ANALYTES[key]
    quantity = as_reported(key, value, unit)
    canonical = quantity.to(analyte.canonical)   # raises on wrong dimension

    if not (analyte.plausible_min <= canonical.value <= analyte.plausible_max):
        raise ImplausibleValueError(
            "{0} of {1} is {2:g} {3} once converted, outside the physiologically "
            "possible range {4:g}-{5:g} {3}. Refusing to predict: check the "
            "unit and the extracted value.".format(
                analyte.label, quantity, canonical.value,
                analyte.canonical.symbol,
                analyte.plausible_min, analyte.plausible_max,
            )
        )
    return canonical


def derive_anc(wbc: Quantity, neutrophils_percent: Quantity) -> Quantity:
    """ANC from WBC and the neutrophil percentage.

    ANC = WBC x neutrophils%. Both operands carry their units, so the result
    carries one too, and no step in the calculation has to be told what scale
    anything is on. The code this replaces multiplied by 1000, divided by
    1000, then divided by 1000 again across three functions
    (app.py:126-134, :183, :149-153).
    """
    percent = neutrophils_percent.to(PERCENT).value
    cells = wbc.to(CELLS_PER_UL).value * percent / 100.0
    return Quantity(cells, CELLS_PER_UL)


@dataclass(frozen=True)
class CBCPanel:
    """A set of CBC values, each canonicalised, each with its unit.

    ``measured`` holds the canonical Quantity per analyte. ``reported`` holds
    what the source actually said, unconverted, so the record of the original
    is never lost -- storage and the PDF read from it.
    """

    measured: dict
    reported: dict
    #: How the units were established: "report" (read off the document),
    #: "operator" (chosen at the keyboard), or "derived" (computed from
    #: other quantities, as ANC is).
    unit_provenance: dict

    def __contains__(self, key) -> bool:
        return key in self.measured

    def get(self, key: str) -> Optional[Quantity]:
        return self.measured.get(key)

    def require(self, key: str) -> Quantity:
        try:
            return self.measured[key]
        except KeyError:
            raise KeyError(
                "{0} is required and was not supplied.".format(
                    ANALYTES[key].label if key in ANALYTES else key
                )
            ) from None

    def with_derived_anc(self) -> "CBCPanel":
        """Return a panel with ANC filled in from WBC and neutrophil %.

        A reported ANC always wins; this only fills a gap.
        """
        if "anc" in self.measured:
            return self
        if "wbc" not in self.measured or "neutrophils_percent" not in self.measured:
            return self

        anc = derive_anc(self.measured["wbc"], self.measured["neutrophils_percent"])
        measured = dict(self.measured, anc=anc)
        reported = dict(self.reported, anc=anc)
        provenance = dict(self.unit_provenance, anc="derived")
        return CBCPanel(measured, reported, provenance)

    def missing_required(self) -> list:
        return [k for k in REQUIRED_FOR_PREDICTION if k not in self.measured]

    def as_dict(self) -> dict:
        """JSON form: every value with its unit, canonical and as recorded."""
        return {
            key: {
                "canonical": quantity.as_dict(),
                "as_reported": self.reported[key].as_dict(),
                "unit_source": self.unit_provenance.get(key, "unknown"),
            }
            for key, quantity in self.measured.items()
        }


def panel_from_extraction(raw: dict, unit_source: str = "report"):
    """Build a panel from an extraction backend's raw output.

    ``raw`` maps analyte to ``{"value": ..., "unit": ...}`` or None. Shared
    by both backends so Tesseract and GPT-4o-mini cannot diverge on what
    counts as an acceptable reading.

    Returns ``(panel, skipped)``. An entry lands in ``skipped`` -- rather
    than in the panel -- when its unit is absent, unrecognised, or of the
    wrong dimension, or when the canonicalised value is impossible. Skipping
    is the point: a value the extractor could not fully read is surfaced to
    the operator to supply deliberately, not quietly repaired.
    """
    values, skipped = {}, {}

    for key, entry in (raw or {}).items():
        if key not in ANALYTES or entry is None:
            continue

        value = entry.get("value") if isinstance(entry, dict) else entry
        unit = entry.get("unit") if isinstance(entry, dict) else None

        # A number the extractor sighted but could not pin down. Surfaced as
        # a refusal rather than dropped: silence would leave the operator
        # unable to tell "the report has no platelet count" from "the count
        # is printed and I could not tell 250,000 from 250.000", and only the
        # second is something they can resolve.
        if isinstance(entry, dict) and entry.get("ambiguous_number") is not None:
            skipped[key] = {
                "value": None,
                "unit": unit,
                "printed": entry["ambiguous_number"],
                "reason": "AmbiguousNumberError",
                "detail": entry.get("detail")
                or "{0} was printed as {1!r}, whose separators do not "
                   "determine its value.".format(
                       ANALYTES[key].label, entry["ambiguous_number"]
                   ),
            }
            continue

        if value is None:
            continue

        try:
            original = as_reported(key, value, unit)
            quantity_for(key, original, None)      # canonicalise to validate
        except (MissingUnitError, ImplausibleValueError, ValueError) as exc:
            skipped[key] = {
                "value": value,
                "unit": unit,
                "reason": type(exc).__name__,
                "detail": str(exc),
            }
            continue

        values[key] = original

    return build_panel(values, unit_source=unit_source), skipped


def panel_from_mapping(data: dict, unit_source: str = "operator") -> CBCPanel:
    """Build a panel from a request body, in any of three accepted shapes.

    Nested, which is the documented form::

        {"wbc": {"value": 5800, "unit": "cells/uL"}}

    Flat, with a sibling ``_unit`` key, for curl and the CLI::

        {"wbc": 5800, "wbc_unit": "cells/uL"}

    Or a shared ``units`` block, which is what the browser form sends::

        {"wbc": 5800, "units": {"wbc": "cells/uL"}}

    A value present in any shape without a unit raises MissingUnitError. The
    shapes differ only in where the unit is written down, never in whether
    one is required.
    """
    shared_units = data.get("units") or {}
    values = {}

    for key in ANALYTES:
        supplied = data.get(key)
        if supplied is None or supplied == "":
            continue

        if isinstance(supplied, dict):
            values[key] = (supplied.get("value"), supplied.get("unit"))
        else:
            unit = data.get(key + "_unit") or shared_units.get(key)
            values[key] = (supplied, unit)

    return build_panel(values, unit_source=unit_source)


def build_panel(values: dict, unit_source: str = "operator") -> CBCPanel:
    """Build a panel from ``{analyte: (value, unit)}`` or ``{analyte: Quantity}``.

    Absent analytes are simply absent. An analyte that is present without a
    unit raises -- that is the boundary this whole change exists to draw.
    """
    measured, reported, provenance = {}, {}, {}

    for key, supplied in values.items():
        if supplied is None:
            continue
        if key not in ANALYTES:
            continue

        if isinstance(supplied, Quantity):
            value, unit = supplied.value, supplied.unit
        elif isinstance(supplied, dict):
            value, unit = supplied.get("value"), supplied.get("unit")
        else:
            value, unit = supplied
        if value is None:
            continue

        original = as_reported(key, value, unit)
        measured[key] = quantity_for(key, original, None)
        reported[key] = original
        provenance[key] = unit_source

    return CBCPanel(measured, reported, provenance)
