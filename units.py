"""Explicit units for laboratory values.

A CBC value is a number *and* a unit. This module makes that pairing the only
representation the rest of the application passes around, and puts every
conversion behind a single method.

Why this exists: the code this replaces inferred the unit from the magnitude
(``if value > 1000: value /= 1000``). Platelets of 900 cells/uL -- profound
thrombocytopenia -- were read as 900 thousand/uL and reported Low risk. Every
error that heuristic makes points the same way, critical to normal, because a
dangerously low count in cells/uL looks like a comfortable count in
thousands/uL. See CLAUDE.md Findings 3, 4 and 7.

Symbols are ASCII on purpose. ``uL``, not the micro sign. Unit strings end up
in log lines and console output, and the Windows console defaults to cp1252,
where a non-ASCII character raises UnicodeEncodeError -- a failure this repo
has already misdiagnosed once (CLAUDE.md, "Fixes that look done and aren't").
Input parsing accepts every spelling; only the canonical symbol is
constrained. ``Unit.display`` carries the typographic form for HTML and PDF,
where the encoding is under our control.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class UnitError(ValueError):
    """Base class for every unit failure. Always fatal to a prediction."""


class MissingUnitError(UnitError):
    """A value arrived with no unit.

    Raised rather than guessed. A prediction made from a number of unknown
    scale is not a weaker prediction, it is an unrelated one.
    """


class UnknownUnitError(UnitError):
    """A unit string was supplied but is not one we recognise.

    Also fatal: an unrecognised unit is not evidence that the value happens to
    be in the unit we would have preferred.
    """


class IncompatibleUnitsError(UnitError):
    """A conversion was attempted between different physical dimensions."""


class Dimension(Enum):
    """What kind of thing a unit measures. Conversion never crosses these."""

    COUNT_CONCENTRATION = "count concentration"   # cells per volume
    MASS_CONCENTRATION = "mass concentration"     # mass per volume
    VOLUME = "volume"
    MASS = "mass"
    FRACTION = "fraction"


@dataclass(frozen=True)
class Unit:
    """A unit of measurement.

    ``to_base`` is the factor that converts a value in this unit to the
    dimension's base unit. It is the only number in the module that
    participates in arithmetic.
    """

    symbol: str
    dimension: Dimension
    to_base: float
    display: str

    def __str__(self) -> str:
        return self.symbol


# Base units, one per dimension: cells/uL, g/dL, fL, pg, %.
# Their to_base is 1.0 by definition.

CELLS_PER_UL = Unit("cells/uL", Dimension.COUNT_CONCENTRATION, 1.0, "cells/µL")
THOUSAND_PER_UL = Unit("thousand/uL", Dimension.COUNT_CONCENTRATION, 1e3, "thousand/µL")
MILLION_PER_UL = Unit("million/uL", Dimension.COUNT_CONCENTRATION, 1e6, "million/µL")

G_PER_DL = Unit("g/dL", Dimension.MASS_CONCENTRATION, 1.0, "g/dL")
G_PER_L = Unit("g/L", Dimension.MASS_CONCENTRATION, 0.1, "g/L")
# Iron-status and inflammation markers live six to seven orders of magnitude
# below haemoglobin, but they are the same dimension and belong on the same
# scale rather than in a parallel one: 1 ug/L = 1e-6 g/L = 1e-7 g/dL, and
# 1 mg/L = 1e-3 g/L = 1e-4 g/dL. Putting them here means a ferritin handed to
# a haemoglobin field converts rather than silently passing, and then fails
# its plausibility bound instead of being believed.
MICROGRAM_PER_L = Unit("ug/L", Dimension.MASS_CONCENTRATION, 1e-7, "µg/L")
MILLIGRAM_PER_L = Unit("mg/L", Dimension.MASS_CONCENTRATION, 1e-4, "mg/L")

FEMTOLITRE = Unit("fL", Dimension.VOLUME, 1.0, "fL")
PICOGRAM = Unit("pg", Dimension.MASS, 1.0, "pg")
PERCENT = Unit("%", Dimension.FRACTION, 1.0, "%")

ALL_UNITS = (
    CELLS_PER_UL,
    THOUSAND_PER_UL,
    MILLION_PER_UL,
    G_PER_DL,
    G_PER_L,
    MICROGRAM_PER_L,
    MILLIGRAM_PER_L,
    FEMTOLITRE,
    PICOGRAM,
    PERCENT,
)


#: A result flag printed inside the units cell. ``test_2.jpg`` prints
#: "cells/uL (Low)"; a report that flags an abnormal value in the same cell as
#: the unit should not thereby make the unit unreadable. Stripped before
#: anything else, because ``_normalise`` drops parentheses further down and
#: "cells/uL (Low)" would otherwise become the key "cellsullow".
_RESULT_FLAG = re.compile(
    r"[\s*#!]*\(\s*(?:low|high|l|h|abn|abnormal|crit|critical|panic)\s*\)[\s*#!]*$",
    re.IGNORECASE,
)

#: A bare marker with no parentheses -- "cells/uL *" or "12.4 g/dL #".
_BARE_MARKER = re.compile(r"[\s*#!]+$")

#: Superscript digits, for the exponent forms reports actually print. A
#: laboratory writes 10⁹/L, not 10^9/L; the caret form is the plain-text
#: transliteration and both have to resolve to the same unit.
_SUPERSCRIPTS = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3",
    "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7",
    "⁸": "8", "⁹": "9",
}

_SUPERSCRIPT_RUN = re.compile("[" + "".join(_SUPERSCRIPTS) + "]+")
#: "10e9/L", "10*9/L" -- E notation and an asterisk standing in for a caret.
_EXPONENT_SPELLING = re.compile(r"10\s*[e*^]?\s*(\d{1,2})\s*/")


def _normalise(text: str) -> str:
    """Reduce a unit string to a comparison key.

    Both micro signs collapse to ``u``, ``mcL`` collapses to ``uL``, every
    exponent spelling collapses to ``10^n``, a trailing result flag is
    dropped, and everything that only affects appearance goes. Applied to the
    alias table as well as to input, so the two cannot drift.
    """
    key = _RESULT_FLAG.sub("", text.strip())
    key = _BARE_MARKER.sub("", key).strip().lower()

    for micro in ("µ", "μ"):          # MICRO SIGN, GREEK SMALL MU
        key = key.replace(micro, "u")
    key = key.replace("×", "x")            # MULTIPLICATION SIGN

    key = _SUPERSCRIPT_RUN.sub(
        lambda m: "^" + "".join(_SUPERSCRIPTS[c] for c in m.group()), key
    )
    key = key.replace("**", "^").replace("e^", "^")
    # Normalise every exponent spelling to a caret, including the two-digit
    # ones: 10^12/L is the SI unit for a red cell count and 10^9/L for a
    # white cell count, and reports print them as 10e12, 10*12 and 10 12 too.
    key = _EXPONENT_SPELLING.sub(lambda m: "10^" + m.group(1) + "/", key)

    for noise in (" ", "\t", ",", ".", "_", "(", ")", "’"):
        key = key.replace(noise, "")

    # mcL is microlitre. Collapsed here rather than registering every mcL
    # spelling separately, so the multiplier forms -- "million cells/mcL" --
    # come along without an entry each. **After** the noise strip, not
    # before: OCR splits "mcL" as "mc L" often enough that collapsing first
    # lets the spaced spelling escape.
    key = key.replace("mcl", "ul")

    key = key.replace("x10^", "10^").replace("*10^", "10^")
    return key


# Spellings seen on real report layouts, mapped to the unit they denote.
#
# 1/mm^3 is exactly 1/uL -- a cubic millimetre is a microlitre -- so /cumm,
# /cmm and /mm3 are cells/uL, not an approximation of it. 10^9/L is exactly
# thousand/uL and 10^12/L is exactly million/uL; both are the SI spellings used
# outside the US, and both are equalities rather than conventions.
_ALIASES: dict = {}


def _register(unit: Unit, *spellings: str) -> None:
    for spelling in spellings:
        _ALIASES[_normalise(spelling)] = unit


_register(
    CELLS_PER_UL,
    "cells/uL", "cell/uL", "/uL", "uL^-1",
    "cells/mm3", "cells/mm^3", "/mm3", "/mm^3", "/cumm", "/cu mm", "/cmm",
    "cells per uL", "cells/cumm", "count/uL", "10^0/uL",
    # Spellings the extraction benchmark found on real report layouts.
    # mcL is handled in _normalise, so these are the remaining shapes.
    # A bare "cells" is deliberately absent: it is a count, not a count
    # concentration, and accepting it would be the module guessing a
    # denominator it was not told.
    "cumm", "cmm", "cells/cmm",
)
_register(
    THOUSAND_PER_UL,
    "thousand/uL", "thousands/uL", "K/uL", "10^3/uL", "x10^3/uL",
    "10^9/L", "x10^9/L", "K/cumm", "thousand/cumm", "thou/uL", "10e3/uL",
    "Th/uL", "thsd/uL", "thousand cells/uL", "thousands cells/uL",
)
_register(
    MILLION_PER_UL,
    "million/uL", "millions/uL", "M/uL", "mil/uL",
    "10^6/uL", "x10^6/uL", "10^12/L", "x10^12/L", "10e6/uL",
    "million/cumm", "millions/cumm",
    "mill/uL", "million cells/uL", "millions cells/uL",
)
# Giga- and tera-per-litre are the European count spellings, and they are
# deliberately NOT registered. ``_normalise`` lower-cases before lookup, so
# "G/L" (a count, 10^9/L) and "g/L" (a mass concentration) collapse onto the
# same key, and "G/L" currently resolves to grams per litre. Distinguishing
# them means making the normaliser case-sensitive for one pair of units,
# which is a safety judgement with an argument on both sides -- a wrongly
# case-folded haemoglobin and a wrongly case-folded white cell count are
# different kinds of harm. Left for a human decision rather than settled
# here. Today the dimension check in ``Quantity.to`` refuses "G/L" for a
# count analyte, so it fails safe; see docs/extraction-benchmark.md.
_register(G_PER_DL, "g/dL", "gm/dL", "gms/dL", "gm%", "g%", "gram/dL")
_register(G_PER_L, "g/L", "gm/L")
# ng/mL is exactly ug/L -- a nanogram per millilitre is a microgram per litre
# -- which is why NHANES publishes ferritin twice, as LBXFER in ng/mL and
# LBDFERSI in ug/L, and says the conversion factor is 1.00.
_register(MICROGRAM_PER_L, "ug/L", "mcg/L", "ng/mL", "ng/ml", "ug/l")
_register(MILLIGRAM_PER_L, "mg/L", "mg/l", "ug/mL")
_register(FEMTOLITRE, "fL", "femtolitre", "femtoliter", "cu microns", "um^3")
_register(PICOGRAM, "pg", "picogram", "picogramme", "pcg")
_register(PERCENT, "%", "percent", "pct", "per cent")


def parse_unit(text):
    """Resolve a unit string to a Unit.

    Raises MissingUnitError for None or blank, UnknownUnitError otherwise.
    Never returns a default -- that is the entire point of the module.
    """
    if isinstance(text, Unit):
        return text
    if text is None or not str(text).strip():
        raise MissingUnitError(
            "No unit supplied. A laboratory value without a unit cannot be "
            "interpreted, and this application will not guess one from its "
            "magnitude."
        )
    key = _normalise(str(text))
    try:
        return _ALIASES[key]
    except KeyError:
        known = ", ".join(sorted({u.symbol for u in ALL_UNITS}))
        raise UnknownUnitError(
            "Unrecognised unit {0!r}. Known units: {1}.".format(str(text), known)
        ) from None


def known_unit_symbols():
    """Canonical symbols, for error messages and API documentation."""
    return sorted({unit.symbol for unit in ALL_UNITS})


@dataclass(frozen=True)
class Quantity:
    """A number with the unit it was measured in.

    Immutable, so a Quantity that has been validated stays validated. ``to()``
    is the only conversion in the codebase; anything needing a bare float asks
    for one explicitly and names the unit it wants.
    """

    value: float
    unit: Unit

    def __post_init__(self) -> None:
        if not isinstance(self.unit, Unit):
            raise MissingUnitError(
                "Quantity requires a Unit, got {0}. Build it with "
                "Quantity.of(value, 'cells/uL').".format(type(self.unit).__name__)
            )
        try:
            object.__setattr__(self, "value", float(self.value))
        except (TypeError, ValueError):
            raise UnitError("{0!r} is not a number".format(self.value)) from None

    @classmethod
    def of(cls, value, unit) -> "Quantity":
        """Build from a value and a unit string, parsing the unit."""
        return cls(value, parse_unit(unit))

    def to(self, unit) -> "Quantity":
        """Convert to another unit of the same dimension.

        The one place a laboratory value is ever scaled.
        """
        target = parse_unit(unit)
        if target.dimension is not self.unit.dimension:
            raise IncompatibleUnitsError(
                "Cannot convert {0} ({1}) to {2} ({3}).".format(
                    self.unit.symbol, self.unit.dimension.value,
                    target.symbol, target.dimension.value,
                )
            )
        if target is self.unit:
            return self
        return Quantity(self.value * self.unit.to_base / target.to_base, target)

    def magnitude_in(self, unit) -> float:
        """The bare number in a named unit.

        Use only where a float is genuinely required -- a model's feature
        vector, a threshold comparison -- and never to hand onward.
        """
        return self.to(unit).value

    def as_dict(self) -> dict:
        """JSON form. The unit travels with the value, always."""
        return {"value": self.value, "unit": self.unit.symbol}

    def __str__(self) -> str:
        return "{0:g} {1}".format(self.value, self.unit.symbol)
