"""Alias coverage: which printed unit spellings survive ``units.parse_unit``.

Direct probe, no images. A rendered corpus can only exercise the spellings it
was told to draw, and the number of spellings real laboratories print is much
larger than any corpus worth generating. So this sweeps the spellings
directly and reports three outcomes per string:

``accepted``        ``parse_unit`` resolves it
``needs_alias``     unambiguous, and ``parse_unit`` does not resolve it. A
                    one-line addition to the alias table.
``needs_dimension`` not expressible in the current type system at all, so not
                    an alias question. Haemoglobin in mmol/L needs a molar
                    mass; a haematocrit in L/L needs a dimensionless
                    dimension that ``units.Dimension`` does not have.

The ``needs_dimension`` bucket is kept separate on purpose. Lumping it in
with the missing aliases would imply a cheap fix and invite somebody to
register ``mmol/L`` against a made-up factor, which is the kind of quiet
fabrication ``units.py`` exists to prevent -- the g/dL-to-mmol/L factor for
haemoglobin (0.6206) is specific to the haemoglobin tetramer and is not a
unit conversion at all.

The ``expected`` column records what each spelling denotes, so a future
alias addition can be checked against an intent rather than against whatever
the table happens to do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

ACCEPTED = "accepted"
NEEDS_ALIAS = "needs_alias"
NEEDS_DIMENSION = "needs_dimension"

MU = "µ"        # MICRO SIGN
GREEK_MU = "μ"  # GREEK SMALL LETTER MU
SUP_3 = "³"
SUP_9 = "⁹"
SUP_1 = "¹"
SUP_2 = "²"
TIMES = "×"


@dataclass(frozen=True)
class Probe:
    """One spelling, what it denotes, and where it was seen."""

    text: str
    #: Canonical symbol it denotes, or None when it is not expressible.
    expected: Optional[str]
    note: str


#: Grouped by dimension. Every entry is a spelling that appears on real
#: laboratory output; the ones marked as seen in this repository's own
#: fixtures are called out, because those are not hypothetical.
PROBES = (
    # --- count concentration: the micro sign and its substitutes ----------
    Probe("cells/" + MU + "L", "cells/uL", "micro sign; test_2.jpg"),
    Probe("cells/" + GREEK_MU + "L", "cells/uL", "Greek mu, a common mojibake"),
    Probe("cells/uL", "cells/uL", "ASCII fallback"),
    Probe("cells/mcL", "cells/uL", "mcL for microlitre; test_4.png"),
    Probe("cells/mcl", "cells/uL", "mcL, lower case"),
    Probe("/mcL", "cells/uL", "bare per-microlitre; test_4.png"),
    Probe("million cells/mcL", "million/uL", "multiplier plus 'cells'; test_4.png"),
    Probe("million cells/" + MU + "L", "million/uL", "same, micro sign"),
    Probe("cells/" + MU + "L (Low)", "cells/uL", "flag in the units cell; test_2.jpg"),
    Probe("cells/" + MU + "L (H)", "cells/uL", "single-letter flag"),
    Probe("cells/" + MU + "L *", "cells/uL", "asterisk marking an abnormal result"),
    Probe("K/" + MU + "L", "thousand/uL", "test1.jpg"),
    Probe("Th/" + MU + "L", "thousand/uL", "Th abbreviation"),
    Probe("thou/" + MU + "L", "thousand/uL", None),
    Probe("10" + SUP_3 + "/" + MU + "L", "thousand/uL", "superscript exponent"),
    Probe("M/" + MU + "L", "million/uL", "test1.jpg"),
    Probe("mill/" + MU + "L", "million/uL", "abbreviated multiplier"),
    # --- count concentration: SI -----------------------------------------
    Probe("10^9/L", "thousand/uL", "plain-text SI"),
    Probe("x10^9/L", "thousand/uL", "explicit multiplier"),
    Probe("10" + SUP_9 + "/L", "thousand/uL", "superscript SI, the printed form"),
    Probe(TIMES + "10" + SUP_9 + "/L", "thousand/uL", "multiplication sign plus superscript"),
    Probe("10E9/L", "thousand/uL", "E notation"),
    Probe("10*9/L", "thousand/uL", "asterisk exponent"),
    Probe("10^12/L", "million/uL", "SI for RBC"),
    Probe("10" + SUP_1 + SUP_2 + "/L", "million/uL", "superscript SI for RBC"),
    Probe("G/L", "thousand/uL", "giga-per-litre, used in parts of Europe"),
    Probe("T/L", "million/uL", "tera-per-litre, same convention"),
    # --- count concentration: per cubic millimetre -----------------------
    Probe("/cumm", "cells/uL", None),
    Probe("/cu mm", "cells/uL", None),
    Probe("/mm3", "cells/uL", None),
    Probe("/mm" + SUP_3, "cells/uL", "superscript"),
    Probe("cmm", "cells/uL", "bare cmm"),
    # --- mass concentration ----------------------------------------------
    Probe("g/dL", "g/dL", "test1.jpg, test_2.jpg"),
    Probe("gm/dL", "g/dL", None),
    Probe("g%", "g/dL", "older Indian and UK reports"),
    Probe("g/L", "g/L", "SI haemoglobin"),
    Probe("mmol/L", None, "SI haemoglobin in NL/DK/SE; needs a molar mass"),
    # --- volume, mass ----------------------------------------------------
    Probe("fL", "fL", None),
    Probe("fl", "fL", "lower case"),
    Probe("femtoliter", "fL", None),
    Probe(MU + "m" + SUP_3, "fL", "cubic micron, superscript"),
    Probe("um^3", "fL", "ASCII cubic micron"),
    Probe("pg", "pg", None),
    Probe("pcg", "pg", None),
    # --- fraction --------------------------------------------------------
    Probe("%", "%", None),
    Probe("percent", "%", None),
    Probe("L/L", None, "SI haematocrit as a fraction; needs a dimensionless dimension"),
    Probe("ratio", None, "same"),
)


@dataclass(frozen=True)
class ProbeResult:
    probe: Probe
    status: str
    resolved: Optional[str]
    #: True when the spelling resolves only after ``score.repair_unit``.
    via_repair: bool
    #: Set when it resolved to something other than what it denotes -- a
    #: worse outcome than rejection, because it is silently wrong.
    misresolved: bool


def run() -> List[ProbeResult]:
    """Probe every spelling against the live alias table."""
    from units import UnitError, parse_unit

    from chemoguard.extraction_bench.score import repair_unit

    results = []

    for probe in PROBES:
        resolved = None
        via_repair = False

        try:
            resolved = parse_unit(probe.text).symbol
        except UnitError:
            repaired = repair_unit(probe.text)
            if repaired != probe.text:
                try:
                    resolved = parse_unit(repaired).symbol
                    via_repair = True
                except UnitError:
                    resolved = None

        if probe.expected is None:
            status = NEEDS_DIMENSION if resolved is None else ACCEPTED
        elif resolved == probe.expected and not via_repair:
            status = ACCEPTED
        else:
            status = NEEDS_ALIAS

        results.append(
            ProbeResult(
                probe=probe,
                status=status,
                resolved=resolved,
                via_repair=via_repair,
                misresolved=(
                    resolved is not None
                    and probe.expected is not None
                    and resolved != probe.expected
                ),
            )
        )

    return results


def summary(results) -> dict:
    """Counts by status, plus the two lists worth reading."""
    counts = {ACCEPTED: 0, NEEDS_ALIAS: 0, NEEDS_DIMENSION: 0}
    for result in results:
        counts[result.status] += 1

    return {
        "counts": counts,
        "total": len(results),
        "needs_alias": [r.probe.text for r in results if r.status == NEEDS_ALIAS],
        "needs_dimension": [
            r.probe.text for r in results if r.status == NEEDS_DIMENSION
        ],
        "misresolved": [
            (r.probe.text, r.probe.expected, r.resolved)
            for r in results
            if r.misresolved
        ],
    }
