"""Reference-range flags for the four cytopenias, graded CTCAE v5.0.

These are threshold comparisons against published grading criteria. They are
not a model and not a diagnosis: they say "this number is below this
published cut-off", which a clinician can verify by eye in a second. They
exist for three reasons.

1. The regression suite needs an assertion about severity that does not
   depend on the quarantined estimators (CLAUDE.md Finding 5, models/ is
   empty and expected to stay so until the NHANES artefacts land).
2. A flag is the only severity signal the application can honestly offer
   while there is no model. It is stated as a threshold comparison, never as
   a prediction.
3. They corroborate the model when there is one. A model returning Low
   against a grade 4 flag is a unit bug, and the two disagreeing is worth
   surfacing rather than hiding.

Grades follow CTCAE v5.0 (November 2017), Blood and lymphatic system
disorders / Investigations. CTCAE states counts in 10^9/L, which is exactly
thousand/uL, and haemoglobin in g/dL.

Limitation, stated rather than papered over: the haemoglobin lower limit of
normal is sex-specific (WHO: 13.0 g/dL male, 12.0 g/dL female). This
application records no sex, so 12.0 is used throughout and anaemia is
under-flagged in men between 12.0 and 13.0 g/dL. Sex-adjusted severity is on
the roadmap in CLAUDE.md under Direction.

Assistive only. A flag is a prompt to look, not a clinical decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from cbc import CBCPanel
from units import Quantity

#: Lower limits of normal, in the unit each criterion is written in.
#: CTCAE writes grade 1 as "<LLN - <first cut-off>", leaving LLN to the
#: institution. These are conventional adult values.
LLN = {
    "wbc": 4.0,          # thousand/uL
    "anc": 1.8,          # thousand/uL
    "platelets": 150.0,  # thousand/uL
    "hb": 12.0,          # g/dL -- sex-neutral, see module docstring
}

SEVERITY_BY_GRADE = {
    0: "none",
    1: "mild",
    2: "moderate",
    3: "severe",
    4: "life-threatening",
}

#: A grade at or above this is "severe" for reporting purposes.
SEVERE_GRADE = 3


@dataclass(frozen=True)
class Flag:
    """One analyte graded against its published cut-offs."""

    analyte: str
    condition: str
    grade: int
    severity: str
    measured: Quantity
    #: The value as compared, in the unit the CTCAE criterion is written in.
    compared_value: float
    compared_unit: str
    description: str

    @property
    def severe(self) -> bool:
        return self.grade >= SEVERE_GRADE

    def as_dict(self) -> dict:
        return {
            "analyte": self.analyte,
            "condition": self.condition,
            "grade": self.grade,
            "severity": self.severity,
            "severe": self.severe,
            "measured": self.measured.as_dict(),
            "compared_as": {"value": self.compared_value, "unit": self.compared_unit},
            "description": self.description,
            "criteria": "CTCAE v5.0",
        }


def _grade(value: float, lln: float, g2: float, g3: float, g4) -> int:
    """Grade a decreasing-is-worse analyte against descending cut-offs.

    At or above ``lln`` is grade 0, and each further cut-off steps the grade
    up by one. ``g4`` may be None, for a criterion whose grade 4 is defined
    clinically rather than numerically -- haemoglobin, where CTCAE grade 4 is
    "life-threatening consequences; urgent intervention indicated" and no
    threshold is published. Inventing a number there would be exactly the
    kind of quiet fabrication this change is removing, so such an analyte
    tops out at grade 3.
    """
    if value >= lln:
        return 0
    if value >= g2:
        return 1
    if value >= g3:
        return 2
    if g4 is None or value >= g4:
        return 3
    return 4


# Each entry: analyte key, condition name, the unit the criterion is written
# in, and the descending cut-offs (LLN, then the floor of grades 1, 2 and 3).
# CTCAE v5.0, verbatim:
#   WBC decreased        G1 <LLN-3000  G2 <3000-2000  G3 <2000-1000  G4 <1000
#   Neutrophil decreased G1 <LLN-1500  G2 <1500-1000  G3 <1000-500   G4 <500
#   Platelet decreased   G1 <LLN-75000 G2 <75000-50000 G3 <50000-25000 G4 <25000
#   Anemia (Hb)          G1 <LLN-10.0  G2 <10.0-8.0   G3 <8.0        G4 clinical
_CRITERIA = (
    ("wbc", "leukopenia", "thousand/uL", (LLN["wbc"], 3.0, 2.0, 1.0)),
    ("anc", "neutropenia", "thousand/uL", (LLN["anc"], 1.5, 1.0, 0.5)),
    ("platelets", "thrombocytopenia", "thousand/uL", (LLN["platelets"], 75.0, 50.0, 25.0)),
    ("hb", "anemia", "g/dL", (LLN["hb"], 10.0, 8.0, None)),
)


def flag_for(analyte: str, quantity: Quantity) -> Flag:
    """Grade a single analyte. Raises KeyError if it has no criteria."""
    for key, condition, unit, cutoffs in _CRITERIA:
        if key != analyte:
            continue
        value = quantity.magnitude_in(unit)
        grade = _grade(value, *cutoffs)
        severity = SEVERITY_BY_GRADE[grade]
        if grade == 0:
            description = "{0:g} {1} is at or above the lower limit of normal ({2:g} {1}).".format(
                value, unit, cutoffs[0]
            )
        else:
            description = "{0} {1}: {2:g} {3} (CTCAE grade {4}).".format(
                severity.capitalize(), condition, value, unit, grade
            )
        return Flag(
            analyte=analyte,
            condition=condition,
            grade=grade,
            severity=severity,
            measured=quantity,
            compared_value=value,
            compared_unit=unit,
            description=description,
        )
    raise KeyError("no CTCAE criteria defined for {0!r}".format(analyte))


def flags_for(panel: CBCPanel) -> list:
    """Grade every analyte in the panel that has published criteria.

    ANC is graded only when it is present; a panel with no ANC and no
    neutrophil percentage simply carries no neutropenia flag, rather than one
    computed from a substitute.
    """
    return [
        flag_for(key, panel.get(key))
        for key, _condition, _unit, _cutoffs in _CRITERIA
        if panel.get(key) is not None
    ]


def severe_flags(panel: CBCPanel) -> list:
    """Only the grade 3 and grade 4 flags."""
    return [flag for flag in flags_for(panel) if flag.severe]
