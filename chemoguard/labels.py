"""WHO anaemia severity, and morphological subtype from MCV.

Thresholds are WHO's **2024** guideline, *Haemoglobin cutoffs to define
anaemia in individuals and populations* (WHO, March 2024, ISBN
978-92-4-008854-2), Table 3 "Haemoglobin cutoffs to define anaemia severity
in individuals". Retrieved and transcribed 2026-09-21.

They are declared below **in g/L, the unit WHO states them in**, and
converted to the canonical g/dL by ``Quantity.to``. That is deliberate: a
hand-converted table is a transcription error waiting to happen, and the
whole point of the boundary in ``units.py`` is that this application does not
do arithmetic on laboratory values in its head.

What changed in 2024, against the 2011 table this replaces
-----------------------------------------------------------

The adult rows did **not** change. Men 15-65 and non-pregnant women 15-65
carry exactly the 2011 bands. Three things around them did:

* **Children 6-59 months was split.** 6-23 months now has its own lower
  cutoff (105 g/L, from 110), on evidence that haemoglobin is physiologically
  lower in the second year of life. 24-59 months keeps 110 g/L.
* **Pregnancy became trimester-specific**: 110 / 105 / 110 g/L for the first,
  second and third trimesters, where 2011 gave a single 110 g/L.
* **The adult rows are explicitly scoped to ages 15-65.** WHO lists
  haemoglobin distributions in adults over 65 as a research gap: "Future
  studies should prospectively define 5th percentiles in healthy populations
  across all geographical regions, including infants, and men and women aged
  >65 years." There is no WHO cutoff for this group.

WHO also now recommends adjusting haemoglobin for **elevation of residence**
and for **smoking** before applying these cutoffs. Neither adjustment is
applied here -- see ``UNADJUSTED_FOR`` below, which is not a detail.

Three stated decisions
----------------------

Each of these excludes rows. None is a side effect of an adult-only
implementation; each is a choice with a reason, and each is counted in
``LabelReport`` so the cost is visible rather than inferred.

1. **Over-65s are not labelled by default** (3,707 of 19,749 NHANES rows,
   18.8%). Applying the 15-65 cutoff to a 78-year-old is using a threshold
   outside the range its own source scopes it to, and it would be invisible
   downstream. Pass ``allow_over_65=True`` to label them anyway; they are
   then marked ``extrapolated`` so a model card can say so.
2. **Pregnant participants are not labelled** (113 rows, 0.57%). The 2024
   cutoffs are trimester-specific and NHANES publishes ``RIDEXPRG`` without
   a trimester, so choosing a cutoff would mean inventing one -- and the
   choice moves the boundary by 5 g/L, which is half a severity band. Rows
   where pregnancy could not be ascertained (``RIDEXPRG == 3``) are excluded
   for the same reason.
3. **Women of child-bearing age with no pregnancy record are treated as not
   pregnant** (3,577 rows). NHANES only ascertains pregnancy over a limited
   age range, so a null is mostly "not asked", not "unknown but plausible".
   This is an assumption, not a measurement, and it is reported as one.

Morphological subtype is a separate, independent label: MCV < 80 fL
microcytic, 80-100 fL normocytic, > 100 fL macrocytic. It does not read
haemoglobin, so it survives the Hb hold-out that the anaemia task needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from units import FEMTOLITRE, G_PER_DL, G_PER_L, Quantity

#: Adjustments WHO 2024 recommends that are NOT applied to these labels.
#:
#: Both push the same way. Haemoglobin rises with altitude and with smoking,
#: so leaving both unadjusted makes a genuinely anaemic person at altitude, or
#: a genuinely anaemic smoker, look less anaemic than they are. Every error
#: this introduces points from abnormal toward normal, which is the direction
#: CLAUDE.md Finding 4 is about.
#:
#: Neither is fixable from the public NHANES release as currently joined:
#: elevation of residence is restricted-use geographic data, and smoking is in
#: the SMQ questionnaire file, which the loader does not read.
UNADJUSTED_FOR = ("elevation of residence", "smoking")


class Severity(Enum):
    """WHO anaemia severity. Ordered, so comparisons are meaningful."""

    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class Morphology(Enum):
    """Red cell size class, from MCV alone."""

    MICROCYTIC = "microcytic"
    NORMOCYTIC = "normocytic"
    MACROCYTIC = "macrocytic"


@dataclass(frozen=True)
class SeverityBand:
    """One row of WHO 2024 Table 3.

    Three cutoffs describe the whole row, because the bands are contiguous.
    WHO writes men as ">=130 / 110-129 / 80-109 / <80"; that is the same as
    cutoffs at 130, 110 and 80 with each band closed below and open above.
    Haemoglobin is reported to one decimal in g/dL, which is whole numbers in
    g/L, so "110-129" and "[110, 130)" select the same values.
    """

    key: str
    label: str
    #: At or above this, not anaemic.
    anaemia: Quantity
    #: At or above this (and below ``anaemia``), mild.
    mild: Quantity
    #: At or above this (and below ``mild``), moderate. Below it, severe.
    moderate: Quantity
    #: The WHO table row this came from, quoted, so it is checkable.
    source_row: str = ""

    def classify(self, hb: Quantity) -> Severity:
        """Severity for one haemoglobin Quantity.

        Comparison happens in g/dL for all four operands, so the unit the
        caller happens to hold cannot change the answer.
        """
        value = hb.magnitude_in(G_PER_DL)
        if value >= self.anaemia.magnitude_in(G_PER_DL):
            return Severity.NONE
        if value >= self.mild.magnitude_in(G_PER_DL):
            return Severity.MILD
        if value >= self.moderate.magnitude_in(G_PER_DL):
            return Severity.MODERATE
        return Severity.SEVERE

    def as_dict(self) -> dict:
        return {
            "band": self.key,
            "label": self.label,
            "anaemia_below": self.anaemia.as_dict(),
            "mild_from": self.mild.as_dict(),
            "moderate_from": self.moderate.as_dict(),
            "source_row": self.source_row,
        }


def _band(key, label, anaemia, mild, moderate, source_row):
    """Build a band from WHO's own numbers, in WHO's own unit (g/L)."""
    return SeverityBand(
        key, label,
        Quantity(anaemia, G_PER_L),
        Quantity(mild, G_PER_L),
        Quantity(moderate, G_PER_L),
        source_row,
    )


#: WHO 2024, Table 3, transcribed in g/L exactly as printed.
WHO_2024_BANDS = {
    b.key: b
    for b in (
        _band("children_6_23_months", "Children, 6-23 months",
              105, 95, 70, ">=105 | 95-104 | 70-94 | <70"),
        _band("children_24_59_months", "Children, 24-59 months",
              110, 100, 70, ">=110 | 100-109 | 70-99 | <70"),
        _band("children_5_11_years", "Children, 5-11 years",
              115, 110, 80, ">=115 | 110-114 | 80-109 | <80"),
        # WHO prints boys and non-pregnant girls as separate rows with
        # identical numbers. One band here; the distinction only matters for
        # a pregnant 12-14-year-old, who takes a pregnancy band instead.
        _band("children_12_14_years", "Children, 12-14 years",
              120, 110, 80, ">=120 | 110-119 | 80-109 | <80"),
        _band("adult_female", "Adults, 15-65 years, nonpregnant women",
              120, 110, 80, ">=120 | 110-119 | 80-109 | <80"),
        _band("adult_male", "Adults, 15-65 years, men",
              130, 110, 80, ">=130 | 110-129 | 80-109 | <80"),
        _band("pregnant_first_trimester", "Pregnancy, first trimester",
              110, 100, 70, ">=110 | 100-109 | 70-99 | <70"),
        _band("pregnant_second_trimester", "Pregnancy, second trimester",
              105, 95, 70, ">=105 | 95-104 | 70-94 | <70"),
        _band("pregnant_third_trimester", "Pregnancy, third trimester",
              110, 100, 70, ">=110 | 100-109 | 70-99 | <70"),
    )
}

#: The oldest age WHO's adult rows are scoped to.
WHO_ADULT_MAX_AGE_YEARS = 65
#: The youngest age any WHO band covers, in months.
WHO_MIN_AGE_MONTHS = 6


class Unlabelled(Enum):
    """Why a row carries no WHO severity. Never silently dropped."""

    OVER_65 = "no WHO cutoff above 65 years"
    UNDER_6_MONTHS = "no WHO cutoff below 6 months"
    PREGNANT_TRIMESTER_UNKNOWN = "pregnant, trimester not reported by NHANES"
    PREGNANCY_UNASCERTAINED = "pregnancy status could not be ascertained"
    NO_HAEMOGLOBIN = "no haemoglobin measurement"
    NO_AGE = "no age"


def band_for(age_years, sex, pregnant=False, pregnancy_ascertained=True,
             allow_over_65=False):
    """The WHO band that applies, or an ``Unlabelled`` reason.

    ``age_years`` is whole years, as NHANES reports it (``RIDAGEYR``). The
    whole-year bands line up exactly with WHO's month bands: an age of 1 is
    12-23 months, entirely inside WHO's 6-23 month row, and ages 2-4 are
    24-59 months.

    Returns ``(band, extrapolated)`` on success, or ``(reason, False)``.
    """
    if age_years is None or age_years != age_years:
        return Unlabelled.NO_AGE, False

    age = int(age_years)

    if pregnant:
        return Unlabelled.PREGNANT_TRIMESTER_UNKNOWN, False
    if not pregnancy_ascertained:
        return Unlabelled.PREGNANCY_UNASCERTAINED, False

    if age < 1:
        # A whole-year age of 0 spans 0-11 months and so straddles WHO's
        # lower limit. NHANES does not run a CBC below age 1, so this is
        # defensive rather than load-bearing.
        return Unlabelled.UNDER_6_MONTHS, False
    if age == 1:
        return WHO_2024_BANDS["children_6_23_months"], False
    if age <= 4:
        return WHO_2024_BANDS["children_24_59_months"], False
    if age <= 11:
        return WHO_2024_BANDS["children_5_11_years"], False
    if age <= 14:
        return WHO_2024_BANDS["children_12_14_years"], False

    if age > WHO_ADULT_MAX_AGE_YEARS and not allow_over_65:
        return Unlabelled.OVER_65, False

    extrapolated = age > WHO_ADULT_MAX_AGE_YEARS
    if sex == "male":
        return WHO_2024_BANDS["adult_male"], extrapolated
    if sex == "female":
        return WHO_2024_BANDS["adult_female"], extrapolated
    # Sex is required for the adult bands and they differ by 10 g/L. No
    # default: guessing here is a one-band error for half the population.
    return Unlabelled.NO_AGE, False


def severity_for(hb, age_years, sex, pregnant=False,
                 pregnancy_ascertained=True, allow_over_65=False):
    """WHO severity for one participant, or an ``Unlabelled`` reason.

    ``hb`` is a ``Quantity``. Passing a bare number is a TypeError, not a
    guess -- a haemoglobin of "12" is 12 g/dL or 12 g/L depending on nothing
    the caller has told us, and those are different diagnoses.
    """
    if not isinstance(hb, Quantity):
        raise TypeError(
            "severity_for needs a Quantity for haemoglobin, got {0}. Build it "
            "with Quantity.of(value, 'g/dL').".format(type(hb).__name__)
        )
    band, _ = band_for(age_years, sex, pregnant, pregnancy_ascertained,
                       allow_over_65)
    if isinstance(band, Unlabelled):
        return band
    return band.classify(hb)


# --- morphological subtype --------------------------------------------------

#: MCV band edges, as Quantities in the unit they are defined in.
#:
#: Microcytic below 80 fL, normocytic 80-100 fL inclusive at both ends,
#: macrocytic above 100 fL. These are the conventional adult cut-points. They
#: are *not* age-adjusted, and MCV genuinely is lower in young children --
#: roughly 70-86 fL at age 1-2 -- so a healthy toddler classifies as
#: microcytic here. Recorded rather than corrected, because there is no
#: single agreed paediatric MCV table to correct against; see
#: ``MORPHOLOGY_CAVEAT``.
MICROCYTIC_BELOW = Quantity(80.0, FEMTOLITRE)
MACROCYTIC_ABOVE = Quantity(100.0, FEMTOLITRE)

MORPHOLOGY_CAVEAT = (
    "MCV bands are adult cut-points unless an age is supplied. MCV is "
    "physiologically lower in early childhood, so a toddler scored against "
    "the adult 80 fL is over-called microcytic."
)

#: Age-specific lower limit of normal for MCV, in fL.
#:
#: ``LLN = 70 + age in years`` for children under 10, the rule in general
#: paediatric use, which reaches the adult 80 fL at age 10 and so joins the
#: adult range without a step.
#:
#: Koerper & Dallman (*Developmental change in red blood cell volume:
#: implication in screening infants and children for iron deficiency and
#: thalassemia trait*, J Pediatr 1976; PMID 956999) give a coarser measured
#: table -- 70 fL at 10-17 months, 74 between 1.5 and 4 years, 76 between 4
#: and 7 years -- which the formula tracks to within about 2 fL. Where they
#: differ the formula is the more conservative (lower) bound in the second
#: year and the less conservative one at 2-3 years. Recorded because a 2 fL
#: disagreement at these ages is roughly a quarter of the gap between the
#: paediatric and adult cut-points.
#:
#: **No age-specific upper limit is applied.** The published paediatric
#: upper bounds are less consistently reported, and the direction does not
#: matter for iron deficiency: young children's MCV is displaced downward,
#: so the macrocytic call is not the one at risk.
PAEDIATRIC_MCV_RULE = "LLN = 70 + age(years), for age < 10"
ADULT_MCV_LLN_AGE = 10


def mcv_lower_limit(age_years=None) -> Quantity:
    """The lower limit of normal for MCV at a given age.

    Returns the adult 80 fL when ``age_years`` is None -- which is the
    current behaviour of every caller that does not pass an age, so adding
    the parameter changes nothing until someone uses it.
    """
    if age_years is None or age_years != age_years:
        return MICROCYTIC_BELOW
    age = int(age_years)
    if age >= ADULT_MCV_LLN_AGE:
        return MICROCYTIC_BELOW
    return Quantity(70.0 + max(age, 0), FEMTOLITRE)


def morphology_for(mcv, age_years=None):
    """Red cell size class from an MCV ``Quantity``.

    With ``age_years``, the microcytic cut-point is age-specific; without
    it, the adult 80 fL applies and ``MORPHOLOGY_CAVEAT`` holds.
    """
    if not isinstance(mcv, Quantity):
        raise TypeError(
            "morphology_for needs a Quantity for MCV, got {0}.".format(
                type(mcv).__name__)
        )
    value = mcv.magnitude_in(FEMTOLITRE)
    if value < mcv_lower_limit(age_years).magnitude_in(FEMTOLITRE):
        return Morphology.MICROCYTIC
    if value > MACROCYTIC_ABOVE.magnitude_in(FEMTOLITRE):
        return Morphology.MACROCYTIC
    return Morphology.NORMOCYTIC


# --- vectorised labelling ---------------------------------------------------


@dataclass(frozen=True)
class LabelReport:
    """Labels for a dataset, and an accounting of every row that has none."""

    severity: object
    morphology: object
    band: object
    extrapolated: object
    unlabelled_reasons: object
    counts: dict = field(default_factory=dict)
    assumptions: tuple = field(default_factory=tuple)

    @property
    def labelled(self):
        return int(self.severity.notna().sum())


def label_dataset(dataset, allow_over_65=False):
    """Label a ``Dataset`` from ``chemoguard.data``.

    Reads ``dataset.values`` (canonical units, guaranteed by the loader) and
    ``dataset.meta`` (age, sex, pregnancy). Returns a ``LabelReport`` whose
    ``counts`` says how many rows each exclusion cost.

    Thresholds are converted from WHO's g/L to the canonical g/dL once, via
    ``Quantity``, and then compared against the whole column. No value is
    rescaled by a literal.
    """
    import numpy as np
    import pandas as pd

    values, meta = dataset.values, dataset.meta
    if meta is None:
        raise ValueError(
            "{0} carries no demographics, so WHO bands cannot be selected. "
            "The bands differ by sex and by age.".format(dataset.name)
        )

    declared = (dataset.units or {}).get("hb")
    if declared and declared != G_PER_DL.symbol:
        raise ValueError(
            "haemoglobin column is in {0}, expected the canonical {1}.".format(
                declared, G_PER_DL.symbol)
        )

    hb = values["hb"]
    age = meta["age_years"]
    sex = meta["sex"]
    pregnancy = meta.get("pregnancy_code")
    if pregnancy is None:
        pregnancy = pd.Series(np.nan, index=meta.index)

    severity = pd.Series(pd.NA, index=values.index, dtype="object")
    band_key = pd.Series(pd.NA, index=values.index, dtype="object")
    extrapolated = pd.Series(False, index=values.index)
    reasons = pd.Series(pd.NA, index=values.index, dtype="object")

    pregnant = pregnancy == 1
    unascertained = pregnancy == 3

    # Row-wise: the band depends on three fields and there are nine bands.
    # 19,749 rows is a fraction of a second, and a vectorised version of this
    # would be nine masks that have to stay mutually exclusive by hand.
    for index, age_value, sex_value, is_pregnant, is_unsure, hb_value in zip(
        values.index, age, sex, pregnant, unascertained, hb
    ):
        band, is_extrapolated = band_for(
            age_value, sex_value, bool(is_pregnant), not bool(is_unsure),
            allow_over_65,
        )
        if isinstance(band, Unlabelled):
            reasons.at[index] = band.value
            continue
        if hb_value != hb_value:
            reasons.at[index] = Unlabelled.NO_HAEMOGLOBIN.value
            continue
        severity.at[index] = band.classify(Quantity(hb_value, G_PER_DL)).value
        band_key.at[index] = band.key
        extrapolated.at[index] = is_extrapolated

    mcv = values.get("mcv")
    if mcv is None:
        morphology = pd.Series(pd.NA, index=values.index, dtype="object")
    else:
        # Thresholds converted once, through Quantity, then applied to the
        # whole column.
        low = MICROCYTIC_BELOW.magnitude_in(dataset.units.get("mcv", "fL"))
        high = MACROCYTIC_ABOVE.magnitude_in(dataset.units.get("mcv", "fL"))
        morphology = pd.Series(
            np.where(mcv.isna(), None,
                     np.where(mcv < low, Morphology.MICROCYTIC.value,
                              np.where(mcv > high, Morphology.MACROCYTIC.value,
                                       Morphology.NORMOCYTIC.value))),
            index=values.index, dtype="object",
        )

    counts = {"rows": int(len(values)), "labelled": int(severity.notna().sum())}
    for reason in Unlabelled:
        n = int((reasons == reason.value).sum())
        if n:
            counts[reason.name.lower()] = n
    counts["extrapolated_over_65"] = int(extrapolated.sum())

    assumed_not_pregnant = int(
        ((sex == "female") & age.between(15, WHO_ADULT_MAX_AGE_YEARS)
         & pregnancy.isna()).sum()
    )

    return LabelReport(
        severity=severity,
        morphology=morphology,
        band=band_key,
        extrapolated=extrapolated,
        unlabelled_reasons=reasons,
        counts=counts,
        assumptions=(
            "{0} women aged 15-{1} have no NHANES pregnancy record and are "
            "treated as not pregnant".format(
                assumed_not_pregnant, WHO_ADULT_MAX_AGE_YEARS),
            "haemoglobin is NOT adjusted for {0}; both adjustments would "
            "raise the anaemia threshold, so unadjusted labels under-call "
            "anaemia".format(" or ".join(UNADJUSTED_FOR)),
            MORPHOLOGY_CAVEAT,
        ),
    )
