"""Iron deficiency from serum ferritin: the target that is not in the CBC.

Finding 8 killed the anaemia target: WHO anaemia severity is a threshold on
haemoglobin, and haemoglobin is algebraically recoverable from the rest of a
CBC panel, so holding it out holds out nothing. Ferritin has no such problem.
No CBC identity produces it, it is measured on a separate assay from a
separate aliquot, and the CBC's relationship to it is statistical -- which is
the definition of a learnable task rather than an arithmetic one.

It is also the question the red-cell indices were invented to answer. Mentzer
(1973) and Green & King (1989) exist to separate iron deficiency from
thalassaemia trait in a microcytic patient. Predicting iron status from a CBC
is what those features are *for*.

Thresholds
----------

WHO 2020, *Serum ferritin concentrations for the assessment of iron status in
individuals and populations: technical brief* (WHO, 2020), Table 1, which
reproduces the recommendations of the *WHO guideline on use of ferritin
concentrations to assess iron status in individuals and populations* (2020,
https://apps.who.int/iris/handle/10665/331505). Transcribed 2026-09-21.

**Unlike the anaemia guideline, there is no upper age gap.** WHO prints an
explicit "Older persons (60+ years)" row at the same <15 ug/L as adults. The
over-65 exclusion that Finding 8's labels needed does not apply here, and the
3,707 rows it cost come back -- to the extent NHANES measured their ferritin,
which is a separate and larger problem. See ``nhanes.FERRITIN_ELIGIBILITY_WARNING``.

Inflammation
------------

Ferritin is an acute-phase reactant: it rises with inflammation independently
of iron stores, so a fixed threshold under-calls deficiency in inflamed
people. WHO gives four ways to handle this:

1. raise the cut-off for inflamed individuals (<30 / <70 ug/L by age);
2. exclude individuals with CRP > 5 mg/L or AGP > 1 g/L;
3. an arithmetic correction by inflammation group;
4. a linear regression (BRINDA) correction on CRP, AGP and malaria.

**Of these, only (1) is available here -- and it is an explicit flag, not a
default.** ``adjust_for_inflammation`` is keyword-only and required, and
both prevalences are reported whichever way it is set. Neither number is
authoritative: WHO states plainly that "the application of different
adjustment approaches will result in a high degree of variability in the
estimated prevalence of depleted iron stores", and the method that would
settle it (BRINDA) needs AGP, which NHANES does not measure. Quoting one
number alone would overstate what this project knows.

Why (1) rather than (3) or (4):

* 3 and 4 both require AGP, which NHANES does not measure. A CRP-only BRINDA
  regression is possible but is no longer the published method, and its
  coefficients would have to be fitted here rather than cited.
* 2 discards the inflamed rather than classifying them, and inflammation is
  common -- it would bias the cohort toward the healthy in a way that a model
  trained on it would inherit silently.
* 1 is a WHO recommendation stated as a threshold table, needs only an
  inflammation indicator, and NHANES measures high-sensitivity CRP on
  essentially the whole examined sample.

Inflammation is CRP > 5 mg/L, which is WHO's own worked example in the same
document and the BRINDA reference value.

The cost is that this is a step function where the biology is continuous: a
CRP of 4.9 and a CRP of 5.1 get thresholds 4.7x apart. ``label_iron_status``
reports prevalence under both the raw and the inflammation-adjusted rule so
the size of that effect is visible rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from units import MICROGRAM_PER_L, MILLIGRAM_PER_L, Quantity


class IronStatus(Enum):
    """Binary iron status. Ordered so "deficient" is the positive class."""

    REPLETE = "replete"
    DEFICIENT = "deficient"


class UnlabelledIron(Enum):
    """Why a row carries no iron-status label."""

    NO_FERRITIN = "ferritin not measured (NHANES eligibility subsample)"
    NO_CRP = "no CRP, so inflammation status is unknown"
    PREGNANT = "pregnant, and WHO gives a threshold for the first trimester only"
    PREGNANCY_UNASCERTAINED = "pregnancy status could not be ascertained"
    NO_AGE = "no age"


@dataclass(frozen=True)
class FerritinBand:
    """One row of WHO 2020 Table 1."""

    key: str
    label: str
    #: Below this, iron deficient, in an apparently healthy individual.
    healthy: Quantity
    #: Below this, iron deficient, in the presence of infection/inflammation.
    inflamed: Quantity
    source_row: str = ""

    def threshold(self, inflamed: bool) -> Quantity:
        return self.inflamed if inflamed else self.healthy

    def classify(self, ferritin: Quantity, inflamed: bool) -> IronStatus:
        cutoff = self.threshold(inflamed).magnitude_in(MICROGRAM_PER_L)
        if ferritin.magnitude_in(MICROGRAM_PER_L) < cutoff:
            return IronStatus.DEFICIENT
        return IronStatus.REPLETE

    def as_dict(self) -> dict:
        return {
            "band": self.key,
            "label": self.label,
            "healthy_below": self.healthy.as_dict(),
            "inflamed_below": self.inflamed.as_dict(),
            "source_row": self.source_row,
        }


def _band(key, label, healthy, inflamed, source_row):
    return FerritinBand(
        key, label,
        Quantity(healthy, MICROGRAM_PER_L),
        Quantity(inflamed, MICROGRAM_PER_L),
        source_row,
    )


#: WHO 2020 Table 1, transcribed in ug/L exactly as printed.
#:
#: The numbers collapse to two tiers -- 12/30 below five years, 15/70 from
#: five up -- but all six rows are written out because that is how WHO prints
#: them and because a future revision is more likely to split a row than to
#: change both tiers.
WHO_2020_FERRITIN_BANDS = {
    b.key: b
    for b in (
        _band("infants_0_23_months", "Infants and young children (0-23 months)",
              12, 30, "<12 healthy | <30 inflamed"),
        _band("children_24_59_months", "Children under 5 years (24-59 months)",
              12, 30, "<12 healthy | <30 inflamed"),
        _band("children_5_9_years", "Children (5 to less than 10 years)",
              15, 70, "<15 healthy | <70 inflamed"),
        _band("adolescents_10_19_years", "Adolescents (10 to less than 20 years)",
              15, 70, "<15 healthy | <70 inflamed"),
        _band("adults_20_59_years", "Adults (20-59 years)",
              15, 70, "<15 healthy | <70 inflamed"),
        # The row the anaemia guideline does not have.
        _band("older_persons_60_plus", "Older persons (60+ years)",
              15, 70, "<15 healthy | <70 inflamed"),
    )
}

#: Inflammation, as WHO's own worked example defines it.
INFLAMMATION_CRP_THRESHOLD = Quantity(5.0, MILLIGRAM_PER_L)

#: Soluble transferrin receptor cut-off, for the **sensitivity analysis
#: only**. This is not a WHO value and must not be presented as one.
#:
#: WHO declines to set an sTfR threshold because the assays are not
#: standardised against a common reference, and the published cut-offs bear
#: that out: one comparative study puts them between 2.48 and 6.70 mg/L
#: depending on method. 6.70 is the figure for the particle-enhanced
#: immunoturbidimetric Roche assay, which is the one NHANES runs (stated in
#: both the TFR_L and P_TFR codebooks), so it is the only one of the range
#: that applies to this data.
#:
#: Using it costs more than half the cohort: sTfR is a smaller subsample than
#: ferritin in 2017-2020. Hence ferritin alone is the primary definition and
#: this is reported beside it, never instead of it.
STFR_ID_THRESHOLD = Quantity(6.70, MILLIGRAM_PER_L)
STFR_CAVEAT = (
    "sTfR cut-offs are assay-specific and WHO sets none; 6.70 mg/L is the "
    "Roche immunoturbidimetric figure, matching the NHANES assay. Sensitivity "
    "analysis only."
)

#: The two target definitions this module can build.
DEFINITIONS = ("ferritin", "ferritin-or-stfr")

#: The cycles models are trained on. **2017-2020 only.**
#:
#: Not a sample-size decision -- pooling would add 1,945 rows -- but a
#: validity one. NHANES ran ferritin on "1-5 years old and 12-49 years old
#: females" in 2021-2023 and on "1-5 years old and 12 years and older" in
#: 2017-2020, so the pooled cohort's sex and age distribution is partly a
#: function of which release a row came from. Sex and age are *features of
#: this model*, so that confound lands directly on the inputs rather than
#: staying in the sampling frame where it could be weighted away.
#:
#: The pooled fit is kept as a declared sensitivity analysis, never as the
#: headline; ``scripts/train_iron.py --cycles pooled`` runs it.
TRAINING_CYCLES = ("2017-2020",)
SENSITIVITY_CYCLES = ("2017-2020", "2021-2023")


def band_for_age(age_years):
    """The WHO 2020 ferritin band for an age in whole years.

    NHANES reports whole years, and WHO's bands fall on whole-year
    boundaries except the first, where 0-23 months is ages 0 and 1.
    """
    if age_years is None or age_years != age_years:
        return UnlabelledIron.NO_AGE

    age = int(age_years)
    if age <= 1:
        return WHO_2020_FERRITIN_BANDS["infants_0_23_months"]
    if age <= 4:
        return WHO_2020_FERRITIN_BANDS["children_24_59_months"]
    if age <= 9:
        return WHO_2020_FERRITIN_BANDS["children_5_9_years"]
    if age <= 19:
        return WHO_2020_FERRITIN_BANDS["adolescents_10_19_years"]
    if age <= 59:
        return WHO_2020_FERRITIN_BANDS["adults_20_59_years"]
    return WHO_2020_FERRITIN_BANDS["older_persons_60_plus"]


def is_inflamed(crp):
    """Whether CRP indicates inflammation. ``None`` if CRP is absent.

    Absent is not "not inflamed". Assuming the healthy threshold for someone
    whose inflammation status is unknown applies the *lower* cut-off, which
    under-calls deficiency -- an error pointing from abnormal to normal, the
    direction Finding 4 is about. So this returns None and the caller
    excludes the row.
    """
    if crp is None:
        return None
    if not isinstance(crp, Quantity):
        raise TypeError(
            "is_inflamed needs a Quantity for CRP, got {0}.".format(
                type(crp).__name__)
        )
    threshold = INFLAMMATION_CRP_THRESHOLD.magnitude_in(MILLIGRAM_PER_L)
    return crp.magnitude_in(MILLIGRAM_PER_L) > threshold


def iron_status_for(ferritin, age_years, crp, *, adjust_for_inflammation,
                    pregnant=False, pregnancy_ascertained=True, stfr=None,
                    definition="ferritin"):
    """Iron status for one participant, or an ``UnlabelledIron`` reason.

    ``adjust_for_inflammation`` is keyword-only and has no default. Neither
    setting is authoritative -- WHO warns that adjustment methods vary
    widely and the BRINDA regression is unavailable without AGP -- so the
    choice is the caller's to make and to state, never one this module makes
    quietly. Reporting one number alone overstates what is known.

    **CRP is required either way.** When ``adjust_for_inflammation`` is
    False the healthy threshold is applied to everyone, but a row with no
    CRP is still excluded. That keeps the adjusted and unadjusted
    prevalences computed over exactly the same rows, so the difference
    between them measures the adjustment and nothing else.

    ``ferritin`` and ``crp`` are Quantities. A bare number raises: ferritin of
    "15" is 15 ug/L or 15 mg/L depending on nothing the caller has said, and
    those differ by a thousand-fold.
    """
    if ferritin is not None and not isinstance(ferritin, Quantity):
        raise TypeError(
            "iron_status_for needs a Quantity for ferritin, got {0}. Build it "
            "with Quantity.of(value, 'ug/L').".format(type(ferritin).__name__)
        )

    if pregnant:
        return UnlabelledIron.PREGNANT
    if not pregnancy_ascertained:
        return UnlabelledIron.PREGNANCY_UNASCERTAINED

    band = band_for_age(age_years)
    if isinstance(band, UnlabelledIron):
        return band
    if ferritin is None:
        return UnlabelledIron.NO_FERRITIN

    inflamed = is_inflamed(crp)
    if inflamed is None:
        # Required even when the adjustment is off, so that the two
        # prevalences are computed over identical rows.
        return UnlabelledIron.NO_CRP

    status = band.classify(ferritin, inflamed and adjust_for_inflammation)

    if definition == "ferritin-or-stfr" and status is IronStatus.REPLETE:
        if stfr is None:
            return UnlabelledIron.NO_FERRITIN
        if not isinstance(stfr, Quantity):
            raise TypeError("sTfR must be a Quantity")
        if stfr.magnitude_in(MILLIGRAM_PER_L) > STFR_ID_THRESHOLD.magnitude_in(
                MILLIGRAM_PER_L):
            return IronStatus.DEFICIENT
    return status


@dataclass(frozen=True)
class IronLabelReport:
    """Iron-status labels, the exclusions, and the class balance."""

    status: object
    band: object
    inflamed: object
    unlabelled_reasons: object
    counts: dict = field(default_factory=dict)
    prevalence: dict = field(default_factory=dict)
    assumptions: tuple = field(default_factory=tuple)

    @property
    def labelled(self):
        return int(self.status.notna().sum())

    @property
    def baseline(self):
        """Majority-class share among labelled rows -- the accuracy to beat."""
        if not self.labelled:
            return float("nan")
        counts = self.status.value_counts()
        return float(counts.max() / counts.sum())


def label_iron_status(dataset, *, adjust_for_inflammation,
                      definition="ferritin"):
    """Label a Dataset carrying ferritin, CRP and demographics.

    ``adjust_for_inflammation`` is keyword-only and required; see
    ``iron_status_for``. Whichever is chosen, ``prevalence`` carries *both*
    numbers, because quoting one alone overstates what is known.
    """
    import numpy as np
    import pandas as pd

    if definition not in DEFINITIONS:
        raise ValueError(
            "Unknown definition {0!r}. Known: {1}".format(
                definition, ", ".join(DEFINITIONS))
        )

    values, meta, units = dataset.values, dataset.meta, dataset.units or {}
    if meta is None:
        raise ValueError("{0} carries no demographics".format(dataset.name))
    for required in ("ferritin", "crp"):
        if required not in values.columns:
            raise KeyError(
                "{0} does not carry {1}; join the iron markers with "
                "nhanes.with_iron_markers().".format(dataset.name, required)
            )
        declared = units.get(required)
        canonical = {"ferritin": "ug/L", "crp": "mg/L"}[required]
        if declared and declared != canonical:
            raise ValueError(
                "{0} is in {1}, expected the canonical {2}.".format(
                    required, declared, canonical)
            )

    ferritin = values["ferritin"]
    crp = values["crp"]
    stfr = values["stfr"] if "stfr" in values.columns else None
    age = meta["age_years"]
    pregnancy = meta.get("pregnancy_code")
    if pregnancy is None:
        pregnancy = pd.Series(np.nan, index=meta.index)

    status = pd.Series(pd.NA, index=values.index, dtype="object")
    band_key = pd.Series(pd.NA, index=values.index, dtype="object")
    inflamed_flag = pd.Series(pd.NA, index=values.index, dtype="object")
    reasons = pd.Series(pd.NA, index=values.index, dtype="object")

    raw_deficient = adjusted_deficient = 0

    for position, key in enumerate(values.index):
        ferritin_value = ferritin.iloc[position]
        crp_value = crp.iloc[position]
        age_value = age.iloc[position]
        pregnancy_value = pregnancy.iloc[position]

        ferritin_q = (None if ferritin_value != ferritin_value
                      else Quantity(ferritin_value, MICROGRAM_PER_L))
        crp_q = (None if crp_value != crp_value
                 else Quantity(crp_value, MILLIGRAM_PER_L))
        stfr_q = None
        if stfr is not None:
            stfr_value = stfr.iloc[position]
            if stfr_value == stfr_value:
                stfr_q = Quantity(stfr_value, MILLIGRAM_PER_L)

        outcome = iron_status_for(
            ferritin_q, age_value, crp_q,
            adjust_for_inflammation=adjust_for_inflammation,
            pregnant=(pregnancy_value == 1),
            pregnancy_ascertained=(pregnancy_value != 3),
            stfr=stfr_q, definition=definition,
        )
        if isinstance(outcome, UnlabelledIron):
            reasons.iat[position] = outcome.value
            continue

        band = band_for_age(age_value)
        inflamed = is_inflamed(crp_q)
        status.iat[position] = outcome.value
        band_key.iat[position] = band.key
        inflamed_flag.iat[position] = bool(inflamed)

        # Both rules, on every labelled row, whichever one was chosen. The
        # pair is the honest report; either alone is a claim the evidence
        # does not support.
        if band.classify(ferritin_q, False) is IronStatus.DEFICIENT:
            raw_deficient += 1
        if band.classify(ferritin_q, bool(inflamed)) is IronStatus.DEFICIENT:
            adjusted_deficient += 1

    labelled = int(status.notna().sum())
    counts = {"rows": int(len(values)), "labelled": labelled}
    for reason in UnlabelledIron:
        n = int((reasons == reason.value).sum())
        if n:
            counts[reason.name.lower()] = n
    counts["inflamed"] = int((inflamed_flag == True).sum())  # noqa: E712

    chosen = adjusted_deficient if adjust_for_inflammation else raw_deficient
    prevalence = {
        "definition": definition,
        "adjust_for_inflammation": adjust_for_inflammation,
        "deficient": chosen,
        "replete": labelled - chosen,
        "rate": (chosen / labelled) if labelled else float("nan"),
        "rate_adjusted": (
            (adjusted_deficient / labelled) if labelled else float("nan")),
        "rate_unadjusted": (
            (raw_deficient / labelled) if labelled else float("nan")),
    }

    assumptions = (
        "inflammation adjustment is {0}, by explicit choice; neither setting "
        "is authoritative and both prevalences are reported".format(
            "ON" if adjust_for_inflammation else "OFF"),
        "inflammation is CRP > {0}; WHO's own worked example".format(
            INFLAMMATION_CRP_THRESHOLD),
        "AGP is not measured by NHANES, so the BRINDA regression correction "
        "is unavailable and WHO's raised-cutoff option is used instead",
        "rows with no CRP are excluded rather than assumed uninflamed: the "
        "healthy cutoff is the lower one, so assuming it would under-call "
        "deficiency",
    )
    if definition == "ferritin-or-stfr":
        assumptions += (STFR_CAVEAT,)

    return IronLabelReport(
        status=status,
        band=band_key,
        inflamed=inflamed_flag,
        unlabelled_reasons=reasons,
        counts=counts,
        prevalence=prevalence,
        assumptions=assumptions,
    )
