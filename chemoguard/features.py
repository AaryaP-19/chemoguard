"""Derived CBC features, each with the units of its inputs declared.

A ratio of two laboratory values is only meaningful in the units it was
defined for. The Mentzer index is MCV in fL over RBC in million/uL and
discriminates at 13; hand it RBC in cells/uL and every value is 10^6 times
smaller and the threshold is meaningless -- no exception, no warning, just a
feature column that is uniformly wrong. That is CLAUDE.md Finding 7 in a
derived feature instead of a model input.

So every feature below declares ``(analyte, unit)`` for each input, the same
shape as ``model_contract.Feature``, and ``compute`` converts to those units
through ``Quantity`` before the arithmetic. A feature whose inputs are not
declared cannot be built.

The returned values are bare floats, which is correct and is the boundary
working as designed: ``Quantity.magnitude_in`` is documented as the exit for
"a model's feature vector", and a ratio like fL/(million/uL) is not a
quantity ``units.py`` models. The unit safety is in the *inputs*, recorded on
each feature, not in the output type.
"""

from __future__ import annotations

from dataclasses import dataclass

from clinical_flags import LLN
from units import Quantity

from .data.analytes import DATASET_ANALYTES, conversion_factor


@dataclass(frozen=True)
class FeatureInput:
    """One analyte a feature reads, and the unit it must read it in."""

    analyte: str
    unit: str

    def __post_init__(self):
        if self.analyte not in DATASET_ANALYTES:
            raise KeyError(
                "{0!r} is not a dataset analyte".format(self.analyte))


@dataclass(frozen=True)
class DerivedFeature:
    """A computed feature: its inputs, their units, and how it combines them."""

    name: str
    description: str
    inputs: tuple
    #: Human-readable formula, in the declared input units.
    expression: str
    #: What the result is in. Descriptive only -- often a compound ratio.
    result_unit: str
    #: True if the feature reads haemoglobin, directly or through a column
    #: that is algebraically equivalent to it. Declared, and independently
    #: re-derived by ``leakage.audit`` rather than trusted.
    reads_haemoglobin: bool
    fn: object
    #: Demographic columns the feature reads from ``dataset.meta``. Not
    #: analytes: they carry no laboratory unit.
    meta_inputs: tuple = ()

    @property
    def analytes(self):
        return tuple(i.analyte for i in self.inputs)

    def as_dict(self):
        return {
            "name": self.name,
            "expression": self.expression,
            "inputs": [{"analyte": i.analyte, "unit": i.unit}
                       for i in self.inputs],
            "result_unit": self.result_unit,
            "reads_haemoglobin": self.reads_haemoglobin,
        }


def _inputs(*pairs):
    return tuple(FeatureInput(a, u) for a, u in pairs)


# --- the feature functions --------------------------------------------------
#
# Each takes a mapping of analyte -> values already in the declared unit, and
# returns the feature. They work elementwise, so the same function serves a
# pandas Series and a plain float.


def _mentzer(v):
    return v["mcv"] / v["rbc"]


def _rdw_weighted_mentzer(v):
    return v["rdw"] * v["mcv"] / v["rbc"]


def _green_king(v):
    # Green & King (1989), as published: MCV^2 x RDW / (Hb x 100).
    return v["mcv"] ** 2 * v["rdw"] / (v["hb"] * 100.0)


def _mcv_vs_age_lln(v):
    """MCV minus the age-specific lower limit of normal, in fL.

    Negative means microcytic *for that age*. The point of the feature is
    that the adult 80 fL cut-point is wrong for young children -- normal MCV
    at 1-4 years runs roughly 70-86 -- so a model that learns its splits
    from a mostly-adult cohort places them where toddler data is not.
    """
    import numpy as np

    from .labels import mcv_lower_limit

    ages = np.asarray(v["age_years"], dtype=float)
    limits = np.array([
        mcv_lower_limit(None if a != a else a).magnitude_in("fL")
        for a in ages
    ])
    return np.asarray(v["mcv"], dtype=float) - limits


def _nlr(v):
    return _safe_ratio(v["anc"], v["lymphocytes"])


def _plr(v):
    return _safe_ratio(v["platelets"], v["lymphocytes"])


def _hb_per_rbc(v):
    return v["hb"] / v["rbc"]


def _cytopenia_count(v):
    """How many of the three lineages are below their lower limit of normal.

    Thresholds are ``clinical_flags.LLN``, so the count and the CTCAE grading
    the application already serves cannot drift apart.
    """
    return (
        (v["hb"] < LLN["hb"]).astype(int)
        + (v["anc"] < LLN["anc"]).astype(int)
        + (v["platelets"] < LLN["platelets"]).astype(int)
    )


def _safe_ratio(numerator, denominator):
    """Divide, returning NaN where the denominator is zero.

    A lymphocyte count of zero is a real reading (profound lymphopenia), and
    the ratio is genuinely undefined there rather than infinite.
    """
    import numpy as np

    try:
        return numerator / denominator.replace(0, np.nan)
    except AttributeError:
        return float("nan") if denominator == 0 else numerator / denominator


FEATURES = {
    f.name: f
    for f in (
        DerivedFeature(
            "mentzer_index",
            "MCV/RBC. Below 13 favours thalassaemia trait, above 13 iron "
            "deficiency, in a microcytic patient.",
            _inputs(("mcv", "fL"), ("rbc", "million/uL")),
            "MCV[fL] / RBC[million/uL]",
            "fL/(million/uL)",
            False,
            _mentzer,
        ),
        DerivedFeature(
            "rdw_weighted_mentzer",
            "RDW x MCV / RBC -- the Mentzer index scaled by red cell "
            "distribution width. An Hb-free discriminant with no published "
            "provenance: it was carried under the name green_king until "
            "2026-09-21 and renamed because it is not that index. Kept "
            "because it is genuinely Hb-free, which the real Green & King "
            "is not.",
            _inputs(("rdw", "%"), ("mcv", "fL"), ("rbc", "million/uL")),
            "RDW[%] x MCV[fL] / RBC[million/uL]",
            "%.fL/(million/uL)",
            False,
            _rdw_weighted_mentzer,
        ),
        DerivedFeature(
            "green_king",
            "The Green & King index as published (1989): "
            "MCV^2 x RDW / (Hb x 100). Designed to separate iron deficiency "
            "from beta-thalassaemia trait in a microcytic patient, which is "
            "the iron-status question directly. It reads haemoglobin, so it "
            "is available on the iron target and barred from the anaemia "
            "one -- that asymmetry is the point of Finding 8.",
            _inputs(("mcv", "fL"), ("rdw", "%"), ("hb", "g/dL")),
            "MCV[fL]^2 x RDW[%] / (Hb[g/dL] x 100)",
            "fL^2.%/(g/dL)",
            True,
            _green_king,
        ),
        DerivedFeature(
            "mcv_vs_age_lln",
            "MCV minus the age-specific lower limit of normal (70 + age "
            "under 10, else 80). Negative is microcytic for age. Added "
            "2026-09-21 to test whether the under-5 subgroup's weak AUC is "
            "an age-scaling problem or a sample-size one.",
            _inputs(("mcv", "fL")),
            "MCV[fL] - LLN(age)[fL]",
            "fL",
            False,
            _mcv_vs_age_lln,
            meta_inputs=("age_years",),
        ),
        DerivedFeature(
            "nlr",
            "Neutrophil-to-lymphocyte ratio, from the two absolute counts. "
            "A general inflammation marker.",
            _inputs(("anc", "cells/uL"), ("lymphocytes", "cells/uL")),
            "ANC[cells/uL] / lymphocytes[cells/uL]",
            "dimensionless",
            False,
            _nlr,
        ),
        DerivedFeature(
            "plr",
            "Platelet-to-lymphocyte ratio.",
            _inputs(("platelets", "cells/uL"), ("lymphocytes", "cells/uL")),
            "platelets[cells/uL] / lymphocytes[cells/uL]",
            "dimensionless",
            False,
            _plr,
        ),
        DerivedFeature(
            "hb_per_rbc",
            "Hb/RBC. This is mean cell haemoglobin divided by 10: MCH[pg] = "
            "Hb[g/dL]/RBC[million/uL] x 10, exactly. It reads haemoglobin, "
            "and because RBC is itself a feature, Hb = hb_per_rbc x RBC "
            "recovers the held-out value exactly. See leakage.py.",
            _inputs(("hb", "g/dL"), ("rbc", "million/uL")),
            "Hb[g/dL] / RBC[million/uL]",
            "g/dL/(million/uL), == MCH[pg]/10",
            True,
            _hb_per_rbc,
        ),
        DerivedFeature(
            "cytopenia_count",
            "How many of haemoglobin, ANC and platelets are below their "
            "lower limit of normal (clinical_flags.LLN). Reads haemoglobin, "
            "so it is a partial anaemia label, not an input to one.",
            _inputs(("hb", "g/dL"), ("anc", "thousand/uL"),
                    ("platelets", "thousand/uL")),
            "[Hb < {0}] + [ANC < {1}] + [PLT < {2}]".format(
                LLN["hb"], LLN["anc"], LLN["platelets"]),
            "count of 0-3",
            True,
            _cytopenia_count,
        ),
    )
}

#: Features that read haemoglobin. Excluded from an Hb-held-out matrix by
#: ``build_matrix``; see ``leakage.py`` for why name-based exclusion is not
#: on its own sufficient.
HAEMOGLOBIN_FEATURES = tuple(
    name for name, f in FEATURES.items() if f.reads_haemoglobin
)


def _column(dataset, analyte, unit):
    """One analyte column, converted to a named unit through Quantity."""
    canonical = (dataset.units or {}).get(analyte)
    if canonical is None:
        raise KeyError(
            "{0} does not report {1}, which is needed in {2}.".format(
                dataset.name, analyte, unit)
        )
    return dataset.values[analyte] * conversion_factor(canonical, unit)


def compute(feature, dataset):
    """Compute one feature over a whole ``Dataset``."""
    if not isinstance(feature, DerivedFeature):
        feature = FEATURES[feature]
    columns = {i.analyte: _column(dataset, i.analyte, i.unit)
               for i in feature.inputs}
    for name in feature.meta_inputs:
        if dataset.meta is None or name not in dataset.meta.columns:
            raise KeyError(
                "{0} needs the demographic column {1!r}, which {2} does not "
                "carry.".format(feature.name, name, dataset.name)
            )
        columns[name] = dataset.meta[name]
    return feature.fn(columns)


def compute_one(feature, quantities):
    """Compute one feature from a mapping of analyte -> ``Quantity``.

    The scalar counterpart of ``compute``. Every input must be a Quantity;
    a bare float raises, because the feature's meaning depends on its unit.
    """
    if not isinstance(feature, DerivedFeature):
        feature = FEATURES[feature]

    values = {}
    for item in feature.inputs:
        quantity = quantities.get(item.analyte)
        if not isinstance(quantity, Quantity):
            raise TypeError(
                "{0} needs {1} as a Quantity in {2}, got {3}.".format(
                    feature.name, item.analyte, item.unit,
                    type(quantity).__name__)
            )
        values[item.analyte] = quantity.magnitude_in(item.unit)
    return feature.fn(values)


def available(dataset, feature):
    """Whether a dataset reports everything a feature needs."""
    if not isinstance(feature, DerivedFeature):
        feature = FEATURES[feature]
    if not all(a in (dataset.units or {}) for a in feature.analytes):
        return False
    if feature.meta_inputs:
        meta = dataset.meta
        if meta is None or any(m not in meta.columns
                               for m in feature.meta_inputs):
            return False
    return True


# --- the anaemia feature matrix --------------------------------------------

#: Held out of the anaemia task: the label is computed from haemoglobin, and
#: haematocrit is the same measurement by another route (HCT = Hb/MCHC x 100).
HELD_OUT_FOR_ANAEMIA = ("hb", "hematocrit")

#: Analytes offered to the anaemia matrix under the naive hold-out -- every
#: analyte the loader carries except the two named above.
#:
#: This list is the *question*, not the answer. Removing hb and hematocrit by
#: name leaves MCH, MCHC, MCV and RBC in place, and those reconstruct
#: haemoglobin arithmetically. ``chemoguard.leakage`` measures it; the
#: ``"reconstruction-safe"`` profile below is what survives.
_NAIVE_ANALYTES = (
    "rbc", "mcv", "mch", "mchc", "rdw",
    "platelets", "mpv", "wbc", "anc", "lymphocytes", "neutrophils_percent",
)

#: What is left once the closed-form reconstructions go. See leakage.py: MCH
#: and MCHC each recover Hb to well within the identity tolerance when paired
#: with RBC, so holding out Hb while keeping them holds out nothing.
#:
#: This is still not enough. MCV x RBC recovers haematocrit exactly, and
#: MCHC varies so little between people that red cell mass alone predicts
#: haemoglobin. Measured, not argued: see the audit.
_SAFE_ANALYTES = (
    "rbc", "mcv", "rdw",
    "platelets", "mpv", "wbc", "anc", "lymphocytes", "neutrophils_percent",
)

#: Everything that carries red cell mass removed -- MCV and RBC both, and the
#: derived indices that reconstruct MCV from their own definition
#: (mentzer_index x RBC = MCV).
#:
#: This is the only profile in which haemoglobin is genuinely held out, and
#: it is included to show what that costs rather than as a recommendation.
#: Anaemia *is* a red cell mass deficit; a matrix with no red cell mass in it
#: is not a harder version of the task, it is a different one.
_RED_CELL_FREE_ANALYTES = (
    "rdw", "platelets", "mpv", "wbc", "anc", "lymphocytes",
    "neutrophils_percent",
)

#: The iron-deficiency profile: the whole CBC panel, nothing held out.
#:
#: Ferritin is measured on a separate assay and no CBC identity produces it,
#: so haemoglobin and haematocrit are ordinary features here. That is the
#: whole reason for the pivot: the asymmetry between this profile and the
#: anaemia ones *is* Finding 8.
_IRON_ANALYTES = _NAIVE_ANALYTES + ("hb", "hematocrit")

#: Columns that define a target and must therefore never be a feature of it.
#: Asserted in ``build_matrix`` rather than left to the profile lists.
NEVER_A_FEATURE = ("ferritin", "stfr", "crp")

PROFILES = {
    # Exclusion by column name only. The profile the brief asked for, kept so
    # the audit has something to measure and the leak stays reproducible.
    "name-excluded": (_NAIVE_ANALYTES,
                      ("mentzer_index", "rdw_weighted_mentzer", "nlr", "plr")),
    # Additionally drops every column with a closed-form route back to Hb.
    "reconstruction-safe": (_SAFE_ANALYTES,
                            ("mentzer_index", "rdw_weighted_mentzer",
                             "nlr", "plr")),
    # Additionally drops red cell mass entirely.
    "red-cell-free": (_RED_CELL_FREE_ANALYTES, ("nlr", "plr")),
    # The iron target: everything the CBC carries, including the real Green
    # & King index, which reads Hb and is the index designed for exactly
    # this discrimination.
    "iron-deficiency": (_IRON_ANALYTES,
                        ("mentzer_index", "rdw_weighted_mentzer",
                         "green_king", "nlr", "plr", "hb_per_rbc",
                         "cytopenia_count")),
    # Same, plus the age-normalised MCV. Exists to be compared against the
    # profile above: if the under-5 AUC moves, the weak subgroup was an
    # age-scaling problem; if it does not, it is sample size.
    "iron-deficiency-age-aware": (
        _IRON_ANALYTES,
        ("mentzer_index", "rdw_weighted_mentzer", "green_king", "nlr",
         "plr", "hb_per_rbc", "cytopenia_count", "mcv_vs_age_lln")),
}

#: Profiles that hold something out, i.e. the anaemia ones. The iron profile
#: holds nothing out, so the hold-out tests do not apply to it -- and that
#: difference is the whole content of Finding 8.
#: Which columns each profile deliberately withholds.
PROFILE_HELD_OUT = {
    "name-excluded": HELD_OUT_FOR_ANAEMIA,
    "reconstruction-safe": HELD_OUT_FOR_ANAEMIA,
    "red-cell-free": HELD_OUT_FOR_ANAEMIA,
    "iron-deficiency": (),
    "iron-deficiency-age-aware": (),
}

#: Demographics the anaemia task may read. Age and sex select the WHO band,
#: so a model that does not see them cannot reproduce a sex-adjusted label
#: even in principle.
ANAEMIA_DEMOGRAPHICS = ("age_years", "sex_male")


@dataclass(frozen=True)
class FeatureMatrix:
    """A built feature matrix, and the record of what was kept out of it."""

    name: str
    frame: object
    analyte_columns: tuple
    derived_columns: tuple
    demographic_columns: tuple
    held_out: tuple
    units: dict
    excluded: dict

    @property
    def columns(self):
        return tuple(self.frame.columns)

    def as_dict(self):
        return {
            "profile": self.name,
            "columns": list(self.columns),
            "held_out": list(self.held_out),
            "excluded": dict(self.excluded),
            "input_units": dict(self.units),
        }


ANAEMIA_PROFILES = tuple(
    name for name, held in PROFILE_HELD_OUT.items() if held
)


def build_matrix(dataset, profile, include_demographics=True):
    """Build a feature matrix for one profile.

    ``profile`` names an entry of ``PROFILES`` and decides both which
    analytes are offered and which derived features are computed.
    ``PROFILE_HELD_OUT`` records what it deliberately withholds.

    Returns a ``FeatureMatrix``. Every analyte column is in its canonical
    unit and ``units`` records which; every derived column was computed
    through ``compute``, which converts inputs by their declared units.

    Columns naming a target -- ferritin, sTfR, CRP -- are refused outright,
    whatever a profile says. A label input in the feature matrix is the
    failure mode this whole package exists to catch.
    """
    import pandas as pd

    try:
        analytes, feature_names = PROFILES[profile]
    except KeyError:
        raise KeyError(
            "Unknown profile {0!r}. Known: {1}".format(
                profile, ", ".join(sorted(PROFILES)))
        ) from None

    smuggled = [a for a in analytes if a in NEVER_A_FEATURE]
    if smuggled:
        raise ValueError(
            "Profile {0!r} lists {1}, which define a target and cannot be "
            "features of it.".format(profile, ", ".join(smuggled))
        )

    held_out = PROFILE_HELD_OUT.get(profile, ())
    available_units = dataset.units or {}
    columns, units, excluded = {}, {}, {}

    for analyte in held_out:
        excluded[analyte] = "held out: the anaemia label is computed from it"
    for analyte in NEVER_A_FEATURE:
        if analyte in available_units:
            excluded[analyte] = "defines the iron-status target"

    candidates = _IRON_ANALYTES
    for analyte in candidates:
        if analyte in held_out:
            continue
        if analyte not in analytes:
            excluded.setdefault(analyte, (
                "drops out under the {0} profile: reconstructs haemoglobin"
                .format(profile)))
            continue
        if analyte not in available_units:
            excluded[analyte] = "not reported by {0}".format(dataset.name)
            continue
        columns[analyte] = dataset.values[analyte]
        units[analyte] = available_units[analyte]

    for name in FEATURES:
        feature = FEATURES[name]
        if name not in feature_names:
            excluded[name] = (
                "reads haemoglobin" if feature.reads_haemoglobin
                else "not in the {0} profile".format(profile))
            continue
        if not available(dataset, feature):
            missing = [a for a in feature.analytes if a not in available_units]
            excluded[name] = "needs {0}, not reported".format(
                ", ".join(missing))
            continue
        columns[name] = compute(feature, dataset)

    demographic_columns = ()
    if include_demographics and dataset.meta is not None:
        meta = dataset.meta
        columns["age_years"] = meta["age_years"]
        # One binary column rather than a label encoding: there are two
        # values and an ordinal code would imply an order.
        columns["sex_male"] = (meta["sex"] == "male").astype(float)
        demographic_columns = ANAEMIA_DEMOGRAPHICS

    frame = pd.DataFrame(columns, index=dataset.values.index)
    kept_features = tuple(n for n in feature_names if n in frame.columns)

    return FeatureMatrix(
        name=profile,
        frame=frame,
        analyte_columns=tuple(a for a in analytes if a in frame.columns),
        derived_columns=kept_features,
        demographic_columns=demographic_columns,
        held_out=held_out,
        units=units,
        excluded=excluded,
    )


def build_anaemia_matrix(dataset, profile="reconstruction-safe",
                         include_demographics=True):
    """The anaemia matrix, with haemoglobin and HCT held out.

    A thin wrapper over ``build_matrix`` that refuses the iron profile, so
    an anaemia caller cannot pick the one profile that keeps haemoglobin.
    """
    if not PROFILE_HELD_OUT.get(profile):
        raise ValueError(
            "Profile {0!r} holds nothing out and is not an anaemia profile. "
            "Use build_matrix() if that is what you meant.".format(profile)
        )
    return build_matrix(dataset, profile, include_demographics)


def build_iron_matrix(dataset, include_demographics=True, age_aware=False):
    """The iron-deficiency matrix: the whole CBC panel, nothing held out."""
    profile = ("iron-deficiency-age-aware" if age_aware else "iron-deficiency")
    return build_matrix(dataset, profile, include_demographics)
