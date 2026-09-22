"""Staged coherence and plausibility checks for a CBC dataset.

A CBC is internally redundant: several reported columns are arithmetic
functions of others, because the analyser computes them that way. Those
identities are a test of whether a row is a real measurement. Synthetic rows
drawn per-column do not satisfy them, and CLAUDE.md Finding 2 is the result of
applying them to the withdrawn dataset.

This module is deliberately **staged** rather than one joint filter, because
the five checks are not five independent pieces of evidence and a flat filter
over all of them reports a survivor count that is wrong in a specific
direction -- too low, for reasons that have nothing to do with the data.

Stage 1  Exact analyser identities, tight tolerance.
         MCV, MCH and PCT. These are definitions, not correlations: the
         instrument computes each one from the others. Only these form the
         joint filter.

Stage 2  MCHC. Checked and reported, and **excluded from the joint filter**,
         because it is algebraically implied by the two identities already in
         it:

             MCH / MCV x 100  =  (Hb/RBC x 10) / (HCT/RBC x 10) x 100
                              =  Hb / HCT x 100
                              =  MCHC

         RBC cancels. A row that satisfies the MCV and MCH identities
         satisfies the MCHC one as a matter of arithmetic, so counting it in
         the joint filter double-counts the same evidence and deflates the
         survivor count without adding information.
         ``CoherenceReport.mchc_redundancy`` measures this rather than
         asserting it.

Stage 3  HCT ~ 3 x Hb. A bedside rule of thumb, not an analyser identity. It
         is the MCHC identity with MCHC pinned at 33.3: HCT = Hb/MCHC x 100,
         so HCT = 3 x Hb exactly when MCHC = 33.3 g/dL. Real MCHC varies over
         roughly 30-37, so the rule is wrong by the amount MCHC differs from
         33.3 even on flawless data. Checked at a wide band, flagged
         approximate, and **excluded from the joint filter**. CLAUDE.md
         Finding 6 notes that a 5% bound on this check was dragging the joint
         count down; at a band wide enough to be fair to it the check is
         nearly vacuous on real data (99.9% of NHANES rows pass at 15%), so it
         earns its place as a report line and not as a filter.

Stage 4  Plausibility bounds. A separate stage with a separate count, because
         it asks a different question. An identity asks whether a row is
         self-consistent; a bound asks whether a value could be a measurement
         at all. A row can fail either without failing the other.

Survivor counts are reported at every stage. There is no headline number,
because the number depends on which question is being asked and the previous
single figure (CLAUDE.md Finding 2's "3,000-3,800") came from mixing two.

Tolerances are named constants below, each with its justification. They are
not arguments with defaults and are not chosen per run -- Finding 2 records
what happens when they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .analytes import DATASET_ANALYTES, conversion_factor

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

#: Relative tolerance for the exact analyser identities (Stage 1).
#:
#: Justified by quantisation, measured rather than assumed. NHANES reports Hb,
#: HCT, MCV, MCH and MCHC to one decimal and RBC to two, so the identities
#: cannot hold exactly even on flawless data -- the reported values are
#: rounded before we see them. The tightest identity is MCH = Hb/RBC x 10,
#: whose worst case is at the low end of each column (Hb 5.4, RBC 2.32,
#: MCH 10.2 over the two cycles loaded):
#:
#:     0.05/5.4 + 0.005/2.32 + 0.05/10.2  =  0.93% + 0.22% + 0.49%  =  1.63%
#:
#: The observed maximum over 19,749 NHANES rows is 1.41%, inside that bound.
#: 2% clears both and rejects nothing on rounding alone: at 2% every NHANES
#: row passes the MCH and MCHC identities, and all but two pass MCV.
#:
#: The choice is also not load-bearing. On the quarantined legacy data the
#: pass rate is flat in the tolerance -- MCV moves 74.9% -> 76.0% between 1%
#: and 5% -- because those rows are not near-misses, they are unrelated. A
#: result that barely moves across a 5x change in tolerance is not a result
#: that was chosen by the tolerance.
EXACT_IDENTITY_TOLERANCE = 0.02

#: Relative tolerance for the approximate rule of thumb (Stage 3).
#:
#: HCT ~ 3 x Hb holds exactly at MCHC = 33.3 g/dL and degrades linearly as
#: MCHC departs from it. A physiological MCHC range of 30-37 g/dL implies a
#: spread of about -10% to +11% in the ratio on data that is entirely
#: correct, so any band tighter than that rejects real rows for being
#: normal. 15% covers it with margin; 99.9% of NHANES rows pass. This is
#: reported, never filtered on.
APPROXIMATE_IDENTITY_TOLERANCE = 0.15

#: Dimensional bridges, not unit conversions.
#:
#: Each falls out of the canonical units and is written here so no magic
#: number sits inside a formula. Derivations, with canonical units in
#: brackets:
#:
#:   MCV [fL]  = (HCT[%]/100) / (RBC[million/uL] x 1e12 /L)
#:             = HCT/RBC x 1e-14 L = HCT/RBC x 10 fL
#:   MCH [pg]  = (Hb[g/dL] x 10 g/L) / (RBC[million/uL] x 1e12 /L)
#:             = Hb/RBC x 1e-11 g = Hb/RBC x 10 pg
#:   MCHC[g/dL]= Hb[g/dL] / (HCT[%]/100) = Hb/HCT x 100
#:   PCT [%]   = PLT[1e9/L] x MPV[fL] x 1e-15 L x 100
#:             = PLT x MPV / 10000
HCT_RBC_TO_MCV = 10.0
HB_RBC_TO_MCH = 10.0
HB_HCT_TO_MCHC = 100.0
PLT_MPV_TO_PCT = 10000.0

#: The rule-of-thumb constant: 100/33.3, i.e. the reciprocal of the MCHC the
#: rule silently assumes.
_HCT_HB_RULE_OF_THUMB = 3.0

#: PCT is defined on platelets in 10^9/L (= thousand/uL); the canonical unit
#: for platelets is cells/uL. The factor comes from the two declared units,
#: via Quantity -- never from a literal 1000.
_PLATELETS_TO_THOUSAND = conversion_factor("cells/uL", "thousand/uL")


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Identity:
    """One redundancy in a CBC panel, and how strictly to hold it."""

    key: str
    label: str
    formula: str
    #: The analyte whose reported column is being tested.
    reported: str
    #: The analytes the derivation reads.
    inputs: tuple
    tolerance: float
    #: "exact" -- the analyser computes it this way.
    #: "approximate" -- a clinical rule of thumb.
    kind: str
    #: Whether this check contributes to the joint survivor count.
    in_joint_filter: bool
    #: Why it is or is not in the joint filter.
    rationale: str

    def derive(self, values):
        return _DERIVATIONS[self.key](values)

    @property
    def needs(self):
        return (self.reported,) + tuple(self.inputs)


def _derive_mcv(v):
    return v["hematocrit"] / v["rbc"] * HCT_RBC_TO_MCV


def _derive_mch(v):
    return v["hb"] / v["rbc"] * HB_RBC_TO_MCH


def _derive_mchc(v):
    return v["hb"] / v["hematocrit"] * HB_HCT_TO_MCHC


def _derive_pct(v):
    return (v["platelets"] * _PLATELETS_TO_THOUSAND) * v["mpv"] / PLT_MPV_TO_PCT


def _derive_hct(v):
    return v["hb"] * _HCT_HB_RULE_OF_THUMB


_DERIVATIONS = {
    "mcv": _derive_mcv,
    "mch": _derive_mch,
    "mchc": _derive_mchc,
    "pct": _derive_pct,
    "hct_rule": _derive_hct,
}


IDENTITIES = (
    Identity(
        "mcv", "MCV", "MCV = HCT / RBC x 10",
        "mcv", ("hematocrit", "rbc"),
        EXACT_IDENTITY_TOLERANCE, "exact", True,
        "Analyser identity. Independent of the MCH check: it reads HCT where "
        "MCH reads Hb.",
    ),
    Identity(
        "mch", "MCH", "MCH = Hb / RBC x 10",
        "mch", ("hb", "rbc"),
        EXACT_IDENTITY_TOLERANCE, "exact", True,
        "Analyser identity. Independent of the MCV check: it reads Hb where "
        "MCV reads HCT.",
    ),
    Identity(
        "pct", "Plateletcrit", "PCT = PLATELETS x MPV / 10000",
        "pct", ("platelets", "mpv"),
        EXACT_IDENTITY_TOLERANCE, "exact", True,
        "Analyser identity over the platelet channel, and the only joint "
        "check that does not read the red cell columns.",
    ),
    Identity(
        "mchc", "MCHC", "MCHC = Hb / HCT x 100",
        "mchc", ("hb", "hematocrit"),
        EXACT_IDENTITY_TOLERANCE, "exact", False,
        "EXCLUDED from the joint filter: algebraically implied by the MCV and "
        "MCH identities, since MCH/MCV x 100 = Hb/HCT x 100. Counting it "
        "again double-counts the same evidence and deflates the survivor "
        "count.",
    ),
    Identity(
        "hct_rule", "HCT rule of thumb", "HCT ~ 3 x Hb",
        "hematocrit", ("hb",),
        APPROXIMATE_IDENTITY_TOLERANCE, "approximate", False,
        "EXCLUDED from the joint filter: a bedside approximation, exact only "
        "at MCHC = 33.3 g/dL, so it fails on correct data whenever MCHC is "
        "normal but not 33.3. Reported at a wide band and flagged.",
    ),
)

JOINT_IDENTITIES = tuple(i for i in IDENTITIES if i.in_joint_filter)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentityResult:
    """What one identity did on one dataset."""

    identity: Identity
    evaluable: int
    passed: int
    #: None when the source does not report the columns at all.
    not_evaluable_reason: str = ""
    median_error: float = float("nan")
    p95_error: float = float("nan")
    max_error: float = float("nan")
    mae: float = float("nan")
    correlation: float = float("nan")

    @property
    def is_evaluable(self):
        return not self.not_evaluable_reason

    @property
    def pass_rate(self):
        return (self.passed / self.evaluable) if self.evaluable else float("nan")


@dataclass(frozen=True)
class Stage:
    """A named filtering step and how many rows came out of it."""

    name: str
    description: str
    survivors: int
    considered: int
    #: True when the stage only reports and does not filter.
    reported_only: bool = False
    detail: str = ""

    @property
    def rate(self):
        return (self.survivors / self.considered) if self.considered else float("nan")


@dataclass(frozen=True)
class CoherenceReport:
    dataset: str
    rows: int
    identities: tuple
    stages: tuple
    plausibility: dict = field(default_factory=dict)
    mchc_redundancy: dict = field(default_factory=dict)
    notes: tuple = field(default_factory=tuple)

    def identity(self, key):
        for result in self.identities:
            if result.identity.key == key:
                return result
        raise KeyError(key)

    def stage(self, name):
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _relative_error(reported, derived):
    """|reported - derived| / |derived|, with the undefined cases masked.

    **The denominator is the derived value, not the reported one.** The
    derived value is the reference -- what the row would say if the analyser
    had computed it -- and an error is measured against a reference. Dividing
    by the reported value instead scales the error by the very quantity under
    test, so a row that overstates a column is judged more leniently than one
    that understates it by the same absolute amount.

    This is the one deliberate methodological difference from CLAUDE.md
    Finding 6, and it accounts for the whole gap between its numbers and
    these. Finding 6 divided by the reported value; doing that here
    reproduces its counts exactly (158 at 1%, 709 at 5%, 1524 at 10% over all
    five identities jointly). Dividing by the derived value gives 157, 683
    and 1392 for the same filter.

    Derived values at or near zero make a relative error meaningless rather
    than large, so they are excluded from the comparison instead of counted
    as failures.
    """
    import numpy as np

    denominator = derived.abs()
    usable = denominator > 0
    error = (reported - derived).abs() / denominator.where(usable)
    return error.where(np.isfinite(error))


def _evaluate_identity(identity, values):
    import numpy as np

    missing = [a for a in identity.needs if a not in values.columns]
    if missing:
        return IdentityResult(
            identity, 0, 0,
            not_evaluable_reason="source does not report {0}".format(
                ", ".join(missing)),
        )

    frame = values[list(identity.needs)].dropna()
    if frame.empty:
        return IdentityResult(
            identity, 0, 0,
            not_evaluable_reason="every row null in {0}".format(
                ", ".join(identity.needs)),
        )

    reported = frame[identity.reported]
    derived = identity.derive(frame)
    error = _relative_error(reported, derived).dropna()
    if error.empty:
        return IdentityResult(
            identity, 0, 0,
            not_evaluable_reason="no row with a non-zero derived value",
        )

    aligned_reported = reported.loc[error.index]
    aligned_derived = derived.loc[error.index]
    if len(error) > 1 and aligned_derived.std() > 0 and aligned_reported.std() > 0:
        correlation = float(aligned_reported.corr(aligned_derived))
    else:
        correlation = float("nan")

    return IdentityResult(
        identity=identity,
        evaluable=int(len(error)),
        passed=int((error <= identity.tolerance).sum()),
        median_error=float(error.median()),
        p95_error=float(np.percentile(error, 95)),
        max_error=float(error.max()),
        mae=float((aligned_reported - aligned_derived).abs().mean()),
        correlation=correlation,
    )


def _joint_mask(values, identities, tolerance_scale=1.0):
    """Rows evaluable on every identity and passing all of them.

    Returns ``(mask, evaluable_mask)``. Keeping the two apart matters: a row
    that cannot be checked has not passed, and it has not failed either.
    Folding non-evaluable rows into either bucket misstates the result.
    """
    import pandas as pd

    evaluable = pd.Series(True, index=values.index)
    passing = pd.Series(True, index=values.index)

    for identity in identities:
        if any(a not in values.columns for a in identity.needs):
            continue
        frame = values[list(identity.needs)]
        row_ok = frame.notna().all(axis=1)
        derived = identity.derive(frame)
        error = _relative_error(frame[identity.reported], derived)
        row_ok &= error.notna()
        evaluable &= row_ok
        passing &= (error <= identity.tolerance * tolerance_scale).fillna(False)

    return (passing & evaluable), evaluable


def _plausibility(values):
    """Per-analyte bound violations, using the declared physiological ranges.

    Bounds come from ``cbc.ANALYTES`` (extended by ``analytes.py`` for MPV and
    plateletcrit) -- the same ones the runtime prediction path enforces, so
    this stage asks of the training data exactly what the API asks of a
    request.
    """
    import pandas as pd

    per_analyte, within = {}, pd.Series(True, index=values.index)

    for key in values.columns:
        analyte = DATASET_ANALYTES.get(key)
        if analyte is None:
            continue
        column = values[key]
        present = column.notna()
        ok = (column >= analyte.plausible_min) & (column <= analyte.plausible_max)
        violations = int((present & ~ok).sum())
        per_analyte[key] = {
            "present": int(present.sum()),
            "violations": violations,
            "bounds": (analyte.plausible_min, analyte.plausible_max),
            "unit": analyte.canonical.symbol,
        }
        within &= (ok | ~present)

    return per_analyte, within


def _mchc_redundancy(values):
    """Measure the claim that MCHC adds nothing once MCV and MCH are held.

    If MCHC really is implied, then among rows passing both the MCV and MCH
    identities, essentially all should pass the MCHC one. Reported as a
    number so the exclusion in Stage 2 rests on evidence and not only on the
    algebra.
    """
    by_key = {i.key: i for i in IDENTITIES}
    needed = ("mcv", "mch", "mchc")
    if any(
        a not in values.columns
        for key in needed for a in by_key[key].needs
    ):
        return {}

    mcv_mch, evaluable = _joint_mask(values, [by_key["mcv"], by_key["mch"]])
    mchc_mask, mchc_evaluable = _joint_mask(values, [by_key["mchc"]])

    both = evaluable & mchc_evaluable
    given = mcv_mch & both
    n_given = int(given.sum())
    if not n_given:
        return {}

    return {
        "n_passing_mcv_and_mch": n_given,
        "of_which_pass_mchc": int((given & mchc_mask).sum()),
        "conditional_pass_rate": float((given & mchc_mask).sum() / n_given),
        "marginal_pass_rate": float(
            (mchc_mask & mchc_evaluable).sum() / max(int(mchc_evaluable.sum()), 1)
        ),
    }


def evaluate(dataset):
    """Run every stage over a ``Dataset`` and return a ``CoherenceReport``.

    Takes the ``Dataset`` rather than a bare frame so the units travel with
    the values. Values must already be in their analytes' canonical units,
    which is what the loaders in this package return.
    """
    values = dataset.values
    units = dataset.units or {}

    for key in values.columns:
        analyte = DATASET_ANALYTES.get(key)
        if analyte is None:
            continue
        declared = units.get(key)
        if declared and declared != analyte.canonical.symbol:
            raise ValueError(
                "Column {0!r} is in {1} but the canonical unit is {2}. "
                "coherence.evaluate requires canonical values; load through "
                "analytes.rescale_frame.".format(
                    key, declared, analyte.canonical.symbol)
            )

    total = int(len(values))
    results = tuple(_evaluate_identity(i, values) for i in IDENTITIES)

    per_analyte, within_bounds = _plausibility(values)
    n_plausible = int(within_bounds.sum())

    joint, joint_evaluable = _joint_mask(values, JOINT_IDENTITIES)
    n_joint_evaluable = int(joint_evaluable.sum())
    n_joint = int(joint.sum())
    n_both = int((joint & within_bounds).sum())

    usable_joint = [
        i for i in JOINT_IDENTITIES
        if all(a in values.columns for a in i.needs)
    ]
    skipped_joint = [
        i for i in JOINT_IDENTITIES
        if not all(a in values.columns for a in i.needs)
    ]

    stages = (
        Stage("loaded", "Rows loaded, in canonical units", total, total),
        Stage(
            "plausibility",
            "Every reported value within its physiological bounds",
            n_plausible, total,
            detail="{0} analytes bounded".format(len(per_analyte)),
        ),
        Stage(
            "exact-evaluable",
            "Rows with every joint-filter input present",
            n_joint_evaluable, total,
            detail="joint filter over {0}{1}".format(
                ", ".join(i.label for i in usable_joint),
                "" if not skipped_joint else
                "; NOT APPLIED: {0} ({1} absent from source)".format(
                    ", ".join(i.label for i in skipped_joint),
                    ", ".join(sorted({
                        a for i in skipped_joint for a in i.needs
                        if a not in values.columns
                    })),
                ),
            ),
        ),
        Stage(
            "exact-identities",
            "...and satisfying all of them at {0:.0%}".format(
                EXACT_IDENTITY_TOLERANCE),
            n_joint, n_joint_evaluable,
        ),
        Stage(
            "mchc-reported",
            "MCHC identity (reported, NOT filtered)",
            next(r.passed for r in results if r.identity.key == "mchc"),
            next(r.evaluable for r in results if r.identity.key == "mchc"),
            reported_only=True,
            detail="algebraically implied by MCV and MCH",
        ),
        Stage(
            "hct-rule-reported",
            "HCT ~ 3xHb at {0:.0%} (reported, NOT filtered)".format(
                APPROXIMATE_IDENTITY_TOLERANCE),
            next(r.passed for r in results if r.identity.key == "hct_rule"),
            next(r.evaluable for r in results if r.identity.key == "hct_rule"),
            reported_only=True,
            detail="rule of thumb, exact only at MCHC 33.3",
        ),
        Stage(
            "coherent-and-plausible",
            "Joint exact identities AND plausibility",
            n_both, total,
        ),
    )

    notes = tuple(dataset.notes or ())
    if skipped_joint:
        notes += (
            "joint filter ran over {0} of {1} exact identities; {2} not "
            "evaluable because the source does not report {3}".format(
                len(usable_joint), len(JOINT_IDENTITIES),
                ", ".join(i.label for i in skipped_joint),
                ", ".join(sorted({
                    a for i in skipped_joint for a in i.needs
                    if a not in values.columns
                })),
            ),
        )

    return CoherenceReport(
        dataset=dataset.name,
        rows=total,
        identities=results,
        stages=stages,
        plausibility=per_analyte,
        mchc_redundancy=_mchc_redundancy(values),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------


def _pct(value):
    return "n/a" if value != value else "{0:.2f}%".format(value * 100)


def format_report(report, indent=""):
    """A single dataset's staged result, as plain ASCII text.

    ASCII throughout: this prints to the Windows cp1252 console, which is
    where a stray non-ASCII character has already cost this repo a
    misdiagnosed bug (CLAUDE.md, "Fixes that look done and aren't").
    """
    lines = []
    add = lines.append
    add("{0}{1}".format(indent, report.dataset))
    add("{0}{1}".format(indent, "=" * len(report.dataset)))
    for note in report.notes:
        add("{0}  note: {1}".format(indent, note))
    add("")

    add("{0}  Per-identity".format(indent))
    add("{0}  {1:<22}{2:>9}{3:>9}{4:>9}  {5:>8} {6:>8} {7:>7}".format(
        indent, "identity", "evaluable", "pass", "rate",
        "med err", "p95 err", "corr"))
    for result in report.identities:
        identity = result.identity
        flag = "" if identity.in_joint_filter else "   [reported only]"
        if not result.is_evaluable:
            add("{0}  {1:<22}{2:>9}{3:>9}{4:>9}  {5}".format(
                indent, identity.label, "-", "-", "n/a",
                result.not_evaluable_reason))
            continue
        add("{0}  {1:<22}{2:>9}{3:>9}{4:>9}  {5:>8} {6:>8} {7:>7}{8}".format(
            indent, identity.label, result.evaluable, result.passed,
            _pct(result.pass_rate), _pct(result.median_error),
            _pct(result.p95_error),
            "n/a" if result.correlation != result.correlation
            else "{0:.3f}".format(result.correlation),
            flag))
    add("")

    add("{0}  Stages".format(indent))
    for stage in report.stages:
        marker = "(reported)" if stage.reported_only else ""
        add("{0}  {1:<24}{2:>8} / {3:<8} {4:>8}  {5} {6}".format(
            indent, stage.name, stage.survivors, stage.considered,
            _pct(stage.rate), marker, stage.detail))
    add("")

    if report.mchc_redundancy:
        r = report.mchc_redundancy
        add("{0}  MCHC redundancy check".format(indent))
        add("{0}    rows passing MCV and MCH : {1}".format(
            indent, r["n_passing_mcv_and_mch"]))
        add("{0}    of which pass MCHC       : {1} ({2})".format(
            indent, r["of_which_pass_mchc"], _pct(r["conditional_pass_rate"])))
        add("{0}    MCHC pass rate overall   : {1}".format(
            indent, _pct(r["marginal_pass_rate"])))
        add("{0}    -> conditional rate at or near 100% is the evidence that "
            "MCHC".format(indent))
        add("{0}       carries no information the joint filter does not "
            "already have.".format(indent))
        add("")

    violations = {k: v for k, v in report.plausibility.items() if v["violations"]}
    add("{0}  Plausibility bound violations".format(indent))
    if not violations:
        add("{0}    none".format(indent))
    for key, info in sorted(violations.items(),
                            key=lambda kv: -kv[1]["violations"]):
        add("{0}    {1:<20} {2:>6} of {3:<6} outside {4:g}-{5:g} {6}".format(
            indent, key, info["violations"], info["present"],
            info["bounds"][0], info["bounds"][1], info["unit"]))
    return "\n".join(lines)


def format_comparison(reports):
    """Stage-by-stage survivor counts for several datasets, side by side."""
    if not reports:
        return ""

    names = [r.dataset for r in reports]
    width = max(22, max(len(n) for n in names) + 2)
    stage_names = [s.name for s in reports[0].stages]

    lines = []
    add = lines.append
    add("Side by side: survivors at each stage")
    add("=" * (26 + width * len(reports)))
    add("{0:<26}{1}".format(
        "stage", "".join("{0:>{1}}".format(n, width) for n in names)))
    for name in stage_names:
        cells = []
        for report in reports:
            try:
                stage = report.stage(name)
            except KeyError:
                cells.append("{0:>{1}}".format("-", width))
                continue
            cells.append("{0:>{1}}".format(
                "{0} ({1})".format(stage.survivors, _pct(stage.rate)), width))
        add("{0:<26}{1}".format(name, "".join(cells)))

    add("")
    add("{0:<26}{1}".format(
        "identity pass rate",
        "".join("{0:>{1}}".format(n, width) for n in names)))
    for identity in IDENTITIES:
        cells = []
        for report in reports:
            result = report.identity(identity.key)
            cells.append("{0:>{1}}".format(
                "not reported" if not result.is_evaluable
                else _pct(result.pass_rate), width))
        label = identity.label + ("" if identity.in_joint_filter else " *")
        add("{0:<26}{1}".format(label, "".join(cells)))
    add("")
    add("* reported only -- not part of the joint filter.")
    return "\n".join(lines)
