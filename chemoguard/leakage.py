"""Does the anaemia feature matrix still contain haemoglobin?

Dropping the ``hb`` column is not the same as holding haemoglobin out. A CBC
is internally redundant by construction -- that redundancy is exactly what
``data/coherence.py`` filters on -- and every one of those identities is a
route back to Hb:

    MCH  = Hb/RBC x 10      ->  Hb = MCH x RBC / 10
    MCHC = Hb/HCT x 100     ->  Hb = MCHC x HCT / 100
    MCV  = HCT/RBC x 10     ->  HCT = MCV x RBC / 10
                            ->  Hb = MCHC x MCV x RBC / 1000

The better the data, the worse this is. On the withdrawn dataset roughly a
quarter of rows satisfied these identities, so reconstruction would have been
noise on most of the file and the leak might have gone unnoticed. On NHANES
19,747 of 19,749 rows satisfy them to within 2%, so the reconstruction is
essentially exact everywhere. Fixing the data turned a latent leak into a
total one.

Two checks, because they fail differently:

* ``closed_form`` walks the identities above and measures how well each
  recovers Hb. It is exact, needs no fitting, and names the mechanism.
* ``empirical`` fits a model from the matrix to Hb on a train split and
  measures recovery on held-out rows. It catches routes nobody enumerated,
  including ones that only work statistically -- MCHC varies little between
  people, so MCV x RBC alone approximates Hb without any identity being
  involved.

The metric that matters is neither R^2 nor error in g/dL: it is whether the
reconstructed haemoglobin reproduces the *WHO severity label*. A model that
recovers Hb to 0.3 g/dL has not leaked a number, it has leaked the answer.
``label_recovery`` measures that directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from units import G_PER_DL, Quantity

from .data.analytes import conversion_factor
from .data.coherence import (
    EXACT_IDENTITY_TOLERANCE,
    HB_HCT_TO_MCHC,
    HB_RBC_TO_MCH,
    HCT_RBC_TO_MCV,
)

#: Reconstruction counts as successful at the same tolerance the coherence
#: filter uses. Using a looser one here would let a leak pass a test that the
#: data itself would fail.
RECONSTRUCTION_TOLERANCE = EXACT_IDENTITY_TOLERANCE


@dataclass(frozen=True)
class Path:
    """One closed-form route from matrix columns back to haemoglobin."""

    name: str
    #: Columns it needs, with the unit each must be in.
    inputs: tuple
    expression: str
    fn: object


# The bridges come from coherence.py rather than being retyped: these are
# the inverses of the identities the filter checks, and a constant that
# disagreed between the two modules would mean the audit was testing a
# different arithmetic from the one the data was filtered on.
def _hb_from_mch(v):
    # MCH[pg] = Hb[g/dL]/RBC[million/uL] x 10
    return v["mch"] * v["rbc"] / HB_RBC_TO_MCH


def _hb_from_mchc_mcv(v):
    # HCT = MCV x RBC / 10, then Hb = MCHC x HCT / 100
    return v["mchc"] * (v["mcv"] * v["rbc"] / HCT_RBC_TO_MCV) / HB_HCT_TO_MCHC


def _hb_from_hb_per_rbc(v):
    return v["hb_per_rbc"] * v["rbc"]


def _hct_from_mcv(v):
    return v["mcv"] * v["rbc"] / HCT_RBC_TO_MCV


PATHS = (
    Path("Hb from MCH x RBC",
         (("mch", "pg"), ("rbc", "million/uL")),
         "Hb = MCH x RBC / 10", _hb_from_mch),
    Path("Hb from MCHC x MCV x RBC",
         (("mchc", "g/dL"), ("mcv", "fL"), ("rbc", "million/uL")),
         "Hb = MCHC x (MCV x RBC / 10) / 100", _hb_from_mchc_mcv),
    Path("Hb from hb_per_rbc x RBC",
         (("hb_per_rbc", None), ("rbc", "million/uL")),
         "Hb = hb_per_rbc x RBC  (exact: hb_per_rbc is Hb/RBC)",
         _hb_from_hb_per_rbc),
)

#: The same question for haematocrit, which is held out alongside Hb.
HCT_PATHS = (
    Path("HCT from MCV x RBC",
         (("mcv", "fL"), ("rbc", "million/uL")),
         "HCT = MCV x RBC / 10", _hct_from_mcv),
)

#: Closed-form routes from a CBC to ferritin: **there are none, and that is
#: the finding**, not an omission.
#:
#: A CBC panel is a closed arithmetic system -- ten reported columns tied by
#: five identities, which is why haemoglobin could not be held out of it.
#: Ferritin is outside that system. It is a different assay on a different
#: aliquot measuring a different protein, and no combination of red cell,
#: white cell or platelet columns produces it. The empirical check therefore
#: carries the whole weight here, and its job is the opposite of what it did
#: for haemoglobin: a *low* recovery rate is the result we want, because it
#: means the relationship is statistical rather than arithmetic.
FERRITIN_PATHS = ()

#: Interpreting the empirical numbers depends on which of those two worlds
#: the target is in, so the audit is told rather than left to guess.
ARITHMETIC = "arithmetic"      # a closed-form route is known to exist
STATISTICAL = "statistical"    # no route exists; recovery is signal, not leak


@dataclass(frozen=True)
class Recovery:
    """How well one route recovered a held-out value."""

    name: str
    expression: str
    evaluable: int
    within_tolerance: int
    median_error: float
    p95_error: float
    available: bool = True
    missing: tuple = ()

    @property
    def rate(self):
        if not self.evaluable:
            return float("nan")
        return self.within_tolerance / self.evaluable

    @property
    def leaks(self):
        return self.available and self.rate > 0.5


@dataclass(frozen=True)
class LeakReport:
    matrix: str
    closed_form: tuple
    hct_closed_form: tuple = ()
    empirical: dict = field(default_factory=dict)
    label_agreement: dict = field(default_factory=dict)
    target: str = "hb"
    kind: str = ARITHMETIC

    @property
    def leaked(self):
        if any(r.leaks for r in self.closed_form):
            return True
        # On a statistical target a high R^2 is the model working, not a
        # leak. Only a recovery tight enough to be arithmetic counts, and
        # "within the identity tolerance on most rows" is that line.
        return bool(self.empirical.get("leaks"))


def _relative_error(actual, predicted):
    import numpy as np

    denominator = actual.abs()
    error = (predicted - actual).abs() / denominator.where(denominator > 0)
    return error.where(np.isfinite(error)).dropna()


def _column_in(matrix, dataset, name, unit):
    """A matrix column in a named unit, converted through Quantity."""
    if name not in matrix.frame.columns:
        return None
    series = matrix.frame[name]
    if unit is None:
        return series
    canonical = matrix.units.get(name) or (dataset.units or {}).get(name)
    if canonical is None or canonical == unit:
        return series
    return series * conversion_factor(canonical, unit)


def closed_form(matrix, dataset, truth, paths=PATHS):
    """Walk each declared route and measure how well it recovers ``truth``.

    ``truth`` is the held-out column, in canonical units.
    """
    import numpy as np

    results = []
    for path in paths:
        columns, missing = {}, []
        for name, unit in path.inputs:
            series = _column_in(matrix, dataset, name, unit)
            if series is None:
                missing.append(name)
            else:
                columns[name] = series
        if missing:
            results.append(Recovery(
                path.name, path.expression, 0, 0, float("nan"), float("nan"),
                available=False, missing=tuple(missing)))
            continue

        predicted = path.fn(columns)
        error = _relative_error(truth, predicted)
        results.append(Recovery(
            path.name, path.expression,
            int(len(error)),
            int((error <= RECONSTRUCTION_TOLERANCE).sum()),
            float(error.median()) if len(error) else float("nan"),
            float(np.percentile(error, 95)) if len(error) else float("nan"),
        ))
    return tuple(results)


def empirical(matrix, truth, test_size=0.3, seed=0):
    """Fit matrix -> haemoglobin on a train split, score on the rest.

    Deliberately a gradient-boosted tree rather than a linear model: the
    reconstructions are products, which a linear model cannot represent, and
    the question is whether *any* learnable route exists, not whether a
    particular one does.
    """
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import train_test_split

    frame = matrix.frame
    usable = truth.notna()
    X, y = frame[usable], truth[usable]
    if len(X) < 50:
        return {"available": False, "reason": "too few rows"}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    model = HistGradientBoostingRegressor(random_state=seed)
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    error = np.abs(predicted - y_test) / np.abs(y_test)
    within = float((error <= RECONSTRUCTION_TOLERANCE).mean())
    return {
        "available": True,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "median_relative_error": float(np.median(error)),
        "p95_relative_error": float(np.percentile(error, 95)),
        "mae_g_per_dl": float(np.abs(predicted - y_test).mean()),
        "within_tolerance": within,
        "r2": float(model.score(X_test, y_test)),
        "leaks": within > 0.5,
        "predicted": predicted,
        "index": y_test.index,
        "actual": y_test,
    }


def label_recovery(empirical_result, meta, allow_over_65=False):
    """Do the reconstructed haemoglobins reproduce the WHO severity labels?

    The measure that actually decides whether the task is leaked. An error of
    0.3 g/dL sounds small until it is compared against a mild band 0.9 g/dL
    wide; conversely a model can be wrong about the number and still right
    about the class. This asks the question the label asks.
    """
    from .labels import Unlabelled, band_for

    if not empirical_result.get("available"):
        return {"available": False}

    index = empirical_result["index"]
    actual = empirical_result["actual"]
    predicted = empirical_result["predicted"]
    rows = meta.reindex(index)

    agree = total = 0
    per_class = {}
    for position, key in enumerate(index):
        row = rows.loc[key]
        pregnancy = row.get("pregnancy_code")
        band, _ = band_for(
            row.get("age_years"), row.get("sex"),
            pregnancy == 1, pregnancy != 3, allow_over_65,
        )
        if isinstance(band, Unlabelled):
            continue
        true_label = band.classify(Quantity(float(actual.iloc[position]), G_PER_DL))
        predicted_label = band.classify(
            Quantity(float(predicted[position]), G_PER_DL))
        total += 1
        correct = int(true_label is predicted_label)
        agree += correct
        bucket = per_class.setdefault(true_label.value, [0, 0])
        bucket[0] += correct
        bucket[1] += 1

    if not total:
        return {"available": False}

    # CLAUDE.md hard rule: never a bare accuracy. 91.7% of labelled rows are
    # "none", so an agreement of 97% is six points over saying "none" every
    # time -- and the classes that matter are the rare ones.
    majority = max(n for _, n in per_class.values())
    return {
        "available": True,
        "n": total,
        "agreement": agree / total,
        "majority_baseline": majority / total,
        "recall": {k: (c / n) for k, (c, n) in sorted(per_class.items())},
        "support": {k: n for k, (_, n) in sorted(per_class.items())},
    }


def iron_label_recovery(empirical_result, dataset):
    """Do reconstructed ferritins reproduce the WHO 2020 iron-status label?

    The ferritin counterpart of ``label_recovery``. Inflammation comes from
    the row's *measured* CRP, not from the reconstruction -- the question is
    whether the CBC recovers ferritin well enough to reproduce the label,
    not whether it also recovers CRP.
    """
    from .iron import UnlabelledIron, band_for_age, is_inflamed
    from units import MICROGRAM_PER_L, MILLIGRAM_PER_L

    if not empirical_result.get("available"):
        return {"available": False}

    index = empirical_result["index"]
    actual = empirical_result["actual"]
    predicted = empirical_result["predicted"]
    meta = dataset.meta.reindex(index)
    crp = dataset.values["crp"].reindex(index)

    agree = total = 0
    per_class = {}
    for position, key in enumerate(index):
        band = band_for_age(meta.loc[key, "age_years"])
        if isinstance(band, UnlabelledIron):
            continue
        crp_value = crp.iloc[position]
        if crp_value != crp_value:
            continue
        inflamed = is_inflamed(Quantity(crp_value, MILLIGRAM_PER_L))

        true_label = band.classify(
            Quantity(float(actual.iloc[position]), MICROGRAM_PER_L), inflamed)
        predicted_label = band.classify(
            Quantity(float(predicted[position]), MICROGRAM_PER_L), inflamed)
        total += 1
        correct = int(true_label is predicted_label)
        agree += correct
        bucket = per_class.setdefault(true_label.value, [0, 0])
        bucket[0] += correct
        bucket[1] += 1

    if not total:
        return {"available": False}

    majority = max(n for _, n in per_class.values())
    return {
        "available": True,
        "n": total,
        "agreement": agree / total,
        "majority_baseline": majority / total,
        "recall": {k: (c / n) for k, (c, n) in sorted(per_class.items())},
        "support": {k: n for k, (_, n) in sorted(per_class.items())},
    }


def audit_target(matrix, dataset, target, paths=(), kind=ARITHMETIC,
                 label_recovery_fn=None, **kwargs):
    """Run the closed-form and empirical checks for one held-out target."""
    truth = dataset.values[target]
    result = empirical(matrix, truth)
    return LeakReport(
        matrix=matrix.name,
        target=target,
        kind=kind,
        closed_form=closed_form(matrix, dataset, truth, paths) if paths else (),
        empirical=result,
        label_agreement=(
            label_recovery_fn(result, dataset, **kwargs)
            if label_recovery_fn else {}),
    )


def audit(matrix, dataset, allow_over_65=False):
    """The haemoglobin audit: closed-form routes, learned fit, label recovery."""
    truth_hb = dataset.values["hb"]
    truth_hct = dataset.values.get("hematocrit")

    result = empirical(matrix, truth_hb)
    return LeakReport(
        matrix=matrix.name,
        target="hb",
        kind=ARITHMETIC,
        closed_form=closed_form(matrix, dataset, truth_hb, PATHS),
        hct_closed_form=(
            closed_form(matrix, dataset, truth_hct, HCT_PATHS)
            if truth_hct is not None else ()),
        empirical=result,
        label_agreement=label_recovery(result, dataset.meta, allow_over_65),
    )


def audit_ferritin(matrix, dataset):
    """The ferritin audit. Same checks, opposite expectation.

    There is no closed-form route to declare, so the empirical check decides.
    A high R^2 here is the model working; only recovery tight enough to be
    arithmetic would be a leak.
    """
    return audit_target(
        matrix, dataset, "ferritin",
        paths=FERRITIN_PATHS, kind=STATISTICAL,
        label_recovery_fn=iron_label_recovery,
    )


def format_report(report):
    """ASCII summary, for the cp1252 console."""
    heading = "Leakage audit: {0}  (target: {1})".format(
        report.matrix, report.target)
    lines = [heading, "=" * len(heading)]

    if report.kind == STATISTICAL:
        lines.append("  Target is OUTSIDE the CBC's arithmetic. No closed-form")
        lines.append("  route exists to declare, so the empirical check decides,")
        lines.append("  and a high R^2 here is signal rather than leakage. Only")
        lines.append("  recovery tight enough to be arithmetic would be a leak.")
        lines.append("")

    def block(title, recoveries):
        lines.append("  {0}".format(title))
        if not recoveries:
            lines.append("    none declared -- no identity produces this "
                         "target from a CBC")
            return
        for r in recoveries:
            if not r.available:
                lines.append("    {0:<32} not available (needs {1})".format(
                    r.name, ", ".join(r.missing)))
                continue
            lines.append(
                "    {0:<32} {1:6.2f}% of {2} rows within {3:.0%}  "
                "(median err {4:.3f}%)  {5}".format(
                    r.name, r.rate * 100, r.evaluable,
                    RECONSTRUCTION_TOLERANCE, r.median_error * 100,
                    "LEAK" if r.leaks else "ok"))

    block("Closed-form routes to {0}:".format(report.target),
          report.closed_form)
    if report.hct_closed_form:
        lines.append("")
        block("Closed-form routes to haematocrit:", report.hct_closed_form)
    lines.append("")

    e = report.empirical
    lines.append("  Learned reconstruction (HistGradientBoosting, held-out split)")
    if not e.get("available"):
        lines.append("    unavailable: {0}".format(e.get("reason")))
    else:
        lines.append("    n_train={0}  n_test={1}".format(
            e["n_train"], e["n_test"]))
        lines.append("    R^2                       {0:.4f}".format(e["r2"]))
        lines.append("    median relative error     {0:.3f}%".format(
            e["median_relative_error"] * 100))
        lines.append("    MAE                       {0:.4f}".format(
            e["mae_g_per_dl"]))
        lines.append("    within {0:.0%} of true value  {1:.2f}%   {2}".format(
            RECONSTRUCTION_TOLERANCE, e["within_tolerance"] * 100,
            "LEAK" if e["leaks"] else "ok"))

    a = report.label_agreement
    if a.get("available"):
        lines.append("")
        lines.append("  Label recovered from the reconstructed target")
        lines.append("    majority-class baseline    {0:.2f}%".format(
            a["majority_baseline"] * 100))
        lines.append("    agreement                  {0:.2f}%  of {1} rows "
                     "(+{2:.2f} over baseline)".format(
                         a["agreement"] * 100, a["n"],
                         (a["agreement"] - a["majority_baseline"]) * 100))
        lines.append("    per-class recall:")
        for label, recall in a["recall"].items():
            lines.append("      {0:<12}{1:>8.2f}%   n={2}".format(
                label, recall * 100, a["support"][label]))
    return "\n".join(lines)
