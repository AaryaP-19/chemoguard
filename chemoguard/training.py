"""Cross-validated evaluation of the iron-deficiency classifier.

Repeated stratified 5-fold, five seeds, 25 fits. Every metric is reported as
mean +- standard deviation across those folds, because a single split on a
13.5% positive class moves by more than the effects being measured.

**AUC is the headline metric and accuracy is nearly meaningless here.** The
majority class is 86.5%, so a model that predicts "replete" for everyone
scores 86.5%; the whole accessible range is 13.5 points wide and most of what
looks like performance is the base rate. Accuracy is still reported, always
beside its baseline, so that a +3 can be read as the +3 it is.

The depth-3 leakage test moved to AUC for the same reason -- see
``LEAKAGE_AUC_MARGIN``.

Nothing here picks an operating point. ``threshold_table`` lays out
candidates with what each one costs and buys, and stops there: choosing
between missing deficient patients and over-referring healthy ones is a
clinical judgement about harm, and this project is not positioned to make it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Folds and seeds. 5 x 5 = 25 fits per configuration.
N_SPLITS = 5
SEEDS = (0, 1, 2, 3, 4)

#: The depth-3 leakage test, in AUC rather than accuracy.
#:
#: CLAUDE.md's hard rule used to read "within 1% of the ensemble" on
#: accuracy. On a 13.5% positive class that is the wrong scale: the observed
#: gap was 1.27 accuracy points, which is inside fold-to-fold noise, while
#: the same two models were 0.061 apart in AUC, which is not. Accuracy is
#: compressed against the base rate; AUC is not.
#:
#: 0.02 is set above the fold-to-fold standard deviation of the AUC estimate
#: (measured at roughly 0.01-0.02 over 25 folds) so that ordinary variation
#: cannot trip it, and far below the 0.061 gap a genuinely learnable target
#: shows.
LEAKAGE_AUC_MARGIN = 0.02

#: Candidate operating points, as recall targets. Deliberately a spread
#: rather than a recommendation.
RECALL_TARGETS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

#: Age bands for the subgroup breakdown. WHO's ferritin bands, collapsed to
#: the four that carry enough rows to report.
AGE_BANDS = (
    ("1-4", 1, 4),
    ("5-19", 5, 19),
    ("20-59", 20, 59),
    ("60+", 60, 200),
)


@dataclass(frozen=True)
class FoldMetrics:
    """Everything measured on one held-out fold."""

    seed: int
    fold: int
    auc: float
    accuracy: float
    baseline: float
    brier: float
    precision: dict
    recall: dict
    f1: dict
    support: dict
    confusion: tuple      # (tn, fp, fn, tp)


@dataclass(frozen=True)
class TrainingReport:
    name: str
    n: int
    positive_rate: float
    baseline: float
    folds: tuple
    #: Out-of-fold predicted probability, averaged over seeds.
    oof_probability: object = None
    truth: object = None
    meta: object = None
    leakage: dict = field(default_factory=dict)
    notes: tuple = field(default_factory=tuple)

    def mean_std(self, attribute, key=None):
        import numpy as np

        values = []
        for fold in self.folds:
            value = getattr(fold, attribute)
            values.append(value[key] if key is not None else value)
        values = np.asarray(values, dtype=float)
        return float(values.mean()), float(values.std(ddof=1))


def _estimator(seed):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(random_state=seed)


def _depth3(seed):
    from sklearn.tree import DecisionTreeClassifier

    return DecisionTreeClassifier(max_depth=3, random_state=seed)


def _fold_metrics(seed, fold, y_true, y_prob, threshold=0.5):
    import numpy as np
    from sklearn.metrics import (
        brier_score_loss,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    names = ("replete", "deficient")

    return FoldMetrics(
        seed=seed,
        fold=fold,
        auc=float(roc_auc_score(y_true, y_prob)),
        accuracy=float((y_pred == y_true).mean()),
        baseline=float(max(np.mean(y_true), 1 - np.mean(y_true))),
        brier=float(brier_score_loss(y_true, y_prob)),
        precision=dict(zip(names, precision)),
        recall=dict(zip(names, recall)),
        f1=dict(zip(names, f1)),
        support=dict(zip(names, support)),
        confusion=(int(tn), int(fp), int(fn), int(tp)),
    )


def cross_validate(X, y, name="iron-deficiency", meta=None, seeds=SEEDS,
                   n_splits=N_SPLITS):
    """Repeated stratified k-fold. Returns a ``TrainingReport``.

    Each seed produces a complete out-of-fold probability for every row; the
    reported ``oof_probability`` is the mean across seeds, which is what the
    calibration curve, the subgroup breakdown and the threshold table are
    computed on. Fold-level metrics stay per-fold so their spread is real.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y).astype(int)
    folds, oof_per_seed = [], []

    for seed in seeds:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed)
        oof = np.full(len(y), np.nan)

        for index, (train, test) in enumerate(splitter.split(X, y)):
            model = _estimator(seed)
            model.fit(X.iloc[train], y[train])
            probability = model.predict_proba(X.iloc[test])[:, 1]
            oof[test] = probability
            folds.append(_fold_metrics(seed, index, y[test], probability))

        oof_per_seed.append(oof)

    return TrainingReport(
        name=name,
        n=int(len(y)),
        positive_rate=float(y.mean()),
        baseline=float(max(y.mean(), 1 - y.mean())),
        folds=tuple(folds),
        oof_probability=np.mean(oof_per_seed, axis=0),
        truth=y,
        meta=meta,
    )


def leakage_probe(X, y, seeds=SEEDS, n_splits=N_SPLITS):
    """The depth-3 leakage test, scored on AUC.

    If a three-level tree comes within ``LEAKAGE_AUC_MARGIN`` of the full
    ensemble, the label is reachable by a handful of threshold comparisons
    and is a rule rather than a target. That is what happened to the anaemia
    label in Finding 8.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y).astype(int)
    scores = {"ensemble": [], "depth3": []}

    for seed in seeds:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed)
        for train, test in splitter.split(X, y):
            for key, build in (("ensemble", _estimator), ("depth3", _depth3)):
                model = build(seed)
                model.fit(X.iloc[train], y[train])
                probability = model.predict_proba(X.iloc[test])[:, 1]
                scores[key].append(roc_auc_score(y[test], probability))

    ensemble = np.asarray(scores["ensemble"])
    depth3 = np.asarray(scores["depth3"])
    gap = float(ensemble.mean() - depth3.mean())

    return {
        "ensemble_auc": float(ensemble.mean()),
        "ensemble_auc_std": float(ensemble.std(ddof=1)),
        "depth3_auc": float(depth3.mean()),
        "depth3_auc_std": float(depth3.std(ddof=1)),
        "gap": gap,
        "margin": LEAKAGE_AUC_MARGIN,
        "label_is_a_rule": gap < LEAKAGE_AUC_MARGIN,
    }


def reliability(report, bins=10):
    """Calibration curve on the out-of-fold probabilities."""
    import numpy as np

    probability = report.oof_probability
    truth = report.truth
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (
            probability < high if high < 1.0 else probability <= high)
        if not mask.any():
            rows.append({"low": low, "high": high, "n": 0})
            continue
        rows.append({
            "low": float(low), "high": float(high),
            "n": int(mask.sum()),
            "predicted": float(probability[mask].mean()),
            "observed": float(truth[mask].mean()),
        })
    return rows


def threshold_table(report, recall_targets=RECALL_TARGETS):
    """Candidate operating points and what each one costs.

    Deliberately a table and not a choice. Each row is the threshold that
    first reaches a recall target on the out-of-fold predictions, with the
    false-positive rate and precision that come with it.
    """
    import numpy as np
    from sklearn.metrics import precision_recall_curve

    probability = report.oof_probability
    truth = report.truth
    precision, recall, thresholds = precision_recall_curve(truth, probability)

    negatives = int((truth == 0).sum())
    rows = []
    for target in recall_targets:
        # precision_recall_curve returns recall descending; find the highest
        # threshold that still achieves the target.
        eligible = np.where(recall[:-1] >= target)[0]
        if not len(eligible):
            continue
        index = eligible[-1]
        cutoff = float(thresholds[index])
        predicted = probability >= cutoff
        false_positives = int((predicted & (truth == 0)).sum())
        rows.append({
            "target_recall": target,
            "threshold": cutoff,
            "recall": float(recall[index]),
            "precision": float(precision[index]),
            "false_positive_rate": false_positives / negatives,
            "flagged": int(predicted.sum()),
            "flagged_share": float(predicted.mean()),
        })
    return rows


def precision_recall_points(report, n=11):
    """A coarse precision/recall curve, as points, for printing."""
    import numpy as np
    from sklearn.metrics import average_precision_score, precision_recall_curve

    precision, recall, _ = precision_recall_curve(
        report.truth, report.oof_probability)
    wanted = np.linspace(0.0, 1.0, n)
    points = []
    for target in wanted:
        index = int(np.argmin(np.abs(recall - target)))
        points.append((float(recall[index]), float(precision[index])))
    return {
        "points": points,
        "average_precision": float(average_precision_score(
            report.truth, report.oof_probability)),
        "no_skill": float(np.mean(report.truth)),
    }


def hanley_mcneil_se(auc, n_positive, n_negative):
    """Standard error of an AUC, Hanley & McNeil (1982).

        SE = sqrt( [A(1-A) + (n1-1)(Q1 - A^2) + (n2-1)(Q2 - A^2)] / (n1 n2) )
        Q1 = A / (2 - A)        Q2 = 2A^2 / (1 + A)

    The exponential-distribution approximation, which is the standard
    closed form and slightly conservative. It exists here because subgroup
    AUCs are read as if they were comparable, and on a rare positive class
    they are not: the interval is driven by the number of *cases*, not the
    number of rows. A band with 683 rows and 37 cases carries an interval
    several times wider than one with 4,862 rows and 700 cases, and quoting
    the two point estimates side by side hides that entirely.
    """
    import math

    if not n_positive or not n_negative:
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    variance = (
        auc * (1 - auc)
        + (n_positive - 1) * (q1 - auc * auc)
        + (n_negative - 1) * (q2 - auc * auc)
    ) / (n_positive * n_negative)
    return math.sqrt(max(variance, 0.0))


def auc_with_interval(y_true, scores, z=1.96):
    """AUC plus a Hanley-McNeil 95% interval and the case count driving it."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    n_positive = int((y_true == 1).sum())
    n_negative = int((y_true == 0).sum())
    if not n_positive or not n_negative:
        return {"auc": float("nan"), "n_positive": n_positive,
                "n_negative": n_negative}

    auc = float(roc_auc_score(y_true, scores))
    se = hanley_mcneil_se(auc, n_positive, n_negative)
    return {
        "auc": auc,
        "se": se,
        "ci_low": max(0.0, auc - z * se),
        "ci_high": min(1.0, auc + z * se),
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def intervals_overlap(a, b):
    """Whether two AUC intervals overlap -- i.e. are not distinguishable."""
    if any(k not in a or k not in b for k in ("ci_low", "ci_high")):
        return None
    return not (a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"])


def subgroups(report, threshold=0.5):
    """AUC and recall by sex and by age band, on out-of-fold predictions.

    A model whose discrimination holds overall but collapses in one band is
    not a model that works; the aggregate just hid it.
    """
    import numpy as np

    if report.meta is None:
        return {}

    probability = report.oof_probability
    truth = report.truth
    predicted = (probability >= threshold).astype(int)
    meta = report.meta

    def block(mask, label):
        mask = np.asarray(mask)
        if mask.sum() < 30:
            return {"group": label, "n": int(mask.sum()), "note": "too few rows"}
        y = truth[mask]
        if len(np.unique(y)) < 2:
            return {"group": label, "n": int(mask.sum()),
                    "note": "one class only"}
        positives = y == 1
        interval = auc_with_interval(y, probability[mask])
        return {
            "group": label,
            "n": int(mask.sum()),
            "positive_rate": float(y.mean()),
            "recall": float(predicted[mask][positives].mean()),
            "baseline": float(max(y.mean(), 1 - y.mean())),
            "accuracy": float((predicted[mask] == y).mean()),
            **interval,
        }

    results = {"sex": [], "age_band": []}
    sex = meta["sex"].to_numpy()
    for value in ("female", "male"):
        results["sex"].append(block(sex == value, value))

    age = meta["age_years"].to_numpy()
    for label, low, high in AGE_BANDS:
        results["age_band"].append(block((age >= low) & (age <= high), label))
    return results
