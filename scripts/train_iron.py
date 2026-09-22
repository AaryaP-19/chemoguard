"""Train and evaluate the iron-deficiency classifier on NHANES.

Usage, from the repository root::

    python scripts/train_iron.py                  # 2017-2020, both label rules
    python scripts/train_iron.py --cycles pooled  # sensitivity: both cycles
    python scripts/train_iron.py --adjustment adjusted

Default cohort is **2017-2020 only**. The 2021-2023 ferritin subsample is
93% female with nobody over 49, and sex and age are features of this model,
so pooling puts a cycle-demographic confound directly on the inputs. The
pooled fit runs under ``--cycles pooled`` and is a sensitivity analysis.

Both inflammation-adjustment settings are evaluated by default, because
neither is authoritative and reporting one alone overstates what is known.

No operating point is chosen. The threshold table is the deliverable.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chemoguard import training  # noqa: E402
from chemoguard.data.nhanes import (  # noqa: E402
    DEFAULT_CACHE,
    load_nhanes,
    with_iron_markers,
)
from chemoguard.features import build_iron_matrix  # noqa: E402
from chemoguard.iron import (  # noqa: E402
    SENSITIVITY_CYCLES,
    TRAINING_CYCLES,
    label_iron_status,
)

BAR = "-" * 78


def build(dataset, adjust, age_aware=False):
    """Matrix, target and metadata for one adjustment setting."""
    report = label_iron_status(dataset, adjust_for_inflammation=adjust)
    matrix = build_iron_matrix(dataset, age_aware=age_aware)
    mask = report.status.notna().to_numpy()
    y = (report.status[mask] == "deficient").astype(int)
    return report, matrix.frame[mask], y, dataset.meta[mask]


def show_fold_metrics(report):
    auc, auc_sd = report.mean_std("auc")
    acc, acc_sd = report.mean_std("accuracy")
    brier, brier_sd = report.mean_std("brier")

    print("  Headline (mean +- sd over {0} folds)".format(len(report.folds)))
    print("    ROC AUC                 {0:.4f} +- {1:.4f}   <- headline".format(
        auc, auc_sd))
    print("    accuracy                {0:.4f} +- {1:.4f}".format(acc, acc_sd))
    print("    majority baseline       {0:.4f}            (+{1:.2f} pts)".format(
        report.baseline, (acc - report.baseline) * 100))
    print("    Brier score             {0:.4f} +- {1:.4f}".format(
        brier, brier_sd))
    print()
    print("  Per class at threshold 0.50")
    print("    {0:<12}{1:>16}{2:>16}{3:>16}{4:>8}".format(
        "class", "precision", "recall", "F1", "n"))
    for label in ("deficient", "replete"):
        row = []
        for metric in ("precision", "recall", "f1"):
            mean, sd = report.mean_std(metric, label)
            row.append("{0:.3f} +- {1:.3f}".format(mean, sd))
        support, _ = report.mean_std("support", label)
        print("    {0:<12}{1:>16}{2:>16}{3:>16}{4:>8.0f}".format(
            label, row[0], row[1], row[2], support))
    print()

    tn = sum(f.confusion[0] for f in report.folds)
    fp = sum(f.confusion[1] for f in report.folds)
    fn = sum(f.confusion[2] for f in report.folds)
    tp = sum(f.confusion[3] for f in report.folds)
    total = tn + fp + fn + tp
    print("  Confusion matrix, summed over all folds (n={0}, {1} seeds)".format(
        total, len(training.SEEDS)))
    print("                      predicted replete   predicted deficient")
    print("    actual replete    {0:>17,}   {1:>19,}".format(tn, fp))
    print("    actual deficient  {0:>17,}   {1:>19,}".format(fn, tp))
    print()


def show_reliability(report):
    print("  Reliability (out-of-fold, averaged over seeds)")
    print("    {0:<14}{1:>8}{2:>12}{3:>12}   {4}".format(
        "bin", "n", "predicted", "observed", "calibration"))
    for row in training.reliability(report):
        if not row["n"]:
            continue
        gap = row["observed"] - row["predicted"]
        bar = "#" * min(20, int(abs(gap) * 100))
        direction = "under" if gap > 0 else "over "
        print("    {0:<14}{1:>8}{2:>12.3f}{3:>12.3f}   {4} {5}{6}".format(
            "{0:.1f}-{1:.1f}".format(row["low"], row["high"]),
            row["n"], row["predicted"], row["observed"],
            direction if abs(gap) > 0.01 else "ok   ", bar,
            "" if abs(gap) <= 0.01 else " {0:+.3f}".format(gap)))
    print()


def show_pr_curve(report):
    curve = training.precision_recall_points(report)
    print("  Precision/recall curve (out-of-fold)")
    print("    average precision {0:.4f}   no-skill floor {1:.4f}".format(
        curve["average_precision"], curve["no_skill"]))
    print("    {0:>8}{1:>12}   {2}".format("recall", "precision", ""))
    for recall, precision in curve["points"]:
        bar = "#" * int(precision * 50)
        print("    {0:>8.2f}{1:>12.3f}   {2}".format(recall, precision, bar))
    print()


def show_thresholds(report):
    print("  Candidate operating points -- NOT a recommendation")
    print("    {0:>8}{1:>12}{2:>10}{3:>12}{4:>10}{5:>10}".format(
        "target", "threshold", "recall", "precision", "FPR", "flagged"))
    for row in training.threshold_table(report):
        print("    {0:>8.0%}{1:>12.4f}{2:>10.3f}{3:>12.3f}{4:>10.3f}"
              "{5:>9.1f}%".format(
                  row["target_recall"], row["threshold"], row["recall"],
                  row["precision"], row["false_positive_rate"],
                  row["flagged_share"] * 100))
    print()
    print("    Choosing among these rows trades missed iron deficiency against")
    print("    over-referral. That is a clinical judgement about relative harm,")
    print("    and this project is not positioned to make it. No default is set.")
    print()


def show_subgroups(report):
    groups = training.subgroups(report)
    if not groups:
        return
    print("  Subgroup breakdown (out-of-fold, threshold 0.50)")
    print("    AUC with Hanley-McNeil 95% interval. The interval is driven by")
    print("    the number of CASES, not the number of rows.")
    for kind in ("sex", "age_band"):
        print("    by {0}".format(kind.replace("_", " ")))
        print("      {0:<10}{1:>7}{2:>7}{3:>9}{4:>9}{5:>20}{6:>9}".format(
            "group", "n", "cases", "pos rate", "AUC", "95% CI", "recall"))
        for row in groups[kind]:
            if "note" in row:
                print("      {0:<10}{1:>7}   {2}".format(
                    row["group"], row["n"], row["note"]))
                continue
            print("      {0:<10}{1:>7}{2:>7}{3:>9.3f}{4:>9.4f}"
                  "{5:>20}{6:>9.3f}".format(
                      row["group"], row["n"], row["n_positive"],
                      row["positive_rate"], row["auc"],
                      "[{0:.3f}, {1:.3f}]".format(
                          row["ci_low"], row["ci_high"]),
                      row["recall"]))
    print()
    print("    Which groups are NOT distinguishable from each other")
    print("    (overlapping 95% intervals). Comparing only against the")
    print("    best-powered band would be misleading: its interval is so")
    print("    tight that everything separates from it.")
    scored = [r for r in groups["age_band"] + groups["sex"] if "note" not in r]
    for row in scored:
        others = [
            other["group"] for other in scored
            if other is not row and training.intervals_overlap(row, other)
        ]
        print("      {0:<10}{1}".format(
            row["group"],
            ", ".join(others) if others else "separated from all others"))
    print()


def show_leakage(result):
    print("  Depth-3 leakage test (AUC, not accuracy)")
    print("    ensemble AUC            {0:.4f} +- {1:.4f}".format(
        result["ensemble_auc"], result["ensemble_auc_std"]))
    print("    depth-3 tree AUC        {0:.4f} +- {1:.4f}".format(
        result["depth3_auc"], result["depth3_auc_std"]))
    print("    gap                     {0:.4f}   (margin {1:.2f})".format(
        result["gap"], result["margin"]))
    print("    verdict                 {0}".format(
        "LABEL IS A RULE" if result["label_is_a_rule"]
        else "passes -- the ensemble is doing more than thresholding"))
    print()


def compare_age_aware(dataset):
    """Does an age-normalised MCV feature rescue the under-5 subgroup?

    Same cohort, same label, same estimator, same seeds. The only
    difference is one feature. If the under-5 AUC does not move, the weak
    band is a power problem and should be recorded as one.
    """
    print(BAR)
    print("AGE-NORMALISED MCV: does it move the under-5 band?")
    print(BAR)
    print("  MCV lower limit of normal = 70 + age(years) below 10, else 80.")
    print("  Koerper & Dallman, J Pediatr 1976 (PMID 956999).")
    print()

    results = {}
    for age_aware in (False, True):
        _, X, y, meta = build(dataset, True, age_aware)
        report = training.cross_validate(
            X, y, name="age_aware={0}".format(age_aware), meta=meta)
        groups = training.subgroups(report)
        auc, sd = report.mean_std("auc")
        results[age_aware] = {
            "overall": (auc, sd),
            "bands": {r["group"]: r for r in groups["age_band"]
                      if "note" not in r},
            "columns": len(X.columns),
        }

    print("  {0:<14}{1:>22}{2:>22}".format(
        "band", "without mcv_vs_age_lln", "with mcv_vs_age_lln"))
    base, aware = results[False], results[True]
    print("  {0:<14}{1:>22}{2:>22}".format(
        "OVERALL",
        "{0:.4f} +- {1:.4f}".format(*base["overall"]),
        "{0:.4f} +- {1:.4f}".format(*aware["overall"])))
    for band in ("1-4", "5-19", "20-59", "60+"):
        if band not in base["bands"] or band not in aware["bands"]:
            continue
        b, a = base["bands"][band], aware["bands"][band]
        print("  {0:<14}{1:>22}{2:>22}".format(
            band,
            "{0:.4f} [{1:.3f},{2:.3f}]".format(
                b["auc"], b["ci_low"], b["ci_high"]),
            "{0:.4f} [{1:.3f},{2:.3f}]".format(
                a["auc"], a["ci_low"], a["ci_high"])))

    b, a = base["bands"].get("1-4"), aware["bands"].get("1-4")
    if b and a:
        shift = a["auc"] - b["auc"]
        print()
        print("  under-5 AUC shift: {0:+.4f}  ({1} cases, "
              "half-width {2:.3f})".format(
                  shift, b["n_positive"], (b["ci_high"] - b["ci_low"]) / 2))
        print("  VERDICT: {0}".format(
            "the age-scaling hypothesis is supported"
            if shift > (b["ci_high"] - b["ci_low"]) / 2
            else "shift is inside the interval -- the limitation is sample "
                 "size, not age scaling"))
    print()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--cycles", choices=("training", "pooled"),
                        default="training")
    parser.add_argument("--adjustment",
                        choices=("both", "adjusted", "unadjusted"),
                        default="both")
    parser.add_argument("--age-aware", action="store_true",
                        help="add the age-normalised MCV feature")
    parser.add_argument("--compare-age-aware", action="store_true",
                        help="run both feature sets and compare the under-5 band")
    args = parser.parse_args(argv)

    cycles = (TRAINING_CYCLES if args.cycles == "training"
              else SENSITIVITY_CYCLES)
    print("=" * 78)
    print("Iron deficiency from CBC -- cross-validated evaluation")
    print("=" * 78)
    print("cohort   : {0}{1}".format(
        ", ".join(cycles),
        "" if args.cycles == "training" else "   [SENSITIVITY ANALYSIS]"))
    if args.cycles == "pooled":
        print("           2021-2023 ferritin is 93% female, nobody over 49;")
        print("           sex and age are features, so this confound is on the")
        print("           inputs. Headline results use 2017-2020 only.")
    print("design   : stratified {0}-fold x {1} seeds = {2} fits".format(
        training.N_SPLITS, len(training.SEEDS),
        training.N_SPLITS * len(training.SEEDS)))
    print()

    dataset = with_iron_markers(
        load_nhanes(cycles=list(cycles), cache_dir=args.cache_dir),
        cache_dir=args.cache_dir)

    settings = ((True, "adjusted"), (False, "unadjusted"))
    if args.adjustment != "both":
        settings = tuple(s for s in settings if s[1] == args.adjustment)

    if args.compare_age_aware:
        return compare_age_aware(dataset)

    for adjust, name in settings:
        label_report, X, y, meta = build(dataset, adjust, args.age_aware)
        prevalence = label_report.prevalence

        print(BAR)
        print("LABEL RULE: inflammation adjustment {0}".format(name.upper()))
        print(BAR)
        print("  labelled rows           {0:,}".format(len(y)))
        print("  deficient               {0:,}  ({1:.2f}%)".format(
            int(y.sum()), y.mean() * 100))
        print("  majority baseline       {0:.2f}%".format(
            max(y.mean(), 1 - y.mean()) * 100))
        print("  prevalence adjusted     {0:.2f}%".format(
            prevalence["rate_adjusted"] * 100))
        print("  prevalence unadjusted   {0:.2f}%".format(
            prevalence["rate_unadjusted"] * 100))
        print("    Neither is authoritative. WHO reports high variability across")
        print("    adjustment methods, and BRINDA is unavailable without AGP.")
        print()

        report = training.cross_validate(
            X, y, name="{0}/{1}".format(args.cycles, name), meta=meta)
        show_fold_metrics(report)
        show_leakage(training.leakage_probe(X, y))
        show_reliability(report)
        show_pr_curve(report)
        show_thresholds(report)
        show_subgroups(report)

    print(BAR)
    print("These are research metrics on public survey data. Nothing here is")
    print("clinically validated, and no threshold is endorsed for use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
