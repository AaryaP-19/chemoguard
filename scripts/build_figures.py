"""Render the report figures to reports/figures/ as PNG + CSV.

Usage, from the repository root::

    python scripts/build_figures.py
    python scripts/build_figures.py --only coherence

Every figure ships with the CSV it was drawn from, so a number in the write-up
can be traced to a row without rerunning the pipeline.

Figures:
  coherence_tolerance_sweep   the saturation curve -- the audit's best evidence
  reliability_curve           calibration of the iron classifier
  precision_recall_curve      with the no-skill floor drawn in
  subgroup_auc                AUC with Hanley-McNeil intervals
  confusion_matrix            at threshold 0.50

Palette is the validated two-slot categorical set (blue #2a78d6, orange
#eb6834): all-pairs CVD Delta E 24.7, normal-vision 33.6, both well clear of
the floors. Every chart with two series also carries a legend, so identity is
never colour alone.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUTPUT = os.path.join("reports", "figures")
DPI = 150

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dedddb"
MUTED = "#8a8985"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"


def _figure(width=7.2, height=4.6):
    figure, axes = plt.subplots(figsize=(width, height), dpi=DPI)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=INK_SOFT, labelsize=9)
    return figure, axes


def _title(axes, title, subtitle=None):
    """Left-aligned title with an optional subtitle above the axes.

    Subtitles are wrapped by hand with newlines rather than left to run:
    a long single-line subtitle widens the saved canvas (bbox_inches is
    "tight") and quietly changes the figure's aspect ratio.
    """
    lines = 0 if not subtitle else len(subtitle.splitlines())
    axes.set_title(title, color=INK, fontsize=12, fontweight="600",
                   loc="left", pad=10 + 12 * lines)
    if subtitle:
        axes.text(0.0, 1.015, subtitle, transform=axes.transAxes,
                  color=INK_SOFT, fontsize=9, va="bottom", linespacing=1.45)


def _save(figure, name):
    os.makedirs(OUTPUT, exist_ok=True)
    path = os.path.join(OUTPUT, name + ".png")
    figure.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    print("  wrote {0}".format(path))
    return path


def _csv(name, header, rows):
    os.makedirs(OUTPUT, exist_ok=True)
    path = os.path.join(OUTPUT, name + ".csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    print("  wrote {0}".format(path))
    return path


# ---------------------------------------------------------------------------
# 1. coherence tolerance sweep
# ---------------------------------------------------------------------------

def coherence_sweep(cache_dir):
    """Joint-identity survivors against tolerance, NHANES vs the legacy copies.

    The shape is the evidence. Real analyser data saturates at 100% almost
    immediately and stays there; the withdrawn dataset saturates at about 27%
    and is flat across a 40x change in tolerance, because its failures are
    not near-misses.
    """
    from chemoguard.data import coherence
    from chemoguard.data.legacy_audit import load_legacy
    from chemoguard.data.nhanes import load_nhanes

    print("coherence_tolerance_sweep")
    nhanes = load_nhanes(cache_dir=cache_dir)
    legacy = load_legacy(os.path.join(
        "Clinically Validated CBC Dataset of 7,196 Samples",
        "BDCBC7196_Hematology_Dataset.csv"))

    tolerances = np.concatenate([
        np.arange(0.0025, 0.02, 0.0025),
        np.arange(0.02, 0.205, 0.01),
    ])

    series, rows = {}, []
    for dataset in (nhanes, legacy):
        rates = []
        for tolerance in tolerances:
            scale = tolerance / coherence.EXACT_IDENTITY_TOLERANCE
            mask, evaluable = coherence._joint_mask(
                dataset.values, coherence.JOINT_IDENTITIES,
                tolerance_scale=scale)
            total = int(evaluable.sum())
            rate = (int(mask.sum()) / total * 100) if total else float("nan")
            rates.append(rate)
            rows.append([dataset.name, round(tolerance * 100, 4),
                         int(mask.sum()), total, round(rate, 4)])
        series[dataset.name] = rates

    figure, axes = _figure()
    labels = list(series)
    for colour, label in zip((SERIES_1, SERIES_2), labels):
        axes.plot(tolerances * 100, series[label], color=colour, linewidth=2,
                  marker="o", markersize=4, label=label, zorder=3)
    axes.axvline(coherence.EXACT_IDENTITY_TOLERANCE * 100, color=MUTED,
                 linewidth=1.2, linestyle="--", zorder=2)
    axes.text(coherence.EXACT_IDENTITY_TOLERANCE * 100 + 0.35, 50,
              "tolerance constant (2%)", color=INK_SOFT, fontsize=8,
              rotation=90, va="center")

    axes.set_xlabel("Relative tolerance on the exact identities (%)",
                    color=INK_SOFT, fontsize=10)
    axes.set_ylabel("Rows passing the joint filter (%)", color=INK_SOFT,
                    fontsize=10)
    axes.set_ylim(0, 108)
    _title(axes, "Real data saturates; synthetic data plateaus",
           "NHANES reaches 100% by 1%. The withdrawn dataset rises only\n"
           "27% to 33% across a 10x tolerance change: not near-misses.")
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc="center right")
    _save(figure, "coherence_tolerance_sweep")
    _csv("coherence_tolerance_sweep",
         ["dataset", "tolerance_percent", "survivors", "evaluable",
          "pass_rate_percent"], rows)


# ---------------------------------------------------------------------------
# iron classifier figures
# ---------------------------------------------------------------------------

def _iron_report(cache_dir, adjust=True):
    from chemoguard import training
    from chemoguard.data.nhanes import load_nhanes, with_iron_markers
    from chemoguard.features import build_iron_matrix
    from chemoguard.iron import TRAINING_CYCLES, label_iron_status

    dataset = with_iron_markers(
        load_nhanes(cycles=list(TRAINING_CYCLES), cache_dir=cache_dir),
        cache_dir=cache_dir)
    labels = label_iron_status(dataset, adjust_for_inflammation=adjust)
    matrix = build_iron_matrix(dataset)
    mask = labels.status.notna().to_numpy()
    y = (labels.status[mask] == "deficient").astype(int)
    return training.cross_validate(
        matrix.frame[mask], y, meta=dataset.meta[mask])


def reliability_curve(report):
    from chemoguard import training

    print("reliability_curve")
    rows = [r for r in training.reliability(report) if r["n"]]
    predicted = [r["predicted"] for r in rows]
    observed = [r["observed"] for r in rows]
    counts = [r["n"] for r in rows]

    figure, axes = _figure()
    axes.plot([0, 1], [0, 1], color=MUTED, linewidth=1.5, linestyle="--",
              zorder=2, label="perfect calibration")
    axes.plot(predicted, observed, color=SERIES_1, linewidth=2, zorder=3,
              label="observed")
    # Marker area carries the bin count; the smallest bins are the noisiest.
    sizes = 40 + 260 * np.sqrt(np.asarray(counts) / max(counts))
    axes.scatter(predicted, observed, s=sizes, color=SERIES_1,
                 edgecolor=SURFACE, linewidth=2, zorder=4)

    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_xlabel("Mean predicted probability", color=INK_SOFT, fontsize=10)
    axes.set_ylabel("Observed frequency of iron deficiency", color=INK_SOFT,
                    fontsize=10)
    _title(axes, "Calibration is close below 0.3 and over-confident above it",
           "Out-of-fold, averaged over 5 seeds. Marker area is the bin count.")
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc="upper left")
    _save(figure, "reliability_curve")
    _csv("reliability_curve",
         ["bin_low", "bin_high", "n", "mean_predicted", "observed_rate"],
         [[round(r["low"], 3), round(r["high"], 3), r["n"],
           round(r["predicted"], 5), round(r["observed"], 5)] for r in rows])


def precision_recall_curve(report):
    from sklearn.metrics import average_precision_score, precision_recall_curve

    print("precision_recall_curve")
    precision, recall, thresholds = precision_recall_curve(
        report.truth, report.oof_probability)
    no_skill = float(np.mean(report.truth))
    average = float(average_precision_score(
        report.truth, report.oof_probability))

    figure, axes = _figure()
    axes.axhline(no_skill, color=MUTED, linewidth=1.5, linestyle="--",
                 zorder=2, label="no-skill floor ({0:.3f})".format(no_skill))
    axes.plot(recall, precision, color=SERIES_1, linewidth=2, zorder=3,
              label="out-of-fold (AP {0:.3f})".format(average))
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1.02)
    axes.set_xlabel("Recall (of iron-deficient participants)", color=INK_SOFT,
                    fontsize=10)
    axes.set_ylabel("Precision", color=INK_SOFT, fontsize=10)
    _title(axes, "Precision/recall, iron deficiency",
           "Average precision {0:.3f} against a no-skill floor of "
           "{1:.3f}.".format(average, no_skill))
    axes.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT,
                loc="upper right")
    _save(figure, "precision_recall_curve")

    # Downsampled to a readable grid; the full curve has one point per
    # distinct score and would be tens of thousands of rows.
    step = max(1, len(recall) // 400)
    rows = [[round(float(recall[i]), 6), round(float(precision[i]), 6),
             ("" if i >= len(thresholds) else round(float(thresholds[i]), 6))]
            for i in range(0, len(recall), step)]
    _csv("precision_recall_curve", ["recall", "precision", "threshold"], rows)


def subgroup_auc(report):
    from chemoguard import training

    print("subgroup_auc")
    groups = training.subgroups(report)
    entries = []
    for kind in ("sex", "age_band"):
        for row in groups[kind]:
            if "note" in row:
                continue
            entries.append((kind, row))

    labels = ["{0}  (n={1}, {2} cases)".format(
        r["group"], r["n"], r["n_positive"]) for _, r in entries]
    aucs = [r["auc"] for _, r in entries]
    low = [r["auc"] - r["ci_low"] for _, r in entries]
    high = [r["ci_high"] - r["auc"] for _, r in entries]
    positions = np.arange(len(entries))[::-1]

    figure, axes = _figure(height=0.62 * len(entries) + 2.2)
    axes.axvline(0.5, color=MUTED, linewidth=1.2, linestyle="--", zorder=2)
    axes.text(0.503, positions.min() - 0.3, "chance", color=INK_SOFT,
              fontsize=8, va="center")
    colours = [SERIES_1 if kind == "sex" else SERIES_2 for kind, _ in entries]
    for position, auc, lo, hi, colour in zip(positions, aucs, low, high, colours):
        axes.errorbar(auc, position, xerr=[[lo], [hi]], fmt="o",
                      color=colour, ecolor=colour, elinewidth=2,
                      capsize=5, markersize=9,
                      markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        axes.text(auc, position + 0.28, "{0:.3f}".format(auc), color=INK,
                  fontsize=9, ha="center")

    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=9, color=INK)
    axes.set_xlim(0.4, 1.0)
    axes.set_xlabel("AUC with Hanley-McNeil 95% interval", color=INK_SOFT,
                    fontsize=10)
    axes.grid(axis="y", visible=False)
    handles = [
        plt.Line2D([], [], color=SERIES_1, marker="o", linestyle="",
                   markersize=8, label="by sex"),
        plt.Line2D([], [], color=SERIES_2, marker="o", linestyle="",
                   markersize=8, label="by age band"),
    ]
    axes.legend(handles=handles, frameon=False, fontsize=9,
                labelcolor=INK_SOFT, loc="lower right")
    _title(axes, "Subgroup AUC, with interval",
           "Interval width is set by case count, not row count. Under-5s\n"
           "carry 37 cases: an interval 6x wider than the 20-59 band's.")
    _save(figure, "subgroup_auc")
    _csv("subgroup_auc",
         ["kind", "group", "n", "n_positive", "n_negative", "positive_rate",
          "auc", "se", "ci_low", "ci_high", "recall_at_0.50"],
         [[kind, r["group"], r["n"], r["n_positive"], r["n_negative"],
           round(r["positive_rate"], 5), round(r["auc"], 5),
           round(r["se"], 5), round(r["ci_low"], 5), round(r["ci_high"], 5),
           round(r["recall"], 5)] for kind, r in entries])


def confusion_matrix(report):
    print("confusion_matrix")
    tn = sum(f.confusion[0] for f in report.folds)
    fp = sum(f.confusion[1] for f in report.folds)
    fn = sum(f.confusion[2] for f in report.folds)
    tp = sum(f.confusion[3] for f in report.folds)
    counts = np.array([[tn, fp], [fn, tp]], dtype=float)
    shares = counts / counts.sum(axis=1, keepdims=True)

    figure, axes = plt.subplots(figsize=(6.0, 4.6), dpi=DPI)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    # Sequential single hue: magnitude, not identity. Normalised by row so
    # the 86% majority class does not flatten the deficient row to one shade.
    axes.imshow(shares, cmap="Blues", vmin=0, vmax=1)

    for i in range(2):
        for j in range(2):
            axes.text(j, i, "{0:,.0f}\n{1:.1%} of row".format(
                counts[i, j], shares[i, j]),
                ha="center", va="center", fontsize=11,
                color=(SURFACE if shares[i, j] > 0.70 else INK))

    axes.set_xticks([0, 1])
    axes.set_xticklabels(["predicted replete", "predicted deficient"],
                         fontsize=9, color=INK)
    axes.set_yticks([0, 1])
    axes.set_yticklabels(["actual replete", "actual deficient"],
                         fontsize=9, color=INK)
    axes.tick_params(length=0)
    for side in axes.spines.values():
        side.set_visible(False)
    axes.set_title(
        "Confusion matrix at threshold 0.50, summed over 25 folds",
        color=INK, fontsize=12, fontweight="600", loc="left", pad=34)
    axes.text(0.0, 1.015,
              "Shaded by row share. 59% of deficient participants\n"
              "are missed at this threshold.",
              transform=axes.transAxes, color=INK_SOFT, fontsize=9,
              va="bottom", linespacing=1.45)
    _save(figure, "confusion_matrix")
    _csv("confusion_matrix",
         ["actual", "predicted", "count", "row_share"],
         [["replete", "replete", int(tn), round(shares[0, 0], 5)],
          ["replete", "deficient", int(fp), round(shares[0, 1], 5)],
          ["deficient", "replete", int(fn), round(shares[1, 0], 5)],
          ["deficient", "deficient", int(tp), round(shares[1, 1], 5)]])


def main(argv=None):
    from chemoguard.data.nhanes import DEFAULT_CACHE

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--only", choices=("coherence", "iron"), default=None)
    args = parser.parse_args(argv)

    print("Writing figures to {0}/ at {1} dpi".format(OUTPUT, DPI))
    print()

    if args.only != "iron":
        coherence_sweep(args.cache_dir)
        print()

    if args.only != "coherence":
        report = _iron_report(args.cache_dir)
        reliability_curve(report)
        precision_recall_curve(report)
        subgroup_auc(report)
        confusion_matrix(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
