"""Degradation curves and the per-parameter heatmap.

One figure per question, and each one plots the counts it was computed from
where they fit. A recall curve whose points rest on four fields each looks
identical to one resting on four hundred, and the reader cannot tell which
unless the figure says.

Matplotlib is pinned at 3.8.4 for the reason ``requirements.txt`` records:
3.9+ resolves numpy to 2.x, which breaks the scikit-learn 1.3.2 ABI. Nothing
here needs a newer matplotlib.
"""

from __future__ import annotations

import os
from typing import List

from chemoguard.extraction_bench import degrade, score as S
from chemoguard.extraction_bench.panels import ANALYTE_ORDER

DEFAULT_DIR = os.path.join("reports", "figures")

#: Families with an ordered severity ladder, so a curve means something.
#: Shadow and crop are included even though their levels are nominal rather
#: than metric -- the three shadow geometries and the three crop intents are
#: ordered by how much they remove, and a reader comparing bars needs them on
#: the same page as the metric families. ``pristine`` and ``combined`` are
#: single points and appear in the table instead.
_LADDERS = (
    degrade.JPEG,
    degrade.SKEW,
    degrade.BLUR,
    degrade.CONTRAST,
    degrade.SHADOW,
    degrade.CROP,
)


def _setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _arms_with_ladder_data(stats, arms):
    """Arms that actually have degraded-image results.

    The pattern-layer arm reads the page text rather than the pixels, so it
    only has a result for the pristine control -- degrading an image does not
    change the text the generator drew. Plotting an empty curve for it would
    put a blank panel in the figure and imply the measurement was made and
    came back at zero.
    """
    return [
        arm
        for arm in arms
        if any(
            key[0] == arm and key[1] in _LADDERS and value["recall"] is not None
            for key, value in stats.items()
        )
    ]


def degradation_curves(scores, manifest, arms, out_dir=DEFAULT_DIR) -> List[str]:
    """Recall against severity, one panel per degradation family."""
    plt = _setup()
    stats = S.by_degradation(scores, manifest)

    arms = _arms_with_ladder_data(stats, arms)
    if not arms:
        return []

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    written = []

    for axis, family in zip(axes.ravel(), _LADDERS):
        labels = {}
        for arm in arms:
            points = sorted(
                (
                    (severity, value["recall"], value["tp"] + value["fn"], label)
                    for (a, fam, severity, label), value in stats.items()
                    if a == arm and fam == family and value["recall"] is not None
                ),
                key=lambda row: row[0],
            )
            if not points:
                continue
            axis.plot(
                [p[0] for p in points],
                [p[1] for p in points],
                marker="o",
                label=arm,
            )
            labels.update({p[0]: p[3] for p in points})

        # Tick labels come from the severity ladder's own names, so a reader
        # sees "q30" and "sigma 1.5" rather than an ordinal index they would
        # have to look up.
        if labels:
            ticks = sorted(labels)
            axis.set_xticks(ticks)
            axis.set_xticklabels(
                [str(labels[t]) for t in ticks], rotation=30, ha="right", fontsize=7
            )

        # The pristine control, as a reference line rather than a point on
        # every curve: it is the same value for all families and drawing it
        # four times as data would imply four measurements.
        for arm in arms:
            control = [
                value["recall"]
                for (a, fam, _s, _l), value in stats.items()
                if a == arm and fam == degrade.PRISTINE and value["recall"] is not None
            ]
            if control:
                axis.axhline(
                    control[0],
                    linestyle=":",
                    linewidth=1,
                    color="grey",
                )

        axis.set_title(family)
        axis.set_ylabel("recall")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    fig.suptitle(
        "Extraction recall against image degradation (stratum C, 27 notation "
        "bases per level)\n"
        "grey dotted: that arm's pristine control at the same notation cells",
        fontsize=11,
    )
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "extraction_degradation_curves.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)
    return written


def per_parameter_heatmap(scores, arms, out_dir=DEFAULT_DIR) -> List[str]:
    """Recall per parameter per stratum, one column block per arm."""
    plt = _setup()
    strata = ("A", "B", "C")

    fig, axes = plt.subplots(
        1, len(arms), figsize=(4.4 * len(arms), 5.2), squeeze=False
    )
    written = []

    for axis, arm in zip(axes[0], arms):
        subset = [s for s in scores if s.backend == arm]
        grid, labels = [], []
        for analyte in ANALYTE_ORDER:
            row = []
            for stratum in strata:
                block = [
                    s for s in subset
                    if s.analyte == analyte and s.stratum == stratum
                ]
                stats = S.aggregate(block, "analyte").get((analyte,))
                row.append(
                    stats["recall"]
                    if stats and stats["recall"] is not None
                    else float("nan")
                )
            grid.append(row)
            labels.append(analyte)

        image = axis.imshow(grid, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        axis.set_xticks(range(len(strata)))
        axis.set_xticklabels(strata)
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels, fontsize=8)
        axis.set_title(arm, fontsize=10)

        for i, row in enumerate(grid):
            for j, value in enumerate(row):
                axis.text(
                    j,
                    i,
                    "n/a" if value != value else "{0:.2f}".format(value),
                    ha="center",
                    va="center",
                    fontsize=7,
                )

        fig.colorbar(image, ax=axis, fraction=0.046, label="recall")

    fig.suptitle("Recall per parameter per stratum", fontsize=11)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "extraction_per_parameter.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)
    return written


def notation_bars(scores, manifest, arms, out_dir=DEFAULT_DIR) -> List[str]:
    """Recall per level of each notation axis, stratum B."""
    plt = _setup()
    axes_to_plot = ("scale", "unit_placement", "decimal_style", "reference_ranges")

    fig, axes = plt.subplots(1, len(axes_to_plot), figsize=(15, 4))
    written = []

    for axis, factor in zip(axes, axes_to_plot):
        stats = S.by_factor(scores, manifest, factor, stratum="B")
        levels = sorted({key[1] for key in stats})
        width = 0.8 / max(len(arms), 1)

        for index, arm in enumerate(arms):
            values = [
                (stats.get((arm, level)) or {}).get("recall") for level in levels
            ]
            positions = [i + index * width for i in range(len(levels))]

            # Undefined recall is not zero recall. The ``absent`` placement
            # level has no extractable fields at all -- every field there is
            # a correct refusal -- so there is nothing to recall and a zero
            # bar would read as total failure at exactly the level where the
            # backend did the right thing every time.
            axis.bar(
                [p for p, v in zip(positions, values) if v is not None],
                [v for v in values if v is not None],
                width=width,
                label=arm,
            )
            for position, value in zip(positions, values):
                if value is None:
                    axis.text(
                        position,
                        0.04,
                        "n/a",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="grey",
                        rotation=90,
                    )

        axis.set_xticks([i + 0.4 - width / 2 for i in range(len(levels))])
        axis.set_xticklabels(levels, rotation=20, ha="right", fontsize=8)
        axis.set_title(factor, fontsize=10)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("recall")
        axis.grid(axis="y", alpha=0.3)
        axis.legend(fontsize=8)

    fig.suptitle("Recall per notation axis level (stratum B, pristine renders)", fontsize=11)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "extraction_notation_axes.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)
    return written


def write_figures(scores, manifest, arms, out_dir=DEFAULT_DIR) -> List[str]:
    """Every figure for which there is data. Others are skipped, not blanked."""
    written = []
    written += per_parameter_heatmap(scores, arms, out_dir)
    written += notation_bars(scores, manifest, arms, out_dir)
    written += degradation_curves(scores, manifest, arms, out_dir)
    return written
