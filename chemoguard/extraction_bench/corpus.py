"""Build the corpus: three strata, and ground truth derived from what was drawn.

**Stratum A** -- the four real images, hand-labelled in ``stratum_a.py``.

**Stratum B** -- the notation cross at pristine render quality. Every
combination of unit scale, unit placement, decimal mark and reference-range
presence, times every panel: 3 x 3 x 2 x 2 x 3 = 108 images. Fully crossed, so
each level of each axis carries the same number of images and a per-level
comparison is not confounded by another axis. ``composition()`` prints the
contingency counts rather than asserting the balance, because a claim of
balance that nobody checked is how a confound survives to publication.

**Stratum C** -- degradation, on a notation subset held fixed. The decimal
mark is pinned to a point and reference ranges to present, so quality is not
confounded with notation; scale and placement still vary across all nine
combinations so that a degradation result is not specific to one notation.
9 cells x 3 panels = 27 base renders, each rendered in 26 quality variants =
702 images.

Keeping ``placement=absent`` in stratum C's base is deliberate. Those fields
have no unit to read, so the only thing degradation can do to them is provoke
a backend into supplying one -- which measures hallucination under duress, the
failure mode with the worst consequence and the one a clean-image benchmark
cannot see.

Crop truth is *derived*, never assumed. ``render`` returns the pixel box of
every cell it drew; a crop is intersected with those boxes, and a field whose
value cell is clipped becomes ``absent`` while a field whose value survives
but whose unit cell is cut becomes ``no_unit``. That second case is the
interesting one: the correct behaviour flips from emitting a quantity to
refusing one, purely because of where the photograph ended.
"""

from __future__ import annotations

import itertools
import shutil
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from chemoguard.extraction_bench import degrade
from chemoguard.extraction_bench.panels import CANONICAL, PANELS, convert
from chemoguard.extraction_bench.render import (
    ABSENT_UNIT,
    COLUMN,
    COMMA,
    LABEL,
    POINT,
    PageSpec,
    Rendered,
    render,
)
from chemoguard.extraction_bench.stratum_a import stratum_a
from chemoguard.extraction_bench.truth import (
    ABSENT,
    EXTRACTABLE,
    NO_UNIT,
    FieldTruth,
    ImageTruth,
    save_manifest,
)

SCALE_LEVELS = ("conventional_cells", "conventional_k", "si")

#: Three levels, as scoped. ``render.VALUE_CELL`` is implemented and excluded;
#: it is the placement stratum A actually uses, and adding it here is a
#: one-element change. See ``render.PLACEMENTS_ALL``.
PLACEMENTS = (COLUMN, LABEL, ABSENT_UNIT)

DECIMAL_STYLES = (POINT, COMMA)
RANGE_LEVELS = (True, False)
PANEL_NAMES = ("normal", "cytopenic", "severe")

#: Notation levels pinned in stratum C, so the quality axis is not confounded.
C_DECIMAL_STYLE = POINT
C_REFERENCE_RANGES = True


def _cell_id(scale, placement, decimal_style, ranges) -> str:
    return "{0}__{1}__{2}__{3}".format(
        scale,
        placement,
        decimal_style,
        "ranges" if ranges else "noranges",
    )


def notation_cells():
    """The 36 cells of the stratum-B cross, in a stable order."""
    return [
        (scale, placement, decimal_style, ranges)
        for scale, placement, decimal_style, ranges in itertools.product(
            SCALE_LEVELS, PLACEMENTS, DECIMAL_STYLES, RANGE_LEVELS
        )
    ]


def degradation_cells():
    """The 9 notation cells stratum C degrades, in a stable order."""
    return [
        (scale, placement, C_DECIMAL_STYLE, C_REFERENCE_RANGES)
        for scale, placement in itertools.product(SCALE_LEVELS, PLACEMENTS)
    ]


def _clipped(box, keep: Tuple[int, int]) -> bool:
    """True when any part of ``box`` falls outside the retained region."""
    _left, _top, right, bottom = box
    return right > keep[0] or bottom > keep[1]


def truth_from_render(
    rendered: Rendered,
    keep: Optional[Tuple[int, int]] = None,
) -> Dict[str, FieldTruth]:
    """Ground truth for a rendered page, after an optional crop.

    ``keep`` is the retained ``(width, height)`` in the rendered page's own
    pixels. Every cell box is tested against it:

    * value cell clipped  -> ``absent``. The number cannot be fully read, so
      there is no correct value and silence is the right answer.
    * value intact, unit cell clipped -> ``no_unit``. The number is readable
      and its scale is not, which is precisely the state ``units.py`` exists
      to refuse.
    * both intact -> unchanged.

    A unit printed inside the label or the value cell travels with that cell,
    so it cannot be clipped independently of the number it belongs to.
    """
    fields = {}

    for row in rendered.rows:
        analyte = row.analyte
        value_gone = keep is not None and _clipped(row.value.box, keep)
        label_gone = keep is not None and _clipped(row.label.box, keep)

        if value_gone or label_gone:
            fields[analyte] = FieldTruth(analyte=analyte, status=ABSENT)
            continue

        # A unit printed in a column can also be printed in the row label, and
        # then a crop that takes the column does not take the unit. "Neutrophils
        # %" beside a units column reading "%" is not a contrivance -- test1.jpg
        # does exactly that -- so the test has to be whether the unit survives
        # *anywhere*, not whether its column did.
        #
        # Getting this wrong cost real credibility once: the first OCR run
        # reported nine fields where the backend had supposedly invented a unit
        # under crop, which would have read as a dangerous hallucination. The
        # backend had read "%" out of the label, correctly, and the ground truth
        # was what was wrong.
        unit_in_label = bool(row.printed_unit) and row.printed_unit in row.label.text

        unit_gone = (
            keep is not None
            and row.unit is not None
            and _clipped(row.unit.box, keep)
            and not unit_in_label
        )

        if row.printed_unit is None or unit_gone:
            fields[analyte] = FieldTruth(
                analyte=analyte,
                status=NO_UNIT,
                printed_value=row.printed_value,
                value=float(row.true_value),
            )
            continue

        fields[analyte] = FieldTruth(
            analyte=analyte,
            status=EXTRACTABLE,
            printed_value=row.printed_value,
            printed_unit=row.printed_unit,
            unit_placement=row.unit_shares or COLUMN,
            value=float(row.true_value),
            unit=row.canonical_unit,
            canonical_value=float(
                convert(
                    Decimal(str(row.true_value)),
                    row.canonical_unit,
                    CANONICAL[analyte],
                )
            ),
        )

    return fields


def _crop_consequence(before: Dict[str, FieldTruth], after: Dict[str, FieldTruth]) -> str:
    """What a crop actually did to the ground truth.

    ``none`` -- the cut removed only page furniture and every field's correct
    outcome is unchanged. ``units_lost`` -- at least one field that was
    extractable now has a readable number and no readable unit, so its
    correct outcome flipped from a quantity to a refusal. ``rows_lost`` --
    rows left the page. Both when a cut did both.
    """
    units_lost = any(
        before[key].status == EXTRACTABLE and after[key].status == NO_UNIT
        for key in before
    )
    rows_lost = any(
        before[key].status != ABSENT and after[key].status == ABSENT
        for key in before
    )

    if units_lost and rows_lost:
        return "units_and_rows_lost"
    if units_lost:
        return "units_lost"
    if rows_lost:
        return "rows_lost"
    return "none"


def _spec(scale, placement, decimal_style, ranges, panel_name) -> PageSpec:
    return PageSpec(
        panel=PANELS[panel_name],
        scale=scale,
        placement=placement,
        decimal_style=decimal_style,
        reference_ranges=ranges,
    )


def build(root, include_degradation: bool = True, progress=None) -> List[ImageTruth]:
    """Write every image and return the manifest.

    Stratum A images are copied into the corpus so the manifest is
    self-contained: the benchmark should be runnable from the corpus directory
    alone, without depending on paths elsewhere in the tree.
    """
    root = Path(root)
    for stratum in ("A", "B", "C"):
        (root / stratum).mkdir(parents=True, exist_ok=True)
    (root / "text").mkdir(parents=True, exist_ok=True)

    manifest: List[ImageTruth] = []
    repo = Path(__file__).resolve().parents[2]

    # --- stratum A: copy the real images alongside their hand labels -------
    #
    # The transcriptions in tests/fixtures/ are carried in as page text, so
    # the pattern layer can be scored on the real images too. That gives
    # stratum A the same ceiling the generated strata have: a field the
    # patterns cannot read from a perfect transcription was never going to
    # survive OCR either, and the loss belongs to the patterns or the alias
    # table rather than to Tesseract.
    for image in stratum_a():
        source = repo / image.path
        if not source.exists():
            raise FileNotFoundError(
                "stratum A image {0} is missing. Expected it at {1}.".format(
                    image.path, source
                )
            )
        target = root / "A" / Path(image.path).name
        shutil.copyfile(source, target)

        text_rel = None
        transcription = repo / "tests" / "fixtures" / (Path(image.path).stem + ".txt")
        if transcription.exists():
            text_rel = "text/A__{0}.txt".format(image.image_id)
            shutil.copyfile(transcription, root / text_rel)

        manifest.append(
            ImageTruth(
                image_id=image.image_id,
                path="A/" + Path(image.path).name,
                stratum="A",
                sub=image.sub,
                factors=image.factors,
                fields=image.fields,
                text_path=text_rel,
            )
        )

    # --- stratum B: the notation cross, pristine ---------------------------
    for scale, placement, decimal_style, ranges in notation_cells():
        cell = _cell_id(scale, placement, decimal_style, ranges)
        for panel_name in PANEL_NAMES:
            spec = _spec(scale, placement, decimal_style, ranges, panel_name)
            rendered = render(spec)
            image_id = "B__{0}__{1}".format(cell, panel_name)
            relative = "B/{0}.png".format(image_id)
            rendered.image.save(root / relative, "PNG")

            text_rel = "text/{0}.txt".format(image_id)
            (root / text_rel).write_text(rendered.text, encoding="utf-8")

            manifest.append(
                ImageTruth(
                    image_id=image_id,
                    path=relative,
                    stratum="B",
                    sub=cell,
                    factors={
                        "source": "generated",
                        "scale": scale,
                        "unit_placement": placement,
                        "decimal_style": decimal_style,
                        "reference_ranges": ranges,
                        "panel": panel_name,
                        "degradation_family": degrade.PRISTINE,
                        "degradation": degrade.PRISTINE,
                        "severity": 0,
                    },
                    fields=truth_from_render(rendered),
                    text_path=text_rel,
                )
            )
            if progress:
                progress("B", image_id)

    if not include_degradation:
        save_manifest(manifest, root / "manifest.json")
        return manifest

    # --- stratum C: degradation over a fixed notation subset ---------------
    all_variants = degrade.variants()

    for scale, placement, decimal_style, ranges in degradation_cells():
        cell = _cell_id(scale, placement, decimal_style, ranges)
        for panel_name in PANEL_NAMES:
            spec = _spec(scale, placement, decimal_style, ranges, panel_name)
            rendered = render(spec)
            base_id = "{0}__{1}".format(cell, panel_name)

            # The page text is a property of the render, not of the
            # degradation, so it is written once per base and shared by all
            # 26 variants. The pattern layer reads it; the OCR path does not.
            text_rel = "text/C__{0}.txt".format(base_id)
            (root / text_rel).write_text(rendered.text, encoding="utf-8")

            for variant in all_variants:
                image, keep = degrade.apply_to(rendered, variant)
                extension = "jpg" if variant.save_format == "JPEG" else "png"
                image_id = "C__{0}__{1}".format(base_id, variant.name)
                relative = "C/{0}.{1}".format(image_id, extension)
                degrade.save(image, root / relative, variant)

                fields = truth_from_render(rendered, keep=keep)
                factors = {
                    "source": "generated",
                    "scale": scale,
                    "unit_placement": placement,
                    "decimal_style": decimal_style,
                    "reference_ranges": ranges,
                    "panel": panel_name,
                    "degradation_family": variant.family,
                    "degradation": variant.name,
                    "severity": variant.severity,
                    "severity_label": variant.level,
                }
                if keep is not None:
                    # A crop is reported by what it did to the truth, not by
                    # its nominal name. The same cut takes the unit column on
                    # one placement and nothing that matters on another, so
                    # grouping crop results by name would average two
                    # different experiments together.
                    factors.update(
                        crop_removes=variant.removes,
                        crop_keep_fraction=[
                            round(keep[0] / rendered.size[0], 4),
                            round(keep[1] / rendered.size[1], 4),
                        ],
                        crop_consequence=_crop_consequence(
                            truth_from_render(rendered), fields
                        ),
                    )

                manifest.append(
                    ImageTruth(
                        image_id=image_id,
                        path=relative,
                        stratum="C",
                        sub=cell,
                        factors=factors,
                        fields=fields,
                        # Only the pristine variant's pixels match the text;
                        # a crop removes rows the text still lists, so the
                        # pattern layer is scored on stratum C's control only.
                        text_path=text_rel if variant.family == degrade.PRISTINE else None,
                    )
                )
                if progress:
                    progress("C", image_id)

    save_manifest(manifest, root / "manifest.json")
    return manifest


def composition(manifest) -> Dict[str, object]:
    """Strata sizes and the contingency counts behind the balance claim.

    Returns counts rather than a verdict. The point is that a reader can see
    each axis level carries the same number of fields, instead of taking the
    word of whoever wrote the generator.
    """
    by_stratum = Counter()
    fields_by_stratum = Counter()
    status_by_stratum: Dict[str, Counter] = {}
    axes: Dict[str, Counter] = {}
    families = Counter()

    for image in manifest:
        by_stratum[image.stratum] += 1
        counts = image.counts()
        fields_by_stratum[image.stratum] += sum(counts.values())
        status_by_stratum.setdefault(image.stratum, Counter()).update(counts)

        if image.stratum == "B":
            for axis in ("scale", "unit_placement", "decimal_style",
                         "reference_ranges", "panel"):
                axes.setdefault(axis, Counter())[
                    str(image.factors.get(axis))
                ] += 1
        if image.stratum == "C":
            families[str(image.factors.get("degradation_family"))] += 1

    return {
        "images_by_stratum": dict(by_stratum),
        "fields_by_stratum": dict(fields_by_stratum),
        "status_by_stratum": {k: dict(v) for k, v in status_by_stratum.items()},
        "stratum_b_axis_balance": {k: dict(v) for k, v in axes.items()},
        "stratum_c_family_counts": dict(families),
        "total_images": sum(by_stratum.values()),
        "total_fields": sum(fields_by_stratum.values()),
    }
