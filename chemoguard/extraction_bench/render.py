"""Draw a CBC report image, and record exactly what was drawn where.

One layout, held fixed. Table layout is a deferred axis (see
``docs/extraction-benchmark.md``), so varying it here would confound the two
axes this pass does measure. What varies is notation: which scale the values
are in, where the unit is printed, whether the decimal mark is a point or a
comma, and whether a reference-range column is present.

The geometry this module returns is load-bearing, not diagnostic. Crop
variants derive their ground truth from it: a crop that removes a unit column
while leaving the value column intact turns an ``extractable`` field into a
``no_unit`` field, and the correct behaviour flips from "emit a quantity" to
"refuse". That is a real thing a photographed report does, and getting its
truth right requires knowing where the cells were.

Unit strings are declared in ``panels.py`` in ASCII (``uL``) for the reason
``units.py`` gives -- they end up in console output -- but a page that printed
``uL`` would not be a realistic report. ``to_display`` substitutes the micro
sign for rendering, and ground truth records the glyphs actually drawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from chemoguard.extraction_bench.panels import (
    ANALYTE_ORDER,
    CANONICAL,
    LABELS,
    LABELS_FOR_EMBEDDED_UNIT,
    UNIT_ALREADY_IN_LABEL,
    Panel,
    SCALES,
    convert,
    row_for,
)

#: Reference ranges in canonical units. Conventional adult ranges; they are
#: printed as page furniture, never scored. What matters is only whether a
#: second column of numbers beside the value distracts the extractor.
REFERENCE_RANGES = {
    "hb": (Decimal("13.5"), Decimal("17.5")),
    "rbc": (Decimal("4.2"), Decimal("5.4")),
    "wbc": (Decimal("4300"), Decimal("10800")),
    "anc": (Decimal("1500"), Decimal("8000")),
    "neutrophils_percent": (Decimal("40"), Decimal("75")),
    "platelets": (Decimal("150000"), Decimal("450000")),
    "hematocrit": (Decimal("39"), Decimal("50")),
    "mcv": (Decimal("80"), Decimal("100")),
    "mch": (Decimal("27"), Decimal("33")),
    "mchc": (Decimal("32"), Decimal("36")),
}

#: Where the unit is printed.
COLUMN = "column"          # a dedicated units column
LABEL = "label"            # appended to the parameter name: "WBC (K/uL)"
VALUE_CELL = "value_cell"  # inside the value cell: "6.5 K/uL"
ABSENT_UNIT = "absent"     # nowhere on the page

#: ``value_cell`` is implemented and **not** part of the pass-1 cross. It is
#: the placement both ``test_3.jpg`` and the screenshot actually use, so it is
#: represented in stratum A, and adding it to the generated cross is a
#: one-element change to ``corpus.PLACEMENTS``. Left out to keep pass 1 to the
#: three levels specified.
PLACEMENTS_ALL = (COLUMN, LABEL, VALUE_CELL, ABSENT_UNIT)

POINT = "point"   # 4.80 / 250,000  -- decimal point, comma groups
COMMA = "comma"   # 4,80 / 250.000  -- decimal comma, point groups
DECIMAL_STYLES = (POINT, COMMA)

_MARKS = {POINT: (".", ","), COMMA: (",", ".")}

_FONT_DIR = "C:/Windows/Fonts"
_FONT_REGULAR = _FONT_DIR + "/arial.ttf"
_FONT_BOLD = _FONT_DIR + "/arialbd.ttf"

_PAD_X = 14
_PAD_Y = 11
_MARGIN = 34
_GRID = 1


def to_display(unit_text: str) -> str:
    """The glyphs a report would actually print for a unit string.

    ``uL`` becomes the micro sign form. Both checked-in report images print
    the micro sign, and whether Tesseract reads it is part of what is being
    measured -- it is a glyph OCR routinely returns as ``u``, ``y`` or ``p``.
    """
    return unit_text.replace("uL", "\u00b5L")


def style_number(plain: str, decimal_style: str) -> str:
    """Apply a notation to a plain ASCII number.

    ``plain`` uses ``.`` as the decimal mark and carries no thousands
    separator. Grouping starts at four integer digits, so ``900`` stays
    ``900`` and ``250000`` becomes ``250,000`` -- or ``250.000`` under the
    comma style, which is the European convention and the case worth having:
    ``ocr_extraction._NUMBER`` matches it, then ``float("250.000")`` is 250.0.
    """
    decimal_mark, group_sep = _MARKS[decimal_style]

    integer, _, fraction = plain.partition(".")
    if len(integer) > 3:
        head = len(integer) % 3 or 3
        chunks = [integer[:head]] + [
            integer[i : i + 3] for i in range(head, len(integer), 3)
        ]
        integer = group_sep.join(chunks)

    return integer + decimal_mark + fraction if fraction else integer


def _reference_range(analyte: str, unit: str, decimal_style: str) -> str:
    """A reference range, printed the way a report prints one.

    Ranges carry a range's precision, not the value's: a page prints
    "80 - 100" for MCV beside a value of "88.0", and "4.2 - 5.4" for RBC
    beside "4.80". Both endpoints share the precision of whichever needs
    more, so a range never reads "1.5 - 8".

    Never scored. It is page furniture, and the only question the axis asks
    is whether a second column of numbers beside the value distracts the
    extractor -- which for a regex scanning line by line is a real risk.
    """
    low, high = (
        convert(bound, CANONICAL[analyte], unit)
        for bound in REFERENCE_RANGES[analyte]
    )
    places = max(-low.normalize().as_tuple().exponent,
                 -high.normalize().as_tuple().exponent, 0)
    quantum = Decimal(1).scaleb(-places)

    return "{0} - {1}".format(
        style_number(str(low.quantize(quantum)), decimal_style),
        style_number(str(high.quantize(quantum)), decimal_style),
    )


@dataclass(frozen=True)
class Cell:
    """A drawn cell: its text and its pixel box, for crop truth."""

    text: str
    box: Tuple[int, int, int, int]   # left, top, right, bottom


@dataclass(frozen=True)
class RenderedRow:
    """One analyte's row as drawn."""

    analyte: str
    label: Cell
    value: Cell
    #: The cell the unit was drawn in. None when the unit is absent, or when
    #: it sits inside the label or the value cell -- in which case
    #: ``unit_shares`` names which.
    unit: Optional[Cell] = None
    unit_shares: Optional[str] = None
    ranges: Optional[Cell] = None
    #: The printed glyphs, for ground truth.
    printed_value: str = ""
    printed_unit: Optional[str] = None
    true_value: Optional[Decimal] = None
    canonical_unit: Optional[str] = None


@dataclass(frozen=True)
class Rendered:
    """A rendered page: the image, the rows as drawn, and the page text."""

    image: object                      # PIL.Image.Image
    rows: List[RenderedRow]
    text: str
    size: Tuple[int, int]
    header_bottom: int = 0
    extras: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PageSpec:
    """Everything that decides what a generated page looks like."""

    panel: Panel
    scale: str
    placement: str
    decimal_style: str
    reference_ranges: bool
    analytes: Tuple[str, ...] = ANALYTE_ORDER
    title: str = "REGIONAL PATHOLOGY LABORATORY"
    subtitle: str = "Complete Blood Count (CBC)"
    font_size: int = 21


def _fonts(size: int):
    from PIL import ImageFont

    return (
        ImageFont.truetype(_FONT_REGULAR, size),
        ImageFont.truetype(_FONT_BOLD, size),
        ImageFont.truetype(_FONT_BOLD, int(size * 1.35)),
    )


def _width(draw, text, font) -> int:
    if not text:
        return 0
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return right - left


def build_cells(spec: PageSpec):
    """The text of every cell, before any pixel is drawn.

    Split out from rendering so the notation logic can be tested without
    Pillow, and so the ground truth and the drawn page cannot disagree: both
    read this one result.
    """
    rows = []

    for analyte in spec.analytes:
        plain, printed_unit, canonical, true_value = row_for(
            spec.panel, analyte, spec.scale
        )
        value_text = style_number(plain, spec.decimal_style)
        unit_display = to_display(printed_unit)

        label = LABELS[analyte]
        unit_cell = None
        shares = None
        recorded_unit = unit_display

        if spec.placement == COLUMN:
            unit_cell = unit_display
        elif spec.placement == LABEL:
            shares = LABEL
            if analyte not in UNIT_ALREADY_IN_LABEL:
                label = "{0} ({1})".format(
                    LABELS_FOR_EMBEDDED_UNIT[analyte], unit_display
                )
        elif spec.placement == VALUE_CELL:
            value_text = "{0} {1}".format(value_text, unit_display)
            shares = VALUE_CELL
        elif spec.placement == ABSENT_UNIT:
            recorded_unit = None
            # "Neutrophils %" states its own unit, so the label has to change
            # too or this placement is not "absent" for that row. Dropping to
            # "Neutrophils" is what makes the axis honest.
            if analyte in UNIT_ALREADY_IN_LABEL:
                label = LABELS_FOR_EMBEDDED_UNIT[analyte].replace(" %", "")
        else:
            raise ValueError("unknown placement {0!r}".format(spec.placement))

        # Ranges never carry a unit. When the placement is "absent" the unit
        # has to be absent from every column, or it is not absent; keeping
        # ranges bare in all placements holds that axis clean instead of
        # letting the unit leak back in through this column.
        range_text = (
            _reference_range(analyte, canonical, spec.decimal_style)
            if spec.reference_ranges
            else None
        )

        rows.append(
            {
                "analyte": analyte,
                "label": label,
                "value": value_text,
                "unit": unit_cell,
                "unit_shares": shares,
                "ranges": range_text,
                "printed_value": style_number(plain, spec.decimal_style),
                "printed_unit": recorded_unit,
                "true_value": true_value,
                "canonical_unit": canonical if recorded_unit else None,
            }
        )

    return rows


def render(spec: PageSpec) -> Rendered:
    """Draw the page and return it with the geometry of every cell."""
    from PIL import Image, ImageDraw

    body, bold, title_font = _fonts(spec.font_size)
    cells = build_cells(spec)

    probe = ImageDraw.Draw(Image.new("RGB", (8, 8), "white"))

    headers = ["TEST", "RESULT"]
    if spec.placement == COLUMN:
        headers.append("UNITS")
    if spec.reference_ranges:
        headers.append("REFERENCE RANGE")

    keys = ["label", "value"] + (
        ["unit"] if spec.placement == COLUMN else []
    ) + (["ranges"] if spec.reference_ranges else [])

    widths = []
    for index, key in enumerate(keys):
        widest = _width(probe, headers[index], bold)
        for row in cells:
            widest = max(widest, _width(probe, row[key] or "", body))
        widths.append(widest + 2 * _PAD_X)

    line_height = spec.font_size + 2 * _PAD_Y
    table_width = sum(widths) + _GRID * (len(widths) + 1)
    title_h = int(spec.font_size * 1.35)
    top = _MARGIN + title_h + 12 + spec.font_size + 22

    width = table_width + 2 * _MARGIN
    height = top + line_height * (len(cells) + 1) + _GRID * (len(cells) + 2) + _MARGIN

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text(
        (width // 2, _MARGIN), spec.title, fill="black", font=title_font, anchor="ma"
    )
    draw.text(
        (width // 2, _MARGIN + title_h + 12),
        spec.subtitle,
        fill="black",
        font=body,
        anchor="ma",
    )

    # Column left edges, once, so header and body cannot drift apart.
    edges = [_MARGIN + _GRID]
    for w in widths:
        edges.append(edges[-1] + w + _GRID)

    def draw_row(y, values, font):
        for index, text in enumerate(values):
            if not text:
                continue
            # Numbers right-align, labels left-align: the alignment a report
            # uses, and it changes where a crop bites.
            if index == 0:
                draw.text((edges[index] + _PAD_X, y + _PAD_Y), text, fill="black", font=font)
            else:
                draw.text(
                    (edges[index + 1] - _GRID - _PAD_X, y + _PAD_Y),
                    text,
                    fill="black",
                    font=font,
                    anchor="ra",
                )

    def box(index, y):
        return (edges[index], y, edges[index + 1] - _GRID, y + line_height)

    y = top
    draw_row(y, headers, bold)
    header_bottom = y + line_height
    y = header_bottom + _GRID

    rows: List[RenderedRow] = []
    for row in cells:
        draw_row(y, [row[key] for key in keys], body)
        positions = {key: index for index, key in enumerate(keys)}

        rows.append(
            RenderedRow(
                analyte=row["analyte"],
                label=Cell(row["label"], box(positions["label"], y)),
                value=Cell(row["value"], box(positions["value"], y)),
                unit=(
                    Cell(row["unit"], box(positions["unit"], y))
                    if "unit" in positions
                    else None
                ),
                unit_shares=row["unit_shares"],
                ranges=(
                    Cell(row["ranges"], box(positions["ranges"], y))
                    if "ranges" in positions
                    else None
                ),
                printed_value=row["printed_value"],
                printed_unit=row["printed_unit"],
                true_value=row["true_value"],
                canonical_unit=row["canonical_unit"],
            )
        )
        y += line_height + _GRID

    # Grid: horizontal rules under the header and each row, verticals full
    # height. Drawn last so no glyph sits on top of a line.
    table_top, table_bottom = top, y - _GRID
    for edge in edges:
        draw.line([(edge - _GRID, table_top), (edge - _GRID, table_bottom)], fill="black", width=_GRID)
    draw.line([(edges[0] - _GRID, table_bottom), (edges[-1] - _GRID, table_bottom)], fill="black", width=_GRID)
    rule = table_top
    for _ in range(len(cells) + 1):
        draw.line([(edges[0] - _GRID, rule), (edges[-1] - _GRID, rule)], fill="black", width=_GRID)
        rule += line_height + _GRID

    text_lines = [spec.title, spec.subtitle, "   ".join(headers)]
    text_lines += ["   ".join(row[key] or "" for key in keys) for row in cells]

    return Rendered(
        image=image,
        rows=rows,
        text="\n".join(text_lines) + "\n",
        size=(width, height),
        header_bottom=header_bottom,
        extras={"columns": list(keys), "edges": edges},
    )
