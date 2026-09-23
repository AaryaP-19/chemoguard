"""Extraction benchmark: where regex-based OCR extraction fails on CBC tables.

This is a measurement artefact, not a utility. It characterises the failure
surface of the Tesseract-plus-regex extraction path in ``ocr_extraction.py``
along three axes:

* **unit notation** -- which scale the lab reported in, where the unit is
  printed, whether the decimal mark is a point or a comma, and whether a
  reference-range column sits beside the value;
* **image quality** -- JPEG compression, skew, blur, contrast loss, the
  shadow gradient a phone camera puts across a page, and partial crops;
* **alias coverage** -- whether a unit the backend read correctly survives
  ``units.parse_unit``.

The third axis is separated from the first two deliberately. A field can be
transcribed perfectly and still be lost, because the spelling printed on the
page is not in the alias table -- ``cells/mcL`` is not, and neither is a unit
cell carrying an annotation such as ``cells/uL (Low)``. Charging that to the
OCR backend would misattribute the loss and point any fix at the wrong module.
So every failure is attributed to a stage, and the stage boundaries are the
ones the production code already has:

    glyphs --(Tesseract)--> text --(regex)--> value+unit
           --(units.parse_unit)--> Unit --(cbc.quantity_for)--> Quantity

Scoring is per field, where a field is one ``(image, analyte)`` pair. Ground
truth records what the page *shows*, never what the current code can parse,
so the same truth scores a run before and after an alias-table change and the
difference between the two runs is itself a measurement.

The GPT-4o-mini vision arm is **deferred, not abandoned**: ``backends.py``
carries a complete adapter for it, hard-guarded against making a network call.
See ``docs/extraction-benchmark.md``.
"""

from chemoguard.extraction_bench.truth import (
    ABSENT,
    EXTRACTABLE,
    NO_UNIT,
    FieldTruth,
    ImageTruth,
    load_manifest,
    save_manifest,
)

__all__ = [
    "ABSENT",
    "EXTRACTABLE",
    "NO_UNIT",
    "FieldTruth",
    "ImageTruth",
    "load_manifest",
    "save_manifest",
]
