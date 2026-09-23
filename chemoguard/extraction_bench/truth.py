"""Ground truth for one field of one report image.

A *field* is one ``(image, analyte)`` pair, and it is the unit of analysis for
the whole benchmark. Every field carries three things a scorer needs and that
are easy to conflate:

``printed_value`` / ``printed_unit``
    The literal glyphs on the page. ``"4,80"`` is not ``"4.80"`` and
    ``"cells/mcL"`` is not ``"cells/uL"``; keeping the glyphs lets a failure be
    attributed to reading rather than to interpretation.

``value`` / ``unit``
    What those glyphs *mean*: the number in the unit that was printed, and the
    canonical symbol that unit denotes. This is a labelling judgement, not a
    call into ``units.py``. ``cells/mcL`` denotes ``cells/uL`` because mcL is
    microlitre, whether or not the alias table happens to know that today.

``status``
    Whether the page makes the field extractable at all.

That last distinction is what stops the benchmark from punishing correct
behaviour. When no unit is printed, refusing is the right answer and any
number emitted is a guess -- so those fields can only score TN or FP, never
TP. ``cbc.panel_from_extraction`` already implements the refusal; this records
when it is the outcome we want.

Ground truth never depends on what the code under test can parse. That is the
property that makes the before-and-after comparison across an alias-table
change meaningful: the same manifest scores both runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional

#: The page prints both a value and a unit for this analyte. A correct
#: ``Quantity`` is obtainable, and emitting it is the only right answer.
EXTRACTABLE = "extractable"

#: The page prints a value but no unit anywhere on the row -- not in a units
#: column, not in the label, not beside the number. Refusing is correct;
#: emitting a number means a unit was inferred from magnitude, which is the
#: exact behaviour CLAUDE.md Finding 4 is about.
NO_UNIT = "no_unit"

#: The analyte is not on the page: the panel does not carry it, or a crop
#: removed its row. Silence is correct.
ABSENT = "absent"

STATUSES = (EXTRACTABLE, NO_UNIT, ABSENT)


@dataclass(frozen=True)
class FieldTruth:
    """What one analyte's row on one image actually says."""

    analyte: str
    status: str
    #: Literal glyphs in the value cell, e.g. ``"4,80"`` or ``"250.000"``.
    printed_value: Optional[str] = None
    #: Literal glyphs of the unit as printed, wherever it was printed, e.g.
    #: ``"millions/uL"``. None when no unit appears on the row.
    printed_unit: Optional[str] = None
    #: Where the unit was printed: "column", "label", "value_cell" or None.
    unit_placement: Optional[str] = None
    #: The number, in the unit that was printed.
    value: Optional[float] = None
    #: Canonical symbol the printed unit denotes, per ``cbc.ANALYTES``.
    unit: Optional[str] = None
    #: ``value`` expressed in the analyte's canonical unit. The quantity a
    #: correct extraction must arrive at.
    canonical_value: Optional[float] = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(
                "status must be one of {0}, got {1!r}".format(STATUSES, self.status)
            )
        if self.status == EXTRACTABLE:
            missing = [
                name
                for name, val in (
                    ("printed_value", self.printed_value),
                    ("printed_unit", self.printed_unit),
                    ("value", self.value),
                    ("unit", self.unit),
                    ("canonical_value", self.canonical_value),
                )
                if val is None
            ]
            if missing:
                raise ValueError(
                    "{0} is {1} but {2} missing. An extractable field must "
                    "state the quantity a correct reading arrives at.".format(
                        self.analyte, EXTRACTABLE, ", ".join(missing)
                    )
                )
        if self.status == NO_UNIT:
            if self.printed_value is None or self.value is None:
                raise ValueError(
                    "{0} is {1}: the value is printed, only the unit is "
                    "absent.".format(self.analyte, NO_UNIT)
                )
            if self.printed_unit is not None:
                raise ValueError(
                    "{0} is {1} but printed_unit is {2!r}".format(
                        self.analyte, NO_UNIT, self.printed_unit
                    )
                )
        if self.status == ABSENT and self.printed_value is not None:
            raise ValueError(
                "{0} is {1} but a printed value was recorded".format(
                    self.analyte, ABSENT
                )
            )

    @property
    def is_extractable(self) -> bool:
        return self.status == EXTRACTABLE


@dataclass(frozen=True)
class ImageTruth:
    """One image, its provenance, and per-analyte truth."""

    image_id: str
    #: Path relative to the corpus root, so a manifest survives being moved.
    path: str
    #: "A" (real images), "B" (notation cross), "C" (degradation).
    stratum: str
    #: Sub-label within the stratum: "A1".."A4", or a notation cell id.
    sub: str
    #: The factor levels this image realises -- scale, placement, decimal
    #: mark, reference ranges, panel, degradation family and severity.
    factors: Dict[str, object] = field(default_factory=dict)
    fields: Dict[str, FieldTruth] = field(default_factory=dict)
    #: Path to the exact text drawn on the page, relative to the corpus root.
    #: Present for generated images only; it is what lets the pattern layer be
    #: scored with no OCR in the loop.
    text_path: Optional[str] = None

    def counts(self) -> Dict[str, int]:
        out = {status: 0 for status in STATUSES}
        for truth in self.fields.values():
            out[truth.status] += 1
        return out


def save_manifest(manifest, path) -> None:
    """Write a list of ImageTruth as indented JSON.

    Indented on purpose: the manifest is the artefact a reviewer reads to
    check that the ground truth says what the images show, and it diffs
    row-by-row when the generator changes.
    """
    payload = [
        dict(
            asdict(image),
            fields={k: asdict(v) for k, v in image.fields.items()},
        )
        for image in manifest
    ]
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_manifest(path):
    """Read a manifest back into ImageTruth objects.

    Re-running ``FieldTruth.__post_init__`` on load is intentional: a manifest
    hand-edited into an inconsistent state fails here rather than producing
    quietly wrong scores.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        ImageTruth(
            image_id=entry["image_id"],
            path=entry["path"],
            stratum=entry["stratum"],
            sub=entry["sub"],
            factors=entry.get("factors", {}),
            text_path=entry.get("text_path"),
            fields={
                key: FieldTruth(**value) for key, value in entry["fields"].items()
            },
        )
        for entry in payload
    ]
