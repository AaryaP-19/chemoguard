"""Image-quality degradations, one factor at a time.

Twenty-six variants per base render: a pristine control, then five families
varied along a severity ladder, and one combined condition standing in for a
photograph taken of a printed page.

One-factor-at-a-time on purpose. The question is *where* extraction degrades,
which needs a curve per family against a common control, not an average over
a soup of simultaneous distortions. The combined ``phone_photo`` condition is
the single exception, included because the interaction is the realistic case
and because a result that only holds for isolated factors is worth knowing
about.

Only the crop family changes ground truth, and it changes it in a way worth
stating: a crop that takes the units column while leaving the value column
turns an ``extractable`` field into a ``no_unit`` field, so the correct
behaviour flips from emitting a quantity to refusing one. Each crop therefore
reports the region of the original page it kept, and ``corpus.py`` derives the
new truth from the recorded cell geometry rather than assuming rows survived.

Everything except the ``jpeg`` family is written as PNG. Saving a skewed page
as JPEG would put compression inside the skew measurement, and then neither
number would mean anything on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


@dataclass(frozen=True)
class Variant:
    """One degradation: how to apply it and how to record it."""

    name: str
    #: Degradation family, for grouping into a curve.
    family: str
    #: Position on that family's severity ladder. 0 is the control. Ordinal
    #: within a family; not comparable across families.
    severity: float
    #: Human-readable severity, for tables.
    level: str
    apply: Callable
    save_format: str = "PNG"
    jpeg_quality: Optional[int] = None
    #: For the crop family: given the ``Rendered`` page, the ``(width,
    #: height)`` in page pixels that survives. Geometry-aware rather than a
    #: fixed fraction, because a fixed fraction measures something different
    #: on every layout -- see ``_crop_plans``.
    plan_crop: Optional[Callable] = None
    #: What the crop is intended to remove, for grouping results.
    removes: Optional[str] = None


PRISTINE = "pristine"
JPEG = "jpeg"
SKEW = "skew"
BLUR = "blur"
CONTRAST = "contrast"
SHADOW = "shadow"
CROP = "crop"
COMBINED = "combined"


def _identity(image):
    return image


def _blur(radius):
    def apply(image):
        from PIL import ImageFilter

        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    return apply


def _skew(degrees):
    def apply(image):
        from PIL import Image

        return image.rotate(
            degrees, resample=Image.BICUBIC, expand=True, fillcolor="white"
        )

    return apply


def _contrast(black, white):
    """Compress the tonal range into ``[black, white]``.

    Not ``ImageEnhance.Contrast``, which pivots around mid-grey and darkens
    the paper as much as it lightens the ink. A faded photocopy or an
    underexposed scan lifts the black point and drops the white point, which
    is what this does: pure black ink becomes ``black`` and white paper
    becomes ``white``.
    """

    def apply(image):
        import numpy as np
        from PIL import Image

        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        arr = black + arr * (white - black)
        return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), "RGB")

    return apply


def _shadow(kind):
    """Multiply the page by an illumination field.

    A phone held over a page puts a soft gradient across it -- the hand and
    the phone itself block part of the light. The field is smooth and
    multiplicative, so ink and paper darken together and local contrast
    survives where the illumination is high.
    """

    def apply(image):
        import numpy as np
        from PIL import Image

        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        height, width = arr.shape[:2]
        ys, xs = np.mgrid[0:height, 0:width]
        u = xs / max(width - 1, 1)
        v = ys / max(height - 1, 1)

        if kind == "left":
            field = 0.42 + 0.58 * np.clip(u / 0.6, 0, 1)
        elif kind == "corner":
            distance = np.sqrt((1 - u) ** 2 + (1 - v) ** 2) / np.sqrt(2)
            field = 0.35 + 0.65 * np.clip(distance / 0.75, 0, 1)
        elif kind == "diagonal":
            field = 0.40 + 0.60 * np.clip((u + v) / 1.3, 0, 1)
        else:
            raise ValueError("unknown shadow {0!r}".format(kind))

        arr = arr * field[:, :, None]
        return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"), "RGB")

    return apply


# --- crops ------------------------------------------------------------------
#
# Cutting from the right and the bottom, because that is where a page runs out
# of frame when someone photographs it holding the top-left corner, and
# because the rightmost columns are the units and reference-range columns --
# the ones whose loss changes what a correct reading *is*.
#
# The depth is computed from the page's own column geometry rather than fixed
# as a fraction of the width. A fixed fraction measures a different thing on
# every layout: 12% off a four-column page with a units column clips only the
# reference ranges and changes no ground truth at all, while the same 12% off
# a narrower three-column page can take the value column. Planning the cut
# against the drawn edges makes each crop level mean one thing across all
# three placements, and none of them destroys the value column -- a crop that
# removes the number leaves nothing to measure but silence.


def _keep_right_of_last_column(rendered):
    """Remove the rightmost column entirely."""
    return rendered.extras["edges"][-2], rendered.size[1]


def _keep_through_value(rendered):
    """Remove every column to the right of the value column.

    This is the crop that matters. With the unit in its own column it takes
    the unit and leaves the number, so the field's correct outcome flips from
    a quantity to a refusal. With the unit in the label it takes nothing that
    matters, which is the comparison: the same physical crop is survivable or
    fatal depending only on where the unit was printed.
    """
    columns = rendered.extras["columns"]
    edges = rendered.extras["edges"]
    return edges[columns.index("value") + 1], rendered.size[1]


def _keep_above_last_two_rows(rendered):
    """Remove the last two data rows."""
    return rendered.size[0], rendered.rows[-2].label.box[1]


def _crop_plans():
    return (
        ("crop_last_column", _keep_right_of_last_column, "rightmost column"),
        ("crop_to_value", _keep_through_value, "columns right of the value"),
        ("crop_last_rows", _keep_above_last_two_rows, "last two rows"),
    )


def _crop_to(keep):
    def apply(image):
        return image.crop((0, 0, keep[0], keep[1]))

    return apply


def _phone_photo(image):
    """Shadow, then camera angle, then focus, then compression.

    Applied in that order because that is the order the real pipeline
    introduces them: the lighting is on the page, the angle is the camera's,
    the blur is the lens's, and the compression is the file format's.
    """
    staged = _shadow("diagonal")(image)
    staged = _skew(3.0)(staged)
    return _blur(1.0)(staged)


def variants() -> List[Variant]:
    """The 26 variants applied to every base render."""
    out = [
        Variant(PRISTINE, PRISTINE, 0, "none", _identity),
    ]

    for index, quality in enumerate((90, 75, 60, 45, 30, 20), start=1):
        out.append(
            Variant(
                "jpeg_q{0}".format(quality),
                JPEG,
                index,
                "q{0}".format(quality),
                _identity,
                save_format="JPEG",
                jpeg_quality=quality,
            )
        )

    for degrees in (1, 2, 3, 4, 5):
        out.append(
            Variant(
                "skew_{0}deg".format(degrees),
                SKEW,
                degrees,
                "{0} deg".format(degrees),
                _skew(float(degrees)),
            )
        )

    for radius in (0.5, 1.0, 1.5, 2.0):
        out.append(
            Variant(
                "blur_{0}".format(radius),
                BLUR,
                radius,
                "sigma {0}".format(radius),
                _blur(radius),
            )
        )

    for index, (level, black, white) in enumerate(
        (("mild", 40, 215), ("moderate", 75, 195), ("severe", 105, 175)), start=1
    ):
        out.append(
            Variant(
                "contrast_{0}".format(level),
                CONTRAST,
                index,
                "{0} ({1}-{2})".format(level, black, white),
                _contrast(black, white),
            )
        )

    for index, kind in enumerate(("left", "diagonal", "corner"), start=1):
        out.append(
            Variant(
                "shadow_{0}".format(kind),
                SHADOW,
                index,
                kind,
                _shadow(kind),
            )
        )

    for index, (name, plan, removes) in enumerate(_crop_plans(), start=1):
        out.append(
            Variant(
                name,
                CROP,
                index,
                removes,
                _identity,       # replaced per page once the plan is resolved
                plan_crop=plan,
                removes=removes,
            )
        )

    out.append(
        Variant(
            "phone_photo",
            COMBINED,
            1,
            "skew 3 deg + blur 1.0 + diagonal shadow + JPEG q60",
            _phone_photo,
            save_format="JPEG",
            jpeg_quality=60,
        )
    )

    return out


#: Families in report order, control first.
FAMILY_ORDER = (PRISTINE, JPEG, SKEW, BLUR, CONTRAST, SHADOW, CROP, COMBINED)


def apply_to(rendered, variant: Variant):
    """Apply a variant to a rendered page.

    Returns ``(image, keep)`` where ``keep`` is the surviving ``(width,
    height)`` in the page's own pixels for a crop, and None otherwise.
    ``keep`` is what ``corpus.truth_from_render`` needs to work out which
    fields the crop turned into refusals and which it removed altogether.
    """
    if variant.plan_crop is None:
        return variant.apply(rendered.image), None

    keep = variant.plan_crop(rendered)
    return _crop_to(keep)(rendered.image), keep


def save(image, path, variant: Variant) -> None:
    """Write the degraded image in the variant's format."""
    if variant.save_format == "JPEG":
        image.convert("RGB").save(path, "JPEG", quality=variant.jpeg_quality)
    else:
        image.save(path, variant.save_format)
