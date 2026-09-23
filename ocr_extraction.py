"""
OCR extraction module for extracting hematological parameters from medical reports.
Uses OpenCV for image preprocessing and Tesseract OCR for text extraction.

The patterns below capture the unit as well as the number. They always
matched the unit text -- ``cells/ul|thousand/ul|k/ul`` was right there in
every alternation -- but it sat outside the capture group, so only
``group(1)`` survived and the unit was read off the page and thrown away.
Moving one parenthesis recovers it. A row whose unit cannot be read is
reported as unusable rather than returned as a bare number.
"""

import re
import os
from typing import Dict, Optional, Tuple

from cbc import panel_from_extraction

# OpenCV, Tesseract and Poppler are imported inside the methods that need
# them. Reading a unit off a line of text needs none of the three, and
# pinning the pattern layer to a 60 MB vision stack meant it could not be
# exercised anywhere those binaries were missing -- which is most CI.

# Every pattern captures a named `value` group and, where the report shows
# one, a named `unit` group. Named rather than positional because the unit is
# printed before the number as often as after it -- "Neutrophils %  55" puts
# it in the label column -- and group order should not decide whether a unit
# is readable.
#: A printed number, separators and all. Both ``.`` and ``,`` are admitted in
#: both roles and the role is decided in ``parse_number``, not here -- the
#: pattern this replaces was ``\d[\d,]*\.?\d*`` finished with
#: ``float(text.replace(",", ""))``, which hard-coded the US convention:
#: commas stripped, points honoured. On a European page that read 6.500 as
#: 6.5 and 0,9 as 9. See CLAUDE.md Finding 9 / M1.
_NUMBER = r'(?P<value>\d[\d.,]*)'

_U = r'(?P<unit>{0})'
#: Unit alternations per dimension. Spellings only; meaning lives in units.py.
#:
#: ``_MU_CHARS`` is the bare character list and ``_MU`` the class built from
#: it. Both exist because one of them has to be interpolated *inside* another
#: character class below, and nesting a class inside a class does not do what
#: it looks like: ``'[l{0}]'.format('[uµμ]')`` compiles to the class
#: ``[l[uµμ]`` followed by a **required literal ']'``, so the SI
#: alternative matched ``10^9/L]`` and never ``10^9/L``. CLAUDE.md Finding
#: 9 / M2.
_MU_CHARS = r'uµμ'
_MU = r'[' + _MU_CHARS + r']'

#: The same class plus ``p``, for the microlitre *denominator only*.
#:
#: Tesseract reads the micro sign as ``p``: a pristine render at OCR
#: confidence 91 returns ``cells/pL`` where the page prints ``cells/µL``. That
#: one glyph was the dominant cause of lost yield in the OCR arm -- every
#: count analyte prints a microlitre denominator and no other analyte does,
#: which is why the counts scored 0.221-0.305 recall while everything else
#: scored 0.670-0.779. CLAUDE.md Finding 9.
#:
#: Accepting ``pL`` here introduces no real ambiguity: a picolitre is a
#: millionth of a microlitre, so no laboratory reports a cell count per
#: picolitre, and there is nothing for ``cells/pL`` to be confused *with*.
#:
#: Scoped deliberately to the denominator after the slash. A blanket ``p``
#: for ``u`` would rewrite ``pg`` -- picogram, which is MCH's real unit -- and
#: that is a live collision rather than a hypothetical one.
_MU_OCR_CHARS = _MU_CHARS + r'p'
_MU_OCR = r'[' + _MU_OCR_CHARS + r']'

#: Rewrites a known OCR corruption back to the glyph it corrupts. Applied to
#: the captured unit string, never to the alias table: ``units.py`` holds
#: spellings that laboratories actually print, and ``cells/pL`` is not one --
#: it is a misreading, and misreadings belong to the layer that produced them.
_OCR_UNIT_REPAIRS = ((re.compile(r'/\s*p\s*l', re.IGNORECASE), '/uL'),)

#: Superscript digits. A report prints 10⁹/L, not 10^9/L, and a superscript
#: run needs no exponent marker because the raised digits *are* the notation.
_SUPERSCRIPT_DIGITS = r'⁰¹²³⁴⁵⁶⁷⁸⁹'

#: The SI count exponents that exist on a CBC: 10^9/L for the white cell and
#: platelet counts, 10^12/L for the red cell count, and the 10^3 and 10^6
#: per-microlitre forms. ``12`` is listed first so a two-digit exponent is
#: preferred over the ``1`` of a one-digit alternative.
_SI_EXPONENT = r'(?:12|0|3|6|9)'

#: An exponent marker is now **required**. ``10\^?`` made the caret optional,
#: so ``109/L`` matched and was read as an SI unit -- which is both wrong and
#: ambiguous, because ``109`` is a perfectly possible printed value. The cost
#: of requiring it is that OCR flattening a superscript to a plain digit
#: ("10⁹/L" read as "109/L") now yields a refusal instead of a match. That is
#: the right way round: a refusal is recoverable and a wrong scale is not.
_SI_COUNT = (
    r'10\s*(?:[\^e*]\s*' + _SI_EXPONENT + r'|[' + _SUPERSCRIPT_DIGITS + r']{1,2})'
    r'\s*/\s*[l' + _MU_CHARS + r']+'
)

#: Only the microlitre-denominator alternatives take ``_MU_OCR``. The SI and
#: per-cubic-millimetre forms have no micro sign to misread.
_U_COUNT = _U.format(
    r'cells?\s*/\s*{0}l|k\s*/\s*{0}l|thousands?\s*/\s*{0}l'
    r'|millions?\s*/\s*{0}l|m\s*/\s*{0}l|{1}'
    r'|/\s*cu\s*mm|/\s*mm3|/\s*{0}l'.format(_MU_OCR, _SI_COUNT)
)
_U_MASSCONC = _U.format(r'g\s*/\s*dl|gm\s*/\s*dl|gms\s*/\s*dl|g\s*/\s*l')
_U_PERCENT = _U.format(r'%')
_U_VOLUME = _U.format(r'fl')
_U_MASS = _U.format(r'pg')

_SEP = r'[:\s]*'
_GAP = r'\s*'

_SEPARATORS = ".,"
#: A thousands separator is always followed by exactly three digits. That one
#: fact does most of the work below.
_GROUP_SIZE = 3


class AmbiguousNumber(ValueError):
    """A printed number whose separators do not determine its value.

    Raised for the genuinely undecidable case and nothing wider. ``1,234``
    alone is 1234 on a US page and 1.234 on a European one, and there is no
    way to tell from the token. Refusing it is the same stance this module
    already takes on a missing unit: a reading that cannot be pinned down is
    surfaced for an operator to resolve, never guessed at.
    """


def repair_ocr_unit(unit):
    """Undo a known OCR corruption in a captured unit string.

    Only the micro sign, only in a microlitre denominator, and only because
    Tesseract reads it as ``p`` reliably enough to be worth naming. The
    rewrite happens here rather than in the alias table so that ``units.py``
    keeps holding spellings a laboratory would print -- a corruption admitted
    into the unit vocabulary would then be accepted from the keyboard and
    from the API too, where it has no excuse.
    """
    if not unit:
        return unit
    for pattern, replacement in _OCR_UNIT_REPAIRS:
        unit = pattern.sub(replacement, unit)
    return unit


def _split_number(token):
    """``"1.234,56"`` -> ``(["1", "234", "56"], [".", ","])``."""
    groups, separators, current = [], [], ""

    for character in token:
        if character in _SEPARATORS:
            separators.append(character)
            groups.append(current)
            current = ""
        else:
            current += character

    groups.append(current)
    return groups, separators


def parse_number(token, decimal_mark=None):
    """The value a printed number denotes.

    The rule rests on the fact that a thousands separator is always followed
    by exactly three digits:

    * no separator -- a plain integer.
    * the last separator is followed by a run that is **not** three digits
      long -- it can only be a decimal mark. ``0,9`` is nine tenths; ``4,80``
      is four point eight; ``1234,5`` likewise, since a grouped number would
      not have four digits in its leading group either.
    * separators of two different characters -- the last is the decimal mark
      and the others group thousands. ``1.234,56``.
    * every separator the same character, each followed by exactly three
      digits -- ambiguous on the token alone, because ``6,500`` is six and a
      half on a European page and six thousand five hundred on a US one.

    ``decimal_mark`` resolves that last case when the *page* has settled it;
    see ``infer_decimal_mark``. Without it, ambiguity raises.

    Raises ``AmbiguousNumber``, never a guess.
    """
    cleaned = str(token).strip().strip(_SEPARATORS)
    if not cleaned:
        raise AmbiguousNumber("{0!r} holds no digits".format(token))

    groups, separators = _split_number(cleaned)
    if not all(group.isdigit() for group in groups):
        raise AmbiguousNumber("{0!r} is not a number".format(token))

    if not separators:
        return float(cleaned)

    # Any separator before the last one can only be grouping thousands, and
    # grouping is only grouping if every group it makes is three digits long.
    for group in groups[1:-1]:
        if len(group) != _GROUP_SIZE:
            raise AmbiguousNumber(
                "{0!r} has a {1}-digit group; neither separator convention "
                "produces that".format(token, len(group))
            )

    last, tail = separators[-1], groups[-1]

    if len(set(separators)) > 1:
        # Two different separator characters: the last one is the decimal
        # mark, because the other is doing the grouping.
        return float("".join(groups[:-1]) + "." + tail)

    if len(separators) > 1:
        # One character, used more than once. It cannot be the decimal mark,
        # because a number has only one of those -- so every occurrence is
        # grouping thousands and the value is an integer. Unambiguous without
        # any page evidence.
        if len(tail) != _GROUP_SIZE:
            raise AmbiguousNumber(
                "{0!r} repeats {1!r} and then ends in a {2}-digit group; no "
                "convention groups like that".format(token, last, len(tail))
            )
        return float("".join(groups))

    if len(tail) != _GROUP_SIZE:
        # A single separator followed by anything other than three digits can
        # only be a decimal mark, whatever convention the page follows.
        return float(groups[0] + "." + tail)

    if decimal_mark is None:
        raise AmbiguousNumber(
            "{0!r} is undecidable: {1!r} followed by exactly three digits is "
            "a thousands separator on one page and a decimal mark on "
            "another, and nothing else on this page settles which."
            .format(token, last)
        )

    if last == decimal_mark:
        return float("".join(groups[:-1]) + "." + tail)
    return float("".join(groups))


#: Every number-shaped token on a page, for the document-level pass.
_NUMBER_TOKEN = re.compile(r'\d[\d.,]*')


def infer_decimal_mark(text):
    """Which character this page uses as its decimal mark, if it says.

    A page that prints ``13.0`` or ``83.5`` anywhere has shown that ``.`` is
    its decimal mark, and therefore that ``,`` groups its thousands. That
    settles ``6,500`` on the same page without guessing -- the evidence comes
    from the document, not from a convention we picked.

    Returns ``"."``, ``","`` or None. None when the page offers no
    unambiguous number, and also when it offers contradictory ones, which
    happens on a page that mixes conventions or where OCR invented a
    separator. Contradiction has to yield None rather than a majority vote:
    a page we cannot read consistently is a page to refuse.
    """
    seen = set()

    for match in _NUMBER_TOKEN.finditer(text):
        cleaned = match.group().strip(_SEPARATORS)
        groups, separators = _split_number(cleaned)

        if not separators or not all(g.isdigit() for g in groups):
            continue
        if len(set(separators)) > 1:
            seen.add(separators[-1])
            continue
        if len(separators) == 1 and len(groups[-1]) != _GROUP_SIZE:
            seen.add(separators[0])

    return seen.pop() if len(seen) == 1 else None

#: Analyte -> the unit alternation valid for it.
_ANALYTE_UNIT = {
    'hb': _U_MASSCONC,
    'rbc': _U_COUNT,
    'wbc': _U_COUNT,
    'platelets': _U_COUNT,
    'anc': _U_COUNT,
    'neutrophils_percent': _U_PERCENT,
    'hematocrit': _U_PERCENT,
    'mcv': _U_VOLUME,
    'mch': _U_MASS,
    'mchc': _U_MASSCONC,
}

#: Analyte -> the ways its row is labelled. Longest-first within each entry,
#: so "mean corpuscular haemoglobin concentration" is tried before "mchc".
_ANALYTE_LABELS = {
    'hb': (r'h(?:ae|e)moglobin(?:\s*\(hb\))?', r'hgb', r'hb'),
    'rbc': (r'red\s*blood\s*cells?(?:\s*\(rbc\))?', r'rbc'),
    'wbc': (r'white\s*blood\s*cells?(?:\s*\(wbc\))?', r'wbc', r'tlc'),
    'platelets': (r'platelets?', r'plt'),
    'anc': (r'absolute\s*neutrophil\s*count(?:\s*\(anc\))?', r'anc'),
    'neutrophils_percent': (r'neutrophils?', r'neut'),
    'hematocrit': (r'h(?:ae|e)matocrit(?:\s*\(hct\))?', r'hct', r'pcv'),
    'mcv': (r'mean\s*corpuscular\s*volume', r'mcv'),
    'mch': (r'mean\s*corpuscular\s*h(?:ae|e)moglobin(?!\s*concentration)', r'mch'),
    'mchc': (r'mean\s*corpuscular\s*h(?:ae|e)moglobin\s*concentration', r'mchc'),
}


def _value_then_unit(label, unit):
    """"Platelets  85000 cells/uL" -- the unit follows the number."""
    return label + _SEP + _NUMBER + _GAP + unit


def _unit_then_value(label, unit):
    """"Neutrophils %  55" -- the unit is part of the row label."""
    return label + _GAP + r'\(?' + unit + r'\)?' + _SEP + _NUMBER


def _value_only(label, unit):
    """"Platelets  250,000" -- a number with no unit anywhere on the row.

    Matched deliberately, and never turned into a value. Without it a row
    the extractor could not fully read is indistinguishable from an analyte
    the report does not carry, and the operator is told nothing. With it,
    the reading is surfaced as unusable and they can supply the unit.
    """
    return label + _SEP + _NUMBER


def _build(builders):
    return {
        analyte: [
            build(label, _ANALYTE_UNIT[analyte])
            for label in labels
            for build in builders
        ]
        for analyte, labels in _ANALYTE_LABELS.items()
    }


class HematologyExtractor:
    """Extract hematological parameters from medical report images."""

    #: Tried in order. A unit-bearing match always wins over a bare number,
    #: for every label spelling, before any bare number is considered.
    PARAMETER_PATTERNS = _build((_value_then_unit, _unit_then_value))
    UNITLESS_PATTERNS = _build((_value_only,))

    #: Tesseract page-segmentation mode.
    #:
    #: **Was 6, which reads almost nothing off a CBC table.** psm 6 means
    #: "assume a single uniform block of text", and a bordered table with
    #: columns is not that -- Tesseract runs the rows together across the
    #: grid lines and returns rubbish. On a pristine, high-contrast generated
    #: report at OCR confidence 90, psm 6 returned
    #: ``REGIONAL PATHOLOGY LABORATORY / Complete Blood Count (CBC) /
    #: Cr ee / rucr ||| ars`` -- the two title lines perfectly and the whole
    #: table destroyed.
    #:
    #: Measured over a stratified sample of 24 images spanning the clean
    #: control and the worst level of every degradation family, end to end
    #: through the real pipeline (CLAUDE.md Finding 9 / M4):
    #:
    #:     psm   precision   recall
    #:     3     0.987       0.419
    #:     4     0.974       0.425
    #:     11    1.000       0.229
    #:     12    0.977       0.235
    #:     6     1.000       0.022    <- as shipped
    #:
    #: 3 over 4 on precision and on silent errors, which is the trade this
    #: application makes everywhere else. The sparse-text modes 11 and 12
    #: read *more* characters and score worse, because they do not preserve
    #: reading order and the patterns need a row's label, value and unit on
    #: one line.
    #:
    #: Note the ceiling this does not reach: the pattern layer scores recall
    #: 0.997 on the same corpus given perfect text. Everything between that
    #: and the number above is glyph reading.
    PAGE_SEGMENTATION_MODE = 3

    #: LSTM engine. Unchanged; only the segmentation mode was wrong.
    OCR_ENGINE_MODE = 3

    def __init__(self, tesseract_cmd: Optional[str] = None, psm: Optional[int] = None):
        """Initialize OCR extractor.

        ``psm`` overrides the page-segmentation mode, so the extraction
        benchmark can report the as-shipped mode alongside the corrected one
        rather than asking the reader to take the difference on trust.
        """
        self._tesseract_cmd = tesseract_cmd
        self.psm = self.PAGE_SEGMENTATION_MODE if psm is None else psm

    def _ocr_config(self) -> str:
        """The Tesseract command-line configuration, in one place."""
        return "--oem {0} --psm {1}".format(self.OCR_ENGINE_MODE, self.psm)

    @staticmethod
    def _vision():
        """Import the vision stack on demand.

        Deferred so extract_parameters -- which is pure text work -- can be
        imported and tested without OpenCV, Tesseract or Poppler installed.
        """
        import cv2
        import numpy as np
        import pytesseract

        return cv2, np, pytesseract

    def _tesseract(self):
        _cv2, _np, pytesseract = self._vision()
        if self._tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self._tesseract_cmd
        return pytesseract

    def preprocess_image(self, image):
        """Preprocess image for better OCR results."""
        cv2, np, _pytesseract = self._vision()
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Apply thresholding
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def extract_text_from_image(self, image_path: str) -> Tuple[str, float]:
        """Extract text from image using OCR."""
        cv2, _np, _pt = self._vision()
        pytesseract = self._tesseract()
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return "", 0.0
            
            # Preprocess
            processed = self.preprocess_image(image)
            
            # Perform OCR
            text = pytesseract.image_to_string(
                processed, config=self._ocr_config()
            )
            
            # Get confidence (average)
            data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return text, avg_confidence
        
        except Exception as e:
            print(f"Error in OCR extraction: {e}")
            return "", 0.0
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, float]:
        """Extract text from PDF by converting to images."""
        from pdf2image import convert_from_path

        cv2, np, _pt = self._vision()
        pytesseract = self._tesseract()
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=300)
            
            all_text = []
            all_confidences = []
            
            for image in images:
                # Convert PIL image to OpenCV format
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Preprocess and extract
                processed = self.preprocess_image(img_array)
                text = pytesseract.image_to_string(
                    processed, config=self._ocr_config()
                )
                all_text.append(text)
                
                # Get confidence
                data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                if confidences:
                    all_confidences.extend(confidences)
            
            combined_text = "\n".join(all_text)
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            
            return combined_text, avg_confidence
        
        except Exception as e:
            print(f"Error extracting from PDF: {e}")
            return "", 0.0
    
    def extract_parameters(self, text: str) -> Dict[str, Optional[dict]]:
        """Transcribe value-and-unit pairs from OCR text.

        Returns ``{analyte: {"value": float, "unit": str}}``. No conversion
        and no range check: the range check that used to live here was doing
        unit inference by another name. Its WBC bound of (0.5, 100.0)
        discarded any reading in cells/uL, so a genuine 5800 cells/uL was
        dropped on the floor while 5.8 thousand/uL passed. Plausibility is
        now checked once, after canonicalisation, in cbc.quantity_for, where
        the unit is known.
        """
        text_lower = text.lower()
        extracted = {}

        # Decide the page's decimal mark once, before reading any row. A
        # number's separators are ambiguous in isolation and often settled by
        # the page around it, so this has to be a document-level pass -- read
        # row by row, "6,500" is undecidable even on a page that prints
        # "13.0" three lines above it.
        decimal_mark = infer_decimal_mark(text_lower)

        for param_name, patterns in self.PARAMETER_PATTERNS.items():
            # Unit-bearing patterns first, across every label spelling. Only
            # if none of them matches is a bare number considered, and then
            # only as a sighting with unit None -- which the caller turns
            # into a refusal, never into a value.
            found = self._first_match(patterns, text_lower, decimal_mark)
            if found is None:
                found = self._first_match(
                    self.UNITLESS_PATTERNS[param_name], text_lower, decimal_mark
                )
            extracted[param_name] = found

        return extracted

    @staticmethod
    def _first_match(patterns, text, decimal_mark=None):
        """The first pattern that matches, as ``{"value", "unit"}`` or None.

        A number whose separators cannot be resolved is returned with
        ``value`` None and ``ambiguous_number`` set to the token as printed.
        It is deliberately *not* skipped here: a row the extractor sighted
        and could not pin down has to reach the operator, and returning None
        would make it indistinguishable from an analyte the report does not
        carry. ``cbc.panel_from_extraction`` turns it into a refusal with a
        reason -- the same treatment a missing unit gets.
        """
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue

            token = match.group("value")
            unit = match.groupdict().get("unit")
            unit = repair_ocr_unit(unit.strip()) if unit else None

            try:
                value = parse_number(token, decimal_mark)
            except AmbiguousNumber as exc:
                return {
                    "value": None,
                    "unit": unit,
                    "ambiguous_number": token,
                    "detail": str(exc),
                }

            return {"value": value, "unit": unit}
        return None

    def process_file(self, file_path: str, file_type: str):
        """Process a file and return ``(panel, text, confidence, skipped)``.

        ``panel`` holds canonicalised Quantities; ``skipped`` holds anything
        read without a usable unit.
        """
        text = ""
        confidence = 0.0

        if file_type.lower() == 'pdf':
            text, confidence = self.extract_text_from_pdf(file_path)
        elif file_type.lower() in ['jpg', 'jpeg', 'png', 'tiff', 'bmp']:
            text, confidence = self.extract_text_from_image(file_path)
        else:
            empty, _ = panel_from_extraction({})
            return empty, "", 0.0, {}

        panel, skipped = panel_from_extraction(self.extract_parameters(text))
        return panel, text, confidence, skipped

