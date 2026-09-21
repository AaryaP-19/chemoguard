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
_NUMBER = r'(?P<value>\d[\d,]*\.?\d*)'

_U = r'(?P<unit>{0})'
#: Unit alternations per dimension. Spellings only; meaning lives in units.py.
_MU = r'[uµμ]'
_U_COUNT = _U.format(
    r'cells?\s*/\s*{0}l|k\s*/\s*{0}l|thousands?\s*/\s*{0}l'
    r'|millions?\s*/\s*{0}l|m\s*/\s*{0}l|10\^?[369]\s*/\s*[l{0}]+'
    r'|/\s*cu\s*mm|/\s*mm3|/\s*{0}l'.format(_MU)
)
_U_MASSCONC = _U.format(r'g\s*/\s*dl|gm\s*/\s*dl|gms\s*/\s*dl|g\s*/\s*l')
_U_PERCENT = _U.format(r'%')
_U_VOLUME = _U.format(r'fl')
_U_MASS = _U.format(r'pg')

_SEP = r'[:\s]*'
_GAP = r'\s*'

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

    def __init__(self, tesseract_cmd: Optional[str] = None):
        """Initialize OCR extractor."""
        self._tesseract_cmd = tesseract_cmd

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
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(processed, config=custom_config)
            
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
                custom_config = r'--oem 3 --psm 6'
                text = pytesseract.image_to_string(processed, config=custom_config)
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

        for param_name, patterns in self.PARAMETER_PATTERNS.items():
            # Unit-bearing patterns first, across every label spelling. Only
            # if none of them matches is a bare number considered, and then
            # only as a sighting with unit None -- which the caller turns
            # into a refusal, never into a value.
            found = self._first_match(patterns, text_lower)
            if found is None:
                found = self._first_match(
                    self.UNITLESS_PATTERNS[param_name], text_lower
                )
            extracted[param_name] = found

        return extracted

    @staticmethod
    def _first_match(patterns, text):
        """The first pattern that matches, as ``{"value", "unit"}`` or None."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue
            try:
                value = float(match.group("value").replace(",", ""))
            except (ValueError, IndexError):
                continue
            unit = match.groupdict().get("unit")
            return {"value": value, "unit": unit.strip() if unit else None}
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

