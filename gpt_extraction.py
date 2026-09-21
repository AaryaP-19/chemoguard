"""
GPT-4o-mini based extraction module for extracting hematological parameters from medical reports.
Uses OpenAI GPT-4o-mini with vision capabilities to extract values from images/PDFs.

The model transcribes; it does not convert. The prompt this replaces told it
to "convert all values to standard units ... If WBC is in cells/uL (e.g.
8000), convert to thousands/uL (8.0)", which put a unit conversion inside a
vision model, discarded the unit the report actually printed, and returned a
bare number that nothing downstream could check. Now it returns the value and
the unit string as printed, and the conversion happens in units.Quantity
where it can be tested.
"""

import base64
import json
import os
from typing import Dict, Optional
from PIL import Image
from pdf2image import convert_from_path
import io

from cbc import ANALYTES, panel_from_extraction as build_panel_from_extraction


class GPTExtractor:
    """Extract hematological parameters using GPT-4o-mini vision model."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize GPT extractor."""
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def convert_pdf_to_image(self, pdf_path: str) -> str:
        """Convert PDF to image and return path."""
        try:
            images = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=1)
            if not images:
                raise ValueError("Could not extract image from PDF")
            
            # Save first page as image
            temp_image_path = pdf_path.replace('.pdf', '_temp.png')
            images[0].save(temp_image_path, 'PNG')
            return temp_image_path
        except Exception as e:
            raise Exception(f"Error converting PDF to image: {str(e)}")
    
    #: Transcribe, do not convert. Every instruction that asked the model to
    #: normalise a scale has been removed, and the unit is now a required
    #: part of each answer.
    PROMPT = """Analyze this medical laboratory report image and transcribe the following hematological parameters.
Return ONLY a valid JSON object. If a parameter is not present on the report, set it to null.

For each parameter, report the number and the unit EXACTLY AS PRINTED on the report:

{
    "rbc":                 {"value": <number>, "unit": "<unit as printed>"} or null,
    "wbc":                 {"value": <number>, "unit": "<unit as printed>"} or null,
    "platelets":           {"value": <number>, "unit": "<unit as printed>"} or null,
    "anc":                 {"value": <number>, "unit": "<unit as printed>"} or null,
    "neutrophils_percent": {"value": <number>, "unit": "%"} or null,
    "hb":                  {"value": <number>, "unit": "<unit as printed>"} or null,
    "hematocrit":          {"value": <number>, "unit": "<unit as printed>"} or null,
    "mcv":                 {"value": <number>, "unit": "<unit as printed>"} or null,
    "mch":                 {"value": <number>, "unit": "<unit as printed>"} or null,
    "mchc":                {"value": <number>, "unit": "<unit as printed>"} or null
}

Critical rules:
- DO NOT convert, rescale or normalise any value. Transcribe the digits as printed.
  If the report says 85000, report 85000. If it says 85, report 85.
- Copy the unit string verbatim from the report: "cells/uL", "K/uL", "10^9/L",
  "millions/uL", "g/dL", "fL", "pg", "%" -- whatever is actually written.
- If a value is printed but its unit is not shown anywhere for that row
  (not in a units column, not in a header, not beside the number), set
  "unit" to null. Do not infer the unit from how large or small the number is.
- Do not guess a value that is not printed.
- Return valid JSON only, no additional text."""

    def extract_from_image(self, image_path: str) -> Dict[str, Optional[dict]]:
        """Transcribe value-and-unit pairs from an image using GPT-4o-mini.

        Returns the raw ``{analyte: {"value": ..., "unit": ...}}`` mapping.
        Interpretation happens in ``process_file``; this method does no
        conversion and no validation beyond parsing the response.
        """
        try:
            # Encode image
            base64_image = self.encode_image(image_path)

            # Call GPT-4o-mini with vision
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=800
            )

            # Parse JSON response
            result_text = response.choices[0].message.content
            extracted_data = json.loads(result_text)

            return {key: extracted_data.get(key) for key in ANALYTES}

        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response from GPT: {str(e)}")
        except Exception as e:
            raise Exception(f"Error extracting from image: {str(e)}")
    
    def process_file(self, file_path: str, file_type: str):
        """Process a file and return ``(panel, confidence, skipped)``.

        ``panel`` is a CBCPanel of canonicalised Quantities. ``skipped`` maps
        analytes the model read but whose unit could not be established, so
        the caller can show them rather than pretend they were never there.

        Confidence counts only the three analytes a prediction requires, and
        counts them only if they made it into the panel -- a value read
        without a usable unit contributes nothing, because it cannot be used.
        """
        temp_image_path = None

        try:
            # Convert PDF to image if needed
            if file_type.lower() == 'pdf':
                temp_image_path = self.convert_pdf_to_image(file_path)
                image_path = temp_image_path
            elif file_type.lower() in ['jpg', 'jpeg', 'png', 'tiff', 'bmp']:
                image_path = file_path
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            raw = self.extract_from_image(image_path)
            panel, skipped = build_panel_from_extraction(raw)

            main_params = ['rbc', 'wbc', 'platelets']
            found = sum(1 for p in main_params if panel.get(p) is not None)
            confidence = (found / len(main_params)) * 100

            return panel, confidence, skipped

        finally:
            # Clean up temporary image file
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                except:
                    pass

