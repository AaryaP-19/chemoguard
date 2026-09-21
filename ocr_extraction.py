"""
OCR extraction module for extracting hematological parameters from medical reports.
Uses OpenCV for image preprocessing and Tesseract OCR for text extraction.
"""

import cv2
import pytesseract
import re
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import os
from typing import Dict, Optional, Tuple


class HematologyExtractor:
    """Extract hematological parameters from medical report images."""
    
    # Common parameter patterns and units
    PARAMETER_PATTERNS = {
        'hb': [
            r'hb[:\s]*(\d+\.?\d*)\s*(?:g/dl|g/dL|gm/dl|gm/dL)',
            r'hemoglobin[:\s]*(\d+\.?\d*)\s*(?:g/dl|g/dL|gm/dl|gm/dL)',
            r'hgb[:\s]*(\d+\.?\d*)\s*(?:g/dl|g/dL|gm/dl|gm/dL)',
        ],
        'rbc': [
            r'rbc[:\s]*(\d+\.?\d*)\s*(?:million/μl|million/ul|millions/μl|millions/ul|10\^6/μl|10\^6/ul|million)',
            r'red\s*blood\s*cell[:\s]*(\d+\.?\d*)',
        ],
        'wbc': [
            r'wbc[:\s]*(\d+\.?\d*)\s*(?:cells/μl|cells/ul|/μl|/ul|thousand/μl|thousand/ul)',
            r'white\s*blood\s*cell[:\s]*(\d+\.?\d*)',
            r'tlc[:\s]*(\d+\.?\d*)',  # Total Leukocyte Count
        ],
        'platelets': [
            r'platelet[:\s]*(\d+\.?\d*)\s*(?:cells/μl|cells/ul|/μl|/ul|thousand/μl|thousand/ul|k/μl|k/ul)',
            r'plt[:\s]*(\d+\.?\d*)',
        ],
        'anc': [
            r'anc[:\s]*(\d+\.?\d*)\s*(?:cells/μl|cells/ul|/μl|/ul)',
            r'absolute\s*neutrophil[:\s]*(\d+\.?\d*)',
        ],
        'neutrophils_percent': [
            r'neutrophil[:\s]*(\d+\.?\d*)\s*%',
            r'neutrophils[:\s]*(\d+\.?\d*)\s*%',
            r'neut[:\s]*(\d+\.?\d*)\s*%',
        ],
        'hematocrit': [
            r'hematocrit[:\s]*(\d+\.?\d*)\s*%',
            r'hct[:\s]*(\d+\.?\d*)\s*%',
            r'pcv[:\s]*(\d+\.?\d*)\s*%',  # Packed Cell Volume
        ],
        'mcv': [
            r'mcv[:\s]*(\d+\.?\d*)\s*(?:fl|fL)',
            r'mean\s*corpuscular\s*volume[:\s]*(\d+\.?\d*)',
        ],
        'mch': [
            r'mch[:\s]*(\d+\.?\d*)\s*(?:pg|pico\s*gram)',
            r'mean\s*corpuscular\s*hemoglobin[:\s]*(\d+\.?\d*)',
        ],
        'mchc': [
            r'mchc[:\s]*(\d+\.?\d*)\s*(?:g/dl|g/dL)',
            r'mean\s*corpuscular\s*hemoglobin\s*concentration[:\s]*(\d+\.?\d*)',
        ],
    }
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """Initialize OCR extractor."""
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
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
    
    def extract_parameters(self, text: str) -> Dict[str, Optional[float]]:
        """Extract hematological parameters from text."""
        text_lower = text.lower()
        extracted = {}
        
        for param_name, patterns in self.PARAMETER_PATTERNS.items():
            value = None
            for pattern in patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1))
                        # Validate ranges
                        if self._validate_value(param_name, value):
                            extracted[param_name] = value
                            break
                    except (ValueError, IndexError):
                        continue
            
            if param_name not in extracted:
                extracted[param_name] = None
        
        return extracted
    
    def _validate_value(self, param_name: str, value: float) -> bool:
        """Validate if extracted value is within reasonable ranges."""
        ranges = {
            'hb': (3.0, 20.0),  # g/dL
            'rbc': (1.0, 8.0),  # million/μL
            'wbc': (0.5, 100.0),  # thousand/μL or cells/μL (will normalize)
            'platelets': (10.0, 2000.0),  # thousand/μL or cells/μL
            'anc': (0.1, 50.0),  # thousand/μL
            'neutrophils_percent': (0.0, 100.0),
            'hematocrit': (10.0, 60.0),
            'mcv': (50.0, 150.0),  # fL
            'mch': (15.0, 40.0),  # pg
            'mchc': (25.0, 40.0),  # g/dL
        }
        
        if param_name in ranges:
            min_val, max_val = ranges[param_name]
            return min_val <= value <= max_val
        
        return True
    
    def process_file(self, file_path: str, file_type: str) -> Tuple[Dict[str, Optional[float]], str, float]:
        """Process file and extract parameters."""
        text = ""
        confidence = 0.0
        
        if file_type.lower() == 'pdf':
            text, confidence = self.extract_text_from_pdf(file_path)
        elif file_type.lower() in ['jpg', 'jpeg', 'png', 'tiff', 'bmp']:
            text, confidence = self.extract_text_from_image(file_path)
        else:
            return {}, "", 0.0
        
        # Extract parameters from text
        parameters = self.extract_parameters(text)
        
        return parameters, text, confidence

