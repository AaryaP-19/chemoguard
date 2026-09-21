"""
GPT-4o-mini based extraction module for extracting hematological parameters from medical reports.
Uses OpenAI GPT-4o-mini with vision capabilities to extract values from images/PDFs.
"""

import base64
import json
import os
from typing import Dict, Optional
from PIL import Image
from pdf2image import convert_from_path
import io


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
    
    def extract_from_image(self, image_path: str) -> Dict[str, Optional[float]]:
        """Extract parameters from image using GPT-4o-mini."""
        try:
            # Encode image
            base64_image = self.encode_image(image_path)
            
            # Prepare prompt for JSON mode
            prompt = """Analyze this medical laboratory report image and extract the following hematological parameters. 
Return ONLY a valid JSON object with the extracted values. If a parameter is not found, set it to null.

Required JSON format:
{
    "rbc": <number in millions/μL or null>,
    "wbc": <number in cells/μL or thousands/μL or null>,
    "platelets": <number in cells/μL or thousands/μL or null>,
    "anc": <number in cells/μL or thousands/μL or null>,
    "neutrophils_percent": <percentage 0-100 or null>,
    "hb": <number in g/dL or null>,
    "hematocrit": <percentage or null>,
    "mcv": <number in fL or null>,
    "mch": <number in pg or null>,
    "mchc": <number in g/dL or null>
}

Important:
- Convert all values to standard units (RBC: millions/μL, WBC: thousands/μL, Platelets: thousands/μL)
- If WBC is in cells/μL (e.g., 8000), convert to thousands/μL (8.0)
- If Platelets is in cells/μL (e.g., 250000), convert to thousands/μL (250)
- Extract exact numeric values only
- Return valid JSON only, no additional text"""
            
            # Call GPT-4o-mini with vision
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
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
                max_tokens=500
            )
            
            # Parse JSON response
            result_text = response.choices[0].message.content
            extracted_data = json.loads(result_text)
            
            # Validate and convert values
            parameters = {}
            for key in ['rbc', 'wbc', 'platelets', 'anc', 'neutrophils_percent', 
                       'hb', 'hematocrit', 'mcv', 'mch', 'mchc']:
                value = extracted_data.get(key)
                if value is not None:
                    try:
                        parameters[key] = float(value)
                    except (ValueError, TypeError):
                        parameters[key] = None
                else:
                    parameters[key] = None
            
            return parameters
            
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response from GPT: {str(e)}")
        except Exception as e:
            raise Exception(f"Error extracting from image: {str(e)}")
    
    def process_file(self, file_path: str, file_type: str) -> tuple[Dict[str, Optional[float]], float]:
        """Process file and extract parameters."""
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
            
            # Extract parameters
            parameters = self.extract_from_image(image_path)
            
            # Calculate confidence (simple heuristic - if we got main parameters, confidence is high)
            main_params = ['rbc', 'wbc', 'platelets']
            found_params = sum(1 for p in main_params if parameters.get(p) is not None)
            confidence = (found_params / len(main_params)) * 100
            
            return parameters, confidence
            
        finally:
            # Clean up temporary image file
            if temp_image_path and os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                except:
                    pass

