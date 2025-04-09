import os
import sys
from PIL import Image
import numpy as np
import cv2
import pydicom
from skimage import feature, filters, measure
from scipy import ndimage
import logging
import traceback
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

logger = logging.getLogger(__name__)

class MedicalImageAnalyzer:
    def __init__(self):
        self.supported_formats = ['.dcm', '.jpg', '.jpeg', '.png', '.tiff']
        # Initialize OCR model for text recognition
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
        if torch.cuda.is_available():
            self.model.to('cuda')
    
    def load_image(self, image_path):
        """Load medical image from various formats including DICOM"""
        try:
            ext = os.path.splitext(image_path)[1].lower()
            if ext not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {ext}")
            
            if ext == '.dcm':
                return self._load_dicom(image_path)
            else:
                return Image.open(image_path)
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise
    
    def _load_dicom(self, path):
        """Load DICOM image"""
        try:
            dcm = pydicom.dcmread(path)
            return Image.fromarray(dcm.pixel_array)
        except Exception as e:
            logger.error(f"Error loading DICOM: {str(e)}")
            raise
    
    def analyze_medical_report(self, image_path):
        """Analyze medical report image and extract information"""
        try:
            # Load and preprocess image
            image = self.load_image(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Perform OCR on the image
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            if torch.cuda.is_available():
                pixel_values = pixel_values.to('cuda')
            
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Process and structure the extracted text
            analysis_result = self._process_medical_text(generated_text)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}\n{traceback.format_exc()}")
            raise
    
    def _process_medical_text(self, text):
        """Process and structure the extracted medical text"""
        sections = {
            'Patient Information': [],
            'Diagnosis': [],
            'Findings': [],
            'Recommendations': [],
            'Other Information': []
        }
        
        # Simple rule-based text classification
        lines = text.split('\n')
        current_section = 'Other Information'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Classify line into sections based on keywords
            lower_line = line.lower()
            if any(keyword in lower_line for keyword in ['patient', 'name:', 'age:', 'dob:', 'sex:']):
                current_section = 'Patient Information'
            elif any(keyword in lower_line for keyword in ['diagnosis:', 'assessment:', 'condition:']):
                current_section = 'Diagnosis'
            elif any(keyword in lower_line for keyword in ['finding', 'observation', 'shows', 'reveals']):
                current_section = 'Findings'
            elif any(keyword in lower_line for keyword in ['recommend', 'advise', 'follow', 'plan:']):
                current_section = 'Recommendations'
            
            sections[current_section].append(line)
        
        # Format the results
        result = "Medical Report Analysis\n" + "="*20 + "\n\n"
        for section, content in sections.items():
            if content:
                result += f"{section}:\n"
                result += "\n".join(f"• {line}" for line in content)
                result += "\n\n"
        
        return result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_image.py <path_to_image>")
        sys.exit(1)
        
    analyzer = MedicalImageAnalyzer()
    result = analyzer.analyze_medical_report(sys.argv[1])
    print(result)
