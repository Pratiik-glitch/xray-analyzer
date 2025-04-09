import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.target_size = (224, 224)  # Standard size for many deep learning models
        logger.info("Initializing Image Processor")
    
    def process_image(self, image_path):
        """Process the image for analysis"""
        try:
            # Open image
            if isinstance(image_path, str):
                image = Image.open(image_path)
            else:
                image = image_path
                
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance image
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)  # Slight contrast enhancement
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)  # Slight sharpness enhancement
            
            # Resize image while maintaining aspect ratio
            image.thumbnail(self.target_size, Image.Resampling.LANCZOS)
            
            # Create new image with padding to get exact target size
            new_image = Image.new('RGB', self.target_size, (0, 0, 0))
            
            # Paste the resized image in the center
            offset = ((self.target_size[0] - image.size[0]) // 2,
                     (self.target_size[1] - image.size[1]) // 2)
            new_image.paste(image, offset)
            
            return new_image
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise