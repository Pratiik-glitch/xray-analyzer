import os
import redis
from rq import Worker, Queue, Connection
from image_processor import ImageProcessor
from ai_models import MedicalImageAnalyzer
import torch

# Configure Redis connection
redis_conn = redis.Redis(host='localhost', port=6379, db=0)

# Initialize processors
image_processor = ImageProcessor()
ai_analyzer = MedicalImageAnalyzer()

def process_image(image_data):
    """Process image with model"""
    with torch.no_grad():
        try:
            # Preprocess image
            processed = image_processor.preprocess_image(image_data)
            
            # Get predictions
            results = ai_analyzer.analyze_image(processed)
            insights = ai_analyzer.generate_insights(results)
            
            return results, insights
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None

if __name__ == '__main__':
    # Start worker
    with Connection(redis_conn):
        worker = Worker(Queue('medical_analysis'))
        worker.work()
