import os
import json
import shutil
from PIL import Image
import hashlib
from datetime import datetime
import threading
from image_processor import ImageProcessor
from ai_models import MedicalImageAnalyzer

class StaticProcessor:
    def __init__(self, static_dir='static/cache', max_cache_size_mb=1024):
        self.static_dir = static_dir
        self.results_dir = os.path.join(static_dir, 'results')
        self.images_dir = os.path.join(static_dir, 'images')
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        self.cache_lock = threading.Lock()
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize processors
        self.image_processor = ImageProcessor()
        self.ai_analyzer = MedicalImageAnalyzer()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_old_files)
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()
    
    def get_file_hash(self, image):
        """Generate hash for image file"""
        img_bytes = image.tobytes()
        return hashlib.md5(img_bytes).hexdigest()
    
    def cache_exists(self, file_hash):
        """Check if analysis results exist in cache"""
        result_path = os.path.join(self.results_dir, f"{file_hash}.json")
        image_path = os.path.join(self.images_dir, f"{file_hash}.jpg")
        return os.path.exists(result_path) and os.path.exists(image_path)
    
    def get_cached_result(self, file_hash):
        """Get cached analysis results"""
        result_path = os.path.join(self.results_dir, f"{file_hash}.json")
        image_path = os.path.join(self.images_dir, f"{file_hash}.jpg")
        
        if not self.cache_exists(file_hash):
            return None
            
        try:
            with open(result_path, 'r') as f:
                results = json.load(f)
            
            # Update access time
            os.utime(result_path, None)
            os.utime(image_path, None)
            
            return {
                'results': results,
                'image_path': image_path
            }
        except:
            return None
    
    def cache_result(self, image, results):
        """Cache analysis results and optimized image"""
        try:
            with self.cache_lock:
                # Generate hash
                file_hash = self.get_file_hash(image)
                
                # Save optimized image
                image_path = os.path.join(self.images_dir, f"{file_hash}.jpg")
                image.save(image_path, 'JPEG', quality=85, optimize=True)
                
                # Save results
                result_path = os.path.join(self.results_dir, f"{file_hash}.json")
                with open(result_path, 'w') as f:
                    json.dump(results, f)
                
                return file_hash
        except:
            return None
    
    def _cleanup_old_files(self):
        """Cleanup old cache files periodically"""
        while True:
            try:
                total_size = 0
                files = []
                
                # Get all cached files with their timestamps
                for root, _, filenames in os.walk(self.static_dir):
                    for filename in filenames:
                        filepath = os.path.join(root, filename)
                        size = os.path.getsize(filepath)
                        accessed = os.path.getatime(filepath)
                        files.append((filepath, size, accessed))
                        total_size += size
                
                # If cache is too large, remove oldest files
                if total_size > self.max_cache_size:
                    # Sort by access time (oldest first)
                    files.sort(key=lambda x: x[2])
                    
                    # Remove files until we're under the limit
                    for filepath, size, _ in files:
                        if total_size <= self.max_cache_size:
                            break
                            
                        try:
                            os.remove(filepath)
                            total_size -= size
                        except:
                            continue
            
            except Exception as e:
                print(f"Error in cleanup: {str(e)}")
            
            # Sleep for 1 hour before next cleanup
            threading.Event().wait(3600)
