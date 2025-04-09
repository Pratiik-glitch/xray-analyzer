from gevent import monkey
monkey.patch_all()

from flask import Flask, request, jsonify, send_from_directory, send_file, render_template
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename
import os
import logging
import traceback
import base64
from io import BytesIO
import threading
import queue
import uuid
from functools import lru_cache
import torch
import time
from gevent.pywsgi import WSGIServer
from gevent.pool import Pool

from analyze_image import MedicalImageAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobQueue:
    def __init__(self, max_size=50):
        self.queue = queue.Queue(maxsize=max_size)
        self.results = {}
        self.processing = set()
        self._stop = False
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def add_job(self, job_id, image_path):
        """Add a job to the queue"""
        try:
            self.queue.put((job_id, image_path), block=False)
            return True
        except queue.Full:
            return False
    
    def get_result(self, job_id):
        """Get job result"""
        if job_id in self.results:
            return self.results.pop(job_id)
        if job_id in self.processing:
            return {'status': 'processing'}
        return {'status': 'not_found'}
    
    def _process_queue(self):
        """Process jobs in the queue"""
        analyzer = MedicalImageAnalyzer()
        
        while not self._stop:
            try:
                job_id, image_path = self.queue.get(timeout=1)
                self.processing.add(job_id)
                
                try:
                    # Process image
                    results = analyzer.analyze_medical_report(image_path)
                    
                    self.results[job_id] = {
                        'status': 'completed',
                        'results': results
                    }
                except Exception as e:
                    self.results[job_id] = {
                        'status': 'error',
                        'error': str(e)
                    }
                finally:
                    self.processing.remove(job_id)
                    if os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                        except:
                            pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing queue: {str(e)}")

class RateLimiter:
    def __init__(self, max_requests=20, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
        self.lock = threading.Lock()
    
    def is_allowed(self, client_id):
        with self.lock:
            now = time.time()
            if client_id not in self.requests:
                self.requests[client_id] = []
            
            # Remove old requests
            self.requests[client_id] = [t for t in self.requests[client_id] 
                                      if now - t < self.time_window]
            
            # Check if under limit
            if len(self.requests[client_id]) < self.max_requests:
                self.requests[client_id].append(now)
                return True
            return False

# Initialize Flask app
app = Flask(__name__, 
           static_folder='static',
           template_folder='static/templates')

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize components
job_queue = JobQueue(max_size=50)
rate_limiter = RateLimiter(max_requests=20, time_window=60)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Check rate limit
    client_id = request.remote_addr
    if not rate_limiter.is_allowed(client_id):
        return jsonify({'error': 'Rate limit exceeded'}), 429
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                   f"{uuid.uuid4()}_{filename}")
            file.save(file_path)
            
            # Add job to queue
            job_id = str(uuid.uuid4())
            if job_queue.add_job(job_id, file_path):
                return jsonify({'job_id': job_id})
            else:
                os.remove(file_path)
                return jsonify({'error': 'Server is busy'}), 503
    
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/status/<job_id>')
def get_job_status(job_id):
    try:
        result = job_queue.get_result(job_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}")
        return jsonify({'error': str(e)}), 500

def cleanup_old_files():
    """Clean up old files in upload directory"""
    try:
        current_time = time.time()
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.getctime(file_path) < current_time - 3600:  # 1 hour old
                os.remove(file_path)
    except Exception as e:
        logger.error(f"Error cleaning up files: {str(e)}")

if __name__ == '__main__':
    # Enable CORS for development
    from flask_cors import CORS
    CORS(app)
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_files)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    # Start server
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    print("Server running on http://localhost:5000")
    http_server.serve_forever()