# Portable Medical Diagnostic Tool

A computer vision and GenAI-powered diagnostic tool for analyzing medical images in resource-constrained settings.

## Features

- Medical image analysis using computer vision
- Disease detection using deep learning models
- Portable web interface for easy access
- Support for common medical image formats (X-ray, MRI, CT scans)
- Preliminary diagnostic insights generation

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd medical-diagnostic-tool
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Upload a medical image through the web interface

4. View the analysis results and preliminary diagnostic insights

## Project Structure

- `app.py`: Main application file
- `image_processor.py`: Image processing utilities
- `ai_models.py`: AI/ML model implementations
- `templates/`: Web interface templates
- `static/`: Static assets for web interface
- `models/`: Pre-trained model storage

## Disclaimer

This tool is intended for preliminary screening purposes only. Always consult with healthcare professionals for proper medical diagnosis and treatment.

## License


MIT License 
