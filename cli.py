import os
import sys
from PIL import Image
import numpy as np

def print_header(text):
    print("\n" + "="*50)
    print(text)
    print("="*50 + "\n")

def print_result(label, value):
    print(f"{label:20}: {value}")

def analyze_image(image):
    """Simple image analysis for demonstration"""
    # Convert to grayscale and get basic stats
    if image.mode != 'L':
        image = image.convert('L')
    
    # Get image array
    img_array = np.array(image)
    
    # Calculate basic statistics
    mean = np.mean(img_array)
    std = np.std(img_array)
    min_val = np.min(img_array)
    max_val = np.max(img_array)
    
    # Simple contrast measure
    contrast = (max_val - min_val) / (max_val + min_val + 1e-6)
    
    # Simulate analysis results
    results = {
        'image_size': image.size,
        'mean_intensity': f"{mean:.2f}",
        'std_intensity': f"{std:.2f}",
        'contrast_ratio': f"{contrast:.2f}",
        'predictions': [
            "Demo: Image statistics calculated",
            f"Demo: Contrast level is {'high' if contrast > 0.5 else 'low'}",
            "Demo: This is a simplified analysis"
        ]
    }
    
    return results

def main():
    print("Initializing simple image analysis system...")
    
    while True:
        print_header("Medical Image Analysis CLI (Demo Version)")
        print("1. Analyze image")
        print("2. Exit")
        
        choice = input("\nEnter your choice (1-2): ")
        
        if choice == '2':
            print("\nExiting...")
            break
            
        if choice == '1':
            # Get image path
            image_path = input("\nEnter the path to your medical image: ").strip('"')
            
            if not os.path.exists(image_path):
                print("\nError: File not found!")
                continue
            
            try:
                # Load and process image
                print("\nLoading image...")
                image = Image.open(image_path)
                
                print("Processing image...")
                results = analyze_image(image)
                
                # Display results
                print_header("Analysis Results (Demo)")
                print_result("Image Size", f"{results['image_size'][0]}x{results['image_size'][1]}")
                print_result("Mean Intensity", results['mean_intensity'])
                print_result("Std Intensity", results['std_intensity'])
                print_result("Contrast Ratio", results['contrast_ratio'])
                
                print("\nDemo Insights:")
                for pred in results['predictions']:
                    print(f"- {pred}")
                
                input("\nPress Enter to continue...")
                
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue

if __name__ == "__main__":
    main()
