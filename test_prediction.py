#!/usr/bin/env python
import os
import sys

# Add the project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.predict import get_model, predict_all, majority_prediction
import cv2
import numpy as np

# Test directory
test_dir = 'static/input_images'

print(f"Test directory: {test_dir}")
print(f"Directory exists: {os.path.exists(test_dir)}")

if os.path.exists(test_dir):
    files = os.listdir(test_dir)
    print(f"Files in directory: {files}")
    print(f"Number of files: {len(files)}")
    
    image_files = [f for f in files if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))]
    print(f"Image files: {image_files}")
    print(f"Number of image files: {len(image_files)}")
    
    if len(image_files) > 0:
        print("\nTesting model loading...")
        try:
            model = get_model()
            print("Model loaded successfully!")
            
            print("\nTesting prediction on first image...")
            predictions = predict_all(test_dir)
            print(f"Predictions: {predictions}")
            
            if predictions:
                final = majority_prediction(predictions)
                print(f"Final prediction: {final}")
        except Exception as e:
            import traceback
            print(f"Error: {e}")
            print(traceback.format_exc())
else:
    print("Directory does not exist!")
