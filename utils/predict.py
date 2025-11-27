import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import cv2
from pathlib import Path

# Lazy load model on first use
_model = None

def get_model():
    global _model
    if _model is None:
        import warnings
        warnings.filterwarnings('ignore')
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        
        MODEL_PATH = Path(__file__).parent.parent / "model" / "final_best_efficientnetb0_model_final.keras"
        _model = tf.keras.models.load_model(str(MODEL_PATH))
    return _model

CLASS_LABELS = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
TARGET_SIZE = (103, 96)

def preprocess_image(img_path):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        img = cv2.resize(img, TARGET_SIZE)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.repeat(img, 3, axis=-1)
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error preprocessing {img_path}: {e}")
        raise

def predict_all(image_dir):
    predictions = []
    model = get_model()
    try:
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))]
        print(f"Found {len(image_files)} images in {image_dir}")
        if not image_files:
            raise ValueError(f"No images found in {image_dir}")
        
        for filename in sorted(image_files):
            try:
                full_path = os.path.join(image_dir, filename)
                print(f"Processing: {filename}")
                img_tensor = preprocess_image(full_path)
                preds = model.predict(img_tensor, verbose=0)[0]
                label = CLASS_LABELS[np.argmax(preds)]
                predictions.append({
                    "filename": os.path.basename(filename),
                    "label": label,
                    "confidence": preds.tolist()
                })
                print(f"✓ {filename} -> {label}")
            except Exception as e:
                print(f"❌ Error predicting for {filename}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not predictions:
            raise ValueError("No predictions were generated from any images")
    except Exception as e:
        print(f"❌ Error in predict_all: {e}")
        raise
    
    return predictions

def majority_prediction(predictions):
    labels = [p['label'] for p in predictions]
    return max(set(labels), key=labels.count)

def predict_single_softmax(img_path):
    model = get_model()
    img_tensor = preprocess_image(img_path)
    preds = model.predict(img_tensor, verbose=0)[0]
    label_index = int(np.argmax(preds))
    label = CLASS_LABELS[label_index]
    return {
        'filename': os.path.basename(img_path),
        'label': label,
        'confidence': preds.tolist()
    }
