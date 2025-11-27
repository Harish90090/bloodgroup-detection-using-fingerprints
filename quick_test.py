import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Importing modules...")
try:
    from utils.predict import get_model, predict_all, majority_prediction
    print("Modules imported successfully!")
    
    print("\nLoading model...")
    model = get_model()
    print("Model loaded successfully!")
    
    print("\nRunning prediction on static/input_images...")
    predictions = predict_all('static/input_images')
    print(f"Got {len(predictions)} predictions:")
    for p in predictions:
        print(f"  {p['filename']}: {p['label']} (confidence: {max(p['confidence']):.4f})")
    
    print("\nCalculating final prediction...")
    final = majority_prediction(predictions)
    print(f"Final blood group: {final}")
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    print(traceback.format_exc())
