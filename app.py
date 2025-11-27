import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, render_template, jsonify, request
import shutil
from utils.predict import predict_all, majority_prediction

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/input_images'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Upload images endpoint
@app.route('/upload-images', methods=['POST'])
def upload_images():
    try:
        input_dir = app.config['UPLOAD_FOLDER']
        
        # Create directory if it doesn't exist
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        
        # Clear existing files
        for f in os.listdir(input_dir):
            try:
                os.remove(os.path.join(input_dir, f))
            except:
                pass
        
        # Get uploaded files
        files = request.files.getlist('files')
        
        if len(files) != 10:
            return jsonify({'error': f'Please upload exactly 10 images. You uploaded {len(files)}.'})
        
        # Save files
        for file in files:
            if file and file.filename:
                filename = file.filename
                filepath = os.path.join(input_dir, filename)
                file.save(filepath)
        
        return jsonify({'success': True})
    
    except Exception as e:
        import traceback
        print("❌ Upload Error:", str(e))
        print(traceback.format_exc())
        return jsonify({'error': f'Upload failed: {str(e)}'})

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_dir = app.config['UPLOAD_FOLDER']
        
        # Check if directory exists
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        
        # Get list of images
        images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))]
        
        if len(images) != 10:
            return jsonify({'error': f'Please select 10 images first. Found {len(images)} images.'})

        predictions = predict_all(input_dir)
        final_prediction = majority_prediction(predictions)

        return jsonify({
            'success': True,
            'final_prediction': final_prediction,
            'predictions': predictions
        })

    except Exception as e:
        import traceback
        error_msg = str(e)
        tb = traceback.format_exc()
        print("❌ Prediction Error:", error_msg)
        print(tb)
        return jsonify({'error': f'Prediction failed: {error_msg}'})

# Detail view for a single fingerprint (optional)
@app.route('/fingerprint-detail/<filename>')
def fingerprint_detail(filename):
    try:
        from utils.predict import predict_single_softmax
        input_dir = app.config['UPLOAD_FOLDER']
        filepath = os.path.join(input_dir, filename)
        result = predict_single_softmax(filepath)

        return render_template('fingerprint_detail.html',
                               image_file=filename,
                               confidences=result['confidence'],
                               predicted_label=result['label'])
    except Exception as e:
        print("❌ Fingerprint Detail Error:", str(e))
        return "Error loading fingerprint details.", 500

if __name__ == '__main__':
    app.run(debug=True)
