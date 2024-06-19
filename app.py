from flask import Flask, request, jsonify, render_template
import os
import torch
from PIL import Image
from collections import Counter
import concurrent.futures
import threading
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

app.config['UPLOAD_FOLDER'] = 'uploads/'
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Lock for thread safety
lock = threading.Lock()

# Global variable to store predictions
all_predictions = []

def preprocess_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def detect_objects(image_path):
    img = Image.open(image_path)
    img = preprocess_image(img)
    results = model(img)
    predictions = results.pandas().xyxy[0].to_dict(orient="records")
    names = [prediction['name'] for prediction in predictions]
    return names

def process_images(image_files):
    global all_predictions
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for image_file in image_files:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)
            future = executor.submit(detect_objects, image_path)
            futures.append((future, image_path))
        
        for future, image_path in futures:
            predictions = future.result()
            all_predictions.append({
                "image_path": image_path,
                "predictions": predictions
            })
            os.remove(image_path)
    
    return all_predictions

def calculate_percentage(predictions):
    flat_predictions = [item for pred in predictions for item in pred["predictions"]]
    label_counts = Counter(flat_predictions)
    total_predictions = len(flat_predictions)
    percentage_focus = 0.0
    if total_predictions > 0:
        percentage_focus = (label_counts['focus'] / total_predictions) * 100
    return percentage_focus

@app.route('/')
def upload_form():
    return render_template('images.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'images[]' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    images = request.files.getlist('images[]')
    
    # Perform object detection on each image
    predictions = process_images(images)
    
    # Calculate live percentage focus
    with lock:
        percentage_focus = calculate_percentage(predictions)
    
    return jsonify({
        "predictions": predictions,
        "percentage_focus": percentage_focus
    }), 200

@app.route('/reset', methods=['POST'])
def reset_predictions():
    global all_predictions
    all_predictions = []
    return jsonify({"message": "Predictions reset successfully"}), 200


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(host='0.0.0.0', port=5000, debug=True)
