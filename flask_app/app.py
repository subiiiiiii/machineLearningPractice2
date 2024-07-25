from flask import Flask, request, render_template
import os
import sys
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Ensure the project directory is in the Python path
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from scripts.preprocess import preprocess_image
from scripts.text_detection import detect_text

# Load the models
text_detection_model = load_model(os.path.join(project_dir, 'models/text_detection_model.h5'))
text_recognition_model = load_model(os.path.join(project_dir, 'models/text_recognition_model.h5'))

app = Flask(__name__)

def recognize_text(image_path):
    image = preprocess_image(image_path)
    detected_boxes = detect_text(image_path)

    recognized_texts = []
    for box in detected_boxes:
        x, y, w, h = box
        text_region = image[int(y):int(y+h), int(x):int(x+w)]
        text_region = cv2.cvtColor(text_region, cv2.COLOR_RGB2GRAY)
        text_region = cv2.resize(text_region, (128, 32))
        text_region = np.expand_dims(text_region, axis=[0, -1])

        predicted_text = text_recognition_model.predict(text_region)
        recognized_texts.append(predicted_text)

    return ''.join([str(txt) for txt in recognized_texts])

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = os.path.join('static', file.filename)
        file.save(file_path)
        recognized_text = recognize_text(file_path)
        return render_template('index.html', text=recognized_text)
    return 'File upload failed'

if __name__ == '__main__':
    app.run(debug=True)
