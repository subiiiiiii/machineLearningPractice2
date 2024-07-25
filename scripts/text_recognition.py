import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import preprocess_image
from text_detection import detect_text

# Load the text recognition model
text_recognition_model = load_model('../models/text_recognition_model.h5')

def recognize_text(image_path):
    image = preprocess_image(image_path)
    detected_boxes = detect_text(image_path)

    recognized_texts = []
    for box in detected_boxes:
        x, y, w, h = box
        text_region = image[y:y+h, x:x+w]
        text_region = cv2.cvtColor(text_region, cv2.COLOR_RGB2GRAY)
        text_region = cv2.resize(text_region, (128, 32))
        text_region = np.expand_dims(text_region, axis=[0, -1])

        predicted_text = text_recognition_model.predict(text_region)
        recognized_texts.append(predicted_text)

    return recognized_texts

# Example usage
image_path = 'data/images/labReport_Page_001.jpg'  # Replace with your image path
recognized_texts = recognize_text(image_path)
print(recognized_texts)
