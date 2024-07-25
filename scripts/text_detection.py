import numpy as np
import os
import sys
from tensorflow.keras.models import load_model

# Add the path to the scripts directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.preprocess import preprocess_image

# Load the text detection model
text_detection_model = load_model(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'models/text_detection_model.h5'))

def detect_text(image_path):
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)
    detected_boxes = text_detection_model.predict(image)
    
    # Print detected_boxes to understand their structure
    print("Detected Boxes:", detected_boxes)

    # Process detected_boxes to extract (x, y, w, h)
    boxes = []
    for box in detected_boxes:
        # Print each box to understand its structure
        print("Box:", box)
        x, y, w, h = process_box_output(box)
        boxes.append((x, y, w, h))
    
    return boxes

def process_box_output(box):
    # Flatten the box array if needed
    if isinstance(box, np.ndarray):
        box = box.flatten()
    print("Box (inside process_box_output):", box)  # Print the box to understand its structure
    
     # Convert to integers using astype
    if len(box) >= 4:
        box = box[:4].astype(int)  # Ensure we only take the first four elements
        x, y, w, h = box[0], box[1], box[2], box[3]
        return x, y, w, h
    else:
        raise ValueError("Box does not have enough elements")


    # Ensure the box has the expected number of elements
    if len(box) >= 4:
        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])
        return x, y, w, h
    else:
        raise ValueError("Box does not have enough elements")
