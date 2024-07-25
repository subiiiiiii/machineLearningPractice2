# OCR App

This project demonstrates an OCR (Optical Character Recognition) system using Python and TensorFlow. The system consists of two main parts: text detection and text recognition.

## Directory Structure
machineLearningPractice2/
├── data/
│ ├── images/
│ └── annotations/
├── notebooks/
│ ├── data_preprocessing.ipynb
│ ├── text_detection_training.ipynb
│ └── text_recognition_training.ipynb
├── models/
│ ├── text_detection_model.h5
│ └── text_recognition_model.h5
├── scripts/
│ ├── preprocess.py
│ ├── text_detection.py
│ └── text_recognition.py
├── requirements.txt
└── README.md


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/machineLearningPractice2.git
   cd machineLearningPractice2

2. Install the required packages:

pip install -r requirements.txt

Usage
1. Preprocess the dataset using the data_preprocessing.ipynb notebook.
2. Train the text detection model using the text_detection_training.ipynb notebook.
3. Train the text recognition model using the text_recognition_training.ipynb notebook.
4. Use the scripts in the scripts directory to integrate the models and perform OCR on images.


To recognize text in an image, run the following script:

python scripts/text_recognition.py