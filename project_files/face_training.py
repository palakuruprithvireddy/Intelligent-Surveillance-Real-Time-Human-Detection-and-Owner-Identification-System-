import torch
import cv2
import numpy as np
import os
from ultralytics import YOLO

# Path for face image database
path = '/Users/prithvireddy/Documents/MLGroupProject/new_preprocessed_images'

# Load the YOLOv11 model
#model = torch.load('yolo11n.pt', map_location='cpu')
model = YOLO('yolo11n.pt')

recognizer = cv2.face.LBPHFaceRecognizer_create()  # For face recognition

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  # Get all image paths
    faceSamples = []  # To store detected face regions
    ids = []          # To store labels

    for imagePath in imagePaths:
        # Read the image
        img = cv2.imread(imagePath)
        results = model(img)  # Run YOLOv11 inference
        detections = results[0].boxes.data.cpu().numpy()  # Access bounding boxes and confidence scores

        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection  # Unpack detection
            if int(cls) == 0 and conf > 0.6:  # Class 0: person, confidence > 60%
                # Crop face region
                cropped_face = img[int(y1):int(y2), int(x1):int(x2)]
                gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

                # Append detected face and assign label
                faceSamples.append(gray_face)
                ids.append(1)  # Assign label 0 for owner

    return faceSamples, ids


print("\n[INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Ensure the 'trainer/' directory exists
if not os.path.exists('trainer'):
    os.makedirs('trainer')

# Save the trained model
recognizer.write('trainer/trainer4.yml')
print(f"\n[INFO] Model trained and saved at 'trainer/trainer.yml'")
