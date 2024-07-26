import cv2
import numpy as np
import pickle

# Load the face data and labels
with open('data/faces_data.pkl', 'rb') as f:
    faces_data = pickle.load(f)

with open('data/names.pkl', 'rb') as f:
    labels = pickle.load(f)

# Convert labels to integers
unique_labels = np.unique(labels)
label_map = {name: idx for idx, name in enumerate(unique_labels)}
int_labels = np.array([label_map[label] for label in labels])

# Initialize the face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the recognizer
face_recognizer.train(faces_data.reshape(-1, 50, 50), int_labels)

# Save the trained model
face_recognizer.save('data/face_recognizer.yml')

# Save the label map
with open('data/label_map.pkl', 'wb') as f:
    pickle.dump(label_map, f)
