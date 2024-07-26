import cv2
import numpy as np
import os
import pickle

# Initialize video capture
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Check if the classifier loaded correctly
if facedetect.empty():
    print("Error loading cascade classifier")
    exit()

faces_data = []
labels = []

num_persons = int(input("Enter the number of persons: "))

for i in range(num_persons):
    name = input(f"Enter the name of person {i + 1}: ")
    count = 0

    while count < 40:  # Capture 40 face samples for each person
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Capture only the face region
            face_img = gray[y:y + h, x:x + w]
            resized_img = cv2.resize(face_img, (50, 50))

            # Store the face data
            faces_data.append(resized_img.flatten())
            labels.append(name)

            count += 1

            cv2.putText(frame, f"{name} {count}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

        cv2.imshow("Capturing Faces", frame)
        cv2.waitKey(1)

# Save the collected data
faces_data = np.array(faces_data)
labels = np.array(labels)

if not os.path.exists('data'):
    os.makedirs('data')

# Save labels
with open('data/names.pkl', 'wb') as f:
    pickle.dump(labels, f)

# Save faces data
with open('data/faces_data.pkl', 'wb') as f:
    pickle.dump(faces_data, f)

# Release video capture and close all windows
video.release()
cv2.destroyAllWindows()
